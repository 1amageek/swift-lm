import Metal
import LMIR

public struct MetalInferenceModel: @unchecked Sendable {

    public let plan: MetalDispatchPlan
    public var prefillPlan: MetalPrefillPlan?
    public let commandQueue: MTLCommandQueue
    public var position: Int = 0

    private var pendingCommandBuffer: MTLCommandBuffer?
    private var hasPendingResult: Bool = false

    public init(plan: MetalDispatchPlan, device: MTLDevice) throws {
        self.plan = plan
        self.prefillPlan = nil
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create command queue")
        }
        self.commandQueue = queue
    }

    // MARK: - Decode

    private func encodeSteps(_ enc: MTLComputeCommandEncoder) {
        for step in plan.steps {
            if step.sync == .bufferBarrier { enc.memoryBarrier(scope: .buffers) }
            enc.setComputePipelineState(step.pipeline)
            for (index, buffer, offset) in step.bufferBindings {
                enc.setBuffer(buffer, offset: offset, index: index)
            }
            for (index, value) in step.bytesBindings {
                value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) }
            }
            if step.threadgroupMemoryLength > 0 {
                enc.setThreadgroupMemoryLength(step.threadgroupMemoryLength, index: 0)
            }
            enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
        }
    }

    public mutating func decode(tokenID: Int32) -> Int32 {
        let b = plan.buffers
        var result: Int32 = -1
        if hasPendingResult {
            pendingCommandBuffer?.waitUntilCompleted()
            result = b.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        }
        b.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        b.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return result }
        encodeSteps(enc)
        enc.endEncoding()
        cb.commit()
        pendingCommandBuffer = cb
        hasPendingResult = true
        position += 1
        return result
    }

    public mutating func decodeSync(tokenID: Int32) -> Int32 {
        let b = plan.buffers
        b.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        b.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return -1 }
        encodeSteps(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { print("[MetalInference] GPU error: \(error)") }


        position += 1
        return b.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    // MARK: - Prefill

    /// Prefill the KV cache with prompt tokens and return the first predicted token.
    ///
    /// Returns the argmax of the prefill logits (the model's first generated token).
    /// The caller should output this token and feed it to the first decode step.
    @discardableResult
    public mutating func prefill(tokens: [Int32]) -> Int32 {
        guard let prefill = prefillPlan else {
            var lastOutput: Int32 = -1
            for token in tokens { lastOutput = decodeSync(tokenID: token) }
            return lastOutput
        }
        guard !tokens.isEmpty else { return -1 }
        guard tokens.count <= prefill.maximumSequenceLength else { return -1 }

        let seqLen = tokens.count
        let prefillStart = CFAbsoluteTimeGetCurrent()

        // Fill tokenIDs and positions
        let tokenPtr = prefill.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefill.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = tokens[i]
            posPtr[i] = UInt32(position + i)
        }

        // Execute sequence graph
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return -1 }

        var dispatchCount = 0
        for step in prefill.steps {
            switch step.mode {
            case .batch:
                encodeBatchStep(encoder, step: step, sequenceLength: UInt32(seqLen))
                dispatchCount += 1
            case .lastToken:
                if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
                encoder.setComputePipelineState(step.pipeline)
                let lastPos = seqLen - 1
                for (index, buffer, baseOffset) in step.bufferBindings {
                    encoder.setBuffer(buffer, offset: baseOffset + lastPos * (step.perPositionStrides[index] ?? 0), index: index)
                }
                for (index, value) in step.bytesBindings {
                    value.withUnsafeBufferPointer { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
                }
                encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                dispatchCount += 1
            case .perPosition:
                for pos in 0..<seqLen {
                    if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
                    encoder.setComputePipelineState(step.pipeline)
                    for (index, buffer, baseOffset) in step.bufferBindings {
                        encoder.setBuffer(buffer, offset: baseOffset + pos * (step.perPositionStrides[index] ?? 0), index: index)
                    }
                    for (index, value) in step.bytesBindings {
                        value.withUnsafeBufferPointer { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
                    }
                    if let posIndex = step.positionBufferIndex {
                        var posValue = UInt32(position + pos)
                        withUnsafeBytes(of: &posValue) { encoder.setBytes($0.baseAddress!, length: $0.count, index: posIndex) }
                    }
                    encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                    dispatchCount += 1
                }
            }
        }

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            print("[MetalInference] PREFILL FAILED: \(error.localizedDescription)")
            return -1
        }

        // Transfer hidden: F32→F16 conversion via GPU blit
        // Decode hidden is storageModePrivate — cannot use CPU .contents().
        // Use a GPU blit to copy the last token from prefill (F32, shared) → decode (F16, private).
        let decodeHiddenSize = self.plan.buffers.hidden.length / MemoryLayout<Float16>.size
        let prefillHiddenStride = decodeHiddenSize * MemoryLayout<Float32>.size
        let lastTokenOffset = (seqLen - 1) * prefillHiddenStride

        if lastTokenOffset + prefillHiddenStride <= prefill.buffers.hidden.length {
            // Prefill hidden is shared — CPU-accessible for F32→F16 conversion.
            // Write F16 values to a temp shared buffer, then blit to private decode hidden.
            if self.plan.buffers.hidden.storageMode == .private {
                // Convert F32→F16 and blit to private decode hidden.
                // Use prefill scratch buffer as staging (shared, large enough).
                let src = (prefill.buffers.hidden.contents() + lastTokenOffset)
                    .bindMemory(to: Float32.self, capacity: decodeHiddenSize)
                let staging = prefill.buffers.scratch.contents()
                    .bindMemory(to: Float16.self, capacity: decodeHiddenSize)
                for i in 0..<decodeHiddenSize {
                    staging[i] = Float16(src[i])
                }
                // Blit from staging (shared) to decode hidden (private)
                if let blitCB = commandQueue.makeCommandBuffer(),
                   let blit = blitCB.makeBlitCommandEncoder() {
                    blit.copy(from: prefill.buffers.scratch, sourceOffset: 0,
                              to: self.plan.buffers.hidden, destinationOffset: 0,
                              size: decodeHiddenSize * MemoryLayout<Float16>.size)
                    blit.endEncoding()
                    blitCB.commit()
                    blitCB.waitUntilCompleted()
                }
            } else {
                // Shared mode — direct CPU copy (test path)
                let src = (prefill.buffers.hidden.contents() + lastTokenOffset)
                    .bindMemory(to: Float32.self, capacity: decodeHiddenSize)
                let dst = self.plan.buffers.hidden.contents()
                    .bindMemory(to: Float16.self, capacity: decodeHiddenSize)
                for i in 0..<decodeHiddenSize {
                    dst[i] = Float16(src[i])
                }
            }
        }

        // Copy KV cache (skip if shared — same buffer used by both prefill and decode)
        if let prefillKV = prefill.buffers.kvCache,
           let decodeKV = self.plan.buffers.kvCache,
           prefillKV.keys !== decodeKV.keys {
            if let blitCB = commandQueue.makeCommandBuffer(),
               let blit = blitCB.makeBlitCommandEncoder() {
                blit.copy(from: prefillKV.keys, sourceOffset: 0,
                          to: decodeKV.keys, destinationOffset: 0,
                          size: min(decodeKV.keys.length, prefillKV.keys.length))
                blit.copy(from: prefillKV.values, sourceOffset: 0,
                          to: decodeKV.values, destinationOffset: 0,
                          size: min(decodeKV.values.length, prefillKV.values.length))
                blit.endEncoding()
                blitCB.commit()
                blitCB.waitUntilCompleted()
            }
        }

        // Copy conv_state (skip if shared — same buffer used by both prefill and decode)
        if let prefillConvState = prefill.buffers.convState,
           let decodeConvState = self.plan.buffers.convState,
           prefillConvState !== decodeConvState {
            if let blitCB = commandQueue.makeCommandBuffer(),
               let blit = blitCB.makeBlitCommandEncoder() {
                blit.copy(from: prefillConvState, sourceOffset: 0,
                          to: decodeConvState, destinationOffset: 0,
                          size: min(decodeConvState.length, prefillConvState.length))
                blit.endEncoding()
                blitCB.commit()
                blitCB.waitUntilCompleted()
            }
        }

        position += seqLen

        // Return the first predicted token (argmax of prefill logits)
        return prefill.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    private func encodeBatchStep(_ encoder: MTLComputeCommandEncoder, step: MetalPrefillStep, sequenceLength: UInt32) {
        if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
        encoder.setComputePipelineState(step.pipeline)
        for (index, buffer, offset) in step.bufferBindings { encoder.setBuffer(buffer, offset: offset, index: index) }
        for (index, value) in step.bytesBindings {
            value.withUnsafeBufferPointer { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
        }
        if let seqLenIndex = step.sequenceLengthBindingIndex {
            var seqLen = sequenceLength
            withUnsafeBytes(of: &seqLen) { encoder.setBytes($0.baseAddress!, length: $0.count, index: seqLenIndex) }
        }
        var grid = step.gridSize
        if step.sequenceLengthBindingIndex != nil && grid.height > 1 {
            grid = MTLSize(width: grid.width, height: Int(sequenceLength), depth: grid.depth)
        }
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: step.threadgroupSize)
    }

    // MARK: - Lifecycle

    public mutating func flush() -> Int32 {
        guard hasPendingResult else { return -1 }
        pendingCommandBuffer?.waitUntilCompleted()
        hasPendingResult = false
        pendingCommandBuffer = nil
        return plan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    public mutating func resetCaches() {
        pendingCommandBuffer?.waitUntilCompleted()
        pendingCommandBuffer = nil
        hasPendingResult = false
        position = 0
        if let convState = plan.buffers.convState {
            memset(convState.contents(), 0, convState.length)
        }
    }
}
