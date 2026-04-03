import Metal

struct MetalPrefillExecutor: Sendable {
    private let transferPlanner = MetalPrefillTransferPlanner()

    func prefill(
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        position: inout Int,
        tokens: [Int32]
    ) -> Int32 {
        guard !tokens.isEmpty else { return -1 }
        guard tokens.count <= prefillPlan.maximumSequenceLength else { return -1 }

        let sequenceLength = tokens.count
        let tokenPointer = prefillPlan.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: sequenceLength)
        let positionPointer = prefillPlan.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: sequenceLength)
        for index in 0..<sequenceLength {
            tokenPointer[index] = tokens[index]
            positionPointer[index] = UInt32(position + index)
        }

        var transferPlan = transferPlanner.makeTransferPlan(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            sequenceLength: sequenceLength
        )

        do {
            if transferPlan.canEncodeInSameTransaction {
                _ = try submission.withTransaction(label: "prefill+postprocess") { transaction in
                    try transaction.withComputeEncoder { compute in
                        encodePrefillSteps(
                            encoder: compute,
                            prefillPlan: prefillPlan,
                            basePosition: position,
                            sequenceLength: sequenceLength
                        )
                    }
                    try transaction.withBlitEncoder { blit in
                        encodePostPrefillCopies(
                            blit: blit,
                            prefillPlan: prefillPlan,
                            decodePlan: decodePlan,
                            transferPlan: transferPlan
                        )
                    }
                }
                transferPlan = transferPlan.afterInlineBlit()
            } else {
                _ = try submission.withCompute(label: "prefill") { encoder in
                    encodePrefillSteps(
                        encoder: encoder,
                        prefillPlan: prefillPlan,
                        basePosition: position,
                        sequenceLength: sequenceLength
                    )
                }
            }
        } catch {
            print("[MetalInference] PREFILL FAILED: \(error)")
            return -1
        }

        stageHiddenIfNeeded(
            transferPlan: transferPlan,
            prefillPlan: prefillPlan,
            decodePlan: decodePlan
        )

        if transferPlan.needsStandaloneBlit {
            do {
                try submission.withBlit(label: "prefill.postprocess") { blit in
                    if transferPlan.shouldStageHiddenOnCPU {
                        blit.copy(
                            from: prefillPlan.buffers.scratch,
                            sourceOffset: 0,
                            to: decodePlan.buffers.hidden,
                            destinationOffset: 0,
                            size: transferPlan.hiddenCopySize
                        )
                    }
                    encodePostPrefillCopies(
                        blit: blit,
                        prefillPlan: prefillPlan,
                        decodePlan: decodePlan,
                        transferPlan: transferPlan.shouldStageHiddenOnCPU
                            ? transferPlan.withoutHiddenCopy()
                            : transferPlan
                    )
                }
            } catch {
                print("[MetalInference] Failed to copy post-prefill state: \(error)")
                return -1
            }
        }

        position += sequenceLength
        return prefillPlan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    private func stageHiddenIfNeeded(
        transferPlan: PostPrefillTransferPlan,
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan
    ) {
        guard transferPlan.shouldStageHiddenOnCPU || transferPlan.usesSharedDecodeHidden else {
            return
        }

        let decodeElementSize = decodePlan.buffers.bufferPrecision.byteSize
        let decodeHiddenSize = decodePlan.buffers.hidden.length / decodeElementSize
        let source = (prefillPlan.buffers.hidden.contents() + transferPlan.hiddenSourceOffset)
            .bindMemory(to: Float32.self, capacity: decodeHiddenSize)

        if transferPlan.shouldStageHiddenOnCPU {
            switch decodePlan.buffers.bufferPrecision {
            case .float16:
                let staging = prefillPlan.buffers.scratch.contents()
                    .bindMemory(to: Float16.self, capacity: decodeHiddenSize)
                for index in 0..<decodeHiddenSize {
                    staging[index] = Float16(source[index])
                }
            case .bfloat16:
                let staging = prefillPlan.buffers.scratch.contents()
                    .bindMemory(to: BFloat16.self, capacity: decodeHiddenSize)
                for index in 0..<decodeHiddenSize {
                    staging[index] = BFloat16(source[index])
                }
            case .float32:
                let staging = prefillPlan.buffers.scratch.contents()
                    .bindMemory(to: Float32.self, capacity: decodeHiddenSize)
                for index in 0..<decodeHiddenSize {
                    staging[index] = source[index]
                }
            }
            return
        }

        switch decodePlan.buffers.bufferPrecision {
        case .float16:
            let destination = decodePlan.buffers.hidden.contents()
                .bindMemory(to: Float16.self, capacity: decodeHiddenSize)
            for index in 0..<decodeHiddenSize {
                destination[index] = Float16(source[index])
            }
        case .bfloat16:
            let destination = decodePlan.buffers.hidden.contents()
                .bindMemory(to: BFloat16.self, capacity: decodeHiddenSize)
            for index in 0..<decodeHiddenSize {
                destination[index] = BFloat16(source[index])
            }
        case .float32:
            let destination = decodePlan.buffers.hidden.contents()
                .bindMemory(to: Float32.self, capacity: decodeHiddenSize)
            for index in 0..<decodeHiddenSize {
                destination[index] = source[index]
            }
        }
    }

    private func encodePrefillSteps(
        encoder: MTLComputeCommandEncoder,
        prefillPlan: MetalPrefillPlan,
        basePosition: Int,
        sequenceLength: Int
    ) {
        for step in prefillPlan.steps {
            switch step.mode {
            case .batch:
                encodeBatchStep(
                    encoder: encoder,
                    step: step,
                    sequenceLength: UInt32(sequenceLength)
                )
            case .lastToken:
                if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
                encoder.setComputePipelineState(step.pipeline)
                let lastPosition = sequenceLength - 1
                step.bindStaticArguments(encoder: encoder, position: lastPosition)
                encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
            case .perPosition:
                for positionOffset in 0..<sequenceLength {
                    if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
                    encoder.setComputePipelineState(step.pipeline)
                    step.bindStaticArguments(encoder: encoder, position: positionOffset)
                    if let positionBufferIndex = step.positionBufferIndex {
                        var positionValue = UInt32(basePosition + positionOffset)
                        withUnsafeBytes(of: &positionValue) { bytes in
                            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: positionBufferIndex)
                        }
                    }
                    encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }
            }
        }
    }

    private func encodeBatchStep(
        encoder: MTLComputeCommandEncoder,
        step: MetalPrefillStep,
        sequenceLength: UInt32
    ) {
        step.bindings.bind(to: encoder)
        step.bindRuntimeArguments(encoder: encoder, sequenceLength: sequenceLength)
        let gridSize = step.resolvedGridSize(sequenceLength: Int(sequenceLength))
        step.descriptor.encode(on: encoder, gridSize: gridSize)
    }

    private func encodePostPrefillCopies(
        blit: MTLBlitCommandEncoder,
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        transferPlan: PostPrefillTransferPlan
    ) {
        if transferPlan.hiddenCopySize > 0 {
            blit.copy(
                from: prefillPlan.buffers.hidden,
                sourceOffset: transferPlan.hiddenSourceOffset,
                to: decodePlan.buffers.hidden,
                destinationOffset: 0,
                size: transferPlan.hiddenCopySize
            )
        }
        if transferPlan.kvCopySize > 0,
           let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache {
            blit.copy(from: prefillKV.keys, sourceOffset: 0, to: decodeKV.keys, destinationOffset: 0, size: transferPlan.kvCopySize)
        }
        if transferPlan.valueCopySize > 0,
           let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache {
            blit.copy(from: prefillKV.values, sourceOffset: 0, to: decodeKV.values, destinationOffset: 0, size: transferPlan.valueCopySize)
        }
        if transferPlan.convCopySize > 0,
           let prefillConvState = prefillPlan.buffers.convState,
           let decodeConvState = decodePlan.buffers.convState {
            blit.copy(from: prefillConvState, sourceOffset: 0, to: decodeConvState, destinationOffset: 0, size: transferPlan.convCopySize)
        }
    }
}
