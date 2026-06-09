import Metal
import Testing
@testable import MetalCompiler
import LMIR

@Suite("Sparse MoE Decode Routing", .serialized)
struct SparseMoEDecodeRoutingTests {
    @Test("A1B BF16 down projection prefers blocked decode layout")
    func a1bBFloatDownProjectionPrefersBlockedDecodeLayout() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }
        let tensorName = "model.layers.0.feed_forward.experts.down_proj"
        let store = try makeStore(
            device: device,
            tensorName: tensorName,
            shape: [32, 2_048, 1_792],
            payloadBytes: 0
        )
        let fragment = makeA1BFragment()
        let entry = DispatchEntry(
            index: 0,
            fragment: fragment,
            parameterBindings: [
                .init(role: "expert_down_proj", tensorName: tensorName),
            ]
        )
        let binding = try #require(entry.parameterBindings.first)
        let resolver = ProjectionWeightAccessPolicyResolver()

        let decodeRequest = resolver.accessRequest(
            for: entry,
            role: "expert_down_proj",
            binding: binding,
            executionPhase: .decode,
            stafWeightStore: store
        )
        let prefillRequest = resolver.accessRequest(
            for: entry,
            role: "expert_down_proj",
            binding: binding,
            executionPhase: .prefill,
            stafWeightStore: store
        )

        #expect(decodeRequest.preferredLayout == .blockedRows8Tiles128)
        #expect(prefillRequest.preferredLayout == .rowMajor)

        let disabledRequest = withEnvironmentValue("SWIFTLM_SPARSE_MOE_DISABLE_DOWN_BLOCKED8X128", value: "1") {
            resolver.accessRequest(
                for: entry,
                role: "expert_down_proj",
                binding: binding,
                executionPhase: .decode,
                stafWeightStore: store
            )
        }
        let packed8Request = withEnvironmentValue("SWIFTLM_SPARSE_MOE_ENABLE_PACKED8", value: "1") {
            resolver.accessRequest(
                for: entry,
                role: "expert_down_proj",
                binding: binding,
                executionPhase: .decode,
                stafWeightStore: store
            )
        }
        let packed8TrueRequest = withEnvironmentValue("SWIFTLM_SPARSE_MOE_ENABLE_PACKED8", value: "true") {
            resolver.accessRequest(
                for: entry,
                role: "expert_down_proj",
                binding: binding,
                executionPhase: .decode,
                stafWeightStore: store
            )
        }

        #expect(disabledRequest.preferredLayout == .rowMajor)
        #expect(packed8Request.preferredLayout == .rowMajor)
        #expect(packed8TrueRequest.preferredLayout == .rowMajor)
    }

    @Test("Split decode uses blocked down only for resolved blocked layout")
    func splitDecodeUsesBlockedDownOnlyForResolvedBlockedLayout() throws {
        let rowMajorNames = try withSparseMoEEnvironmentCleared {
            try makeDecodeKernelNames(weightLayouts: .rowMajor)
        }
        let blockedNames = try withSparseMoEEnvironmentCleared {
            try makeDecodeKernelNames(weightLayouts: .init(
                gateUp: .rowMajor,
                down: .blockedRows8Tiles128
            ))
        }
        let stagedNames = try withSparseMoEEnvironmentCleared {
            try withEnvironmentValue("SWIFTLM_SPARSE_MOE_ENABLE_DOWN_STAGED_ACTIVATION", value: "1") {
                try makeDecodeKernelNames(weightLayouts: .init(
                    gateUp: .rowMajor,
                    down: .blockedRows8Tiles128
                ))
            }
        }

        #expect(rowMajorNames.contains("sparse_moe_bf16_down_packed4"))
        #expect(!rowMajorNames.contains("sparse_moe_bf16_down_blocked8x128_packed4"))
        #expect(!rowMajorNames.contains("sparse_moe_bf16_down_blocked8x128_staged_act"))
        #expect(blockedNames.contains("sparse_moe_bf16_down_blocked8x128_packed4"))
        #expect(!blockedNames.contains("sparse_moe_bf16_down_blocked8x128_staged_act"))
        #expect(!blockedNames.contains("sparse_moe_bf16_down_packed4"))
        #expect(stagedNames.contains("sparse_moe_bf16_down_blocked8x128_staged_act"))
    }

    @Test("Specialized builder prepares SparseMoE expert down access")
    func specializedBuilderPreparesSparseMoEExpertDownAccess() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }
        let tensorName = "tiny.expert_down_proj"
        let outputDimension = 8
        let intermediateDimension = 128
        let expertCount = 2
        let elementCount = expertCount * outputDimension * intermediateDimension
        let payloadBytes = elementCount * MemoryLayout<UInt16>.stride
        let store = try makeStore(
            device: device,
            tensorName: tensorName,
            shape: [expertCount, outputDimension, intermediateDimension],
            payloadBytes: payloadBytes
        )
        let fragment = SparseMoEFragment(
            expertCount: expertCount,
            expertsPerToken: 1,
            gateKind: .sigmoidTopK,
            inputDimension: 128,
            outputDimension: outputDimension,
            intermediateDimension: intermediateDimension,
            normalizeRoutingWeights: true,
            routedScalingFactor: 1,
            useExpertBias: false
        )
        let entry = DispatchEntry(
            index: 0,
            fragment: fragment,
            parameterBindings: [
                .init(role: "expert_down_proj", tensorName: tensorName),
            ]
        )
        let resolver = ProjectionWeightAccessPolicyResolver(
            override: .prefer(.optimized(.blockedRows8Tiles128), forTensorNames: [tensorName])
        )
        let builder = STAFSpecializedWeightStoreBuilder(
            device: device,
            accessPolicyResolver: resolver
        )

        let specializedStore = try #require(try builder.prepare(store: store, entries: [entry]))
        let access = try #require(specializedStore.bufferAccess(for: tensorName, layout: .blockedRows8Tiles128))

        #expect(access.layout == .blockedRows8Tiles128)
        #expect(access.size == payloadBytes)
    }

    @Test("Blocked down kernel matches CPU dot product")
    func blockedDownKernelMatchesCPUDotProduct() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }
        let tensorName = "tiny.expert_down_proj"
        let outputDimension = 8
        let intermediateDimension = 128
        let expertCount = 2
        let expertsPerToken = 2
        let selectedExperts = [1, 0]
        let scratchRowStride = 2 * expertsPerToken + 2 * 128 + expertsPerToken * intermediateDimension
        let activationOffset = 2 * expertsPerToken + 2 * 128

        let rowMajorPayload = (0..<(expertCount * outputDimension * intermediateDimension)).map { index in
            let value = Float((index * 17) % 31 - 15) * 0.00390625
            return BFloat16(value).bitPattern
        }
        let store = try makeStore(
            device: device,
            tensorName: tensorName,
            shape: [expertCount, outputDimension, intermediateDimension],
            payload: rowMajorPayload
        )
        let builder = STAFSpecializedWeightStoreBuilder(device: device)
        let blockedAccess = try builder.makeBlockedRows8Tiles128Access(for: tensorName, store: store)

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateSparseMoE(
                name: "test_sparse_moe_bf16",
                bufferPrecision: .float32Decode,
                weightFormat: .bfloat16,
                gateKind: .sigmoidTopK
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_sparse_moe_bf16_down_blocked8x128_packed4"))
        )

        let scratchBuffer = try #require(device.makeBuffer(
            length: scratchRowStride * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let selectedExpertPointer = scratchBuffer.contents().bindMemory(
            to: UInt32.self,
            capacity: scratchRowStride
        )
        for (index, expert) in selectedExperts.enumerated() {
            selectedExpertPointer[index] = UInt32(expert)
        }
        let activationPointer = scratchBuffer.contents().bindMemory(
            to: Float.self,
            capacity: scratchRowStride
        )
        var activations = [Float](repeating: .zero, count: expertsPerToken * intermediateDimension)
        for index in activations.indices {
            activations[index] = Float((index * 11) % 37 - 18) * 0.0078125
            activationPointer[activationOffset + index] = activations[index]
        }
        scratchBuffer.didModifyRange(0..<(scratchRowStride * MemoryLayout<Float>.stride))

        let outputBuffer = try #require(device.makeBuffer(
            length: outputDimension * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        let downSimdWidth = max(pipeline.threadExecutionWidth, 1)
        let downSimdgroups = max(1, min(32, pipeline.maxTotalThreadsPerThreadgroup / downSimdWidth))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(scratchBuffer, offset: 0, index: 0)
        encoder.setBuffer(blockedAccess.buffer, offset: blockedAccess.offset, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes([UInt32(outputDimension)], length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes([UInt32(intermediateDimension)], length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes([UInt32(expertsPerToken)], length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes([UInt32(1)], length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes([UInt32(outputDimension)], length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes([UInt32(scratchRowStride)], length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (outputDimension + downSimdgroups - 1) / downSimdgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: downSimdgroups * downSimdWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let actualPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputDimension)
        let actual = (0..<outputDimension).map { actualPointer[$0] }
        let expected = (0..<outputDimension).map { row in
            selectedExperts.enumerated().reduce(Float.zero) { total, pair in
                let (k, expert) = pair
                let rowBase = (expert * outputDimension + row) * intermediateDimension
                return total + (0..<intermediateDimension).reduce(Float.zero) { partial, middle in
                    let weight = BFloat16(bitPattern: rowMajorPayload[rowBase + middle]).floatValue
                    return partial + weight * activations[k * intermediateDimension + middle]
                }
            }
        }
        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }

        #expect(maxError < 0.001, "Blocked Sparse MoE down kernel drifted: maxError=\(maxError)")
    }

    private func makeDecodeKernelNames(
        weightLayouts: SparseMoEFragment.WeightLayoutSelection
    ) throws -> [String] {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return []
        }
        let fragment = makeA1BFragment()
        let kernelContext = KernelContext(
            bufferPrecision: .bfloat16,
            weightFormat: WeightFormats.bfloat16
        )
        let baseName = fragment.kernelName(context: kernelContext)
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + fragment.kernelSource(
                name: baseName,
                bufferPrecision: kernelContext.bufferPrecision,
                weightFormat: kernelContext.weightFormat
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        var pipelines: [String: MTLComputePipelineState] = [:]
        for functionName in [
            "\(baseName)_router_scores",
            "\(baseName)_router_select",
            "\(baseName)_router_parallel",
            "\(baseName)_router_parallel_staged_packed4",
            "\(baseName)_gate_up_staged_packed4",
            "\(baseName)_down_packed4",
            "\(baseName)_down_blocked8x128_packed4",
            "\(baseName)_down_blocked8x128_staged_act",
        ] {
            let function = try #require(library.makeFunction(name: functionName))
            pipelines[functionName] = try device.makeComputePipelineState(function: function)
        }

        let input = try makeBuffer(device: device, length: 1)
        let router = try makeBuffer(device: device, length: 1)
        let gateUp = try makeBuffer(device: device, length: 1)
        let down = try makeBuffer(device: device, length: 1)
        let bias = try makeBuffer(device: device, length: 1)
        let output = try makeBuffer(device: device, length: 1)
        let scratch = try makeBuffer(device: device, length: 1)

        let steps = try fragment.splitDecodeSteps(
            bindings: (
                buffers: [
                    (0, input, 0),
                    (1, router, 0),
                    (2, gateUp, 0),
                    (3, down, 0),
                    (4, bias, 0),
                    (5, output, 0),
                    (6, scratch, 0),
                ],
                bytes: []
            ),
            pipelineCache: pipelines,
            kernelContext: kernelContext,
            weightLayouts: weightLayouts
        )

        return steps.compactMap(\.metadata.kernelName)
    }

    private func makeA1BFragment() -> SparseMoEFragment {
        SparseMoEFragment(
            expertCount: 32,
            expertsPerToken: 4,
            gateKind: .sigmoidTopK,
            inputDimension: 2_048,
            outputDimension: 2_048,
            intermediateDimension: 1_792,
            normalizeRoutingWeights: true,
            routedScalingFactor: 1,
            useExpertBias: true
        )
    }

    private func makeStore(
        device: MTLDevice,
        tensorName: String,
        shape: [Int],
        payloadBytes: Int
    ) throws -> STAFWeightStore {
        let buffer = try #require(device.makeBuffer(
            length: max(payloadBytes, 1),
            options: .storageModeShared
        ))
        let entry = STAFTensorEntry(
            name: tensorName,
            payloadOffset: 0,
            payloadSize: payloadBytes,
            schemeIdentifier: .bf16RowMajor,
            semanticRole: .moeExpertDown,
            shape: shape,
            blockSize: 0,
            groupSize: 0,
            bufferOffset: 0
        )
        return STAFWeightStore(
            buffer: buffer,
            entries: [tensorName: entry],
            metadata: .empty,
            specializedBufferAccesses: [:]
        )
    }

    private func makeStore(
        device: MTLDevice,
        tensorName: String,
        shape: [Int],
        payload: [UInt16]
    ) throws -> STAFWeightStore {
        let payloadBytes = payload.count * MemoryLayout<UInt16>.stride
        let buffer = try #require(device.makeBuffer(
            length: max(payloadBytes, 1),
            options: .storageModeShared
        ))
        payload.withUnsafeBufferPointer { payloadPointer in
            guard let baseAddress = payloadPointer.baseAddress else { return }
            buffer.contents()
                .bindMemory(to: UInt16.self, capacity: payload.count)
                .update(from: baseAddress, count: payload.count)
        }
        buffer.didModifyRange(0..<payloadBytes)
        let entry = STAFTensorEntry(
            name: tensorName,
            payloadOffset: 0,
            payloadSize: payloadBytes,
            schemeIdentifier: .bf16RowMajor,
            semanticRole: .moeExpertDown,
            shape: shape,
            blockSize: 0,
            groupSize: 0,
            bufferOffset: 0
        )
        return STAFWeightStore(
            buffer: buffer,
            entries: [tensorName: entry],
            metadata: .empty,
            specializedBufferAccesses: [:]
        )
    }

    private func makeBuffer(device: MTLDevice, length: Int) throws -> MTLBuffer {
        try #require(device.makeBuffer(length: max(length, 1), options: .storageModeShared))
    }

    private func withSparseMoEEnvironmentCleared<T>(_ body: () throws -> T) rethrows -> T {
        try withEnvironmentValue("SWIFTLM_SPARSE_MOE_DISABLE_PACKED4", value: nil) {
            try withEnvironmentValue("SWIFTLM_SPARSE_MOE_ENABLE_PACKED8", value: nil) {
                try withEnvironmentValue("SWIFTLM_SPARSE_MOE_GATE_UP_ROW2", value: nil) {
                    try withEnvironmentValue("SWIFTLM_SPARSE_MOE_GATE_UP_SPLIT2", value: nil) {
                        try withEnvironmentValue("SWIFTLM_SPARSE_MOE_DOWN_SPLIT2", value: nil) {
                            try withEnvironmentValue("SWIFTLM_SPARSE_MOE_DISABLE_DOWN_STAGED_ACTIVATION", value: nil) {
                                try withEnvironmentValue("SWIFTLM_SPARSE_MOE_ENABLE_DOWN_STAGED_ACTIVATION", value: nil) {
                                    try body()
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func withEnvironmentValue<T>(
        _ key: String,
        value: String?,
        _ body: () throws -> T
    ) rethrows -> T {
        try TestEnvironmentVariables.withValue(key, value: value, body)
    }
}
