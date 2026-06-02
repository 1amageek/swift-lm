import Darwin
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

        #expect(disabledRequest.preferredLayout == .rowMajor)
        #expect(packed8Request.preferredLayout == .rowMajor)
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
        let previous = ProcessInfo.processInfo.environment[key]
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let previous {
                setenv(key, previous, 1)
            } else {
                unsetenv(key)
            }
        }
        return try body()
    }
}
