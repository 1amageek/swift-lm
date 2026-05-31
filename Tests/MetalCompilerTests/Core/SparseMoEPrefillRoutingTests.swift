import Darwin
import Metal
import Testing
@testable import MetalCompiler

@Suite("Sparse MoE Prefill Routing", .serialized)
struct SparseMoEPrefillRoutingTests {
    @Test("BF16 split prefill uses packed4 projection kernels")
    func bf16SplitPrefillUsesPacked4ProjectionKernels() throws {
        let names = try withSparseMoEEnvironmentCleared {
            try makePrefillKernelNames()
        }

        #expect(names == [
            "sparse_moe_seq_bf16_f32_router_scores",
            "sparse_moe_seq_bf16_f32_router_select",
            "sparse_moe_seq_bf16_f32_gate_up_packed4",
            "sparse_moe_seq_bf16_f32_down_packed4",
        ])
    }

    @Test("Split prefill ignores decode-only experimental projection routes")
    func splitPrefillIgnoresDecodeOnlyExperimentalProjectionRoutes() throws {
        let names = try withEnvironmentValue("SWIFTLM_SPARSE_MOE_ENABLE_PACKED8", value: "1") {
            try withEnvironmentValue("SWIFTLM_SPARSE_MOE_GATE_UP_ROW2", value: "1") {
                try withEnvironmentValue("SWIFTLM_SPARSE_MOE_GATE_UP_SPLIT2", value: "1") {
                    try withEnvironmentValue("SWIFTLM_SPARSE_MOE_DOWN_SPLIT2", value: "1") {
                        try withEnvironmentValue("SWIFTLM_SPARSE_MOE_DISABLE_PACKED4", value: nil) {
                            try makePrefillKernelNames()
                        }
                    }
                }
            }
        }

        #expect(names.contains("sparse_moe_seq_bf16_f32_gate_up_packed4"))
        #expect(names.contains("sparse_moe_seq_bf16_f32_down_packed4"))
        #expect(!names.contains("sparse_moe_seq_bf16_f32_gate_up_packed8"))
        #expect(!names.contains("sparse_moe_seq_bf16_f32_gate_up_row2_packed4"))
        #expect(!names.contains("sparse_moe_seq_bf16_f32_gate_up_split2"))
        #expect(!names.contains("sparse_moe_seq_bf16_f32_down_packed8"))
        #expect(!names.contains("sparse_moe_seq_bf16_f32_down_split2"))
    }

    @Test("Split prefill falls back to scalar projection kernels when packed4 is disabled")
    func splitPrefillFallsBackToScalarProjectionKernelsWhenPacked4IsDisabled() throws {
        let names = try withEnvironmentValue("SWIFTLM_SPARSE_MOE_DISABLE_PACKED4", value: "1") {
            try withEnvironmentValue("SWIFTLM_SPARSE_MOE_ENABLE_PACKED8", value: "1") {
                try makePrefillKernelNames()
            }
        }

        #expect(names.contains("sparse_moe_seq_bf16_f32_gate_up"))
        #expect(names.contains("sparse_moe_seq_bf16_f32_down"))
        #expect(!names.contains("sparse_moe_seq_bf16_f32_gate_up_packed4"))
        #expect(!names.contains("sparse_moe_seq_bf16_f32_down_packed4"))
        #expect(!names.contains("sparse_moe_seq_bf16_f32_gate_up_packed8"))
        #expect(!names.contains("sparse_moe_seq_bf16_f32_down_packed8"))
    }

    private func makePrefillKernelNames() throws -> [String] {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return []
        }

        let fragment = SparseMoEFragment(
            expertCount: 4,
            expertsPerToken: 2,
            gateKind: .sigmoidTopK,
            inputDimension: 16,
            outputDimension: 16,
            intermediateDimension: 16,
            normalizeRoutingWeights: true,
            routedScalingFactor: 1,
            useExpertBias: true
        )
        let baseName = fragment.kernelName(
            context: KernelContext(bufferPrecision: .float32, weightFormat: WeightFormats.bfloat16)
        )
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + fragment.kernelSource(
                name: baseName,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        var pipelines: [String: MTLComputePipelineState] = [:]
        for functionName in [
            baseName,
            "\(baseName)_router_scores",
            "\(baseName)_router_select",
            "\(baseName)_gate_up",
            "\(baseName)_gate_up_packed4",
            "\(baseName)_gate_up_packed8",
            "\(baseName)_gate_up_row2_packed4",
            "\(baseName)_gate_up_split2",
            "\(baseName)_down",
            "\(baseName)_down_packed4",
            "\(baseName)_down_packed8",
            "\(baseName)_down_split2",
        ] {
            let function = try #require(library.makeFunction(name: functionName))
            pipelines[functionName] = try device.makeComputePipelineState(function: function)
        }

        let maximumSequenceLength = 5
        let rowBytes = 16 * MemoryLayout<Float>.stride
        let hidden = try makeBuffer(device: device, length: maximumSequenceLength * rowBytes)
        let residual = try makeBuffer(device: device, length: maximumSequenceLength * rowBytes)
        let scratch = try makeBuffer(device: device, length: maximumSequenceLength * rowBytes * 2)
        let moeScratch = try makeBuffer(
            device: device,
            length: maximumSequenceLength * fragment.scratchElementsPerToken * MemoryLayout<Float>.stride
        )
        let logits = try makeBuffer(device: device, length: rowBytes)
        let tokenIDs = try makeBuffer(device: device, length: maximumSequenceLength * MemoryLayout<Int32>.stride)
        let positions = try makeBuffer(device: device, length: maximumSequenceLength * MemoryLayout<UInt32>.stride)
        let ropePositionAxes = try makeBuffer(device: device, length: maximumSequenceLength * MemoryLayout<UInt32>.stride)
        let tokenOut = try makeBuffer(device: device, length: MemoryLayout<Int32>.stride)
        let runtimeConstants = try makeBuffer(
            device: device,
            length: PrefillBufferSet.runtimeConstantBufferSize(maximumSequenceLength: maximumSequenceLength)
        )
        let weightBuffer = try makeBuffer(device: device, length: 16 * 16 * 8)
        let buffers = PrefillBufferSet(
            bufferPrecision: .float32,
            hidden: hidden,
            residual: residual,
            scratch: scratch,
            moeScratch: moeScratch,
            weights: [weightBuffer],
            kvCache: nil,
            convState: nil,
            recurrentState: nil,
            convStateDimension: 0,
            convStateKernelSize: 0,
            recurrentStateBytesPerLayer: 0,
            perLayerInputs: nil,
            perLayerInputDimension: 0,
            perLayerInputLayerCount: 0,
            logits: logits,
            tokenIDs: tokenIDs,
            positions: positions,
            ropePositionAxes: ropePositionAxes,
            tokenOut: tokenOut,
            ssmConvDebug: nil,
            dequantScratch: nil,
            compactProjectionScratch: nil,
            runtimeConstantBuffer: runtimeConstants
        )
        let context = PrefillBindingContext(
            buffers: buffers,
            slotDimension: 16,
            scratchElementSize: MemoryLayout<Float>.stride,
            maximumSequenceLength: maximumSequenceLength,
            currentInputBuffer: scratch,
            currentInputOffset: 0,
            layerIndex: 0,
            kvCacheIndex: 0,
            convLayerIndex: 0,
            recurrentLayerIndex: 0,
            kernelContext: KernelContext(bufferPrecision: .float32, weightFormat: WeightFormats.bfloat16),
            resolveWeight: { _ in (weightBuffer, 0) },
            getPipeline: { name in
                guard let pipeline = pipelines[name] else {
                    throw MetalCompilerError.kernelNotFound(name)
                }
                return pipeline
            }
        )

        return try fragment.prefillSteps(context: context).steps.map {
            try #require($0.metadata.kernelName)
        }
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
                            try body()
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
