import Metal
import Testing
import LMArchitecture
import LMIR
import ModelDeclarations
@testable import MetalCompiler

@Suite("Quantization Planning", .serialized)
struct QuantizationPlanningTests {

    @Test("q4 prefill projection uses MPP acceleration when strides are compatible")
    func q4PrefillProjectionUsesMPPWhenStridesAreCompatible() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstPrefillProjection(in: graph, device: device) {
            $0.inputDimension == config.intermediateSize
        }
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: target.tensorName,
            withShape: [target.outputDimension, target.inputDimension],
            schemeIdentifier: .q4Group64ScaleF16
        )

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        let quantizedEntry = try #require(
            plan.quantizationPlan.entries.first(where: { $0.tensorName == target.tensorName })
        )
        #expect(quantizedEntry.path == .prefillProjection)
        #expect(quantizedEntry.schemeIdentifier == .q4Group64ScaleF16)
        #expect(quantizedEntry.kernelFamily == .mppGEMM)
        #expect(!quantizedEntry.usedFallback)
        let prefillGEMM = try #require(quantizedEntry.prefillGEMM)
        #expect(prefillGEMM.selectedKernelName.contains("gemm"))
        #expect(prefillGEMM.inputDimension == target.inputDimension)
        #expect(prefillGEMM.outputDimension == target.outputDimension)
        #expect(prefillGEMM.inputRowStride == target.inputDimension)
        #expect(prefillGEMM.outputRowStride == target.outputDimension)
        #expect(prefillGEMM.maximumSequenceLength == 16)
    }

    @Test("dense prefill projection records runtime MPP tile variants")
    func densePrefillProjectionRecordsRuntimeMPPTileVariants() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstPrefillProjection(in: graph, device: device) {
            $0.inputDimension == config.intermediateSize
        }
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: target.tensorName,
            withShape: [target.outputDimension, target.inputDimension],
            schemeIdentifier: .fp16RowMajor
        )

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 128),
            stafWeightStore: store,
            device: device
        )

        let denseEntry = try #require(
            plan.quantizationPlan.entries.first(where: { $0.tensorName == target.tensorName })
        )
        #expect(denseEntry.kernelFamily == .mppGEMM)
        #expect(!denseEntry.usedFallback)
        let prefillGEMM = try #require(denseEntry.prefillGEMM)
        #expect(prefillGEMM.sequenceTileHeight == 64)
        #expect(prefillGEMM.tileVariantHeights == [16, 32, 64])

        let gemmStep = try #require(plan.steps.first {
            $0.metadata.kernelName?.contains("gemm") == true && !$0.tileVariants.isEmpty
        })
        #expect(gemmStep.resolvedDescriptor(sequenceLength: 16).pipeline.label?.hasSuffix("_mtile16") == true)
        #expect(gemmStep.resolvedDescriptor(sequenceLength: 33).pipeline.label?.hasSuffix("_mtile64") == true)
        #expect(gemmStep.resolvedDescriptor(sequenceLength: 96).pipeline.label?.hasSuffix("_mtile64") == true)
        #expect(gemmStep.resolvedGridSize(sequenceLength: 96).height == 2)

        let isolatedPlan = try plan.makeRuntimeIsolatedCopy(device: device)
        let isolatedGEMMStep = try #require(isolatedPlan.steps.first {
            $0.metadata.kernelName?.contains("gemm") == true && !$0.tileVariants.isEmpty
        })
        #expect(isolatedGEMMStep.tileVariants.map(\.tileHeight) == [16, 32, 64])
        #expect(isolatedGEMMStep.resolvedDescriptor(sequenceLength: 96).pipeline.label?.hasSuffix("_mtile64") == true)
    }

    @Test("dense batched prefill projection rejects MPP when scratch output stride is padded")
    func denseBatchedPrefillProjectionRejectsMPPWhenScratchOutputStrideIsPadded() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstBatchedProjection(in: graph, device: device)
        #expect(target.projections.allSatisfy { $0.outputDimension < config.intermediateSize })
        let overrides = Dictionary(uniqueKeysWithValues: target.projections.map {
            (
                $0.tensorName,
                TensorOverride(
                    shape: [$0.outputDimension, $0.inputDimension],
                    schemeIdentifier: .fp16RowMajor
                )
            )
        })
        let store = try makeWeightStore(for: graph, device: device, overrides: overrides)

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        let tensorNames = Set(target.projections.map(\.tensorName))
        let projectionEntries = plan.quantizationPlan.entries.filter {
            guard let tensorName = $0.tensorName else { return false }
            return tensorNames.contains(tensorName)
        }
        #expect(projectionEntries.count == target.projections.count)
        #expect(projectionEntries.allSatisfy { $0.kernelFamily != .mppGEMM })
        #expect(projectionEntries.allSatisfy { $0.prefillGEMM?.selectedKernelName.hasPrefix("batched_gemm") != true })
    }

    @Test("q4 prefill projection records direct kernel fallback when input stride is incompatible")
    func q4PrefillProjectionRecordsDirectKernelStrideFallback() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstPrefillProjection(in: graph, device: device) {
            $0.inputDimension != config.intermediateSize
        }
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: target.tensorName,
            withShape: [target.outputDimension, target.inputDimension],
            schemeIdentifier: .q4Group64ScaleF16
        )

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        let quantizedEntry = try #require(
            plan.quantizationPlan.entries.first(where: { $0.tensorName == target.tensorName })
        )
        #expect(quantizedEntry.path == .prefillProjection)
        #expect(quantizedEntry.schemeIdentifier == .q4Group64ScaleF16)
        #expect(quantizedEntry.kernelFamily == .q4G64GEMM)
        #expect(quantizedEntry.usedFallback)
        #expect(quantizedEntry.fallbackReason == .inputStrideMismatch)
        let prefillGEMM = try #require(quantizedEntry.prefillGEMM)
        #expect(prefillGEMM.selectedKernelName == "gemm_q4_g64_f32s")
        #expect(prefillGEMM.inputDimension == target.inputDimension)
        #expect(prefillGEMM.outputDimension == target.outputDimension)
        #expect(prefillGEMM.inputRowStride != target.inputDimension)
        #expect(prefillGEMM.outputRowStride >= target.outputDimension)
        #expect(prefillGEMM.maximumSequenceLength == 16)
    }

    @Test("dense prefill projection records disabled environment fallback when MPP is off")
    func densePrefillProjectionRecordsDisabledEnvironmentFallback() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstPrefillProjection(in: graph, device: device) {
            $0.inputDimension == config.intermediateSize
        }
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: target.tensorName,
            withShape: [target.outputDimension, target.inputDimension],
            schemeIdentifier: .fp16RowMajor
        )

        setenv("SWIFTLM_DISABLE_MPP", "1", 1)
        defer { unsetenv("SWIFTLM_DISABLE_MPP") }

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        let denseEntry = try #require(
            plan.quantizationPlan.entries.first(where: { $0.tensorName == target.tensorName })
        )
        #expect(denseEntry.schemeIdentifier == .fp16RowMajor)
        #expect(denseEntry.kernelFamily == .naiveGEMM)
        #expect(denseEntry.usedFallback)
        #expect(denseEntry.fallbackReason == .disabledByEnvironment)
        let prefillGEMM = try #require(denseEntry.prefillGEMM)
        #expect(prefillGEMM.selectedKernelName.hasPrefix("naive::gemm"))
        #expect(prefillGEMM.inputDimension == target.inputDimension)
        #expect(prefillGEMM.outputDimension == target.outputDimension)
        #expect(prefillGEMM.inputRowStride == target.inputDimension)
        #expect(prefillGEMM.outputRowStride == target.outputDimension)
        #expect(prefillGEMM.maximumSequenceLength == 16)
    }

    @Test("prefill diagnostics include quantization summary")
    func prefillDiagnosticsIncludeQuantizationSummary() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let store = try makeWeightStore(for: graph, device: device)
        let diagnostics = try MetalInferenceCompiler().dumpCompiledPrefillPlan(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        #expect(diagnostics.contains("quantization: entries="))
        #expect(diagnostics.contains("prefillAccel="))
        #expect(diagnostics.contains("kernelName="))
        #expect(diagnostics.contains(" in="))
        #expect(diagnostics.contains(" out="))
        #expect(diagnostics.contains(" seq=16"))
    }

    @Test("q4 prefill embedding lookup records quantized embedding kernel family")
    func q4PrefillEmbeddingLookupUsesQuantizedKernelFamily() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let embeddingBinding = try firstEmbeddingBinding(in: graph, device: device, phase: .prefill)
        let resolvedEmbeddingBinding = try #require(embeddingBinding)
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: resolvedEmbeddingBinding.tensorName,
            withShape: [config.vocabSize, config.hiddenSize],
            schemeIdentifier: .q4Group64ScaleF16
        )

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        let embeddingEntry = try #require(
            plan.quantizationPlan.entries.first {
                $0.tensorName == resolvedEmbeddingBinding.tensorName && $0.path == .embeddingLookup
            }
        )
        #expect(embeddingEntry.schemeIdentifier == .q4Group64ScaleF16)
        #expect(embeddingEntry.kernelFamily == .q4G64EmbeddingLookup)
        #expect(!embeddingEntry.usedFallback)
    }

    @Test("q3 prefill projection records kernel family and enables sequence ingestion")
    func q3PrefillProjectionEnablesSequencePromptIngestion() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstPrefillProjection(in: graph, device: device)
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: target.tensorName,
            withShape: [target.outputDimension, target.inputDimension],
            schemeIdentifier: .q3Group64ScaleF16
        )

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        let quantizedEntry = try #require(
            plan.quantizationPlan.entries.first(where: { $0.tensorName == target.tensorName })
        )
        #expect(quantizedEntry.path == .prefillProjection)
        #expect(quantizedEntry.schemeIdentifier == .q3Group64ScaleF16)
        #expect(quantizedEntry.kernelFamily == .naiveGEMM)
        #expect(quantizedEntry.usedFallback)
        #expect(quantizedEntry.fallbackReason == .inputStrideMismatch)
        let prefillGEMM = try #require(quantizedEntry.prefillGEMM)
        #expect(prefillGEMM.selectedKernelName.contains("gemm"))
        #expect(prefillGEMM.inputDimension == target.inputDimension)
        #expect(prefillGEMM.outputDimension == target.outputDimension)
        #expect(prefillGEMM.maximumSequenceLength == 16)
        #expect(!plan.requiresSequentialPromptIngestion)
        #expect(plan.sequencePrefillFallbackReason == nil)
    }

    @Test("all q3 prefill projection schemes enable sequence ingestion")
    func allQ3PrefillProjectionSchemesEnableSequencePromptIngestion() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let schemes: [QuantizationSchemeIdentifier] = [
            .q3Group16ScaleF16,
            .q3Group32ScaleF16,
            .q3Group64ScaleF16,
        ]
        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstPrefillProjection(in: graph, device: device)

        for scheme in schemes {
            let store = try makeWeightStore(
                for: graph,
                device: device,
                overriding: target.tensorName,
                withShape: [target.outputDimension, target.inputDimension],
                schemeIdentifier: scheme
            )

            let plan = try MetalInferenceCompiler().compilePrefill(
                graph: graph,
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                vocabSize: config.vocabSize,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
                stafWeightStore: store,
                device: device
            )

            let quantizedEntry = try #require(
                plan.quantizationPlan.entries.first(where: { $0.tensorName == target.tensorName })
            )
            #expect(quantizedEntry.path == .prefillProjection)
            #expect(quantizedEntry.schemeIdentifier == scheme)
            #expect(!plan.requiresSequentialPromptIngestion)
            #expect(plan.sequencePrefillFallbackReason == nil)
        }
    }

    @Test("all q3 prefill embedding schemes enable sequence ingestion")
    func allQ3PrefillEmbeddingSchemesEnableSequencePromptIngestion() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let schemes: [QuantizationSchemeIdentifier] = [
            .q3Group16ScaleF16,
            .q3Group32ScaleF16,
            .q3Group64ScaleF16,
        ]
        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let embeddingBinding = try firstEmbeddingBinding(in: graph, device: device, phase: .prefill)
        let resolvedEmbeddingBinding = try #require(embeddingBinding)

        for scheme in schemes {
            let store = try makeWeightStore(
                for: graph,
                device: device,
                overriding: resolvedEmbeddingBinding.tensorName,
                withShape: [config.vocabSize, config.hiddenSize],
                schemeIdentifier: scheme
            )

            let plan = try MetalInferenceCompiler().compilePrefill(
                graph: graph,
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                vocabSize: config.vocabSize,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
                stafWeightStore: store,
                device: device
            )

            let embeddingEntry = try #require(
                plan.quantizationPlan.entries.first {
                    $0.tensorName == resolvedEmbeddingBinding.tensorName && $0.path == .embeddingLookup
                }
            )
            #expect(embeddingEntry.schemeIdentifier == scheme)
            #expect(!plan.requiresSequentialPromptIngestion)
            #expect(plan.sequencePrefillFallbackReason == nil)
        }
    }

    @Test("q8 decode embedding lookup records quantized embedding kernel family")
    func q8DecodeEmbeddingLookupUsesQuantizedKernelFamily() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let embeddingBinding = try firstEmbeddingBinding(in: graph, device: device, phase: .decode)
        let resolvedEmbeddingBinding = try #require(embeddingBinding)
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: resolvedEmbeddingBinding.tensorName,
            withShape: [config.vocabSize, config.hiddenSize],
            schemeIdentifier: .q8Group32ScaleF16
        )

        let compiled = try MetalInferenceCompiler().compile(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            stafWeightStore: store,
            device: device
        )

        let embeddingEntry = try #require(
            compiled.decodePlan.quantizationPlan.entries.first {
                $0.tensorName == resolvedEmbeddingBinding.tensorName && $0.path == .embeddingLookup
            }
        )
        #expect(embeddingEntry.schemeIdentifier == .q8Group32ScaleF16)
        #expect(embeddingEntry.kernelFamily == .q8G32EmbeddingLookup)
        #expect(!embeddingEntry.usedFallback)
    }

    @Test("q4 decode keeps sibling projections batched")
    func q4DecodeKeepsSiblingProjectionsBatched() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstBatchedProjection(in: graph, device: device)
        let overrides = Dictionary(uniqueKeysWithValues: target.projections.map {
            (
                $0.tensorName,
                TensorOverride(
                    shape: [$0.outputDimension, $0.inputDimension],
                    schemeIdentifier: .q4Group64ScaleF16
                )
            )
        })
        let store = try makeWeightStore(for: graph, device: device, overrides: overrides)

        let compiled = try MetalInferenceCompiler().compile(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            stafWeightStore: store,
            device: device
        )

        let step = try #require(
            compiled.decodePlan.steps.first { $0.metadata.entryIndex == target.entry.index }
        )
        #expect(step.metadata.kernelName?.hasPrefix("batched_gemv") == true)
        #expect(step.metadata.kernelName?.contains("_q4_g64") == true)

        let tensorNames = Set(target.projections.map(\.tensorName))
        let quantizedEntries = compiled.decodePlan.quantizationPlan.entries.filter {
            guard let tensorName = $0.tensorName else { return false }
            return tensorNames.contains(tensorName)
        }
        #expect(quantizedEntries.count == target.projections.count)
        #expect(quantizedEntries.allSatisfy { $0.kernelFamily == .q4G64GEMV })
        #expect(quantizedEntries.allSatisfy { !$0.usedFallback })
    }

    @Test("decode diagnostics include quantization summary")
    func decodeDiagnosticsIncludeQuantizationSummary() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let store = try makeWeightStore(for: graph, device: device)
        let diagnostics = try MetalInferenceCompiler().dumpCompiledDecodePlan(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            stafWeightStore: store,
            device: device
        )

        #expect(diagnostics.contains("quantization: entries="))
    }

    @Test("non-aligned quantized kernels are classified by concrete family")
    func nonAlignedQuantizedKernelFamiliesAreClassified() {
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "embedding_lookup_seq_q3_g64",
            usesMPP: false
        ) == .q3G64EmbeddingLookup)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "gemv_q3_g64",
            usesMPP: false
        ) == .q3G64GEMV)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "gemm_q3_g64_f32s",
            usesMPP: false
        ) == .q3G64GEMM)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "embedding_lookup_seq_q5_g32",
            usesMPP: false
        ) == .q5G32EmbeddingLookup)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "gemv_q5_g64",
            usesMPP: false
        ) == .q5G64GEMV)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "gemm_q6_g16_f32s",
            usesMPP: false
        ) == .q6G16GEMM)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "batched_gemv3_q4_g64",
            usesMPP: false
        ) == .q4G64GEMV)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "batched_gemv2_q8_g32",
            usesMPP: false
        ) == .q8G32GEMV)
    }

    private func makeConfig() -> ModelConfig {
        ModelConfig(
            hiddenSize: 128,
            layerCount: 1,
            intermediateSize: 512,
            vocabSize: 1024,
            attentionHeads: 4,
            kvHeads: 4,
            headDim: 32,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-5,
            normKind: .rmsNorm,
            ropeTheta: 10000,
            ropeDimension: 32,
            ropeScaling: nil,
            tiedEmbeddings: true,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: nil,
            ssmNumHeads: nil,
            ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil,
            convKernelSize: nil,
            partialRotaryFactor: nil,
            slidingWindow: nil
        )
    }

    private func resolvedGraph(config: ModelConfig) throws -> ModelGraph {
        let graph = try ModelGraph(Transformer(config: config))
        return ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
    }

    private struct ProjectionTarget {
        let entry: DispatchEntry
        let tensorName: String
        let inputDimension: Int
        let outputDimension: Int
    }

    private struct BatchedProjectionTarget {
        let entry: DispatchEntry
        let projections: [ProjectionTarget]
    }

    private func firstPrefillProjection(
        in graph: ModelGraph,
        device: MTLDevice
    ) throws -> ProjectionTarget {
        try firstPrefillProjection(in: graph, device: device) { _ in true }
    }

    private func firstPrefillProjection(
        in graph: ModelGraph,
        device: MTLDevice,
        matching predicate: (ProjectionTarget) -> Bool
    ) throws -> ProjectionTarget {
        let context = CompileContext(
            graph: graph,
            hiddenSize: 128,
            intermediateSize: 512,
            vocabSize: 1024,
            inferencePolicy: .default,
            stafWeightStore: nil,
            device: device,
            weightFormat: .float16,
            decodeBufferPrecision: .float16,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver()
        )
        let entries = MetalEntryCollector().collect(
            using: context,
            kernelContext: context.prefillKernelContext
        ).fusedEntries

        for entry in entries {
            // Find standalone LinearFragment (e.g. o_proj, down_proj)
            if let projection = entry.fragment as? LinearFragment,
               let binding = entry.parameterBindings.first(where: { $0.role == projection.field }) {
                let target = ProjectionTarget(
                    entry: entry,
                    tensorName: binding.tensorName,
                    inputDimension: projection.inputDimension,
                    outputDimension: projection.outputDimension
                )
                if predicate(target) {
                    return target
                }
            }
        }

        throw QuantizationPlanningError.missingProjection
    }

    private func firstBatchedProjection(
        in graph: ModelGraph,
        device: MTLDevice
    ) throws -> BatchedProjectionTarget {
        let context = CompileContext(
            graph: graph,
            hiddenSize: 128,
            intermediateSize: 512,
            vocabSize: 1024,
            inferencePolicy: .default,
            stafWeightStore: nil,
            device: device,
            weightFormat: .float16,
            decodeBufferPrecision: .float16,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver()
        )
        let entries = MetalEntryCollector().collect(
            using: context,
            kernelContext: context.decodeKernelContext
        ).fusedEntries

        for entry in entries {
            guard let batched = entry.fragment as? BatchedProjection else {
                continue
            }
            let projections = batched.projections.compactMap { projection -> ProjectionTarget? in
                guard let binding = entry.parameterBindings.first(where: { $0.role == projection.field }) else {
                    return nil
                }
                return ProjectionTarget(
                    entry: entry,
                    tensorName: binding.tensorName,
                    inputDimension: projection.inputDimension,
                    outputDimension: projection.outputDimension
                )
            }
            if projections.count == batched.projections.count {
                return BatchedProjectionTarget(entry: entry, projections: projections)
            }
        }

        throw QuantizationPlanningError.missingProjection
    }

    private func firstEmbeddingBinding(
        in graph: ModelGraph,
        device: MTLDevice,
        phase: STAFWeightExecutionPhase
    ) throws -> ParameterBinding? {
        let context = CompileContext(
            graph: graph,
            hiddenSize: 128,
            intermediateSize: 512,
            vocabSize: 1024,
            inferencePolicy: .default,
            stafWeightStore: nil,
            device: device,
            weightFormat: .float16,
            decodeBufferPrecision: .float16,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver()
        )
        let kernelContext: KernelContext
        switch phase {
        case .decode:
            kernelContext = context.decodeKernelContext
        case .prefill:
            kernelContext = context.prefillKernelContext
        }
        let entries = MetalEntryCollector().collect(
            using: context,
            kernelContext: kernelContext
        ).fusedEntries

        for entry in entries {
            guard entry.fragment is GatherFragment else {
                continue
            }
            return entry.parameterBindings.first(where: { $0.role == "embedding_table" })
        }
        return nil
    }

    private func makeWeightStore(
        for graph: ModelGraph,
        device: MTLDevice,
        overriding tensorName: String? = nil,
        withShape shape: [Int] = [1],
        schemeIdentifier: QuantizationSchemeIdentifier = .passthrough
    ) throws -> STAFWeightStore {
        let overrides = tensorName.map {
            [
                $0: TensorOverride(
                    shape: shape,
                    schemeIdentifier: schemeIdentifier
                )
            ]
        } ?? [:]
        return try makeWeightStore(for: graph, device: device, overrides: overrides)
    }

    private struct TensorOverride {
        let shape: [Int]
        let schemeIdentifier: QuantizationSchemeIdentifier
    }

    private func makeWeightStore(
        for graph: ModelGraph,
        device: MTLDevice,
        overrides: [String: TensorOverride]
    ) throws -> STAFWeightStore {
        let overridePayloadSize = max(
            1,
            overrides.values.map { $0.shape.reduce(1, *) }.max() ?? 1
        ) * MemoryLayout<UInt16>.size
        let buffer = try #require(device.makeBuffer(length: overridePayloadSize, options: .storageModeShared))
        var entries: [String: STAFTensorEntry] = [:]
        for name in tensorNames(in: graph.rootRegion) {
            let tensorOverride = overrides[name]
            let isOverride = tensorOverride != nil
            let entryShape = tensorOverride?.shape ?? [1]
            entries[name] = STAFTensorEntry(
                name: name,
                payloadOffset: 0,
                payloadSize: isOverride ? overridePayloadSize : buffer.length,
                schemeIdentifier: tensorOverride?.schemeIdentifier ?? .passthrough,
                semanticRole: .other,
                shape: entryShape,
                blockSize: 64,
                groupSize: 64,
                bufferOffset: 0
            )
        }

        return STAFWeightStore(
            buffer: buffer,
            entries: entries,
            metadata: .empty,
            specializedBufferAccesses: [:]
        )
    }

    private func tensorNames(in region: Region) -> Set<String> {
        var names = Set(region.operations.flatMap { operation in
            operation.parameterBindings.map(\.tensorName)
        })
        for operation in region.operations {
            switch operation.kind {
            case .primitive:
                break
            case .residual(_, let body):
                names.formUnion(tensorNames(in: body))
            case .parallel(_, let branches):
                for branch in branches {
                    names.formUnion(tensorNames(in: branch))
                }
            case .repeating(_, let body):
                names.formUnion(tensorNames(in: body))
            case .conditional(_, let thenRegion, let elseRegion):
                names.formUnion(tensorNames(in: thenRegion))
                names.formUnion(tensorNames(in: elseRegion))
            }
        }
        return names
    }
}

private enum QuantizationPlanningError: Error {
    case missingProjection
}
