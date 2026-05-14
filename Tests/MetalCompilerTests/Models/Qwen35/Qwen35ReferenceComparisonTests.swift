import Foundation
import CryptoKit
import Metal
import Testing
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
@Suite("Qwen35 Reference Comparison", .serialized)
struct Qwen35ReferenceComparisonTests {
    private struct ReferenceCase: Sendable {
        let index: Int
        let tokens: [Int32]

        var prefix: String {
            "ref.case_\(index)"
        }
    }

    private static let defaultPromptTokens: [Int32] = [
        248045, 846, 198, 3710, 369, 279, 6511, 314, 6124, 30,
        248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ]

    private static let referenceCases = [
        ReferenceCase(index: 0, tokens: defaultPromptTokens),
        ReferenceCase(index: 1, tokens: Array(defaultPromptTokens.prefix(8))),
    ]
    private static let expectedLinearBlockOrdinals: [Int32] = [0, 9, 17]

    private static let referencePath = URL(fileURLWithPath: BenchmarkSupport.testDataPath)
        .appendingPathComponent("qwen35_reference.safetensors")
        .path

    @Test("Reference snapshot schema is complete")
    func referenceSnapshotSchemaIsComplete() throws {
        let env = try Self.setupReferenceOrSkip()
        let requiredTensors = [
            "ref.meta.schema_version",
            "ref.meta.case_count",
            "ref.meta.decode_steps",
            "ref.meta.linear_block_ordinals",
            "ref.meta.config_sha256",
            "ref.meta.torch_version_utf8",
            "ref.meta.transformers_version_utf8",
            "ref.meta.fast_backend_available",
        ]
        for name in requiredTensors {
            #expect(env.ref.tensors[name] != nil, "Missing Qwen35 reference tensor: \(name)")
        }

        let schemaVersion = try Self.readRefInt32(env.ref, name: "ref.meta.schema_version")
        let caseCount = try Self.readRefInt32(env.ref, name: "ref.meta.case_count")
        let decodeSteps = try Self.readRefInt32(env.ref, name: "ref.meta.decode_steps")
        let fastBackendAvailable = try Self.readRefInt32(env.ref, name: "ref.meta.fast_backend_available")
        let linearBlockOrdinals = try Self.readRefInt32Array(env.ref, name: "ref.meta.linear_block_ordinals")
        let referenceConfigHash = try Self.readRefUInt8Array(env.ref, name: "ref.meta.config_sha256")
        let torchVersion = try Self.readRefUInt8Array(env.ref, name: "ref.meta.torch_version_utf8")
        let transformersVersion = try Self.readRefUInt8Array(env.ref, name: "ref.meta.transformers_version_utf8")
        let bundleConfigHash = try Self.configSHA256(bundlePath: env.bundlePath)

        #expect(schemaVersion == 6, "Unexpected Qwen35 reference schema version: \(schemaVersion)")
        #expect(Int(caseCount) == Self.referenceCases.count)
        #expect(decodeSteps >= 1)
        #expect(fastBackendAvailable == 0 || fastBackendAvailable == 1)
        #expect(linearBlockOrdinals == Self.expectedLinearBlockOrdinals)
        #expect(!torchVersion.isEmpty)
        #expect(!transformersVersion.isEmpty)
        #expect(referenceConfigHash == bundleConfigHash, "Qwen35 reference config does not match the Swift bundle config")

        for ordinal in linearBlockOrdinals {
            let layerIndex = try Self.readRefInt32(
                env.ref,
                name: "ref.meta.linear_ordinal_\(ordinal).layer_index"
            )
            #expect(layerIndex >= 0)
            let partitionCount = try Self.readRefInt32(
                env.ref,
                name: "ref.meta.linear_ordinal_\(ordinal).partition_count"
            )
            #expect(partitionCount > 1)
        }

        for referenceCase in Self.referenceCases {
            let prefix = referenceCase.prefix
            let requiredCaseTensors = [
                "\(prefix).meta.input_tokens",
                "\(prefix).meta.prefill_token_count",
                "\(prefix).prefill.embedding",
                "\(prefix).prefill.final_hidden",
                "\(prefix).prefill.logits_last",
                "\(prefix).prefill.next_token",
                "\(prefix).prefill.layer_0.after_op",
                "\(prefix).prefill.layer_3.after_op",
                "\(prefix).prefill.conv_state.0",
                "\(prefix).prefill.recurrent_state.0",
                "\(prefix).prefill.attn_layer_3.keys",
                "\(prefix).decode_0.final_hidden",
                "\(prefix).decode_0.logits_last",
                "\(prefix).decode_0.next_token",
            ]
            for name in requiredCaseTensors {
                #expect(env.ref.tensors[name] != nil, "Missing Qwen35 reference tensor: \(name)")
            }

            let tokenCount = try Self.readRefInt32(env.ref, name: "\(prefix).meta.prefill_token_count")
            let referenceTokens = try Self.readRefInt32Array(env.ref, name: "\(prefix).meta.input_tokens")
            #expect(Int(tokenCount) == referenceCase.tokens.count)
            #expect(referenceTokens == referenceCase.tokens)

            for ordinal in linearBlockOrdinals {
                let requiredBlockTensors = [
                    "\(prefix).prefill.linear_ordinal_\(ordinal).block.projected_qkv",
                    "\(prefix).prefill.linear_ordinal_\(ordinal).block.projected_z",
                    "\(prefix).prefill.linear_ordinal_\(ordinal).block.projected_beta",
                    "\(prefix).prefill.linear_ordinal_\(ordinal).block.projected_alpha",
                    "\(prefix).prefill.linear_ordinal_\(ordinal).block.conv_silu",
                    "\(prefix).prefill.linear_ordinal_\(ordinal).block.gated_recurrent_output",
                    "\(prefix).prefill.linear_ordinal_\(ordinal).block.out_projection",
                    "\(prefix).prefill.linear_ordinal_\(ordinal).block.out_projection_partials",
                    "\(prefix).prefill.linear_ordinal_\(ordinal).block.out_projection_reduced",
                ]
                for name in requiredBlockTensors {
                    #expect(env.ref.tensors[name] != nil, "Missing Qwen35 reference tensor: \(name)")
                }
            }
        }
    }

    @Test("Prefill linear-attention block boundaries match HuggingFace reference")
    func prefillLinearAttentionBlockBoundariesMatchReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let env = try Self.setupOrSkip(enableSSMConvDebug: true, loadSTAF: true)
        let referenceCase = Self.referenceCases[0]
        let linearBlockOrdinals = try Self.readRefInt32Array(env.ref, name: "ref.meta.linear_block_ordinals")
        #expect(!linearBlockOrdinals.isEmpty)

        var model = try Self.makeRuntimeIsolatedModel(from: env.model)
        let prefillPlan = try #require(model.prefillPlan)

        var lastProbeStepIndex = 0
        var stages: [LinearAttentionBoundaryStage] = []
        var fanInChecks: [LinearAttentionFanInCheck] = []
        for ordinal in linearBlockOrdinals.map(Int.init) {
            let layerIndex = Int(try Self.readRefInt32(
                env.ref,
                name: "ref.meta.linear_ordinal_\(ordinal).layer_index"
            ))
            let partitionCount = Int(try Self.readRefInt32(
                env.ref,
                name: "ref.meta.linear_ordinal_\(ordinal).partition_count"
            ))
            let steps = try Self.linearAttentionBoundarySteps(
                prefillPlan: prefillPlan,
                layerIndex: layerIndex
            )
            lastProbeStepIndex = max(lastProbeStepIndex, steps.outProjection)
            if let partialProjection = steps.partialProjection {
                lastProbeStepIndex = max(lastProbeStepIndex, partialProjection)
            }
            stages.append(contentsOf: Self.linearAttentionBoundaryStages(
                referencePrefix: referenceCase.prefix,
                ordinal: ordinal,
                steps: steps,
                slotDimension: prefillPlan.slotDimension
            ))
            fanInChecks.append(LinearAttentionFanInCheck(
                referencePrefix: referenceCase.prefix,
                ordinal: ordinal,
                layerIndex: layerIndex,
                partitionCount: partitionCount,
                partialProjectionStepIndex: steps.partialProjection,
                partialProjectionBindingIndex: steps.partialProjectionBindingIndex
            ))
        }

        var probes: [MetalInferenceModel.DebugPrefillBindingProbe] = []
        for stage in stages {
            for rowIndex in referenceCase.tokens.indices {
                probes.append(MetalInferenceModel.DebugPrefillBindingProbe(
                    label: "\(stage.name).row_\(rowIndex)",
                    stepIndex: stage.stepIndex,
                    bindingIndex: stage.bindingIndex,
                    phase: .afterStep,
                    rowIndex: rowIndex,
                    rowStride: stage.rowStride,
                    count: stage.count,
                    precision: .float32
                ))
            }
        }
        for check in fanInChecks {
            guard let partialProjectionStepIndex = check.partialProjectionStepIndex else { continue }
            for partition in 0..<check.partitionCount {
                for rowIndex in referenceCase.tokens.indices {
                    probes.append(MetalInferenceModel.DebugPrefillBindingProbe(
                        label: check.partialProbeLabel(partition: partition, rowIndex: rowIndex),
                        stepIndex: partialProjectionStepIndex,
                        bindingIndex: check.partialProjectionBindingIndex,
                        phase: .afterStep,
                        rowIndex: partition * referenceCase.tokens.count + rowIndex,
                        rowStride: prefillPlan.slotDimension,
                        count: 1024,
                        precision: .float32
                    ))
                }
            }
        }

        model.resetState()
        let snapshots = try model.debugPrefillBindingProbes(
            tokens: referenceCase.tokens,
            stepIndex: lastProbeStepIndex,
            probes: probes
        )

        for stage in stages {
            let referenceValues = try Self.readRefTensorAsFloats(env.ref, name: stage.referenceName)
            let referenceRowStride = referenceValues.count / referenceCase.tokens.count
            #expect(
                referenceValues.count % referenceCase.tokens.count == 0,
                "\(stage.referenceName) cannot be split by token rows"
            )
            #expect(
                referenceRowStride >= stage.count,
                "\(stage.referenceName) row stride \(referenceRowStride) is smaller than expected count \(stage.count)"
            )

            var worstError: Float = 0
            for rowIndex in referenceCase.tokens.indices {
                guard let metalValues = snapshots["\(stage.name).row_\(rowIndex)"] else {
                    throw SetupError.tensorNotFound("metal.\(stage.name).row_\(rowIndex)")
                }
                let referenceStart = rowIndex * referenceRowStride
                let reference = Array(referenceValues[referenceStart..<(referenceStart + stage.count)])
                let error = Self.maxAbsoluteError(metalValues, reference)
                worstError = max(worstError, error)
                #expect(
                    error <= stage.tolerance,
                    "\(stage.name) row \(rowIndex) drifted: maxErr=\(error)"
                )
            }
            print("[Qwen35Ref] \(referenceCase.prefix).prefill.\(stage.name) maxErr=\(String(format: "%.4f", worstError))")
        }

        for check in fanInChecks {
            try Self.expectLinearAttentionFanInMatchesReference(
                check: check,
                env: env,
                snapshots: snapshots,
                tokenCount: referenceCase.tokens.count
            )
        }
    }

    @Test("Prefill final hidden and logits match HuggingFace reference")
    func prefillFinalHiddenAndLogitsMatchReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let env = try Self.setupOrSkip()
        let prefillGate = try #require(env.model.prefillPlan)
        #expect(!prefillGate.requiresSequentialPromptIngestion)
        #expect(prefillGate.sequencePrefillFallbackReason == nil)

        for referenceCase in Self.referenceCases {
            var hiddenModel = try Self.makeRuntimeIsolatedModel(from: env.model)
            let metalFinalHidden = try hiddenModel.debugPrefillLastTokenFinalHidden(tokens: referenceCase.tokens)

            let refFinalHiddenAll = try Self.readRefTensorAsFloats(env.ref, name: "\(referenceCase.prefix).prefill.final_hidden")
            let hiddenSize = metalFinalHidden.count
            let refFinalHidden = Array(refFinalHiddenAll.suffix(hiddenSize))
            let finalHiddenError = Self.maxAbsoluteError(metalFinalHidden, refFinalHidden)

            var logitsModel = try Self.makeRuntimeIsolatedModel(from: env.model)
            logitsModel.resetState()
            let metalToken = logitsModel.prefill(tokens: referenceCase.tokens)
            let refToken = try Self.readRefInt32(env.ref, name: "\(referenceCase.prefix).prefill.next_token")

            let prefillPlan = try #require(logitsModel.prefillPlan)
            let metalLogits = try Self.readBuffer(prefillPlan.buffers.logits, precision: .float32)
            let refLogits = try Self.readRefTensorAsFloats(env.ref, name: "\(referenceCase.prefix).prefill.logits_last")
            let metalTop = Self.argmax(metalLogits)
            let refTop = Self.argmax(refLogits)
            let logitsError = Self.maxAbsoluteError(metalLogits, refLogits)

            print("[Qwen35Ref] \(referenceCase.prefix).prefill finalHidden maxErr=\(String(format: "%.4f", finalHiddenError))")
            print("[Qwen35Ref] \(referenceCase.prefix).prefill logits maxErr=\(String(format: "%.4f", logitsError))")
            print("[Qwen35Ref] \(referenceCase.prefix).prefill Metal token=\(metalToken) top=\(metalTop.index), HF token=\(refToken) top=\(refTop.index)")

            #expect(metalToken == refToken, "\(referenceCase.prefix).prefill next token drifted: Metal=\(metalToken) HF=\(refToken)")
            #expect(metalTop.index == refTop.index, "\(referenceCase.prefix).prefill argmax drifted: Metal=\(metalTop.index) HF=\(refTop.index)")
            #expect(finalHiddenError < 1.25, "\(referenceCase.prefix).prefill final hidden drifted: maxErr=\(finalHiddenError)")
            #expect(logitsError < 1.0, "\(referenceCase.prefix).prefill logits drifted: maxErr=\(logitsError)")
        }
    }

    @Test("Prefill state matches HuggingFace reference")
    func prefillStateMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let env = try Self.setupOrSkip()
        for referenceCase in Self.referenceCases {
            var model = try Self.makeRuntimeIsolatedModel(from: env.model)
            model.resetState()
            _ = model.prefill(tokens: referenceCase.tokens)

            guard let convState = model.buffers.convState else {
                Issue.record("Qwen35 decode plan did not allocate conv state")
                return
            }
            guard let recurrentState = model.buffers.recurrentState else {
                Issue.record("Qwen35 decode plan did not allocate recurrent state")
                return
            }

            let convDim = model.buffers.convStateDimension
            let kernelSize = model.buffers.convStateKernelSize
            let convLayerCount = convState.length / (convDim * kernelSize * MemoryLayout<BFloat16>.stride)
            let metalConv = try Self.readBuffer(convState, precision: .bfloat16)
            let prefillPrefix = "\(referenceCase.prefix).prefill"
            let referenceLinearCount = Self.referenceLinearOrdinalCount(env.ref, prefix: prefillPrefix)
            #expect(referenceLinearCount == convLayerCount)

            for ordinal in 0..<min(referenceLinearCount, convLayerCount) {
                let ref = try Self.readRefTensorAsFloats(env.ref, name: "\(prefillPrefix).linear_ordinal_\(ordinal).conv_state")
                let base = ordinal * convDim * kernelSize
                let metal = Array(metalConv[base..<(base + convDim * kernelSize)])
                let error = Self.maxAbsoluteError(metal, ref)
                print("[Qwen35Ref] \(prefillPrefix).linear_ordinal_\(ordinal).conv_state maxErr=\(String(format: "%.4f", error))")
                #expect(error < 1.25, "\(prefillPrefix).linear_ordinal_\(ordinal).conv_state drifted: maxErr=\(error)")
            }

            let recurrentLayerValues = model.buffers.recurrentStateBytesPerLayer / MemoryLayout<Float>.stride
            let recurrentLayerCount = recurrentState.length / model.buffers.recurrentStateBytesPerLayer
            let metalRecurrent = try Self.readBuffer(recurrentState, precision: .float32)
            #expect(referenceLinearCount == recurrentLayerCount)
            for ordinal in 0..<min(referenceLinearCount, recurrentLayerCount) {
                let ref = try Self.readRefTensorAsFloats(env.ref, name: "\(prefillPrefix).linear_ordinal_\(ordinal).recurrent_state")
                let base = ordinal * recurrentLayerValues
                let metal = Array(metalRecurrent[base..<(base + recurrentLayerValues)])
                let error = Self.maxAbsoluteError(metal, ref)
                print("[Qwen35Ref] \(prefillPrefix).linear_ordinal_\(ordinal).recurrent_state maxErr=\(String(format: "%.4f", error))")
                #expect(error < 0.75, "\(prefillPrefix).linear_ordinal_\(ordinal).recurrent_state drifted: maxErr=\(error)")
            }

            try Self.expectKVCacheMatchesReference(
                model: model,
                ref: env.ref,
                prefix: prefillPrefix,
                tokenCount: referenceCase.tokens.count
            )
        }
    }

    @Test("Decode step zero matches HuggingFace reference")
    func decodeStepZeroMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let env = try Self.setupOrSkip()
        for referenceCase in Self.referenceCases {
            var model = try Self.makeRuntimeIsolatedModel(from: env.model)
            model.resetState()
            let prefillToken = model.prefill(tokens: referenceCase.tokens)
            let metalToken = model.decodeSync(tokenID: prefillToken)
            let decodePrefix = "\(referenceCase.prefix).decode_0"
            let refToken = try Self.readRefInt32(env.ref, name: "\(decodePrefix).next_token")
            let refLogits = try Self.readRefTensorAsFloats(env.ref, name: "\(decodePrefix).logits_last")
            let metalLogits = try Self.readBuffer(model.buffers.logits, precision: model.buffers.bufferPrecision)
            let metalTop = Self.argmax(metalLogits)
            let refTop = Self.argmax(refLogits)
            let logitsError = Self.maxAbsoluteError(metalLogits, refLogits)

            print("[Qwen35Ref] \(decodePrefix) Metal token=\(metalToken) top=\(metalTop.index), HF token=\(refToken) top=\(refTop.index), logits maxErr=\(String(format: "%.4f", logitsError))")

            #expect(metalToken == refToken, "\(decodePrefix) next token drifted: Metal=\(metalToken) HF=\(refToken)")
            #expect(metalTop.index == refTop.index, "\(decodePrefix) argmax drifted: Metal=\(metalTop.index) HF=\(refTop.index)")
            #expect(logitsError < 1.0, "\(decodePrefix) logits drifted: maxErr=\(logitsError)")

            try Self.expectLinearStatesMatchReference(
                model: model,
                ref: env.ref,
                prefix: decodePrefix
            )
            try Self.expectKVCacheMatchesReference(
                model: model,
                ref: env.ref,
                prefix: decodePrefix,
                tokenCount: referenceCase.tokens.count + 1
            )
        }
    }

    private struct TestEnvironment {
        let model: MetalInferenceModel
        let ref: MetalWeightFile
        let staf: STAFWeightStore?
        let bundlePath: String
    }

    private struct ReferenceEnvironment {
        let ref: MetalWeightFile
        let bundlePath: String
    }

    private struct LinearAttentionBoundarySteps {
        let projection: Int
        let recurrence: Int
        let partialProjection: Int?
        let partialProjectionBindingIndex: Int
        let outProjection: Int
        let outProjectionBindingIndex: Int
    }

    private struct LinearAttentionBoundaryStage {
        let name: String
        let referenceName: String
        let stepIndex: Int
        let bindingIndex: Int
        let rowStride: Int
        let count: Int
        let tolerance: Float
    }

    private struct LinearAttentionFanInCheck {
        let referencePrefix: String
        let ordinal: Int
        let layerIndex: Int
        let partitionCount: Int
        let partialProjectionStepIndex: Int?
        let partialProjectionBindingIndex: Int

        var labelPrefix: String {
            "linear_ordinal_\(ordinal).block"
        }

        var referenceBlockPrefix: String {
            "\(referencePrefix).prefill.linear_ordinal_\(ordinal).block"
        }

        func gatedLabel(rowIndex: Int) -> String {
            "\(labelPrefix).gated_recurrent_output.row_\(rowIndex)"
        }

        func partialProbeLabel(partition: Int, rowIndex: Int) -> String {
            "\(labelPrefix).out_projection_partials.partition_\(partition).row_\(rowIndex)"
        }
    }

    private static func setupReferenceOrSkip() throws -> ReferenceEnvironment {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SetupError.noDevice
        }

        let refURL = URL(fileURLWithPath: referencePath)
        guard FileManager.default.fileExists(atPath: refURL.path) else {
            Issue.record("Qwen35 reference not found. Run: python3 scripts/hf/dump_qwen35_reference.py")
            throw SetupError.noReference
        }

        guard let bundlePath = try resolveBundle() else {
            Issue.record("Qwen35 original bundle is not cached. Expected ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B")
            throw SetupError.noBundle
        }

        let ref = try SafetensorsLoader().load(at: refURL, device: device)
        return ReferenceEnvironment(ref: ref, bundlePath: bundlePath)
    }

    private static func setupOrSkip(
        enableSSMConvDebug: Bool = false,
        loadSTAF: Bool = false
    ) throws -> TestEnvironment {
        let referenceEnv = try setupReferenceOrSkip()
        let previousDebugValue = getenv("SWIFTLM_PREFILL_DEBUG_SSM_CONV").map { String(cString: $0) }
        if enableSSMConvDebug {
            setenv("SWIFTLM_PREFILL_DEBUG_SSM_CONV", "1", 1)
        }
        defer {
            if let previousDebugValue {
                setenv("SWIFTLM_PREFILL_DEBUG_SSM_CONV", previousDebugValue, 1)
            } else {
                unsetenv("SWIFTLM_PREFILL_DEBUG_SSM_CONV")
            }
        }
        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: referenceEnv.bundlePath,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64)
        )
        let staf: STAFWeightStore?
        if loadSTAF {
            let stafURL = URL(fileURLWithPath: referenceEnv.bundlePath).appendingPathComponent("model.staf")
            guard FileManager.default.fileExists(atPath: stafURL.path) else {
                throw SetupError.tensorNotFound(stafURL.path)
            }
            staf = try STAFLoader().load(at: stafURL, device: model.device)
        } else {
            staf = nil
        }
        return TestEnvironment(
            model: model,
            ref: referenceEnv.ref,
            staf: staf,
            bundlePath: referenceEnv.bundlePath
        )
    }

    private static func resolveBundle() throws -> String? {
        if let envPath = ProcessInfo.processInfo.environment["SWIFTLM_QWEN35_REFERENCE_MODEL"],
           FileManager.default.fileExists(atPath: envPath) {
            return envPath
        }

        let hubRoot = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        let snapshotsDir = "\(hubRoot)/models--Qwen--Qwen3.5-0.8B/snapshots"
        guard FileManager.default.fileExists(atPath: snapshotsDir) else { return nil }
        let entries = try FileManager.default.contentsOfDirectory(atPath: snapshotsDir).sorted()
        for entry in entries {
            let candidate = "\(snapshotsDir)/\(entry)"
            if FileManager.default.fileExists(atPath: "\(candidate)/config.json") {
                return candidate
            }
        }
        return nil
    }

    private static func makeRuntimeIsolatedModel(from model: MetalInferenceModel) throws -> MetalInferenceModel {
        let isolated = try model.compiledModel.makeRuntimeIsolatedCopy(device: model.device)
        return try MetalInferenceModel(compiledModel: isolated, device: model.device)
    }

    private static func linearAttentionBoundarySteps(
        prefillPlan: MetalPrefillPlan,
        layerIndex: Int
    ) throws -> LinearAttentionBoundarySteps {
        let layerToken = ".layers.\(layerIndex).linear_attn."
        guard let projection = prefillPlan.steps.firstIndex(where: { step in
            step.metadata.weightTensorName?.contains("\(layerToken)in_proj_qkv.weight") == true
        }) else {
            throw SetupError.stepNotFound("linear attention projection for layer \(layerIndex)")
        }
        guard let recurrence = prefillPlan.steps.indices.first(where: { index in
            guard index > projection else { return false }
            let step = prefillPlan.steps[index]
            return (step.metadata.kernelName ?? step.pipeline.label ?? "").contains("ssm_recurrence_seq")
        }) else {
            throw SetupError.stepNotFound("linear attention recurrence for layer \(layerIndex)")
        }
        let outProjectionCandidates = prefillPlan.steps.indices.filter { index in
            let step = prefillPlan.steps[index]
            return step.metadata.weightTensorName?.contains("\(layerToken)out_proj.weight") == true
        }
        guard let firstOutProjection = outProjectionCandidates.first else {
            throw SetupError.stepNotFound("linear attention output projection for layer \(layerIndex)")
        }
        guard recurrence < firstOutProjection else {
            throw SetupError.stepNotFound("linear attention recurrence before output projection for layer \(layerIndex)")
        }
        let firstOutProjectionKernel = prefillPlan.steps[firstOutProjection].metadata.kernelName
            ?? prefillPlan.steps[firstOutProjection].pipeline.label
            ?? ""
        let partialProjection: Int?
        let partialProjectionBindingIndex: Int
        let outProjection: Int
        let outProjectionBindingIndex: Int
        if firstOutProjectionKernel.hasPrefix("recurrent_block_partial_projection") {
            guard let reduceStep = outProjectionCandidates.dropFirst().first(where: { index in
                let step = prefillPlan.steps[index]
                let kernel = step.metadata.kernelName ?? step.pipeline.label ?? ""
                return kernel.hasPrefix("recurrent_block_partial_reduce")
            }) else {
                throw SetupError.stepNotFound("linear attention partial output reduce for layer \(layerIndex)")
            }
            partialProjection = firstOutProjection
            partialProjectionBindingIndex = 2
            outProjection = reduceStep
            outProjectionBindingIndex = 1
        } else if firstOutProjectionKernel.hasPrefix("recurrent_block_partial_reduce") {
            partialProjection = recurrence
            partialProjectionBindingIndex = 22
            outProjection = firstOutProjection
            outProjectionBindingIndex = 1
        } else {
            partialProjection = nil
            partialProjectionBindingIndex = 2
            outProjection = firstOutProjection
            outProjectionBindingIndex = 2
        }
        return LinearAttentionBoundarySteps(
            projection: projection,
            recurrence: recurrence,
            partialProjection: partialProjection,
            partialProjectionBindingIndex: partialProjectionBindingIndex,
            outProjection: outProjection,
            outProjectionBindingIndex: outProjectionBindingIndex
        )
    }

    private static func linearAttentionBoundaryStages(
        referencePrefix: String,
        ordinal: Int,
        steps: LinearAttentionBoundarySteps,
        slotDimension: Int
    ) -> [LinearAttentionBoundaryStage] {
        let blockPrefix = "\(referencePrefix).prefill.linear_ordinal_\(ordinal).block"
        let labelPrefix = "linear_ordinal_\(ordinal).block"
        return [
            LinearAttentionBoundaryStage(
                name: "\(labelPrefix).projected_qkv",
                referenceName: "\(blockPrefix).projected_qkv",
                stepIndex: steps.projection,
                bindingIndex: 5,
                rowStride: slotDimension,
                count: 6144,
                tolerance: 1.25
            ),
            LinearAttentionBoundaryStage(
                name: "\(labelPrefix).projected_z",
                referenceName: "\(blockPrefix).projected_z",
                stepIndex: steps.projection,
                bindingIndex: 6,
                rowStride: slotDimension,
                count: 2048,
                tolerance: 1.25
            ),
            LinearAttentionBoundaryStage(
                name: "\(labelPrefix).projected_beta",
                referenceName: "\(blockPrefix).projected_beta",
                stepIndex: steps.projection,
                bindingIndex: 7,
                rowStride: slotDimension,
                count: 16,
                tolerance: 0.25
            ),
            LinearAttentionBoundaryStage(
                name: "\(labelPrefix).projected_alpha",
                referenceName: "\(blockPrefix).projected_alpha",
                stepIndex: steps.projection,
                bindingIndex: 8,
                rowStride: slotDimension,
                count: 16,
                tolerance: 0.25
            ),
            LinearAttentionBoundaryStage(
                name: "\(labelPrefix).conv_silu",
                referenceName: "\(blockPrefix).conv_silu",
                stepIndex: steps.recurrence,
                bindingIndex: 18,
                rowStride: slotDimension,
                count: 6144,
                tolerance: 1.25
            ),
            LinearAttentionBoundaryStage(
                name: "\(labelPrefix).gated_recurrent_output",
                referenceName: "\(blockPrefix).gated_recurrent_output",
                stepIndex: steps.recurrence,
                bindingIndex: 10,
                rowStride: slotDimension,
                count: 2048,
                tolerance: 1.25
            ),
            LinearAttentionBoundaryStage(
                name: "\(labelPrefix).out_projection",
                referenceName: "\(blockPrefix).out_projection",
                stepIndex: steps.outProjection,
                bindingIndex: steps.outProjectionBindingIndex,
                rowStride: 1024,
                count: 1024,
                tolerance: 1.25
            ),
            LinearAttentionBoundaryStage(
                name: "\(labelPrefix).out_projection_reduced",
                referenceName: "\(blockPrefix).out_projection_reduced",
                stepIndex: steps.outProjection,
                bindingIndex: steps.outProjectionBindingIndex,
                rowStride: 1024,
                count: 1024,
                tolerance: 1.25
            ),
        ]
    }

    private static func expectLinearAttentionFanInMatchesReference(
        check: LinearAttentionFanInCheck,
        env: TestEnvironment,
        snapshots: [String: [Float]],
        tokenCount: Int
    ) throws {
        let gatedRows = try (0..<tokenCount).map { rowIndex in
            guard let row = snapshots[check.gatedLabel(rowIndex: rowIndex)] else {
                throw SetupError.tensorNotFound("metal.\(check.gatedLabel(rowIndex: rowIndex))")
            }
            return row
        }
        let weightTensorName = "model.language_model.layers.\(check.layerIndex).linear_attn.out_proj.weight"
        guard let staf = env.staf else {
            throw SetupError.tensorNotFound("model.staf")
        }
        let (weight, weightShape) = try readSTAFTensorAsFloats(staf, name: weightTensorName)
        guard weightShape.count == 2 else {
            throw SetupError.unsupportedTensorDType(weightTensorName, "rank\(weightShape.count)")
        }
        let outputDimension = weightShape[0]
        let inputDimension = weightShape[1]
        let computedPartials = try computePartitionedOutputProjection(
            inputRows: gatedRows,
            weight: weight,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            partitionCount: check.partitionCount
        )
        let referencePartials = try readRefTensorAsFloats(
            env.ref,
            name: "\(check.referenceBlockPrefix).out_projection_partials"
        )
        let partialError = maxAbsoluteError(computedPartials.partials, referencePartials)
        print("[Qwen35Ref] \(check.referenceBlockPrefix).out_projection_partials computed maxErr=\(String(format: "%.4f", partialError))")
        #expect(
            partialError <= 1.25,
            "\(check.referenceBlockPrefix).out_projection_partials computed drifted: maxErr=\(partialError)"
        )

        if check.partialProjectionStepIndex != nil {
            var metalPartials: [Float] = []
            metalPartials.reserveCapacity(referencePartials.count)
            for rowIndex in 0..<tokenCount {
                for partition in 0..<check.partitionCount {
                    guard let values = snapshots[check.partialProbeLabel(partition: partition, rowIndex: rowIndex)] else {
                        throw SetupError.tensorNotFound("metal.\(check.partialProbeLabel(partition: partition, rowIndex: rowIndex))")
                    }
                    metalPartials.append(contentsOf: values)
                }
            }
            let metalPartialError = maxAbsoluteError(metalPartials, referencePartials)
            print("[Qwen35Ref] \(check.referenceBlockPrefix).out_projection_partials metal maxErr=\(String(format: "%.4f", metalPartialError))")
            if metalPartialError > 1.25,
               let mismatch = largestMismatch(
                actual: metalPartials,
                expected: referencePartials,
                outputDimension: outputDimension,
                partitionCount: check.partitionCount
               ) {
                print(
                    "[Qwen35Ref] partial mismatch token=\(mismatch.token) partition=\(mismatch.partition) output=\(mismatch.output) metal=\(mismatch.actual) ref=\(mismatch.expected)"
                )
            }
            #expect(
                metalPartialError <= 1.25,
                "\(check.referenceBlockPrefix).out_projection_partials metal drifted: maxErr=\(metalPartialError)"
            )
        }

        let referenceReduced = try readRefTensorAsFloats(
            env.ref,
            name: "\(check.referenceBlockPrefix).out_projection_reduced"
        )
        let reducedError = maxAbsoluteError(computedPartials.reduced, referenceReduced)
        #expect(
            reducedError <= 1.25,
            "\(check.referenceBlockPrefix).out_projection_reduced computed drifted: maxErr=\(reducedError)"
        )
        let referenceOut = try readRefTensorAsFloats(
            env.ref,
            name: "\(check.referenceBlockPrefix).out_projection"
        )
        let reducedOutError = maxAbsoluteError(referenceReduced, referenceOut)
        #expect(
            reducedOutError <= 1.25,
            "\(check.referenceBlockPrefix).out_projection_reduced differs from out_projection: maxErr=\(reducedOutError)"
        )
    }

    private static func computePartitionedOutputProjection(
        inputRows: [[Float]],
        weight: [Float],
        inputDimension: Int,
        outputDimension: Int,
        partitionCount: Int
    ) throws -> (partials: [Float], reduced: [Float]) {
        guard inputDimension % partitionCount == 0 else {
            throw SetupError.unsupportedTensorDType("partitioned output projection", "inputDim=\(inputDimension), partitions=\(partitionCount)")
        }
        let partitionInputDimension = inputDimension / partitionCount
        var partials = [Float](repeating: 0, count: inputRows.count * partitionCount * outputDimension)
        var reduced = [Float](repeating: 0, count: inputRows.count * outputDimension)
        for rowIndex in inputRows.indices {
            let input = inputRows[rowIndex]
            guard input.count >= inputDimension else {
                throw SetupError.unsupportedTensorDType("partitioned output projection input", "count=\(input.count)")
            }
            for partition in 0..<partitionCount {
                let inputBase = partition * partitionInputDimension
                for row in 0..<outputDimension {
                    var sum: Float = 0
                    let weightBase = row * inputDimension + inputBase
                    for column in 0..<partitionInputDimension {
                        sum += input[inputBase + column] * weight[weightBase + column]
                    }
                    let partialIndex = rowIndex * partitionCount * outputDimension
                        + partition * outputDimension
                        + row
                    partials[partialIndex] = sum
                    reduced[rowIndex * outputDimension + row] += sum
                }
            }
        }
        return (partials, reduced)
    }

    private static func largestMismatch(
        actual: [Float],
        expected: [Float],
        outputDimension: Int,
        partitionCount: Int
    ) -> (token: Int, partition: Int, output: Int, actual: Float, expected: Float)? {
        guard actual.count == expected.count, outputDimension > 0, partitionCount > 0 else {
            return nil
        }
        var bestIndex = 0
        var bestError = Float.zero
        for index in actual.indices {
            let error = abs(actual[index] - expected[index])
            if error > bestError {
                bestError = error
                bestIndex = index
            }
        }
        let output = bestIndex % outputDimension
        let partition = (bestIndex / outputDimension) % partitionCount
        let token = bestIndex / (partitionCount * outputDimension)
        return (token, partition, output, actual[bestIndex], expected[bestIndex])
    }

    private static func readRefTensorAsFloats(
        _ file: MetalWeightFile,
        name: String
    ) throws -> [Float] {
        guard let info = file.tensors[name] else {
            throw SetupError.tensorNotFound(name)
        }
        let count = info.shape.reduce(1, *)
        let base = file.buffer.contents() + file.dataSectionOffset + info.dataOffset
        switch info.dtype {
        case .float16:
            let pointer = base.bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let pointer = base.bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .float32:
            let pointer = base.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        default:
            throw SetupError.unsupportedTensorDType(name, info.dtype.rawValue)
        }
    }

    private static func readSTAFTensorAsFloats(
        _ store: STAFWeightStore,
        name: String
    ) throws -> ([Float], [Int]) {
        guard let entry = store.entries[name] else {
            throw SetupError.tensorNotFound(name)
        }
        let count = entry.shape.reduce(1, *)
        let base = store.buffer.contents() + entry.bufferOffset
        switch entry.schemeIdentifier {
        case .fp16RowMajor:
            let pointer = base.bindMemory(to: Float16.self, capacity: count)
            return ((0..<count).map { Float(pointer[$0]) }, entry.shape)
        case .bf16RowMajor:
            let pointer = base.bindMemory(to: BFloat16.self, capacity: count)
            return ((0..<count).map { Float(pointer[$0]) }, entry.shape)
        case .fp32RowMajor:
            let pointer = base.bindMemory(to: Float.self, capacity: count)
            return (Array(UnsafeBufferPointer(start: pointer, count: count)), entry.shape)
        default:
            throw SetupError.unsupportedTensorDType(name, "\(entry.schemeIdentifier)")
        }
    }

    private static func readRefInt32(_ file: MetalWeightFile, name: String) throws -> Int32 {
        guard let info = file.tensors[name] else {
            throw SetupError.tensorNotFound(name)
        }
        guard info.dtype == .int32, info.shape.reduce(1, *) >= 1 else {
            throw SetupError.unsupportedTensorDType(name, info.dtype.rawValue)
        }
        let pointer = (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: Int32.self, capacity: 1)
        return pointer.pointee
    }

    private static func readRefInt32Array(_ file: MetalWeightFile, name: String) throws -> [Int32] {
        guard let info = file.tensors[name] else {
            throw SetupError.tensorNotFound(name)
        }
        guard info.dtype == .int32 else {
            throw SetupError.unsupportedTensorDType(name, info.dtype.rawValue)
        }
        let count = info.shape.reduce(1, *)
        let pointer = (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: Int32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private static func readRefUInt8Array(_ file: MetalWeightFile, name: String) throws -> [UInt8] {
        guard let info = file.tensors[name] else {
            throw SetupError.tensorNotFound(name)
        }
        guard info.dtype == .uint8 else {
            throw SetupError.unsupportedTensorDType(name, info.dtype.rawValue)
        }
        let count = info.shape.reduce(1, *)
        let pointer = (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: UInt8.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private static func configSHA256(bundlePath: String) throws -> [UInt8] {
        let configURL = URL(fileURLWithPath: bundlePath).appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        return Array(SHA256.hash(data: data))
    }

    private static func referenceLinearOrdinalCount(_ file: MetalWeightFile, prefix: String) -> Int {
        var count = 0
        while file.tensors["\(prefix).linear_ordinal_\(count).conv_state"] != nil {
            count += 1
        }
        return count
    }

    private static func expectLinearStatesMatchReference(
        model: MetalInferenceModel,
        ref: MetalWeightFile,
        prefix: String
    ) throws {
        guard let convState = model.buffers.convState else {
            Issue.record("Qwen35 decode plan did not allocate conv state")
            return
        }
        guard let recurrentState = model.buffers.recurrentState else {
            Issue.record("Qwen35 decode plan did not allocate recurrent state")
            return
        }

        let convDim = model.buffers.convStateDimension
        let kernelSize = model.buffers.convStateKernelSize
        let convLayerCount = convState.length / (convDim * kernelSize * MemoryLayout<BFloat16>.stride)
        let recurrentLayerValues = model.buffers.recurrentStateBytesPerLayer / MemoryLayout<Float>.stride
        let recurrentLayerCount = recurrentState.length / model.buffers.recurrentStateBytesPerLayer
        let referenceLinearCount = referenceLinearOrdinalCount(ref, prefix: prefix)
        let metalConv = try readBuffer(convState, precision: .bfloat16)
        let metalRecurrent = try readBuffer(recurrentState, precision: .float32)

        #expect(referenceLinearCount == convLayerCount)
        #expect(referenceLinearCount == recurrentLayerCount)

        for ordinal in 0..<min(referenceLinearCount, convLayerCount) {
            let refConv = try readRefTensorAsFloats(ref, name: "\(prefix).linear_ordinal_\(ordinal).conv_state")
            let convBase = ordinal * convDim * kernelSize
            let conv = Array(metalConv[convBase..<(convBase + convDim * kernelSize)])
            let convError = maxAbsoluteError(conv, refConv)
            print("[Qwen35Ref] \(prefix).linear_ordinal_\(ordinal).conv_state maxErr=\(String(format: "%.4f", convError))")
            #expect(convError < 1.25, "\(prefix).linear_ordinal_\(ordinal).conv_state drifted: maxErr=\(convError)")

            let refRecurrent = try readRefTensorAsFloats(ref, name: "\(prefix).linear_ordinal_\(ordinal).recurrent_state")
            let recurrentBase = ordinal * recurrentLayerValues
            let recurrent = Array(metalRecurrent[recurrentBase..<(recurrentBase + recurrentLayerValues)])
            let recurrentError = maxAbsoluteError(recurrent, refRecurrent)
            print("[Qwen35Ref] \(prefix).linear_ordinal_\(ordinal).recurrent_state maxErr=\(String(format: "%.4f", recurrentError))")
            #expect(recurrentError < 0.75, "\(prefix).linear_ordinal_\(ordinal).recurrent_state drifted: maxErr=\(recurrentError)")
        }
    }

    private static func expectKVCacheMatchesReference(
        model: MetalInferenceModel,
        ref: MetalWeightFile,
        prefix: String,
        tokenCount: Int
    ) throws {
        guard let cache = model.buffers.kvCache else {
            Issue.record("Qwen35 decode plan did not allocate KV cache")
            return
        }

        let keyPrecision = try denseKVPrecision(for: cache.specification.keyQuantizationScheme)
        let valuePrecision = try denseKVPrecision(for: cache.specification.valueQuantizationScheme)
        let attentionCount = referenceAttentionOrdinalCount(ref, prefix: prefix)
        #expect(attentionCount == cache.specification.layerCount)

        for ordinal in 0..<min(attentionCount, cache.specification.layerCount) {
            let metalKeys = try readKVCacheLayer(
                cache: cache,
                layerIndex: ordinal,
                tokenCount: tokenCount,
                precision: keyPrecision,
                kind: .keys
            )
            let refKeys = try readRefTensorAsFloats(ref, name: "\(prefix).attn_ordinal_\(ordinal).keys")
            let keyError = maxAbsoluteError(metalKeys, refKeys)
            print("[Qwen35Ref] \(prefix).attn_ordinal_\(ordinal).keys maxErr=\(String(format: "%.4f", keyError))")
            #expect(keyError < 1.0, "\(prefix).attn_ordinal_\(ordinal).keys drifted: maxErr=\(keyError)")

            let metalValues = try readKVCacheLayer(
                cache: cache,
                layerIndex: ordinal,
                tokenCount: tokenCount,
                precision: valuePrecision,
                kind: .values
            )
            let refValues = try readRefTensorAsFloats(ref, name: "\(prefix).attn_ordinal_\(ordinal).values")
            let valueError = maxAbsoluteError(metalValues, refValues)
            print("[Qwen35Ref] \(prefix).attn_ordinal_\(ordinal).values maxErr=\(String(format: "%.4f", valueError))")
            #expect(valueError < 1.0, "\(prefix).attn_ordinal_\(ordinal).values drifted: maxErr=\(valueError)")
        }
    }

    private enum KVKind {
        case keys
        case values
    }

    private static func readKVCacheLayer(
        cache: MetalKVCache,
        layerIndex: Int,
        tokenCount: Int,
        precision: BufferPrecision,
        kind: KVKind
    ) throws -> [Float] {
        let spec = cache.specification
        let scheme: QuantizationSchemeIdentifier
        let buffer: MTLBuffer
        switch kind {
        case .keys:
            scheme = spec.keyQuantizationScheme
            buffer = cache.keys
        case .values:
            scheme = spec.valueQuantizationScheme
            buffer = cache.values
        }

        var values: [Float] = []
        values.reserveCapacity(spec.kvHeadCount * tokenCount * spec.headDimension)
        for head in 0..<spec.kvHeadCount {
            for position in 0..<tokenCount {
                let offset = spec.offset(layer: layerIndex, head: head, position: position, scheme: scheme)
                values.append(
                    contentsOf: try readBufferSlice(
                        buffer,
                        offset: offset,
                        count: spec.headDimension,
                        precision: precision
                    )
                )
            }
        }
        return values
    }

    private static func denseKVPrecision(for scheme: QuantizationSchemeIdentifier) throws -> BufferPrecision {
        switch scheme {
        case .fp16RowMajor:
            return .float16
        case .bf16RowMajor:
            return .bfloat16
        case .fp32RowMajor:
            return .float32
        default:
            throw SetupError.unsupportedKVScheme("\(scheme)")
        }
    }

    private static func referenceAttentionOrdinalCount(_ file: MetalWeightFile, prefix: String) -> Int {
        var count = 0
        while file.tensors["\(prefix).attn_ordinal_\(count).keys"] != nil {
            count += 1
        }
        return count
    }

    private static func readBuffer(_ buffer: MTLBuffer, precision: BufferPrecision) throws -> [Float] {
        if buffer.storageMode == .private {
            let device = buffer.device
            guard let staging = device.makeBuffer(length: buffer.length, options: .storageModeShared),
                  let queue = device.makeCommandQueue(),
                  let commandBuffer = queue.makeCommandBuffer(),
                  let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw SetupError.metalReadbackUnavailable
            }
            blit.copy(from: buffer, sourceOffset: 0, to: staging, destinationOffset: 0, size: buffer.length)
            blit.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw SetupError.metalReadbackFailed(error.localizedDescription)
            }
            return readSharedBuffer(staging, precision: precision)
        }
        return readSharedBuffer(buffer, precision: precision)
    }

    private static func readBufferSlice(
        _ buffer: MTLBuffer,
        offset: Int,
        count: Int,
        precision: BufferPrecision
    ) throws -> [Float] {
        let byteCount = count * precision.byteSize
        guard offset >= 0, count >= 0, offset + byteCount <= buffer.length else {
            throw SetupError.bufferRangeOutOfBounds(offset: offset, byteCount: byteCount, length: buffer.length)
        }
        let device = buffer.device
        guard let staging = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let queue = device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw SetupError.metalReadbackUnavailable
        }
        blit.copy(from: buffer, sourceOffset: offset, to: staging, destinationOffset: 0, size: byteCount)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SetupError.metalReadbackFailed(error.localizedDescription)
        }
        return readSharedBuffer(staging, precision: precision)
    }

    private static func readSharedBuffer(_ buffer: MTLBuffer, precision: BufferPrecision) -> [Float] {
        switch precision {
        case .float16:
            let count = buffer.length / MemoryLayout<Float16>.stride
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let count = buffer.length / MemoryLayout<BFloat16>.stride
            let pointer = buffer.contents().bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .float32, .float32Decode:
            let count = buffer.length / MemoryLayout<Float>.stride
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        }
    }

    private struct IndexedValue {
        let index: Int
        let value: Float
    }

    private static func argmax(_ values: [Float]) -> IndexedValue {
        var index = 0
        var value = -Float.infinity
        for candidateIndex in 0..<values.count where values[candidateIndex] > value {
            index = candidateIndex
            value = values[candidateIndex]
        }
        return IndexedValue(index: index, value: value)
    }

    private static func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return .infinity }
        let count = min(lhs.count, rhs.count)
        var error: Float = 0
        for index in 0..<count {
            guard lhs[index].isFinite, rhs[index].isFinite else { return .infinity }
            error = max(error, abs(lhs[index] - rhs[index]))
        }
        return error
    }

    private enum SetupError: Error {
        case noDevice
        case noReference
        case noBundle
        case tensorNotFound(String)
        case unsupportedTensorDType(String, String)
        case unsupportedKVScheme(String)
        case metalReadbackUnavailable
        case metalReadbackFailed(String)
        case bufferRangeOutOfBounds(offset: Int, byteCount: Int, length: Int)
        case stepNotFound(String)
    }
}
#endif
