import Foundation
import Metal
import Testing
@testable import MetalCompiler

@Suite("LFM Strict Prompt Reference", .serialized)
struct LFMStrictPromptReferenceTests {
    private static let referencePath = URL(fileURLWithPath: BenchmarkSupport.testDataPath)
        .appendingPathComponent("lfm2_strict_chat_reference.safetensors")
        .path
    private static let hiReferencePath = "/tmp/lfm2_hi_reference.safetensors"
    private static let stafPath = BenchmarkSupport.stafPath
    private static let modelDirectoryPath = BenchmarkSupport.lfmBundlePath
    private static let hiPromptTokenIDs: [Int32] = [1, 6, 6423, 708, 6928, 7, 708, 6, 64015, 708]
    private static let hiExpectedFirstToken: Int32 = 64400
    private static let hiExpectedDecodeTokenIDs: [Int32] = [9095, 892, 521, 2944, 1090, 2130, 1620, 779]

    private struct TestEnvironment {
        var model: MetalInferenceModel
        let ref: MetalWeightFile
    }

    private struct ParsedDispatchEntry {
        let layer: Int?
        let kind: String
    }

    @Test("Strict prompt prefill logits match HuggingFace reference", .timeLimit(.minutes(2)))
    func strictPromptPrefillLogitsMatchReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try setupOrSkip()
        let tokens = try readInputTokens(env.ref)
        var model = env.model

        let firstToken = model.prefill(tokens: tokens)
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.prefill.logits_last")
        let metalLogits = readF32Buffer(try #require(model.prefillPlan).buffers.logits)
        let metalTop = argmax(metalLogits)
        let refTop = argmax(refLogits)

        print("[StrictRef] firstToken metal=\(firstToken)")
        print("[StrictRef] prefill argmax metal=\(metalTop.index) python=\(refTop.index)")
        print("[StrictRef] prefill top-10 metal=\(formatTopK(topK(metalLogits, k: 10)))")
        print("[StrictRef] prefill top-10 python=\(formatTopK(topK(refLogits, k: 10)))")

        let prefillPlan = try #require(model.prefillPlan)
        for (index, step) in prefillPlan.steps.prefix(20).enumerated() {
            let kernel = step.metadata.kernelName ?? step.pipeline.label ?? "unknown"
            print("[StrictRef] step[\(index)] layer=\(step.metadata.layerIndex.map(String.init) ?? "-") kernel=\(kernel) mode=\(step.mode)")
        }
        for (index, step) in prefillPlan.steps.suffix(20).enumerated() {
            let absoluteIndex = prefillPlan.steps.count - 20 + index
            let kernel = step.metadata.kernelName ?? step.pipeline.label ?? "unknown"
            print("[StrictRef] tailStep[\(absoluteIndex)] layer=\(step.metadata.layerIndex.map(String.init) ?? "-") kernel=\(kernel) mode=\(step.mode)")
        }
        let finalHiddenSource = prefillPlan.finalHiddenSource(sequenceLength: tokens.count)
        print("[StrictRef] prefill stepCount=\(prefillPlan.stepCount)")
        print("[StrictRef] finalHiddenSource buffer=\(bufferLabel(finalHiddenSource.buffer, prefillPlan: prefillPlan)) offset=\(finalHiddenSource.offset)")

        #if ENABLE_METAL_PROBES
        let metalFinal = try model.debugPrefillLastTokenFinalHidden(tokens: tokens)
        let refFinalAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.final_hidden")
        let refFinal = Array(refFinalAll.suffix(2048))
        let finalErr = maxAbsoluteError(metalFinal, refFinal)
        print("[StrictRef] prefill final_hidden maxErr=\(String(format: "%.4f", finalErr))")

        let allCheckpoints: [(name: String, stepCount: Int, threshold: Float)] = [
            ("ref.prefill.layer_0.after_op", 7, 0.25),
            ("ref.prefill.layer_0.mlp_out", 13, 0.50),
            ("ref.prefill.layer_0.after_mlp", 14, 0.50),
            ("ref.prefill.layer_2.after_op", 40, 0.50),
            ("ref.prefill.layer_5.after_op", 87, 0.50),
            ("ref.prefill.layer_8.after_op", 134, 0.50),
            ("ref.prefill.layer_8.after_mlp", 141, 0.50),
            ("ref.prefill.layer_9.after_op", 148, 0.50),
            ("ref.prefill.layer_9.after_mlp", 155, 0.50),
            ("ref.prefill.layer_10.after_op", 167, 0.50),
            ("ref.prefill.layer_10.after_mlp", 174, 0.50),
            ("ref.prefill.layer_11.after_op", 181, 0.50),
            ("ref.prefill.layer_11.after_mlp", 188, 0.50),
            ("ref.prefill.layer_12.after_op", 200, 0.50),
            ("ref.prefill.layer_12.after_mlp", 207, 0.50),
            ("ref.prefill.layer_13.after_op", 214, 0.50),
            ("ref.prefill.layer_13.after_mlp", 221, 0.50),
            ("ref.prefill.layer_14.after_op", 233, 0.50),
            ("ref.prefill.layer_14.after_mlp", 240, 0.50),
            ("ref.prefill.layer_15.after_op", 247, 0.50),
            ("ref.prefill.layer_15.after_mlp", 254, 0.50),
        ]
        let checkpoints = allCheckpoints.filter { $0.stepCount < prefillPlan.steps.count }
        let skippedCheckpoints = allCheckpoints.filter { $0.stepCount >= prefillPlan.steps.count }
        if !skippedCheckpoints.isEmpty {
            print("[StrictRef] skipping \(skippedCheckpoints.count) stale checkpoints for \(prefillPlan.stepCount)-step prefill plan")
        }

        let snapshots = try model.debugPrefillLastTokenHiddenSnapshots(
            tokens: tokens,
            stepIndices: Set(checkpoints.map(\.stepCount))
        )
        for checkpoint in checkpoints {
            let metal = try #require(snapshots[checkpoint.stepCount], "Missing step snapshot \(checkpoint.stepCount)")
            let referenceAll = try readRefTensorAsFloats(env.ref, name: checkpoint.name)
            let reference = Array(referenceAll.suffix(2048))
            let error = maxAbsoluteError(metal, reference)
            print("[StrictRef] \(checkpoint.name) maxErr=\(String(format: "%.4f", error))")
            #expect(error < checkpoint.threshold, "\(checkpoint.name) diverged: maxErr=\(error)")
        }
        #expect(finalErr < 0.25, "Strict prompt final hidden diverged: maxErr=\(finalErr)")
        #endif

        #expect(firstToken == Int32(refTop.index), "Strict prompt first token mismatch")
        #expect(metalTop.index == refTop.index, "Strict prompt prefill argmax mismatch")
    }

    @Test("Strict prompt first decode step matches HuggingFace reference", .timeLimit(.minutes(2)))
    func strictPromptFirstDecodeStepMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try setupOrSkip()
        let tokens = try readInputTokens(env.ref)
        var model = env.model

        let firstToken = model.prefill(tokens: tokens)
        let decodedToken = model.decodeSync(tokenID: firstToken)
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_0.logits")
        let metalLogits = readDecodeBuffer(model.buffers.logits, precision: model.buffers.bufferPrecision)
        let metalTop = argmax(metalLogits)
        let refTop = argmax(refLogits)
        let maxErr = maxAbsoluteError(metalLogits, refLogits)

        print("[StrictRef] decode0 input metal=\(firstToken)")
        print("[StrictRef] decode0 output metal=\(decodedToken)")
        print("[StrictRef] decode0 argmax metal=\(metalTop.index) python=\(refTop.index)")
        print("[StrictRef] decode0 top-10 metal=\(formatTopK(topK(metalLogits, k: 10)))")
        print("[StrictRef] decode0 top-10 python=\(formatTopK(topK(refLogits, k: 10)))")
        print("[StrictRef] decode0 maxErr=\(String(format: "%.4f", maxErr))")

        #expect(firstToken == 64400, "Strict prompt first token should be <think>")
        #expect(Int(decodedToken) == refTop.index, "Strict prompt first decode token mismatch")
        #expect(metalTop.index == refTop.index, "Strict prompt decode0 argmax mismatch")
    }

    @Test("Strict prompt decode prefix matches HuggingFace reference", .timeLimit(.minutes(2)))
    func strictPromptDecodePrefixMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try setupOrSkip()
        let tokens = try readInputTokens(env.ref)
        var model = env.model
        var inputToken = model.prefill(tokens: tokens)

        for step in 0..<3 {
            let decodedToken = model.decodeSync(tokenID: inputToken)
            let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).logits")
            let metalLogits = readDecodeBuffer(model.buffers.logits, precision: model.buffers.bufferPrecision)
            let metalTop = argmax(metalLogits)
            let refTop = argmax(refLogits)
            let maxErr = maxAbsoluteError(metalLogits, refLogits)

            print("[StrictRef] decode\(step) input metal=\(inputToken)")
            print("[StrictRef] decode\(step) output metal=\(decodedToken)")
            print("[StrictRef] decode\(step) argmax metal=\(metalTop.index) python=\(refTop.index)")
            print("[StrictRef] decode\(step) top-10 metal=\(formatTopK(topK(metalLogits, k: 10)))")
            print("[StrictRef] decode\(step) top-10 python=\(formatTopK(topK(refLogits, k: 10)))")
            print("[StrictRef] decode\(step) maxErr=\(String(format: "%.4f", maxErr))")

            #expect(Int(decodedToken) == refTop.index, "Strict prompt decode\(step) token mismatch")
            #expect(metalTop.index == refTop.index, "Strict prompt decode\(step) argmax mismatch")
            inputToken = decodedToken
        }
    }

    @Test("Hi prompt decode prefix matches HuggingFace reference when available", .timeLimit(.minutes(2)))
    func hiPromptDecodePrefixMatchesReferenceWhenAvailable() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        var model = try setupModelOnly()
        let tokens = Self.hiPromptTokenIDs
        var inputToken = model.prefill(tokens: tokens)
        let stepLimit = Int(ProcessInfo.processInfo.environment["SWIFTLM_HI_PREFIX_STEPS"] ?? "") ?? 8
        #expect(inputToken == Self.hiExpectedFirstToken, "Hi prompt prefill token mismatch")
        #expect(stepLimit <= Self.hiExpectedDecodeTokenIDs.count, "Hi prompt prefix step limit exceeds expected token coverage")

        for step in 0..<stepLimit {
            let decodedToken = model.decodeSync(tokenID: inputToken)
            let expectedToken = Self.hiExpectedDecodeTokenIDs[step]

            print("[StrictRef][hi] decode\(step) input metal=\(inputToken)")
            print("[StrictRef][hi] decode\(step) output metal=\(decodedToken)")
            print("[StrictRef][hi] decode\(step) expected=\(expectedToken)")

            #expect(decodedToken == expectedToken, "Hi prompt decode\(step) token mismatch")
            inputToken = expectedToken
        }
        print("[StrictRef][hi] prefix parity completed steps=\(stepLimit)")
    }

    @Test("Layer13 dense MLP STAF rows match safetensors", .timeLimit(.minutes(2)))
    func layer13DenseMLPSTAFRowsMatchSafetensors() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let device = try #require(MTLCreateSystemDefaultDevice())
        let modelDirectory = URL(fileURLWithPath: Self.modelDirectoryPath)
        let safetensorURLs = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let safetensors = try SafetensorsLoader().loadAll(urls: safetensorURLs, device: device)
        let stafStore = try STAFLoader().load(
            at: modelDirectory.appendingPathComponent("model.staf"),
            device: device
        )

        let tensorNames = [
            "model.layers.13.feed_forward.w1.weight",
            "model.layers.13.feed_forward.w2.weight",
            "model.layers.13.feed_forward.w3.weight",
        ]

        for tensorName in tensorNames {
            let safetensorsTensor = try #require(safetensors.tensor(for: tensorName))
            let rowCount = try #require(safetensorsTensor.shape.first)
            let columnCount = try #require(safetensorsTensor.shape.dropFirst().first)
            let sampledRows = Array(Set([0, rowCount / 2, max(rowCount - 1, 0)])).sorted()
            for rowIndex in sampledRows {
                let safetensorsRow = try readTensorRow(
                    tensor: safetensorsTensor,
                    rowIndex: rowIndex,
                    columnCount: columnCount
                )
                let stafRow = try readSTAFRow(
                    tensorName: tensorName,
                    rowIndex: rowIndex,
                    columnCount: columnCount,
                    store: stafStore
                )
                let error = maxAbsoluteError(safetensorsRow, stafRow)
                print("[StrictRef] \(tensorName) row=\(rowIndex) maxErr=\(String(format: "%.6f", error))")
                #expect(error == 0, "\(tensorName) row \(rowIndex) diverged: maxErr=\(error)")
            }
        }
    }

    #if ENABLE_METAL_PROBES
    @Test("Hi prompt decode4 layerwise diagnostic", .timeLimit(.minutes(2)))
    func hiPromptDecode4LayerwiseDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let referencePath = ProcessInfo.processInfo.environment["SWIFTLM_LFM_HI_REFERENCE_PATH"]
            ?? Self.hiReferencePath
        guard FileManager.default.fileExists(atPath: referencePath) else {
            print("[Skip] Generate \(referencePath) with scripts/hf/dump_lfm2_reference.py before running hi prompt parity.")
            return
        }

        let env = try setupOrSkip(referencePath: referencePath)
        let tokens = try readInputTokens(env.ref)
        let compiler = MetalInferenceCompiler(
            decodeBufferPrecisionOverride: decodeBufferPrecisionOverrideFromEnvironment()
        )
        let dispatchDump = try makeDispatchDump(compiler: compiler)
        let entries = parseDispatchEntries(from: dispatchDump)
        var model = env.model

        let warmupPrefillCount = Int(
            ProcessInfo.processInfo.environment["SWIFTLM_WARMUP_PREFILL_COUNT"] ?? "0"
        ) ?? 0
        if warmupPrefillCount > 0 {
            for _ in 0..<warmupPrefillCount {
                _ = model.prefill(tokens: tokens)
                model.resetState()
            }
        }

        var inputToken = model.prefill(tokens: tokens)
        let transferredFinalHidden = Array(
            readDecodeBuffer(finalHiddenInputBuffer(for: model.decodePlan), precision: model.buffers.bufferPrecision)
                .prefix(2048)
        )
        let refPrefillFinalHidden = Array(
            try readRefTensorAsFloats(env.ref, name: "ref.prefill.final_hidden")
                .suffix(2048)
        )
        print("[StrictRef][hi] prefill transferredFinalHidden maxErr=\(String(format: "%.4f", maxAbsoluteError(transferredFinalHidden, refPrefillFinalHidden)))")
        let skipPrefillStateProbes = ProcessInfo.processInfo
            .environment["SWIFTLM_SKIP_PREFILL_STATE_PROBES"] == "1"
        if !skipPrefillStateProbes, let convState = model.buffers.convState {
            let convDim = model.buffers.convStateDimension
            let kernelSize = model.buffers.convStateKernelSize
            let allConvState = readDecodeBuffer(convState, precision: .bfloat16)
            for convIdx in 0..<10 {
                let refConvState = try readRefTensorAsFloats(env.ref, name: "ref.prefill.conv_state.\(convIdx)")
                let offset = convIdx * kernelSize * convDim
                let metalConvState = Array(allConvState[offset..<(offset + kernelSize * convDim)])
                print("[StrictRef][hi] prefill conv_state[\(convIdx)] maxErr=\(String(format: "%.4f", maxAbsoluteError(metalConvState, refConvState)))")
            }
        }
        if !skipPrefillStateProbes {
        for kvIdx in 0..<6 {
            if let keySnapshot = try model.debugPrefillKVCacheLayerSnapshot(
                tokens: tokens,
                layerIndex: kvIdx,
                kind: .keys
            ) {
                let metalKeys = readKVCacheSnapshotValues(keySnapshot, sequenceLength: tokens.count)
                let refKeys = try readRefTensorAsFloats(env.ref, name: "ref.prefill.kv_cache.\(kvIdx).keys")
                print("[StrictRef][hi] prefill kv_cache[\(kvIdx)].keys maxErr=\(String(format: "%.4f", maxAbsoluteError(metalKeys, refKeys)))")
            }
            if let valueSnapshot = try model.debugPrefillKVCacheLayerSnapshot(
                tokens: tokens,
                layerIndex: kvIdx,
                kind: .values
            ) {
                let metalValues = readKVCacheSnapshotValues(valueSnapshot, sequenceLength: tokens.count)
                let refValues = try readRefTensorAsFloats(env.ref, name: "ref.prefill.kv_cache.\(kvIdx).values")
                print("[StrictRef][hi] prefill kv_cache[\(kvIdx)].values maxErr=\(String(format: "%.4f", maxAbsoluteError(metalValues, refValues)))")
            }
        }
        inputToken = model.prefill(tokens: tokens)
        }
        for step in 0..<4 {
            _ = model.decodeSync(tokenID: inputToken)
            let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).logits")
            inputToken = Int32(argmax(refLogits).index)
        }

        print("[StrictRef][hi] decode4 diagnostic input=\(inputToken) position=\(model.position)")
        model.buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(model.position)
        let ropeAxes = model.buffers.ropePositionAxes.contents().bindMemory(to: UInt32.self, capacity: 3)
        ropeAxes[0] = UInt32(model.position)
        ropeAxes[1] = UInt32(model.position)
        ropeAxes[2] = UInt32(model.position)
        model.buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = inputToken
        let finalNormStepIndex = finalNormStepIndex(for: model.decodePlan)

        var currentLayer = 0
        var currentKVLayer = 0
        var waitingForOperatorResidual = false
        var firstLargeDrift: (layer: Int, phase: String, error: Float)?
        var submission = try MetalSubmissionContext(device: model.device)

        for (stepIndex, step) in model.decodePlan.steps.enumerated() {
            guard stepIndex < entries.count else { break }

            if stepIndex == finalNormStepIndex {
                let kernel = step.metadata.kernelName ?? step.pipeline.label ?? "unknown"
                let inputBinding = try #require(step.bufferBindings.first(where: { $0.index == 0 }))
                let finalNormInput = Array(
                    readDecodeBuffer(inputBinding.buffer, precision: model.buffers.bufferPrecision)
                        .dropFirst(inputBinding.offset / model.buffers.bufferPrecision.byteSize)
                        .prefix(2048)
                )
                let expectedInput = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.layer_15.after_mlp")
                let inputError = maxAbsoluteError(finalNormInput, expectedInput)
                let bindingSummary = step.bufferBindings
                    .map { binding in
                        "i\(binding.index)=\(decodeBufferLabel(binding.buffer, model: model))@\(binding.offset)"
                    }
                    .joined(separator: ", ")
                print("[StrictRef][hi] decode4 finalNorm step=\(stepIndex) kernel=\(kernel) inputMaxErr=\(String(format: "%.4f", inputError))")
                print("[StrictRef][hi] decode4 finalNorm bindings \(bindingSummary)")
                if let residualBinding = step.bufferBindings.first(where: { $0.buffer === model.buffers.residual }) {
                    let residual = Array(
                        readDecodeBuffer(residualBinding.buffer, precision: model.buffers.bufferPrecision)
                            .dropFirst(residualBinding.offset / model.buffers.bufferPrecision.byteSize)
                            .prefix(2048)
                    )
                    let fusedInput = zip(finalNormInput, residual).map(+)
                    let fusedInputError = maxAbsoluteError(fusedInput, expectedInput)
                    print("[StrictRef][hi] decode4 finalNorm fusedInputMaxErr=\(String(format: "%.4f", fusedInputError))")
                }
            }

            try submission.withCompute { encoder, argumentTable in
                MetalDecodeEncoder.encodeStep(
                    step: step,
                    encoder: encoder,
                    argumentTable: argumentTable
                )
            }

            let entry = entries[stepIndex]
            if entry.kind.contains("FlashAttentionFragment") {
                let kvLayer = currentKVLayer
                currentKVLayer += 1
                let attentionOutput = try readDecodeScratchSlot(
                    model: model,
                    slotIndex: 0,
                    count: 32 * 64
                )
                let refAttentionOutput = try readRefTensorAsFloats(
                    env.ref,
                    name: "ref.decode_4.layer_\(currentLayer).attn_pre_o_proj"
                )
                let liveKey = try readLiveKVCacheToken(
                    model: model,
                    layerIndex: kvLayer,
                    position: model.position,
                    kind: .keys
                )
                let liveValue = try readLiveKVCacheToken(
                    model: model,
                    layerIndex: kvLayer,
                    position: model.position,
                    kind: .values
                )
                let refKey = try readRefTensorAsFloats(
                    env.ref,
                    name: "ref.decode_4.kv_cache.\(kvLayer).current_key"
                )
                let refValue = try readRefTensorAsFloats(
                    env.ref,
                    name: "ref.decode_4.kv_cache.\(kvLayer).current_value"
                )
                print("[StrictRef][hi] decode4 layer=\(currentLayer) kv=\(kvLayer) current_key maxErr=\(String(format: "%.4f", maxAbsoluteError(liveKey, refKey)))")
                print("[StrictRef][hi] decode4 layer=\(currentLayer) kv=\(kvLayer) current_value maxErr=\(String(format: "%.4f", maxAbsoluteError(liveValue, refValue)))")
                print("[StrictRef][hi] decode4 layer=\(currentLayer) kv=\(kvLayer) attn_pre_o_proj maxErr=\(String(format: "%.4f", maxAbsoluteError(attentionOutput, refAttentionOutput)))")
                if currentLayer == 2 {
                    let cache = try #require(model.buffers.kvCache)
                    print("[StrictRef][hi] decode4 kv schemes key=\(cache.specification.keyQuantizationScheme) value=\(cache.specification.valueQuantizationScheme) layout=\(cache.specification.layoutMode)")
                    print("[StrictRef][hi] decode4 layer=2 liveKey prefix=\(formatPrefix(liveKey, count: 12))")
                    print("[StrictRef][hi] decode4 layer=2 refKey prefix=\(formatPrefix(refKey, count: 12))")
                }
            }
            if entry.kind.contains("projection(o_proj") || entry.kind.contains("projection(out_proj") {
                if currentLayer == 2 {
                    let qNorm = try readDecodeScratchSlot(
                        model: model,
                        slotIndex: 1,
                        count: 32 * 64
                    )
                    let kNorm = try readDecodeScratchSlot(
                        model: model,
                        slotIndex: 2,
                        count: 8 * 64
                    )
                    let refQNorm = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.layer_2.q_norm")
                    let refKNorm = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.layer_2.k_norm")
                    print("[StrictRef][hi] decode4 layer=2 q_norm scratch maxErr=\(String(format: "%.4f", maxAbsoluteError(qNorm, refQNorm)))")
                    print("[StrictRef][hi] decode4 layer=2 k_norm scratch maxErr=\(String(format: "%.4f", maxAbsoluteError(kNorm, refKNorm)))")
                }
                let metal = readDecodeBuffer(model.buffers.hidden, precision: model.buffers.bufferPrecision)
                let ref = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.layer_\(currentLayer).after_op")
                let error = maxAbsoluteError(metal, ref)
                print("[StrictRef][hi] decode4 layer=\(currentLayer) phase=after_op kind=\(entry.kind) maxErr=\(String(format: "%.4f", error))")
                if firstLargeDrift == nil, error > 1.0 {
                    firstLargeDrift = (currentLayer, "after_op", error)
                }
                waitingForOperatorResidual = true
            }

            if entry.kind.contains("projection(down_proj") {
                let metal = readDecodeBuffer(model.buffers.hidden, precision: model.buffers.bufferPrecision)
                let ref = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.layer_\(currentLayer).mlp_out")
                let error = maxAbsoluteError(metal, ref)
                print("[StrictRef][hi] decode4 layer=\(currentLayer) phase=mlp_out kind=\(entry.kind) maxErr=\(String(format: "%.4f", error))")
                if firstLargeDrift == nil, error > 1.0 {
                    firstLargeDrift = (currentLayer, "mlp_out", error)
                }
            }

            if entry.kind.contains("ResidualAddFragment") || entry.kind.contains("synthesized_3way") {
                if waitingForOperatorResidual {
                    let residual = readDecodeBuffer(model.buffers.residual, precision: model.buffers.bufferPrecision)
                    let hidden = readDecodeBuffer(model.buffers.hidden, precision: model.buffers.bufferPrecision)
                    let refResidual = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.layer_\(currentLayer).after_op_residual")
                    let refFFNNorm = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.layer_\(currentLayer).ffn_norm")
                    let residualError = maxAbsoluteError(residual, refResidual)
                    let normError = maxAbsoluteError(hidden, refFFNNorm)
                    print("[StrictRef][hi] decode4 layer=\(currentLayer) phase=after_op_residual kind=\(entry.kind) maxErr=\(String(format: "%.4f", residualError))")
                    print("[StrictRef][hi] decode4 layer=\(currentLayer) phase=ffn_norm kind=\(entry.kind) maxErr=\(String(format: "%.4f", normError))")
                    if firstLargeDrift == nil, residualError > 1.0 {
                        firstLargeDrift = (currentLayer, "after_op_residual", residualError)
                    }
                    if firstLargeDrift == nil, normError > 1.0 {
                        firstLargeDrift = (currentLayer, "ffn_norm", normError)
                    }
                    waitingForOperatorResidual = false
                } else {
                    // In the fused ResidualAdd + Copy + Reduction kernel, hidden is
                    // the normalized output for the next sublayer. The post-MLP
                    // residual stream that HuggingFace calls layer.after_mlp is
                    // copied into the residual buffer.
                    let metal = readDecodeBuffer(model.buffers.residual, precision: model.buffers.bufferPrecision)
                    let ref = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.layer_\(currentLayer).after_mlp")
                    let error = maxAbsoluteError(metal, ref)
                    print("[StrictRef][hi] decode4 layer=\(currentLayer) phase=after_mlp kind=\(entry.kind) maxErr=\(String(format: "%.4f", error))")
                    if firstLargeDrift == nil, error > 1.0 {
                        firstLargeDrift = (currentLayer, "after_mlp", error)
                    }
                    currentLayer += 1
                }
            }
        }

        let metalLogits = readDecodeBuffer(model.buffers.logits, precision: model.buffers.bufferPrecision)
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.logits")
        let metalFinalHidden = Array(
            readDecodeBuffer(finalHiddenInputBuffer(for: model.decodePlan), precision: model.buffers.bufferPrecision)
                .prefix(2048)
        )
        let refFinalHidden = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.final_hidden")
        let metalTop = argmax(metalLogits)
        let refTop = argmax(refLogits)
        let logitsError = maxAbsoluteError(metalLogits, refLogits)
        let finalHiddenError = maxAbsoluteError(metalFinalHidden, refFinalHidden)
        print("[StrictRef][hi] decode4 final argmax metal=\(metalTop.index) python=\(refTop.index) maxErr=\(String(format: "%.4f", logitsError))")
        print("[StrictRef][hi] decode4 final_hidden maxErr=\(String(format: "%.4f", finalHiddenError))")
        if let drift = firstLargeDrift {
            print("[StrictRef][hi] decode4 firstLargeDrift layer=\(drift.layer) phase=\(drift.phase) maxErr=\(String(format: "%.4f", drift.error))")
        } else {
            print("[StrictRef][hi] decode4 firstLargeDrift none above threshold")
        }

        writeDecodeBuffer(refFinalHidden, to: finalHiddenInputBuffer(for: model.decodePlan), precision: model.buffers.bufferPrecision)
        try submission.withCompute { encoder, argumentTable in
            for step in model.decodePlan.steps.suffix(2) {
                MetalDecodeEncoder.encodeStep(
                    step: step,
                    encoder: encoder,
                    argumentTable: argumentTable
                )
            }
        }
        let logitsFromPythonHidden = readDecodeBuffer(model.buffers.logits, precision: model.buffers.bufferPrecision)
        let pythonHiddenTop = argmax(logitsFromPythonHidden)
        let pythonHiddenLogitsError = maxAbsoluteError(logitsFromPythonHidden, refLogits)
        print("[StrictRef][hi] decode4 outputHeadFromPythonHidden argmax metal=\(pythonHiddenTop.index) python=\(refTop.index) maxErr=\(String(format: "%.4f", pythonHiddenLogitsError))")

        let refAfterMLP = try readRefTensorAsFloats(env.ref, name: "ref.decode_4.layer_15.after_mlp")
        writeDecodeBuffer(refAfterMLP, to: model.buffers.hidden, precision: model.buffers.bufferPrecision)
        zeroDecodeBuffer(model.buffers.residual, precision: model.buffers.bufferPrecision)
        try submission.withCompute { encoder, argumentTable in
            MetalDecodeEncoder.encodeStep(
                step: model.decodePlan.steps[finalNormStepIndex],
                encoder: encoder,
                argumentTable: argumentTable
            )
        }
        let finalNormFromPythonInput = Array(
            readDecodeBuffer(finalHiddenInputBuffer(for: model.decodePlan), precision: model.buffers.bufferPrecision)
                .prefix(2048)
        )
        let finalNormKernelError = maxAbsoluteError(finalNormFromPythonInput, refFinalHidden)
        print("[StrictRef][hi] decode4 finalNormFromPythonAfterMLP maxErr=\(String(format: "%.4f", finalNormKernelError))")

        #expect(metalTop.index == refTop.index, "Hi prompt decode4 argmax mismatch")
    }

    @Test("Layer13 MLP chain localizes execution drift", .timeLimit(.minutes(2)))
    func layer13MLPChainLocalizesExecutionDrift() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try setupOrSkip()
        let tokens = try readInputTokens(env.ref)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let safetensors = try loadModelSafetensors(device: device)
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        var model = env.model

        let prefillPlan = try #require(model.prefillPlan)
        let hiddenRowStride = prefillPlan.buffers.hidden.length
            / MemoryLayout<Float>.stride
            / prefillPlan.maximumSequenceLength
        let scratchRowStride = prefillPlan.slotDimension
        let scratchSlotByteStride = prefillPlan.maximumSequenceLength
            * prefillPlan.slotDimension
            * MemoryLayout<Float>.stride

        let inputSnapshots = try model.debugPrefillLastTokenHiddenSnapshots(
            tokens: tokens,
            stepIndices: [214]
        )
        let input = try #require(inputSnapshots[214])
        let debugPassRanges = prefillPassRanges(for: prefillPlan.steps, within: 0..<(217))
        let debugPassRangeLabels = debugPassRanges.map { "\($0.lowerBound)..<\($0.upperBound)" }
        print("[StrictRef] debugPassRanges(0..<217)=\(debugPassRangeLabels)")
        let normStep = prefillPlan.steps[216]
        dumpStepBindings(normStep, stepIndex: 216, prefillPlan: prefillPlan)
        print("[StrictRef] step[216] barrierPolicy=\(String(describing: normStep.barrierPolicy))")
        let normScaleBinding = try #require(normStep.bindings.buffers.first(where: { $0.index == 1 }))
        let boundNormScale = readBufferSlice(
            buffer: normScaleBinding.buffer,
            offset: normScaleBinding.offset,
            count: 2048,
            precision: .bfloat16
        )
        let boundNormEpsilon = normStep.bytesBindings
            .first(where: { $0.index == 4 })
            .map { readFloatBinding($0.value) }
            ?? spec.config.normEps
        let normOut = try model.debugPrefillLastTokenBufferSnapshot(
            tokens: tokens,
            stepIndex: 216,
            buffer: prefillPlan.buffers.hidden,
            baseOffset: 0,
            rowStride: hiddenRowStride,
            count: 2048
        )
        let gateOut = try model.debugPrefillLastTokenBufferSnapshot(
            tokens: tokens,
            stepIndex: 217,
            buffer: prefillPlan.buffers.scratch,
            baseOffset: scratchSlotByteStride,
            rowStride: scratchRowStride,
            count: 8192
        )
        let upOut = try model.debugPrefillLastTokenBufferSnapshot(
            tokens: tokens,
            stepIndex: 218,
            buffer: prefillPlan.buffers.scratch,
            baseOffset: 2 * scratchSlotByteStride,
            rowStride: scratchRowStride,
            count: 8192
        )
        let swigluOut = try model.debugPrefillLastTokenBufferSnapshot(
            tokens: tokens,
            stepIndex: 219,
            buffer: prefillPlan.buffers.scratch,
            baseOffset: 0,
            rowStride: scratchRowStride,
            count: 8192
        )
        let downOut = try model.debugPrefillLastTokenBufferSnapshot(
            tokens: tokens,
            stepIndex: 220,
            buffer: prefillPlan.buffers.scratch,
            baseOffset: scratchSlotByteStride,
            rowStride: scratchRowStride,
            count: 2048
        )

        let normScale = try readTensorVector(
            try #require(safetensors.tensor(for: "model.layers.13.ffn_norm.weight"))
        )
        let gateTensor = try #require(safetensors.tensor(for: "model.layers.13.feed_forward.w1.weight"))
        let upTensor = try #require(safetensors.tensor(for: "model.layers.13.feed_forward.w3.weight"))
        let downTensor = try #require(safetensors.tensor(for: "model.layers.13.feed_forward.w2.weight"))

        let normReference = manualRMSNorm(
            input: input,
            scale: normScale,
            epsilon: spec.config.normEps
        )
        let boundNormReference = manualRMSNorm(
            input: input,
            scale: boundNormScale,
            epsilon: boundNormEpsilon
        )
        let gateReference = try manualDenseProjection(
            input: normReference,
            tensor: gateTensor,
            outputDimension: 8192,
            inputDimension: 2048
        )
        let upReference = try manualDenseProjection(
            input: normReference,
            tensor: upTensor,
            outputDimension: 8192,
            inputDimension: 2048
        )
        let swigluReference = manualSwiGLU(gate: gateReference, up: upReference)
        let downReference = try manualDenseProjection(
            input: swigluReference,
            tensor: downTensor,
            outputDimension: 2048,
            inputDimension: 8192
        )

        let normErr = maxAbsoluteError(normOut, normReference)
        let normErrFromBoundScale = maxAbsoluteError(normOut, boundNormReference)
        let normScaleErr = maxAbsoluteError(boundNormScale, normScale)
        let gateErr = maxAbsoluteError(gateOut, gateReference)
        let upErr = maxAbsoluteError(upOut, upReference)
        let swigluErr = maxAbsoluteError(swigluOut, swigluReference)
        let downErr = maxAbsoluteError(downOut, downReference)

        print("[StrictRef] layer13 normErr=\(String(format: "%.6f", normErr))")
        print("[StrictRef] layer13 normErr(boundScale)=\(String(format: "%.6f", normErrFromBoundScale))")
        print("[StrictRef] layer13 normScaleErr=\(String(format: "%.6f", normScaleErr))")
        print("[StrictRef] layer13 gateErr=\(String(format: "%.6f", gateErr))")
        print("[StrictRef] layer13 upErr=\(String(format: "%.6f", upErr))")
        print("[StrictRef] layer13 swigluErr=\(String(format: "%.6f", swigluErr))")
        print("[StrictRef] layer13 downErr=\(String(format: "%.6f", downErr))")

        #expect(normErr < 0.01, "layer13 ffn_norm drifted: maxErr=\(normErr)")
        #expect(gateErr < 0.05, "layer13 gate_proj drifted: maxErr=\(gateErr)")
        #expect(upErr < 0.05, "layer13 up_proj drifted: maxErr=\(upErr)")
        #expect(swigluErr < 0.05, "layer13 swiglu drifted: maxErr=\(swigluErr)")
        #expect(downErr < 0.05, "layer13 down_proj drifted: maxErr=\(downErr)")
    }

    @Test("Layer13 ffn_norm scale binding matches safetensors", .timeLimit(.minutes(2)))
    func layer13FFNNormScaleBindingMatchesSafetensors() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try setupOrSkip()
        let device = try #require(MTLCreateSystemDefaultDevice())
        let safetensors = try loadModelSafetensors(device: device)
        let prefillPlan = try #require(env.model.prefillPlan)
        let normStep = prefillPlan.steps[216]
        let normScaleBinding = try #require(normStep.bindings.buffers.first(where: { $0.index == 1 }))
        let boundNormScale = readBufferSlice(
            buffer: normScaleBinding.buffer,
            offset: normScaleBinding.offset,
            count: 2048,
            precision: .bfloat16
        )
        let referenceScale = try readTensorVector(
            try #require(safetensors.tensor(for: "model.layers.13.ffn_norm.weight"))
        )
        let scaleErr = maxAbsoluteError(boundNormScale, referenceScale)
        print("[StrictRef] layer13 ffn_norm scale binding maxErr=\(String(format: "%.6f", scaleErr))")
        #expect(scaleErr < 0.0001, "layer13 ffn_norm scale binding drifted: maxErr=\(scaleErr)")
    }

    @Test("Layer13 ffn_norm manual separate-output dispatch matches reference", .timeLimit(.minutes(2)))
    func layer13FFNNormManualSeparateOutputDispatchMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try setupOrSkip()
        let tokens = try readInputTokens(env.ref)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let safetensors = try loadModelSafetensors(device: device)
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        var model = env.model

        let prefillPlan = try #require(model.prefillPlan)
        let normStep = prefillPlan.steps[216]
        let hiddenRowStride = prefillPlan.buffers.hidden.length
            / MemoryLayout<Float>.stride
            / prefillPlan.maximumSequenceLength
        let outputLength = prefillPlan.maximumSequenceLength
            * hiddenRowStride
            * MemoryLayout<Float>.stride
        let outputBuffer = try #require(
            device.makeBuffer(length: outputLength, options: .storageModeShared)
        )
        let manualOutput = try model.debugPrefillLastTokenBufferSnapshotManualDispatch(
            tokens: tokens,
            prefixThroughStepIndex: 215,
            pipeline: normStep.pipeline,
            gridSize: normStep.gridSize,
            threadgroupSize: normStep.threadgroupSize,
            threadgroupMemoryLength: normStep.threadgroupMemoryLength,
            bufferBindings: normStep.bufferBindings.map { binding in
                if binding.index == 2 {
                    return (index: binding.index, buffer: outputBuffer, offset: 0)
                }
                return binding
            },
            bytesBindings: residentBytesBindings(from: normStep.bindings),
            runtimeSequenceLengthBindingIndex: normStep.sequenceLengthPolicy.bindingIndex,
            outputBuffer: outputBuffer,
            outputBaseOffset: 0,
            outputRowStride: hiddenRowStride,
            count: 2048
        )
        let splitPassHiddenOutput = try model.debugPrefillLastTokenBufferSnapshotSplitPass(
            tokens: tokens,
            prefixThroughStepIndex: 215,
            isolatedStepIndex: 216,
            buffer: prefillPlan.buffers.hidden,
            baseOffset: 0,
            rowStride: hiddenRowStride,
            count: 2048
        )
        let inputSnapshots = try model.debugPrefillLastTokenHiddenSnapshots(
            tokens: tokens,
            stepIndices: [214]
        )
        let input = try #require(inputSnapshots[214])
        let normScale = try readTensorVector(
            try #require(safetensors.tensor(for: "model.layers.13.ffn_norm.weight"))
        )
        let reference = manualRMSNorm(
            input: input,
            scale: normScale,
            epsilon: spec.config.normEps
        )
        let manualErr = maxAbsoluteError(manualOutput, reference)
        let splitPassErr = maxAbsoluteError(splitPassHiddenOutput, reference)
        print("[StrictRef] layer13 manual separate-output normErr=\(String(format: "%.6f", manualErr))")
        print("[StrictRef] layer13 split-pass hidden-output normErr=\(String(format: "%.6f", splitPassErr))")
        #expect(manualErr < 0.01, "layer13 manual separate-output norm drifted: maxErr=\(manualErr)")
    }
    #endif

    private func setupOrSkip(referencePath: String = Self.referencePath) throws -> TestEnvironment {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SetupError.noDevice
        }

        let refURL = URL(fileURLWithPath: referencePath)
        guard FileManager.default.fileExists(atPath: refURL.path) else {
            throw SetupError.noReference
        }

        let stafURL = URL(fileURLWithPath: Self.stafPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            throw SetupError.noSTAF
        }

        let ref = try SafetensorsLoader().load(at: refURL, device: device)
        let store = try STAFLoader().load(at: stafURL, device: device)
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        let compiler = MetalInferenceCompiler(
            decodeBufferPrecisionOverride: decodeBufferPrecisionOverrideFromEnvironment()
        )
        let decodePlan = try compiler.compile(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            stafWeightStore: store,
            device: device
        )
        let prefillPlan = try compiler.compilePrefill(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            sharedKVCache: decodePlan.buffers.kvCache,
            sharedConvState: decodePlan.buffers.convState,
            sharedConvStateDimension: decodePlan.buffers.convStateDimension,
            sharedConvStateKernelSize: decodePlan.buffers.convStateKernelSize,
            sharedRecurrentState: decodePlan.buffers.recurrentState,
            sharedRecurrentStateBytesPerLayer: decodePlan.buffers.recurrentStateBytesPerLayer,
            device: device
        )

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan
        return TestEnvironment(model: model, ref: ref)
    }

    private func setupModelOnly() throws -> MetalInferenceModel {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SetupError.noDevice
        }

        let stafURL = URL(fileURLWithPath: Self.stafPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            throw SetupError.noSTAF
        }

        let store = try STAFLoader().load(at: stafURL, device: device)
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        let compiler = MetalInferenceCompiler(
            decodeBufferPrecisionOverride: decodeBufferPrecisionOverrideFromEnvironment()
        )
        let decodePlan = try compiler.compile(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            stafWeightStore: store,
            device: device
        )
        let prefillPlan = try compiler.compilePrefill(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            sharedKVCache: decodePlan.buffers.kvCache,
            sharedConvState: decodePlan.buffers.convState,
            sharedConvStateDimension: decodePlan.buffers.convStateDimension,
            sharedConvStateKernelSize: decodePlan.buffers.convStateKernelSize,
            sharedRecurrentState: decodePlan.buffers.recurrentState,
            sharedRecurrentStateBytesPerLayer: decodePlan.buffers.recurrentStateBytesPerLayer,
            device: device
        )

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan
        return model
    }

    private func readInputTokens(_ file: MetalWeightFile) throws -> [Int32] {
        guard let info = file.tensors["ref.input_tokens"] else {
            throw SetupError.tensorNotFound("ref.input_tokens")
        }
        let count = info.shape.reduce(1, *)
        let pointer = (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: Int32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func decodeBufferPrecisionOverrideFromEnvironment() -> BufferPrecision? {
        let value = ProcessInfo.processInfo.environment["SWIFTLM_DECODE_BUFFER_PRECISION"]?
            .lowercased()
        switch value {
        case "f32", "float32":
            return .float32Decode
        case "bf16", "bfloat16":
            return .bfloat16
        case "f16", "float16":
            return .float16
        default:
            return nil
        }
    }

    private func readRefTensorAsFloats(_ file: MetalWeightFile, name: String) throws -> [Float] {
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
            throw SetupError.tensorNotFound("Unsupported dtype \(info.dtype) for \(name)")
        }
    }

    private func loadModelSafetensors(device: MTLDevice) throws -> MetalWeightStore {
        let modelDirectory = URL(fileURLWithPath: Self.modelDirectoryPath)
        let safetensorURLs = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return try SafetensorsLoader().loadAll(urls: safetensorURLs, device: device)
    }

    private func readTensorRow(
        tensor: MetalTensor,
        rowIndex: Int,
        columnCount: Int
    ) throws -> [Float] {
        guard tensor.shape.count >= 2 else {
            throw SetupError.tensorNotFound("Expected rank-2 tensor row access")
        }
        let rowCount = tensor.shape[0]
        guard rowIndex >= 0, rowIndex < rowCount else {
            throw SetupError.tensorNotFound("Tensor row index out of bounds")
        }
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)
        let start = rowIndex * columnCount
        switch tensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: tensor.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                Float(Float16(bitPattern: pointer[start + index]))
            }
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: tensor.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                Float(bitPattern: UInt32(pointer[start + index]) << 16)
            }
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: tensor.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                pointer[start + index]
            }
        case .quantized:
            throw SetupError.tensorNotFound("Quantized safetensors row access unsupported")
        }
    }

    private func readTensorVector(_ tensor: MetalTensor) throws -> [Float] {
        let count = tensor.shape.reduce(1, *)
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)
        switch tensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { index in
                Float(Float16(bitPattern: pointer[index]))
            }
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { index in
                Float(bitPattern: UInt32(pointer[index]) << 16)
            }
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: count)
            return (0..<count).map { index in
                pointer[index]
            }
        case .quantized:
            throw SetupError.tensorNotFound("Quantized safetensors vector access unsupported")
        }
    }

    #if ENABLE_METAL_PROBES
    private func residentBytesBindings(
        from table: MetalBindingTable
    ) -> [(index: Int, value: [UInt8])] {
        table.constants.compactMap { constant in
            guard case .buffer(let binding) = constant else {
                return nil
            }
            let base = binding.buffer.contents().advanced(by: binding.offset)
            let bytes = Array(
                UnsafeBufferPointer(
                    start: base.assumingMemoryBound(to: UInt8.self),
                    count: binding.length
                )
            )
            return (index: binding.index, value: bytes)
        }
        .sorted { $0.index < $1.index }
    }
    #endif

    private func readBufferSlice(
        buffer: MTLBuffer,
        offset: Int,
        count: Int,
        precision: BufferPrecision
    ) -> [Float] {
        switch precision {
        case .float16:
            let pointer = (buffer.contents() + offset).bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { index in
                Float(pointer[index])
            }
        case .bfloat16:
            let pointer = (buffer.contents() + offset).bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { index in
                Float(pointer[index])
            }
        case .float32, .float32Decode:
            let pointer = (buffer.contents() + offset).bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        }
    }

    private func readDecodeScratchSlot(
        model: MetalInferenceModel,
        slotIndex: Int,
        count: Int
    ) throws -> [Float] {
        let precision = model.buffers.bufferPrecision
        let slotDimension = model.buffers.scratch.length / (5 * precision.byteSize)
        let offset = slotIndex * slotDimension * precision.byteSize
        return try readBufferSliceViaBlit(
            buffer: model.buffers.scratch,
            offset: offset,
            count: count,
            precision: precision,
            device: model.device
        )
    }

    private func readBufferSliceViaBlit(
        buffer: MTLBuffer,
        offset: Int,
        count: Int,
        precision: BufferPrecision,
        device: MTLDevice
    ) throws -> [Float] {
        let byteCount = count * precision.byteSize
        guard let staging = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let queue = device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder()
        else {
            throw SetupError.noDevice
        }
        blit.copy(
            from: buffer,
            sourceOffset: offset,
            to: staging,
            destinationOffset: 0,
            size: byteCount
        )
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return readBufferSlice(buffer: staging, offset: 0, count: count, precision: precision)
    }

    private enum LiveKVKind {
        case keys
        case values
    }

    private func readLiveKVCacheToken(
        model: MetalInferenceModel,
        layerIndex: Int,
        position: Int,
        kind: LiveKVKind
    ) throws -> [Float] {
        let cache = try #require(model.buffers.kvCache)
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
        let precision = try denseKVPrecision(for: scheme)
        var values: [Float] = []
        values.reserveCapacity(spec.kvHeadCount * spec.headDimension)
        for head in 0..<spec.kvHeadCount {
            let offset = spec.offset(layer: layerIndex, head: head, position: position, scheme: scheme)
            values.append(contentsOf: try readBufferSliceViaBlit(
                buffer: buffer,
                offset: offset,
                count: spec.headDimension,
                precision: precision,
                device: model.device
            ))
        }
        return values
    }

    private func denseKVPrecision(for scheme: QuantizationSchemeIdentifier) throws -> BufferPrecision {
        switch scheme {
        case .fp16RowMajor:
            return .float16
        case .bf16RowMajor:
            return .bfloat16
        case .fp32RowMajor:
            return .float32
        default:
            throw SetupError.tensorNotFound("Diagnostic only supports dense KV cache, got \(scheme)")
        }
    }

    private func readFloatBinding(_ bytes: [UInt8]) -> Float {
        precondition(bytes.count == MemoryLayout<Float>.size)
        return bytes.withUnsafeBytes { $0.load(as: Float.self) }
    }

    private func finalHiddenInputBuffer(for plan: MetalDispatchPlan) -> MTLBuffer {
        let projectionStepIndex = plan.steps.count - 2
        let projectionStep = plan.steps[projectionStepIndex]
        guard let binding = projectionStep.bufferBindings.first(where: { $0.index == 0 }) else {
            fatalError("Output head projection missing input buffer binding")
        }
        return binding.buffer
    }

    private func decodeBufferLabel(_ buffer: MTLBuffer, model: MetalInferenceModel) -> String {
        if buffer === model.buffers.hidden { return "hidden" }
        if buffer === model.buffers.residual { return "residual" }
        if buffer === model.buffers.scratch { return "scratch" }
        if buffer === model.buffers.logits { return "logits" }
        if buffer === model.buffers.tokenIn { return "tokenIn" }
        if buffer === model.buffers.position { return "position" }
        return "weight/other"
    }

    private func finalNormStepIndex(for plan: MetalDispatchPlan) -> Int {
        precondition(plan.steps.count >= 3, "Decode plan missing final norm/output head steps")
        return plan.steps.count - 3
    }

    private func writeDecodeBuffer(_ values: [Float], to buffer: MTLBuffer, precision: BufferPrecision) {
        switch precision {
        case .float32, .float32Decode:
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: values.count)
            for index in values.indices {
                pointer[index] = values[index]
            }
        case .float16:
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: values.count)
            for index in values.indices {
                pointer[index] = Float16(values[index])
            }
        case .bfloat16:
            let pointer = buffer.contents().bindMemory(to: BFloat16.self, capacity: values.count)
            for index in values.indices {
                pointer[index] = BFloat16(values[index])
            }
        }
    }

    private func zeroDecodeBuffer(_ buffer: MTLBuffer, precision: BufferPrecision) {
        switch precision {
        case .float32, .float32Decode:
            let count = buffer.length / MemoryLayout<Float>.stride
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
            for index in 0..<count {
                pointer[index] = 0
            }
        case .float16:
            let count = buffer.length / MemoryLayout<Float16>.stride
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            for index in 0..<count {
                pointer[index] = 0
            }
        case .bfloat16:
            let count = buffer.length / MemoryLayout<BFloat16>.stride
            let pointer = buffer.contents().bindMemory(to: BFloat16.self, capacity: count)
            for index in 0..<count {
                pointer[index] = 0
            }
        }
    }

    private func readKVCacheSnapshotValues(
        _ snapshot: MetalInferenceModel.DebugKVCacheLayerSnapshot,
        sequenceLength: Int
    ) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(snapshot.kvHeadCount * sequenceLength * 64)
        for head in 0..<snapshot.kvHeadCount {
            for position in 0..<sequenceLength {
                let base = head * snapshot.maximumSequenceLength * snapshot.bytesPerHeadSlot
                    + position * snapshot.bytesPerHeadSlot
                switch snapshot.scheme {
                case .bf16RowMajor:
                    for index in 0..<64 {
                        let byteOffset = base + index * MemoryLayout<UInt16>.stride
                        let bits = UInt16(snapshot.bytes[byteOffset])
                            | (UInt16(snapshot.bytes[byteOffset + 1]) << 8)
                        values.append(Float(bitPattern: UInt32(bits) << 16))
                    }
                case .fp16RowMajor:
                    for index in 0..<64 {
                        let byteOffset = base + index * MemoryLayout<UInt16>.stride
                        let bits = UInt16(snapshot.bytes[byteOffset])
                            | (UInt16(snapshot.bytes[byteOffset + 1]) << 8)
                        values.append(Float(Float16(bitPattern: bits)))
                    }
                case .fp32RowMajor:
                    for index in 0..<64 {
                        let byteOffset = base + index * MemoryLayout<Float>.stride
                        let bits = UInt32(snapshot.bytes[byteOffset])
                            | (UInt32(snapshot.bytes[byteOffset + 1]) << 8)
                            | (UInt32(snapshot.bytes[byteOffset + 2]) << 16)
                            | (UInt32(snapshot.bytes[byteOffset + 3]) << 24)
                        values.append(Float(bitPattern: bits))
                    }
                default:
                    fatalError("Unsupported dense KV snapshot scheme: \(snapshot.scheme)")
                }
            }
        }
        return values
    }

    private func makeDispatchDump(compiler: MetalInferenceCompiler) throws -> String {
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        return compiler.dumpDispatchEntries(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize
        )
    }

    private func parseDispatchEntries(from dump: String) -> [ParsedDispatchEntry] {
        dump.split(separator: "\n").compactMap { line in
            guard let bracketEnd = line.firstIndex(of: "]") else { return nil }
            let tail = line[line.index(after: bracketEnd)...].trimmingCharacters(in: .whitespaces)
            if tail.hasPrefix("-- ") {
                return ParsedDispatchEntry(layer: nil, kind: String(tail.dropFirst(3)))
            }
            guard tail.first == "L" else { return nil }
            let pieces = tail.split(separator: " ", maxSplits: 1).map(String.init)
            guard pieces.count == 2, let layer = Int(pieces[0].dropFirst()) else { return nil }
            return ParsedDispatchEntry(layer: layer, kind: pieces[1])
        }
    }

    private func readSTAFRow(
        tensorName: String,
        rowIndex: Int,
        columnCount: Int,
        store: STAFWeightStore
    ) throws -> [Float] {
        guard let entry = store.entries[tensorName],
              let access = store.bufferAccess(for: tensorName),
              let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier) else {
            throw SetupError.tensorNotFound(tensorName)
        }
        let start = rowIndex * columnCount
        let basePointer = access.buffer.contents().advanced(by: access.offset)
        switch format.schemeIdentifier {
        case .fp16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: entry.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                Float(Float16(bitPattern: pointer[start + index]))
            }
        case .bf16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: entry.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                Float(bitPattern: UInt32(pointer[start + index]) << 16)
            }
        case .fp32RowMajor:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: entry.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                pointer[start + index]
            }
        default:
            throw SetupError.tensorNotFound("Unsupported STAF format for \(tensorName)")
        }
    }

    private func readF32Buffer(_ buffer: MTLBuffer) -> [Float] {
        let count = buffer.length / MemoryLayout<Float>.stride
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readDecodeBuffer(_ buffer: MTLBuffer, precision: BufferPrecision) -> [Float] {
        if buffer.storageMode == .private {
            let device = buffer.device
            guard let staging = device.makeBuffer(length: buffer.length, options: .storageModeShared),
                  let queue = device.makeCommandQueue(),
                  let commandBuffer = queue.makeCommandBuffer(),
                  let blit = commandBuffer.makeBlitCommandEncoder() else { return [] }
            blit.copy(from: buffer, sourceOffset: 0, to: staging, destinationOffset: 0, size: buffer.length)
            blit.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return readDecodeBuffer(staging, precision: precision)
        }

        switch precision {
        case .float32, .float32Decode:
            return readF32Buffer(buffer)
        case .float16:
            let count = buffer.length / MemoryLayout<Float16>.stride
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let count = buffer.length / MemoryLayout<UInt16>.stride
            let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Float(bitPattern: UInt32(pointer[$0]) << 16) }
        }
    }

    private func argmax(_ values: [Float]) -> (index: Int, value: Float) {
        var bestIndex = 0
        var bestValue = values.first ?? -.infinity
        for (index, value) in values.enumerated() where value > bestValue {
            bestIndex = index
            bestValue = value
        }
        return (bestIndex, bestValue)
    }

    private func topK(_ values: [Float], k: Int) -> [(index: Int, value: Float)] {
        values.enumerated()
            .sorted { lhs, rhs in
                if lhs.element == rhs.element { return lhs.offset < rhs.offset }
                return lhs.element > rhs.element
            }
            .prefix(k)
            .map { ($0.offset, $0.element) }
    }

    private func formatTopK(_ values: [(index: Int, value: Float)]) -> [String] {
        values.map { entry in
            "(\(entry.index),\(String(format: "%.1f", entry.value)))"
        }
    }

    private func formatPrefix(_ values: [Float], count: Int) -> [String] {
        values.prefix(count).map { String(format: "%.4f", $0) }
    }

    private func dumpStepBindings(
        _ step: MetalPrefillStep,
        stepIndex: Int,
        prefillPlan: MetalPrefillPlan
    ) {
        let kernel = step.metadata.kernelName ?? step.pipeline.label ?? "unknown"
        let bindings = step.bufferBindings.map { binding in
            "i\(binding.index)=\(bufferLabel(binding.buffer, prefillPlan: prefillPlan))@\(binding.offset)"
        }.joined(separator: ", ")
        print("[StrictRef] step[\(stepIndex)] \(kernel) bindings: \(bindings)")
    }

    private func bufferLabel(
        _ buffer: MTLBuffer,
        prefillPlan: MetalPrefillPlan
    ) -> String {
        if buffer === prefillPlan.buffers.hidden { return "hidden" }
        if buffer === prefillPlan.buffers.scratch { return "scratch" }
        if buffer === prefillPlan.buffers.residual { return "residual" }
        if buffer === prefillPlan.buffers.logits { return "logits" }
        if buffer === prefillPlan.buffers.tokenIDs { return "tokenIDs" }
        if buffer === prefillPlan.buffers.positions { return "positions" }
        if buffer === prefillPlan.buffers.runtimeConstantBuffer { return "runtimeConstant" }
        return "buffer"
    }

    private func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0 }
        var maxErr: Float = 0
        for index in 0..<count {
            maxErr = max(maxErr, abs(lhs[index] - rhs[index]))
        }
        return maxErr
    }

    private func manualRMSNorm(
        input: [Float],
        scale: [Float],
        epsilon: Float
    ) -> [Float] {
        let meanSquare = input.reduce(Float.zero) { partial, value in
            partial + value * value
        } / Float(max(input.count, 1))
        let invRMS = 1 / sqrt(meanSquare + epsilon)
        return zip(input, scale).map { value, weight in
            value * invRMS * weight
        }
    }

    private func manualDenseProjection(
        input: [Float],
        tensor: MetalTensor,
        outputDimension: Int,
        inputDimension: Int
    ) throws -> [Float] {
        guard tensor.shape.count >= 2 else {
            throw SetupError.tensorNotFound("Expected rank-2 tensor projection access")
        }
        return try (0..<outputDimension).map { rowIndex in
            let row = try readTensorRow(
                tensor: tensor,
                rowIndex: rowIndex,
                columnCount: inputDimension
            )
            var sum: Float = 0
            for index in 0..<inputDimension {
                sum += row[index] * input[index]
            }
            return sum
        }
    }

    private func manualSwiGLU(gate: [Float], up: [Float]) -> [Float] {
        zip(gate, up).map { gate, up in
            let activated = gate * (1 / (1 + exp(-gate)))
            return activated * up
        }
    }

    private enum SetupError: Error {
        case noDevice
        case noReference
        case noSTAF
        case tensorNotFound(String)
    }
}
