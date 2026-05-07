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

    private static let referencePath = URL(fileURLWithPath: BenchmarkSupport.testDataPath)
        .appendingPathComponent("qwen35_reference.safetensors")
        .path

    @Test("Reference snapshot schema is complete")
    func referenceSnapshotSchemaIsComplete() throws {
        let env = try Self.setupOrSkip()
        let requiredTensors = [
            "ref.meta.schema_version",
            "ref.meta.case_count",
            "ref.meta.decode_steps",
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
        let referenceConfigHash = try Self.readRefUInt8Array(env.ref, name: "ref.meta.config_sha256")
        let torchVersion = try Self.readRefUInt8Array(env.ref, name: "ref.meta.torch_version_utf8")
        let transformersVersion = try Self.readRefUInt8Array(env.ref, name: "ref.meta.transformers_version_utf8")
        let bundleConfigHash = try Self.configSHA256(bundlePath: env.bundlePath)

        #expect(schemaVersion == 4, "Unexpected Qwen35 reference schema version: \(schemaVersion)")
        #expect(Int(caseCount) == Self.referenceCases.count)
        #expect(decodeSteps >= 1)
        #expect(fastBackendAvailable == 0 || fastBackendAvailable == 1)
        #expect(!torchVersion.isEmpty)
        #expect(!transformersVersion.isEmpty)
        #expect(referenceConfigHash == bundleConfigHash, "Qwen35 reference config does not match the Swift bundle config")

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
        let bundlePath: String
    }

    private static func setupOrSkip() throws -> TestEnvironment {
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
        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: bundlePath,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64)
        )
        return TestEnvironment(model: model, ref: ref, bundlePath: bundlePath)
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
    }
}
#endif
