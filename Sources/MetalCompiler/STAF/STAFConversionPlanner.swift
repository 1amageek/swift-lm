import Foundation

struct STAFConversionPlanner: Sendable {

    func plan(
        safetensorsURLs: [URL],
        quantization: MLXQuantizationHint?
    ) throws -> STAFConversionPlan {
        let sortedURLs = safetensorsURLs.sorted { $0.lastPathComponent < $1.lastPathComponent }

        let loader = SafetensorsLoader()
        var allTensors: [(name: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)] = []

        for (shardIndex, url) in sortedURLs.enumerated() {
            let tensors = try loader.parseHeader(at: url)
            for tensor in tensors {
                allTensors.append((name: tensor.name, info: tensor, shardIndex: shardIndex, shardURL: url))
            }
        }

        let consumedCompanions = consumedCompanions(in: allTensors)
        var entries: [STAFConversionEntry] = []
        entries.reserveCapacity(allTensors.count)

        for (name, info, shardIndex, shardURL) in allTensors {
            if consumedCompanions.contains(name) {
                continue
            }

            entries.append(
                STAFConversionEntry(
                    name: name,
                    info: info,
                    shardIndex: shardIndex,
                    shardURL: shardURL,
                    schemeIdentifier: try determineScheme(
                        name: name,
                        info: info,
                        allTensors: allTensors,
                        quantization: quantization
                    ),
                    semanticRole: inferSemanticRole(name: name),
                    originalDType: mapOriginalDType(info.dtype)
                )
            )
        }

        return STAFConversionPlan(sortedURLs: sortedURLs, entries: entries)
    }

    private func consumedCompanions(
        in allTensors: [(name: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)]
    ) -> Set<String> {
        var consumed = Set<String>()
        for (name, _, _, _) in allTensors where name.hasSuffix(".weight") {
            let modulePath = String(name.dropLast(".weight".count))
            let scalesName = modulePath + ".scales"
            let biasesName = modulePath + ".biases"
            if allTensors.contains(where: { $0.name == scalesName }),
               allTensors.contains(where: { $0.name == biasesName }) {
                consumed.insert(scalesName)
                consumed.insert(biasesName)
            }
        }
        return consumed
    }

    private func determineScheme(
        name: String,
        info: SafetensorsTensorInfo,
        allTensors: [(name: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)],
        quantization: MLXQuantizationHint?
    ) throws -> QuantizationSchemeIdentifier {
        if name.hasSuffix(".weight") {
            let modulePath = String(name.dropLast(".weight".count))
            let hasScales = allTensors.contains { $0.name == modulePath + ".scales" }
            let hasBiases = allTensors.contains { $0.name == modulePath + ".biases" }

            if hasScales && hasBiases {
                guard let hint = quantization else {
                    throw STAFConversionError.missingQuantizationHint(name)
                }
                guard let format = QuantizationFormatRegistry.formatForMLXQuantization(
                    bits: hint.bits,
                    groupSize: hint.groupSize
                ) else {
                    throw STAFConversionError.unsupportedQuantization(
                        bits: hint.bits,
                        groupSize: hint.groupSize
                    )
                }
                try verifyTensorShape(
                    name: name,
                    weightShape: info.shape,
                    scalesInfo: allTensors.first { $0.name == modulePath + ".scales" }?.info,
                    hint: hint
                )
                return format.schemeIdentifier
            }
        }

        if name.hasSuffix(".scales") || name.hasSuffix(".biases") {
            return .passthrough
        }

        switch info.dtype {
        case .float16: return .fp16RowMajor
        case .bfloat16: return .bf16RowMajor
        case .float32: return .fp32RowMajor
        default: return .passthrough
        }
    }

    /// Confirm that the tensor shapes are consistent with the quantization hint.
    ///
    /// `input_dim = packed_dim × (32 / bits) = num_groups × group_size`. Any
    /// mismatch indicates a corrupt bundle or wrong hint — fail loudly instead
    /// of silently mislabeling the scheme.
    private func verifyTensorShape(
        name: String,
        weightShape: [Int],
        scalesInfo: SafetensorsTensorInfo?,
        hint: MLXQuantizationHint
    ) throws {
        guard weightShape.count >= 2 else {
            throw STAFConversionError.inconsistentQuantizationShape(
                name: name,
                reason: "weight shape has <2 dims: \(weightShape)"
            )
        }
        let packedDimension = weightShape[weightShape.count - 1]
        let inputDimFromWeight = packedDimension * 32 / hint.bits
        if (packedDimension * 32) % hint.bits != 0 {
            throw STAFConversionError.inconsistentQuantizationShape(
                name: name,
                reason: "packed_dim=\(packedDimension) is not divisible by bits=\(hint.bits)"
            )
        }
        if inputDimFromWeight % hint.groupSize != 0 {
            throw STAFConversionError.inconsistentQuantizationShape(
                name: name,
                reason: "input_dim=\(inputDimFromWeight) is not divisible by group_size=\(hint.groupSize)"
            )
        }
        if let scalesShape = scalesInfo?.shape, scalesShape.count >= 2 {
            let numberOfGroups = scalesShape[scalesShape.count - 1]
            let inputDimFromScales = numberOfGroups * hint.groupSize
            if inputDimFromScales != inputDimFromWeight {
                throw STAFConversionError.inconsistentQuantizationShape(
                    name: name,
                    reason: "input_dim from weight=\(inputDimFromWeight) " +
                            "!= input_dim from scales=\(inputDimFromScales) " +
                            "(bits=\(hint.bits), group_size=\(hint.groupSize))"
                )
            }
        }
    }

    private func inferSemanticRole(name: String) -> SemanticRole {
        if name.contains("embed_tokens") || name.contains("token_embd") {
            return .tokenEmbedding
        }
        if name.contains("q_proj") { return .attentionQuery }
        if name.contains("k_proj") { return .attentionKey }
        if name.contains("v_proj") { return .attentionValue }
        if name.contains("o_proj") || name.contains("out_proj") { return .attentionOutput }
        if name.contains("gate_proj") || name.contains(".w1.") { return .mlpGate }
        if name.contains("up_proj") || name.contains(".w3.") { return .mlpUp }
        if name.contains("down_proj") || name.contains(".w2.") { return .mlpDown }
        if name.contains("layernorm") || name.contains("norm") && name.hasSuffix(".weight") {
            return .normWeight
        }
        if name.contains("lm_head") { return .languageModelHead }
        if name.contains("experts") && name.contains("gate") { return .moeExpertGate }
        if name.contains("experts") && name.contains("up") { return .moeExpertUp }
        if name.contains("experts") && name.contains("down") { return .moeExpertDown }
        if name.contains("router") || name.contains("gate.weight") && name.contains("moe") {
            return .moeRouter
        }
        return .unknown
    }

    private func mapOriginalDType(_ dtype: SafetensorsDType) -> OriginalDType {
        switch dtype {
        case .float32: return .float32
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .int32: return .int32
        case .int16: return .int16
        case .int8: return .int8
        default: return .unknown
        }
    }
}
