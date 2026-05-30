import Foundation

struct STAFConversionPlanner: Sendable {

    func plan(
        safetensorsURLs: [URL],
        quantization: MLXQuantizationHint?
    ) throws -> STAFConversionPlan {
        let sortedURLs = safetensorsURLs.sorted { $0.lastPathComponent < $1.lastPathComponent }

        let loader = SafetensorsLoader()
        var rawTensors: [(sourceName: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)] = []

        for (shardIndex, url) in sortedURLs.enumerated() {
            let tensors = try loader.parseHeader(at: url)
            for tensor in tensors {
                rawTensors.append((sourceName: tensor.name, info: tensor, shardIndex: shardIndex, shardURL: url))
            }
        }

        // Detect the source convention from raw names, then canonicalize every
        // tensor before companion discovery so `.weight`/`.scales`/`.biases`
        // sibling matching sees a single consistent namespace.
        let canonicalizer = TensorNameCanonicalizer.detect(from: rawTensors.map { $0.sourceName })
        let allTensors = rawTensors.map { raw -> (name: String, sourceName: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL) in
            (name: canonicalizer.canonicalize(raw.sourceName),
             sourceName: raw.sourceName,
             info: raw.info,
             shardIndex: raw.shardIndex,
             shardURL: raw.shardURL)
        }

        let packedMoEEntries = try packedMoEEntries(in: allTensors, quantization: quantization)
        let consumedPackedMoETensors = Set(
            packedMoEEntries.flatMap { entry in entry.packedMoE?.consumedTensorNames ?? [] }
        )
        let consumedCompanions = consumedCompanions(in: allTensors)
        var entries: [STAFConversionEntry] = []
        entries.reserveCapacity(allTensors.count + packedMoEEntries.count)

        for tensor in allTensors {
            if consumedCompanions.contains(tensor.name) || consumedPackedMoETensors.contains(tensor.name) {
                continue
            }

            entries.append(
                STAFConversionEntry(
                    name: tensor.name,
                    sourceName: tensor.sourceName,
                    info: tensor.info,
                    shardIndex: tensor.shardIndex,
                    shardURL: tensor.shardURL,
                    schemeIdentifier: try determineScheme(
                        name: tensor.name,
                        info: tensor.info,
                        allTensors: allTensors,
                        quantization: quantization
                    ),
                    semanticRole: inferSemanticRole(name: tensor.name),
                    originalDType: mapOriginalDType(tensor.info.dtype)
                )
            )
        }
        entries.append(contentsOf: packedMoEEntries)

        return STAFConversionPlan(sortedURLs: sortedURLs, entries: entries)
    }

    private func consumedCompanions(
        in allTensors: [(name: String, sourceName: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)]
    ) -> Set<String> {
        var consumed = Set<String>()
        for tensor in allTensors where tensor.name.hasSuffix(".weight") {
            let modulePath = String(tensor.name.dropLast(".weight".count))
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

    private func packedMoEEntries(
        in allTensors: [(name: String, sourceName: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)],
        quantization: MLXQuantizationHint?
    ) throws -> [STAFConversionEntry] {
        let expertPattern = #/^(.+\.feed_forward)\.experts\.(\d+)\.(w[123])\.weight$/#
        var groups: [String: [Int: [String: (sourceName: String, info: SafetensorsTensorInfo, shardURL: URL)]]] = [:]
        for tensor in allTensors {
            guard let match = tensor.name.wholeMatch(of: expertPattern) else { continue }
            let prefix = String(match.1)
            let expert = Int(match.2) ?? -1
            let role = String(match.3)
            guard expert >= 0 else { continue }
            groups[prefix, default: [:]][expert, default: [:]][role] = (
                sourceName: tensor.sourceName,
                info: tensor.info,
                shardURL: tensor.shardURL
            )
        }

        var entries: [STAFConversionEntry] = []
        for (prefix, expertsByIndex) in groups {
            let expertIndices = expertsByIndex.keys.sorted()
            guard !expertIndices.isEmpty,
                  expertIndices == Array(0..<expertIndices.count) else { continue }

            var experts: [STAFPackedMoEExpertSources] = []
            experts.reserveCapacity(expertIndices.count)
            var gateShape: [Int]?
            var downShape: [Int]?
            var allBF16 = true
            var complete = true
            for expertIndex in expertIndices {
                guard let sources = expertsByIndex[expertIndex],
                      let gate = sources["w1"],
                      let down = sources["w2"],
                      let up = sources["w3"] else {
                    complete = false
                    break
                }
                gateShape = gateShape ?? gate.info.shape
                downShape = downShape ?? down.info.shape
                guard gate.info.shape == gateShape,
                      up.info.shape == gateShape,
                      down.info.shape == downShape else {
                    complete = false
                    break
                }
                allBF16 = allBF16
                    && gate.info.dtype == .bfloat16
                    && up.info.dtype == .bfloat16
                    && down.info.dtype == .bfloat16
                experts.append(STAFPackedMoEExpertSources(
                    gate: STAFPackedMoETensorSource(name: gate.sourceName, shardURL: gate.shardURL),
                    up: STAFPackedMoETensorSource(name: up.sourceName, shardURL: up.shardURL),
                    down: STAFPackedMoETensorSource(name: down.sourceName, shardURL: down.shardURL)
                ))
            }
            guard complete, allBF16,
                  let gateShape, gateShape.count == 2,
                  let downShape, downShape.count == 2 else { continue }

            let expertCount = experts.count
            let intermediateDimension = gateShape[0]
            let inputDimension = gateShape[1]
            let outputDimension = downShape[0]
            let gateUpInfo = SafetensorsTensorInfo(
                name: "\(prefix).experts.gate_up_proj",
                dtype: .bfloat16,
                shape: [expertCount, 2 * intermediateDimension, inputDimension],
                dataOffset: 0,
                byteCount: expertCount * 2 * intermediateDimension * inputDimension * MemoryLayout<UInt16>.stride
            )
            entries.append(STAFConversionEntry(
                name: gateUpInfo.name,
                sourceName: gateUpInfo.name,
                info: gateUpInfo,
                shardIndex: 0,
                shardURL: experts[0].gate.shardURL,
                schemeIdentifier: .bf16RowMajor,
                semanticRole: .moeExpertGate,
                originalDType: .bfloat16,
                packedMoE: STAFPackedMoEEntry(kind: .gateUp, experts: experts)
            ))

            let downInfo = SafetensorsTensorInfo(
                name: "\(prefix).experts.down_proj",
                dtype: .bfloat16,
                shape: [expertCount, outputDimension, intermediateDimension],
                dataOffset: 0,
                byteCount: expertCount * outputDimension * intermediateDimension * MemoryLayout<UInt16>.stride
            )
            entries.append(STAFConversionEntry(
                name: downInfo.name,
                sourceName: downInfo.name,
                info: downInfo,
                shardIndex: 0,
                shardURL: experts[0].down.shardURL,
                schemeIdentifier: .bf16RowMajor,
                semanticRole: .moeExpertDown,
                originalDType: .bfloat16,
                packedMoE: STAFPackedMoEEntry(kind: .down, experts: experts)
            ))
        }
        entries.append(contentsOf: try bulkSwitchMLPMoEEntries(in: allTensors, quantization: quantization))
        return entries
    }

    private func bulkSwitchMLPMoEEntries(
        in allTensors: [(name: String, sourceName: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)],
        quantization: MLXQuantizationHint?
    ) throws -> [STAFConversionEntry] {
        let gateSuffix = ".switch_mlp.gate_proj.weight"
        var entries: [STAFConversionEntry] = []
        for gateTensor in allTensors where gateTensor.name.hasSuffix(gateSuffix) {
            let prefix = String(gateTensor.name.dropLast(gateSuffix.count))
            let upName = prefix + ".switch_mlp.up_proj.weight"
            let downName = prefix + ".switch_mlp.down_proj.weight"
            guard let upTensor = allTensors.first(where: { $0.name == upName }),
                  let downTensor = allTensors.first(where: { $0.name == downName }) else {
                continue
            }
            guard gateTensor.info.shape.count == 3,
                  upTensor.info.shape == gateTensor.info.shape,
                  downTensor.info.shape.count == 3 else {
                continue
            }

            let expertCount = gateTensor.info.shape[0]
            let intermediateDimension = gateTensor.info.shape[1]
            let gatePackedDimension = gateTensor.info.shape[2]
            let outputDimension = downTensor.info.shape[1]
            let downPackedDimension = downTensor.info.shape[2]
            let gateScheme = try determineScheme(
                name: gateTensor.name,
                info: gateTensor.info,
                allTensors: allTensors,
                quantization: quantization
            )
            let upScheme = try determineScheme(
                name: upTensor.name,
                info: upTensor.info,
                allTensors: allTensors,
                quantization: quantization
            )
            let downScheme = try determineScheme(
                name: downTensor.name,
                info: downTensor.info,
                allTensors: allTensors,
                quantization: quantization
            )
            guard gateScheme == upScheme, gateScheme == downScheme else {
                continue
            }

            let bulk = STAFPackedMoEBulkSources(
                gate: STAFPackedMoETensorSource(
                    name: gateTensor.sourceName,
                    shardURL: gateTensor.shardURL,
                    info: gateTensor.info,
                    schemeIdentifier: gateScheme
                ),
                up: STAFPackedMoETensorSource(
                    name: upTensor.sourceName,
                    shardURL: upTensor.shardURL,
                    info: upTensor.info,
                    schemeIdentifier: upScheme
                ),
                down: STAFPackedMoETensorSource(
                    name: downTensor.sourceName,
                    shardURL: downTensor.shardURL,
                    info: downTensor.info,
                    schemeIdentifier: downScheme
                ),
                expertCount: expertCount,
                intermediateDimension: intermediateDimension,
                outputDimension: outputDimension
            )

            let gateUpInfo = SafetensorsTensorInfo(
                name: "\(prefix).experts.gate_up_proj",
                dtype: gateTensor.info.dtype,
                shape: [expertCount * 2 * intermediateDimension, gatePackedDimension],
                dataOffset: 0,
                byteCount: gateTensor.info.byteCount + upTensor.info.byteCount
            )
            entries.append(STAFConversionEntry(
                name: gateUpInfo.name,
                sourceName: gateUpInfo.name,
                info: gateUpInfo,
                shardIndex: gateTensor.shardIndex,
                shardURL: gateTensor.shardURL,
                schemeIdentifier: gateScheme,
                semanticRole: .moeExpertGate,
                originalDType: mapOriginalDType(gateTensor.info.dtype),
                packedMoE: STAFPackedMoEEntry(kind: .gateUp, bulk: bulk)
            ))

            let downInfo = SafetensorsTensorInfo(
                name: "\(prefix).experts.down_proj",
                dtype: downTensor.info.dtype,
                shape: [expertCount * outputDimension, downPackedDimension],
                dataOffset: 0,
                byteCount: downTensor.info.byteCount
            )
            entries.append(STAFConversionEntry(
                name: downInfo.name,
                sourceName: downInfo.name,
                info: downInfo,
                shardIndex: downTensor.shardIndex,
                shardURL: downTensor.shardURL,
                schemeIdentifier: downScheme,
                semanticRole: .moeExpertDown,
                originalDType: mapOriginalDType(downTensor.info.dtype),
                packedMoE: STAFPackedMoEEntry(kind: .down, bulk: bulk)
            ))
        }
        return entries
    }

    private func determineScheme(
        name: String,
        info: SafetensorsTensorInfo,
        allTensors: [(name: String, sourceName: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)],
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

        // The SSM (DeltaNet) per-head RMS norm weight is read by the Metal
        // recurrence kernel via a hardcoded `device const float* normWeight`
        // signature. HF Qwen3.5 stores this tensor as float32 (matches), but
        // MLX bundles (incl. quantized variants) collapse it to bfloat16.
        // Reading bf16 bytes as f32 produces garbage scales and corrupts the
        // DeltaNet output. Force f32 storage here so the kernel contract holds
        // regardless of source dtype.
        if isSSMNormWeight(name: name) {
            return .fp32RowMajor
        }

        switch info.dtype {
        case .float16: return .fp16RowMajor
        case .bfloat16: return .bf16RowMajor
        case .float32: return .fp32RowMajor
        default: return .passthrough
        }
    }

    private func isSSMNormWeight(name: String) -> Bool {
        name.hasSuffix(".linear_attn.norm.weight")
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
        if name.contains("experts.gate_up_proj") { return .moeExpertGate }
        if name.contains("experts") && name.contains("gate") { return .moeExpertGate }
        if name.contains("experts") && name.contains("up") { return .moeExpertUp }
        if name.contains("experts") && name.contains("down") { return .moeExpertDown }
        if name.contains("router")
            || name.contains("gate.weight") && name.contains("moe")
            || name.contains("feed_forward.gate.weight") {
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
