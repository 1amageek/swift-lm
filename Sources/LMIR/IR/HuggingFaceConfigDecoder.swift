import Foundation

/// Decodes Hugging Face config.json metadata into the backend-independent model contract.
public struct HuggingFaceConfigDecoder: Sendable {
    public init() {}

    public func decode(from url: URL) throws -> HuggingFaceModelConfiguration {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw HuggingFaceConfigError.configReadFailed(url, String(describing: error))
        }
        return try decode(from: data)
    }

    public func decode(from data: Data) throws -> HuggingFaceModelConfiguration {
        let rawJSON = try object(from: data)
        guard let modelType = rawJSON["model_type"] as? String else {
            throw HuggingFaceConfigError.missingField("model_type")
        }

        var json = rawJSON
        if let textConfig = rawJSON["text_config"] as? [String: Any],
           textConfig["hidden_size"] != nil {
            for (key, value) in textConfig {
                json[key] = value
            }
        }

        let hiddenSize = try requiredInt("hidden_size", in: json)
        let layerCount = try requiredInt("num_hidden_layers", in: json)
        let vocabSize = try requiredInt("vocab_size", in: json)

        let rawIntermediateSize = jsonInt("intermediate_size", in: json)
            ?? jsonInt("block_ff_dim", in: json)
            ?? hiddenSize * 4
        let autoAdjust = jsonBool("block_auto_adjust_ff_dim", in: json)
            ?? jsonBool("block_use_swiglu", in: json)
            ?? false
        let intermediateSize: Int
        if autoAdjust {
            var adjusted = rawIntermediateSize * 2 / 3
            if let multiplier = jsonDouble("block_ffn_dim_multiplier", in: json) {
                adjusted = Int(multiplier * Double(adjusted))
            }
            let multipleOf = jsonInt("block_multiple_of", in: json) ?? 256
            adjusted = multipleOf * ((adjusted + multipleOf - 1) / multipleOf)
            intermediateSize = adjusted
        } else {
            intermediateSize = rawIntermediateSize
        }

        let attentionHeads = jsonInt("num_attention_heads", in: json) ?? 32
        let kvHeads = jsonInt("num_key_value_heads", in: json) ?? attentionHeads
        let headDim = jsonInt("head_dim", in: json) ?? (hiddenSize / attentionHeads)
        let normEps = jsonDouble("rms_norm_eps", in: json)
            ?? jsonDouble("layer_norm_eps", in: json)
            ?? jsonDouble("norm_eps", in: json)
            ?? jsonDouble("block_norm_eps", in: json)
            ?? 1e-6

        let ropeParameters = json["rope_parameters"] as? [String: Any]
        let slidingAttentionRoPE = ropeParameters?["sliding_attention"] as? [String: Any]
        let fullAttentionRoPE = ropeParameters?["full_attention"] as? [String: Any]
        let localAttentionRopeTheta = jsonDouble("rope_local_base_freq", in: json)
            ?? jsonDouble("rope_theta", in: slidingAttentionRoPE)
        let ropeTheta = localAttentionRopeTheta
            ?? jsonDouble("rope_theta", in: json)
            ?? jsonDouble("rope_theta", in: ropeParameters)
            ?? 500_000.0

        let modelTypeLowercased = modelType.lowercased()
        let mropeAxes: MRoPEAxes?
        if let sections = ropeParameters?["mrope_section"] as? [Int], !sections.isEmpty {
            let interleaved = jsonBool("mrope_interleaved", in: ropeParameters) ?? false
            mropeAxes = MRoPEAxes(sections: sections, interleaved: interleaved)
        } else {
            mropeAxes = nil
        }

        let maxContextLength = jsonInt("max_position_embeddings", in: json) ?? 2048
        guard maxContextLength > 0 else {
            throw HuggingFaceConfigError.invalidValue(
                "max_position_embeddings must be positive"
            )
        }

        let modelConfig = ModelConfig(
            hiddenSize: hiddenSize,
            layerCount: layerCount,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            attentionBias: jsonBool("attention_bias", in: json) ?? false,
            mlpBias: jsonBool("mlp_bias", in: json) ?? false,
            normEps: Float(normEps),
            normKind: modelTypeLowercased == "cohere" ? .layerNorm : .rmsNorm,
            ropeTheta: Float(ropeTheta),
            ropeDimension: jsonInt("rope_dim", in: json) ?? headDim,
            ropeScaling: nil,
            tiedEmbeddings: jsonBool("tie_word_embeddings", in: json)
                ?? jsonBool("tie_embedding", in: json)
                ?? tiedEmbeddingDefault(for: modelTypeLowercased),
            expertCount: jsonInt("num_local_experts", in: json)
                ?? jsonInt("num_experts", in: json),
            expertsPerToken: jsonInt("num_experts_per_tok", in: json),
            moeIntermediateSize: jsonInt("moe_intermediate_size", in: json),
            moeNormalizeRoutingWeights: jsonBool("norm_topk_prob", in: json) ?? false,
            moeRoutedScalingFactor: Float(
                jsonDouble("routed_scaling_factor", in: json) ?? 1.0
            ),
            moeUseExpertBias: jsonBool("use_expert_bias", in: json) ?? false,
            qkNorm: jsonBool("qk_norm", in: json)
                ?? ["lfm2", "lfm2_moe"].contains(modelTypeLowercased),
            fullAttentionInterval: jsonInt("full_attention_interval", in: json),
            ssmNumHeads: jsonInt("ssm_num_heads", in: json)
                ?? jsonInt("linear_num_value_heads", in: json),
            ssmGroupCount: jsonInt("linear_num_key_heads", in: json),
            ssmKeyHeadDim: jsonInt("ssm_state_size", in: json)
                ?? jsonInt("linear_key_head_dim", in: json),
            ssmValueHeadDim: jsonInt("ssm_state_size", in: json)
                ?? jsonInt("linear_value_head_dim", in: json),
            convKernelSize: jsonInt("conv_kernel_size", in: json)
                ?? jsonInt("linear_conv_kernel_dim", in: json),
            convLCache: jsonInt("conv_L_cache", in: json),
            partialRotaryFactor: (
                jsonDouble("partial_rotary_factor", in: json)
                    ?? jsonDouble("partial_rotary_factor", in: ropeParameters)
            ).map(Float.init),
            slidingWindow: jsonInt("sliding_window", in: json),
            useBidirectionalAttention: jsonBool("use_bidirectional_attention", in: json)
                ?? false,
            queryPreAttentionScalar: jsonDouble("query_pre_attn_scalar", in: json).map(Float.init),
            localAttentionRopeTheta: localAttentionRopeTheta.map(Float.init),
            layerTypes: layerTypes(from: json, layerCount: layerCount),
            hiddenSizePerLayerInput: jsonInt("hidden_size_per_layer_input", in: json),
            vocabSizePerLayerInput: jsonInt("vocab_size_per_layer_input", in: json),
            globalHeadDim: jsonInt("global_head_dim", in: json),
            globalKVHeads: jsonInt("num_global_key_value_heads", in: json),
            numKVSharedLayers: jsonInt("num_kv_shared_layers", in: json),
            useDoubleWideMLP: jsonBool("use_double_wide_mlp", in: json) ?? false,
            attentionKEqualsV: jsonBool("attention_k_eq_v", in: json) ?? false,
            fullAttentionRopeTheta: (
                jsonDouble("rope_theta", in: fullAttentionRoPE)
                    ?? jsonDouble("rope_theta", in: json)
            ).map(Float.init),
            fullAttentionPartialRotaryFactor: jsonDouble(
                "partial_rotary_factor",
                in: fullAttentionRoPE
            ).map(Float.init),
            fullAttentionRoPEScaling: fullAttentionRoPEScaling(from: fullAttentionRoPE),
            finalLogitSoftcapping: jsonDouble("final_logit_softcapping", in: json).map(Float.init),
            numDenseLayers: jsonInt("num_dense_layers", in: json) ?? 0,
            mropeAxes: mropeAxes
        )

        return HuggingFaceModelConfiguration(
            modelType: modelTypeLowercased,
            modelConfig: modelConfig,
            maxContextLength: maxContextLength
        )
    }

    private func object(from data: Data) throws -> [String: Any] {
        do {
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw HuggingFaceConfigError.invalidJSON("root must be an object")
            }
            return object
        } catch let error as HuggingFaceConfigError {
            throw error
        } catch {
            throw HuggingFaceConfigError.invalidJSON(String(describing: error))
        }
    }

    private func requiredInt(_ key: String, in json: [String: Any]) throws -> Int {
        guard let value = jsonInt(key, in: json), value > 0 else {
            throw HuggingFaceConfigError.missingField(key)
        }
        return value
    }

    private func jsonInt(_ key: String, in json: [String: Any]?) -> Int? {
        guard let json else { return nil }
        if let value = json[key] as? Int { return value }
        if let value = json[key] as? NSNumber { return value.intValue }
        return nil
    }

    private func jsonDouble(_ key: String, in json: [String: Any]?) -> Double? {
        guard let json else { return nil }
        if let value = json[key] as? Double { return value }
        if let value = json[key] as? Int { return Double(value) }
        if let value = json[key] as? NSNumber { return value.doubleValue }
        return nil
    }

    private func jsonBool(_ key: String, in json: [String: Any]?) -> Bool? {
        guard let json else { return nil }
        return json[key] as? Bool
    }

    private func layerTypes(from json: [String: Any], layerCount: Int) -> [String]? {
        if let types = json["layer_types"] as? [String] {
            return types
        }
        guard let attentionIndices = json["full_attn_idxs"] as? [Int] else {
            return nil
        }
        let attentionSet = Set(attentionIndices)
        return (0..<layerCount).map { attentionSet.contains($0) ? "full_attention" : "conv" }
    }

    private func tiedEmbeddingDefault(for modelType: String) -> Bool {
        switch modelType {
        case "gemma", "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text",
             "lfm2", "lfm2_moe":
            return true
        default:
            return false
        }
    }

    private func fullAttentionRoPEScaling(
        from json: [String: Any]?
    ) -> RoPEScaling? {
        guard let ropeType = json?["rope_type"] as? String,
              ropeType != "default" else {
            return nil
        }
        return RoPEScaling(kind: .custom(ropeType), factor: 1.0)
    }
}
