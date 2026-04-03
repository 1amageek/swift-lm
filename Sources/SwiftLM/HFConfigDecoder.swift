import Foundation
import LMArchitecture

struct HFConfigDecoder {
    func modelType(from data: Data) throws -> String {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = json["model_type"] as? String else {
            throw ModelBundleLoaderError.invalidConfig("Missing model_type in config.json")
        }
        return modelType
    }

    func decode(from data: Data) throws -> ModelConfig {
        guard let rawJson = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig("config.json is not a JSON object")
        }

        let json: [String: Any]
        if let textConfig = rawJson["text_config"] as? [String: Any], textConfig["hidden_size"] != nil {
            var merged = rawJson
            for (key, value) in textConfig { merged[key] = value }
            json = merged
        } else {
            json = rawJson
        }

        guard let hiddenSize = json["hidden_size"] as? Int else {
            throw ModelBundleLoaderError.invalidConfig("Missing hidden_size")
        }
        guard let layerCount = json["num_hidden_layers"] as? Int else {
            throw ModelBundleLoaderError.invalidConfig("Missing num_hidden_layers")
        }

        let rawIntermediateSize = json["intermediate_size"] as? Int
            ?? json["block_ff_dim"] as? Int
            ?? hiddenSize * 4
        let autoAdjust = json["block_auto_adjust_ff_dim"] as? Bool
            ?? json["block_use_swiglu"] as? Bool
            ?? false
        let intermediateSize: Int
        if autoAdjust {
            var adjusted = rawIntermediateSize * 2 / 3
            if let multiplier = json["block_ffn_dim_multiplier"] as? Double {
                adjusted = Int(multiplier * Double(adjusted))
            }
            let multipleOf = json["block_multiple_of"] as? Int ?? 256
            adjusted = multipleOf * ((adjusted + multipleOf - 1) / multipleOf)
            intermediateSize = adjusted
        } else {
            intermediateSize = rawIntermediateSize
        }

        guard let vocabSize = json["vocab_size"] as? Int else {
            throw ModelBundleLoaderError.invalidConfig("Missing vocab_size")
        }
        let attentionHeads = json["num_attention_heads"] as? Int ?? 32
        let kvHeads = json["num_key_value_heads"] as? Int ?? attentionHeads
        let headDim = json["head_dim"] as? Int ?? (hiddenSize / attentionHeads)
        let normEps = (json["rms_norm_eps"] as? Double
            ?? json["layer_norm_eps"] as? Double
            ?? json["norm_eps"] as? Double
            ?? json["block_norm_eps"] as? Double
            ?? 1e-6)

        let ropeParams = json["rope_parameters"] as? [String: Any]
        let ropeTheta = json["rope_theta"] as? Double
            ?? (ropeParams?["rope_theta"] as? Double)
            ?? 500000.0
        let tiedEmbeddings = json["tie_word_embeddings"] as? Bool
            ?? json["tie_embedding"] as? Bool
            ?? false

        let mropeAxes: MRoPEAxes?
        if let sections = ropeParams?["mrope_section"] as? [Int], !sections.isEmpty {
            let interleaved = ropeParams?["mrope_interleaved"] as? Bool ?? false
            mropeAxes = MRoPEAxes(sections: sections, interleaved: interleaved)
        } else {
            mropeAxes = nil
        }

        return ModelConfig(
            hiddenSize: hiddenSize,
            layerCount: layerCount,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            attentionBias: json["attention_bias"] as? Bool ?? false,
            mlpBias: json["mlp_bias"] as? Bool ?? false,
            normEps: Float(normEps),
            normKind: json["model_type"] as? String == "cohere" ? .layerNorm : .rmsNorm,
            ropeTheta: Float(ropeTheta),
            ropeDimension: json["rope_dim"] as? Int ?? headDim,
            ropeScaling: nil,
            tiedEmbeddings: tiedEmbeddings,
            expertCount: json["num_local_experts"] as? Int ?? json["num_experts"] as? Int,
            expertsPerToken: json["num_experts_per_tok"] as? Int,
            moeIntermediateSize: json["moe_intermediate_size"] as? Int,
            qkNorm: json["qk_norm"] as? Bool
                ?? (["lfm2", "lfm2_moe"].contains(json["model_type"] as? String ?? "")),
            fullAttentionInterval: json["full_attention_interval"] as? Int,
            ssmNumHeads: json["ssm_num_heads"] as? Int
                ?? json["linear_num_value_heads"] as? Int,
            ssmGroupCount: json["linear_num_key_heads"] as? Int,
            ssmKeyHeadDim: json["ssm_state_size"] as? Int
                ?? json["linear_key_head_dim"] as? Int,
            ssmValueHeadDim: json["ssm_state_size"] as? Int
                ?? json["linear_value_head_dim"] as? Int,
            convKernelSize: json["conv_kernel_size"] as? Int
                ?? json["linear_conv_kernel_dim"] as? Int,
            convLCache: json["conv_L_cache"] as? Int,
            partialRotaryFactor: (json["partial_rotary_factor"] as? Double
                ?? ropeParams?["partial_rotary_factor"] as? Double).map { Float($0) },
            slidingWindow: json["sliding_window"] as? Int,
            layerTypes: {
                if let types = json["layer_types"] as? [String] { return types }
                if let attnIdxs = json["full_attn_idxs"] as? [Int] {
                    let attnSet = Set(attnIdxs)
                    return (0..<layerCount).map { attnSet.contains($0) ? "full_attention" : "conv" }
                }
                return nil
            }(),
            numDenseLayers: json["num_dense_layers"] as? Int ?? 0,
            mropeAxes: mropeAxes
        )
    }
}
