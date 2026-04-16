import LMIR

/// Tensor naming convention for generic Llama-family models.
///
/// Covers Llama, Mistral, Qwen2, Phi, StarCoder2, and Mixtral — any model
/// matching the standard `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj`
/// layout under `model.layers.{i}`.
public struct LlamaFamilyNaming: WeightNamingConvention {

    public init() {}

    public func bindings(
        for attributes: any OperationAttributes,
        scope: WeightNamingScope,
        residualIndex: Int,
        normIndex: Int
    ) -> [ParameterBinding] {
        if let _ = attributes as? TokenEmbeddingAttributes {
            return [ParameterBinding(role: "embedding_table", tensorName: "model.embed_tokens.weight")]
        }

        if let attrs = attributes as? OutputHeadAttributes {
            if attrs.tiedToEmbedding {
                return [ParameterBinding(role: "weight", tensorName: "model.embed_tokens.weight")]
            }
            return [ParameterBinding(role: "weight", tensorName: "lm_head.weight")]
        }

        guard case .layer(let layerIndex) = scope else {
            if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
                return [ParameterBinding(role: "scale", tensorName: "model.norm.weight")]
            }
            return []
        }

        let prefix = "model.layers.\(layerIndex)"

        if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
            let normName = residualIndex == 0 ? "input_layernorm" : "post_attention_layernorm"
            return [ParameterBinding(role: "scale", tensorName: "\(prefix).\(normName).weight")]
        }

        if let attrs = attributes as? AttentionAttributes {
            let attnPrefix = "\(prefix).self_attn"
            var bindings = [
                ParameterBinding(role: "q_proj", tensorName: "\(attnPrefix).q_proj.weight"),
                ParameterBinding(role: "k_proj", tensorName: "\(attnPrefix).k_proj.weight"),
                ParameterBinding(role: "o_proj", tensorName: "\(attnPrefix).o_proj.weight"),
            ]
            bindings.append(contentsOf: WeightNamingHelpers.valueProjection(attributes: attrs, attentionPrefix: attnPrefix))
            if let qkNorm = attrs.qkNorm, qkNorm != .none {
                bindings.append(ParameterBinding(role: "q_layernorm", tensorName: "\(attnPrefix).q_layernorm.weight"))
                bindings.append(ParameterBinding(role: "k_layernorm", tensorName: "\(attnPrefix).k_layernorm.weight"))
            }
            return bindings
        }

        if let _ = attributes as? MLPAttributes {
            let mlpPrefix = "\(prefix).mlp"
            return [
                ParameterBinding(role: "gate_proj", tensorName: "\(mlpPrefix).gate_proj.weight"),
                ParameterBinding(role: "up_proj", tensorName: "\(mlpPrefix).up_proj.weight"),
                ParameterBinding(role: "down_proj", tensorName: "\(mlpPrefix).down_proj.weight"),
            ]
        }

        if let _ = attributes as? MoEAttributes {
            let moePrefix = "\(prefix).block_sparse_moe"
            return [
                ParameterBinding(role: "router", tensorName: "\(moePrefix).gate.weight"),
            ]
        }

        if let _ = attributes as? StateSpaceAttributes {
            let ssPrefix = "\(prefix).linear_attn"
            return [
                ParameterBinding(role: "in_proj_qkv", tensorName: "\(ssPrefix).in_proj_qkv.weight"),
                ParameterBinding(role: "in_proj_z", tensorName: "\(ssPrefix).in_proj_z.weight"),
                ParameterBinding(role: "in_proj_b", tensorName: "\(ssPrefix).in_proj_b.weight"),
                ParameterBinding(role: "in_proj_a", tensorName: "\(ssPrefix).in_proj_a.weight"),
                ParameterBinding(role: "out_proj", tensorName: "\(ssPrefix).out_proj.weight"),
                ParameterBinding(role: "scale", tensorName: "\(ssPrefix).norm.weight"),
                ParameterBinding(role: "conv_weight", tensorName: "\(ssPrefix).conv1d.weight"),
                ParameterBinding(role: "dt_bias", tensorName: "\(ssPrefix).dt_bias"),
                ParameterBinding(role: "A_log", tensorName: "\(ssPrefix).A_log"),
            ]
        }

        if let _ = attributes as? ShortConvAttributes {
            let convPrefix = "\(prefix).conv"
            return [
                ParameterBinding(role: "in_proj", tensorName: "\(convPrefix).in_proj.weight"),
                ParameterBinding(role: "conv_weight", tensorName: "\(convPrefix).conv.weight"),
                ParameterBinding(role: "out_proj", tensorName: "\(convPrefix).out_proj.weight"),
            ]
        }

        return []
    }
}

public extension WeightNamingConvention where Self == LlamaFamilyNaming {
    /// Generic Llama-family naming (also used as the default fallback).
    static var llamaFamily: LlamaFamilyNaming { LlamaFamilyNaming() }
}

/// Shared helpers used by multiple `WeightNamingConvention` conformances.
enum WeightNamingHelpers {
    /// Value projection binding for Attention with dedicated V projection.
    /// Returns empty if V is folded into QKV or sourced from another op.
    static func valueProjection(
        attributes: AttentionAttributes,
        attentionPrefix: String,
        tensorSuffix: String = "v_proj.weight"
    ) -> [ParameterBinding] {
        guard attributes.valueProjectionSource == .dedicatedProjection else {
            return []
        }
        return [ParameterBinding(role: "v_proj", tensorName: "\(attentionPrefix).\(tensorSuffix)")]
    }
}
