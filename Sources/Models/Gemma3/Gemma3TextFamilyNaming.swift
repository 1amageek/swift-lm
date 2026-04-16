import LMIR

/// Tensor naming convention for Gemma3-family text models.
///
/// Uses Gemma-style sandwich norms (`input_layernorm`, `post_attention_layernorm`,
/// `pre_feedforward_layernorm`, `post_feedforward_layernorm`) and
/// `q_norm`/`k_norm` attention weights under `model.layers.{i}`.
public struct Gemma3TextFamilyNaming: WeightNamingConvention {

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
            let normName: String
            switch (residualIndex, normIndex) {
            case (0, 0): normName = "input_layernorm"
            case (0, 1): normName = "post_attention_layernorm"
            case (1, 0): normName = "pre_feedforward_layernorm"
            case (1, 1): normName = "post_feedforward_layernorm"
            default: return []
            }
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
                bindings.append(ParameterBinding(role: "q_layernorm", tensorName: "\(attnPrefix).q_norm.weight"))
                bindings.append(ParameterBinding(role: "k_layernorm", tensorName: "\(attnPrefix).k_norm.weight"))
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

        return []
    }
}

public extension WeightNamingConvention where Self == Gemma3TextFamilyNaming {
    static var gemma3TextFamily: Gemma3TextFamilyNaming { Gemma3TextFamilyNaming() }
}
