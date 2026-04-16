import LMIR

/// Tensor naming convention for Gemma4-family text decoder.
///
/// Uses `model.language_model.*` prefix with Gemma-style sandwich norms,
/// per-layer input weights, layer scalar, and optional v_norm
/// (`v_layernorm` when `valueNorm == .rmsNormUnitOffset`).
public struct Gemma4FamilyNaming: WeightNamingConvention {

    public init() {}

    public func bindings(
        for attributes: any OperationAttributes,
        scope: WeightNamingScope,
        residualIndex: Int,
        normIndex: Int
    ) -> [ParameterBinding] {
        if let _ = attributes as? TokenEmbeddingAttributes {
            return [ParameterBinding(role: "embedding_table", tensorName: "model.language_model.embed_tokens.weight")]
        }

        if let attrs = attributes as? OutputHeadAttributes {
            if attrs.tiedToEmbedding {
                return [ParameterBinding(role: "weight", tensorName: "model.language_model.embed_tokens.weight")]
            }
            return [ParameterBinding(role: "weight", tensorName: "lm_head.weight")]
        }

        guard case .layer(let layerIndex) = scope else {
            if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
                return [ParameterBinding(role: "scale", tensorName: "model.language_model.norm.weight")]
            }
            return []
        }

        let prefix = "model.language_model.layers.\(layerIndex)"

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
            if attrs.valueNorm == .rmsNormUnitOffset {
                bindings.append(ParameterBinding(role: "v_layernorm", tensorName: "\(attnPrefix).v_norm.weight"))
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

        if let _ = attributes as? PerLayerInputAttributes {
            return [
                ParameterBinding(role: "per_layer_embedding_table",
                                 tensorName: "model.language_model.embed_tokens_per_layer.weight"),
                ParameterBinding(role: "per_layer_model_projection",
                                 tensorName: "model.language_model.per_layer_model_projection.weight"),
                ParameterBinding(role: "per_layer_projection_norm",
                                 tensorName: "model.language_model.per_layer_projection_norm.weight"),
                ParameterBinding(role: "per_layer_input_gate",
                                 tensorName: "\(prefix).per_layer_input_gate.weight"),
                ParameterBinding(role: "per_layer_projection",
                                 tensorName: "\(prefix).per_layer_projection.weight"),
                ParameterBinding(role: "post_per_layer_input_norm",
                                 tensorName: "\(prefix).post_per_layer_input_norm.weight"),
            ]
        }

        if let _ = attributes as? LayerScaleAttributes {
            return [
                ParameterBinding(role: "layer_scalar",
                                 tensorName: "\(prefix).layer_scalar")
            ]
        }

        return []
    }
}

public extension WeightNamingConvention where Self == Gemma4FamilyNaming {
    static var gemma4Family: Gemma4FamilyNaming { Gemma4FamilyNaming() }
}
