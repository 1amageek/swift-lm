import LMIR

/// Tensor naming convention for the LFM2 family.
///
/// Differs from Llama in:
/// - Attention output is `out_proj` (not `o_proj`)
/// - QK norm is `q_layernorm`/`k_layernorm` (not `q_norm`/`k_norm`)
/// - Per-layer norms are `operator_norm`/`ffn_norm`
/// - Root norm is `model.embedding_norm`
/// - MLP is `feed_forward.w{1,2,3}` (w1=gate, w3=up, w2=down)
/// - Uses `ShortConv` at conv layers
public struct LFM2FamilyNaming: WeightNamingConvention {

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
                return [ParameterBinding(role: "scale", tensorName: "model.embedding_norm.weight")]
            }
            return []
        }

        let prefix = "model.layers.\(layerIndex)"

        if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
            let normName = residualIndex == 0 ? "operator_norm" : "ffn_norm"
            return [ParameterBinding(role: "scale", tensorName: "\(prefix).\(normName).weight")]
        }

        if let attrs = attributes as? AttentionAttributes {
            let attnPrefix = "\(prefix).self_attn"
            var bindings = [
                ParameterBinding(role: "q_proj", tensorName: "\(attnPrefix).q_proj.weight"),
                ParameterBinding(role: "k_proj", tensorName: "\(attnPrefix).k_proj.weight"),
                ParameterBinding(role: "o_proj", tensorName: "\(attnPrefix).out_proj.weight"),
            ]
            bindings.append(contentsOf: WeightNamingHelpers.valueProjection(attributes: attrs, attentionPrefix: attnPrefix))
            if let qkNorm = attrs.qkNorm, qkNorm != .none {
                bindings.append(ParameterBinding(role: "q_layernorm", tensorName: "\(attnPrefix).q_layernorm.weight"))
                bindings.append(ParameterBinding(role: "k_layernorm", tensorName: "\(attnPrefix).k_layernorm.weight"))
            }
            return bindings
        }

        if let _ = attributes as? MLPAttributes {
            let ffPrefix = "\(prefix).feed_forward"
            return [
                ParameterBinding(role: "gate_proj", tensorName: "\(ffPrefix).w1.weight"),
                ParameterBinding(role: "up_proj", tensorName: "\(ffPrefix).w3.weight"),
                ParameterBinding(role: "down_proj", tensorName: "\(ffPrefix).w2.weight"),
            ]
        }

        if let attrs = attributes as? MoEAttributes {
            let ffPrefix = "\(prefix).feed_forward"
            var bindings = [
                ParameterBinding(role: "router", tensorName: "\(ffPrefix).gate.weight"),
                ParameterBinding(role: "expert_bias", tensorName: "\(ffPrefix).expert_bias"),
            ]
            for i in 0..<attrs.expertCount {
                bindings.append(contentsOf: [
                    ParameterBinding(role: "expert_\(i)_gate_proj", tensorName: "\(ffPrefix).experts.\(i).w1.weight"),
                    ParameterBinding(role: "expert_\(i)_up_proj", tensorName: "\(ffPrefix).experts.\(i).w3.weight"),
                    ParameterBinding(role: "expert_\(i)_down_proj", tensorName: "\(ffPrefix).experts.\(i).w2.weight"),
                ])
            }
            return bindings
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

public extension WeightNamingConvention where Self == LFM2FamilyNaming {
    static var lfm2Family: LFM2FamilyNaming { LFM2FamilyNaming() }
}
