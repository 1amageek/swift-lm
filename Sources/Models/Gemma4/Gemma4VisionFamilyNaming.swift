import LMIR

/// Tensor naming convention for the Gemma4 vision encoder.
///
/// Uses `model.vision_tower.*` prefix with sandwich norms and `.linear.weight`
/// suffix (Gemma4 vision wraps each Linear inside a `linear.*` module).
public struct Gemma4VisionFamilyNaming: WeightNamingConvention {

    public init() {}

    public func bindings(
        for attributes: any OperationAttributes,
        scope: WeightNamingScope,
        residualIndex: Int,
        normIndex: Int
    ) -> [ParameterBinding] {
        if let _ = attributes as? PatchEmbeddingAttributes {
            return [
                ParameterBinding(role: "weight", tensorName: "model.vision_tower.patch_embedder.input_proj.weight"),
            ]
        }

        if let _ = attributes as? LinearAttributes {
            // Root-level linear is the vision-to-text projection.
            if case .root = scope {
                return [
                    ParameterBinding(role: "weight", tensorName: "model.embed_vision.embedding_projection.weight"),
                ]
            }
            return []
        }

        if let _ = attributes as? PoolingAttributes {
            return []
        }

        if let _ = attributes as? PositionEmbeddingAttributes {
            return [
                ParameterBinding(role: "position_embedding_table",
                                 tensorName: "model.vision_tower.patch_embedder.position_embedding_table"),
            ]
        }

        if let _ = attributes as? StandardizeAttributes {
            return [
                ParameterBinding(role: "std_bias", tensorName: "model.vision_tower.std_bias"),
                ParameterBinding(role: "std_scale", tensorName: "model.vision_tower.std_scale"),
            ]
        }

        guard case .layer(let layerIndex) = scope else {
            // Root-level norm: withScale=false means no learnable weight.
            if let rmsAttrs = attributes as? RMSNormAttributes, rmsAttrs.withScale {
                return [ParameterBinding(role: "scale", tensorName: "model.vision_tower.norm.weight")]
            }
            if attributes is LayerNormAttributes {
                return [ParameterBinding(role: "scale", tensorName: "model.vision_tower.norm.weight")]
            }
            return []
        }

        let prefix = "model.vision_tower.encoder.layers.\(layerIndex)"

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
                ParameterBinding(role: "q_proj", tensorName: "\(attnPrefix).q_proj.linear.weight"),
                ParameterBinding(role: "k_proj", tensorName: "\(attnPrefix).k_proj.linear.weight"),
                ParameterBinding(role: "o_proj", tensorName: "\(attnPrefix).o_proj.linear.weight"),
            ]
            if attrs.valueProjectionSource == .dedicatedProjection {
                bindings.append(ParameterBinding(role: "v_proj", tensorName: "\(attnPrefix).v_proj.linear.weight"))
            }
            if let qkNorm = attrs.qkNorm, qkNorm != .none {
                bindings.append(ParameterBinding(role: "q_layernorm", tensorName: "\(attnPrefix).q_norm.weight"))
                bindings.append(ParameterBinding(role: "k_layernorm", tensorName: "\(attnPrefix).k_norm.weight"))
            }
            return bindings
        }

        if let _ = attributes as? MLPAttributes {
            let mlpPrefix = "\(prefix).mlp"
            return [
                ParameterBinding(role: "gate_proj", tensorName: "\(mlpPrefix).gate_proj.linear.weight"),
                ParameterBinding(role: "up_proj", tensorName: "\(mlpPrefix).up_proj.linear.weight"),
                ParameterBinding(role: "down_proj", tensorName: "\(mlpPrefix).down_proj.linear.weight"),
            ]
        }

        return []
    }
}

public extension WeightNamingConvention where Self == Gemma4VisionFamilyNaming {
    static var gemma4VisionFamily: Gemma4VisionFamilyNaming { Gemma4VisionFamilyNaming() }
}
