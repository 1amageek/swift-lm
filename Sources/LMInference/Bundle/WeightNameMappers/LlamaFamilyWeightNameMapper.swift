import LMCompiler
import SwiftLM

/// Weight name mapper for Llama-family models.
///
/// Handles: Llama, Qwen2, Mistral, Gemma, Phi, StarCoder2, Qwen3.5, Mixtral.
/// HF tensor names follow the standard convention:
/// - Attention: self_attn.{q_proj, k_proj, v_proj, o_proj}
/// - MLP: mlp.{gate_proj, down_proj, up_proj}
/// - Norms: input_layernorm, post_attention_layernorm
/// - Final norm: model.norm
/// - QK norm: q_norm, k_norm
public struct LlamaFamilyWeightNameMapper: WeightNameMapper {

    public init() {}

    public func manifest(for graph: ModelGraph) -> [SlotManifestEntry] {
        ModelGraphSlotEnumerator().enumerate(graph, naming: .llamaFamily)
    }
}
