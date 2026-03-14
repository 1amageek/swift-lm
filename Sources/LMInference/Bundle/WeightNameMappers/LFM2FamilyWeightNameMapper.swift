import LMCompiler
import SwiftLM

/// Weight name mapper for LFM2-family models.
///
/// Handles: LiquidAI LFM2, LFM2.5.
/// HF tensor names differ from Llama convention:
/// - Attention: self_attn.{q_proj, k_proj, v_proj, out_proj}
/// - QK norm: q_layernorm, k_layernorm
/// - MLP: feed_forward.{w1, w2, w3}
/// - Norms: operator_norm, ffn_norm
/// - Shared norm: model.embedding_norm (used for both post-embed and final norm)
public struct LFM2FamilyWeightNameMapper: WeightNameMapper {

    public init() {}

    public func manifest(for graph: ModelGraph) -> [SlotManifestEntry] {
        var entries = ModelGraphSlotEnumerator().enumerate(graph, naming: .lfm2Family)
        // LFM2 shares embedding_norm for both post-embed and final norm.
        // The enumerator produces "model.norm.weight" for the final norm,
        // but the actual tensor in safetensors is "model.embedding_norm.weight".
        entries = entries.map { entry in
            if entry.mlxWeightPath == "model.norm.weight" {
                return SlotManifestEntry(
                    slot: entry.slot,
                    mlxWeightPath: "model.embedding_norm.weight"
                )
            }
            return entry
        }
        return entries
    }
}
