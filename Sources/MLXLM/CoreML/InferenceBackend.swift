import Foundation

/// Inference backend selection for LLM execution.
///
/// CoreML is the default for standard transformer architectures.
/// MLX is used as fallback for unsupported architectures or when CoreML is unavailable.
public enum InferenceBackend: Sendable {
    /// Automatically select the best backend based on architecture and availability.
    ///
    /// - Transformer, ParallelAttentionMLP, MoE → CoreML (1.6x faster)
    /// - HybridDeltaNet, HybridConvAttention → MLX (recurrent state unsupported in CoreML)
    /// - CoreML compilation failure → MLX fallback
    case auto

    /// Force CoreML execution. Fails if CoreML compilation is not possible.
    case coreml

    /// Force MLX Metal execution. Always available.
    case mlx
}
