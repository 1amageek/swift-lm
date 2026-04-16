/// A naming convention that maps IR operations to concrete tensor names
/// in safetensors/STAF files.
///
/// Each model family (Llama, Gemma, Qwen, LFM2, etc.) has a distinct
/// tensor-name layout. A `WeightNamingConvention` produces the
/// `ParameterBinding`s for a single primitive operation given its
/// structural context (layer index, residual position, sandwich-norm
/// position).
///
/// Conformances live with the model declaration (`Sources/Models/<Family>/`),
/// not in MetalCompiler. This keeps family-specific knowledge out of the
/// backend and allows new models to be added without touching the compiler.
public protocol WeightNamingConvention: Sendable {
    /// Produce parameter bindings for a single primitive operation.
    ///
    /// - Parameters:
    ///   - attributes: The operation's attributes (type-switched to
    ///     identify Attention/MLP/Norm/etc.).
    ///   - scope: Whether the operation is at root or inside a layer
    ///     (with its index).
    ///   - residualIndex: 0 for the operator/attention residual block,
    ///     1 for the FFN residual block within a layer.
    ///   - normIndex: Sandwich norm position within the residual body
    ///     (0 for pre-op, 1 for post-op).
    /// - Returns: Bindings that map operation parameter roles to tensor
    ///   names. Empty if the operation has no weights.
    func bindings(
        for attributes: any OperationAttributes,
        scope: WeightNamingScope,
        residualIndex: Int,
        normIndex: Int
    ) -> [ParameterBinding]
}

/// Structural position of an operation during weight resolution.
public enum WeightNamingScope: Sendable, Hashable {
    /// Operation sits at the model root (e.g. token embedding, final norm,
    /// output head).
    case root
    /// Operation sits inside a decoder/encoder layer with the given index.
    case layer(index: Int)
}
