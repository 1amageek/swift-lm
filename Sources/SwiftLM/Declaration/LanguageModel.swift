// MARK: - Structure-only Operations

extension ModelComponent {

    /// Produce the normalized (structurally closed) semantic IR for this component.
    ///
    /// Returns the `NormalizedModel` containing both the semantic graph
    /// and diagnostic metadata. The graph is well-formed but NOT
    /// canonicalized. For equivalence comparison, pass `result.graph`
    /// through `canonicalize(_:)`.
    public func makeNormalizedModel() throws -> NormalizedModel {
        try normalize(self)
    }

    /// Convenience: produce just the semantic graph (discarding metadata).
    public func makeModelGraph() throws -> ModelGraph {
        try normalize(self).graph
    }
}

// MARK: - Weight Modifier

extension ModelComponent {

    /// Attach a weight declaration to this model.
    ///
    /// Returns a `WeightedModel` that bundles the architecture and weight source.
    /// The model itself is unchanged — weights are an external annotation.
    ///
    /// ```swift
    /// let model = Qwen35(config: .qwen35_0_8B)
    /// let weighted = model.weights(.gguf(location: "model.gguf"))
    /// ```
    public func weights(_ declaration: WeightsDeclaration) -> WeightedModel<Self> {
        WeightedModel(model: self, weightsDeclaration: declaration)
    }

    /// Attach a composed weight declaration using a builder.
    ///
    /// ```swift
    /// let weighted = model.weights {
    ///     WeightsDeclaration.gguf(location: "base.gguf")
    ///     WeightsDeclaration.safetensors(directory: "adapter/", indexFile: nil)
    /// }
    /// ```
    public func weights(
        @WeightsBuilder _ builder: () -> WeightsDeclaration
    ) -> WeightedModel<Self> {
        WeightedModel(model: self, weightsDeclaration: builder())
    }
}

/// A model bundled with a weight declaration.
///
/// `WeightedModel` is produced by the `.weights(_:)` modifier on `ModelComponent`.
/// It carries both the structural graph and the weight source, ready for
/// resolution and compilation.
///
/// `WeightedModel` is NOT a `ModelComponent`. This is intentional — it represents
/// a different concept: a structure-plus-weights bundle, not a pure structure.
///
/// ```swift
/// let weighted = Qwen35(config: .qwen35_0_8B)
///     .weights(.gguf(location: "model.gguf"))
///
/// let graph = try weighted.makeModelGraph()
/// let weightsDecl = weighted.weightsDeclaration
/// ```
public struct WeightedModel<M: ModelComponent>: Sendable {

    /// The underlying model (structure only).
    public let model: M

    /// The weight declaration attached to this model.
    public let weightsDeclaration: WeightsDeclaration

    public init(model: M, weightsDeclaration: WeightsDeclaration) {
        self.model = model
        self.weightsDeclaration = weightsDeclaration
    }

    /// Produce the normalized semantic IR for the model.
    public func makeNormalizedModel() throws -> NormalizedModel {
        try model.makeNormalizedModel()
    }

    /// Produce just the semantic graph.
    public func makeModelGraph() throws -> ModelGraph {
        try model.makeModelGraph()
    }

    /// Replace the weight declaration with a different one.
    public func weights(_ declaration: WeightsDeclaration) -> WeightedModel<M> {
        WeightedModel(model: model, weightsDeclaration: declaration)
    }

    /// Replace the weight declaration using a builder.
    ///
    /// ```swift
    /// let updated = weighted.weights {
    ///     WeightsDeclaration.gguf(location: "new-base.gguf")
    ///     WeightsDeclaration.safetensors(directory: "adapter/", indexFile: nil)
    /// }
    /// ```
    public func weights(
        @WeightsBuilder _ builder: () -> WeightsDeclaration
    ) -> WeightedModel<M> {
        WeightedModel(model: model, weightsDeclaration: builder())
    }
}
