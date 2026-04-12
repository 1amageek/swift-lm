// MARK: - Structure-only Operations

extension NormalizedModel {
    /// Produce the normalized (structurally closed) semantic IR for this component.
    ///
    /// Returns the `NormalizedModel` containing both the semantic graph
    /// and diagnostic metadata. The graph is well-formed but NOT
    /// canonicalized. For equivalence comparison, pass `result.graph`
    /// through `canonicalize(_:)`.
    public init(_ component: some ModelComponent) throws {
        self = try normalize(component)
    }
}

extension ModelGraph {
    /// Convenience: produce just the semantic graph (discarding metadata).
    public init(_ component: some ModelComponent) throws {
        self = try NormalizedModel(component).graph
    }
}
