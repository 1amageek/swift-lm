/// Attributes for a standardize operation.
///
/// Applies per-element affine transformation: `(x - bias) * scale`.
/// Used by Gemma4 vision encoder's post-pooling standardization step.
public struct StandardizeAttributes: OperationAttributes, Codable, Equatable {

    /// Dimension of the input vector.
    public let dimension: Int

    public init(dimension: Int) {
        precondition(dimension > 0, "dimension must be positive")
        self.dimension = dimension
    }
}
