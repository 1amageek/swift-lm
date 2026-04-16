/// Per-element affine standardization component.
///
/// Applies `(x - bias) * scale` using learnable per-element weights.
///
/// ```swift
/// Standardize(dimension: 1152)
/// ```
public struct Standardize: ModelComponent {

    public typealias Attributes = StandardizeAttributes

    public let dimension: Int

    public init(dimension: Int) {
        precondition(dimension > 0, "dimension must be positive")
        self.dimension = dimension
    }
}

extension Standardize {

    public var attributes: StandardizeAttributes {
        StandardizeAttributes(dimension: dimension)
    }
}
