/// Spatial average pooling component.
///
/// Reduces spatial resolution by averaging non-overlapping kernel-sized regions.
///
/// ```swift
/// Pooling(kernelSize: 4, hiddenSize: 1152, rescale: sqrtf(1152))
/// ```
public struct Pooling: ModelComponent {

    public typealias Attributes = PoolingAttributes

    public let kernelSize: Int
    public let hiddenSize: Int
    public let rescale: Float?

    public init(kernelSize: Int, hiddenSize: Int, rescale: Float? = nil) {
        precondition(kernelSize > 0, "kernelSize must be positive")
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        self.kernelSize = kernelSize
        self.hiddenSize = hiddenSize
        self.rescale = rescale
    }
}

extension Pooling {

    public var attributes: PoolingAttributes {
        PoolingAttributes(
            kernelSize: kernelSize,
            hiddenSize: hiddenSize,
            rescale: rescale
        )
    }
}
