/// Short convolution component for hybrid conv-attention architectures.
///
/// Represents a depthwise short convolution with input/output projections,
/// used in LFM2-family models as an alternative to attention in some layers.
///
/// ```swift
/// ShortConv(hiddenSize: 2048, kernelSize: 3)
/// ```
public struct ShortConv: ModelComponent {

    public typealias Attributes = ShortConvAttributes

    public let hiddenSize: Int
    public let kernelSize: Int

    public init(hiddenSize: Int, kernelSize: Int) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(kernelSize > 0, "kernelSize must be positive")
        self.hiddenSize = hiddenSize
        self.kernelSize = kernelSize
    }
}

extension ShortConv {

    public var attributes: ShortConvAttributes {
        ShortConvAttributes(
            hiddenSize: hiddenSize,
            kernelSize: kernelSize
        )
    }
}
