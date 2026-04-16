/// Attributes for a spatial average pooling node.
///
/// Reduces spatial resolution by averaging non-overlapping kernel-sized regions.
/// Used after vision encoder layers to downsample patch embeddings.
public struct PoolingAttributes: OperationAttributes, Codable, Equatable {

    /// Spatial kernel size for the pooling window.
    public let kernelSize: Int

    /// Hidden dimension of each spatial position.
    public let hiddenSize: Int

    /// Optional rescale factor applied after averaging (e.g. sqrt(hiddenSize)).
    public let rescale: Float?

    public init(kernelSize: Int, hiddenSize: Int, rescale: Float? = nil) {
        self.kernelSize = kernelSize
        self.hiddenSize = hiddenSize
        self.rescale = rescale
    }
}
