/// Attributes for a patch embedding node.
///
/// Projects flattened image patches to dense embedding vectors via a linear
/// transformation. This is the vision encoder counterpart to
/// `TokenEmbeddingAttributes` for text models.
public struct PatchEmbeddingAttributes: OperationAttributes, Codable, Equatable {

    /// Dimensionality of each flattened patch (e.g. patchSize² × channels).
    public let patchPixelDimension: Int

    /// Output embedding dimensionality.
    public let hiddenSize: Int

    public init(patchPixelDimension: Int, hiddenSize: Int) {
        self.patchPixelDimension = patchPixelDimension
        self.hiddenSize = hiddenSize
    }
}
