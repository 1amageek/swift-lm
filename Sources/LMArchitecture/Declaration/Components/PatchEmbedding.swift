/// Patch embedding component.
///
/// Projects flattened image patches to dense embedding vectors.
/// Vision encoder counterpart to `TokenEmbedding`.
///
/// ```swift
/// PatchEmbedding(patchPixelDimension: 768, hiddenSize: 1152)
/// ```
public struct PatchEmbedding: ModelComponent {

    public typealias Attributes = PatchEmbeddingAttributes

    public let patchPixelDimension: Int
    public let hiddenSize: Int

    public init(patchPixelDimension: Int, hiddenSize: Int) {
        precondition(patchPixelDimension > 0, "patchPixelDimension must be positive")
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        self.patchPixelDimension = patchPixelDimension
        self.hiddenSize = hiddenSize
    }
}

extension PatchEmbedding {

    public var attributes: PatchEmbeddingAttributes {
        PatchEmbeddingAttributes(
            patchPixelDimension: patchPixelDimension,
            hiddenSize: hiddenSize
        )
    }

    public var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(0), resultArity: .exact(1))
    }
}
