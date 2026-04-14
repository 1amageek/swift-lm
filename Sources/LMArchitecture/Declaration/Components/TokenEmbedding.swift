/// Token embedding component.
///
/// Maps discrete token IDs to dense embedding vectors.
///
/// ```swift
/// TokenEmbedding(vocabSize: 32000, embeddingSize: 4096)
/// ```
public struct TokenEmbedding: ModelComponent {

    public typealias Attributes = TokenEmbeddingAttributes

    public let vocabSize: Int
    public let embeddingSize: Int
    public let dtypeHint: DTypeHint?
    public let embeddingScale: Float?

    public init(
        vocabSize: Int,
        embeddingSize: Int,
        dtypeHint: DTypeHint? = nil,
        embeddingScale: Float? = nil
    ) {
        precondition(vocabSize > 0, "vocabSize must be positive")
        precondition(embeddingSize > 0, "embeddingSize must be positive")
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.dtypeHint = dtypeHint
        self.embeddingScale = embeddingScale
    }
}

extension TokenEmbedding {

    public var attributes: TokenEmbeddingAttributes {
        TokenEmbeddingAttributes(
            vocabSize: vocabSize,
            embeddingSize: embeddingSize,
            dtypeHint: dtypeHint,
            embeddingScale: embeddingScale
        )
    }

    public var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(0), resultArity: .exact(1))
    }
}
