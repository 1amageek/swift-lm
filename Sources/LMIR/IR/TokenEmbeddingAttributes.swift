/// Attributes for a token embedding node.
///
/// Maps discrete token IDs to dense vectors.
public struct TokenEmbeddingAttributes: OperationAttributes, Codable, Equatable {

    /// Size of the vocabulary (number of distinct tokens).
    public let vocabSize: Int

    /// Dimensionality of each embedding vector.
    public let embeddingSize: Int

    /// Optional dtype hint for the embedding table.
    public let dtypeHint: DTypeHint?

    /// Optional scale factor applied after embedding lookup.
    ///
    /// Gemma models multiply embeddings by `sqrt(hidden_size)` to counterbalance
    /// the tied embedding/output head weight sharing.
    public let embeddingScale: Float?

    public init(
        vocabSize: Int,
        embeddingSize: Int,
        dtypeHint: DTypeHint? = nil,
        embeddingScale: Float? = nil
    ) {
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.dtypeHint = dtypeHint
        self.embeddingScale = embeddingScale
    }
}
