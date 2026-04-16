/// 2D separable position embedding component.
///
/// Adds learnable position embeddings to the hidden state using a
/// separable table: `hidden[pos] += table_x[col(pos)] + table_y[row(pos)]`.
///
/// Grid width is image-specific. The vision encoder is compiled per
/// unique grid configuration.
///
/// ```swift
/// PositionEmbedding(hiddenSize: 1152, tableSize: 48, gridWidth: 16)
/// ```
public struct PositionEmbedding: ModelComponent {

    public typealias Attributes = PositionEmbeddingAttributes

    public let hiddenSize: Int
    public let tableSize: Int
    public let gridWidth: Int

    public init(hiddenSize: Int, tableSize: Int, gridWidth: Int) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(tableSize > 0, "tableSize must be positive")
        precondition(gridWidth > 0, "gridWidth must be positive")
        self.hiddenSize = hiddenSize
        self.tableSize = tableSize
        self.gridWidth = gridWidth
    }
}

extension PositionEmbedding {

    public var attributes: PositionEmbeddingAttributes {
        PositionEmbeddingAttributes(hiddenSize: hiddenSize, tableSize: tableSize, gridWidth: gridWidth)
    }
}
