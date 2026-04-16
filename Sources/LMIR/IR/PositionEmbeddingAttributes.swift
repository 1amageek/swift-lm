/// Attributes for a 2D separable position embedding operation.
///
/// Adds position embeddings from a separable 2D table: for each grid
/// position (row, col), the embedding is `table_y[row] + table_x[col]`.
/// Table shape: `[2, tableSize, hiddenSize]` where axis 0 holds the X
/// (column) table and axis 1 holds the Y (row) table.
///
/// Grid dimensions (`gridWidth`) are image-specific. The vision encoder
/// is compiled per unique grid configuration — compilation is fast
/// (~0.2s) and cacheable by grid dimensions.
///
/// Used by Gemma4 vision encoder's patch position embeddings.
public struct PositionEmbeddingAttributes: OperationAttributes, Codable, Equatable {

    /// Hidden dimension of the position embedding vectors.
    public let hiddenSize: Int

    /// Maximum grid positions per axis in the table.
    public let tableSize: Int

    /// Grid width (columns) for the current image.
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
