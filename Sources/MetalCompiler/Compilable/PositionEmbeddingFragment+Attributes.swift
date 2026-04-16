import LMIR

extension PositionEmbeddingAttributes: MetalCompilable {

    /// Fragment expansion for 2D separable position embedding.
    package func fragment(context: KernelContext) -> PositionEmbeddingFragment {
        PositionEmbeddingFragment(
            hiddenSize: hiddenSize,
            tableSize: tableSize,
            gridWidth: gridWidth
        )
    }
}
