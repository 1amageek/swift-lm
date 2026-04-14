import LMIR

extension TokenEmbeddingAttributes: MetalCompilable {
    package func fragment(context: KernelContext) -> GatherFragment {
        GatherFragment(vocabularySize: vocabSize, embeddingDimension: embeddingSize, embeddingScale: embeddingScale)
    }
}
