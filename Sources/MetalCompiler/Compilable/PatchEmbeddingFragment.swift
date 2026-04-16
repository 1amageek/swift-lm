import LMIR

extension PatchEmbeddingAttributes: MetalCompilable {
    package func fragment(context: KernelContext) -> LinearFragment {
        LinearFragment(
            field: "weight",
            inputDimension: patchPixelDimension,
            outputDimension: hiddenSize
        )
    }
}
