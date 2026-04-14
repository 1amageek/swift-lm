import LMIR

extension OutputHeadAttributes: MetalCompilable {
    @MetalKernelFragmentBuilder
    package func fragment(context: KernelContext) -> some MetalKernelFragment {
        LinearFragment(field: "weight", inputDimension: inputSize, outputDimension: vocabSize)
        ArgmaxFragment(vocabularySize: vocabSize)
    }
}
