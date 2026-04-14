import LMIR

extension ShortConvAttributes: MetalCompilable {
    @MetalKernelFragmentBuilder
    package func fragment(context: KernelContext) -> some MetalKernelFragment {
        LinearFragment(field: "in_proj", inputDimension: hiddenSize, outputDimension: hiddenSize * 3)
        Conv1dFragment(dimension: hiddenSize, kernelSize: kernelSize)
        LinearFragment(field: "out_proj", inputDimension: hiddenSize, outputDimension: hiddenSize)
    }
}
