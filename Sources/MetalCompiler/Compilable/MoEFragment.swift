import LMIR

extension MoEAttributes: MetalCompilable {
    @MetalKernelFragmentBuilder
    package func fragment(context: KernelContext) -> some MetalKernelFragment {
        LinearFragment(field: "router", inputDimension: expertMLP.inputSize, outputDimension: expertCount)
    }
}
