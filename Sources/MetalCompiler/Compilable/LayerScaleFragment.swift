import LMIR

extension LayerScaleAttributes: MetalCompilable {
    @MetalKernelFragmentBuilder
    package func fragment(context: KernelContext) -> some MetalKernelFragment {
        ScalarMultiplyFragment(
            count: dimension,
            weightRole: "layer_scalar"
        )
    }
}
