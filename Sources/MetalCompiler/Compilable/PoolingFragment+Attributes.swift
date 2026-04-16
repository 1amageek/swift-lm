import LMIR

extension PoolingAttributes: MetalCompilable {
    package func fragment(context: KernelContext) -> PoolingFragment {
        PoolingFragment(
            kernelSize: kernelSize,
            hiddenSize: hiddenSize,
            rescale: rescale
        )
    }
}
