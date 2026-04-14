import LMIR

extension RMSNormAttributes: MetalCompilable {

    /// Fragment expansion for RMSNorm.
    package func fragment(context: KernelContext) -> Reduction {
        Reduction(dimension: dimension, epsilon: epsilon, weightBias: weightBias)
    }
}

extension LayerNormAttributes: MetalCompilable {

    /// Fragment expansion for LayerNorm.
    package func fragment(context: KernelContext) -> Reduction {
        Reduction(dimension: dimension, epsilon: epsilon)
    }
}
