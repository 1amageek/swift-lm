import LMIR

extension MoEAttributes: MetalCompilable {
    @MetalKernelFragmentBuilder
    package func fragment(context: KernelContext) -> some MetalKernelFragment {
        SparseMoEFragment(
            expertCount: expertCount,
            expertsPerToken: expertsPerToken,
            gateKind: gateKind,
            inputDimension: expertMLP.inputSize,
            outputDimension: expertMLP.outputSize,
            intermediateDimension: expertMLP.intermediateSize,
            normalizeRoutingWeights: normalizeRoutingWeights,
            routedScalingFactor: routedScalingFactor,
            useExpertBias: useExpertBias
        )
    }
}
