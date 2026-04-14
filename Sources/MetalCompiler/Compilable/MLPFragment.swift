import LMIR

extension MLPAttributes: MetalCompilable {

    /// Fragment expansion for MLP: batched gate+up projections, activation, down projection.
    @MetalKernelFragmentBuilder
    package func fragment(context: KernelContext) -> some MetalKernelFragment {
        BatchedProjection(projections: [
            .init(field: "gate_proj", inputDimension: inputSize, outputDimension: intermediateSize),
            .init(field: "up_proj", inputDimension: inputSize, outputDimension: intermediateSize),
        ])
        ElementwiseFragment(count: intermediateSize, kind: elementwiseKind)
        LinearFragment(field: "down_proj", inputDimension: intermediateSize, outputDimension: outputSize)
    }

    private var elementwiseKind: ElementwiseFragment.ElementwiseKind {
        switch activation {
        case .custom(let kind) where kind == "gelu_pytorch_tanh" || kind == "gelu_new" || kind == "gelu_fast":
            return .geluGated
        case .gelu:
            return .geluGated
        case .silu, .swish, .relu:
            return .swiglu
        case .custom:
            return .swiglu
        }
    }
}
