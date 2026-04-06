import LMIR

extension MLPAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public func fragment(context: KernelContext) -> some MetalKernelFragment {
        LinearFragment(field: "gate_proj", inputDimension: inputSize, outputDimension: intermediateSize)
        LinearFragment(field: "up_proj", inputDimension: inputSize, outputDimension: intermediateSize)
        ElementwiseFragment(count: intermediateSize, kind: elementwiseKind)
        LinearFragment(field: "down_proj", inputDimension: intermediateSize, outputDimension: outputSize)
    }
    public var isFusable: Bool { false }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment(context: context)) }

    /// Select the elementwise kernel kind based on the MLP activation function.
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
