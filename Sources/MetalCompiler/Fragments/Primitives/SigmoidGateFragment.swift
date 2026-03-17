/// Sigmoid-gated element-wise operation.
public struct SigmoidGateFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
    }

    public var isFusable: Bool { true }
    public func kernelName(context: KernelContext) -> String { "sigmoid_gate" }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }
}
