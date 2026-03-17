/// Depthwise temporal convolution with double gating (decode: state update).
public struct Conv1dFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let kernelSize: Int

    public init(dimension: Int, kernelSize: Int) {
        self.dimension = dimension
        self.kernelSize = kernelSize
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        if context.bufferPrecision == .float32 { return "conv1d_causal_seq_f32" }
        return context.weightFormat == .bfloat16 ? "conv_state_update_bf16" : "conv_state_update"
    }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: "conv_weight", role: .weight)] }
    public var cacheSlots: [MetalCacheSlot] { [MetalCacheSlot(name: "conv_cache", kind: .conv)] }
}
