/// Per-head RMS normalization for Q or K projections.
public struct QKNormFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let headDimension: Int
    public let epsilon: Float
    public let weightRole: String  // "q_layernorm" or "k_layernorm"

    public init(headCount: Int, headDimension: Int, epsilon: Float, weightRole: String) {
        self.headCount = headCount
        self.headDimension = headDimension
        self.epsilon = epsilon
        self.weightRole = weightRole
    }

    public var isFusable: Bool { true }
    public var isInPlace: Bool { true }
    public var normEpsilon: Float? { epsilon }
    public func kernelName(context: KernelContext) -> String {
        if context.bufferPrecision == .float32 { return "qk_rms_norm_seq_f32" }
        return context.weightFormat == .bfloat16 ? "qk_rms_norm_bf16" : "qk_rms_norm"
    }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: weightRole, role: .weight)] }
}
