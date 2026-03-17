/// Rotary position embedding (in-place on Q and K).
public struct RoPEFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let ropeDimension: Int
    public let base: Float

    public init(headCount: Int, kvHeadCount: Int, headDimension: Int,
                ropeDimension: Int, base: Float) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.ropeDimension = ropeDimension
        self.base = base
    }

    public var isFusable: Bool { false }
    public var isInPlace: Bool { true }
    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "rope_seq_f32" : "rope"
    }
    public var dispatchDimension: MetalDispatchDimension {
        .perHead(headCount: max(headCount, kvHeadCount))
    }
}
