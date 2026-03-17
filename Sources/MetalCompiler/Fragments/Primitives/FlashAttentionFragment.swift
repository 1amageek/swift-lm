/// Single-token attention against KV cache.
public struct FlashAttentionFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let ropeDimension: Int
    public let ropeBase: Float

    public init(headCount: Int, kvHeadCount: Int, headDimension: Int,
                ropeDimension: Int = 0, ropeBase: Float = 0) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.ropeDimension = ropeDimension
        self.ropeBase = ropeBase
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "flash_attn_decode_f32" : "flash_attn_decode"
    }
    public var dispatchDimension: MetalDispatchDimension {
        .perHead(headCount: headCount)
    }
    public var cacheSlots: [MetalCacheSlot] { [MetalCacheSlot(name: "kv_cache", kind: .kv)] }
}
