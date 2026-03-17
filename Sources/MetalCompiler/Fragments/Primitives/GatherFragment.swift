/// Token ID → embedding vector lookup.
public struct GatherFragment: PrimitiveMetalKernelFragment {
    public let vocabularySize: Int
    public let embeddingDimension: Int

    public init(vocabularySize: Int, embeddingDimension: Int) {
        self.vocabularySize = vocabularySize
        self.embeddingDimension = embeddingDimension
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        let bf16 = context.weightFormat == .bfloat16
        if context.bufferPrecision == .float32 {
            return bf16 ? "embedding_lookup_seq_bf16_f32" : "embedding_lookup_seq_f32"
        }
        return bf16 ? "embedding_lookup_bf16" : "embedding_lookup"
    }
    public var dispatchDimension: MetalDispatchDimension { .gather(count: embeddingDimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: nil, role: .weight)] }
}
