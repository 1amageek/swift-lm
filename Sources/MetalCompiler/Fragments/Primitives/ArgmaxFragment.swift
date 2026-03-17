/// Argmax over vocabulary: logits → token ID.
public struct ArgmaxFragment: PrimitiveMetalKernelFragment {
    public let vocabularySize: Int

    public init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "argmax_f32" : "argmax"
    }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: vocabularySize) }
}
