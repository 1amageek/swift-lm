/// Matrix-vector (decode) or matrix-matrix (prefill) multiply.
///
/// Represents a single linear projection (GEMV for decode, GEMM for prefill).
/// `isOutput` is set by the compiler to mark the last projection in a composite
/// as writing back to the hidden buffer (or logits for output head).
public struct LinearFragment: PrimitiveMetalKernelFragment {
    public let field: String
    public let inputDimension: Int
    public let outputDimension: Int
    /// Whether this projection writes to hidden (or logits). Set by the compiler.
    public var isOutput: Bool

    public init(field: String, inputDimension: Int, outputDimension: Int, isOutput: Bool = false) {
        self.field = field
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        self.isOutput = isOutput
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String { "gemv" }
    public var dispatchDimension: MetalDispatchDimension {
        .gemv(outputDimension: outputDimension, inputDimension: inputDimension)
    }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: field, role: .weight)] }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        guard role == field else { return 0 }
        return inputDimension * outputDimension * bytesPerScalar
    }
}

extension LinearFragment: ProjectionDescribing {
    public var projectionFields: [ProjectionFieldDescriptor] {
        [ProjectionFieldDescriptor(field: field, inputDimension: inputDimension, outputDimension: outputDimension)]
    }
    public var isOutputProjection: Bool { isOutput }
    public func withOutputProjectionEnabled() -> any PrimitiveMetalKernelFragment {
        var copy = self
        copy.isOutput = true
        return copy
    }
}
