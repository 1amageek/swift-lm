import Metal

/// Elementwise: one thread per element, trivially parallel.
/// Used by: SwiGLU, GEGLU, SigmoidGate.
public struct ElementwiseFragment: PrimitiveMetalKernelFragment {
    public let count: Int
    public let kind: ElementwiseKind

    public enum ElementwiseKind: Sendable, Equatable {
        /// gate * sigmoid(gate) * up — SiLU-gated (Llama, LFM2)
        case swiglu
        /// gelu_tanh(gate) * up — GELU-gated (Gemma4)
        case geluGated
        case sigmoidGate
    }

    public init(count: Int, kind: ElementwiseKind = .swiglu) {
        self.count = count
        self.kind = kind
    }

    /// Whether this elementwise operation uses a gated activation pattern
    /// (two input buffers: gate and up).
    public var isGatedActivation: Bool {
        switch kind {
        case .swiglu, .geluGated: return true
        case .sigmoidGate: return false
        }
    }

    public var isFusable: Bool { true }
    public func kernelName(context: KernelContext) -> String {
        switch kind {
        case .swiglu:
            return context.bufferPrecision == .float32 ? "swiglu_seq_f32" : "swiglu"
        case .geluGated:
            return context.bufferPrecision == .float32 ? "geglu_seq_f32" : "geglu"
        case .sigmoidGate:
            return "sigmoid_gate"
        }
    }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: count) }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateGatedActivation(
            name: name, bufferPrecision: bufferPrecision,
            activation: gatedActivation, isSequence: bufferPrecision == .float32)
    }

    /// The activation function used by this elementwise operation.
    public var gatedActivation: MetalSourceGenerator.GatedActivation {
        switch kind {
        case .swiglu, .sigmoidGate: return .silu
        case .geluGated: return .geluTanh
        }
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 1 * slotBytes),
                (1, context.bufferSet.scratch, 2 * slotBytes),
                (2, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(count)),
            ],
            outputIsHidden: false,
            resetsProjectionIndex: true
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (count + tgSize - 1) / tgSize
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, 1 * scratchSlotSize),
                    (1, context.buffers.scratch, 2 * scratchSlotSize),
                    (2, context.buffers.scratch, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(count)),
                    uint32Binding(4, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: false,
            resetsProjectionIndex: true
        )
    }
}
