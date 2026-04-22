import LMIR
import Metal

/// Metal dispatch types used by the compiler.
///
/// Shared types: MetalDispatchDimension, MetalWeightSlot,
/// MetalCacheSlot, fused operation structs, SynchronizationKind,
/// BufferPrecision, WeightFormat.

// MARK: - Buffer Precision

/// Buffer precision for intermediate values (hidden, scratch, residual).
public enum BufferPrecision: Sendable {
    /// Float16 — used in decode (single token, no accumulation).
    case float16
    /// BFloat16 — used in decode for BF16-native models.
    case bfloat16
    /// Float32 — used in prefill (multi-token, prevents accumulation error).
    case float32

    public var metalType: String {
        switch self {
        case .float16: return "half"
        case .bfloat16: return "bfloat"
        case .float32: return "float"
        }
    }

    public var byteSize: Int {
        switch self {
        case .float16: return 2
        case .bfloat16: return 2
        case .float32: return 4
        }
    }
}

// MARK: - Weight Format

/// Weight data format is a `QuantizationFormat` instance.
///
/// Previously a closed enum; now a type alias over the protocol so that
/// new formats can be added without enum case proliferation. Dense formats
/// (fp16 / bf16 / fp32) conform to the same protocol as quantized formats.
///
/// Pattern matching is replaced by protocol property queries:
/// `format.isQuantized`, `format.bits`, `format.groupSize`, `format.schemeIdentifier`.
public typealias WeightFormat = any QuantizationFormat

/// Factory constants that mirror the legacy enum-case API used throughout the compiler.
///
/// Dense formats are provided as singletons; quantized factories resolve to the
/// registry so that unsupported (bits, groupSize) pairs fail loudly.
public enum WeightFormats {
    public static let float16: any QuantizationFormat = Float16Format()
    public static let bfloat16: any QuantizationFormat = BFloat16Format()
    public static let float32: any QuantizationFormat = Float32Format()

    public static func quantized2Bit(groupSize: Int) -> any QuantizationFormat {
        guard let format = QuantizationFormatRegistry.formatForMLXQuantization(bits: 2, groupSize: groupSize) else {
            fatalError("Unsupported Q2 group size \(groupSize)")
        }
        return format
    }

    public static func quantized3Bit(groupSize: Int) -> any QuantizationFormat {
        guard let format = QuantizationFormatRegistry.formatForMLXQuantization(bits: 3, groupSize: groupSize) else {
            fatalError("Unsupported Q3 group size \(groupSize)")
        }
        return format
    }

    public static func quantized4Bit(groupSize: Int) -> any QuantizationFormat {
        guard let format = QuantizationFormatRegistry.formatForMLXQuantization(bits: 4, groupSize: groupSize) else {
            fatalError("Unsupported Q4 group size \(groupSize)")
        }
        return format
    }

    public static func quantized5Bit(groupSize: Int) -> any QuantizationFormat {
        guard let format = QuantizationFormatRegistry.formatForMLXQuantization(bits: 5, groupSize: groupSize) else {
            fatalError("Unsupported Q5 group size \(groupSize)")
        }
        return format
    }

    public static func quantized6Bit(groupSize: Int) -> any QuantizationFormat {
        guard let format = QuantizationFormatRegistry.formatForMLXQuantization(bits: 6, groupSize: groupSize) else {
            fatalError("Unsupported Q6 group size \(groupSize)")
        }
        return format
    }

    public static func quantized8Bit(groupSize: Int) -> any QuantizationFormat {
        guard let format = QuantizationFormatRegistry.formatForMLXQuantization(bits: 8, groupSize: groupSize) else {
            fatalError("Unsupported Q8 group size \(groupSize)")
        }
        return format
    }
}

public extension QuantizationFormat {
    /// Compare formats by their STAF scheme identifier (single source of truth).
    func matches(_ other: any QuantizationFormat) -> Bool {
        schemeIdentifier == other.schemeIdentifier
    }

    /// Bit width if quantized; nil for dense formats.
    var quantizationBits: Int? { isQuantized ? bits : nil }

    /// Group size if quantized; nil for dense formats.
    var quantizationGroupSize: Int? { isQuantized ? groupSize : nil }

    /// Returns self if quantized, nil for dense formats. Preserved for call sites
    /// that used the old `WeightFormat.quantizationFormat` bridge.
    var quantizationFormat: (any QuantizationFormat)? { isQuantized ? self : nil }
}

// MARK: - Kernel Context

/// Context passed to fragment tree traversal for kernel name resolution.
///
/// Carries the buffer precision (F16 decode / F32 prefill) and weight format
/// (from STAF) so that fragments can resolve context-dependent kernel names
/// without hardcoding variants.
public struct KernelContext: Sendable {
    public let bufferPrecision: BufferPrecision
    public let weightFormat: WeightFormat

    public init(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) {
        self.bufferPrecision = bufferPrecision
        self.weightFormat = weightFormat
    }
}

// MARK: - Buffer Binding Context

/// Context provided by the compiler for fragment buffer binding resolution.
///
/// Fragments use this to declare their decode-path buffer layout
/// without knowing the concrete MTLBuffer allocation details.
public struct BufferBindingContext: @unchecked Sendable {
    public let bufferSet: MetalBufferSet
    public let slotDimension: Int
    public let elementSize: Int
    public let currentInputBuffer: MTLBuffer
    public let currentInputOffset: Int
    public let layerIndex: Int?
    public let kvCacheIndex: Int
    public let convLayerIndex: Int
    public let recurrentLayerIndex: Int
    /// Current projection index for scratch slot allocation.
    /// Projection outputs use scratch slot `projectionIndex + 1`.
    public let projectionIndex: Int
    public let resolveWeight: (String) -> (buffer: MTLBuffer, offset: Int)

    public init(bufferSet: MetalBufferSet, slotDimension: Int, elementSize: Int,
                currentInputBuffer: MTLBuffer, currentInputOffset: Int,
                layerIndex: Int?, kvCacheIndex: Int, convLayerIndex: Int, recurrentLayerIndex: Int,
                projectionIndex: Int = 0,
                resolveWeight: @escaping (String) -> (buffer: MTLBuffer, offset: Int)) {
        self.bufferSet = bufferSet
        self.slotDimension = slotDimension
        self.elementSize = elementSize
        self.currentInputBuffer = currentInputBuffer
        self.currentInputOffset = currentInputOffset
        self.layerIndex = layerIndex
        self.kvCacheIndex = kvCacheIndex
        self.convLayerIndex = convLayerIndex
        self.recurrentLayerIndex = recurrentLayerIndex
        self.projectionIndex = projectionIndex
        self.resolveWeight = resolveWeight
    }
}

/// Buffer bindings declared by a fragment for decode dispatch.
public struct FragmentBindings: @unchecked Sendable {
    // @unchecked: contains MTLBuffer (Metal protocol, not Sendable)
    public let buffers: [(index: Int, buffer: MTLBuffer, offset: Int)]
    public let bytes: [(index: Int, value: [UInt8])]
    /// Whether this fragment's output goes to hidden (true) or scratch (false).
    public let outputIsHidden: Bool
    /// Whether to reset projection index after this fragment.
    public let resetsProjectionIndex: Bool
    /// Whether this fragment consumes a KV cache layer slot.
    public let consumesKVCacheLayer: Bool
    /// Whether this fragment consumes a conv state layer slot.
    public let consumesConvLayer: Bool
    /// Whether this fragment consumes a recurrent state layer slot.
    public let consumesRecurrentLayer: Bool
    /// Buffer access pattern for barrier optimization.
    /// Indices into the `buffers` array that are written by this fragment.
    /// Buffers not listed here are treated as read-only.
    /// When nil, the barrier optimizer falls back to conservative (all read+write).
    public let writeBufferIndices: Set<Int>?

    /// Number of projection scratch slots consumed by this fragment.
    /// The routing planner advances projectionIndex by this amount.
    /// Default 0 for non-projection fragments.
    public let projectionSlotsConsumed: Int

    public init(buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
                bytes: [(index: Int, value: [UInt8])],
                outputIsHidden: Bool,
                resetsProjectionIndex: Bool = false,
                consumesKVCacheLayer: Bool = false,
                consumesConvLayer: Bool = false,
                consumesRecurrentLayer: Bool = false,
                writeBufferIndices: Set<Int>? = nil,
                projectionSlotsConsumed: Int = 0) {
        self.buffers = buffers
        self.bytes = bytes
        self.outputIsHidden = outputIsHidden
        self.resetsProjectionIndex = resetsProjectionIndex
        self.consumesKVCacheLayer = consumesKVCacheLayer
        self.consumesConvLayer = consumesConvLayer
        self.consumesRecurrentLayer = consumesRecurrentLayer
        self.writeBufferIndices = writeBufferIndices
        self.projectionSlotsConsumed = projectionSlotsConsumed
    }
}

// MARK: - Prefill Binding Context

/// Context provided by the compiler for fragment prefill step generation.
public struct PrefillBindingContext: @unchecked Sendable {
    // @unchecked: contains MTLBuffer (via PrefillBufferSet) and MTLComputePipelineState
    public let buffers: PrefillBufferSet
    public let slotDimension: Int
    public let scratchElementSize: Int
    public let maximumSequenceLength: Int
    public let currentInputBuffer: MTLBuffer
    public let currentInputOffset: Int
    public let layerIndex: Int?
    public let kvCacheIndex: Int
    public let convLayerIndex: Int
    public let recurrentLayerIndex: Int
    /// Current projection index for scratch slot allocation.
    public let projectionIndex: Int
    public let kernelContext: KernelContext
    public let resolveWeight: (String) -> (buffer: MTLBuffer, offset: Int)
    public let getPipeline: (String) throws -> MTLComputePipelineState

    public init(buffers: PrefillBufferSet, slotDimension: Int, scratchElementSize: Int,
                maximumSequenceLength: Int, currentInputBuffer: MTLBuffer, currentInputOffset: Int,
                layerIndex: Int?, kvCacheIndex: Int, convLayerIndex: Int, recurrentLayerIndex: Int,
                projectionIndex: Int = 0,
                kernelContext: KernelContext,
                resolveWeight: @escaping (String) -> (buffer: MTLBuffer, offset: Int),
                getPipeline: @escaping (String) throws -> MTLComputePipelineState) {
        self.buffers = buffers
        self.slotDimension = slotDimension
        self.scratchElementSize = scratchElementSize
        self.maximumSequenceLength = maximumSequenceLength
        self.currentInputBuffer = currentInputBuffer
        self.currentInputOffset = currentInputOffset
        self.layerIndex = layerIndex
        self.kvCacheIndex = kvCacheIndex
        self.convLayerIndex = convLayerIndex
        self.recurrentLayerIndex = recurrentLayerIndex
        self.projectionIndex = projectionIndex
        self.kernelContext = kernelContext
        self.resolveWeight = resolveWeight
        self.getPipeline = getPipeline
    }
}

/// Prefill steps declared by a fragment.
public struct FragmentPrefillSteps: @unchecked Sendable {
    // @unchecked: contains [MetalPrefillStep] which has MTLBuffer/MTLComputePipelineState
    public let steps: [MetalPrefillStep]
    public let outputIsHidden: Bool
    public let resetsProjectionIndex: Bool
    public let consumesKVCacheLayer: Bool
    public let consumesConvLayer: Bool
    public let consumesRecurrentLayer: Bool

    /// Number of projection scratch slots consumed by this fragment.
    /// The prefill routing planner advances projectionIndex by this amount.
    public let projectionSlotsConsumed: Int

    public init(steps: [MetalPrefillStep], outputIsHidden: Bool,
                resetsProjectionIndex: Bool = false,
                consumesKVCacheLayer: Bool = false,
                consumesConvLayer: Bool = false,
                consumesRecurrentLayer: Bool = false,
                projectionSlotsConsumed: Int = 0) {
        self.steps = steps
        self.outputIsHidden = outputIsHidden
        self.resetsProjectionIndex = resetsProjectionIndex
        self.consumesKVCacheLayer = consumesKVCacheLayer
        self.consumesConvLayer = consumesConvLayer
        self.consumesRecurrentLayer = consumesRecurrentLayer
        self.projectionSlotsConsumed = projectionSlotsConsumed
    }
}

// MARK: - Binding Helpers

/// Create a bytes binding for a UInt32 constant.
public func uint32Binding(_ index: Int, _ value: UInt32) -> (index: Int, value: [UInt8]) {
    withUnsafeBytes(of: value) { (index: index, value: Array($0)) }
}

/// Create a bytes binding for a Float constant.
public func floatBinding(_ index: Int, _ value: Float) -> (index: Int, value: [UInt8]) {
    withUnsafeBytes(of: value) { (index: index, value: Array($0)) }
}

// MARK: - Dispatch Dimension

/// GPU dispatch pattern for grid/threadgroup sizing.
public enum MetalDispatchDimension: Sendable, Equatable {
    /// Single threadgroup reduces across the dimension (RMSNorm, Argmax).
    case reduction(dimension: Int)
    /// One thread per element (SwiGLU, SigmoidGate, Copy, Add).
    case elementwise(count: Int)
    /// One threadgroup per head (FlashAttention, RoPE, QKNorm, SSM).
    case perHead(headCount: Int)
    /// Token → embedding gather.
    case gather(count: Int)
    /// GEMV/GEMM projection.
    case gemv(outputDimension: Int, inputDimension: Int)
    /// Multiple independent threadgroups, each reducing over its own partition
    /// (SSM recurrence: one threadgroup per key-group, disjoint conv channels
    /// and recurrent state slices). `threadsPerPartition` is clamped to the
    /// pipeline's maxTotalThreadsPerThreadgroup at dispatch time.
    case partitionedReduction(partitionCount: Int, threadsPerPartition: Int)
}

// MARK: - Batched Operations

/// Batched projection: multiple GEMV projections in a single dispatch.
/// All projections share the same input but have different weights and outputs.
public struct BatchedProjection: Sendable {
    public struct Entry: Sendable {
        public let field: String
        public let inputDimension: Int
        public let outputDimension: Int
        public init(field: String, inputDimension: Int, outputDimension: Int) {
            self.field = field
            self.inputDimension = inputDimension
            self.outputDimension = outputDimension
        }
    }
    public let projections: [Entry]

    public var totalOutputDimension: Int {
        projections.reduce(0) { $0 + $1.outputDimension }
    }

    public var inputDimension: Int {
        projections[0].inputDimension
    }

    public init(projections: [Entry]) {
        self.projections = projections
    }
}

/// Batched in-place fragments with the same dispatch dimension.
/// The compiler dispatches all instances in a single kernel, routing
/// threadgroups to the correct data/weight buffers.
public struct BatchedFragment: Sendable {
    public let fragments: [any PrimitiveMetalKernelFragment]
    public let dispatchDimension: MetalDispatchDimension

    public init(fragments: [any PrimitiveMetalKernelFragment], dispatchDimension: MetalDispatchDimension) {
        self.fragments = fragments
        self.dispatchDimension = dispatchDimension
    }
}

// MARK: - Weight / Cache Slots

public struct MetalWeightSlot: Sendable {
    public let field: String?
    public let role: MetalWeightRole

    public init(field: String? = nil, role: MetalWeightRole) {
        self.field = field
        self.role = role
    }
}

public struct MetalCacheSlot: Sendable {
    public let name: String
    public let kind: MetalCacheKind
    /// Temporal window size for conv cache (kernelSize), or 0 for non-conv caches.
    public let temporalSize: Int
    /// KV head count (for .kv caches).
    public let kvHeadCount: Int
    /// Head dimension (for .kv caches).
    public let headDimension: Int

    public init(name: String, kind: MetalCacheKind = .kv, temporalSize: Int = 0,
                kvHeadCount: Int = 0, headDimension: Int = 0) {
        self.name = name
        self.kind = kind
        self.temporalSize = temporalSize
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
    }
}

public enum MetalCacheKind: Sendable {
    case kv
    case conv
    case recurrent
}

public enum MetalWeightRole: Sendable {
    case weight
    case scale
    case embeddingTable
}

public enum SynchronizationKind: Sendable {
    case none
    case bufferBarrier
}
