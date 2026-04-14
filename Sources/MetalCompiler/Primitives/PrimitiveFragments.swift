/// Primitive Metal kernel fragments — leaf nodes of the fragment tree.
///
/// Each primitive fragment is a single dispatch unit. The compiler:
/// 1. Reads fragment parameters (dimension, epsilon, etc.)
/// 2. Determines buffer precision (F16 decode / F32 prefill) and weight format (from STAF)
/// 3. Calls MetalSourceGenerator to produce MSL on-demand
///
/// No hardcoded kernel names. The compiler derives names from fragment type + context.
///
/// Individual fragment types are in the Primitives/ subdirectory (1 file per type).

// MARK: - Primitive Protocol

/// Leaf fragment that the compiler translates into a single Metal dispatch.
///
/// The compiler uses these properties for generic graph optimization:
/// - `dispatchDimension`: kernel scaffold and batching strategy
/// - `isFusable`: whether this fragment participates in compiler optimizations
/// - `isInPlace`: whether this fragment modifies its primary buffer in-place
/// - `kernelName(context:)`: context-aware kernel name resolution
/// - `kernelBody()`: composable MSL computation body for kernel composition
///
/// Only fragments that explicitly opt in participate in decode-time in-place
/// batching. This keeps the optimizer generic while avoiding unsound routing
/// for fragments whose scratch layout cannot be reconstructed from dispatch
/// dimension alone.
public protocol PrimitiveMetalKernelFragment: MetalKernelFragment where Fragment == Never {
    /// GPU dispatch pattern (determines grid/threadgroup sizing).
    var dispatchDimension: MetalDispatchDimension { get }
    /// Weight tensors this fragment reads from STAF.
    var weightSlots: [MetalWeightSlot] { get }
    /// Persistent cache slots (KV cache, conv state).
    var cacheSlots: [MetalCacheSlot] { get }

    /// Optional override for the KV cache layer index this fragment reads.
    ///
    /// Used by Gemma 4 shared-KV attention, which reads another layer's cache
    /// without allocating or advancing a new KV slot.
    var kvCacheIndexOverride: Int? { get }

    /// Whether this fragment modifies its primary data buffer in-place.
    ///
    /// In-place fragments within the same composite fragment that have different
    /// data sources (different preceding projection outputs) are independent.
    /// The optimizer uses this to determine batchability.
    var isInPlace: Bool { get }

    /// Per-head dimension for batching eligibility.
    /// Non-nil for fragments that operate per-head (QKNorm, RoPE, etc.)
    /// and declare how many elements each head processes.
    var perHeadDimension: Int? { get }

    /// Epsilon value for normalization-type fragments.
    /// nil for fragments that don't perform normalization.
    var normEpsilon: Float? { get }

    /// Additive bias applied to normalization weights.
    ///
    /// Non-zero values indicate a family-specific affine convention such as
    /// Qwen 3.5's `1 + weight`, which current fused norm kernels do not model.
    var normWeightBias: Float? { get }

    /// Whether the decode optimizer may merge consecutive in-place instances
    /// of this fragment family into a batched dispatch.
    ///
    /// Opt-in is required because the batched route builder must preserve each
    /// fragment's concrete buffer bindings, not just its dispatch dimension.
    var supportsInPlaceBatching: Bool { get }

    /// Kernel name for this fragment, resolved using the kernel context.
    ///
    /// Returns the correct kernel name for both decode (F16) and prefill (F32)
    /// based on `context.bufferPrecision` and `context.weightFormat`.
    func kernelName(context: KernelContext) -> String

    /// Declare buffer bindings for decode dispatch.
    ///
    /// The compiler provides a context with buffer set, slot dimensions,
    /// and weight resolution. The fragment returns its concrete bindings
    /// and routing state updates.
    func decodeBindings(context: BufferBindingContext) -> FragmentBindings

    /// Build prefill steps for this fragment.
    ///
    /// The compiler provides a context with prefill buffers, slot dimensions,
    /// pipeline cache, and kernel context. The fragment returns its prefill
    /// steps (batch, perPosition, or lastToken mode).
    func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps

    /// Fusion contract declaring this fragment's data flow interface.
    ///
    /// The compiler uses this — and ONLY this — to determine fusion eligibility.
    /// No concrete fragment types are inspected.
    ///
    /// Returns nil for non-fusable (opaque) fragments (FlashAttention, GEMV, etc.).
    var fusionContract: FusionContract? { get }

    /// Generate the composable MSL computation body for this fragment.
    ///
    /// Fusable fragments return a body using standardized variable names
    /// that match port names declared in `fusionContract`. The compiler
    /// wraps the body in a scaffold (kernel signature, buffer declarations,
    /// row pointer computation, sequence iteration).
    ///
    /// Body conventions by parallelism:
    /// - `.perRow`: body contains its own loops (`for (uint i = tid; ...)`)
    ///   and threadgroup barriers. Uses `tid`, `threadgroupSize`, `dimension`,
    ///   `shared` (threadgroup float[32]).
    /// - `.perElement`: body is a single-element expression using `i` as index.
    ///   No loop — scaffold wraps with iteration context.
    /// - `.perHead`: body contains its own loops using `tid` within `headDimension`.
    ///
    /// Output writes are always `float`. The scaffold adds precision cast
    /// (`half()`, `float()`) only at the final output of a fused group.
    ///
    /// Returns nil for non-fusable (opaque) fragments.
    /// The compiler calls `kernelSource()` instead.
    func kernelBody(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String?

    /// Generate the complete MSL kernel for non-fusable (opaque) fragments.
    ///
    /// Called only when `kernelBody()` returns nil. The compiler uses the
    /// returned source as-is, without composition or wrapping.
    func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String

    /// Scalar constant values declared by this fragment's fusion contract.
    ///
    /// The keys must match the `name` fields in `fusionContract.scalarConstants`.
    /// Used by SynthesizedFragment to bind scalar constant values in the
    /// fused kernel's buffer bindings.
    var scalarConstantValues: [String: ScalarConstantValue] { get }

    /// Required fallback buffer size for a given weight role.
    ///
    /// Each fragment computes its own requirement based on internal dimensions.
    /// The compiler calls this instead of inspecting concrete fragment types
    /// for buffer sizing.
    func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int
}

/// A typed scalar constant value for kernel buffer binding.
public enum ScalarConstantValue: Sendable {
    case float(Float)
    case uint32(UInt32)

    /// Convert to raw bytes for buffer binding.
    public var bytes: [UInt8] {
        switch self {
        case .float(let v): return withUnsafeBytes(of: v) { Array($0) }
        case .uint32(let v): return withUnsafeBytes(of: v) { Array($0) }
        }
    }
}

extension PrimitiveMetalKernelFragment {
    public func fragment(context: KernelContext) -> Never { fatalError() }
    public var weightSlots: [MetalWeightSlot] { [] }
    public var cacheSlots: [MetalCacheSlot] { [] }
    public var kvCacheIndexOverride: Int? { nil }
    public var isInPlace: Bool { false }
    public var perHeadDimension: Int? { nil }
    public var normEpsilon: Float? { nil }
    public var normWeightBias: Float? { nil }
    public var supportsInPlaceBatching: Bool { false }
    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        fatalError("Fragment \(type(of: self)) must implement decodeBindings(context:)")
    }
    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        fatalError("Fragment \(type(of: self)) must implement prefillSteps(context:)")
    }
    public var fusionContract: FusionContract? { nil }
    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? { nil }
    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        fatalError("Fragment \(type(of: self)) must implement either kernelBody() or kernelSource()")
    }
    public var scalarConstantValues: [String: ScalarConstantValue] { [:] }
    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int { 0 }
}
