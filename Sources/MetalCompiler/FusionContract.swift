// MARK: - Kernel Parallelism

/// Thread organization pattern for a fragment's computation.
///
/// Determines the scaffold template the compiler uses to wrap `kernelBody()`.
///
/// Convention per parallelism:
/// - `.perRow`: body contains its own loops (`for (uint i = tid; ...)`) and
///   threadgroup barriers. Scaffold provides row-offset pointers, `tid`,
///   `threadgroupSize`, `dimension`, and `threadgroup float shared[32]`.
/// - `.perElement`: body is a single-element expression using `i` as the
///   element index. No loop. Scaffold wraps with iteration context.
/// - `.perHead`: body contains its own loops using `tid` within `headDimension`.
///   Scaffold provides head-offset pointer and shared memory.
public enum KernelParallelism: Sendable, Equatable {
    /// One threadgroup per row. Threads cooperate to process `dimension` elements.
    case perRow(dimension: Int)

    /// One thread per element. Trivially parallel, no inter-thread cooperation.
    case perElement(count: Int)

    /// One threadgroup per head. Threads cooperate within `headDimension`.
    case perHead(headCount: Int, headDimension: Int)

    /// The number of elements processed per work unit (row, element, or head).
    public var dimension: Int {
        switch self {
        case .perRow(let dimension): return dimension
        case .perElement(let count): return count
        case .perHead(_, let headDimension): return headDimension
        }
    }
}

// MARK: - Buffer Intent

/// Declares which physical buffer a `.buffer` role port binds to.
///
/// KernelScaffold generates buffer parameters in `contract.ports` order.
/// When the compiler builds buffer bindings for a SynthesizedFragment,
/// it uses `bufferIntent` to decide which physical buffer (hidden, scratch,
/// residual) to bind to each port index.
///
/// Only meaningful for `.buffer` role ports. `.weight` role ports always
/// bind to STAF weight storage regardless of intent.
public enum BufferIntent: Sendable {
    /// Bind according to routing state (hidden when lastOutputIsHidden,
    /// scratch otherwise). This is the default for data flow ports.
    case dataFlow

    /// Always bind to the residual buffer, regardless of routing state.
    case residual
}

// MARK: - Fusion Port

/// A named data port in a fragment's data flow interface.
///
/// Port names correspond to variable names in `kernelBody()` MSL code.
/// The compiler uses port names to connect fragments during fusion
/// (renaming producer output → consumer input).
public struct FusionPort: Sendable {
    /// Variable name used in `kernelBody()` MSL code.
    public let name: String

    /// Data flow direction.
    public let direction: PortDirection

    /// What kind of data source this port binds to.
    public let role: PortRole

    /// How the consumer reads elements from this port.
    /// Determines intermediate storage strategy during fusion.
    public let accessPattern: AccessPattern

    /// Which physical buffer this port binds to.
    /// Only meaningful for `.buffer` role ports.
    public let bufferIntent: BufferIntent

    public init(
        name: String,
        direction: PortDirection,
        role: PortRole,
        accessPattern: AccessPattern = .singlePass,
        bufferIntent: BufferIntent = .dataFlow
    ) {
        self.name = name
        self.direction = direction
        self.role = role
        self.accessPattern = accessPattern
        self.bufferIntent = bufferIntent
    }

    // MARK: Direction

    public enum PortDirection: Sendable {
        /// Read-only input.
        case input
        /// Write-only output.
        case output
    }

    // MARK: Role

    public enum PortRole: Sendable {
        /// Device buffer (hidden, scratch, residual).
        /// Scaffold computes row/element pointer based on parallelism.
        case buffer

        /// STAF weight tensor. Compiler resolves format from parameterBindings.
        /// `field` matches the weight role in parameterBindings for STAF resolution.
        case weight(field: String)
    }

    // MARK: Access Pattern

    /// How a consumer fragment reads elements from an input port.
    ///
    /// Determines intermediate storage when fusing producer → consumer:
    /// - `.singlePass`: register variable (zero cost, merged loop iteration)
    /// - `.multiPass`: threadgroup memory (producer writes → barrier → consumer reads)
    public enum AccessPattern: Sendable {
        /// Each element read once, in producer's output order.
        /// Intermediate can be a register variable within the same loop iteration.
        case singlePass

        /// Elements read multiple times or in different loop passes.
        /// Intermediate must be stored in threadgroup memory.
        case multiPass
    }
}

// MARK: - Scalar Constant

/// A scalar constant parameter emitted in the kernel signature.
///
/// The scaffold emits `constant <metalType>& <name> [[buffer(N)]]` for each
/// scalar constant. The body references the constant by `name`.
///
/// Buffer indices are assigned after ports and dimension, before sequenceLength.
/// This preserves binding compatibility with existing fragment bindings.
public struct ScalarConstant: Sendable {
    /// Variable name used in `kernelBody()` MSL code.
    public let name: String

    /// Metal type string (e.g., "float", "uint", "int").
    public let metalType: String

    public init(name: String, metalType: String) {
        self.name = name
        self.metalType = metalType
    }
}

// MARK: - Fusion Contract

/// Declarative specification of a fragment's fusion interface.
///
/// The compiler uses this — and ONLY this — to determine fusion eligibility
/// between adjacent fragments. No concrete fragment types are inspected.
///
/// A fragment with a non-nil `fusionContract` participates in automatic fusion.
/// A fragment with nil `fusionContract` is treated as an opaque barrier
/// that cannot be fused with neighbors.
public struct FusionContract: Sendable {
    /// Named data ports — the fragment's external data interface.
    /// Port names correspond to variables in `kernelBody()`.
    public let ports: [FusionPort]

    /// Scalar constant parameters referenced by `kernelBody()`.
    /// Emitted as `constant <type>& <name> [[buffer(N)]]` in the kernel signature.
    /// Buffer indices: after ports + dimension, before sequenceLength.
    public let scalarConstants: [ScalarConstant]

    /// Thread organization pattern.
    public let parallelism: KernelParallelism

    /// Threadgroup shared memory required by this fragment (bytes).
    /// Does not include intermediate storage — the compiler adds that separately.
    public let threadgroupMemoryBytes: Int

    /// Whether the fragment uses SIMD reduction (`simd_sum`, threadgroup barriers).
    public let requiresSIMDReduction: Bool

    public init(
        ports: [FusionPort],
        scalarConstants: [ScalarConstant] = [],
        parallelism: KernelParallelism,
        threadgroupMemoryBytes: Int = 0,
        requiresSIMDReduction: Bool = false
    ) {
        self.ports = ports
        self.scalarConstants = scalarConstants
        self.parallelism = parallelism
        self.threadgroupMemoryBytes = threadgroupMemoryBytes
        self.requiresSIMDReduction = requiresSIMDReduction
    }

    /// All input ports (direction == .input).
    public var inputPorts: [FusionPort] {
        ports.filter { $0.direction == .input }
    }

    /// All output ports (direction == .output).
    public var outputPorts: [FusionPort] {
        ports.filter { $0.direction == .output }
    }

    /// The primary buffer input port (first input with role == .buffer).
    public var primaryInput: FusionPort? {
        ports.first { $0.direction == .input && isBufferRole($0.role) }
    }

    /// The primary output port (first output).
    public var primaryOutput: FusionPort? {
        ports.first { $0.direction == .output }
    }
}

// MARK: - Intermediate Storage

/// Storage strategy for intermediate values between fused fragments.
public enum IntermediateStorage: Sendable, Equatable {
    /// Value stays in a register variable within the same loop iteration.
    /// Zero cost. Only valid when consumer access pattern is `.singlePass`.
    case register

    /// Value stored in threadgroup memory with a barrier between writes and reads.
    /// Cost: `dimension * sizeof(float)` bytes of threadgroup memory + 1 TG barrier.
    case threadgroupMemory(dimension: Int)
}

// MARK: - Parallelism Compatibility

extension KernelParallelism {
    /// Whether this parallelism can be fused with another.
    ///
    /// Compatible pairs:
    /// - `.perRow(D)` ↔ `.perRow(D)` — same dimension
    /// - `.perElement(C)` ↔ `.perElement(C)` — same count
    /// - `.perRow(D)` ↔ `.perElement(D)` — coerce to `.perRow`
    /// - `.perHead(H,D)` ↔ `.perHead(H,D)` — same head count and dimension
    public func isCompatible(with other: KernelParallelism) -> Bool {
        switch (self, other) {
        case (.perRow(let d1), .perRow(let d2)):
            return d1 == d2
        case (.perElement(let c1), .perElement(let c2)):
            return c1 == c2
        case (.perRow(let d), .perElement(let c)),
             (.perElement(let c), .perRow(let d)):
            return d == c
        case (.perHead(let h1, let d1), .perHead(let h2, let d2)):
            return h1 == h2 && d1 == d2
        default:
            return false
        }
    }

    /// Resolve the effective parallelism when fusing two compatible patterns.
    /// `.perRow` takes precedence over `.perElement` (more constrained).
    public func resolved(with other: KernelParallelism) -> KernelParallelism {
        switch (self, other) {
        case (.perRow, _): return self
        case (_, .perRow): return other
        case (.perHead, _): return self
        case (_, .perHead): return other
        default: return self
        }
    }
}

// MARK: - Fusion Eligibility

extension FusionContract {
    /// Determine the intermediate storage strategy for connecting
    /// this contract's output to another contract's input.
    public func intermediateStorage(to consumer: FusionContract) -> IntermediateStorage {
        guard let consumerInput = consumer.primaryInput else {
            return .threadgroupMemory(dimension: parallelism.dimension)
        }

        switch consumerInput.accessPattern {
        case .singlePass:
            let effective = parallelism.resolved(with: consumer.parallelism)
            if case .perElement = effective {
                return .register
            }
            // perRow with singlePass: use TG memory for now (loop merging optimization deferred)
            return .threadgroupMemory(dimension: effective.dimension)

        case .multiPass:
            let effective = parallelism.resolved(with: consumer.parallelism)
            return .threadgroupMemory(dimension: effective.dimension)
        }
    }
}

// MARK: - Private Helpers

private func isBufferRole(_ role: FusionPort.PortRole) -> Bool {
    if case .buffer = role { return true }
    return false
}
