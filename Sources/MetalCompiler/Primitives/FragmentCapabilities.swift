/// Capability protocols for fragment self-description.
///
/// The compiler queries these protocols via capability cast (`as? ProjectionDescribing`)
/// instead of inspecting concrete fragment types (`as? LinearFragment`).
/// Each protocol captures a single cross-cutting concern shared by multiple fragment types.

// MARK: - Projection Capability

/// Describes a weight-bearing projection field within a fragment.
public struct ProjectionFieldDescriptor: Sendable {
    public let field: String
    public let inputDimension: Int
    public let outputDimension: Int

    public init(field: String, inputDimension: Int, outputDimension: Int) {
        self.field = field
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
    }
}

/// Fragments that represent weight-bearing projections (GEMV/GEMM).
///
/// The compiler uses this for buffer sizing, weight resolution, quantization planning,
/// and output projection marking — without knowing whether the fragment is a
/// LinearFragment, BatchedProjection, or any future projection type.
public protocol ProjectionDescribing: PrimitiveMetalKernelFragment {
    /// Projection fields this fragment represents.
    var projectionFields: [ProjectionFieldDescriptor] { get }

    /// Whether this projection writes back to the hidden buffer (or logits).
    var isOutputProjection: Bool { get }

    /// Create a copy with the output projection flag enabled.
    func withOutputProjectionEnabled() -> any PrimitiveMetalKernelFragment
}

// MARK: - Conv State Capability

/// Fragments that require conv state buffers for temporal convolution.
///
/// The compiler uses this for conv state buffer sizing.
/// Both Conv1dFragment and SSMRecurrenceFragment conform.
public protocol ConvStateRequiring: PrimitiveMetalKernelFragment {
    /// Total feature dimension for conv state allocation.
    var convStateDimension: Int { get }
}

// MARK: - Recurrent State Capability

/// Fragments that require recurrent state buffers for SSM recurrence.
///
/// The compiler uses this for recurrent state buffer sizing.
public protocol RecurrentStateRequiring: PrimitiveMetalKernelFragment {
    /// Bytes per layer for recurrent state allocation.
    var recurrentStateBytesPerLayer: Int { get }
}

// MARK: - Per-Layer Input Capability

/// Fragments that consume per-layer input vectors.
///
/// The compiler uses this for per-layer input buffer sizing.
public protocol PerLayerInputCapable: PrimitiveMetalKernelFragment {
    /// Dimension of the per-layer input vector this fragment consumes.
    var perLayerInputDimension: Int { get }
}
