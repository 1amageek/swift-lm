import LMIR

// MARK: - Dispatch Entry

/// A single kernel dispatch in the plan (IR → fusion → steps).
public struct DispatchEntry: Sendable {
    public let index: Int
    public let fragment: any PrimitiveMetalKernelFragment
    public let parameterBindings: [ParameterBinding]
    public let layerIndex: Int?
    public let compositeID: Int?

    public init(
        index: Int,
        fragment: any PrimitiveMetalKernelFragment,
        parameterBindings: [ParameterBinding] = [],
        layerIndex: Int? = nil,
        compositeID: Int? = nil
    ) {
        self.index = index
        self.fragment = fragment
        self.parameterBindings = parameterBindings
        self.layerIndex = layerIndex
        self.compositeID = compositeID
    }

    /// Human-readable description of the fragment for diagnostics and logging.
    public var fragmentDescription: String {
        if let projection = fragment as? ProjectionDescribing {
            let fields = projection.projectionFields.map(\.field).joined(separator: ",")
            return "projection(\(fields), isOutput: \(projection.isOutputProjection))"
        }
        return String(describing: type(of: fragment))
    }
}
