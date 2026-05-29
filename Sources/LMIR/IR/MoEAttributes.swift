/// Attributes for a Mixture-of-Experts node.
///
/// Routes tokens to a subset of expert MLPs via a gating mechanism.
public struct MoEAttributes: OperationAttributes, Codable, Equatable {

    /// Total number of experts.
    public let expertCount: Int

    /// Number of experts activated per token.
    public let expertsPerToken: Int

    /// Gating mechanism for expert selection.
    public let gateKind: MoEGateKind

    /// Whether top-k routing probabilities are normalized before expert aggregation.
    public let normalizeRoutingWeights: Bool

    /// Multiplicative scale applied to routing weights after optional normalization.
    public let routedScalingFactor: Float

    /// Whether routing uses an additive expert-bias tensor for expert selection.
    public let useExpertBias: Bool

    /// MLP attributes shared by all experts.
    public let expertMLP: MLPAttributes

    public init(
        expertCount: Int,
        expertsPerToken: Int,
        gateKind: MoEGateKind = .topK,
        normalizeRoutingWeights: Bool = false,
        routedScalingFactor: Float = 1.0,
        useExpertBias: Bool = false,
        expertMLP: MLPAttributes
    ) {
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.gateKind = gateKind
        self.normalizeRoutingWeights = normalizeRoutingWeights
        self.routedScalingFactor = routedScalingFactor
        self.useExpertBias = useExpertBias
        self.expertMLP = expertMLP
    }
}

/// Gating mechanism for expert selection.
public enum MoEGateKind: Codable, Equatable, Sendable {
    case topK
    case sigmoidTopK
    case custom(String)
}
