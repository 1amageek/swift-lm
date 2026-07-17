/// State-space model component.
///
/// Represents SSM variants such as Mamba, DeltaNet, or similar
/// recurrent/selective-state-space architectures.
///
/// ```swift
/// StateSpace(hiddenSize: 4096, stateSize: 16, variant: "mamba")
/// ```
public struct StateSpace: ModelComponent {

    public typealias Attributes = StateSpaceAttributes

    public let hiddenSize: Int
    public let numHeads: Int
    public let groupCount: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let convKernelSize: Int
    public let normEpsilon: Float
    public let variant: String

    public init(
        hiddenSize: Int, numHeads: Int,
        groupCount: Int? = nil,
        keyHeadDim: Int, valueHeadDim: Int,
        convKernelSize: Int = 1,
        normEpsilon: Float = 1e-6,
        variant: String
    ) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(numHeads > 0, "numHeads must be positive")
        precondition((groupCount ?? numHeads) > 0, "groupCount must be positive")
        precondition(keyHeadDim > 0, "keyHeadDim must be positive")
        precondition(valueHeadDim > 0, "valueHeadDim must be positive")
        precondition(convKernelSize > 0, "convKernelSize must be positive")
        precondition(normEpsilon > 0, "normEpsilon must be positive")
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.groupCount = groupCount ?? numHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convKernelSize = convKernelSize
        self.normEpsilon = normEpsilon
        self.variant = variant
    }
}

extension StateSpace {

    public var attributes: StateSpaceAttributes {
        StateSpaceAttributes(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            groupCount: groupCount,
            keyHeadDim: keyHeadDim,
            valueHeadDim: valueHeadDim,
            convKernelSize: convKernelSize,
            normEpsilon: normEpsilon,
            variant: variant
        )
    }
}
