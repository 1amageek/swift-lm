/// DeltaNet state-space family component.
///
/// This is a semantic wrapper over the generic `stateSpace` IR node so model
/// declarations can name the paper-level family explicitly while still lowering
/// to the same optimized kernel path.
public struct DeltaNet: ModelComponent {

    public typealias Attributes = StateSpaceAttributes

    public enum Variant: String, Sendable, Equatable {
        case standard = "deltanet"
        case gated = "gated_deltanet"
    }

    public let hiddenSize: Int
    public let numHeads: Int
    public let groupCount: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let convKernelSize: Int
    public let variant: Variant

    public init(
        hiddenSize: Int,
        numHeads: Int,
        groupCount: Int? = nil,
        keyHeadDim: Int,
        valueHeadDim: Int,
        convKernelSize: Int = 1,
        variant: Variant = .standard
    ) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(numHeads > 0, "numHeads must be positive")
        precondition((groupCount ?? numHeads) > 0, "groupCount must be positive")
        precondition(keyHeadDim > 0, "keyHeadDim must be positive")
        precondition(valueHeadDim > 0, "valueHeadDim must be positive")
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.groupCount = groupCount ?? numHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convKernelSize = convKernelSize
        self.variant = variant
    }
}

extension DeltaNet {

    public var attributes: StateSpaceAttributes {
        StateSpaceAttributes(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            groupCount: groupCount,
            keyHeadDim: keyHeadDim,
            valueHeadDim: valueHeadDim,
            convKernelSize: convKernelSize,
            variant: variant.rawValue
        )
    }
}
