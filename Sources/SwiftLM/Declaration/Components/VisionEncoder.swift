/// Vision encoder component for vision-language models.
///
/// Represents a Vision Transformer (ViT) that processes image pixels
/// into dense feature vectors. This is a source operation that reads
/// image data from the runtime context.
///
/// ```swift
/// VisionEncoder(
///     hiddenSize: 768,
///     outputSize: 1024,
///     depth: 12,
///     headCount: 12,
///     patchSize: 16,
///     intermediateSize: 3072
/// )
/// ```
public struct VisionEncoder: ModelComponent {

    public typealias Body = Never

    public let hiddenSize: Int
    public let outputSize: Int
    public let depth: Int
    public let headCount: Int
    public let patchSize: Int
    public let spatialMergeSize: Int
    public let inChannels: Int
    public let intermediateSize: Int
    public let mlpActivation: ActivationKind
    public let mlpGating: GatingKind
    public let normEpsilon: Float
    public let bias: Bool
    public let temporalPatchSize: Int?

    public init(
        hiddenSize: Int,
        outputSize: Int,
        depth: Int,
        headCount: Int,
        patchSize: Int,
        spatialMergeSize: Int = 2,
        inChannels: Int = 3,
        intermediateSize: Int,
        mlpActivation: ActivationKind = .gelu,
        mlpGating: GatingKind = .none,
        normEpsilon: Float = 1e-6,
        bias: Bool = true,
        temporalPatchSize: Int? = nil
    ) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(outputSize > 0, "outputSize must be positive")
        precondition(depth > 0, "depth must be positive")
        precondition(headCount > 0, "headCount must be positive")
        precondition(hiddenSize % headCount == 0, "hiddenSize must be divisible by headCount")
        precondition(patchSize > 0, "patchSize must be positive")
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.depth = depth
        self.headCount = headCount
        self.patchSize = patchSize
        self.spatialMergeSize = spatialMergeSize
        self.inChannels = inChannels
        self.intermediateSize = intermediateSize
        self.mlpActivation = mlpActivation
        self.mlpGating = mlpGating
        self.normEpsilon = normEpsilon
        self.bias = bias
        self.temporalPatchSize = temporalPatchSize
    }
}

extension VisionEncoder: PrimitiveComponent {

    package var operationKind: OperationKind {
        .visionEncoder(VisionEncoderAttributes(
            hiddenSize: hiddenSize,
            outputSize: outputSize,
            depth: depth,
            headCount: headCount,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            inChannels: inChannels,
            intermediateSize: intermediateSize,
            mlpActivation: mlpActivation,
            mlpGating: mlpGating,
            normEpsilon: normEpsilon,
            bias: bias,
            temporalPatchSize: temporalPatchSize
        ))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(0), resultArity: .exact(1))
    }
}
