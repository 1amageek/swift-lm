/// Attributes for a vision encoder operation.
///
/// Describes a Vision Transformer (ViT) that processes image pixels
/// into dense feature vectors suitable for merging with text embeddings.
///
/// The vision encoder is a source operation (zero operands) that reads
/// image data from the runtime context (`ModelInputs.imagePixels`).
///
/// Architectural variants are captured by configuration:
/// - Qwen 3.5: Conv2d patches, GELU MLP, full attention, 2D-RoPE
/// - Qwen 2.5-VL: Conv3d patches, SwiGLU MLP, window+full attention, 2D-RoPE
public struct VisionEncoderAttributes: Codable, Equatable, Sendable {

    /// Hidden dimension of the vision transformer.
    public let hiddenSize: Int

    /// Output dimension projected to match the text model's hidden size.
    public let outputSize: Int

    /// Number of transformer blocks in the vision encoder.
    public let depth: Int

    /// Number of attention heads per vision block.
    public let headCount: Int

    /// Patch size for spatial tokenization (e.g., 14 or 16).
    public let patchSize: Int

    /// Spatial merge factor for the PatchMerger (e.g., 2 for 2x2 merge).
    public let spatialMergeSize: Int

    /// Number of input image channels.
    public let inChannels: Int

    /// Intermediate MLP dimension within each vision block.
    public let intermediateSize: Int

    /// MLP activation function (e.g., `.gelu` for Qwen 3.5, `.silu` for Qwen 2.5-VL).
    public let mlpActivation: ActivationKind

    /// MLP gating strategy (e.g., `.none` for standard, `.swiglu` for gated).
    public let mlpGating: GatingKind

    /// Normalization epsilon for RMSNorm / LayerNorm in vision blocks.
    public let normEpsilon: Float

    /// Whether projections include bias terms.
    public let bias: Bool

    /// Whether the patch embedding uses temporal convolution (Conv3d for video).
    public let temporalPatchSize: Int?

    /// Per-head dimension (derived from hiddenSize / headCount if not specified).
    public var headDimension: Int { hiddenSize / headCount }

    /// The image factor for smart resizing (patchSize * spatialMergeSize).
    public var imageFactor: Int { patchSize * spatialMergeSize }

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
        precondition(spatialMergeSize > 0, "spatialMergeSize must be positive")
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
