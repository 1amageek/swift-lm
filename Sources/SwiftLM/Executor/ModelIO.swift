/// Input tensors for model execution.
public struct ModelInputs: Sendable {

    /// Token IDs to process.
    public let tokenIDs: TensorData

    /// Optional position IDs (for models requiring explicit positions).
    public let positionIDs: TensorData?

    /// Optional attention mask.
    public let attentionMask: TensorData?

    /// Optional KV cache from a previous execution step.
    public let cache: KVCacheState?

    /// Optional image pixel data for vision-language models.
    ///
    /// Shape depends on the vision encoder:
    /// - 2D (Conv2d): `[N, H, W, C]` where N = number of images
    /// - 3D (Conv3d): `[N, T, H, W, C]` where T = temporal frames
    public let imagePixels: TensorData?

    /// Grid dimensions for each image/video in the batch.
    ///
    /// Each entry is a `[T, H, W]` triple describing the grid of patches:
    /// - T: number of temporal frames (1 for images)
    /// - H: number of patch rows after smart resize
    /// - W: number of patch columns after smart resize
    public let imageGridSizes: TensorData?

    public init(
        tokenIDs: TensorData,
        positionIDs: TensorData? = nil,
        attentionMask: TensorData? = nil,
        cache: KVCacheState? = nil,
        imagePixels: TensorData? = nil,
        imageGridSizes: TensorData? = nil
    ) {
        self.tokenIDs = tokenIDs
        self.positionIDs = positionIDs
        self.attentionMask = attentionMask
        self.cache = cache
        self.imagePixels = imagePixels
        self.imageGridSizes = imageGridSizes
    }
}

/// Output tensors from model execution.
public struct ModelOutputs: Sendable {

    /// Logits over the vocabulary for each position.
    public let logits: TensorData

    /// Updated KV cache state after execution.
    public let cache: KVCacheState?

    /// Optional hidden states from intermediate layers.
    public let hiddenStates: TensorData?

    public init(
        logits: TensorData,
        cache: KVCacheState? = nil,
        hiddenStates: TensorData? = nil
    ) {
        self.logits = logits
        self.cache = cache
        self.hiddenStates = hiddenStates
    }
}

/// Opaque KV cache state for autoregressive generation.
///
/// The concrete representation depends on the runtime backend.
/// SwiftLM core defines only the contract; backend-specific
/// implementations provide typed access to cache contents.
public struct KVCacheState: Sendable {

    /// Opaque backend-specific cache data.
    public let storage: any Sendable

    /// Number of cached positions (sequence length already processed).
    public let cachedLength: Int

    public init(storage: any Sendable, cachedLength: Int) {
        self.storage = storage
        self.cachedLength = cachedLength
    }
}
