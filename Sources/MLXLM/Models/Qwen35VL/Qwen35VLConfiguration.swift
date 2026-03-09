/// Configuration for Qwen 3.5 vision-language model.
///
/// Groups the text decoder config, vision encoder config, and
/// vision token IDs for model construction.
struct Qwen35VLConfiguration: Sendable {

    let text: Qwen35Configuration
    let vision: VisionConfiguration

    /// Token IDs for vision placeholders in the text sequence.
    let imageTokenId: Int
    let videoTokenId: Int
    let visionStartTokenId: Int
    let visionEndTokenId: Int

    init(
        text: Qwen35Configuration,
        vision: VisionConfiguration,
        imageTokenId: Int,
        videoTokenId: Int,
        visionStartTokenId: Int,
        visionEndTokenId: Int
    ) {
        self.text = text
        self.vision = vision
        self.imageTokenId = imageTokenId
        self.videoTokenId = videoTokenId
        self.visionStartTokenId = visionStartTokenId
        self.visionEndTokenId = visionEndTokenId
    }

    // MARK: - Vision Configuration

    /// Vision encoder configuration for Qwen 3.5.
    ///
    /// Uses GELU MLP (fc1+fc2), Conv2d patch embedding, and no window attention.
    struct VisionConfiguration: Sendable {
        var hiddenSize: Int
        var intermediateSize: Int
        var depth: Int
        var numHeads: Int
        var outHiddenSize: Int
        var patchSize: Int
        var spatialMergeSize: Int
        var inChannels: Int
        var normEps: Float
        var imageMean: [Float]
        var imageStd: [Float]
        var minPixels: Int
        var maxPixels: Int

        init(
            hiddenSize: Int,
            intermediateSize: Int,
            depth: Int,
            numHeads: Int,
            outHiddenSize: Int,
            patchSize: Int = 16,
            spatialMergeSize: Int = 2,
            inChannels: Int = 3,
            normEps: Float = 1e-6,
            imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073],
            imageStd: [Float] = [0.26862954, 0.26130258, 0.27577711],
            minPixels: Int = 3136,
            maxPixels: Int = 12_845_056
        ) {
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.depth = depth
            self.numHeads = numHeads
            self.outHiddenSize = outHiddenSize
            self.patchSize = patchSize
            self.spatialMergeSize = spatialMergeSize
            self.inChannels = inChannels
            self.normEps = normEps
            self.imageMean = imageMean
            self.imageStd = imageStd
            self.minPixels = minPixels
            self.maxPixels = maxPixels
        }

        var headDim: Int { hiddenSize / numHeads }

        /// Factor for image dimensions (must be divisible by this).
        var imageFactor: Int { patchSize * spatialMergeSize }
    }
}

// MARK: - VLMInputConfig Conformance

extension Qwen35VLConfiguration: VLMInputConfig {
    var spatialMergeSize: Int { vision.spatialMergeSize }
}
