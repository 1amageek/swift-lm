/// Configuration for Qwen2.5-VL vision-language model.
struct Qwen25VLConfiguration: Sendable {

    let text: TextConfiguration
    let vision: VisionConfiguration
    let mrope: MRoPEConfiguration

    /// Token IDs for vision placeholders in the text sequence.
    let imageTokenId: Int
    let videoTokenId: Int
    let visionStartTokenId: Int
    let visionEndTokenId: Int

    init(
        text: TextConfiguration,
        vision: VisionConfiguration,
        mrope: MRoPEConfiguration = MRoPEConfiguration(),
        imageTokenId: Int,
        videoTokenId: Int,
        visionStartTokenId: Int,
        visionEndTokenId: Int
    ) {
        self.text = text
        self.vision = vision
        self.mrope = mrope
        self.imageTokenId = imageTokenId
        self.videoTokenId = videoTokenId
        self.visionStartTokenId = visionStartTokenId
        self.visionEndTokenId = visionEndTokenId
    }

    // MARK: - Text Configuration

    /// LLM decoder configuration (standard Qwen2 architecture).
    struct TextConfiguration: Sendable {
        var hiddenSize: Int
        var hiddenLayers: Int
        var intermediateSize: Int
        var attentionHeads: Int
        var kvHeads: Int
        var vocabularySize: Int
        var normEps: Float
        var ropeTheta: Float
        var maxPositionEmbeddings: Int?
        var tieWordEmbeddings: Bool

        init(
            hiddenSize: Int,
            hiddenLayers: Int,
            intermediateSize: Int,
            attentionHeads: Int,
            kvHeads: Int? = nil,
            vocabularySize: Int,
            normEps: Float = 1e-6,
            ropeTheta: Float = 1_000_000,
            maxPositionEmbeddings: Int? = nil,
            tieWordEmbeddings: Bool = true
        ) {
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads ?? attentionHeads
            self.vocabularySize = vocabularySize
            self.normEps = normEps
            self.ropeTheta = ropeTheta
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.tieWordEmbeddings = tieWordEmbeddings
        }

        var headDim: Int { hiddenSize / attentionHeads }
    }

    // MARK: - Vision Configuration

    /// Vision encoder configuration.
    ///
    /// All dimension fields are extracted from mmproj GGUF metadata.
    /// No model-specific default values — the loader must provide them.
    struct VisionConfiguration: Sendable {
        var hiddenSize: Int
        var intermediateSize: Int
        var depth: Int
        var numHeads: Int
        var outHiddenSize: Int
        var patchSize: Int
        var spatialMergeSize: Int
        var temporalPatchSize: Int
        var inChannels: Int
        var normEps: Float
        var windowSize: Int
        var fullAttBlockIndexes: [Int]

        /// Image normalization parameters from mmproj metadata.
        var imageMean: [Float]
        var imageStd: [Float]

        /// Image dimension constraints.
        var minPixels: Int
        var maxPixels: Int

        init(
            hiddenSize: Int,
            intermediateSize: Int,
            depth: Int,
            numHeads: Int,
            outHiddenSize: Int,
            patchSize: Int = 14,
            spatialMergeSize: Int = 2,
            temporalPatchSize: Int = 2,
            inChannels: Int = 3,
            normEps: Float = 1e-6,
            windowSize: Int = 112,
            fullAttBlockIndexes: [Int],
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
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.normEps = normEps
            self.windowSize = windowSize
            self.fullAttBlockIndexes = fullAttBlockIndexes
            self.imageMean = imageMean
            self.imageStd = imageStd
            self.minPixels = minPixels
            self.maxPixels = maxPixels
        }

        var headDim: Int { hiddenSize / numHeads }

        /// Factor for image dimensions (must be divisible by this).
        var imageFactor: Int { patchSize * spatialMergeSize }
    }

    // MARK: - M-RoPE Configuration

    /// Multimodal Rotary Position Embedding configuration.
    ///
    /// Splits head dimensions into 3 sections for temporal/height/width positioning.
    struct MRoPEConfiguration: Sendable {
        /// Head dimension split: [temporal, height, width].
        var sections: [Int]

        init(sections: [Int] = [16, 24, 24]) {
            self.sections = sections
        }

        var totalDimensions: Int { sections.reduce(0, +) }
    }
}

// MARK: - VLMInputConfig Conformance

extension Qwen25VLConfiguration: VLMInputConfig {
    var spatialMergeSize: Int { vision.spatialMergeSize }
}
