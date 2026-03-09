import Foundation
import GGUFParser
import GGUFTokenizer
import MLX
import MLXFast
import MLXNN

/// Qwen 3.5 vision-language model.
///
/// Combines a ViT-based vision encoder (GELU MLP, Conv2d, no window attention)
/// with the hybrid DeltaNet + Full Attention text decoder. All shared VLM
/// orchestration (vision encoding, embedding merging, M-RoPE position IDs)
/// is provided by ``VisionLanguageModel`` protocol default implementations.
class Qwen35VLModel: Module, VisionLanguageModel, KVCacheDimensionProvider {

    let configuration: Qwen35VLConfiguration

    @ModuleInfo(key: "visual") var visionEncoder: Qwen35VLVisionTransformer
    @ModuleInfo(key: "model") var model: Qwen35ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    // MARK: VisionLanguageModel

    var nextPosition: Int = 0
    var imageTokenId: Int { configuration.imageTokenId }
    var videoTokenId: Int { configuration.videoTokenId }
    var spatialMergeSize: Int { configuration.vision.spatialMergeSize }

    // MARK: KVCacheDimensionProvider

    var vocabularySize: Int { configuration.text.vocabularySize }
    var layerCount: Int { model.layers.count }
    var kvHeads: [Int] {
        (0..<configuration.text.hiddenLayers).map { i in
            configuration.text.isFullAttentionLayer(i) ? configuration.text.kvHeads : 0
        }
    }

    init(_ config: Qwen35VLConfiguration) {
        self.configuration = config

        self._visionEncoder.wrappedValue = Qwen35VLVisionTransformer(config.vision)
        self._model.wrappedValue = Qwen35ModelInner(config.text)
        if !config.text.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.text.hiddenSize, config.text.vocabularySize, bias: false
            )
        }
    }

    // MARK: - Bridge Methods (used by VisionLanguageModel default implementations)

    func encodeVision(
        image: LMInput.ProcessedImage?,
        video: LMInput.ProcessedVideo?
    ) throws -> MLXArray? {
        if let image {
            let gridTHW = image.frames ?? [LMInput.THW(t: 1, h: 1, w: 1)]
            return visionEncoder(image.pixels, gridTHW: gridTHW)
        }
        if let video {
            let gridTHW = video.frames ?? [LMInput.THW(t: 1, h: 1, w: 1)]
            return visionEncoder(video.pixels, gridTHW: gridTHW)
        }
        return nil
    }

    func embedTokens(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    func forwardTextModel(
        _ inputs: MLXArray, cache: [KVCache]?,
        inputEmbeddings: MLXArray?, positionIds: MLXArray?
    ) -> MLXArray {
        model(inputs, cache: cache, inputEmbeddings: inputEmbeddings, positionIds: positionIds)
    }

    func forwardLogits(_ h: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(h) }
        return model.embedTokens.asLinear(h)
    }

    // MARK: - Model-Specific (not shared)

    /// Hybrid cache: DeltaNetCache for DeltaNet layers, KVCacheSimple for full attention.
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        nextPosition = 0
        let params = parameters ?? GenerateParameters()
        return (0..<configuration.text.hiddenLayers).map { i in
            if configuration.text.isFullAttentionLayer(i) {
                if let maxSize = params.maxKVSize {
                    return RotatingKVCache(maxSize: maxSize)
                }
                return KVCacheSimple()
            }
            return DeltaNetCache()
        }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        for key in Array(result.keys) where key.contains("conv1d.weight") {
            if let w = result[key] {
                if w.ndim == 2 {
                    result[key] = w.expandedDimensions(axis: -1)
                } else if w.ndim == 3 {
                    result[key] = w.transposed(0, 2, 1)
                }
            }
        }
        return result
    }
}

// MARK: - GGUF Loading

extension Qwen35VLModel: GGUFLoadableModel {

    /// Detects Qwen 3.5 VL: requires mmproj URL and DeltaNet SSM tensors.
    package static func canLoad(from file: GGUFFile, context: GGUFLoadContext) -> Bool {
        context.mmprojURL != nil
            && file.tensors.contains { $0.name == "blk.0.ssm_beta.weight" }
    }

    package static func load(
        from file: GGUFFile, context: GGUFLoadContext
    ) throws -> GGUFLoadResult {
        guard let mmprojURL = context.mmprojURL else {
            throw GGUFLoadError.missingMetadata("mmproj file required for VLM")
        }

        var textConfig = try Qwen35Configuration(from: file)
        textConfig.mropeSections = [11, 11, 10]

        // Load vision encoder from mmproj
        let visionLoader = GGUFVisionLoader()
        let (loadedVisionEncoder, loadedVisionConfig) = try visionLoader.loadQwen35(url: mmprojURL)
        var visionConfig = loadedVisionConfig
        visionConfig.outHiddenSize = textConfig.hiddenSize

        // Resolve vision token IDs
        let tokenizer = context.tokenizer
        guard let imageTokenId = tokenizer.tokenID(for: "<|image_pad|>") else {
            throw GGUFLoadError.missingMetadata("tokenizer vocabulary: <|image_pad|>")
        }
        guard let videoTokenId = tokenizer.tokenID(for: "<|video_pad|>") else {
            throw GGUFLoadError.missingMetadata("tokenizer vocabulary: <|video_pad|>")
        }
        guard let visionStartTokenId = tokenizer.tokenID(for: "<|vision_start|>") else {
            throw GGUFLoadError.missingMetadata("tokenizer vocabulary: <|vision_start|>")
        }
        guard let visionEndTokenId = tokenizer.tokenID(for: "<|vision_end|>") else {
            throw GGUFLoadError.missingMetadata("tokenizer vocabulary: <|vision_end|>")
        }

        let vlmConfig = Qwen35VLConfiguration(
            text: textConfig,
            vision: visionConfig,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            visionStartTokenId: visionStartTokenId,
            visionEndTokenId: visionEndTokenId
        )
        let vlmModel = Qwen35VLModel(vlmConfig)

        let imageProcessor = Qwen35VLImageProcessor(config: visionConfig)

        let capturedVisionEncoder = loadedVisionEncoder
        let capturedVLMModel = vlmModel

        return GGUFLoadResult(
            model: vlmModel,
            mapper: Qwen35TensorNameMapper(),
            visionLoader: { _ in
                let visionParams = capturedVisionEncoder.parameters()
                capturedVLMModel.visionEncoder.update(parameters: visionParams)
                eval(capturedVLMModel.visionEncoder)
            },
            makeProcessor: { tokenizer, chatTemplate, bosToken, eosToken, addBosToken in
                VLMUserInputProcessor(
                    tokenizer: tokenizer,
                    chatTemplate: chatTemplate,
                    bosToken: bosToken,
                    eosToken: eosToken,
                    addBosToken: addBosToken,
                    vlmInputConfig: vlmConfig,
                    preprocessImage: { try imageProcessor.preprocess(image: $0) }
                )
            }
        )
    }
}

// MARK: - LoRA

extension Qwen35VLModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
