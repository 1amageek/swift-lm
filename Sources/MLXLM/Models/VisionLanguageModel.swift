import MLX
import MLXFast
import MLXNN

/// Configuration interface for VLM input processors.
///
/// Provides the token IDs and spatial merge information needed to
/// process vision placeholders in the text token sequence.
protocol VLMInputConfig {
    var visionStartTokenId: Int { get }
    var visionEndTokenId: Int { get }
    var imageTokenId: Int { get }
    var videoTokenId: Int { get }
    var spatialMergeSize: Int { get }
}

/// A language model that can process visual input alongside text.
///
/// VLMs encode images/videos through a vision encoder and merge the resulting
/// embeddings into the text sequence before autoregressive generation.
///
/// Conforming types provide thin bridge methods (``encodeVision(image:video:)``,
/// ``embedTokens(_:)``, ``forwardTextModel(_:cache:inputEmbeddings:positionIds:)``,
/// ``forwardLogits(_:)``), and the protocol extension supplies default
/// implementations for ``LanguageModel/prepare(_:cache:windowSize:)``,
/// ``LanguageModel/callAsFunction(_:cache:state:)-1mh8j``, and the shared
/// M-RoPE position ID computation.
public protocol VisionLanguageModel: LanguageModel {

    /// Encode visual input into embeddings aligned with the LLM's hidden dimension.
    func encodeVision(
        image: LMInput.ProcessedImage?,
        video: LMInput.ProcessedVideo?
    ) throws -> MLXArray?

    /// Token ID used as placeholder for image content in the text sequence.
    var imageTokenId: Int { get }

    /// Token ID used as placeholder for video content in the text sequence.
    var videoTokenId: Int { get }

    /// Spatial merge factor used by the vision encoder (typically 2).
    var spatialMergeSize: Int { get }

    /// Current text position for sequential M-RoPE generation.
    var nextPosition: Int { get set }

    /// Embed token IDs into continuous representations.
    func embedTokens(_ tokens: MLXArray) -> MLXArray

    /// Forward pass through the text model with optional embedding injection.
    func forwardTextModel(
        _ inputs: MLXArray, cache: [KVCache]?,
        inputEmbeddings: MLXArray?, positionIds: MLXArray?
    ) -> MLXArray

    /// Compute logits from hidden states (via LM head or tied embeddings).
    func forwardLogits(_ h: MLXArray) -> MLXArray
}

// MARK: - Default Implementations

extension VisionLanguageModel {

    /// Shared prefill: encode vision, merge embeddings, compute M-RoPE, forward.
    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        let prefillOffset = cache.first?.offset ?? 0

        if prefillOffset == 0 {
            let visionEmbeddings = try encodeVision(
                image: input.image, video: input.video)

            let (positionIds, nextTextPos) = computeMRoPEPositionIds(
                inputIds: tokens, image: input.image, video: input.video)

            var embeddings = embedTokens(tokens)
            if let visionEmbeddings {
                embeddings = mergeVisionEmbeddings(
                    textEmbeddings: embeddings,
                    visionEmbeddings: visionEmbeddings,
                    inputIds: tokens)
            }

            let h = forwardTextModel(
                tokens, cache: cache,
                inputEmbeddings: embeddings, positionIds: positionIds)
            let logits = forwardLogits(h)
            self.nextPosition = nextTextPos
            return .logits(LMOutput(logits: logits))
        }

        let remaining = tokens[0..., prefillOffset...]
        let output = callAsFunction(
            LMInput.Text(tokens: remaining), cache: cache, state: nil)
        return .logits(output)
    }

    /// Generation phase: sequential M-RoPE positions.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let positionIds = input.positionIds ?? makeSequentialPositionIds(
            batchSize: input.tokens.dim(0),
            seqLen: input.tokens.dim(1),
            startPosition: nextPosition)

        let h = forwardTextModel(
            input.tokens, cache: cache,
            inputEmbeddings: nil, positionIds: positionIds)
        let logits = forwardLogits(h)
        self.nextPosition += input.tokens.dim(1)
        return LMOutput(logits: logits)
    }

    /// Raw forward pass without position tracking.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = forwardTextModel(
            inputs, cache: cache, inputEmbeddings: nil, positionIds: nil)
        return forwardLogits(h)
    }

    // MARK: - Shared Helpers

    /// Replace placeholder token embeddings with vision encoder output.
    func mergeVisionEmbeddings(
        textEmbeddings: MLXArray,
        visionEmbeddings: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let B = textEmbeddings.dim(0)
        let S = textEmbeddings.dim(1)
        let D = textEmbeddings.dim(2)

        let flatIds = inputIds.reshaped(-1)
        let isImage = flatIds .== MLXArray(Int32(imageTokenId))
        let isVideo = flatIds .== MLXArray(Int32(videoTokenId))
        let isVision = isImage .|| isVideo

        let visionCumsum = cumsum(isVision.asType(DType.int32)) - 1
        let clampedIdx = clip(
            visionCumsum, min: 0,
            max: max(visionEmbeddings.dim(0) - 1, 0))
        let visionGathered = visionEmbeddings[clampedIdx]

        let flatEmbeddings = textEmbeddings.reshaped(B * S, D)
        let mask = isVision.reshaped(B * S, 1)
        let merged = which(mask, visionGathered, flatEmbeddings)

        return merged.reshaped(B, S, D)
    }

    /// Compute M-RoPE 3D position IDs for mixed text+vision sequences.
    ///
    /// Text tokens: all 3 axes get the same sequential position.
    /// Vision tokens: temporal/height/width from grid layout (after spatial merge).
    ///
    /// - Returns: `(positionIds: [3, B, S], nextTextPosition: Int)`.
    func computeMRoPEPositionIds(
        inputIds: MLXArray,
        image: LMInput.ProcessedImage?,
        video: LMInput.ProcessedVideo?
    ) -> (MLXArray, Int) {
        let B = inputIds.dim(0)
        let S = inputIds.dim(1)

        var temporalPos = [Int32](repeating: 0, count: B * S)
        var heightPos = [Int32](repeating: 0, count: B * S)
        var widthPos = [Int32](repeating: 0, count: B * S)

        let flatIds = inputIds.reshaped(-1)

        var currentTextPos: Int32 = 0
        var visionTokenIdx = 0
        let allGrids = (image?.frames ?? []) + (video?.frames ?? [])
        var gridIdx = 0

        for i in 0..<(B * S) {
            let tokenId: Int32 = flatIds[i].item()

            if tokenId == Int32(imageTokenId) || tokenId == Int32(videoTokenId) {
                if gridIdx < allGrids.count {
                    let grid = allGrids[gridIdx]
                    let mergedH = grid.h / spatialMergeSize
                    let mergedW = grid.w / spatialMergeSize
                    let totalMerged = grid.t * mergedH * mergedW

                    let posInGrid = visionTokenIdx
                    let tPos = posInGrid / (mergedH * mergedW)
                    let hPos = (posInGrid % (mergedH * mergedW)) / mergedW
                    let wPos = posInGrid % mergedW

                    temporalPos[i] = currentTextPos + Int32(tPos)
                    heightPos[i] = currentTextPos + Int32(hPos)
                    widthPos[i] = currentTextPos + Int32(wPos)

                    visionTokenIdx += 1
                    if visionTokenIdx >= totalMerged {
                        currentTextPos += Int32(max(grid.t, max(mergedH, mergedW)))
                        visionTokenIdx = 0
                        gridIdx += 1
                    }
                }
            } else {
                temporalPos[i] = currentTextPos
                heightPos[i] = currentTextPos
                widthPos[i] = currentTextPos
                currentTextPos += 1
            }
        }

        let tArray = MLXArray(temporalPos).reshaped(B, S)
        let hArray = MLXArray(heightPos).reshaped(B, S)
        let wArray = MLXArray(widthPos).reshaped(B, S)

        return (stacked([tArray, hArray, wArray], axis: 0), Int(currentTextPos))
    }

    /// Create sequential M-RoPE position IDs for text-only generation.
    func makeSequentialPositionIds(
        batchSize: Int, seqLen: Int, startPosition: Int
    ) -> MLXArray {
        let positions = tiled(
            MLXArray(Int32(startPosition)..<Int32(startPosition + seqLen))
                .reshaped(1, seqLen),
            repetitions: [batchSize, 1])
        return stacked([positions, positions, positions], axis: 0)
    }
}

/// A vision encoder that transforms pixel data into feature embeddings.
///
/// Implementations include ViT variants (CLIP, SigLIP, custom ViTs)
/// paired with optional spatial merging and projection layers.
protocol VisionEncoder: Module {

    /// Encode pixel data into feature embeddings.
    ///
    /// - Parameters:
    ///   - pixels: Pixel tensor (layout is model-specific, e.g. `[N, C, T, H, W]`).
    ///   - gridTHW: Per-image/video grid dimensions for position encoding.
    /// - Returns: Feature embeddings of shape `[totalTokens, outputDim]`.
    func callAsFunction(_ pixels: MLXArray, gridTHW: [LMInput.THW]) -> MLXArray
}
