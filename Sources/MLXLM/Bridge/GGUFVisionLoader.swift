import Foundation
import GGUFParser
import MLX
import MLXNN

/// Loads a vision encoder from a GGUF mmproj file.
///
/// Parses the mmproj GGUF to extract vision configuration from metadata
/// and loads vision encoder weights using the vision tensor name mapper.
struct GGUFVisionLoader {

    /// Load vision encoder from an mmproj GGUF file URL.
    ///
    /// - Parameter url: Path to the mmproj.gguf file.
    /// - Returns: Configured and loaded vision encoder with its configuration.
    func load(url: URL) throws -> (Qwen25VLVisionTransformer, Qwen25VLConfiguration.VisionConfiguration) {
        let file = try GGUFFile.parse(url: url)
        let config = try extractConfig(from: file)
        let encoder = Qwen25VLVisionTransformer(config)

        try loadWeights(into: encoder, from: file)
        eval(encoder)

        return (encoder, config)
    }

    // MARK: - Config Extraction

    private func extractConfig(from file: GGUFFile) throws -> Qwen25VLConfiguration.VisionConfiguration {
        func visionMeta(_ key: String) -> GGUFMetadataValue? {
            file.metadata["clip.vision.\(key)"]
        }

        // Required fields — throw if missing
        guard let hiddenSize = visionMeta("embedding_length")?.intValue else {
            throw GGUFLoadError.missingMetadata("clip.vision.embedding_length")
        }
        guard let depth = visionMeta("block_count")?.intValue else {
            throw GGUFLoadError.missingMetadata("clip.vision.block_count")
        }
        guard let numHeads = visionMeta("head_count")?.intValue else {
            throw GGUFLoadError.missingMetadata("clip.vision.head_count")
        }

        // Optional fields with safe defaults
        let intermediateSize = visionMeta("feed_forward_length")?.intValue ?? (hiddenSize * 4)
        let outHiddenSize = visionMeta("projection_dim")?.intValue ?? hiddenSize
        let patchSize = visionMeta("patch_size")?.intValue ?? 14
        let windowSize = visionMeta("image_size")?.intValue ?? (patchSize * 8)
        let spatialMergeSize = file.metadata["clip.vision.spatial_merge_size"]?.intValue ?? 2
        let normEps = visionMeta("attention.layer_norm_epsilon")?.float32Value ?? 1e-6

        // Parse full attention block pattern from metadata, or compute from depth
        let fullAttBlocks = parseFullAttBlockIndexes(from: file, depth: depth)

        // Image normalization from mmproj metadata
        let imageMean = extractFloatArray(from: file, key: "clip.vision.image_mean",
                                          fallback: [0.48145466, 0.4578275, 0.40821073])
        let imageStd = extractFloatArray(from: file, key: "clip.vision.image_std",
                                         fallback: [0.26862954, 0.26130258, 0.27577711])

        return Qwen25VLConfiguration.VisionConfiguration(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            depth: depth,
            numHeads: numHeads,
            outHiddenSize: outHiddenSize,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            normEps: normEps,
            windowSize: windowSize,
            fullAttBlockIndexes: fullAttBlocks,
            imageMean: imageMean,
            imageStd: imageStd
        )
    }

    /// Parse full attention block indexes from GGUF metadata.
    private func parseFullAttBlockIndexes(from file: GGUFFile, depth: Int) -> [Int] {
        // Try n_wa_pattern metadata (llama.cpp format)
        if let pattern = file.metadata["clip.vision.n_wa_pattern"]?.intValue, pattern > 0 {
            // Pattern value indicates interval: every `pattern` layers gets full attention
            // e.g. pattern=8 → blocks at indices 7, 15, 23, 31
            var indexes = [Int]()
            for i in stride(from: pattern - 1, to: depth, by: pattern) {
                indexes.append(i)
            }
            return indexes
        }

        // Compute from depth: full attention every depth/4 blocks
        let interval = max(1, depth / 4)
        return Array(stride(from: interval - 1, to: depth, by: interval))
    }

    /// Extract a float array from GGUF metadata.
    ///
    /// GGUF stores arrays as `.array([GGUFMetadataValue])` with each element
    /// being `.float32(Float)` or `.float64(Double)`.
    private func extractFloatArray(from file: GGUFFile, key: String, fallback: [Float]) -> [Float] {
        guard let value = file.metadata[key],
              case .array(let elements) = value else {
            return fallback
        }
        let floats = elements.compactMap { $0.float32Value }
        return floats.isEmpty ? fallback : floats
    }

    // MARK: - Qwen 3.5 Vision Loading

    /// Load Qwen 3.5 vision encoder from an mmproj GGUF file.
    func loadQwen35(url: URL) throws -> (Qwen35VLVisionTransformer, Qwen35VLConfiguration.VisionConfiguration) {
        let file = try GGUFFile.parse(url: url)
        let config = try extractQwen35Config(from: file)
        let encoder = Qwen35VLVisionTransformer(config)

        try loadQwen35Weights(into: encoder, from: file)
        eval(encoder)

        return (encoder, config)
    }

    private func extractQwen35Config(from file: GGUFFile) throws -> Qwen35VLConfiguration.VisionConfiguration {
        func visionMeta(_ key: String) -> GGUFMetadataValue? {
            file.metadata["clip.vision.\(key)"]
        }

        guard let hiddenSize = visionMeta("embedding_length")?.intValue else {
            throw GGUFLoadError.missingMetadata("clip.vision.embedding_length")
        }
        guard let depth = visionMeta("block_count")?.intValue else {
            throw GGUFLoadError.missingMetadata("clip.vision.block_count")
        }
        guard let numHeads = visionMeta("head_count")?.intValue else {
            throw GGUFLoadError.missingMetadata("clip.vision.head_count")
        }

        let intermediateSize = visionMeta("feed_forward_length")?.intValue ?? (hiddenSize * 4)
        let outHiddenSize = visionMeta("projection_dim")?.intValue ?? hiddenSize
        let patchSize = visionMeta("patch_size")?.intValue ?? 16
        let spatialMergeSize = file.metadata["clip.vision.spatial_merge_size"]?.intValue ?? 2
        let normEps = visionMeta("attention.layer_norm_epsilon")?.float32Value ?? 1e-6

        let imageMean = extractFloatArray(from: file, key: "clip.vision.image_mean",
                                          fallback: [0.48145466, 0.4578275, 0.40821073])
        let imageStd = extractFloatArray(from: file, key: "clip.vision.image_std",
                                         fallback: [0.26862954, 0.26130258, 0.27577711])

        return Qwen35VLConfiguration.VisionConfiguration(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            depth: depth,
            numHeads: numHeads,
            outHiddenSize: outHiddenSize,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            normEps: normEps,
            imageMean: imageMean,
            imageStd: imageStd
        )
    }

    private func loadQwen35Weights(
        into encoder: Qwen35VLVisionTransformer,
        from file: GGUFFile
    ) throws {
        let mapper = Qwen35VLVisionTensorNameMapper()
        let bridge = GGUFTensorBridge()
        var weights: [String: MLXArray] = [:]

        for tensor in file.tensors {
            guard let mlxName = mapper.mlxName(for: tensor.name) else {
                continue
            }
            let data = try file.tensorData(for: tensor)
            let array = try bridge.convert(tensor: tensor, data: data)
            weights[mlxName] = array
        }

        // Handle split QKV
        weights = fuseQwen35QKVIfNeeded(weights, depth: encoder.config.depth)

        // Sanitize Conv2d weights
        weights = sanitizeQwen35VisionWeights(weights)

        let parameters = ModuleParameters.unflattened(weights)
        try encoder.update(parameters: parameters, verify: .noUnusedKeys)
    }

    private func fuseQwen35QKVIfNeeded(
        _ weights: [String: MLXArray], depth: Int
    ) -> [String: MLXArray] {
        var result = weights

        for i in 0..<depth {
            let prefix = "blocks.\(i).attn"
            let qKey = "\(prefix).q_proj.weight"
            let kKey = "\(prefix).k_proj.weight"
            let vKey = "\(prefix).v_proj.weight"
            let qkvKey = "\(prefix).qkv.weight"

            if let q = result[qKey], let k = result[kKey], let v = result[vKey],
               result[qkvKey] == nil {
                result[qkvKey] = concatenated([q, k, v], axis: 0)
                result.removeValue(forKey: qKey)
                result.removeValue(forKey: kKey)
                result.removeValue(forKey: vKey)
            }

            let qBias = "\(prefix).q_proj.bias"
            let kBias = "\(prefix).k_proj.bias"
            let vBias = "\(prefix).v_proj.bias"
            let qkvBias = "\(prefix).qkv.bias"

            if let qb = result[qBias], let kb = result[kBias], let vb = result[vBias],
               result[qkvBias] == nil {
                result[qkvBias] = concatenated([qb, kb, vb], axis: 0)
                result.removeValue(forKey: qBias)
                result.removeValue(forKey: kBias)
                result.removeValue(forKey: vBias)
            }
        }

        return result
    }

    private func sanitizeQwen35VisionWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights

        for key in Array(result.keys) {
            guard let w = result[key] else { continue }

            // Conv2d patch embedding: PyTorch [O, I, H, W] -> MLX [O, H, W, I]
            if key.contains("patch_embed.proj.weight") && w.ndim == 4 {
                let shape = w.shape
                if shape[1] == 3 || shape[1] == 1 {
                    result[key] = w.transposed(0, 2, 3, 1)
                }
            }
        }

        return result
    }

    // MARK: - Qwen 2.5-VL Weight Loading

    private func loadWeights(
        into encoder: Qwen25VLVisionTransformer,
        from file: GGUFFile
    ) throws {
        let mapper = Qwen25VLVisionTensorNameMapper()
        let bridge = GGUFTensorBridge()
        var weights: [String: MLXArray] = [:]

        for tensor in file.tensors {
            guard let mlxName = mapper.mlxName(for: tensor.name) else {
                continue
            }

            let data = try file.tensorData(for: tensor)
            let array = try bridge.convert(tensor: tensor, data: data)
            weights[mlxName] = array
        }

        // Handle split QKV: if GGUF has separate Q/K/V, fuse them
        weights = fuseQKVIfNeeded(weights, encoder: encoder)

        // Handle split Conv3d: if GGUF has two Conv2d weights, combine them
        weights = fusePatchEmbedIfNeeded(weights)

        // Sanitize weights (transpose for MLX layout)
        let sanitized = sanitizeVisionWeights(weights)

        let parameters = ModuleParameters.unflattened(sanitized)
        try encoder.update(parameters: parameters, verify: .noUnusedKeys)
    }

    /// Fuse separate Q/K/V weight matrices into a single QKV matrix.
    private func fuseQKVIfNeeded(
        _ weights: [String: MLXArray],
        encoder: Qwen25VLVisionTransformer
    ) -> [String: MLXArray] {
        var result = weights

        for i in 0..<encoder.config.depth {
            let prefix = "blocks.\(i).attn"
            let qKey = "\(prefix).q_proj.weight"
            let kKey = "\(prefix).k_proj.weight"
            let vKey = "\(prefix).v_proj.weight"
            let qkvKey = "\(prefix).qkv.weight"

            // Only fuse if we have separate Q/K/V and no fused QKV
            if let q = result[qKey], let k = result[kKey], let v = result[vKey],
               result[qkvKey] == nil {
                result[qkvKey] = concatenated([q, k, v], axis: 0)
                result.removeValue(forKey: qKey)
                result.removeValue(forKey: kKey)
                result.removeValue(forKey: vKey)
            }

            // Same for biases
            let qBias = "\(prefix).q_proj.bias"
            let kBias = "\(prefix).k_proj.bias"
            let vBias = "\(prefix).v_proj.bias"
            let qkvBias = "\(prefix).qkv.bias"

            if let qb = result[qBias], let kb = result[kBias], let vb = result[vBias],
               result[qkvBias] == nil {
                result[qkvBias] = concatenated([qb, kb, vb], axis: 0)
                result.removeValue(forKey: qBias)
                result.removeValue(forKey: kBias)
                result.removeValue(forKey: vBias)
            }
        }

        return result
    }

    /// Combine two Conv2d weights into a single Conv3d weight (llama.cpp split format).
    private func fusePatchEmbedIfNeeded(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights

        let w0Key = "patch_embed.proj.weight"
        let w1Key = "patch_embed.proj.weight.1"

        if let w0 = result[w0Key], let w1 = result[w1Key] {
            // w0, w1 are Conv2d weights: [O, H, W, I]
            // Stack along temporal dimension: [O, 2, H, W, I]
            let combined = stacked([w0, w1], axis: 1)
            result[w0Key] = combined
            result.removeValue(forKey: w1Key)
        }

        return result
    }

    /// Sanitize vision weights for MLX layout conventions.
    private func sanitizeVisionWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights

        for key in Array(result.keys) {
            guard let w = result[key] else { continue }

            // Conv3d patch embedding: PyTorch [O, I, D, H, W] → MLX [O, D, H, W, I]
            if key.contains("patch_embed.proj.weight") && w.ndim == 5 {
                let shape = w.shape
                // Check if channels-first: shape[1] is small (input channels = 3)
                if shape[1] == 3 || shape[1] == 1 {
                    result[key] = w.transposed(0, 2, 3, 4, 1)
                }
            }
        }

        return result
    }
}
