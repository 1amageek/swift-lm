import Foundation
import LMArchitecture
import MetalCompiler

struct HFConfigDecoder {
    func quantizationHint(from data: Data) throws -> MLXQuantizationHint? {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig("config.json is not a JSON object")
        }
        let candidates: [[String: Any]] = [
            json["quantization"] as? [String: Any],
            (json["text_config"] as? [String: Any])?["quantization"] as? [String: Any]
        ].compactMap { $0 }
        guard let dict = candidates.first,
              let bits = dict["bits"] as? Int,
              let groupSize = dict["group_size"] as? Int else {
            return nil
        }
        return MLXQuantizationHint(bits: bits, groupSize: groupSize)
    }

    func modelType(from data: Data) throws -> String {
        do {
            return try HuggingFaceConfigDecoder().decode(from: data).modelType
        } catch let error as HuggingFaceConfigError {
            throw ModelBundleLoaderError.invalidConfig(error.description)
        }
    }

    func decode(from data: Data) throws -> ModelConfig {
        do {
            return try HuggingFaceConfigDecoder().decode(from: data).modelConfig
        } catch let error as HuggingFaceConfigError {
            throw ModelBundleLoaderError.invalidConfig(error.description)
        }
    }

    func inputCapabilities(
        from configData: Data,
        preprocessorConfigData: Data? = nil,
        visionConfiguration: ModelVisionConfiguration? = nil
    ) throws -> ModelInputCapabilities {
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig("config.json is not a JSON object")
        }

        let modelType = (config["model_type"] as? String ?? "").lowercased()
        let resolvedVisionConfiguration: ModelVisionConfiguration
        if let visionConfiguration {
            resolvedVisionConfiguration = visionConfiguration
        } else {
            resolvedVisionConfiguration = try self.visionConfiguration(
                from: configData,
                preprocessorConfigData: preprocessorConfigData
            ) ?? ModelVisionConfiguration()
        }

        let hasVisionConfig = config["vision_config"] != nil || modelType.contains("_vl")
        let supportsImages =
            resolvedVisionConfiguration.imageTokenID != nil &&
            (
                hasVisionConfig ||
                Gemma4Support.supportsImageProcessorClass(
                    resolvedVisionConfiguration.processorClass
                ) ||
                QwenVisionSupport.supportsImageProcessorClass(
                    resolvedVisionConfiguration.processorClass
                ) ||
                QwenVisionSupport.supportsImageProcessorType(
                    resolvedVisionConfiguration.imageProcessorType
                )
            )
        let supportsVideo =
            resolvedVisionConfiguration.videoTokenID != nil &&
            (
                hasVisionConfig ||
                QwenVisionSupport.supportsVideoProcessorClass(
                    resolvedVisionConfiguration.processorClass
                ) ||
                QwenVisionSupport.supportsVideoProcessorType(
                    resolvedVisionConfiguration.videoProcessorType
                )
            )

        return ModelInputCapabilities(
            supportsText: true,
            supportsImages: supportsImages,
            supportsVideo: supportsVideo
        )
    }

    func visionConfiguration(
        from configData: Data,
        preprocessorConfigData: Data? = nil
    ) throws -> ModelVisionConfiguration? {
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig("config.json is not a JSON object")
        }

        let hasVisionConfig = config["vision_config"] != nil
        let visionConfig = config["vision_config"] as? [String: Any]
        let imageTokenID = config["image_token_id"] as? Int
        let videoTokenID = config["video_token_id"] as? Int
        let visionStartTokenID = config["vision_start_token_id"] as? Int
        let visionEndTokenID = config["vision_end_token_id"] as? Int

        let processorConfig: [String: Any]?
        if let preprocessorConfigData {
            guard let json = try JSONSerialization.jsonObject(with: preprocessorConfigData) as? [String: Any] else {
                throw ModelBundleLoaderError.invalidConfig(
                    "preprocessor_config.json is not a JSON object"
                )
            }
            processorConfig = json
        } else {
            processorConfig = nil
        }

        let imageProcessorType = processorConfig?["image_processor_type"] as? String
        let videoProcessorType = processorConfig?["video_processor_type"] as? String
        let processorClass = processorConfig?["processor_class"] as? String
        let patchSize = processorConfig?["patch_size"] as? Int
        let temporalPatchSize = processorConfig?["temporal_patch_size"] as? Int
        let poolingKernelSize = processorConfig?["pooling_kernel_size"] as? Int
        let mergeSize = processorConfig?["merge_size"] as? Int
        let size = processorConfig?["size"] as? [String: Any]
        let imageMean = processorConfig?["image_mean"] as? [Double] ?? []
        let imageStd = processorConfig?["image_std"] as? [Double] ?? []
        let minimumPixelCount =
            size?["shortest_edge"] as? Int
            ?? processorConfig?["min_pixels"] as? Int
        let maximumPixelCount =
            size?["longest_edge"] as? Int
            ?? processorConfig?["max_pixels"] as? Int
        let videoFramesPerSecond =
            processorConfig?["fps"] as? Double
            ?? (processorConfig?["fps"] as? Int).map(Double.init)
        let minimumFrameCount =
            processorConfig?["min_frames"] as? Int
        let maximumFrameCount =
            processorConfig?["max_frames"] as? Int

        let hasKnownImageProcessorClass = QwenVisionSupport.supportsImageProcessorClass(
            processorClass
        ) || Gemma4Support.supportsImageProcessorClass(processorClass)
        let hasKnownImageProcessorType = QwenVisionSupport.supportsImageProcessorType(
            imageProcessorType
        )
        let hasKnownVideoProcessorType = QwenVisionSupport.supportsVideoProcessorType(
            videoProcessorType
        )
        guard hasVisionConfig ||
                imageTokenID != nil ||
                videoTokenID != nil ||
                visionStartTokenID != nil ||
                visionEndTokenID != nil ||
                hasKnownImageProcessorClass ||
                hasKnownImageProcessorType ||
                hasKnownVideoProcessorType ||
                patchSize != nil ||
                temporalPatchSize != nil ||
                mergeSize != nil ||
                minimumPixelCount != nil ||
                maximumPixelCount != nil else {
            return nil
        }

        return ModelVisionConfiguration(
            hiddenSize: visionConfig?["hidden_size"] as? Int,
            depth: visionConfig?["depth"] as? Int
                ?? visionConfig?["num_hidden_layers"] as? Int,
            intermediateSize: visionConfig?["intermediate_size"] as? Int,
            outHiddenSize: visionConfig?["out_hidden_size"] as? Int
                ?? visionConfig?["output_proj_dims"] as? Int,
            headCount: visionConfig?["num_heads"] as? Int
                ?? visionConfig?["num_attention_heads"] as? Int,
            numPositionEmbeddings: visionConfig?["num_position_embeddings"] as? Int,
            inChannels: visionConfig?["in_channels"] as? Int
                ?? visionConfig?["num_channels"] as? Int,
            hiddenAct: visionConfig?["hidden_act"] as? String
                ?? visionConfig?["hidden_activation"] as? String,
            deepstackVisualIndexes: visionConfig?["deepstack_visual_indexes"] as? [Int] ?? [],
            processorClass: processorClass,
            imageTokenID: imageTokenID,
            videoTokenID: videoTokenID,
            visionStartTokenID: visionStartTokenID,
            visionEndTokenID: visionEndTokenID,
            imageProcessorType: imageProcessorType,
            videoProcessorType: videoProcessorType,
            patchSize: patchSize ?? visionConfig?["patch_size"] as? Int,
            poolingKernelSize: poolingKernelSize ?? visionConfig?["pooling_kernel_size"] as? Int,
            temporalPatchSize: temporalPatchSize ?? visionConfig?["temporal_patch_size"] as? Int,
            mergeSize: mergeSize,
            spatialMergeSize: visionConfig?["spatial_merge_size"] as? Int,
            positionEmbeddingSize: visionConfig?["position_embedding_size"] as? Int,
            defaultOutputLength: visionConfig?["default_output_length"] as? Int
                ?? config["vision_soft_tokens_per_image"] as? Int,
            ropeTheta: {
                let ropeParams = visionConfig?["rope_parameters"] as? [String: Any]
                if let theta = ropeParams?["rope_theta"] as? Double {
                    return Float(theta)
                }
                if let theta = ropeParams?["rope_theta"] as? Int {
                    return Float(theta)
                }
                return nil
            }(),
            standardize: visionConfig?["standardize"] as? Bool,
            minimumPixelCount: minimumPixelCount,
            maximumPixelCount: maximumPixelCount,
            videoFramesPerSecond: videoFramesPerSecond,
            minimumFrameCount: minimumFrameCount,
            maximumFrameCount: maximumFrameCount,
            imageMean: imageMean,
            imageStd: imageStd
        )
    }
}
