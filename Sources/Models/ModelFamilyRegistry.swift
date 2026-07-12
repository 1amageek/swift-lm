import LMArchitecture
import LMIR

/// Resolves model metadata to a declarative graph and weight naming convention.
public enum ModelFamilyRegistry {
    public static func family(for modelType: String) -> ModelFamily? {
        switch modelType.lowercased() {
        case "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2",
             "phi", "phi3", "starcoder2", "gpt_neox", "internlm2",
             "deepseek", "yi", "baichuan", "chatglm", "mixtral",
             "qwen2_moe", "deepseek_v2", "arctic", "dbrx":
            return .transformer
        case "gemma3_text":
            return .gemma3Text
        case "gemma4", "gemma4_text":
            return .gemma4
        case "qwen3_5", "qwen3_vl", "qwen2_5_vl", "qwen2_vl":
            return .qwen35
        case "lfm2", "lfm2_moe":
            return .lfm2
        case "cohere", "command-r":
            return .cohere
        default:
            return nil
        }
    }

    public static func resolveModelGraph(
        modelType: String,
        config: ModelConfig
    ) throws -> ModelGraph {
        guard let family = family(for: modelType) else {
            throw ModelGraphBuildError.invalidConfig(
                "Unsupported model_type: \(modelType)"
            )
        }

        switch family {
        case .transformer:
            return try ModelGraph(Transformer(config: config))
        case .gemma3Text:
            try Gemma3Text.validate(config)
            return try ModelGraph(Gemma3Text(config: config))
        case .gemma4:
            try Gemma4.validate(config)
            return try ModelGraph(Gemma4(config: config))
        case .qwen35:
            try Qwen35.validate(config)
            return try ModelGraph(Qwen35(config: config))
        case .lfm2:
            return try ModelGraph(LFM2(config: config))
        case .cohere:
            return try ModelGraph(Cohere(config: config))
        }
    }

    public static func resolveEmbeddingBackboneGraph(
        modelType: String,
        config: ModelConfig
    ) throws -> ModelGraph {
        guard family(for: modelType) == .gemma3Text else {
            throw ModelGraphBuildError.invalidConfig(
                "Text embedding backbone is not supported for model_type: \(modelType)"
            )
        }
        return try ModelGraph(EmbeddingGemma(config: config))
    }

    public static func namingConvention(
        for modelType: String
    ) throws -> any WeightNamingConvention {
        guard let family = family(for: modelType) else {
            throw ModelGraphBuildError.invalidConfig(
                "Unsupported model_type: \(modelType)"
            )
        }

        switch family {
        case .gemma3Text:
            return Gemma3TextFamilyNaming()
        case .gemma4:
            return Gemma4FamilyNaming()
        case .qwen35:
            return Qwen35FamilyNaming()
        case .lfm2:
            return LFM2FamilyNaming()
        case .transformer, .cohere:
            return LlamaFamilyNaming()
        }
    }
}
