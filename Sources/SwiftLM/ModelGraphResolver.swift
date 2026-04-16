import LMArchitecture
import LMIR
import MetalCompiler
import ModelDeclarations

struct ModelGraphResolver {
    func resolveModelGraph(modelType: String, config: ModelConfig) throws -> ModelGraph {
        switch modelType.lowercased() {
        case "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2",
             "phi", "phi3", "starcoder2", "gpt_neox", "internlm2",
             "deepseek", "yi", "baichuan", "chatglm",
             "mixtral", "qwen2_moe", "deepseek_v2", "arctic", "dbrx":
            return try ModelGraph(Transformer(config: config))
        case "gemma4", "gemma4_text":
            do {
                try Gemma4.validate(config)
            } catch let error as ModelGraphBuildError {
                throw ModelBundleLoaderError.invalidConfig(error.description)
            }
            return try ModelGraph(Gemma4(config: config))
        case "gemma3_text":
            do {
                try Gemma3Text.validate(config)
            } catch let error as ModelGraphBuildError {
                throw ModelBundleLoaderError.invalidConfig(error.description)
            }
            return try ModelGraph(Gemma3Text(config: config))
        case "qwen3_5", "qwen3_vl", "qwen2_5_vl", "qwen2_vl":
            do {
                try Qwen35.validate(config)
            } catch let error as ModelGraphBuildError {
                throw ModelBundleLoaderError.invalidConfig(error.description)
            }
            return try ModelGraph(Qwen35(config: config))
        case "lfm2", "lfm2_moe":
            return try ModelGraph(LFM2(config: config))
        case "cohere", "command-r":
            return try ModelGraph(Cohere(config: config))
        case "nemotron_h":
            throw ModelBundleLoaderError.invalidConfig(
                "nemotron_h (Mamba-2 hybrid) is not yet supported"
            )
        default:
            throw ModelBundleLoaderError.invalidConfig(
                "Unsupported model_type: \(modelType)"
            )
        }
    }

    func resolveEmbeddingBackboneGraph(modelType: String, config: ModelConfig) throws -> ModelGraph {
        switch modelType.lowercased() {
        case "gemma3_text":
            do {
                return try ModelGraph(EmbeddingGemma(config: config))
            } catch let error as ModelGraphBuildError {
                throw ModelBundleLoaderError.invalidConfig(error.description)
            }
        default:
            throw ModelBundleLoaderError.invalidConfig(
                "Text embedding backbone is not supported for model_type: \(modelType)"
            )
        }
    }

    func namingConvention(for modelType: String) -> any WeightNamingConvention {
        switch modelType.lowercased() {
        case "gemma3_text":
            return Gemma3TextFamilyNaming()
        case "gemma4", "gemma4_text":
            return Gemma4FamilyNaming()
        case "qwen3_5", "qwen3_vl", "qwen2_5_vl", "qwen2_vl":
            return Qwen35FamilyNaming()
        case "lfm2", "lfm2_moe":
            return LFM2FamilyNaming()
        default:
            return LlamaFamilyNaming()
        }
    }
}
