import LMArchitecture
import MetalCompiler
import ModelDeclarations

struct ModelGraphResolver {
    func resolveModelGraph(modelType: String, config: ModelConfig) throws -> ModelGraph {
        switch modelType.lowercased() {
        case "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2",
             "phi", "phi3", "starcoder2", "gpt_neox", "internlm2",
             "deepseek", "yi", "baichuan", "chatglm",
             "mixtral", "qwen2_moe", "deepseek_v2", "arctic", "dbrx":
            return try Transformer(config: config).makeModelGraph()
        case "qwen3_5":
            return try Qwen35(config: config).makeModelGraph()
        case "lfm2", "lfm2_moe":
            return try LFM2(config: config).makeModelGraph()
        case "cohere", "command-r":
            return try Cohere(config: config).makeModelGraph()
        case "nemotron_h":
            throw ModelBundleLoaderError.invalidConfig(
                "nemotron_h (Mamba-2 hybrid) is not yet supported"
            )
        default:
            return try Transformer(config: config).makeModelGraph()
        }
    }

    func namingConvention(for modelType: String) -> ParameterResolver.WeightNamingConvention {
        switch modelType.lowercased() {
        case "lfm2", "lfm2_moe":
            return .lfm2Family
        default:
            return .llamaFamily
        }
    }
}
