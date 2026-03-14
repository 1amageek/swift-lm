import SwiftLM
import ModelDeclarations
import LMCompiler

/// Registry mapping HF model_type strings to ModelComponent factories and weight name mappers.
///
/// Known models produce verified ModelGraph instances via their ModelComponent declarations.
/// Unknown models fall back to AnyModel for best-effort IR construction.
public struct ModelRegistry: Sendable {

    /// Result of resolving a model_type.
    public struct Result: Sendable {
        /// The assembled model graph.
        public let graph: ModelGraph
        /// The weight name mapper for this model family.
        public let weightNameMapper: any WeightNameMapper
    }

    public init() {}

    /// Resolve a model_type to a ModelGraph and WeightNameMapper.
    ///
    /// - Parameters:
    ///   - modelType: The model_type string from config.json
    ///   - config: Decoded ModelConfig
    ///   - rawConfig: Raw JSON dictionary for model-specific field extraction
    /// - Returns: Result containing the ModelGraph and WeightNameMapper
    /// - Throws: ModelGraphBuildError if required fields are missing
    public func resolve(
        modelType: String,
        config: ModelConfig,
        rawConfig: [String: Any]
    ) throws -> Result {
        let key = modelType.lowercased()

        // Known model lookup
        if let entry = Self.knownModels[key] {
            let graph = try entry.factory(config, rawConfig)
            return Result(graph: graph, weightNameMapper: entry.mapper)
        }

        // VLM text model type check
        if let textConfig = rawConfig["text_config"] as? [String: Any],
           let textModelType = textConfig["model_type"] as? String {
            let textKey = textModelType.lowercased()
            if let entry = Self.textModelTypes[textKey] {
                let graph = try entry.factory(config, rawConfig)
                return Result(graph: graph, weightNameMapper: entry.mapper)
            }
        }

        // Fallback: AnyModel (best-effort)
        let graph = try AnyModel(config: config).makeModelGraph()
        return Result(graph: graph, weightNameMapper: LlamaFamilyWeightNameMapper())
    }

    // MARK: - Registry Entries

    private struct Entry: Sendable {
        let factory: @Sendable (ModelConfig, [String: Any]) throws -> ModelGraph
        let mapper: any WeightNameMapper
    }

    private static let llamaMapper = LlamaFamilyWeightNameMapper()
    private static let lfm2Mapper = LFM2FamilyWeightNameMapper()

    private static let knownModels: [String: Entry] = {
        let transformerEntry = Entry(
            factory: { config, _ in
                try ModelDeclarations.Transformer(config: config).makeModelGraph()
            },
            mapper: llamaMapper
        )

        let moeEntry = Entry(
            factory: { config, _ in
                try ModelDeclarations.Transformer(config: config).makeModelGraph()
            },
            mapper: llamaMapper
        )

        let cohereEntry = Entry(
            factory: { config, _ in
                try Cohere(config: config).makeModelGraph()
            },
            mapper: llamaMapper
        )

        let qwen35Entry = Entry(
            factory: { config, _ in
                try Qwen35.validate(config)
                return try Qwen35(config: config).makeModelGraph()
            },
            mapper: llamaMapper
        )

        let lfm2Entry = Entry(
            factory: { config, _ in
                try LFM2.validate(config)
                return try LFM2(config: config).makeModelGraph()
            },
            mapper: lfm2Mapper
        )

        return [
            // Standard transformer family
            "llama": transformerEntry,
            "qwen2": transformerEntry,
            "qwen3": transformerEntry,
            "mistral": transformerEntry,
            "gemma": transformerEntry,
            "gemma2": transformerEntry,
            "phi": transformerEntry,
            "phi3": transformerEntry,
            "starcoder2": transformerEntry,
            "gpt_neox": transformerEntry,
            "internlm2": transformerEntry,
            "deepseek": transformerEntry,
            "yi": transformerEntry,
            "baichuan": transformerEntry,
            "chatglm": transformerEntry,

            // Parallel attention + MLP family
            "cohere": cohereEntry,
            "command-r": cohereEntry,

            // MoE family
            "mixtral": moeEntry,
            "qwen2_moe": moeEntry,
            "deepseek_v2": moeEntry,
            "arctic": moeEntry,
            "dbrx": moeEntry,

            // Hybrid DeltaNet / attention family
            "qwen3_5": qwen35Entry,

            // Hybrid short-convolution / attention family
            "lfm2": lfm2Entry,
        ]
    }()

    private static let textModelTypes: [String: Entry] = [
        "qwen3_5_text": Entry(
            factory: { config, _ in
                try Qwen35.validate(config)
                return try Qwen35(config: config).makeModelGraph()
            },
            mapper: llamaMapper
        ),
        "qwen2_vl": Entry(
            factory: { config, _ in
                try ModelDeclarations.Transformer(config: config).makeModelGraph()
            },
            mapper: llamaMapper
        ),
    ]
}
