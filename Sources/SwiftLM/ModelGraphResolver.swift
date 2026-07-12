import LMArchitecture
import LMIR
import ModelDeclarations

struct ModelGraphResolver {
    func resolveModelGraph(modelType: String, config: ModelConfig) throws -> ModelGraph {
        do {
            return try ModelFamilyRegistry.resolveModelGraph(
                modelType: modelType,
                config: config
            )
        } catch let error as ModelGraphBuildError {
            throw ModelBundleLoaderError.invalidConfig(error.description)
        }
    }

    func resolveEmbeddingBackboneGraph(modelType: String, config: ModelConfig) throws -> ModelGraph {
        do {
            return try ModelFamilyRegistry.resolveEmbeddingBackboneGraph(
                modelType: modelType,
                config: config
            )
        } catch let error as ModelGraphBuildError {
            throw ModelBundleLoaderError.invalidConfig(error.description)
        }
    }

    func namingConvention(for modelType: String) throws -> any WeightNamingConvention {
        do {
            return try ModelFamilyRegistry.namingConvention(for: modelType)
        } catch let error as ModelGraphBuildError {
            throw ModelBundleLoaderError.invalidConfig(error.description)
        }
    }
}
