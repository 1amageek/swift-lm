import CoreAILanguageModels
import Foundation

/// Strict loader for Apple Core AI language-model bundles.
@available(macOS 27.0, iOS 27.0, *)
public struct SwiftLMFoundationModelBundle: Sendable {
    private let bundle: LanguageBundle

    public init(contentsOf url: URL) throws {
        bundle = try LanguageBundle(at: url)
    }

    public var name: String { bundle.name }
    public var tokenizer: String { bundle.tokenizer }
    public var vocabSize: Int { bundle.vocabSize }
    public var maxContextLength: Int { bundle.maxContextLength }
    public var bundleURL: URL { bundle.bundlePath }
    public var isVisionLanguageModel: Bool { bundle.visionConfig != nil }

    public var visionConfiguration: SwiftLMVisionConfiguration? {
        bundle.visionConfig.map(SwiftLMVisionConfiguration.init)
    }

    public func makeLanguageModel(
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto
    ) async throws -> CoreAILanguageModel {
        guard !isVisionLanguageModel else {
            throw SwiftLMVisionLanguageModelError.visionLanguageModelRequiresVisionAPI
        }
        return try await CoreAILanguageModel(
            resourcesAt: bundle.bundlePath,
            variant: variant,
            kvCacheStrategy: kvCacheStrategy
        )
    }

    public func makeVisionLanguageModel(
        kvCacheStrategy: KVCacheStrategy = .auto
    ) async throws -> SwiftLMVisionLanguageModel {
        let visionConfig = try validatedVisionConfiguration()

        try bundle.bundle.verify()

        let visionURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.vision)
        let embeddingURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.embedding)
        let decoderURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        let functionName = bundle.language.functionMap?.name(for: "main") ?? "main"
        let baseConfig = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [decoderURL.path],
            function: functionName
        )
        let configuration = VLMModelConfig(base: baseConfig, visionConfig: visionConfig)

        let visionModel = try await PreparedModel.prepare(at: visionURL)
        let embeddingModel = try await PreparedModel.prepare(at: embeddingURL)
        let decoderModel = try await PreparedModel.prepare(at: decoderURL)
        let engine = try await CoreAISequentialVLMEngine(
            config: configuration,
            visionModel: visionModel,
            embedModel: embeddingModel,
            llmModel: decoderModel,
            options: EngineOptions(kvCacheStrategy: kvCacheStrategy)
        )
        let tokenizer = try await bundle.loadTokenizer()

        var stopTokenIDs = Set<Int32>()
        if let eosTokenID = tokenizer.eosTokenId {
            stopTokenIDs.insert(Int32(eosTokenID))
        }

        return SwiftLMVisionLanguageModel(
            engine: engine,
            tokenizer: tokenizer,
            configuration: SwiftLMVisionConfiguration(visionConfig),
            maxContextLength: bundle.maxContextLength,
            stopTokenIDs: stopTokenIDs
        )
    }

    private func validatedVisionConfiguration() throws -> VisionConfig {
        guard let configuration = bundle.visionConfig else {
            throw SwiftLMVisionLanguageModelError.languageModelDoesNotSupportVision
        }
        guard configuration.imageSize > 0 else {
            throw invalidVisionConfiguration(
                field: "image_size", reason: "must be greater than zero")
        }
        guard configuration.patchSize > 0 else {
            throw invalidVisionConfiguration(
                field: "patch_size", reason: "must be greater than zero")
        }
        guard configuration.imageSize.isMultiple(of: configuration.patchSize) else {
            throw invalidVisionConfiguration(
                field: "patch_size",
                reason: "must divide image_size exactly"
            )
        }
        guard configuration.imageTokenCount > 0 else {
            throw invalidVisionConfiguration(
                field: "image_token_count",
                reason: "must be greater than zero"
            )
        }
        guard configuration.imageTokenId >= 0 else {
            throw invalidVisionConfiguration(
                field: "image_token_id", reason: "must not be negative")
        }
        guard configuration.imageMean.count == 3 else {
            throw invalidVisionConfiguration(
                field: "image_mean", reason: "must contain three RGB values")
        }
        guard configuration.imageStd.count == 3 else {
            throw invalidVisionConfiguration(
                field: "image_std", reason: "must contain three RGB values")
        }
        guard configuration.imageMean.allSatisfy(\.isFinite) else {
            throw invalidVisionConfiguration(field: "image_mean", reason: "values must be finite")
        }
        guard configuration.imageStd.allSatisfy({ $0.isFinite && $0 != 0 }) else {
            throw invalidVisionConfiguration(
                field: "image_std",
                reason: "values must be finite and nonzero"
            )
        }
        guard configuration.rescaleFactor.isFinite else {
            throw invalidVisionConfiguration(field: "rescale_factor", reason: "must be finite")
        }
        return configuration
    }

    private func invalidVisionConfiguration(
        field: String,
        reason: String
    ) -> SwiftLMVisionLanguageModelError {
        .invalidVisionConfiguration(field: field, reason: reason)
    }
}
