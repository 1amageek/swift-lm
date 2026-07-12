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

    public func makeLanguageModel(
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto
    ) async throws -> CoreAILanguageModel {
        try await CoreAILanguageModel(
            resourcesAt: bundle.bundlePath,
            variant: variant,
            kvCacheStrategy: kvCacheStrategy
        )
    }
}
