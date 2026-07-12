import CoreAI
import Foundation

/// A validated Core AI source or compiled model asset.
@available(macOS 27.0, iOS 27.0, *)
public struct CoreAIModelAsset: Sendable {
    public let url: URL
    public let summary: AIModelAsset.Summary

    public init(contentsOf url: URL, includingStatistics: Bool = false) throws {
        guard AIModelAsset.isValid(at: url) else {
            throw CoreAIModelAssetError.invalidAsset(url)
        }
        let asset = try AIModelAsset(contentsOf: url)
        guard let summary = try asset.summary(includingStatistics: includingStatistics) else {
            throw CoreAIModelAssetError.missingSummary(url)
        }
        self.url = url
        self.summary = summary
    }

    public var functionNames: [String] {
        summary.functions.map(\.name)
    }

    public func function(named name: String) throws -> AIModelAsset.FunctionDescriptor {
        guard let function = summary.functions.first(where: { $0.name == name }) else {
            throw CoreAIModelAssetError.functionNotFound(name)
        }
        return function
    }

    public func specialize(
        options: SpecializationOptions = .default,
        cache: AIModelCache = .default,
        cachePolicy: AIModelCache.Policy = .default
    ) async throws -> AIModel {
        guard !options.expectFrequentReshapes else {
            throw CoreAIModelAssetError.unsupportedSpecializationOption(
                "expectFrequentReshapes is disabled until the current Core AI runtime is compatible"
            )
        }
        return try await AIModel.specialize(
            contentsOf: url,
            options: options,
            cache: cache,
            cachePolicy: cachePolicy
        )
    }
}
