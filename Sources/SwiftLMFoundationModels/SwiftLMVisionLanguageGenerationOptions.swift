import CoreAILanguageModels
import Foundation

/// Generation controls for Core AI VLM inference.
@available(macOS 27.0, iOS 27.0, *)
public struct SwiftLMVisionLanguageGenerationOptions: Sendable, Equatable {
    public let maxTokens: Int
    public let samplingConfiguration: SamplingConfiguration
    public let additionalStopTokenIDs: Set<Int32>

    public init(
        maxTokens: Int = 256,
        samplingConfiguration: SamplingConfiguration = .greedy,
        additionalStopTokenIDs: Set<Int32> = []
    ) {
        self.maxTokens = maxTokens
        self.samplingConfiguration = samplingConfiguration
        self.additionalStopTokenIDs = additionalStopTokenIDs
    }
}
