import CoreAILanguageModels
import Foundation

/// Completed output from Core AI VLM generation.
@available(macOS 27.0, iOS 27.0, *)
public struct SwiftLMVisionLanguageOutput: Sendable, Equatable {
    public let text: String
    public let tokenIDs: [Int32]
    public let stopReason: StopReason

    public init(text: String, tokenIDs: [Int32], stopReason: StopReason) {
        self.text = text
        self.tokenIDs = tokenIDs
        self.stopReason = stopReason
    }
}
