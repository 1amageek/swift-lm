import Foundation

/// Errors raised by the strict Core AI VLM adapter.
@available(macOS 27.0, iOS 27.0, *)
public enum SwiftLMVisionLanguageModelError: Error, LocalizedError, Sendable, Equatable {
    case languageModelDoesNotSupportVision
    case visionLanguageModelRequiresVisionAPI
    case invalidVisionConfiguration(field: String, reason: String)
    case invalidMaximumTokenCount(Int)
    case imageTokenUnavailable(Int32)
    case chatTemplateFailed(String)
    case invalidImagePlaceholderCount(expected: Int, actual: Int)
    case contextLengthExceeded(maximum: Int, requested: Int)
    case generationEndedWithoutReason

    public var errorDescription: String? {
        switch self {
        case .languageModelDoesNotSupportVision:
            "The bundle is a text-only language model."
        case .visionLanguageModelRequiresVisionAPI:
            "The bundle contains vision assets and must be loaded with makeVisionLanguageModel()."
        case .invalidVisionConfiguration(let field, let reason):
            "The VLM bundle has an invalid \(field) value: \(reason)."
        case .invalidMaximumTokenCount(let count):
            "Maximum token count must be greater than zero; received \(count)."
        case .imageTokenUnavailable(let tokenID):
            "The tokenizer cannot resolve image token ID \(tokenID)."
        case .chatTemplateFailed(let reason):
            "The tokenizer chat template could not render the VLM prompt: \(reason)"
        case .invalidImagePlaceholderCount(let expected, let actual):
            "The VLM prompt requires \(expected) image placeholder tokens but contains \(actual)."
        case .contextLengthExceeded(let maximum, let requested):
            "The request needs \(requested) tokens but the model context limit is \(maximum)."
        case .generationEndedWithoutReason:
            "Core AI generation ended without reporting a stop reason."
        }
    }
}
