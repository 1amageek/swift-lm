import Foundation

/// Errors raised while validating and specializing Core AI assets.
@available(macOS 27.0, iOS 27.0, *)
public enum CoreAIModelAssetError: Error, LocalizedError, Sendable, Equatable {
    case invalidAsset(URL)
    case missingSummary(URL)
    case functionNotFound(String)
    case stateNotFound(function: String, state: String)
    case outputUnavailable(String)
    case unsupportedStateCount(Int)
    case unsupportedSpecializationOption(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let url):
            return "Invalid Core AI model asset: \(url.path)"
        case .missingSummary(let url):
            return "Core AI model asset has no summary: \(url.path)"
        case .functionNotFound(let name):
            return "Core AI function not found: \(name)"
        case .stateNotFound(let function, let state):
            return "Core AI state '\(state)' not found in function '\(function)'"
        case .outputUnavailable(let name):
            return "Core AI output is unavailable: \(name)"
        case .unsupportedStateCount(let count):
            return "Core AI state session supports at most four tensor states, got \(count)"
        case .unsupportedSpecializationOption(let message):
            return "Unsupported Core AI specialization option: \(message)"
        }
    }
}
