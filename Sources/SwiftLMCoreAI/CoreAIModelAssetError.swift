import Foundation

/// Errors raised while validating and specializing Core AI assets.
@available(macOS 27.0, iOS 27.0, *)
public enum CoreAIModelAssetError: Error, LocalizedError, Sendable, Equatable {
    case invalidAsset(URL)
    case missingSummary(URL)
    case functionNotFound(String)
    case stateNotFound(function: String, state: String)
    case missingDynamicStateShape(String)
    case missingDynamicOutputShape(String)
    case invalidStateShape(function: String, state: String, expected: [Int], provided: [Int])
    case invalidOutputShape(function: String, output: String, expected: [Int], provided: [Int])
    case unsupportedOutputCount(Int)
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
        case .missingDynamicStateShape(let state):
            return "Core AI dynamic state shape is required: \(state)"
        case .missingDynamicOutputShape(let output):
            return "Core AI dynamic output shape is required: \(output)"
        case .invalidStateShape(let function, let state, let expected, let provided):
            return "Invalid Core AI state shape for '\(function).\(state)': expected \(expected), got \(provided)"
        case .invalidOutputShape(let function, let output, let expected, let provided):
            return "Invalid Core AI output shape for '\(function).\(output)': expected \(expected), got \(provided)"
        case .unsupportedOutputCount(let count):
            return "Core AI state session supports at most one NDArray output, got \(count)"
        case .outputUnavailable(let name):
            return "Core AI output is unavailable: \(name)"
        case .unsupportedStateCount(let count):
            return "Core AI state session supports at most four tensor states, got \(count)"
        case .unsupportedSpecializationOption(let message):
            return "Unsupported Core AI specialization option: \(message)"
        }
    }
}
