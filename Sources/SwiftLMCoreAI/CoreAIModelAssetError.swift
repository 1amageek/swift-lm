import Foundation

/// Errors raised while validating and specializing Core AI assets.
@available(macOS 27.0, iOS 27.0, *)
public enum CoreAIModelAssetError: Error, LocalizedError, Sendable, Equatable {
    case invalidAsset(URL)
    case invalidBundle(URL, String)
    case missingSummary(URL)
    case functionNotFound(String)
    case inputNotFound(function: String, input: String)
    case unexpectedInput(function: String, input: String)
    case unsupportedInput(function: String, input: String)
    case invalidInputShape(function: String, input: String, expected: [Int], provided: [Int])
    case invalidInputDataType(function: String, input: String, expected: String, provided: String)
    case stateNotFound(function: String, state: String)
    case statefulFunctionRequiresStateSession(String)
    case missingDynamicStateShape(String)
    case missingDynamicOutputShape(String)
    case stateAllocationFailed(String)
    case invalidStateShape(function: String, state: String, expected: [Int], provided: [Int])
    case invalidOutputShape(function: String, output: String, expected: [Int], provided: [Int])
    case unsupportedOutputCount(Int)
    case outputUnavailable(String)
    case contractMismatch(function: String, message: String)
    case unsupportedSpecializationOption(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAsset(let url):
            return "Invalid Core AI model asset: \(url.path)"
        case .invalidBundle(let url, let message):
            return "Invalid Core AI model bundle at \(url.path): \(message)"
        case .missingSummary(let url):
            return "Core AI model asset has no summary: \(url.path)"
        case .functionNotFound(let name):
            return "Core AI function not found: \(name)"
        case .inputNotFound(let function, let input):
            return "Core AI input '\(input)' is required by function '\(function)'"
        case .unexpectedInput(let function, let input):
            return "Core AI input '\(input)' is not declared by function '\(function)'"
        case .unsupportedInput(let function, let input):
            return "Core AI input '\(function).\(input)' is not an NDArray"
        case .invalidInputShape(let function, let input, let expected, let provided):
            return "Invalid Core AI input shape for '\(function).\(input)': expected \(expected), got \(provided)"
        case .invalidInputDataType(let function, let input, let expected, let provided):
            return "Invalid Core AI input data type for '\(function).\(input)': expected \(expected), got \(provided)"
        case .stateNotFound(let function, let state):
            return "Core AI state '\(state)' not found in function '\(function)'"
        case .statefulFunctionRequiresStateSession(let function):
            return "Core AI function '\(function)' declares mutable state and requires CoreAIStateSession"
        case .missingDynamicStateShape(let state):
            return "Core AI dynamic state shape is required: \(state)"
        case .missingDynamicOutputShape(let output):
            return "Core AI dynamic output shape is required: \(output)"
        case .stateAllocationFailed(let state):
            return "Core AI could not allocate persistent storage for state '\(state)'"
        case .invalidStateShape(let function, let state, let expected, let provided):
            return "Invalid Core AI state shape for '\(function).\(state)': expected \(expected), got \(provided)"
        case .invalidOutputShape(let function, let output, let expected, let provided):
            return "Invalid Core AI output shape for '\(function).\(output)': expected \(expected), got \(provided)"
        case .unsupportedOutputCount(let count):
            return "Core AI session requires exactly one NDArray output, got \(count)"
        case .outputUnavailable(let name):
            return "Core AI output is unavailable: \(name)"
        case .contractMismatch(let function, let message):
            return "Core AI asset does not match the Swift LMIR contract for function '\(function)': \(message)"
        case .unsupportedSpecializationOption(let message):
            return "Unsupported Core AI specialization option: \(message)"
        }
    }
}
