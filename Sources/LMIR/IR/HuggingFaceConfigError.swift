import Foundation

/// Errors raised while normalizing a Hugging Face model configuration.
public enum HuggingFaceConfigError: Error, Sendable, CustomStringConvertible {
    case configReadFailed(URL, String)
    case invalidJSON(String)
    case missingField(String)
    case invalidValue(String)

    public var description: String {
        switch self {
        case .configReadFailed(let url, let message):
            return "Could not read Hugging Face config at \(url.path): \(message)"
        case .invalidJSON(let message):
            return "Invalid Hugging Face config JSON: \(message)"
        case .missingField(let field):
            return "Missing required Hugging Face config field: \(field)"
        case .invalidValue(let message):
            return "Invalid Hugging Face config value: \(message)"
        }
    }
}
