import Foundation

/// Errors raised while converting a semantic graph into the Core AI export format.
public enum CoreAIExportError: Error, LocalizedError, Sendable, Equatable {
    case invalidConfiguration(String)
    case invalidGraph(String)
    case unsupportedPrimitive(String)
    case invalidAttributePayload(String)
    case unsupportedFormatVersion(Int)
    case serializationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "Invalid Core AI export configuration: \(message)"
        case .invalidGraph(let message):
            return "Invalid Core AI export graph: \(message)"
        case .unsupportedPrimitive(let name):
            return "Unsupported primitive for Core AI export: \(name)"
        case .invalidAttributePayload(let type):
            return "Primitive attributes cannot be represented as JSON: \(type)"
        case .unsupportedFormatVersion(let version):
            return "Unsupported Core AI export format version: \(version)"
        case .serializationFailed(let message):
            return "Core AI export serialization failed: \(message)"
        }
    }
}
