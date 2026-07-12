import Foundation

/// Target-specific settings for a Core AI export document.
public struct CoreAIExportConfiguration: Codable, Equatable, Sendable {
    public let name: String
    public let modelType: String
    public let target: CoreAIExportDocument.Target
    public let maxContextLength: Int
    public let vocabSize: Int

    public init(
        name: String,
        modelType: String,
        target: CoreAIExportDocument.Target,
        maxContextLength: Int,
        vocabSize: Int
    ) throws {
        guard !name.isEmpty else {
            throw CoreAIExportError.invalidConfiguration("name must not be empty")
        }
        guard !modelType.isEmpty else {
            throw CoreAIExportError.invalidConfiguration("modelType must not be empty")
        }
        guard maxContextLength > 0 else {
            throw CoreAIExportError.invalidConfiguration("maxContextLength must be positive")
        }
        guard vocabSize > 0 else {
            throw CoreAIExportError.invalidConfiguration("vocabSize must be positive")
        }
        self.name = name
        self.modelType = modelType
        self.target = target
        self.maxContextLength = maxContextLength
        self.vocabSize = vocabSize
    }

    public var metadata: CoreAIExportDocument.Metadata {
        CoreAIExportDocument.Metadata(
            name: name,
            modelType: modelType,
            target: target,
            maxContextLength: maxContextLength,
            vocabSize: vocabSize
        )
    }
}
