/// Normalized metadata needed by model graph construction and export.
public struct HuggingFaceModelConfiguration: Sendable {
    public let modelType: String
    public let modelConfig: ModelConfig
    public let maxContextLength: Int

    public init(
        modelType: String,
        modelConfig: ModelConfig,
        maxContextLength: Int
    ) {
        self.modelType = modelType
        self.modelConfig = modelConfig
        self.maxContextLength = maxContextLength
    }
}
