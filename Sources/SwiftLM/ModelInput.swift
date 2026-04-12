/// User-facing input for generation.
///
/// `ModelInput` is the primary public input type for `SwiftLM`.
/// It can represent plain text prompts and chat-style prompts, and it is
/// intentionally shaped so multimodal content can be introduced without
/// replacing the public API again.
public struct ModelInput: Sendable {
    /// The prompt payload.
    public var prompt: Prompt
    /// Prompt/render-time options used before token generation begins.
    public var promptOptions: PromptPreparationOptions
    /// Tool definitions available to the model for this request.
    public var tools: [ToolDefinition]?

    public init(_ prompt: String, promptOptions: PromptPreparationOptions = PromptPreparationOptions()) {
        self.prompt = .text(prompt)
        self.promptOptions = promptOptions
    }

    public init(prompt: String, promptOptions: PromptPreparationOptions = PromptPreparationOptions()) {
        self.prompt = .text(prompt)
        self.promptOptions = promptOptions
    }

    public init(
        chat: [InputMessage],
        tools: [ToolDefinition]? = nil,
        promptOptions: PromptPreparationOptions = PromptPreparationOptions()
    ) {
        self.prompt = .chat(chat)
        self.tools = tools
        self.promptOptions = promptOptions
    }

    /// Prompt representation.
    public enum Prompt: Sendable {
        case text(String)
        case chat([InputMessage])
    }
}
