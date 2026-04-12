/// A tool definition describing a callable function available to the model.
///
/// Tool definitions are passed through to the chat template's `tools` variable.
/// The structure mirrors the OpenAI/HuggingFace function-calling convention that
/// most chat templates expect.
public struct ToolDefinition: Sendable {
    public var function: Function

    public init(function: Function) {
        self.function = function
    }

    /// A function specification within a tool definition.
    public struct Function: Sendable {
        public var name: String
        public var description: String
        /// JSON Schema for the function parameters, as a dictionary.
        public var parameters: [String: any Sendable]?

        public init(
            name: String,
            description: String,
            parameters: [String: any Sendable]? = nil
        ) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }
}
