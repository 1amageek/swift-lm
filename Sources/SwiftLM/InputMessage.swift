/// A single message in a chat-style prompt.
public struct InputMessage: Sendable {
    public var role: Role
    public var content: [Content]
    /// Tool calls requested by an assistant message.
    public var toolCalls: [ToolCall]?
    /// The ID of the tool call this message responds to (tool role only).
    public var toolCallID: String?

    public init(role: Role, content: String) {
        self.role = role
        self.content = [.text(content)]
    }

    public init(role: Role, content: [Content]) {
        self.role = role
        self.content = content
    }

    public static func system(_ content: String) -> InputMessage {
        InputMessage(role: .system, content: content)
    }

    public static func system(_ content: [Content]) -> InputMessage {
        InputMessage(role: .system, content: content)
    }

    public static func user(_ content: String) -> InputMessage {
        InputMessage(role: .user, content: content)
    }

    public static func user(_ content: [Content]) -> InputMessage {
        InputMessage(role: .user, content: content)
    }

    public static func assistant(_ content: String, toolCalls: [ToolCall]? = nil) -> InputMessage {
        var message = InputMessage(role: .assistant, content: content)
        message.toolCalls = toolCalls
        return message
    }

    public static func assistant(_ content: [Content], toolCalls: [ToolCall]? = nil) -> InputMessage {
        var message = InputMessage(role: .assistant, content: content)
        message.toolCalls = toolCalls
        return message
    }

    public static func tool(_ content: String, toolCallID: String? = nil) -> InputMessage {
        var message = InputMessage(role: .tool, content: content)
        message.toolCallID = toolCallID
        return message
    }

    public static func tool(_ content: [Content], toolCallID: String? = nil) -> InputMessage {
        var message = InputMessage(role: .tool, content: content)
        message.toolCallID = toolCallID
        return message
    }

    var containsImageContent: Bool {
        content.contains { item in
            if case .image = item {
                return true
            }
            return false
        }
    }

    var containsVideoContent: Bool {
        content.contains { item in
            if case .video = item {
                return true
            }
            return false
        }
    }

    var containsVisualContent: Bool {
        containsImageContent || containsVideoContent
    }

    var textContent: String {
        content.compactMap { item in
            if case .text(let text) = item {
                return text
            }
            return nil
        }
        .joined()
    }

    public enum Content: Sendable {
        case text(String)
        case image(InputImage)
        case video(InputVideo)
    }

    public enum Role: String, Sendable {
        case user
        case assistant
        case system
        case tool
    }

    /// A tool call emitted by the model in an assistant message.
    public struct ToolCall: Sendable {
        public var id: String
        public var function: Function

        public init(id: String, function: Function) {
            self.id = id
            self.function = function
        }

        public struct Function: Sendable {
            public var name: String
            /// JSON-encoded arguments string.
            public var arguments: String

            public init(name: String, arguments: String) {
                self.name = name
                self.arguments = arguments
            }
        }
    }
}
