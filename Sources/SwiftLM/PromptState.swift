import MetalCompiler

/// A reusable snapshot of decode state for a prepared prompt prefix.
///
/// Build a prompt state with ``ModelContainer/makePromptState(input:)-(LMInput)`` or
/// ``ModelContainer/makePromptState(input:)-(UserInput)`` and reuse it with
/// ``ModelContainer/generate(from:parameters:)``.
public struct PromptState: @unchecked Sendable {
    let metalState: MetalPromptState
    /// Number of tokens in the prompt prefix used to create this state.
    public let promptTokenCount: Int

    init(metalState: MetalPromptState, promptTokenCount: Int) {
        self.metalState = metalState
        self.promptTokenCount = promptTokenCount
    }
}

/// Errors produced by ``ModelContainer``.
public enum ModelContainerError: Error {
    /// Prefill did not produce a valid first token.
    case invalidPrefillResult
}
