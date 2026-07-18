import Foundation

/// Prompt input for a Core AI vision-language model.
@available(macOS 27.0, iOS 27.0, *)
public enum SwiftLMVisionLanguagePrompt: Sendable, Equatable {
    /// Text rendered through the tokenizer's chat template.
    case text(String)

    /// Fully rendered token IDs containing exactly the required image placeholders.
    case tokens([Int32])
}
