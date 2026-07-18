import Foundation

/// Image and prompt input for Core AI VLM generation.
@available(macOS 27.0, iOS 27.0, *)
public struct SwiftLMVisionLanguageInput: Sendable, Equatable {
    public let imageURL: URL
    public let prompt: SwiftLMVisionLanguagePrompt

    public init(imageURL: URL, prompt: SwiftLMVisionLanguagePrompt) {
        self.imageURL = imageURL
        self.prompt = prompt
    }
}
