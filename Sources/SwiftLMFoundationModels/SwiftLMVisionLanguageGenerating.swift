import Foundation

/// High-level interface for stateful Core AI VLM generation.
@available(macOS 27.0, iOS 27.0, *)
public protocol SwiftLMVisionLanguageGenerating: Sendable {
    func generate(
        from input: SwiftLMVisionLanguageInput,
        options: SwiftLMVisionLanguageGenerationOptions
    ) async throws -> SwiftLMVisionLanguageOutput

    func reset() async throws
    func cancel() async throws
}
