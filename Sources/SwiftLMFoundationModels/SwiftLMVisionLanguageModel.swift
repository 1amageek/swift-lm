import CoreAILanguageModels
import Foundation
import Tokenizers

/// Stateful Swift interface over Apple's Core AI sequential VLM engine.
@available(macOS 27.0, iOS 27.0, *)
public actor SwiftLMVisionLanguageModel: SwiftLMVisionLanguageGenerating {
    public nonisolated let configuration: SwiftLMVisionConfiguration

    private let engine: CoreAISequentialVLMEngine
    private let tokenizer: any Tokenizer
    private let maxContextLength: Int
    private let stopTokenIDs: Set<Int32>

    init(
        engine: CoreAISequentialVLMEngine,
        tokenizer: any Tokenizer,
        configuration: SwiftLMVisionConfiguration,
        maxContextLength: Int,
        stopTokenIDs: Set<Int32>
    ) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.maxContextLength = maxContextLength
        self.stopTokenIDs = stopTokenIDs
    }

    public func generate(
        from input: SwiftLMVisionLanguageInput,
        options: SwiftLMVisionLanguageGenerationOptions = SwiftLMVisionLanguageGenerationOptions()
    ) async throws -> SwiftLMVisionLanguageOutput {
        guard options.maxTokens > 0 else {
            throw SwiftLMVisionLanguageModelError.invalidMaximumTokenCount(options.maxTokens)
        }

        let embeddedInput = try await engine.encodeImage(at: input.imageURL)
        guard embeddedInput.tokenCount == configuration.imageTokenCount else {
            throw SwiftLMVisionLanguageModelError.invalidImagePlaceholderCount(
                expected: configuration.imageTokenCount,
                actual: embeddedInput.tokenCount
            )
        }

        let promptTokenIDs = try makePromptTokenIDs(from: input.prompt)
        let requestedTokenCount = promptTokenIDs.count + options.maxTokens
        guard requestedTokenCount <= maxContextLength else {
            throw SwiftLMVisionLanguageModelError.contextLengthExceeded(
                maximum: maxContextLength,
                requested: requestedTokenCount
            )
        }

        let sequence = try await engine.generate(
            with: embeddedInput,
            tokens: promptTokenIDs,
            samplingConfiguration: options.samplingConfiguration,
            inferenceOptions: InferenceOptions(maxTokens: options.maxTokens)
        )

        var generatedTokenIDs: [Int32] = []
        for try await output in sequence {
            if stopTokenIDs.contains(output.tokenId)
                || options.additionalStopTokenIDs.contains(output.tokenId)
            {
                sequence.setStopReason(.eos)
                break
            }
            generatedTokenIDs.append(output.tokenId)
        }

        guard let stopReason = sequence.stopReason else {
            throw SwiftLMVisionLanguageModelError.generationEndedWithoutReason
        }
        return SwiftLMVisionLanguageOutput(
            text: tokenizer.decode(tokens: generatedTokenIDs.map(Int.init)),
            tokenIDs: generatedTokenIDs,
            stopReason: stopReason
        )
    }

    public func reset() async throws {
        try await engine.reset()
    }

    public func cancel() async throws {
        try await engine.cancel()
    }

    private func makePromptTokenIDs(
        from prompt: SwiftLMVisionLanguagePrompt
    ) throws -> [Int32] {
        switch prompt {
        case .text(let text):
            return try makeTemplatedPromptTokenIDs(text: text)
        case .tokens(let tokenIDs):
            try promptTokenExpander.validatePretokenizedTokenIDs(tokenIDs)
            return tokenIDs
        }
    }

    private func makeTemplatedPromptTokenIDs(text: String) throws -> [Int32] {
        guard let imageToken = tokenizer.convertIdToToken(Int(configuration.imageTokenID)) else {
            throw SwiftLMVisionLanguageModelError.imageTokenUnavailable(configuration.imageTokenID)
        }

        let renderedTokenIDs: [Int]
        do {
            renderedTokenIDs = try PromptUtils.maybeApplyTokenizerChatTemplate(
                .prompt("\(imageToken)\n\(text)"),
                tokenizer: tokenizer
            )
        } catch {
            throw SwiftLMVisionLanguageModelError.chatTemplateFailed(String(describing: error))
        }

        return try promptTokenExpander.expandTemplatedTokenIDs(
            renderedTokenIDs.map(Int32.init)
        )
    }

    private var promptTokenExpander: SwiftLMVisionPromptTokenExpander {
        SwiftLMVisionPromptTokenExpander(
            imageTokenID: configuration.imageTokenID,
            imageTokenCount: configuration.imageTokenCount
        )
    }
}
