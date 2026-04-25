import Foundation
import Testing
@testable import SwiftLM

@Suite("LFM Output Diagnostics", .serialized)
struct LFMOutputDiagnosticsTests {
    @Test("Inspect LFM hi chat 256 output", .timeLimit(.minutes(10)))
    func inspectLFMHiChat256Output() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loaded = try await ModelBundleLoader().load(directory: localModelDirectory)
        let container = try LanguageModelContext(loaded)
        container.resetState()
        let prepared = try await container.prepare(ModelInput(
            prompt: "hi",
            promptOptions: PromptPreparationOptions(isThinkingEnabled: true)
        ))
        let prompt = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let tokenIDs = try container.debugRawGeneratedTokenIDs(
            prompt: prompt,
            parameters: GenerationParameters(
                maxTokens: 256,
                streamChunkTokenCount: 1,
                temperature: 0
            )
        )
        let rawText = container.tokenizer.decode(tokens: tokenIDs, skipSpecialTokens: false)
        let visibleText = visibleTextDroppingLeadingThinkingBlock(from: rawText)
        let expectedPrefix = [64400, 9095, 892, 521, 2944, 1090, 2130, 1620, 779]

        print("[LFM hi 256 prepared]")
        print(prepared.renderedText)
        print("[LFM hi 256 prepared token ids] \(prepared.tokenIDs)")
        print("[LFM hi 256 generated token ids]")
        print(tokenIDs)
        print("[LFM hi 256 generated raw text]")
        print(rawText)
        print("[LFM hi 256 generated visible text]")
        print(visibleText)

        #expect(Array(tokenIDs.prefix(expectedPrefix.count)) == expectedPrefix)
        #expect(rawText.contains("<think>"), "Thinking model should enter a thinking block for this reference prompt")
        #expect(rawText.contains("</think>"), "Thinking model must close the thinking block within 256 greedy tokens")
        #expect(!visibleText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "Thinking output must include a final answer, not only reasoning")
        #expect(!visibleText.contains("<think>"))
        #expect(!visibleText.contains("</think>"))
    }

    @Test("Inspect LFM real output", .timeLimit(.minutes(10)))
    func inspectLFMRealOutput() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loaded = try await ModelBundleLoader().load(directory: localModelDirectory)
        let container = try LanguageModelContext(loaded)
        container.resetState()
        let prepared = try await container.prepare( ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        print("[LFM prepared]")
        print(prepared.renderedText)
        print("[LFM prepared token count] \(prepared.tokenIDs.count)")

        let prompt = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let topLogits = try container.debugPrefillTopLogits(prompt: prompt, topK: 20)
        print("[LFM prefill top logits]")
        for entry in topLogits {
            let formattedLogit = String(format: "%.4f", entry.logit)
            print("  id=\(entry.tokenID) logit=\(formattedLogit) token=\(String(reflecting: entry.decoded))")
        }

        let tokenIDs = try container.debugRawGeneratedTokenIDs(
            prompt: prompt,
            parameters: GenerationParameters(
                maxTokens: 64,
                streamChunkTokenCount: 8,
                temperature: 0
            )
        )
        let rawText = container.tokenizer.decode(tokens: tokenIDs, skipSpecialTokens: false)
        let visibleText = visibleTextDroppingLeadingThinkingBlock(from: rawText)

        print("[LFM generated token ids]")
        print(tokenIDs)
        print("[LFM generated raw text]")
        print(rawText)
        print("[LFM generated visible text]")
        print(visibleText)
    }

    @Test("Compare LFM direct and chat prompts", .timeLimit(.minutes(10)))
    func compareLFMDirectAndChatPrompts() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loaded = try await ModelBundleLoader().load(directory: localModelDirectory)
        let container = try LanguageModelContext(loaded)
        let parameters = GenerationParameters(
            maxTokens: 16,
            streamChunkTokenCount: 1,
            temperature: 0
        )

        let cases: [(String, PreparedPrompt)] = [
            (
                "short direct",
                RealOutputAssertionSupport.directTextPrompt(
                    RealOutputAssertionSupport.capitalCompletionPrompt,
                    using: container
                )
            ),
            (
                "long direct",
                RealOutputAssertionSupport.directTextPrompt(
                    "Question: What is the capital of Japan? Answer with exactly one word. Answer:",
                    using: container
                )
            ),
            (
                "chat",
                try await container.prepare(
                    ModelInput(
                        prompt: RealOutputAssertionSupport.strictCapitalPrompt,
                        promptOptions: PromptPreparationOptions(isThinkingEnabled: false)
                    )
                )
            ),
            (
                "manual chat with bos",
                PreparedPrompt(
                    renderedText: "<|startoftext|><|im_start|>user\nWhat is the capital of Japan? Answer with exactly one word.<|im_end|>\n<|im_start|>assistant\n",
                    tokenIDs: container.tokenizer.encode(
                        text: "<|startoftext|><|im_start|>user\nWhat is the capital of Japan? Answer with exactly one word.<|im_end|>\n<|im_start|>assistant\n",
                        addSpecialTokens: false
                    )
                )
            ),
            (
                "manual chat no bos",
                PreparedPrompt(
                    renderedText: "<|im_start|>user\nWhat is the capital of Japan? Answer with exactly one word.<|im_end|>\n<|im_start|>assistant\n",
                    tokenIDs: container.tokenizer.encode(
                        text: "<|im_start|>user\nWhat is the capital of Japan? Answer with exactly one word.<|im_end|>\n<|im_start|>assistant\n",
                        addSpecialTokens: false
                    )
                )
            ),
            (
                "manual plain roles",
                PreparedPrompt(
                    renderedText: "user: What is the capital of Japan? Answer with exactly one word.\nassistant:",
                    tokenIDs: container.tokenizer.encode(
                        text: "user: What is the capital of Japan? Answer with exactly one word.\nassistant:",
                        addSpecialTokens: true
                    )
                )
            ),
        ]

        for (label, prepared) in cases {
            container.resetState()
            let prompt = try ExecutablePrompt(preparedPrompt: prepared, using: container)
            let ids = try container.debugRawGeneratedTokenIDs(prompt: prompt, parameters: parameters)
            let text = container.decode(ids)
            print("[LFM \(label) prompt tokens] \(prepared.tokenIDs.count)")
            print("[LFM \(label) prompt] \(prepared.renderedText)")
            print("[LFM \(label) raw ids] \(ids)")
            print("[LFM \(label) raw text] \(text)")
        }
    }

    private func visibleTextDroppingLeadingThinkingBlock(from text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("<think>") else {
            return trimmed
        }
        guard let closingRange = trimmed.range(of: "</think>") else {
            return trimmed
        }
        return trimmed[closingRange.upperBound...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
