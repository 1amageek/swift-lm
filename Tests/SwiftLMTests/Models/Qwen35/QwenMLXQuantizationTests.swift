import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen3.5 MLX Quantization Variants", .serialized)
struct QwenMLXQuantizationTests {
    @Test("Qwen3.5-0.8B-MLX-bf16 generates non-degenerate text", .timeLimit(.minutes(10)))
    func qwen35_0_8B_bf16() async throws {
        try await runGreedyVariant(name: "Qwen3.5-0.8B-MLX-bf16")
    }

    @Test("Qwen3.5-0.8B-4bit generates non-degenerate text", .timeLimit(.minutes(10)))
    func qwen35_0_8B_4bit() async throws {
        try await runGreedyVariant(name: "Qwen3.5-0.8B-4bit")
    }

    @Test("Qwen3.5-0.8B-3bit generates non-degenerate text", .timeLimit(.minutes(10)))
    func qwen35_0_8B_3bit() async throws {
        try await runGreedyVariant(name: "Qwen3.5-0.8B-3bit")
    }

    @Test("Qwen3.5-4B-MLX-4bit generates non-degenerate text", .timeLimit(.minutes(10)))
    func qwen35_4B_4bit() async throws {
        try await runGreedyVariant(name: "Qwen3.5-4B-MLX-4bit")
    }

    private func runGreedyVariant(name: String) async throws {
        guard let directory = try QwenVisionTestSupport.optionalQwen35MLXBundleDirectory(name: name) else {
            print("[Skip] No local \(name) bundle found")
            return
        }
        let loaded = try await ModelBundleLoader().load(directory: directory)
        let container = try LanguageModelContext(loaded)

        container.resetState()
        let prepared = try await container.prepare(ModelInput(
            chat: [.user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])],
            promptOptions: PromptPreparationOptions(isThinkingEnabled: false)
        ))
        let prompt = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        container.resetState()
        let tokenIDs = try container.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 16)
        )
        let text = container.decode(tokenIDs)
        let normalized = RealOutputAssertionSupport.normalized(text)
        print("[\(name) token ids] \(tokenIDs)")
        print("[\(name) text] \(String(text.prefix(200)))")
        print("[\(name) normalized] \(normalized)")

        #expect(!tokenIDs.isEmpty, "[\(name)] expected at least one generated token")
        let uniqueIDs = Set(tokenIDs)
        #expect(
            uniqueIDs.count > 1,
            "[\(name)] degenerate output: only \(uniqueIDs.count) unique token id(s) — \(tokenIDs)"
        )
        #expect(
            !Self.hasLongCharacterRun(text, threshold: 8),
            "[\(name)] degenerate character run detected — \(String(text.prefix(80)))"
        )
        #expect(
            !text.contains("!!!!!!!!"),
            "[\(name)] '!!!!' degeneracy detected — \(String(text.prefix(80)))"
        )
        #expect(!text.contains("<think>"))
        #expect(!text.contains("</think>"))
        #expect(
            normalized.hasPrefix("Tokyo"),
            "[\(name)] expected response to start with 'Tokyo' for the strict capital prompt — got '\(normalized.prefix(80))'"
        )
    }

    private static func hasLongCharacterRun(_ text: String, threshold: Int) -> Bool {
        guard threshold > 1, !text.isEmpty else { return false }
        var previous: Character? = nil
        var run = 0
        for character in text {
            if character == previous {
                run += 1
                if run >= threshold {
                    return true
                }
            } else {
                run = 1
                previous = character
            }
        }
        return false
    }
}
