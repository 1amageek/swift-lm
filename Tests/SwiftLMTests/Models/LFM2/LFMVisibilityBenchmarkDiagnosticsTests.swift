import Foundation
import Testing
@testable import SwiftLM

@Suite("LFM Visibility Benchmark Diagnostics", .serialized)
struct LFMVisibilityBenchmarkDiagnosticsTests {
    @Test("benchmark prompt reports raw-to-visible expansion", .timeLimit(.minutes(10)))
    func benchmarkPromptReportsRawToVisibleExpansion() async throws {
        guard let localModelDirectory = try Self.optionalLFMThinkingDirectory() else {
            print("[Skip] No local LFM bundle found")
            return
        }

        let loaded = try await ModelBundleLoader().load(directory: localModelDirectory)
        let container = try LanguageModelContext(loaded)
        let prompt = ExecutablePrompt(tokenIDs: [1, 1, 6, 6423, 708])
        let parameters = GenerationParameters(maxTokens: 50, streamChunkTokenCount: 8, temperature: 0)

        container.resetState()
        let visibleTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetState()
        let rawTokenIDs = try container.debugRawGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        let visibleText = container.decode( visibleTokenIDs)
        let rawText = container.tokenizer.decode(tokens: rawTokenIDs, skipSpecialTokens: false)
        let expansion = Double(rawTokenIDs.count) / Double(max(visibleTokenIDs.count, 1))

        print("[LFM benchmark visible token count] \(visibleTokenIDs.count)")
        print("[LFM benchmark raw token count] \(rawTokenIDs.count)")
        print(String(format: "[LFM benchmark raw/visible ratio] %.2f", expansion))
        print("[LFM benchmark visible text prefix]")
        print(String(visibleText.prefix(200)))
        print("[LFM benchmark raw text prefix]")
        print(String(rawText.prefix(400)))

        #expect(!visibleTokenIDs.isEmpty)
        #expect(!rawTokenIDs.isEmpty)
        #expect(rawTokenIDs.count >= visibleTokenIDs.count)
        #expect(!visibleText.contains("<think>"))
    }

    @Test("hello prompt reports raw-to-visible expansion", .timeLimit(.minutes(10)))
    func helloPromptReportsRawToVisibleExpansion() async throws {
        guard let localModelDirectory = try Self.optionalLFMThinkingDirectory() else {
            print("[Skip] No local LFM bundle found")
            return
        }

        let loaded = try await ModelBundleLoader().load(directory: localModelDirectory)
        let container = try LanguageModelContext(loaded)
        let prepared = try await container.prepare( ModelInput(prompt: "Hello"))
        let prompt = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let parameters = GenerationParameters(maxTokens: 4, streamChunkTokenCount: 1, temperature: 0)

        container.resetState()
        let visibleTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetState()
        let rawTokenIDs = try container.debugRawGeneratedTokenIDs(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 1024, streamChunkTokenCount: 1, temperature: 0)
        )

        let visibleText = container.decode( visibleTokenIDs)
        let rawText = container.tokenizer.decode(tokens: rawTokenIDs, skipSpecialTokens: false)
        let expansion = Double(rawTokenIDs.count) / Double(max(visibleTokenIDs.count, 1))

        print("[LFM hello visible token count] \(visibleTokenIDs.count)")
        print("[LFM hello raw token count] \(rawTokenIDs.count)")
        print(String(format: "[LFM hello raw/visible ratio] %.2f", expansion))
        print("[LFM hello visible text]")
        print(visibleText)
        print("[LFM hello raw text prefix]")
        print(String(rawText.prefix(400)))

        #expect(!rawTokenIDs.isEmpty)
        #expect(rawTokenIDs.count >= visibleTokenIDs.count)
        #expect(!visibleText.contains("<think>"))
    }

    @Test("thinking opt-in surfaces reasoning in streaming and visible token output", .timeLimit(.minutes(10)))
    func thinkingOptInSurfacesReasoningInStreamingAndVisibleOutput() async throws {
        guard let localModelDirectory = try Self.optionalLFMThinkingDirectory() else {
            print("[Skip] No local LFM bundle found")
            return
        }

        let loaded = try await ModelBundleLoader().load(directory: localModelDirectory)
        let container = try LanguageModelContext(loaded)
        let parameters = GenerationParameters(
            maxTokens: 64,
            streamChunkTokenCount: 1,
            temperature: 0,
            reasoning: .inline
        )
        let prepared = try await container.prepare( ModelInput(
            prompt: "Hello",
            promptOptions: PromptPreparationOptions(isThinkingEnabled: true)
        ))
        let prompt = try ExecutablePrompt(preparedPrompt: prepared, using: container)

        container.resetState()
        let visibleTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetState()
        let rawTokenIDs = try container.debugRawGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetState()
        let stream = try container.generate(from: prompt, parameters: parameters)
        let streamed = await QwenVisionTestSupport.collectGeneration(from: stream)

        let visibleText = container.tokenizer.decode(tokens: visibleTokenIDs, skipSpecialTokens: false)
        let rawText = container.tokenizer.decode(tokens: rawTokenIDs, skipSpecialTokens: false)
        let streamedText = streamed.chunks.joined()

        print("[LFM thinking opt-in visible text prefix]")
        print(String(visibleText.prefix(400)))
        print("[LFM thinking opt-in streamed text prefix]")
        print(String(streamedText.prefix(400)))

        #expect(!rawTokenIDs.isEmpty)
        #expect(visibleTokenIDs == rawTokenIDs)
        #expect(rawText.contains("<think>"))
        #expect(visibleText.contains("<think>"))
        #expect(streamedText.contains("<think>"))
    }

    @Test("thinking separate emits reasoning stream for LFM template semantics", .timeLimit(.minutes(10)))
    func thinkingSeparateEmitsReasoningStream() async throws {
        guard let localModelDirectory = try Self.optionalLFMThinkingDirectory() else {
            print("[Skip] No local LFM bundle found")
            return
        }

        let loaded = try await ModelBundleLoader().load(directory: localModelDirectory)
        let container = try LanguageModelContext(loaded)
        let parameters = GenerationParameters(
            maxTokens: 64,
            streamChunkTokenCount: 1,
            temperature: 0,
            reasoning: .separate
        )
        let prepared = try await container.prepare( ModelInput(
            prompt: "Hello",
            promptOptions: PromptPreparationOptions(isThinkingEnabled: true)
        ))
        let prompt = try ExecutablePrompt(preparedPrompt: prepared, using: container)

        container.resetState()
        let stream = try container.generate(from: prompt, parameters: parameters)

        var answer = ""
        var reasoning = ""
        for await generation in stream {
            switch generation {
            case .text(let chunk):
                answer += chunk
            case .reasoning(let chunk):
                reasoning += chunk
            case .completed:
                break
            }
        }

        print("[LFM separate reasoning prefix]")
        print(String(reasoning.prefix(400)))
        print("[LFM separate answer prefix]")
        print(String(answer.prefix(400)))

        #expect(!reasoning.isEmpty)
        #expect(!answer.contains("<think>"))
        #expect(!reasoning.contains("<think>"))
        #expect(!reasoning.contains("</think>"))
    }

    @Test("public generate E2E emits reasoning stream for LFM template semantics", .timeLimit(.minutes(10)))
    func publicGenerateE2EEmitsReasoningStream() async throws {
        guard let localModelDirectory = try Self.optionalLFMThinkingDirectory() else {
            print("[Skip] No local LFM bundle found")
            return
        }

        let loaded = try await ModelBundleLoader().load(directory: localModelDirectory)
        let container = try LanguageModelContext(loaded)
        let parameters = GenerationParameters(
            maxTokens: 64,
            streamChunkTokenCount: 1,
            temperature: 0,
            reasoning: .separate
        )

        container.resetState()
        let stream = try await container.generate(
            ModelInput(
                prompt: "Hello",
                promptOptions: PromptPreparationOptions(isThinkingEnabled: true)
            ),
            parameters: parameters
        )

        var answer = ""
        var reasoning = ""
        var eventKinds: [String] = []
        for await generation in stream {
            switch generation {
            case .text(let chunk):
                answer += chunk
                eventKinds.append("text")
            case .reasoning(let chunk):
                reasoning += chunk
                eventKinds.append("reasoning")
            case .completed:
                break
            }
        }

        print("[LFM E2E separate event kinds]")
        print(eventKinds.joined(separator: ","))
        print("[LFM E2E separate reasoning prefix]")
        print(String(reasoning.prefix(400)))
        print("[LFM E2E separate answer prefix]")
        print(String(answer.prefix(400)))

        #expect(eventKinds.contains("reasoning"))
        #expect(!reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!answer.contains("<think>"))
        #expect(!answer.contains("</think>"))
        #expect(!reasoning.contains("<think>"))
        #expect(!reasoning.contains("</think>"))
    }

    private static func optionalLFMThinkingDirectory() throws -> URL? {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let directCandidates = [
            repositoryRoot.appendingPathComponent("TestData/LFM2.5-1.2B-Thinking"),
            URL(fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking")
        ]
        for candidate in directCandidates where try isUsableModelDirectory(candidate) {
            return candidate
        }

        let envCandidates = [
            ProcessInfo.processInfo.environment["SWIFTLM_LFM_THINKING_DIR"],
            ProcessInfo.processInfo.environment["SWIFTLM_LFM25_THINKING_DIR"]
        ].compactMap { $0 }
        for candidate in envCandidates {
            let url = URL(fileURLWithPath: NSString(string: candidate).expandingTildeInPath)
            if try isUsableModelDirectory(url) {
                return url
            }
        }

        let cacheRoot = URL(
            fileURLWithPath: NSString(
                string: "~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Thinking"
            ).expandingTildeInPath
        )
        let snapshots = try snapshotDirectories(baseURL: cacheRoot)
        for snapshot in snapshots where try isUsableModelDirectory(snapshot) {
            return snapshot
        }
        return nil
    }

    private static func snapshotDirectories(baseURL: URL) throws -> [URL] {
        let snapshotsURL = baseURL.appendingPathComponent("snapshots")
        guard FileManager.default.fileExists(atPath: snapshotsURL.path) else {
            return []
        }
        return try FileManager.default.contentsOfDirectory(
            at: snapshotsURL,
            includingPropertiesForKeys: nil
        )
    }

    private static func isUsableModelDirectory(_ directory: URL) throws -> Bool {
        let configURL = directory.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: configURL.path)
    }
}
