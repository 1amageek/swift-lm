import Testing
@testable import SwiftLM

@Suite("LFM2.5 8B-A1B Real Bundle", .serialized)
struct LFM25A1BRealBundleTests {
    @Test("Local LFM2.5 8B-A1B loads and prepares text", .timeLimit(.minutes(10)))
    func localLFM25A1BLoadsAndPreparesText() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let prepared = try await container.prepare(ModelInput(prompt: "hi"))

        #expect(container.configuration.name.lowercased() == "lfm2_moe")
        #expect(!container.configuration.inputCapabilities.supportsImages)
        #expect(container.configuration.executionCapabilities.supportsTextGeneration)
        #expect(!prepared.tokenIDs.isEmpty)
    }

    @Test("Local LFM2.5 8B-A1B prepares text/chat inputs and rejects images", .timeLimit(.minutes(10)))
    func localLFM25A1BPreparesPromptVariantsAndRejectsImages() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let directPrompt = try await container.prepare(ModelInput(prompt: "hi"))
        let chatPrompt = try await container.prepare(ModelInput(chat: [
            .user([.text("Say hello in one short sentence.")])
        ]))

        #expect(!directPrompt.tokenIDs.isEmpty)
        #expect(!chatPrompt.tokenIDs.isEmpty)
        #expect(chatPrompt.tokenIDs.count >= directPrompt.tokenIDs.count)

        do {
            _ = try await container.prepare(ModelInput(chat: [
                .user([
                    .text("Describe this image."),
                    .image(InputImage(data: try TestImageFixtures.makeOnePixelPNGData(), mimeType: "image/png")),
                ])
            ]))
            Issue.record("Expected LFM2.5 8B-A1B to reject image-bearing input")
        } catch LanguageModelContextError.unsupportedInputForModel {
        }
    }

    @Test("Local LFM2.5 8B-A1B emits one greedy token", .timeLimit(.minutes(10)))
    func localLFM25A1BEmitsOneGreedyToken() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        let prepared = try await context.prepare(ModelInput(prompt: "hi"))
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
        let tokenIDs = try context.debugRawGeneratedTokenIDs(
            prompt: executable,
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 1)
        )

        print("[LFM2.5 8B-A1B first greedy token ids] \(tokenIDs)")
        print("[LFM2.5 8B-A1B first greedy token text] \(context.decode(tokenIDs, skipSpecialTokens: false))")

        #expect(tokenIDs.count == 1)
        #expect(!container.configuration.eosTokenIds.contains(tokenIDs[0]))
    }
}
