import Foundation
import Metal
import Testing
@testable import MetalCompiler
@testable import SwiftLM

@Suite("LFM2.5 8B-A1B Real Bundle", .serialized)
struct LFM25A1BRealBundleTests {
    private static let hfStrictCapital64TokenIDs = [
        124_901, 207, 597, 4_695, 20_589, 34, 496, 2_992,
        355, 278, 5_205, 302, 3_888, 39, 41_774, 415,
        8_043, 734, 1_858, 2_426, 8, 2_083, 1_946, 6_119,
        415, 8_043, 734, 1_858, 20, 643, 355, 496,
        40_049, 11_053, 2_784, 3_584, 589, 734, 1_858, 22,
        23_620, 513, 1_113, 4_666, 61_049, 39, 440, 10_996,
        34, 496, 39_630, 415, 8_043, 734, 1_858, 2_426,
        1_672, 522, 1_252, 6_159, 496, 40_049, 11_053, 2_784,
    ]

    private static let hfLargestPlanet16TokenIDs = [
        124_901, 207, 597, 4_695, 20_589, 34, 496, 2_617,
        278, 6_083, 7_543, 296, 278, 19_691, 3_922, 22,
    ]

    private static let hfGoodMorningJapanese16TokenIDs = [
        124_901, 207, 597, 4_695, 20_589, 34, 496, 6_977,
        22_157, 1_683, 8_031, 310, 7_314, 22, 41_774, 1_049,
    ]

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

    @Test("Local LFM2.5 8B-A1B matches HF short trace for strict capital chat", .timeLimit(.minutes(10)))
    func localLFM25A1BMatchesHFShortTraceForStrictCapitalChat() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let expectedTokenIDs = Array(Self.hfStrictCapital64TokenIDs.prefix(16))
        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        let prepared = try await context.prepare(ModelInput(chat: [
            .user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])
        ]))
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)

        let topLogits = try context.debugPrefillTopLogits(prompt: executable, topK: 10)
        let start = CFAbsoluteTimeGetCurrent()
        let tokenIDs = try context.debugRawGeneratedTokenIDs(
            prompt: executable,
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: expectedTokenIDs.count)
        )
        let elapsedSeconds = CFAbsoluteTimeGetCurrent() - start

        print("[LFM2.5 8B-A1B strict capital short trace ids] \(tokenIDs)")
        print("[LFM2.5 8B-A1B strict capital short trace text] \(context.decode(tokenIDs, skipSpecialTokens: false))")
        print(String(format: "[LFM2.5 8B-A1B strict capital short trace time] %.3fs", elapsedSeconds))
        print("[LFM2.5 8B-A1B strict capital top logits] \(topLogits.map { ($0.tokenID, $0.logit, $0.decoded) })")

        #expect(topLogits.first?.tokenID == 124_901)
        #expect(tokenIDs == expectedTokenIDs)
        #expect(context.decode([try #require(tokenIDs.first)], skipSpecialTokens: false) == "<think>")
        #expect(elapsedSeconds < 5.0, "16-token HF trace gate should remain a bounded decode smoke")
    }

    @Test("Local LFM2.5 8B-A1B prompt-state restore preserves visible output", .timeLimit(.minutes(10)))
    func localLFM25A1BPromptStateRestorePreservesVisibleOutput() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let expectedVisibleTokenIDs = [207, 40_049, 11_053]
        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        let prepared = try await context.prepare(ModelInput(chat: [
            .user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])
        ]))
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
        let traces = try RealOutputAssertionSupport.assertGreedyDirectMatchesPromptState(
            container: context,
            prompt: executable,
            label: "LFM2.5 A1B strict capital prompt state",
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 8)
        )

        #expect(traces.directTokenIDs == expectedVisibleTokenIDs)
        #expect(traces.restoredTokenIDs == expectedVisibleTokenIDs)
        #expect(traces.directText == "Tokyo")
        #expect(traces.restoredText == "Tokyo")
    }

    @Test("Split Sparse MoE route matches HF first token and clears legacy speed gate", .timeLimit(.minutes(10)))
    func splitSparseMoERouteMatchesHFFirstTokenAndClearsLegacySpeedGate() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let prompt = ModelInput(chat: [
            .user([.text("What is the capital of Japan? Answer with just the city name.")])
        ])
        let monolithic = try await withSparseMoEMonolithicRoute {
            try await measureFirstRawToken(directory: localModelDirectory, input: prompt)
        }
        let split = try await withSparseMoEDefaultRoute {
            try await measureFirstRawToken(directory: localModelDirectory, input: prompt)
        }

        print("[LFM2.5 8B-A1B monolithic token ids] \(monolithic.tokenIDs)")
        print("[LFM2.5 8B-A1B split token ids] \(split.tokenIDs)")
        print(String(
            format: "[LFM2.5 8B-A1B sparse MoE route time] monolithic=%.3fs split=%.3fs speedup=%.1f%%",
            monolithic.elapsedSeconds,
            split.elapsedSeconds,
            monolithic.elapsedSeconds > 0
                ? (monolithic.elapsedSeconds - split.elapsedSeconds) / monolithic.elapsedSeconds * 100
                : 0
        ))

        #expect(split.tokenIDs == [124_901])
        #expect(
            split.elapsedSeconds <= monolithic.elapsedSeconds * 0.70,
            "Split Sparse MoE route must be at least 30% faster than the legacy monolithic diagnostic route"
        )
    }

    @Test("Default Sparse MoE route stays bounded across prompt lengths", .timeLimit(.minutes(10)))
    func defaultSparseMoERouteStaysBoundedAcrossPromptLengths() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        let longPrompt = Array(repeating: "Japan has a capital city and the answer should stay concise.", count: 8)
            .joined(separator: " ")
        let cases: [(label: String, input: ModelInput)] = [
            ("short-direct", ModelInput(prompt: "hi")),
            ("strict-chat", ModelInput(chat: [
                .user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])
            ])),
            ("long-direct", ModelInput(prompt: longPrompt)),
        ]

        for testCase in cases {
            context.resetState()
            let prepared = try await context.prepare(testCase.input)
            let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
            let start = CFAbsoluteTimeGetCurrent()
            let tokenIDs = try context.debugRawGeneratedTokenIDs(
                prompt: executable,
                parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 1)
            )
            let elapsedSeconds = CFAbsoluteTimeGetCurrent() - start
            print(String(
                format: "[LFM2.5 8B-A1B default route latency] %@ tokens=%d elapsed=%.3fs output=%@",
                testCase.label,
                prepared.tokenIDs.count,
                elapsedSeconds,
                context.decode(tokenIDs, skipSpecialTokens: false)
            ))

            #expect(tokenIDs.count == 1)
            #expect(!container.configuration.eosTokenIds.contains(tokenIDs[0]))
            #expect(
                elapsedSeconds < 3.0,
                "Default Sparse MoE route should not regress to diagnostic monolithic latency"
            )
        }
    }

    @Test("Default Sparse MoE route stays bounded across decode lengths", .timeLimit(.minutes(10)))
    func defaultSparseMoERouteStaysBoundedAcrossDecodeLengths() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let expectedTokenIDs = Self.hfStrictCapital64TokenIDs
        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        let prepared = try await context.prepare(ModelInput(chat: [
            .user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])
        ]))
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)

        for tokenCount in [1, 8, 16, 32, 64] {
            context.resetState()
            let start = CFAbsoluteTimeGetCurrent()
            let tokenIDs = try context.debugRawGeneratedTokenIDs(
                prompt: executable,
                parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: tokenCount)
            )
            let elapsedSeconds = CFAbsoluteTimeGetCurrent() - start
            let tokensPerSecond = Double(tokenCount) / max(elapsedSeconds, 0.001)
            print(String(
                format: "[LFM2.5 8B-A1B default route decode sweep] tokens=%d elapsed=%.3fs tok/s=%.1f",
                tokenCount,
                elapsedSeconds,
                tokensPerSecond
            ))

            #expect(tokenIDs == Array(expectedTokenIDs.prefix(tokenCount)))
            #expect(tokenIDs.count == tokenCount)
            #expect(!tokenIDs.contains { container.configuration.eosTokenIds.contains($0) })
            #expect(elapsedSeconds < 10.0, "Default route decode-length sweep should remain bounded")

            if tokenCount == 64 {
                let decodedText = context.decode(tokenIDs, skipSpecialTokens: false)
                print("[LFM2.5 8B-A1B sustained decode \(tokenCount)-token ids] \(tokenIDs)")
                print("[LFM2.5 8B-A1B sustained decode \(tokenCount)-token text] \(decodedText)")
                #expect(decodedText.contains("Tokyo"))
            }
        }

        context.resetState()
        let completeStart = CFAbsoluteTimeGetCurrent()
        let completeTokenIDs = try context.debugRawGeneratedTokenIDs(
            prompt: executable,
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 128)
        )
        let completeElapsedSeconds = CFAbsoluteTimeGetCurrent() - completeStart
        let completeTokensPerSecond = Double(completeTokenIDs.count) / max(completeElapsedSeconds, 0.001)
        let completeText = context.decode(completeTokenIDs, skipSpecialTokens: false)
        print(String(
            format: "[LFM2.5 8B-A1B default route complete decode] tokens=%d elapsed=%.3fs tok/s=%.1f",
            completeTokenIDs.count,
            completeElapsedSeconds,
            completeTokensPerSecond
        ))
        print("[LFM2.5 8B-A1B complete decode ids] \(completeTokenIDs)")
        print("[LFM2.5 8B-A1B complete decode text] \(completeText)")

        #expect(Array(completeTokenIDs.prefix(expectedTokenIDs.count)) == expectedTokenIDs)
        #expect(completeTokenIDs.count >= expectedTokenIDs.count)
        #expect(completeTokenIDs.count <= 128)
        #expect(completeText.contains("</think>"))
        #expect(completeText.contains("Tokyo"))
        #expect(completeElapsedSeconds < 20.0, "Complete decode should remain bounded")
    }

    @Test("Default Sparse MoE route matches HF traces across multiple prompts", .timeLimit(.minutes(10)))
    func defaultSparseMoERouteMatchesHFTracesAcrossMultiplePrompts() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let cases: [(label: String, prompt: String, expectedPrefixTokenIDs: [Int], expectedTextFragment: String)] = [
            (
                "strict-capital",
                RealOutputAssertionSupport.strictCapitalPrompt,
                Array(Self.hfStrictCapital64TokenIDs.prefix(16)),
                "Tokyo"
            ),
            (
                "largest-planet",
                "Name the largest planet in the Solar System. Answer with one word.",
                Self.hfLargestPlanet16TokenIDs,
                "Jupiter"
            ),
            (
                "good-morning-ja",
                "Translate good morning to Japanese. Answer only the translation.",
                Self.hfGoodMorningJapanese16TokenIDs,
                "おは"
            ),
        ]

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        var totalGeneratedTokens = 0
        let totalStart = CFAbsoluteTimeGetCurrent()

        for testCase in cases {
            context.resetState()
            let prepared = try await context.prepare(ModelInput(chat: [
                .user([.text(testCase.prompt)])
            ]))
            let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
            let start = CFAbsoluteTimeGetCurrent()
            let tokenIDs = try context.debugRawGeneratedTokenIDs(
                prompt: executable,
                parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 64)
            )
            let elapsedSeconds = CFAbsoluteTimeGetCurrent() - start
            let tokensPerSecond = Double(tokenIDs.count) / max(elapsedSeconds, 0.001)
            let decodedText = context.decode(tokenIDs, skipSpecialTokens: false)
            totalGeneratedTokens += tokenIDs.count
            print(String(
                format: "[LFM2.5 8B-A1B multi-prompt trace] %@ tokens=%d elapsed=%.3fs tok/s=%.1f",
                testCase.label,
                tokenIDs.count,
                elapsedSeconds,
                tokensPerSecond
            ))
            print("[LFM2.5 8B-A1B multi-prompt ids] \(testCase.label) \(tokenIDs)")
            print("[LFM2.5 8B-A1B multi-prompt text] \(testCase.label) \(decodedText)")

            #expect(Array(tokenIDs.prefix(testCase.expectedPrefixTokenIDs.count)) == testCase.expectedPrefixTokenIDs)
            #expect(decodedText.contains(testCase.expectedTextFragment))
            #expect(tokenIDs.count <= 64)
            #expect(elapsedSeconds < 5.0, "Multi-prompt trace should remain bounded")
        }

        let totalElapsedSeconds = CFAbsoluteTimeGetCurrent() - totalStart
        let aggregateTokensPerSecond = Double(totalGeneratedTokens) / max(totalElapsedSeconds, 0.001)
        print(String(
            format: "[LFM2.5 8B-A1B multi-prompt aggregate] prompts=%d tokens=%d elapsed=%.3fs tok/s=%.1f",
            cases.count,
            totalGeneratedTokens,
            totalElapsedSeconds,
            aggregateTokensPerSecond
        ))
        #expect(totalGeneratedTokens >= cases.reduce(0) { $0 + $1.expectedPrefixTokenIDs.count })
        #expect(aggregateTokensPerSecond > 50.0)
    }

    @Test("Default Sparse MoE route reports decode timing breakdown", .timeLimit(.minutes(10)))
    func defaultSparseMoERouteReportsDecodeTimingBreakdown() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        let prepared = try await context.prepare(ModelInput(chat: [
            .user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])
        ]))
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
        let timing = try context.debugRawGenerationTiming(
            prompt: executable,
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 64)
        )
        let hostOverheadSeconds = max(0, timing.decodeWallSeconds - timing.decodeGPUSeconds)

        print(String(
            format: "[LFM2.5 8B-A1B decode timing] tokens=%d prefill=%.3fs wall=%.3fs gpu=%.3fs wall_tok_s=%.1f gpu_tok_s=%.1f host_overhead=%.3fs steps=%d barriers=%d host_logit_reads=%d",
            timing.tokenIDs.count,
            timing.prefillSeconds,
            timing.decodeWallSeconds,
            timing.decodeGPUSeconds,
            timing.decodeWallTokensPerSecond,
            timing.decodeGPUTokensPerSecond,
            hostOverheadSeconds,
            timing.decodeStepCount,
            timing.decodeBarrierCount,
            timing.hostSamplingLogitReadCount
        ))
        print("[LFM2.5 8B-A1B decode kernel histogram] \(timing.decodeKernelHistogram.prefix(12).map { "\($0.kernelName):\($0.count)" }.joined(separator: ", "))")

        #expect(timing.tokenIDs == Self.hfStrictCapital64TokenIDs)
        if ProcessInfo.processInfo.environment["SWIFTLM_SPARSE_MOE_DISABLE_PACKED4"] == "1" {
            #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "sparse_moe_bf16_gate_up" })
            #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "sparse_moe_bf16_down" })
            #expect(timing.decodeWallTokensPerSecond > 60.0)
            #expect(timing.decodeGPUTokensPerSecond > 60.0)
        } else {
            if ProcessInfo.processInfo.environment["SWIFTLM_SPARSE_MOE_DISABLE_ROUTER_PARALLEL"] == "1" {
                #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "sparse_moe_bf16_router_scores" })
                #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "sparse_moe_bf16_router_select" })
                #expect(timing.decodeStepCount <= 224)
            } else {
                #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "sparse_moe_bf16_router_parallel" })
                #expect(timing.decodeStepCount <= 202)
            }
            if ProcessInfo.processInfo.environment["SWIFTLM_SPARSE_MOE_ENABLE_PACKED8"] == "1" {
                #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "sparse_moe_bf16_gate_up_packed8" })
                #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "sparse_moe_bf16_down_packed8" })
            } else {
                #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "sparse_moe_bf16_gate_up_packed4" })
                #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "sparse_moe_bf16_down_packed4" })
            }
            if ProcessInfo.processInfo.environment["SWIFTLM_OUTPUT_HEAD_PARTIAL_ARGMAX"] == "1" {
                #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "gemv_vocab_bf16_argmax_partial" })
                #expect(timing.decodeKernelHistogram.contains { $0.kernelName == "argmax_partial_reduce" })
            }
            #expect(timing.decodeWallTokensPerSecond > 78.0)
            #expect(timing.decodeGPUTokensPerSecond > 80.0)
            #expect(timing.hostSamplingLogitReadCount == 0)
        }
        #expect(timing.decodeStepCount > 0)
    }

    @Test("Production Sparse MoE route uses A1B optimized kernels", .timeLimit(.minutes(10)))
    func productionSparseMoERouteUsesA1BOptimizedKernels() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let timing = try await withSparseMoEDefaultRoute {
            try await withEnvironmentValue("SWIFTLM_SPARSE_MOE_DISABLE_ROUTER_PARALLEL", value: nil) {
                try await withEnvironmentValue("SWIFTLM_SPARSE_MOE_DISABLE_PACKED4", value: nil) {
                    let loader = ModelBundleLoader()
                    let container = try await loader.load(directory: localModelDirectory)
                    let context = try LanguageModelContext(container)
                    let prepared = try await context.prepare(ModelInput(chat: [
                        .user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])
                    ]))
                    let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
                    return try context.debugRawGenerationTiming(
                        prompt: executable,
                        parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 4)
                    )
                }
            }
        }

        let histogram = Dictionary(
            uniqueKeysWithValues: timing.decodeKernelHistogram.map { ($0.kernelName, $0.count) }
        )
        print("[LFM2.5 8B-A1B production route histogram] \(timing.decodeKernelHistogram.prefix(12).map { "\($0.kernelName):\($0.count)" }.joined(separator: ", "))")

        #expect(Array(timing.tokenIDs.prefix(4)) == Array(Self.hfStrictCapital64TokenIDs.prefix(4)))
        #expect(histogram["sparse_moe_bf16_router_parallel"] == 22)
        #expect(histogram["sparse_moe_bf16_gate_up_packed4"] == 22)
        #expect(histogram["sparse_moe_bf16_down_packed4"] == 22)
        #expect(histogram["sparse_moe_bf16_router_scores"] == nil)
        #expect(histogram["sparse_moe_bf16_router_select"] == nil)
        #expect(histogram["sparse_moe_bf16_gate_up"] == nil)
        #expect(histogram["sparse_moe_bf16_down"] == nil)
        #expect(timing.decodeStepCount <= 202)
    }

    @Test("Opt-in packed8 Sparse MoE projection route matches HF prefix", .timeLimit(.minutes(10)))
    func optInPacked8SparseMoEProjectionRouteMatchesHFPrefix() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }

        let timing = try await withSparseMoEDefaultRoute {
            try await withEnvironmentValue("SWIFTLM_SPARSE_MOE_ENABLE_PACKED8", value: "1") {
                let loader = ModelBundleLoader()
                let container = try await loader.load(directory: localModelDirectory)
                let context = try LanguageModelContext(container)
                let prepared = try await context.prepare(ModelInput(chat: [
                    .user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])
                ]))
                let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
                return try context.debugRawGenerationTiming(
                    prompt: executable,
                    parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 8)
                )
            }
        }

        let histogram = Dictionary(
            uniqueKeysWithValues: timing.decodeKernelHistogram.map { ($0.kernelName, $0.count) }
        )
        print("[LFM2.5 8B-A1B packed8 route histogram] \(timing.decodeKernelHistogram.prefix(12).map { "\($0.kernelName):\($0.count)" }.joined(separator: ", "))")

        #expect(timing.tokenIDs == Array(Self.hfStrictCapital64TokenIDs.prefix(8)))
        #expect(histogram["sparse_moe_bf16_gate_up_packed8"] == 22)
        #expect(histogram["sparse_moe_bf16_down_packed8"] == 22)
        #expect(histogram["sparse_moe_bf16_gate_up_packed4"] == nil)
        #expect(histogram["sparse_moe_bf16_down_packed4"] == nil)
    }

    @Test("Real packed Sparse MoE kernel matches CPU reference", .timeLimit(.minutes(10)))
    func realPackedSparseMoEKernelMatchesCPUReference() throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLFM25A1BModelDirectoryOrSkip() else {
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let store = try STAFLoader().load(
            at: localModelDirectory.appendingPathComponent("model.staf"),
            device: device
        )
        let layerIndex = 2
        let prefix = "model.layers.\(layerIndex).feed_forward"
        let router = try requireWeight(store, "\(prefix).gate.weight")
        let gateUp = try requireWeight(store, "\(prefix).experts.gate_up_proj")
        let down = try requireWeight(store, "\(prefix).experts.down_proj")
        let bias = try requireWeight(store, "\(prefix).expert_bias")

        let inputDimension = 2_048
        let outputDimension = 2_048
        let intermediateDimension = 1_792
        let expertCount = 32
        let expertsPerToken = 4
        let scratchRowStride = 2 * expertsPerToken + 2 * 128 + expertsPerToken * intermediateDimension
        var input = (0..<inputDimension).map { index in
            Float(((index * 37) % 97) - 48) * 0.00390625
        }
        var output = [Float](repeating: .zero, count: outputDimension)
        var moeScratch = [Float](repeating: .zero, count: scratchRowStride)

        let expected = sparseMoECPUReference(
            input: input,
            router: router,
            gateUp: gateUp,
            down: down,
            bias: bias,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            intermediateDimension: intermediateDimension,
            expertCount: expertCount,
            expertsPerToken: expertsPerToken
        )

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateSparseMoE(
                name: "test_real_lfm25_a1b_sparse_moe_seq_bf16",
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                gateKind: .sigmoidTopK
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let routerScoresPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_real_lfm25_a1b_sparse_moe_seq_bf16_router_scores"))
        )
        let routerSelectPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_real_lfm25_a1b_sparse_moe_seq_bf16_router_select"))
        )
        let gateUpPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_real_lfm25_a1b_sparse_moe_seq_bf16_gate_up_packed4"))
        )
        let downPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_real_lfm25_a1b_sparse_moe_seq_bf16_down_packed4"))
        )

        let inputBuffer = try #require(device.makeBuffer(
            bytes: &input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let moeScratchBuffer = try #require(device.makeBuffer(
            bytes: &moeScratch,
            length: moeScratch.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        let sequenceLength = 1
        let normalizeRoutingWeights = true
        var scalingFactor: Float = 1.0

        encoder.setComputePipelineState(routerScoresPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(router.buffer, offset: router.offset, index: 1)
        encoder.setBuffer(bias.buffer, offset: bias.offset, index: 2)
        encoder.setBuffer(moeScratchBuffer, offset: 0, index: 3)
        encoder.setBytes([UInt32(inputDimension)], length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes([UInt32(expertCount)], length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes([UInt32(expertsPerToken)], length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes([normalizeRoutingWeights ? UInt32(1) : UInt32(0)], length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&scalingFactor, length: MemoryLayout<Float>.stride, index: 8)
        encoder.setBytes([UInt32(1)], length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes([UInt32(sequenceLength)], length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes([UInt32(inputDimension)], length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes([UInt32(scratchRowStride)], length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.dispatchThreadgroups(
            MTLSize(width: expertCount, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(width: routerScoresPipeline.threadExecutionWidth, height: 1, depth: 1)
        )

        encoder.setComputePipelineState(routerSelectPipeline)
        encoder.setBuffer(moeScratchBuffer, offset: 0, index: 0)
        encoder.setBytes([UInt32(expertCount)], length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes([UInt32(expertsPerToken)], length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes([normalizeRoutingWeights ? UInt32(1) : UInt32(0)], length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&scalingFactor, length: MemoryLayout<Float>.stride, index: 4)
        encoder.setBytes([UInt32(sequenceLength)], length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes([UInt32(scratchRowStride)], length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: sequenceLength, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: routerSelectPipeline.threadExecutionWidth, height: 1, depth: 1)
        )

        let gateUpSimdWidth = max(gateUpPipeline.threadExecutionWidth, 1)
        let gateUpSimdgroups = max(1, min(32, gateUpPipeline.maxTotalThreadsPerThreadgroup / gateUpSimdWidth))
        encoder.setComputePipelineState(gateUpPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(gateUp.buffer, offset: gateUp.offset, index: 1)
        encoder.setBuffer(moeScratchBuffer, offset: 0, index: 2)
        encoder.setBytes([UInt32(inputDimension)], length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes([UInt32(intermediateDimension)], length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes([UInt32(expertsPerToken)], length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes([UInt32(sequenceLength)], length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes([UInt32(inputDimension)], length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes([UInt32(scratchRowStride)], length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (expertsPerToken * intermediateDimension + gateUpSimdgroups - 1) / gateUpSimdgroups,
                height: sequenceLength,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: gateUpSimdgroups * gateUpSimdWidth, height: 1, depth: 1)
        )

        let downSimdWidth = max(downPipeline.threadExecutionWidth, 1)
        let downSimdgroups = max(1, min(32, downPipeline.maxTotalThreadsPerThreadgroup / downSimdWidth))
        encoder.setComputePipelineState(downPipeline)
        encoder.setBuffer(moeScratchBuffer, offset: 0, index: 0)
        encoder.setBuffer(down.buffer, offset: down.offset, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes([UInt32(outputDimension)], length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes([UInt32(intermediateDimension)], length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes([UInt32(expertsPerToken)], length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes([UInt32(sequenceLength)], length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes([UInt32(outputDimension)], length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes([UInt32(scratchRowStride)], length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (outputDimension + downSimdgroups - 1) / downSimdgroups, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(width: downSimdgroups * downSimdWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let actualPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: output.count)
        let actual = (0..<output.count).map { actualPointer[$0] }
        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        print("[LFM2.5 8B-A1B real Sparse MoE kernel max error] \(maxError)")
        #expect(maxError < 0.02, "Real packed Sparse MoE kernel drifted: maxError=\(maxError)")
    }

    private struct MeasuredTokenTrace {
        let tokenIDs: [Int]
        let elapsedSeconds: Double
    }

    private struct WeightAccess {
        let buffer: MTLBuffer
        let offset: Int
    }

    private func requireWeight(_ store: STAFWeightStore, _ name: String) throws -> WeightAccess {
        guard let access = store.bufferAccess(for: name) else {
            Issue.record("Missing STAF tensor \(name)")
            throw LFM25A1BTestError.missingWeight(name)
        }
        return WeightAccess(buffer: access.buffer, offset: access.offset)
    }

    private enum LFM25A1BTestError: Error {
        case missingWeight(String)
    }

    private func sparseMoECPUReference(
        input: [Float],
        router: WeightAccess,
        gateUp: WeightAccess,
        down: WeightAccess,
        bias: WeightAccess,
        inputDimension: Int,
        outputDimension: Int,
        intermediateDimension: Int,
        expertCount: Int,
        expertsPerToken: Int
    ) -> [Float] {
        var routingWeights = [Float](repeating: .zero, count: expertCount)
        var routingScores = [Float](repeating: .zero, count: expertCount)
        for expert in 0..<expertCount {
            var logit: Float = 0
            for column in 0..<inputDimension {
                logit += readBF16(router, expert * inputDimension + column) * input[column]
            }
            let weight = Float(1.0 / (1.0 + Foundation.exp(Double(-logit))))
            routingWeights[expert] = weight
            routingScores[expert] = weight + readFloat32(bias, expert)
        }

        var selectedExperts: [Int] = []
        var selectedWeights: [Float] = []
        var selectedWeightSum: Float = 0
        for _ in 0..<expertsPerToken {
            var bestScore = -Float.infinity
            var bestExpert = 0
            for expert in 0..<expertCount where !selectedExperts.contains(expert) {
                if routingScores[expert] > bestScore {
                    bestScore = routingScores[expert]
                    bestExpert = expert
                }
            }
            selectedExperts.append(bestExpert)
            selectedWeights.append(routingWeights[bestExpert])
            selectedWeightSum += routingWeights[bestExpert]
        }
        selectedWeights = selectedWeights.map { $0 / (selectedWeightSum + 1.0e-6) }

        var activated = [Float](repeating: .zero, count: expertsPerToken * intermediateDimension)
        for (k, expert) in selectedExperts.enumerated() {
            let routeWeight = selectedWeights[k]
            let expertBase = expert * 2 * intermediateDimension * inputDimension
            let upBase = expertBase + intermediateDimension * inputDimension
            for middle in 0..<intermediateDimension {
                var gateValue: Float = 0
                var upValue: Float = 0
                let gateRow = expertBase + middle * inputDimension
                let upRow = upBase + middle * inputDimension
                for column in 0..<inputDimension {
                    let x = input[column]
                    gateValue += readBF16(gateUp, gateRow + column) * x
                    upValue += readBF16(gateUp, upRow + column) * x
                }
                let silu = gateValue * Float(1.0 / (1.0 + Foundation.exp(Double(-gateValue))))
                activated[k * intermediateDimension + middle] = silu * upValue * routeWeight
            }
        }

        var output = [Float](repeating: .zero, count: outputDimension)
        for row in 0..<outputDimension {
            var total: Float = 0
            for (k, expert) in selectedExperts.enumerated() {
                let downRow = expert * outputDimension * intermediateDimension + row * intermediateDimension
                for middle in 0..<intermediateDimension {
                    total += readBF16(down, downRow + middle) * activated[k * intermediateDimension + middle]
                }
            }
            output[row] = total
        }
        return output
    }

    private func readBF16(_ weight: WeightAccess, _ index: Int) -> Float {
        let pointer = (weight.buffer.contents() + weight.offset)
            .bindMemory(to: UInt16.self, capacity: index + 1)
        return BFloat16(bitPattern: pointer[index]).floatValue
    }

    private func readFloat32(_ weight: WeightAccess, _ index: Int) -> Float {
        let pointer = (weight.buffer.contents() + weight.offset)
            .bindMemory(to: Float.self, capacity: index + 1)
        return pointer[index]
    }

    private func measureFirstRawToken(
        directory: URL,
        input: ModelInput
    ) async throws -> MeasuredTokenTrace {
        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: directory)
        let context = try LanguageModelContext(container)
        let prepared = try await context.prepare(input)
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
        let start = CFAbsoluteTimeGetCurrent()
        let tokenIDs = try context.debugRawGeneratedTokenIDs(
            prompt: executable,
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 1)
        )
        return MeasuredTokenTrace(
            tokenIDs: tokenIDs,
            elapsedSeconds: CFAbsoluteTimeGetCurrent() - start
        )
    }

    private func withSparseMoEMonolithicRoute<T>(
        _ operation: () async throws -> T
    ) async throws -> T {
        try await withEnvironmentValue("SWIFTLM_DIAGNOSTIC_SPARSE_MOE_MONOLITHIC", value: "1", operation)
    }

    private func withSparseMoEDefaultRoute<T>(
        _ operation: () async throws -> T
    ) async throws -> T {
        try await withEnvironmentValue("SWIFTLM_DIAGNOSTIC_SPARSE_MOE_MONOLITHIC", value: nil, operation)
    }

    private func withEnvironmentValue<T>(
        _ key: String,
        value: String?,
        _ operation: () async throws -> T
    ) async throws -> T {
        let previous = ProcessInfo.processInfo.environment[key]
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let previous {
                setenv(key, previous, 1)
            } else {
                unsetenv(key)
            }
        }
        return try await operation()
    }
}
