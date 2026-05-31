import Foundation
@_spi(Benchmark) import SwiftLM

@main
struct LFM25A1BBenchmarkRunner {
    static func main() async {
        do {
            let options = try BenchmarkOptions(arguments: Array(CommandLine.arguments.dropFirst()))
            let result = try await run(options: options)
            print(result.report)
            if options.requiresM5Gate, result.medianWallTokensPerSecond < 90.0 {
                throw BenchmarkError.m5GateFailed(result.medianWallTokensPerSecond)
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            Foundation.exit(1)
        }
    }

    private static func run(options: BenchmarkOptions) async throws -> BenchmarkResult {
        let modelDirectory = try options.modelDirectory ?? HFCacheResolver.resolveSnapshot(
            repoDirectoryName: "models--LiquidAI--LFM2.5-8B-A1B"
        )
        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: modelDirectory)
        let context = try LanguageModelContext(container)
        let prepared = try await context.prepare(ModelInput(chat: [
            .user([.text(options.prompt)])
        ]))
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
        let expectedTokenIDs = Array(Self.expectedStrictCapital64TokenIDs.prefix(options.maxTokens))
        let expectedText = context.decode(expectedTokenIDs, skipSpecialTokens: false)

        var warmups: [LanguageModelContext.DebugRawGenerationTiming] = []
        warmups.reserveCapacity(options.warmupIterations)
        for _ in 0..<options.warmupIterations {
            warmups.append(try runOne(
                context: context,
                executable: executable,
                expectedTokenIDs: expectedTokenIDs,
                expectedText: expectedText,
                options: options
            ))
        }

        var measuredRuns: [LanguageModelContext.DebugRawGenerationTiming] = []
        measuredRuns.reserveCapacity(options.iterations)
        for _ in 0..<options.iterations {
            measuredRuns.append(try runOne(
                context: context,
                executable: executable,
                expectedTokenIDs: expectedTokenIDs,
                expectedText: expectedText,
                options: options
            ))
        }

        guard let best = measuredRuns.max(by: { $0.decodeWallTokensPerSecond < $1.decodeWallTokensPerSecond }) else {
            throw BenchmarkError.noRuns
        }
        return BenchmarkResult(
            modelDirectory: modelDirectory,
            maxTokens: options.maxTokens,
            warmupIterations: options.warmupIterations,
            iterations: options.iterations,
            best: best,
            allWallTokensPerSecond: measuredRuns.map(\.decodeWallTokensPerSecond),
            warmupWallTokensPerSecond: warmups.map(\.decodeWallTokensPerSecond)
        )
    }

    private static func runOne(
        context: LanguageModelContext,
        executable: ExecutablePrompt,
        expectedTokenIDs: [Int],
        expectedText: String,
        options: BenchmarkOptions
    ) throws -> LanguageModelContext.DebugRawGenerationTiming {
        context.resetState()
        let timing = try context.debugRawGenerationWallTiming(
            prompt: executable,
            parameters: GenerationParameters(
                maxTokens: options.maxTokens,
                streamChunkTokenCount: 1,
                temperature: 0
            )
        )
        guard timing.tokenIDs == expectedTokenIDs else {
            throw BenchmarkError.traceMismatch(
                expected: expectedTokenIDs,
                actual: timing.tokenIDs
            )
        }
        let actualText = context.decode(timing.tokenIDs, skipSpecialTokens: false)
        guard actualText == expectedText else {
            throw BenchmarkError.textMismatch(expected: expectedText, actual: actualText)
        }
        return timing
    }

    private static let expectedStrictCapital64TokenIDs = [
        124_901, 207, 597, 4_695, 20_589, 34, 496, 2_992,
        355, 278, 5_205, 302, 3_888, 39, 41_774, 415,
        8_043, 734, 1_858, 2_426, 8, 2_083, 1_946, 6_119,
        415, 8_043, 734, 1_858, 20, 643, 355, 496,
        40_049, 11_053, 2_784, 3_584, 589, 734, 1_858, 22,
        23_620, 513, 1_113, 4_666, 61_049, 39, 440, 10_996,
        34, 496, 39_630, 415, 8_043, 734, 1_858, 2_426,
        1_672, 522, 1_252, 6_159, 496, 40_049, 11_053, 2_784,
    ]
}
