import Foundation
@_spi(Benchmark) import SwiftLM

struct BenchmarkResult {
    let modelDirectory: URL
    let maxTokens: Int
    let iterations: Int
    let best: LanguageModelContext.DebugRawGenerationTiming
    let allWallTokensPerSecond: [Double]

    var bestWallTokensPerSecond: Double { best.decodeWallTokensPerSecond }

    var report: String {
        let values = allWallTokensPerSecond
            .map { String(format: "%.1f", $0) }
            .joined(separator: ",")
        return String(
            format: "[LFM2.5 8B-A1B release benchmark] model=%@ tokens=%d iterations=%d wall=%.3fs wall_tok_s=%.1f prefill=%.3fs steps=%d barriers=%d host_logit_reads=%d all_wall_tok_s=[%@]",
            modelDirectory.path,
            maxTokens,
            iterations,
            best.decodeWallSeconds,
            best.decodeWallTokensPerSecond,
            best.prefillSeconds,
            best.decodeStepCount,
            best.decodeBarrierCount,
            best.hostSamplingLogitReadCount,
            values
        )
    }
}
