import Foundation
@_spi(Benchmark) import SwiftLM

struct BenchmarkResult {
    let modelDirectory: URL
    let maxTokens: Int
    let warmupIterations: Int
    let iterations: Int
    let best: LanguageModelContext.DebugRawGenerationTiming
    let allWallTokensPerSecond: [Double]
    let warmupWallTokensPerSecond: [Double]

    var bestWallTokensPerSecond: Double { best.decodeWallTokensPerSecond }
    var medianWallTokensPerSecond: Double {
        let sorted = allWallTokensPerSecond.sorted()
        guard !sorted.isEmpty else { return 0 }
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    var report: String {
        let values = allWallTokensPerSecond
            .map { String(format: "%.1f", $0) }
            .joined(separator: ",")
        let warmupValues = warmupWallTokensPerSecond
            .map { String(format: "%.1f", $0) }
            .joined(separator: ",")
        let kernels = formatHistogram(best.decodeKernelHistogram, limit: 10)
        let barriers = formatHistogram(best.decodeBarrierKernelHistogram, limit: 10)
        return String(
            format: "[LFM2.5 8B-A1B release benchmark] model=%@ tokens=%d warmup=%d iterations=%d best_wall=%.3fs best_wall_tok_s=%.1f median_wall_tok_s=%.1f prefill=%.3fs steps=%d barriers=%d host_logit_reads=%d warmup_wall_tok_s=[%@] all_wall_tok_s=[%@] kernels=[%@] barrier_kernels=[%@]",
            modelDirectory.path,
            maxTokens,
            warmupIterations,
            iterations,
            best.decodeWallSeconds,
            best.decodeWallTokensPerSecond,
            medianWallTokensPerSecond,
            best.prefillSeconds,
            best.decodeStepCount,
            best.decodeBarrierCount,
            best.hostSamplingLogitReadCount,
            warmupValues,
            values,
            kernels,
            barriers
        )
    }

    private func formatHistogram(
        _ histogram: [(kernelName: String, count: Int)],
        limit: Int
    ) -> String {
        histogram.prefix(limit)
            .map { "\($0.kernelName):\($0.count)" }
            .joined(separator: ",")
    }
}
