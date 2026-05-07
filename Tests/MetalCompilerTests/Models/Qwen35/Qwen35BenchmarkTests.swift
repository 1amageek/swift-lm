import Foundation
import Metal
import Testing
@testable import MetalCompiler

/// Qwen 3.5 0.8B throughput benchmarks.
///
/// Designed for apples-to-apples comparison with mlx-swift-lm's Qwen35BenchmarkTests.
/// Both stacks load the same underlying weights (Qwen/Qwen3.5-0.8B BF16, text path only).
///
/// Bundle resolution:
///   - Direct: $SWIFTLM_QWEN35_BUNDLE  (env override)
///   - Cache : ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<hash>/
///
/// If neither is present the test is skipped (`Issue.record`).
#if ENABLE_METAL_PROBES
@Suite("Qwen35 Benchmark", .serialized)
struct Qwen35BenchmarkTests {

    static let modelLabel = "Qwen3.5-0.8B"
    static let q3ModelLabel = "Qwen3.5-0.8B Q3"

    @Test("MLX-aligned prefill + decode throughput (3-run median)")
    func mlxAlignedBenchmark() throws {
        guard let bundlePath = try Self.resolveBundlePath() else {
            Issue.record("Qwen3.5-0.8B bundle not found. Expected ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B or $SWIFTLM_QWEN35_BUNDLE.")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: bundlePath,
            maximumPrefillLength: 128
        )
        var inferenceModel = model

        // Warmup — resident caches, hot Metal kernels.
        do {
            let warmupTokens: [Int32] = Array(repeating: 1, count: 8)
            var tok = inferenceModel.prefill(tokens: warmupTokens)
            for _ in 0..<4 { tok = inferenceModel.decodeSync(tokenID: tok) }
            inferenceModel.resetState()
        }

        print("=== \(Self.modelLabel) BF16 swift-lm benchmark (MLX-aligned) ===")
        print("bundle: \(bundlePath)")
        print("runs per measurement: 3")
        print()

        print("PREFILL (tok/s — prompt tokens divided by time-to-first-token)")
        let prefillLengths = [16, 32, 64, 128]
        for length in prefillLengths {
            var tps: [Double] = []
            var msList: [Double] = []
            for _ in 0..<3 {
                inferenceModel.resetState()
                let tokens = [Int32](repeating: 1, count: length)
                let start = CFAbsoluteTimeGetCurrent()
                _ = inferenceModel.prefill(tokens: tokens)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                tps.append(Double(length) / elapsed)
                msList.append(elapsed * 1000)
            }
            let s = BenchStats(tps)
            let m = BenchStats(msList)
            print(String(
                format: "  len %3d: median %6.1f tok/s, mean %6.1f ±%.2f (σ/μ %.2f%%) | %.2f ms median",
                length, s.median, s.mean, s.stddev, s.relStddev * 100, m.median))
        }

        print()
        print("DECODE (tok/s — steady-state token generation after prefill)")
        let decodeSteps = 100
        var dtps: [Double] = []
        var dms: [Double] = []
        for _ in 0..<3 {
            inferenceModel.resetState()
            let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
            var tok = inferenceModel.prefill(tokens: promptTokens)
            for _ in 0..<3 { tok = inferenceModel.decodeSync(tokenID: tok) }

            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<decodeSteps {
                tok = inferenceModel.decodeSync(tokenID: tok)
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            dtps.append(Double(decodeSteps) / elapsed)
            dms.append(elapsed * 1000 / Double(decodeSteps))
        }
        let ds = BenchStats(dtps)
        let dm = BenchStats(dms)
        print(String(
            format: "  %3d steps: median %5.1f tok/s, mean %5.1f ±%.2f (σ/μ %.2f%%) | %.2f ms/tok median",
            decodeSteps, ds.median, ds.mean, ds.stddev, ds.relStddev * 100, dm.median))
        print()
    }

    @Test("Q3 sequence prefill smoke benchmark")
    func q3SequencePrefillSmokeBenchmark() throws {
        guard let bundlePath = try Self.resolveBundle(repoName: "mlx-community--Qwen3.5-0.8B-3bit") else {
            Issue.record("Q3 bundle not found. Expected ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-3bit.")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: bundlePath,
            maximumPrefillLength: 128
        )
        let prefillPlan = try #require(model.prefillPlan)
        #expect(!prefillPlan.requiresSequentialPromptIngestion)
        #expect(prefillPlan.sequencePrefillFallbackReason == nil)
        #expect(
            Self.q3SequenceGEMVKernelCount(in: prefillPlan) > 0,
            "Q3 prefill must include packed Q3 sequence GEMV kernels."
        )

        var inferenceModel = model
        let warmupTokens: [Int32] = Array(repeating: 1, count: 8)
        _ = inferenceModel.prefill(tokens: warmupTokens)
        inferenceModel.resetState()

        var sequentialModel = model
        sequentialModel.prefillPlan = nil
        _ = sequentialModel.prefill(tokens: warmupTokens)
        sequentialModel.resetState()

        print("=== \(Self.q3ModelLabel) swift-lm sequence prefill smoke ===")
        print("bundle: \(bundlePath)")
        print("q3 sequence gemv kernels: \(Self.q3SequenceGEMVKernelCount(in: prefillPlan))")
        print("q3 batched sequence gemv kernels: \(Self.q3BatchedSequenceGEMVKernelCount(in: prefillPlan))")
        Self.printQ3SequenceKernelBreakdown(in: prefillPlan)
        print("runs per measurement: 2")
        print()

        let prefillLengths = [16, 64, 128]
        for length in prefillLengths {
            let sequence = Self.measurePrefill(model: &inferenceModel, length: length, runs: 2)
            let sequential = Self.measurePrefill(model: &sequentialModel, length: length, runs: 2)
            let speedup = sequential.milliseconds.median / sequence.milliseconds.median
            #expect(
                sequence.milliseconds.median < sequential.milliseconds.median,
                "Q3 sequence prefill should stay faster than sequential prompt ingestion at length \(length)."
            )
            print(String(
                format: "  len %3d: sequence %.2f ms, sequential %.2f ms, speedup %.2fx | sequence median %.1f tok/s",
                length,
                sequence.milliseconds.median,
                sequential.milliseconds.median,
                speedup,
                sequence.tokensPerSecond.median))
        }
        print()
    }

    // MARK: - Bundle resolution

    private static func resolveBundlePath() throws -> String? {
        if let override = ProcessInfo.processInfo.environment["SWIFTLM_QWEN35_BUNDLE"],
           !override.trimmingCharacters(in: .whitespaces).isEmpty {
            return NSString(string: override).expandingTildeInPath
        }
        let hubRoot = NSString(string: "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots").expandingTildeInPath
        guard FileManager.default.fileExists(atPath: hubRoot) else { return nil }
        let entries = try FileManager.default.contentsOfDirectory(atPath: hubRoot).sorted()
        for entry in entries {
            let candidate = "\(hubRoot)/\(entry)"
            let cfg = "\(candidate)/config.json"
            if FileManager.default.fileExists(atPath: cfg) {
                return candidate
            }
        }
        return nil
    }

    private static func resolveBundle(repoName: String) throws -> String? {
        let hubRoot = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        let snapshotsDir = "\(hubRoot)/models--\(repoName)/snapshots"
        guard FileManager.default.fileExists(atPath: snapshotsDir) else { return nil }
        let entries = try FileManager.default.contentsOfDirectory(atPath: snapshotsDir).sorted()
        for entry in entries {
            let candidate = "\(snapshotsDir)/\(entry)"
            if FileManager.default.fileExists(atPath: "\(candidate)/config.json") {
                return candidate
            }
        }
        return nil
    }

    private static func prefillKernelCount(prefix: String, in prefillPlan: MetalPrefillPlan) -> Int {
        prefillPlan.steps.count { step in
            let name = step.metadata.kernelName ?? step.pipeline.label ?? ""
            return name.hasPrefix(prefix)
        }
    }

    private static func q3SequenceGEMVKernelCount(in prefillPlan: MetalPrefillPlan) -> Int {
        prefillPlan.steps.count { step in
            let name = step.metadata.kernelName ?? step.pipeline.label ?? ""
            return name.hasPrefix("gemv_seq_q3_g")
                || (name.hasPrefix("batched_gemv") && name.contains("_seq_q3_g"))
        }
    }

    private static func q3BatchedSequenceGEMVKernelCount(in prefillPlan: MetalPrefillPlan) -> Int {
        prefillPlan.steps.count { step in
            let name = step.metadata.kernelName ?? step.pipeline.label ?? ""
            return name.hasPrefix("batched_gemv") && name.contains("_seq_q3_g")
        }
    }

    private static func printQ3SequenceKernelBreakdown(in prefillPlan: MetalPrefillPlan) {
        var kernelCounts: [String: Int] = [:]
        var singleProjectionCounts: [String: Int] = [:]
        for step in prefillPlan.steps {
            let name = step.metadata.kernelName ?? step.pipeline.label ?? ""
            guard name.hasPrefix("gemv_seq_q3_g")
                || (name.hasPrefix("batched_gemv") && name.contains("_seq_q3_g")) else {
                continue
            }
            kernelCounts[name, default: 0] += 1
            if name.hasPrefix("gemv_seq_q3_g") {
                let summary = step.metadata.weightTensorName.map(weightRoleSummary) ?? "(unknown)"
                singleProjectionCounts[summary, default: 0] += 1
            }
        }

        print("q3 sequence kernel breakdown:")
        for (name, count) in kernelCounts.sorted(by: { $0.key < $1.key }) {
            print("  \(count)× \(name)")
        }
        print("q3 single sequence projection breakdown:")
        for (role, count) in singleProjectionCounts.sorted(by: { $0.key < $1.key }) {
            print("  \(count)× \(role)")
        }
    }

    private static func weightRoleSummary(_ tensorName: String) -> String {
        var components = tensorName.split(separator: ".").map(String.init)
        if components.last == "weight" {
            components.removeLast()
        }
        if let layerIndex = components.firstIndex(of: "layers"),
           layerIndex + 2 < components.count {
            return components[(layerIndex + 2)...].joined(separator: ".")
        }
        return components.suffix(3).joined(separator: ".")
    }

    private static func measurePrefill(
        model: inout MetalInferenceModel,
        length: Int,
        runs: Int
    ) -> PrefillMeasurement {
        var tps: [Double] = []
        var msList: [Double] = []
        for _ in 0..<runs {
            model.resetState()
            let tokens = [Int32](repeating: 1, count: length)
            let start = CFAbsoluteTimeGetCurrent()
            _ = model.prefill(tokens: tokens)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            tps.append(Double(length) / elapsed)
            msList.append(elapsed * 1000)
        }
        return PrefillMeasurement(
            tokensPerSecond: BenchStats(tps),
            milliseconds: BenchStats(msList)
        )
    }

    private struct PrefillMeasurement {
        let tokensPerSecond: BenchStats
        let milliseconds: BenchStats
    }

    private struct BenchStats {
        let mean: Double
        let median: Double
        let stddev: Double
        init(_ values: [Double]) {
            precondition(!values.isEmpty)
            let sorted = values.sorted()
            let count = values.count
            let sum = values.reduce(0, +)
            let meanValue = sum / Double(count)
            let stddevValue: Double
            if count > 1 {
                let variance = values.reduce(0.0) { acc, v in
                    acc + (v - meanValue) * (v - meanValue)
                } / Double(count - 1)
                stddevValue = variance.squareRoot()
            } else {
                stddevValue = 0
            }
            self.mean = meanValue
            self.median = sorted[count / 2]
            self.stddev = stddevValue
        }
        var relStddev: Double { mean == 0 ? 0 : stddev / mean }
    }
}
#endif
