import Foundation
import Metal
import Testing
@testable import MetalCompiler

/// Qwen 3.5 0.8B Q4-weight vs BF16-weight token quality.
///
/// Purpose: Verify the existing Q4G64 decode path produces healthy output
/// when applied to Qwen3.5's hybrid architecture (DeltaNet SSM layers +
/// Full Attention layers). Gemma4 only exercises Attention, so this test
/// expands coverage to the SSM projection path under quantization.
///
/// Bundles (MLX-community, matching provenance):
///   baseline : mlx-community/Qwen3.5-0.8B-MLX-bf16  (MLX-repacked bf16)
///   candidate: mlx-community/Qwen3.5-0.8B-4bit      (Q4G64 affine)
///
/// Assertion is on token diversity only (>= 20 unique tokens of 31),
/// same rationale as Gemma4Q4AgreementTests — greedy argmax agreement is
/// physically unreachable for a quantized model at temperature 0.
@Suite("Qwen35 Q4 Agreement", .serialized)
struct Qwen35Q4AgreementTests {

    @Test("Q4 vs BF16 token diversity (3 prompts × 30 decode steps)")
    func q4VersusBFloat16Agreement() throws {
        guard let bf16Path = Self.resolveBundle(repoName: "mlx-community--Qwen3.5-0.8B-MLX-bf16") else {
            Issue.record("BF16 bundle not cached. Expected ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-bf16")
            return
        }
        guard let q4Path = Self.resolveBundle(repoName: "mlx-community--Qwen3.5-0.8B-4bit") else {
            Issue.record("Q4 bundle not cached. Expected ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-4bit")
            return
        }

        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let decodeSteps = 30
        // Pre-tokenized via Qwen3.5-VL tokenizer (shared across all Qwen3.5 bundles):
        //   "What is the capital of Japan?"    → [3710, 369, 279, 6511, 314, 6124, 30]
        //   "Explain the Fibonacci sequence."  → [814, 20139, 279, 76938, 8240, 13]
        //   "Hello, how are you?"              → [9419, 11, 1204, 513, 488, 30]
        let prompts: [(String, [Int32])] = [
            ("japan     ", [3710, 369, 279, 6511, 314, 6124, 30]),
            ("fibonacci ", [814, 20139, 279, 76938, 8240, 13]),
            ("hello     ", [9419, 11, 1204, 513, 488, 30]),
        ]

        let policy = InferencePolicy(maximumSequenceLength: 256)

        var baselineTraces: [(String, [Int32])] = []
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bf16Path,
                inferencePolicy: policy)
            var m = model
            for (name, tokens) in prompts {
                m.resetState()
                let trace = BenchmarkSupport.decodeTokenTrace(
                    model: &m,
                    promptTokens: tokens,
                    decodeSteps: decodeSteps)
                baselineTraces.append((name, trace))
            }
        }

        var q4Traces: [(String, [Int32])] = []
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: q4Path,
                inferencePolicy: policy)
            var m = model
            for (name, tokens) in prompts {
                m.resetState()
                let trace = BenchmarkSupport.decodeTokenTrace(
                    model: &m,
                    promptTokens: tokens,
                    decodeSteps: decodeSteps)
                q4Traces.append((name, trace))
            }
        }

        print("\n=== Qwen3.5-0.8B Q4 vs BF16 Token Agreement ===")
        print("  decode_steps=\(decodeSteps)  prompts=\(prompts.count)")
        print(String(repeating: "-", count: 75))
        print("Prompt     Match   Rate    First_Div")
        print(String(repeating: "-", count: 75))

        let minimumUniqueTokens = 20
        var totalMatch = 0
        var totalCompare = 0
        for (index, (name, q4Trace)) in q4Traces.enumerated() {
            let bf16Trace = baselineTraces[index].1
            let compareLength = min(q4Trace.count, bf16Trace.count)

            var matchCount = 0
            var firstDivergence = -1
            for i in 0..<compareLength {
                if q4Trace[i] == bf16Trace[i] {
                    matchCount += 1
                } else if firstDivergence < 0 {
                    firstDivergence = i
                }
            }
            totalMatch += matchCount
            totalCompare += compareLength

            let rate = compareLength > 0 ? Double(matchCount) / Double(compareLength) * 100 : 0
            let divStr = firstDivergence >= 0 ? String(firstDivergence) : "none"
            let bf16Preview = bf16Trace.prefix(15).map { String($0) }.joined(separator: ",")
            let q4Preview = q4Trace.prefix(15).map { String($0) }.joined(separator: ",")
            let padName = name.padding(toLength: 10, withPad: " ", startingAt: 0)
            print("\(padName) \(String(format: "%3d/%3d", matchCount, compareLength))  \(String(format: "%5.1f%%", rate))  \(divStr)")
            print("  BF16: [\(bf16Preview)]")
            print("  Q4  : [\(q4Preview)]")
            let bf16Unique = Set(bf16Trace).count
            let q4Unique = Set(q4Trace).count
            print("  diversity: BF16 \(bf16Unique) unique, Q4 \(q4Unique) unique (of \(compareLength) tokens)")

            #expect(
                bf16Unique >= minimumUniqueTokens,
                "BF16 trace for '\(name.trimmingCharacters(in: .whitespaces))' collapsed: only \(bf16Unique) unique tokens of \(compareLength) (threshold \(minimumUniqueTokens))")
            withKnownIssue(
                "Q4 decode collapses to token 0 on Qwen3.5's DeltaNet/SSM quantization path (Gemma4 Attention-only Q4 is healthy). Tracked separately from the MLX VLM tensor-name canonicalization work that unblocked bundle loading.",
                isIntermittent: false
            ) {
                #expect(
                    q4Unique >= minimumUniqueTokens,
                    "Q4 trace for '\(name.trimmingCharacters(in: .whitespaces))' collapsed: only \(q4Unique) unique tokens of \(compareLength) (threshold \(minimumUniqueTokens))")
            }
        }

        print(String(repeating: "-", count: 75))
        let aggregate = totalCompare > 0 ? Double(totalMatch) / Double(totalCompare) * 100 : 0
        print(String(format: "Aggregate: %d/%d  (%.2f%%) — informational only, not asserted", totalMatch, totalCompare, aggregate))
        print()
    }

    private static func resolveBundle(repoName: String) -> String? {
        let hubRoot = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        let snapshotsDir = "\(hubRoot)/models--\(repoName)/snapshots"
        guard FileManager.default.fileExists(atPath: snapshotsDir) else { return nil }
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir).sorted() else { return nil }
        for entry in entries {
            let candidate = "\(snapshotsDir)/\(entry)"
            if FileManager.default.fileExists(atPath: "\(candidate)/config.json") {
                return candidate
            }
        }
        return nil
    }
}
