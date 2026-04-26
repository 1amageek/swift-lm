import Foundation
import Metal
import Testing
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
/// Compare per-layer hidden state between HF Qwen3.5-0.8B (working) and
/// MLX Qwen3.5-0.8B-MLX-bf16 (broken). Identifies the FIRST layer where
/// divergence appears so we can isolate which kernel/weight is wrong.
///
/// The MLX bf16 bundle should be a 1:1 reproduction of HF: same weight values
/// (after tensor-name canonicalization) and dtypes (after `linear_attn.norm.weight`
/// dtype normalization in STAFConversionPlanner). Any per-layer divergence
/// indicates an unfixed dtype or layout mismatch.
@Suite("Qwen35 MLX vs HF Layer Probe", .serialized)
struct Qwen35MLXLayerProbeTests {

    @Test("HF vs MLX-bf16 per-layer hidden divergence")
    func hfVsMlxBf16LayerDivergence() throws {
        guard let hfPath = try Self.resolveHFBundle() else {
            Issue.record("HF Qwen3.5-0.8B bundle not cached. Expected ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B")
            return
        }
        guard let mlxPath = try Self.resolveMLXBundle(name: "Qwen3.5-0.8B-MLX-bf16") else {
            Issue.record("MLX-bf16 bundle not cached. Expected ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-bf16")
            return
        }

        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        // Short token sequence to keep the probe fast and deterministic.
        let tokens: [Int32] = [248045, 846, 198, 3710, 369, 198]
        let policy = InferencePolicy(maximumSequenceLength: 64)

        let hfData = try captureAllSteps(bundlePath: hfPath, policy: policy, tokens: tokens, label: "HF")
        let mlxData = try captureAllSteps(bundlePath: mlxPath, policy: policy, tokens: tokens, label: "MLX")

        print("\n=== Qwen3.5-0.8B HF vs MLX-bf16 Per-Step Hidden Probe ===")
        print("Tokens: \(tokens)")
        print("step | kernel                                            | HF L2      | MLX L2     | cosine")
        print(String(repeating: "-", count: 110))

        let count = min(hfData.count, mlxData.count)
        var firstDivStep = -1
        for i in 0..<count {
            let stepIdx = hfData[i].step
            let hfK = hfData[i].kernel
            let mlxK = mlxData[i].kernel
            let hfH = hfData[i].hidden
            let mlxH = mlxData[i].hidden
            let hfL = l2Norm(hfH)
            let mlxL = l2Norm(mlxH)
            let cos = cosine(hfH, mlxH)
            let kpad = (hfK == mlxK ? hfK : "HF=\(hfK) / MLX=\(mlxK)")
                .padding(toLength: 50, withPad: " ", startingAt: 0)
            print("\(String(format: "%4d", stepIdx)) | \(kpad) | \(String(format: "%10.4f", hfL)) | \(String(format: "%10.4f", mlxL)) | \(String(format: "%.4f", cos))")
            // Skip "all zero" steps (hidden not updated at this step)
            if hfL > 0.001 && mlxL > 0.001 && cos < 0.95 && firstDivStep < 0 {
                firstDivStep = stepIdx
            }
        }
        print(String(repeating: "-", count: 110))
        print("First step with cosine(HF, MLX) < 0.95 (both non-zero): step \(firstDivStep)")

        if firstDivStep >= 0 {
            for i in 0..<count where hfData[i].step == firstDivStep {
                let hfHead = hfData[i].hidden.prefix(8).map { String(format: "%+.4f", $0) }.joined(separator: ",")
                let mlxHead = mlxData[i].hidden.prefix(8).map { String(format: "%+.4f", $0) }.joined(separator: ",")
                print("\nStep \(firstDivStep) hidden head:")
                print("  HF [0..8]: [\(hfHead)]")
                print("  MLX[0..8]: [\(mlxHead)]")
                break
            }
        }

        #expect(count > 0)
    }

    private struct StepProbe {
        let step: Int
        let kernel: String
        let hidden: [Float]
    }

    private func captureAllSteps(
        bundlePath: String,
        policy: InferencePolicy,
        tokens: [Int32],
        label: String
    ) throws -> [StepProbe] {
        var result: [StepProbe] = []
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bundlePath, inferencePolicy: policy)
            var m = model
            let plan = try #require(m.prefillPlan, "prefillPlan missing")

            // Probe every step: snapshot the hidden buffer after each dispatch.
            // Steps that don't write to hidden will return zeros (filtered later).
            let allSteps = Set(0..<plan.steps.count)
            let snapshots = try m.debugPrefillLastTokenHiddenSnapshots(
                tokens: tokens, stepIndices: allSteps)

            var probes: [StepProbe] = []
            for idx in 0..<plan.steps.count {
                guard let h = snapshots[idx] else { continue }
                let kernel = plan.steps[idx].metadata.kernelName
                    ?? plan.steps[idx].pipeline.label
                    ?? "?"
                probes.append(StepProbe(step: idx, kernel: kernel, hidden: h))
            }
            print("[\(label)] plan.steps.count=\(plan.steps.count) probed_steps=\(probes.count)")
            result = probes
        }
        return result
    }

    private func l2Norm(_ v: [Float]) -> Float {
        var sum: Float = 0
        for x in v { sum += x * x }
        return sqrt(sum)
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        for i in 0..<n {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = sqrt(na) * sqrt(nb)
        return denom > 0 ? dot / denom : 0
    }

    private static func resolveHFBundle() throws -> String? {
        let hubRoot = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        let snapshotsDir = "\(hubRoot)/models--Qwen--Qwen3.5-0.8B/snapshots"
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

    private static func resolveMLXBundle(name: String) throws -> String? {
        let hubRoot = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        let snapshotsDir = "\(hubRoot)/models--mlx-community--\(name)/snapshots"
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
}
#endif
