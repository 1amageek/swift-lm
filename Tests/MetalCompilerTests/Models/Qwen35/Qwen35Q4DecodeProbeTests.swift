import Foundation
import Testing
@testable import MetalCompiler

@Suite("Qwen35 Q4 probe")
struct Qwen35Q4DecodeProbeTests {

    @Test("dump decode kernels per layer")
    func dumpDecodeKernels() throws {
        let hubRoot = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        let snapshotsDir = "\(hubRoot)/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots"
        guard let entry = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir).sorted().first,
              FileManager.default.fileExists(atPath: "\(snapshotsDir)/\(entry)/config.json") else {
            Issue.record("Q4 bundle missing")
            return
        }
        let bundlePath = "\(snapshotsDir)/\(entry)"

        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let policy = InferencePolicy(maximumSequenceLength: 64)
        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: bundlePath, inferencePolicy: policy)

        print("\n=== Q4 Qwen3.5 decode plan ===")
        print("Steps: \(model.decodePlan.steps.count)")
        let layers = Set(model.decodePlan.steps.compactMap { $0.metadata.layerIndex })
        print("Layer indices present: \(layers.sorted())")

        print("\n--- First 90 steps ---")
        for (i, step) in model.decodePlan.steps.enumerated() {
            guard i < 90 else { break }
            let k = step.metadata.kernelName ?? "<nil>"
            let w = step.metadata.weightTensorName ?? "-"
            let li = step.metadata.layerIndex.map { "L\($0)" } ?? "L?"
            print("  [\(i)] \(li) \(k)  weight=\(w)")
        }
    }
}
