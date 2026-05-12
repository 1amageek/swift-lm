import Foundation
import Testing
@testable import MetalCompiler

@Suite("Prefill Profile Harness")
struct PrefillProfileHarnessTests {
    @Test("Prefill execution conditions gate sequence lengths")
    func prefillExecutionConditionsGateSequenceLengths() {
        #expect(PrefillStepExecutionCondition.always.shouldExecute(sequenceLength: 1))
        #expect(PrefillStepExecutionCondition.sequenceLengthAtLeast(64).shouldExecute(sequenceLength: 64))
        #expect(PrefillStepExecutionCondition.sequenceLengthAtLeast(64).shouldExecute(sequenceLength: 128))
        #expect(!PrefillStepExecutionCondition.sequenceLengthAtLeast(64).shouldExecute(sequenceLength: 63))
        #expect(PrefillStepExecutionCondition.sequenceLengthAtMost(63).shouldExecute(sequenceLength: 1))
        #expect(PrefillStepExecutionCondition.sequenceLengthAtMost(63).shouldExecute(sequenceLength: 63))
        #expect(!PrefillStepExecutionCondition.sequenceLengthAtMost(63).shouldExecute(sequenceLength: 64))
    }

    @Test("Profile artifacts encode JSON and CSV summaries")
    func profileArtifactsEncodeJSONAndCSV() throws {
        let entry = MetalPrefillProfile.Entry(
            scope: "step",
            index: 0,
            rangeStart: 0,
            rangeEnd: 1,
            kernelName: "gemv_seq_bf16_f32s",
            category: "projection",
            mode: "batch",
            layerIndex: 0,
            entryIndex: 4,
            weightTensorName: "layers.0.mlp.down_proj.weight",
            gridWidth: 1024,
            gridHeight: 128,
            gridDepth: 1,
            threadgroupWidth: 64,
            threadgroupHeight: 1,
            threadgroupDepth: 1,
            threadgroupMemoryBytes: 1024,
            bufferBindingCount: 4,
            inlineConstantBytes: 16,
            uniqueBoundBufferBytes: 4096,
            estimatedDispatchCount: 1,
            totalGpuMicroseconds: 100,
            averageGpuMicroseconds: 25,
            totalWallMicroseconds: 140,
            averageWallMicroseconds: 35
        )
        let profile = MetalPrefillProfile(
            profileKind: "step",
            sequenceLength: 128,
            maximumSequenceLength: 128,
            iterations: 4,
            warmupIterations: 1,
            stepCount: 1,
            entries: [entry],
            generatedAt: "2026-05-07T00:00:00Z"
        )

        let root = repositoryRoot()
            .appendingPathComponent(".test-artifacts/prefill-profile-harness-test", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let urls = try profile.writeArtifacts(directory: root, basename: "synthetic")

        #expect(urls.count == 6)
        let jsonURL = root.appendingPathComponent("synthetic.json")
        let csvURL = root.appendingPathComponent("synthetic.csv")
        let categoryCSVURL = root.appendingPathComponent("synthetic-categories.csv")
        let kernelCSVURL = root.appendingPathComponent("synthetic-kernels.csv")
        let layerCSVURL = root.appendingPathComponent("synthetic-layers.csv")
        let weightCSVURL = root.appendingPathComponent("synthetic-weights.csv")
        let jsonData = try Data(contentsOf: jsonURL)
        let decoded = try JSONDecoder().decode(MetalPrefillProfile.self, from: jsonData)
        #expect(decoded.summary.entriesByCategory.first?.name == "projection")
        #expect(decoded.summary.totalGpuMicroseconds == 25)

        let csv = try String(contentsOf: csvURL, encoding: .utf8)
        #expect(csv.contains("gemv_seq_bf16_f32s"))
        #expect(csv.contains("averageGpuMicroseconds"))
        let categories = try String(contentsOf: categoryCSVURL, encoding: .utf8)
        #expect(categories.contains("percentageOfGpu"))
        let kernels = try String(contentsOf: kernelCSVURL, encoding: .utf8)
        #expect(kernels.contains("gemv_seq_bf16_f32s,projection,1,1,25.000"))
        let layers = try String(contentsOf: layerCSVURL, encoding: .utf8)
        #expect(layers.contains("0,projection,1,1,25.000"))
        let weights = try String(contentsOf: weightCSVURL, encoding: .utf8)
        #expect(weights.contains("mlp.down_proj,projection,1,1,25.000"))
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
