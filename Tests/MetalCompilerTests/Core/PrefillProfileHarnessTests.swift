import Foundation
import Testing
@testable import MetalCompiler

@Suite("Prefill Profile Harness")
struct PrefillProfileHarnessTests {
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

        #expect(urls.count == 2)
        let jsonURL = root.appendingPathComponent("synthetic.json")
        let csvURL = root.appendingPathComponent("synthetic.csv")
        let jsonData = try Data(contentsOf: jsonURL)
        let decoded = try JSONDecoder().decode(MetalPrefillProfile.self, from: jsonData)
        #expect(decoded.summary.entriesByCategory.first?.name == "projection")
        #expect(decoded.summary.totalGpuMicroseconds == 25)

        let csv = try String(contentsOf: csvURL, encoding: .utf8)
        #expect(csv.contains("gemv_seq_bf16_f32s"))
        #expect(csv.contains("averageGpuMicroseconds"))
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
