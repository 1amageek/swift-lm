import Foundation
import Testing
@testable import MetalCompiler

@Suite("MLP Fusion Window")
struct MLPFusionWindowTests {
    @Test("Scanner finds unfused SwiGLU down windows")
    func scannerFindsUnfusedSwigluDownWindows() {
        let entries = [
            profileEntry(
                index: 0,
                kernelName: "batched_gemv2_seq_bf16_f32s",
                category: "projection",
                weightTensorName: gateUpWeights(layer: 2)
            ),
            profileEntry(
                index: 1,
                kernelName: "swiglu_seq_f32",
                category: "elementwise"
            ),
            profileEntry(
                index: 2,
                kernelName: "gemv_seq_bf16_f32s",
                category: "projection",
                weightTensorName: "model.language_model.layers.2.mlp.down_proj.weight"
            ),
        ]

        let windows = MLPFusionWindowScanner.swigluDownWindows(in: entries)

        #expect(windows.count == 1)
        #expect(windows[0].layerIndex == 2)
        #expect(windows[0].range == 0..<3)
        #expect(windows[0].gateUpProjectionStepIndex == 0)
        #expect(windows[0].activationStepIndex == 1)
        #expect(windows[0].downProjectionStepIndex == 2)
        #expect(windows[0].route == .unfusedSwigluDown)
    }

    @Test("Scanner finds fused SwiGLU down windows")
    func scannerFindsFusedSwigluDownWindows() {
        let entries = [
            profileEntry(
                index: 0,
                kernelName: "batched_gemv2_seq_bf16_f32s",
                category: "projection",
                weightTensorName: gateUpWeights(layer: 5)
            ),
            profileEntry(
                index: 1,
                kernelName: "mlp_fused_swiglu_down_seq_bf16_f32s",
                category: "projection",
                weightTensorName: "model.language_model.layers.5.mlp.down_proj.weight"
            ),
            profileEntry(
                index: 2,
                kernelName: "batched_gemv2_seq_bf16_f32s",
                category: "projection",
                weightTensorName: gateUpWeights(layer: 6)
            ),
        ]

        let windows = MLPFusionWindowScanner.swigluDownWindows(in: entries)

        #expect(windows.count == 1)
        #expect(windows[0].layerIndex == 5)
        #expect(windows[0].range == 0..<2)
        #expect(windows[0].activationStepIndex == nil)
        #expect(windows[0].downProjectionStepIndex == 1)
        #expect(windows[0].route == .fusedSwigluDown)
    }

    @Test("Scanner rejects incomplete and cross-layer windows")
    func scannerRejectsIncompleteAndCrossLayerWindows() {
        let entries = [
            profileEntry(
                index: 0,
                kernelName: "batched_gemv2_seq_bf16_f32s",
                category: "projection",
                weightTensorName: gateUpWeights(layer: 0)
            ),
            profileEntry(
                index: 1,
                kernelName: "swiglu_seq_f32",
                category: "elementwise"
            ),
            profileEntry(
                index: 2,
                kernelName: "gemv_seq_bf16_f32s",
                category: "projection",
                weightTensorName: "model.language_model.layers.1.mlp.down_proj.weight"
            ),
        ]

        let windows = MLPFusionWindowScanner.swigluDownWindows(in: entries)

        #expect(windows.isEmpty)
    }

    @Test("Profile emits MLP fusion window CSV")
    func profileEmitsMLPFusionWindowCSV() {
        let profile = MetalPrefillProfile(
            profileKind: "step",
            sequenceLength: 128,
            maximumSequenceLength: 128,
            iterations: 1,
            warmupIterations: 0,
            stepCount: 3,
            entries: [
                profileEntry(
                    index: 0,
                    kernelName: "batched_gemv2_seq_bf16_f32s",
                    category: "projection",
                    weightTensorName: gateUpWeights(layer: 3),
                    totalGpuMicroseconds: 2,
                    estimatedTotalBytes: 20
                ),
                profileEntry(
                    index: 1,
                    kernelName: "swiglu_seq_f32",
                    category: "elementwise",
                    totalGpuMicroseconds: 1,
                    estimatedTotalBytes: 10
                ),
                profileEntry(
                    index: 2,
                    kernelName: "gemv_seq_bf16_f32s",
                    category: "projection",
                    weightTensorName: "model.language_model.layers.3.mlp.down_proj.weight",
                    totalGpuMicroseconds: 4,
                    estimatedTotalBytes: 40
                ),
            ],
            generatedAt: "2026-05-14T00:00:00Z"
        )

        let csv = profile.mlpFusionWindowCSVString

        #expect(csv.contains("layerIndex,rangeStart,rangeEnd"))
        #expect(csv.contains("gateUpProjectionGpuMicroseconds,activationGpuMicroseconds,downProjectionGpuMicroseconds"))
        #expect(csv.contains("3,0,3,0,1,2,batched_gemv2_seq_bf16_f32s,swiglu_seq_f32,gemv_seq_bf16_f32s,unfused_swiglu_down,3,7.000,2.000,1.000,4.000,70"))
    }

    private func gateUpWeights(layer: Int) -> String {
        [
            "model.language_model.layers.\(layer).mlp.gate_proj.weight",
            "model.language_model.layers.\(layer).mlp.up_proj.weight",
        ].joined(separator: ";")
    }

    private func profileEntry(
        index: Int,
        kernelName: String,
        category: String,
        weightTensorName: String? = nil,
        totalGpuMicroseconds: Double = 1,
        estimatedTotalBytes: Int = 0
    ) -> MetalPrefillProfile.Entry {
        MetalPrefillProfile.Entry(
            scope: "step",
            index: index,
            rangeStart: index,
            rangeEnd: index + 1,
            kernelName: kernelName,
            category: category,
            mode: "batch",
            layerIndex: nil,
            entryIndex: index,
            weightTensorName: weightTensorName,
            gridWidth: 1,
            gridHeight: 1,
            gridDepth: 1,
            threadgroupWidth: 1,
            threadgroupHeight: 1,
            threadgroupDepth: 1,
            threadgroupMemoryBytes: 0,
            bufferBindingCount: 0,
            inlineConstantBytes: 0,
            uniqueBoundBufferBytes: 0,
            estimatedReadBytes: 0,
            estimatedWriteBytes: estimatedTotalBytes,
            estimatedTotalBytes: estimatedTotalBytes,
            estimatedDispatchCount: 1,
            totalGpuMicroseconds: totalGpuMicroseconds,
            averageGpuMicroseconds: totalGpuMicroseconds,
            totalWallMicroseconds: totalGpuMicroseconds,
            averageWallMicroseconds: totalGpuMicroseconds
        )
    }
}
