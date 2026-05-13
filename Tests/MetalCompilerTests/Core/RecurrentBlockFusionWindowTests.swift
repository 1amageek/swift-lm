import Testing
@testable import MetalCompiler

@Suite("Recurrent Block Fusion Windows")
struct RecurrentBlockFusionWindowTests {
    @Test("Scanner finds linear attention recurrent block windows")
    func scannerFindsLinearAttentionWindows() {
        let entries = [
            profileEntry(
                index: 0,
                kernelName: "batched_gemv4_seq_bf16_f32s",
                category: "projection",
                weightTensorName: linearAttentionInputWeights(layer: 0)
            ),
            profileEntry(
                index: 1,
                kernelName: "ssm_recurrence_seq_bf16_f32",
                category: "ssm_recurrence"
            ),
            profileEntry(
                index: 2,
                kernelName: "round_bf16_seq_f32",
                category: "other"
            ),
            profileEntry(
                index: 3,
                kernelName: "gemv_seq_bf16_f32s",
                category: "projection",
                weightTensorName: "model.language_model.layers.0.linear_attn.out_proj.weight"
            ),
            profileEntry(
                index: 4,
                kernelName: "batched_gemv2_seq_bf16_f32s",
                category: "projection",
                weightTensorName: [
                    "model.language_model.layers.0.mlp.gate_proj.weight",
                    "model.language_model.layers.0.mlp.up_proj.weight",
                ].joined(separator: ";")
            ),
            profileEntry(
                index: 5,
                kernelName: "batched_gemv4_seq_bf16_f32s",
                category: "projection",
                weightTensorName: linearAttentionInputWeights(layer: 4)
            ),
            profileEntry(
                index: 6,
                kernelName: "ssm_recurrence_seq_bf16_f32_prewrite_decay",
                category: "ssm_recurrence"
            ),
            profileEntry(
                index: 7,
                kernelName: "gemv_seq_bf16_f32s",
                category: "projection",
                weightTensorName: "model.language_model.layers.4.linear_attn.out_proj.weight"
            ),
        ]

        let windows = RecurrentBlockFusionWindowScanner.linearAttentionWindows(in: entries)

        #expect(windows.count == 2)
        #expect(windows[0].layerIndex == 0)
        #expect(windows[0].range == 0..<4)
        #expect(windows[0].inputProjectionStepIndex == 0)
        #expect(windows[0].recurrenceStepIndex == 1)
        #expect(windows[0].bridgeStepIndices == [2])
        #expect(windows[0].outputProjectionStepIndex == 3)
        #expect(windows[1].layerIndex == 4)
        #expect(windows[1].range == 5..<8)
        #expect(windows[1].recurrenceKernelName == "ssm_recurrence_seq_bf16_f32_prewrite_decay")
    }

    @Test("Scanner rejects incomplete or cross-layer windows")
    func scannerRejectsIncompleteOrCrossLayerWindows() {
        let entries = [
            profileEntry(
                index: 0,
                kernelName: "batched_gemv4_seq_bf16_f32s",
                category: "projection",
                weightTensorName: linearAttentionInputWeights(layer: 0)
            ),
            profileEntry(
                index: 1,
                kernelName: "gemv_seq_bf16_f32s",
                category: "projection",
                weightTensorName: "model.language_model.layers.1.linear_attn.out_proj.weight"
            ),
            profileEntry(
                index: 2,
                kernelName: "batched_gemv4_seq_bf16_f32s",
                category: "projection",
                weightTensorName: linearAttentionInputWeights(layer: 2)
            ),
            profileEntry(
                index: 3,
                kernelName: "ssm_recurrence_seq_bf16_f32",
                category: "ssm_recurrence"
            ),
        ]

        let windows = RecurrentBlockFusionWindowScanner.linearAttentionWindows(in: entries)

        #expect(windows.isEmpty)
    }

    @Test("Profile emits recurrent block window CSV")
    func profileEmitsRecurrentBlockWindowCSV() {
        let profile = MetalPrefillProfile(
            profileKind: "step",
            sequenceLength: 128,
            maximumSequenceLength: 128,
            iterations: 1,
            warmupIterations: 0,
            stepCount: 4,
            entries: [
                profileEntry(
                    index: 0,
                    kernelName: "batched_gemv4_seq_bf16_f32s",
                    category: "projection",
                    weightTensorName: linearAttentionInputWeights(layer: 3)
                ),
                profileEntry(
                    index: 1,
                    kernelName: "ssm_recurrence_seq_bf16_f32",
                    category: "ssm_recurrence"
                ),
                profileEntry(
                    index: 2,
                    kernelName: "round_bf16_seq_f32",
                    category: "other"
                ),
                profileEntry(
                    index: 3,
                    kernelName: "gemv_seq_bf16_f32s",
                    category: "projection",
                    weightTensorName: "model.language_model.layers.3.linear_attn.out_proj.weight"
                ),
            ],
            generatedAt: "2026-05-13T00:00:00Z"
        )

        let csv = profile.recurrentBlockWindowCSVString

        #expect(csv.contains("layerIndex,rangeStart,rangeEnd"))
        #expect(csv.contains("3,0,4,0,1,2,3"))
    }

    private func linearAttentionInputWeights(layer: Int) -> String {
        [
            "model.language_model.layers.\(layer).linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.\(layer).linear_attn.in_proj_z.weight",
            "model.language_model.layers.\(layer).linear_attn.in_proj_b.weight",
            "model.language_model.layers.\(layer).linear_attn.in_proj_a.weight",
        ].joined(separator: ";")
    }

    private func profileEntry(
        index: Int,
        kernelName: String,
        category: String,
        weightTensorName: String? = nil
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
            estimatedWriteBytes: 0,
            estimatedTotalBytes: 0,
            estimatedDispatchCount: 1,
            totalGpuMicroseconds: 1,
            averageGpuMicroseconds: 1,
            totalWallMicroseconds: 1,
            averageWallMicroseconds: 1
        )
    }
}
