import Testing
import LMIR
@testable import MetalCompiler

@Suite("Recurrent Block Fusion Windows")
struct RecurrentBlockFusionWindowTests {
    @Test("Admission scanner finds dispatch-entry linear attention windows")
    func admissionScannerFindsDispatchEntryWindows() {
        let entries = [
            dispatchInputProjection(index: 10, layer: 2),
            DispatchEntry(
                index: 11,
                fragment: SSMRecurrenceFragment(
                    headCount: 16,
                    groupCount: 4,
                    keyHeadDimension: 64,
                    valueHeadDimension: 64,
                    convKernelSize: 4
                ),
                layerIndex: 2
            ),
            DispatchEntry(
                index: 12,
                fragment: ElementwiseFragment(count: 256, kind: .geluGated),
                layerIndex: 2
            ),
            dispatchOutputProjection(index: 13, layer: 2),
            dispatchInputProjection(index: 20, layer: 6),
            dispatchOutputProjection(index: 21, layer: 7),
        ]

        let windows = RecurrentBlockFusionAdmissionScanner.linearAttentionWindows(in: entries)

        #expect(windows.count == 1)
        #expect(windows[0].layerIndex == 2)
        #expect(windows[0].range == 10..<14)
        #expect(windows[0].inputProjectionEntryIndex == 10)
        #expect(windows[0].recurrenceEntryIndex == 11)
        #expect(windows[0].bridgeEntryIndices == [12])
        #expect(windows[0].outputProjectionEntryIndex == 13)
        #expect(windows[0].inputProjectionFields == ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"])
        #expect(windows[0].outputProjectionField == "out_proj")
    }

    @Test("Admission scanner rejects incomplete or cross-layer dispatch windows")
    func admissionScannerRejectsIncompleteOrCrossLayerDispatchWindows() {
        let entries = [
            dispatchInputProjection(index: 0, layer: 0),
            dispatchOutputProjection(index: 1, layer: 0),
            dispatchInputProjection(index: 2, layer: 1),
            DispatchEntry(
                index: 3,
                fragment: SSMRecurrenceFragment(
                    headCount: 16,
                    groupCount: 4,
                    keyHeadDimension: 64,
                    valueHeadDimension: 64,
                    convKernelSize: 4
                ),
                layerIndex: 1
            ),
            dispatchOutputProjection(index: 4, layer: 2),
        ]

        let windows = RecurrentBlockFusionAdmissionScanner.linearAttentionWindows(in: entries)

        #expect(windows.isEmpty)
    }

    @Test("Prototype planner rejects single-dispatch fusion when output projection crosses recurrence groups")
    func prototypePlannerRejectsCrossGroupSingleDispatchFusion() throws {
        let entries = [
            dispatchInputProjection(index: 0, layer: 0, groups: 4),
            dispatchRecurrence(index: 1, layer: 0, groups: 4),
            dispatchOutputProjection(index: 2, layer: 0),
        ]
        let window = try #require(RecurrentBlockFusionAdmissionScanner.linearAttentionWindows(in: entries).first)

        let decision = RecurrentBlockFusionPrototypePlanner.singleDispatchDecision(
            for: window,
            entries: entries
        )

        #expect(decision == .rejected([.outputProjectionRequiresCrossGroupFanIn(partitionCount: 4)]))
    }

    @Test("Prototype planner accepts shape-matched single-group dispatch window")
    func prototypePlannerAcceptsSingleGroupDispatchWindow() throws {
        let entries = [
            dispatchInputProjection(index: 0, layer: 0, groups: 1),
            dispatchRecurrence(index: 1, layer: 0, groups: 1),
            dispatchOutputProjection(index: 2, layer: 0),
        ]
        let window = try #require(RecurrentBlockFusionAdmissionScanner.linearAttentionWindows(in: entries).first)

        let decision = RecurrentBlockFusionPrototypePlanner.singleDispatchDecision(
            for: window,
            entries: entries
        )

        #expect(decision == .eligible)
    }

    @Test("Prototype planner creates reference-gated two-stage plan for multi-group recurrent block")
    func prototypePlannerCreatesTwoStagePlanForMultiGroupWindow() throws {
        let entries = [
            dispatchInputProjection(index: 0, layer: 4, groups: 4),
            dispatchRecurrence(index: 1, layer: 4, groups: 4),
            dispatchOutputProjection(index: 2, layer: 4),
        ]
        let window = try #require(RecurrentBlockFusionAdmissionScanner.linearAttentionWindows(in: entries).first)

        let decision = RecurrentBlockFusionPrototypePlanner.twoStageDecision(
            for: window,
            entries: entries
        )

        #expect(decision == .candidate(RecurrentBlockFusionTwoStagePlan(
            layerIndex: 4,
            partitionCount: 4,
            headsPerPartition: 4,
            partitionInputDimension: 64,
            recurrentOutputDimension: 256,
            outputDimension: 2048,
            partialRowsPerToken: 8192,
            partialScratchBaseSlot: 1,
            partialScratchSlotCount: 4,
            requiredScratchSlotCount: 5,
            numericalContract: .referenceGated
        )))
    }

    @Test("Prototype planner rejects two-stage plan when single dispatch is preferred")
    func prototypePlannerRejectsTwoStagePlanForSingleGroupWindow() throws {
        let entries = [
            dispatchInputProjection(index: 0, layer: 0, groups: 1),
            dispatchRecurrence(index: 1, layer: 0, groups: 1),
            dispatchOutputProjection(index: 2, layer: 0),
        ]
        let window = try #require(RecurrentBlockFusionAdmissionScanner.linearAttentionWindows(in: entries).first)

        let decision = RecurrentBlockFusionPrototypePlanner.twoStageDecision(
            for: window,
            entries: entries
        )

        #expect(decision == .rejected([.singleDispatchPreferred(partitionCount: 1)]))
    }

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

    @Test("Scanner treats partial projection, storage round, and reduce as one logical output projection")
    func scannerIncludesPartialProjectionReduceOutputWindow() throws {
        let entries = [
            profileEntry(
                index: 0,
                kernelName: "batched_gemv4_seq_bf16_f32s",
                category: "projection",
                weightTensorName: linearAttentionInputWeights(layer: 5)
            ),
            profileEntry(
                index: 1,
                kernelName: "ssm_recurrence_seq_bf16_f32",
                category: "ssm_recurrence"
            ),
            profileEntry(
                index: 2,
                kernelName: "recurrent_block_partial_projection_seq_bf16_f32",
                category: "projection",
                weightTensorName: "model.language_model.layers.5.linear_attn.out_proj.weight"
            ),
            profileEntry(
                index: 3,
                kernelName: "round_bf16_seq_f32",
                category: "other"
            ),
            profileEntry(
                index: 4,
                kernelName: "recurrent_block_partial_reduce_seq_f32",
                category: "projection",
                weightTensorName: "model.language_model.layers.5.linear_attn.out_proj.weight"
            ),
            profileEntry(
                index: 5,
                kernelName: "batched_gemv2_seq_bf16_f32s",
                category: "projection",
                weightTensorName: [
                    "model.language_model.layers.5.mlp.gate_proj.weight",
                    "model.language_model.layers.5.mlp.up_proj.weight",
                ].joined(separator: ";")
            ),
        ]

        let window = try #require(RecurrentBlockFusionWindowScanner.linearAttentionWindows(in: entries).first)

        #expect(window.layerIndex == 5)
        #expect(window.range == 0..<5)
        #expect(window.outputProjectionStepIndex == 4)
        #expect(window.outputProjectionStepIndices == [2, 4])
        #expect(window.outputProjectionKernelName == "recurrent_block_partial_reduce_seq_f32")
        #expect(window.outputProjectionKernelNames == [
            "recurrent_block_partial_projection_seq_bf16_f32",
            "recurrent_block_partial_reduce_seq_f32",
        ])
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
        #expect(csv.contains("windowEntryCount,totalGpuMicroseconds"))
        #expect(csv.contains("fusedStageCandidate,currentReplaceableStepCount,targetFusedStageStepCount,estimatedDispatchReduction"))
        #expect(csv.contains("3,0,4,0,1,2,3,3"))
        #expect(csv.contains("4,4.000,1.000,1.000,1.000,1.000,0,true,3,2,1"))
    }

    private func linearAttentionInputWeights(layer: Int) -> String {
        [
            "model.language_model.layers.\(layer).linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.\(layer).linear_attn.in_proj_z.weight",
            "model.language_model.layers.\(layer).linear_attn.in_proj_b.weight",
            "model.language_model.layers.\(layer).linear_attn.in_proj_a.weight",
        ].joined(separator: ";")
    }

    private func dispatchInputProjection(index: Int, layer: Int, groups: Int = 4) -> DispatchEntry {
        DispatchEntry(
            index: index,
            fragment: BatchedProjection(projections: [
                .init(field: "in_proj_qkv", inputDimension: 2048, outputDimension: (2 * groups * 64) + 256),
                .init(field: "in_proj_z", inputDimension: 2048, outputDimension: 256),
                .init(field: "in_proj_b", inputDimension: 2048, outputDimension: 16),
                .init(field: "in_proj_a", inputDimension: 2048, outputDimension: 16),
            ]),
            parameterBindings: [
                .init(role: "in_proj_qkv", tensorName: "model.language_model.layers.\(layer).linear_attn.in_proj_qkv.weight"),
                .init(role: "in_proj_z", tensorName: "model.language_model.layers.\(layer).linear_attn.in_proj_z.weight"),
                .init(role: "in_proj_b", tensorName: "model.language_model.layers.\(layer).linear_attn.in_proj_b.weight"),
                .init(role: "in_proj_a", tensorName: "model.language_model.layers.\(layer).linear_attn.in_proj_a.weight"),
            ],
            layerIndex: layer
        )
    }

    private func dispatchRecurrence(index: Int, layer: Int, groups: Int) -> DispatchEntry {
        DispatchEntry(
            index: index,
            fragment: SSMRecurrenceFragment(
                headCount: 16,
                groupCount: groups,
                keyHeadDimension: 64,
                valueHeadDimension: 16,
                convKernelSize: 4
            ),
            layerIndex: layer
        )
    }

    private func dispatchOutputProjection(index: Int, layer: Int) -> DispatchEntry {
        DispatchEntry(
            index: index,
            fragment: LinearFragment(
                field: "out_proj",
                inputDimension: 256,
                outputDimension: 2048,
                isOutput: true
            ),
            parameterBindings: [
                .init(role: "out_proj", tensorName: "model.language_model.layers.\(layer).linear_attn.out_proj.weight"),
            ],
            layerIndex: layer
        )
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
