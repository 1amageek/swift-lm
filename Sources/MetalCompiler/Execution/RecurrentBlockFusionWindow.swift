import Foundation
import LMIR

struct RecurrentBlockFusionWindow: Sendable, Equatable {
    let layerIndex: Int
    let rangeStart: Int
    let rangeEnd: Int
    let inputProjectionStepIndex: Int
    let recurrenceStepIndex: Int
    let bridgeStepIndices: [Int]
    let outputProjectionStepIndex: Int
    let outputProjectionStepIndices: [Int]
    let inputProjectionKernelName: String
    let recurrenceKernelName: String
    let outputProjectionKernelName: String
    let outputProjectionKernelNames: [String]

    var range: Range<Int> {
        rangeStart..<rangeEnd
    }
}

struct RecurrentBlockFusionAdmissionWindow: Sendable, Equatable {
    let layerIndex: Int
    let rangeStart: Int
    let rangeEnd: Int
    let inputProjectionEntryIndex: Int
    let recurrenceEntryIndex: Int
    let bridgeEntryIndices: [Int]
    let outputProjectionEntryIndex: Int
    let inputProjectionFields: [String]
    let outputProjectionField: String

    var range: Range<Int> {
        rangeStart..<rangeEnd
    }
}

enum RecurrentBlockFusionAdmissionScanner {
    static func linearAttentionWindows(
        in entries: [DispatchEntry]
    ) -> [RecurrentBlockFusionAdmissionWindow] {
        let orderedEntries = entries.sorted { lhs, rhs in
            lhs.index < rhs.index
        }
        var windows: [RecurrentBlockFusionAdmissionWindow] = []
        var cursor = 0
        while cursor < orderedEntries.count {
            let inputEntry = orderedEntries[cursor]
            guard let inputProjection = linearAttentionInputProjection(inputEntry),
                  let inputLayerIndex = inputEntry.layerIndex ?? layerIndex(from: inputEntry.parameterBindings) else {
                cursor += 1
                continue
            }

            var recurrenceEntry: Optional<DispatchEntry> = Optional.none
            var bridgeEntries: [DispatchEntry] = []
            var outputEntry: Optional<DispatchEntry> = Optional.none
            var scanIndex = cursor + 1

            while scanIndex < orderedEntries.count {
                let current = orderedEntries[scanIndex]
                if linearAttentionInputProjection(current) != nil {
                    break
                }
                if recurrenceEntry == nil, isLinearAttentionRecurrence(current) {
                    recurrenceEntry = current
                    scanIndex += 1
                    continue
                }
                if let recurrenceEntry, isLinearAttentionOutputProjection(current),
                   (current.layerIndex ?? layerIndex(from: current.parameterBindings)) == inputLayerIndex {
                    outputEntry = current
                    let bridgeRange = orderedEntries[(cursor + 1)..<scanIndex]
                    bridgeEntries = bridgeRange.filter { entry in
                        entry.index != recurrenceEntry.index
                    }
                    break
                }
                scanIndex += 1
            }

            if let recurrenceEntry, let outputEntry,
               let outputProjection = outputEntry.fragment as? LinearFragment {
                windows.append(
                    RecurrentBlockFusionAdmissionWindow(
                        layerIndex: inputLayerIndex,
                        rangeStart: inputEntry.index,
                        rangeEnd: outputEntry.index + 1,
                        inputProjectionEntryIndex: inputEntry.index,
                        recurrenceEntryIndex: recurrenceEntry.index,
                        bridgeEntryIndices: bridgeEntries.map(\.index),
                        outputProjectionEntryIndex: outputEntry.index,
                        inputProjectionFields: inputProjection.projections.map(\.field),
                        outputProjectionField: outputProjection.field
                    )
                )
                cursor = scanIndex + 1
            } else {
                cursor += 1
            }
        }
        return windows
    }

    private static func linearAttentionInputProjection(_ entry: DispatchEntry) -> BatchedProjection? {
        guard let projection = entry.fragment as? BatchedProjection else {
            return nil
        }
        let fields = Set(projection.projections.map(\.field))
        guard fields == Set(["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]) else {
            return nil
        }
        let tensorNames = entry.parameterBindings.map(\.tensorName)
        guard tensorNames.contains(where: { $0.contains("linear_attn.in_proj_qkv.weight") }),
              tensorNames.contains(where: { $0.contains("linear_attn.in_proj_z.weight") }),
              tensorNames.contains(where: { $0.contains("linear_attn.in_proj_b.weight") }),
              tensorNames.contains(where: { $0.contains("linear_attn.in_proj_a.weight") }) else {
            return nil
        }
        return projection
    }

    private static func isLinearAttentionRecurrence(_ entry: DispatchEntry) -> Bool {
        entry.fragment is SSMRecurrenceFragment
    }

    private static func isLinearAttentionOutputProjection(_ entry: DispatchEntry) -> Bool {
        guard let projection = entry.fragment as? LinearFragment,
              projection.field == "out_proj",
              projection.isOutput else {
            return false
        }
        return entry.parameterBindings.contains {
            $0.tensorName.contains("linear_attn.out_proj.weight")
        }
    }

    private static func layerIndex(from parameterBindings: [LMIR.ParameterBinding]) -> Int? {
        for binding in parameterBindings {
            if let layer = layerIndex(from: binding.tensorName) {
                return layer
            }
        }
        return nil
    }

    private static func layerIndex(from tensorName: String) -> Int? {
        let components = tensorName.split(separator: ".").map(String.init)
        guard let layerTokenIndex = components.firstIndex(of: "layers"),
              layerTokenIndex + 1 < components.count else {
            return nil
        }
        return Int(components[layerTokenIndex + 1])
    }
}

enum RecurrentBlockFusionSingleDispatchRejection: Sendable, Equatable {
    case missingInputProjection(entryIndex: Int)
    case missingRecurrence(entryIndex: Int)
    case missingOutputProjection(entryIndex: Int)
    case inputProjectionShapeMismatch(expectedConvDimension: Int, actualQKVDimension: Int)
    case gateProjectionShapeMismatch(expectedOutputDimension: Int, actualZDimension: Int)
    case outputProjectionShapeMismatch(expectedInputDimension: Int, actualInputDimension: Int)
    case outputProjectionRequiresCrossGroupFanIn(partitionCount: Int)
}

enum RecurrentBlockFusionSingleDispatchDecision: Sendable, Equatable {
    case eligible
    case rejected([RecurrentBlockFusionSingleDispatchRejection])
}

enum RecurrentBlockFusionNumericalContract: Sendable, Equatable {
    case strictDecodeEquivalent
    case referenceGated
}

struct RecurrentBlockFusionTwoStagePlan: Sendable, Equatable {
    let layerIndex: Int
    let partitionCount: Int
    let headsPerPartition: Int
    let partitionInputDimension: Int
    let recurrentOutputDimension: Int
    let outputDimension: Int
    let partialRowsPerToken: Int
    let partialScratchBaseSlot: Int
    let partialScratchSlotCount: Int
    let requiredScratchSlotCount: Int
    let numericalContract: RecurrentBlockFusionNumericalContract
}

enum RecurrentBlockFusionTwoStageRejection: Sendable, Equatable {
    case missingInputProjection(entryIndex: Int)
    case missingRecurrence(entryIndex: Int)
    case missingOutputProjection(entryIndex: Int)
    case singleDispatchPreferred(partitionCount: Int)
    case inputProjectionShapeMismatch(expectedConvDimension: Int, actualQKVDimension: Int)
    case gateProjectionShapeMismatch(expectedOutputDimension: Int, actualZDimension: Int)
    case outputProjectionShapeMismatch(expectedInputDimension: Int, actualInputDimension: Int)
    case unevenHeadPartition(headCount: Int, partitionCount: Int)
}

enum RecurrentBlockFusionTwoStageDecision: Sendable, Equatable {
    case candidate(RecurrentBlockFusionTwoStagePlan)
    case rejected([RecurrentBlockFusionTwoStageRejection])
}

enum RecurrentBlockFusionFusedStageExecutionShape: String, Sendable, Equatable {
    case groupOwnedStateUpdateThenPartialRows = "group-owned-state-update-then-partial-rows"
    case partialPartitionOwnedStateUpdatesThenPartialRows = "partial-partition-owned-state-updates-then-partial-rows"
}

struct RecurrentBlockFusionFusedStagePlan: Sendable, Equatable {
    let layerIndex: Int
    let partitionCount: Int
    let recurrentGroupsPerPartition: Int
    let headsPerPartition: Int
    let partitionInputDimension: Int
    let recurrentOutputDimension: Int
    let outputDimension: Int
    let currentReplaceableStepCount: Int
    let targetFusedStageStepCount: Int
    let estimatedDispatchReduction: Int
    let executionShape: RecurrentBlockFusionFusedStageExecutionShape
    let unsafeRowGridFusionAllowed: Bool
    let numericalContract: RecurrentBlockFusionNumericalContract
}

enum RecurrentBlockFusionFusedStageRejection: Sendable, Equatable {
    case missingInputProjection(entryIndex: Int)
    case missingRecurrence(entryIndex: Int)
    case missingOutputProjection(entryIndex: Int)
    case singleDispatchPreferred(partitionCount: Int)
    case inputProjectionShapeMismatch(expectedConvDimension: Int, actualQKVDimension: Int)
    case gateProjectionShapeMismatch(expectedOutputDimension: Int, actualZDimension: Int)
    case outputProjectionShapeMismatch(expectedInputDimension: Int, actualInputDimension: Int)
    case unevenHeadPartition(headCount: Int, partitionCount: Int)
    case noScratchCompatiblePartition(groupCount: Int, maximumPartialScratchSlotCount: Int)
    case unevenRecurrentGroupPartition(groupCount: Int, partitionCount: Int)
    case noDispatchReduction(currentStepCount: Int, targetStepCount: Int)
}

enum RecurrentBlockFusionFusedStageDecision: Sendable, Equatable {
    case candidate(RecurrentBlockFusionFusedStagePlan)
    case rejected([RecurrentBlockFusionFusedStageRejection])
}

enum RecurrentBlockFusionPrototypePlanner {
    private static let maximumPartialScratchSlotCount = 4
    private static let fusedStageStepCount = 2

    private static func largestDivisor(of value: Int, notExceeding maximum: Int) -> Int {
        guard value > 0, maximum > 0 else { return 0 }
        for candidate in stride(from: min(value, maximum), through: 1, by: -1) {
            if value % candidate == 0 {
                return candidate
            }
        }
        return 0
    }

    static func singleDispatchDecision(
        for window: RecurrentBlockFusionAdmissionWindow,
        entries: [DispatchEntry]
    ) -> RecurrentBlockFusionSingleDispatchDecision {
        let byIndex = Dictionary(uniqueKeysWithValues: entries.map { ($0.index, $0) })
        var rejections: [RecurrentBlockFusionSingleDispatchRejection] = []

        guard let inputEntry = byIndex[window.inputProjectionEntryIndex],
              let inputProjection = inputEntry.fragment as? BatchedProjection else {
            return .rejected([.missingInputProjection(entryIndex: window.inputProjectionEntryIndex)])
        }
        guard let recurrenceEntry = byIndex[window.recurrenceEntryIndex],
              let recurrence = recurrenceEntry.fragment as? SSMRecurrenceFragment else {
            return .rejected([.missingRecurrence(entryIndex: window.recurrenceEntryIndex)])
        }
        guard let outputEntry = byIndex[window.outputProjectionEntryIndex],
              let outputProjection = outputEntry.fragment as? LinearFragment else {
            return .rejected([.missingOutputProjection(entryIndex: window.outputProjectionEntryIndex)])
        }

        let fields = Dictionary(uniqueKeysWithValues: inputProjection.projections.map { ($0.field, $0) })
        if let qkv = fields["in_proj_qkv"], qkv.outputDimension != recurrence.convDimension {
            rejections.append(.inputProjectionShapeMismatch(
                expectedConvDimension: recurrence.convDimension,
                actualQKVDimension: qkv.outputDimension
            ))
        }
        let recurrentOutputDimension = recurrence.headCount * recurrence.valueHeadDimension
        if let z = fields["in_proj_z"], z.outputDimension != recurrentOutputDimension {
            rejections.append(.gateProjectionShapeMismatch(
                expectedOutputDimension: recurrentOutputDimension,
                actualZDimension: z.outputDimension
            ))
        }
        if outputProjection.inputDimension != recurrentOutputDimension {
            rejections.append(.outputProjectionShapeMismatch(
                expectedInputDimension: recurrentOutputDimension,
                actualInputDimension: outputProjection.inputDimension
            ))
        }

        if recurrence.groupCount > 1 {
            rejections.append(.outputProjectionRequiresCrossGroupFanIn(
                partitionCount: recurrence.groupCount
            ))
        }

        return rejections.isEmpty ? .eligible : .rejected(rejections)
    }

    static func twoStageDecision(
        for window: RecurrentBlockFusionAdmissionWindow,
        entries: [DispatchEntry]
    ) -> RecurrentBlockFusionTwoStageDecision {
        let byIndex = Dictionary(uniqueKeysWithValues: entries.map { ($0.index, $0) })
        var rejections: [RecurrentBlockFusionTwoStageRejection] = []

        guard let inputEntry = byIndex[window.inputProjectionEntryIndex],
              let inputProjection = inputEntry.fragment as? BatchedProjection else {
            return .rejected([.missingInputProjection(entryIndex: window.inputProjectionEntryIndex)])
        }
        guard let recurrenceEntry = byIndex[window.recurrenceEntryIndex],
              let recurrence = recurrenceEntry.fragment as? SSMRecurrenceFragment else {
            return .rejected([.missingRecurrence(entryIndex: window.recurrenceEntryIndex)])
        }
        guard let outputEntry = byIndex[window.outputProjectionEntryIndex],
              let outputProjection = outputEntry.fragment as? LinearFragment else {
            return .rejected([.missingOutputProjection(entryIndex: window.outputProjectionEntryIndex)])
        }

        if recurrence.groupCount <= 1 {
            rejections.append(.singleDispatchPreferred(partitionCount: recurrence.groupCount))
        }
        if recurrence.headCount % max(recurrence.groupCount, 1) != 0 {
            rejections.append(.unevenHeadPartition(
                headCount: recurrence.headCount,
                partitionCount: recurrence.groupCount
            ))
        }

        let fields = Dictionary(uniqueKeysWithValues: inputProjection.projections.map { ($0.field, $0) })
        if let qkv = fields["in_proj_qkv"], qkv.outputDimension != recurrence.convDimension {
            rejections.append(.inputProjectionShapeMismatch(
                expectedConvDimension: recurrence.convDimension,
                actualQKVDimension: qkv.outputDimension
            ))
        }
        let recurrentOutputDimension = recurrence.headCount * recurrence.valueHeadDimension
        if let z = fields["in_proj_z"], z.outputDimension != recurrentOutputDimension {
            rejections.append(.gateProjectionShapeMismatch(
                expectedOutputDimension: recurrentOutputDimension,
                actualZDimension: z.outputDimension
            ))
        }
        if outputProjection.inputDimension != recurrentOutputDimension {
            rejections.append(.outputProjectionShapeMismatch(
                expectedInputDimension: recurrentOutputDimension,
                actualInputDimension: outputProjection.inputDimension
            ))
        }

        guard rejections.isEmpty else {
            return .rejected(rejections)
        }

        let partitionCount = largestDivisor(
            of: recurrence.groupCount,
            notExceeding: maximumPartialScratchSlotCount
        )
        guard partitionCount > 1, recurrentOutputDimension % partitionCount == 0 else {
            return .rejected([.singleDispatchPreferred(partitionCount: partitionCount)])
        }
        let headsPerPartition = recurrence.headCount / partitionCount
        let partitionInputDimension = recurrentOutputDimension / partitionCount
        return .candidate(RecurrentBlockFusionTwoStagePlan(
            layerIndex: window.layerIndex,
            partitionCount: partitionCount,
            headsPerPartition: headsPerPartition,
            partitionInputDimension: partitionInputDimension,
            recurrentOutputDimension: recurrentOutputDimension,
            outputDimension: outputProjection.outputDimension,
            partialRowsPerToken: partitionCount * outputProjection.outputDimension,
            partialScratchBaseSlot: 1,
            partialScratchSlotCount: partitionCount,
            requiredScratchSlotCount: 1 + partitionCount,
            numericalContract: .referenceGated
        ))
    }

    static func fusedStageDecision(
        for window: RecurrentBlockFusionAdmissionWindow,
        entries: [DispatchEntry],
        implicitBridgeStepCount: Int = 0
    ) -> RecurrentBlockFusionFusedStageDecision {
        let byIndex = Dictionary(uniqueKeysWithValues: entries.map { ($0.index, $0) })
        var rejections: [RecurrentBlockFusionFusedStageRejection] = []

        guard let inputEntry = byIndex[window.inputProjectionEntryIndex],
              let inputProjection = inputEntry.fragment as? BatchedProjection else {
            return .rejected([.missingInputProjection(entryIndex: window.inputProjectionEntryIndex)])
        }
        guard let recurrenceEntry = byIndex[window.recurrenceEntryIndex],
              let recurrence = recurrenceEntry.fragment as? SSMRecurrenceFragment else {
            return .rejected([.missingRecurrence(entryIndex: window.recurrenceEntryIndex)])
        }
        guard let outputEntry = byIndex[window.outputProjectionEntryIndex],
              let outputProjection = outputEntry.fragment as? LinearFragment else {
            return .rejected([.missingOutputProjection(entryIndex: window.outputProjectionEntryIndex)])
        }

        if recurrence.groupCount <= 1 {
            rejections.append(.singleDispatchPreferred(partitionCount: recurrence.groupCount))
        }
        if recurrence.headCount % max(recurrence.groupCount, 1) != 0 {
            rejections.append(.unevenHeadPartition(
                headCount: recurrence.headCount,
                partitionCount: recurrence.groupCount
            ))
        }

        let fields = Dictionary(uniqueKeysWithValues: inputProjection.projections.map { ($0.field, $0) })
        if let qkv = fields["in_proj_qkv"], qkv.outputDimension != recurrence.convDimension {
            rejections.append(.inputProjectionShapeMismatch(
                expectedConvDimension: recurrence.convDimension,
                actualQKVDimension: qkv.outputDimension
            ))
        }
        let recurrentOutputDimension = recurrence.headCount * recurrence.valueHeadDimension
        if let z = fields["in_proj_z"], z.outputDimension != recurrentOutputDimension {
            rejections.append(.gateProjectionShapeMismatch(
                expectedOutputDimension: recurrentOutputDimension,
                actualZDimension: z.outputDimension
            ))
        }
        if outputProjection.inputDimension != recurrentOutputDimension {
            rejections.append(.outputProjectionShapeMismatch(
                expectedInputDimension: recurrentOutputDimension,
                actualInputDimension: outputProjection.inputDimension
            ))
        }

        let partitionCount = largestDivisor(
            of: recurrence.groupCount,
            notExceeding: maximumPartialScratchSlotCount
        )
        if partitionCount <= 1 || recurrentOutputDimension % max(partitionCount, 1) != 0 {
            rejections.append(.noScratchCompatiblePartition(
                groupCount: recurrence.groupCount,
                maximumPartialScratchSlotCount: maximumPartialScratchSlotCount
            ))
        }
        if partitionCount > 1, recurrence.groupCount % partitionCount != 0 {
            rejections.append(.unevenRecurrentGroupPartition(
                groupCount: recurrence.groupCount,
                partitionCount: partitionCount
            ))
        }

        let currentReplaceableStepCount = 1 + window.bridgeEntryIndices.count + implicitBridgeStepCount + 1
        let estimatedDispatchReduction = currentReplaceableStepCount - fusedStageStepCount
        if estimatedDispatchReduction <= 0 {
            rejections.append(.noDispatchReduction(
                currentStepCount: currentReplaceableStepCount,
                targetStepCount: fusedStageStepCount
            ))
        }

        guard rejections.isEmpty else {
            return .rejected(rejections)
        }

        let recurrentGroupsPerPartition = recurrence.groupCount / partitionCount
        let headsPerPartition = recurrence.headCount / partitionCount
        let partitionInputDimension = recurrentOutputDimension / partitionCount
        let executionShape: RecurrentBlockFusionFusedStageExecutionShape = recurrentGroupsPerPartition == 1
            ? .groupOwnedStateUpdateThenPartialRows
            : .partialPartitionOwnedStateUpdatesThenPartialRows
        return .candidate(RecurrentBlockFusionFusedStagePlan(
            layerIndex: window.layerIndex,
            partitionCount: partitionCount,
            recurrentGroupsPerPartition: recurrentGroupsPerPartition,
            headsPerPartition: headsPerPartition,
            partitionInputDimension: partitionInputDimension,
            recurrentOutputDimension: recurrentOutputDimension,
            outputDimension: outputProjection.outputDimension,
            currentReplaceableStepCount: currentReplaceableStepCount,
            targetFusedStageStepCount: fusedStageStepCount,
            estimatedDispatchReduction: estimatedDispatchReduction,
            executionShape: executionShape,
            unsafeRowGridFusionAllowed: false,
            numericalContract: .referenceGated
        ))
    }
}

enum RecurrentBlockFusionWindowScanner {
    static func linearAttentionWindows(
        in entries: [MetalPrefillProfile.Entry]
    ) -> [RecurrentBlockFusionWindow] {
        let orderedEntries = entries.sorted {
            if $0.rangeStart == $1.rangeStart {
                return $0.index < $1.index
            }
            return $0.rangeStart < $1.rangeStart
        }
        var windows: [RecurrentBlockFusionWindow] = []
        var cursor = 0
        while cursor < orderedEntries.count {
            let inputEntry = orderedEntries[cursor]
            guard isLinearAttentionInputProjection(inputEntry),
                  let inputLayerIndex = layerIndex(from: inputEntry.weightTensorName) else {
                cursor += 1
                continue
            }

            var recurrenceEntry: Optional<MetalPrefillProfile.Entry> = Optional.none
            var bridgeEntries: [MetalPrefillProfile.Entry] = []
            var outputEntry: Optional<MetalPrefillProfile.Entry> = Optional.none
            var scanIndex = cursor + 1

            while scanIndex < orderedEntries.count {
                let current = orderedEntries[scanIndex]
                if isLinearAttentionInputProjection(current) {
                    break
                }
                if recurrenceEntry == nil, isLinearAttentionRecurrence(current) {
                    recurrenceEntry = current
                    scanIndex += 1
                    continue
                }
                if let recurrenceEntry,
                   let outputProjection = linearAttentionOutputProjection(
                    startingAt: scanIndex,
                    in: orderedEntries,
                    layerIndex: inputLayerIndex
                   ) {
                    outputEntry = outputProjection.entries.last
                    let bridgeRange = orderedEntries[(cursor + 1)..<scanIndex]
                    bridgeEntries = bridgeRange.filter { entry in
                        entry.index != recurrenceEntry.index
                    }
                    break
                }
                scanIndex += 1
            }

            if let recurrenceEntry, let outputEntry {
                let outputProjection = linearAttentionOutputProjection(
                    startingAt: scanIndex,
                    in: orderedEntries,
                    layerIndex: inputLayerIndex
                )
                let outputProjectionEntries = outputProjection?.entries ?? [outputEntry]
                windows.append(
                    RecurrentBlockFusionWindow(
                        layerIndex: inputLayerIndex,
                        rangeStart: inputEntry.rangeStart,
                        rangeEnd: outputEntry.rangeEnd,
                        inputProjectionStepIndex: inputEntry.index,
                        recurrenceStepIndex: recurrenceEntry.index,
                        bridgeStepIndices: bridgeEntries.map(\.index),
                        outputProjectionStepIndex: outputEntry.index,
                        outputProjectionStepIndices: outputProjectionEntries.map(\.index),
                        inputProjectionKernelName: inputEntry.kernelName,
                        recurrenceKernelName: recurrenceEntry.kernelName,
                        outputProjectionKernelName: outputEntry.kernelName,
                        outputProjectionKernelNames: outputProjectionEntries.map(\.kernelName)
                    )
                )
                cursor = outputProjection?.nextCursor ?? scanIndex + 1
            } else {
                cursor += 1
            }
        }
        return windows
    }

    private static func isLinearAttentionInputProjection(_ entry: MetalPrefillProfile.Entry) -> Bool {
        guard entry.category == "projection",
              let tensorName = entry.weightTensorName else {
            return false
        }
        return tensorName.contains("linear_attn.in_proj_qkv.weight")
            && tensorName.contains("linear_attn.in_proj_z.weight")
            && tensorName.contains("linear_attn.in_proj_b.weight")
            && tensorName.contains("linear_attn.in_proj_a.weight")
    }

    private static func isLinearAttentionRecurrence(_ entry: MetalPrefillProfile.Entry) -> Bool {
        entry.category == "ssm_recurrence"
            && entry.kernelName.contains("ssm_recurrence_seq")
    }

    private static func isLinearAttentionOutputProjection(_ entry: MetalPrefillProfile.Entry) -> Bool {
        guard entry.category == "projection",
              let tensorName = entry.weightTensorName else {
            return false
        }
        return tensorName.contains("linear_attn.out_proj.weight")
    }

    private struct LinearAttentionOutputProjection {
        let entries: [MetalPrefillProfile.Entry]
        let nextCursor: Int
    }

    private static func linearAttentionOutputProjection(
        startingAt index: Int,
        in entries: [MetalPrefillProfile.Entry],
        layerIndex: Int
    ) -> LinearAttentionOutputProjection? {
        guard index < entries.count else { return nil }
        let first = entries[index]
        guard isLinearAttentionOutputProjection(first),
              Self.layerIndex(from: first.weightTensorName) == layerIndex else {
            return nil
        }
        if first.kernelName.hasPrefix("recurrent_block_partial_reduce") {
            return LinearAttentionOutputProjection(entries: [first], nextCursor: index + 1)
        }
        guard first.kernelName.hasPrefix("recurrent_block_partial_projection") else {
            return LinearAttentionOutputProjection(entries: [first], nextCursor: index + 1)
        }

        var scanIndex = index + 1
        while scanIndex < entries.count {
            let candidate = entries[scanIndex]
            if isLinearAttentionInputProjection(candidate) {
                return nil
            }
            if isLinearAttentionOutputProjection(candidate) {
                guard Self.layerIndex(from: candidate.weightTensorName) == layerIndex,
                      candidate.kernelName.hasPrefix("recurrent_block_partial_reduce") else {
                    return nil
                }
                return LinearAttentionOutputProjection(
                    entries: [first, candidate],
                    nextCursor: scanIndex + 1
                )
            }
            guard isPartialOutputBridge(candidate) else {
                return nil
            }
            scanIndex += 1
        }
        return nil
    }

    private static func isPartialOutputBridge(_ entry: MetalPrefillProfile.Entry) -> Bool {
        entry.category == "other" && entry.kernelName.hasPrefix("round_")
    }

    private static func layerIndex(from tensorName: String?) -> Int? {
        let firstTensorName = tensorName?.split(separator: ";").first.map(String.init)
        guard let components = firstTensorName?.split(separator: ".").map(String.init),
              let layerTokenIndex = components.firstIndex(of: "layers"),
              layerTokenIndex + 1 < components.count else {
            return nil
        }
        return Int(components[layerTokenIndex + 1])
    }
}
