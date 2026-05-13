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
    let inputProjectionKernelName: String
    let recurrenceKernelName: String
    let outputProjectionKernelName: String

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

enum RecurrentBlockFusionPrototypePlanner {
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

        let partitionCount = recurrence.groupCount
        let headsPerPartition = recurrence.headCount / partitionCount
        let partitionInputDimension = headsPerPartition * recurrence.valueHeadDimension
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
                if let recurrenceEntry, isLinearAttentionOutputProjection(current),
                   layerIndex(from: current.weightTensorName) == inputLayerIndex {
                    outputEntry = current
                    let bridgeRange = orderedEntries[(cursor + 1)..<scanIndex]
                    bridgeEntries = bridgeRange.filter { entry in
                        entry.index != recurrenceEntry.index
                    }
                    break
                }
                scanIndex += 1
            }

            if let recurrenceEntry, let outputEntry {
                windows.append(
                    RecurrentBlockFusionWindow(
                        layerIndex: inputLayerIndex,
                        rangeStart: inputEntry.rangeStart,
                        rangeEnd: outputEntry.rangeEnd,
                        inputProjectionStepIndex: inputEntry.index,
                        recurrenceStepIndex: recurrenceEntry.index,
                        bridgeStepIndices: bridgeEntries.map(\.index),
                        outputProjectionStepIndex: outputEntry.index,
                        inputProjectionKernelName: inputEntry.kernelName,
                        recurrenceKernelName: recurrenceEntry.kernelName,
                        outputProjectionKernelName: outputEntry.kernelName
                    )
                )
                cursor = scanIndex + 1
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
