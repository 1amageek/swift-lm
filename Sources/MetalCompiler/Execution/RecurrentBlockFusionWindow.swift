import Foundation

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
