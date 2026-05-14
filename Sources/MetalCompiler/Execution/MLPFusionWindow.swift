import Foundation

struct MLPFusionWindow: Sendable, Equatable {
    let layerIndex: Int
    let rangeStart: Int
    let rangeEnd: Int
    let gateUpProjectionStepIndex: Int
    let activationStepIndex: Int?
    let downProjectionStepIndex: Int
    let gateUpProjectionKernelName: String
    let activationKernelName: String?
    let downProjectionKernelName: String
    let route: Route

    enum Route: String, Sendable, Equatable {
        case unfusedSwigluDown = "unfused_swiglu_down"
        case fusedSwigluDown = "fused_swiglu_down"
    }

    var range: Range<Int> {
        rangeStart..<rangeEnd
    }
}

enum MLPFusionWindowScanner {
    static func swigluDownWindows(
        in entries: [MetalPrefillProfile.Entry]
    ) -> [MLPFusionWindow] {
        let orderedEntries = entries.sorted {
            if $0.rangeStart == $1.rangeStart {
                return $0.index < $1.index
            }
            return $0.rangeStart < $1.rangeStart
        }
        var windows: [MLPFusionWindow] = []
        var cursor = 0
        while cursor < orderedEntries.count {
            let gateUpEntry = orderedEntries[cursor]
            guard isGateUpProjection(gateUpEntry),
                  let layerIndex = extractLayerIndex(from: gateUpEntry.weightTensorName) else {
                cursor += 1
                continue
            }

            if let fusedWindow = fusedSwigluDownWindow(
                gateUpEntry: gateUpEntry,
                layerIndex: layerIndex,
                startingAt: cursor + 1,
                in: orderedEntries
            ) {
                windows.append(fusedWindow.window)
                cursor = fusedWindow.nextCursor
                continue
            }

            if let unfusedWindow = unfusedSwigluDownWindow(
                gateUpEntry: gateUpEntry,
                layerIndex: layerIndex,
                startingAt: cursor + 1,
                in: orderedEntries
            ) {
                windows.append(unfusedWindow.window)
                cursor = unfusedWindow.nextCursor
                continue
            }

            cursor += 1
        }
        return windows
    }

    private struct ScanResult {
        let window: MLPFusionWindow
        let nextCursor: Int
    }

    private static func unfusedSwigluDownWindow(
        gateUpEntry: MetalPrefillProfile.Entry,
        layerIndex: Int,
        startingAt index: Int,
        in entries: [MetalPrefillProfile.Entry]
    ) -> ScanResult? {
        var activationEntry: Optional<MetalPrefillProfile.Entry> = Optional.none
        var scanIndex = index
        while scanIndex < entries.count {
            let entry = entries[scanIndex]
            if isGateUpProjection(entry) {
                return nil
            }
            if let entryLayerIndex = extractLayerIndex(from: entry.weightTensorName),
               entryLayerIndex != layerIndex {
                return nil
            }
            if activationEntry == nil, isSwigluActivation(entry) {
                activationEntry = entry
                scanIndex += 1
                continue
            }
            if let activationEntry,
               isDownProjection(entry, layerIndex: layerIndex),
               !isFusedSwigluDownProjection(entry) {
                return ScanResult(
                    window: MLPFusionWindow(
                        layerIndex: layerIndex,
                        rangeStart: gateUpEntry.rangeStart,
                        rangeEnd: entry.rangeEnd,
                        gateUpProjectionStepIndex: gateUpEntry.index,
                        activationStepIndex: activationEntry.index,
                        downProjectionStepIndex: entry.index,
                        gateUpProjectionKernelName: gateUpEntry.kernelName,
                        activationKernelName: activationEntry.kernelName,
                        downProjectionKernelName: entry.kernelName,
                        route: .unfusedSwigluDown
                    ),
                    nextCursor: scanIndex + 1
                )
            }
            scanIndex += 1
        }
        return nil
    }

    private static func fusedSwigluDownWindow(
        gateUpEntry: MetalPrefillProfile.Entry,
        layerIndex: Int,
        startingAt index: Int,
        in entries: [MetalPrefillProfile.Entry]
    ) -> ScanResult? {
        var scanIndex = index
        while scanIndex < entries.count {
            let entry = entries[scanIndex]
            if isGateUpProjection(entry) || isSwigluActivation(entry) {
                return nil
            }
            if let entryLayerIndex = extractLayerIndex(from: entry.weightTensorName),
               entryLayerIndex != layerIndex {
                return nil
            }
            if isFusedSwigluDownProjection(entry),
               isDownProjection(entry, layerIndex: layerIndex) {
                return ScanResult(
                    window: MLPFusionWindow(
                        layerIndex: layerIndex,
                        rangeStart: gateUpEntry.rangeStart,
                        rangeEnd: entry.rangeEnd,
                        gateUpProjectionStepIndex: gateUpEntry.index,
                        activationStepIndex: nil,
                        downProjectionStepIndex: entry.index,
                        gateUpProjectionKernelName: gateUpEntry.kernelName,
                        activationKernelName: nil,
                        downProjectionKernelName: entry.kernelName,
                        route: .fusedSwigluDown
                    ),
                    nextCursor: scanIndex + 1
                )
            }
            scanIndex += 1
        }
        return nil
    }

    private static func isGateUpProjection(_ entry: MetalPrefillProfile.Entry) -> Bool {
        guard entry.category == "projection",
              let tensorName = entry.weightTensorName else {
            return false
        }
        return tensorName.contains("mlp.gate_proj.weight")
            && tensorName.contains("mlp.up_proj.weight")
    }

    private static func isSwigluActivation(_ entry: MetalPrefillProfile.Entry) -> Bool {
        entry.category == "elementwise"
            && entry.kernelName.lowercased().contains("swiglu")
    }

    private static func isDownProjection(
        _ entry: MetalPrefillProfile.Entry,
        layerIndex: Int
    ) -> Bool {
        guard let tensorName = entry.weightTensorName,
              tensorName.contains("mlp.down_proj.weight"),
              Self.extractLayerIndex(from: tensorName) == layerIndex else {
            return false
        }
        return entry.category == "projection" || isFusedSwigluDownProjection(entry)
    }

    private static func isFusedSwigluDownProjection(_ entry: MetalPrefillProfile.Entry) -> Bool {
        entry.kernelName.lowercased().hasPrefix("mlp_fused_swiglu_down")
    }

    private static func extractLayerIndex(from tensorName: String?) -> Int? {
        let firstTensorName = tensorName?.split(separator: ";").first.map(String.init)
        guard let components = firstTensorName?.split(separator: ".").map(String.init),
              let layerTokenIndex = components.firstIndex(of: "layers"),
              layerTokenIndex + 1 < components.count else {
            return nil
        }
        return Int(components[layerTokenIndex + 1])
    }
}
