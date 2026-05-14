import Foundation
import Metal

struct MetalPrefillProfile: Codable, Sendable {
    struct Entry: Codable, Sendable {
        let scope: String
        let index: Int
        let rangeStart: Int
        let rangeEnd: Int
        let kernelName: String
        let category: String
        let mode: String
        let layerIndex: Int?
        let entryIndex: Int?
        let weightTensorName: String?
        let gridWidth: Int
        let gridHeight: Int
        let gridDepth: Int
        let threadgroupWidth: Int
        let threadgroupHeight: Int
        let threadgroupDepth: Int
        let threadgroupMemoryBytes: Int
        let bufferBindingCount: Int
        let inlineConstantBytes: Int
        let uniqueBoundBufferBytes: Int
        let estimatedReadBytes: Int
        let estimatedWriteBytes: Int
        let estimatedTotalBytes: Int
        let estimatedDispatchCount: Int
        let totalGpuMicroseconds: Double
        let averageGpuMicroseconds: Double
        let totalWallMicroseconds: Double
        let averageWallMicroseconds: Double
    }

    struct Summary: Codable, Sendable {
        struct Category: Codable, Sendable {
            let name: String
            let entryCount: Int
            let totalGpuMicroseconds: Double
            let totalWallMicroseconds: Double
            let percentageOfGpu: Double
        }

        let totalGpuMicroseconds: Double
        let totalWallMicroseconds: Double
        let entriesByCategory: [Category]
    }

    let schemaVersion: Int
    let profileKind: String
    let sequenceLength: Int
    let maximumSequenceLength: Int
    let iterations: Int
    let warmupIterations: Int
    let stepCount: Int
    let generatedAt: String
    let entries: [Entry]
    let summary: Summary

    init(
        profileKind: String,
        sequenceLength: Int,
        maximumSequenceLength: Int,
        iterations: Int,
        warmupIterations: Int,
        stepCount: Int,
        entries: [Entry],
        generatedAt: String = ISO8601DateFormatter().string(from: Date())
    ) {
        self.schemaVersion = 2
        self.profileKind = profileKind
        self.sequenceLength = sequenceLength
        self.maximumSequenceLength = maximumSequenceLength
        self.iterations = iterations
        self.warmupIterations = warmupIterations
        self.stepCount = stepCount
        self.generatedAt = generatedAt
        self.entries = entries
        self.summary = Self.makeSummary(entries: entries)
    }

    var csvString: String {
        var lines: [String] = [
            [
                "scope",
                "index",
                "rangeStart",
                "rangeEnd",
                "kernelName",
                "category",
                "mode",
                "layerIndex",
                "entryIndex",
                "weightTensorName",
                "gridWidth",
                "gridHeight",
                "threadgroupWidth",
                "threadgroupHeight",
                "threadgroupMemoryBytes",
                "bufferBindingCount",
                "inlineConstantBytes",
                "uniqueBoundBufferBytes",
                "estimatedReadBytes",
                "estimatedWriteBytes",
                "estimatedTotalBytes",
                "estimatedDispatchCount",
                "averageGpuMicroseconds",
                "averageWallMicroseconds",
            ].joined(separator: ","),
        ]
        for entry in entries {
            var row: [String] = []
            row.reserveCapacity(24)
            row.append(entry.scope)
            row.append(String(entry.index))
            row.append(String(entry.rangeStart))
            row.append(String(entry.rangeEnd))
            row.append(csvEscape(entry.kernelName))
            row.append(entry.category)
            row.append(entry.mode)
            row.append(entry.layerIndex.map(String.init) ?? "")
            row.append(entry.entryIndex.map(String.init) ?? "")
            row.append(csvEscape(entry.weightTensorName ?? ""))
            row.append(String(entry.gridWidth))
            row.append(String(entry.gridHeight))
            row.append(String(entry.threadgroupWidth))
            row.append(String(entry.threadgroupHeight))
            row.append(String(entry.threadgroupMemoryBytes))
            row.append(String(entry.bufferBindingCount))
            row.append(String(entry.inlineConstantBytes))
            row.append(String(entry.uniqueBoundBufferBytes))
            row.append(String(entry.estimatedReadBytes))
            row.append(String(entry.estimatedWriteBytes))
            row.append(String(entry.estimatedTotalBytes))
            row.append(String(entry.estimatedDispatchCount))
            row.append(String(format: "%.3f", entry.averageGpuMicroseconds))
            row.append(String(format: "%.3f", entry.averageWallMicroseconds))
            lines.append(row.joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    var categoryCSVString: String {
        var lines: [String] = [
            [
                "category",
                "entryCount",
                "totalGpuMicroseconds",
                "totalWallMicroseconds",
                "percentageOfGpu",
            ].joined(separator: ","),
        ]
        for category in summary.entriesByCategory {
            lines.append([
                csvEscape(category.name),
                String(category.entryCount),
                String(format: "%.3f", category.totalGpuMicroseconds),
                String(format: "%.3f", category.totalWallMicroseconds),
                String(format: "%.3f", category.percentageOfGpu),
            ].joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    var kernelCSVString: String {
        aggregateCSVString(
            headers: [
                "kernelName",
                "category",
                "entryCount",
                "estimatedDispatchCount",
                "estimatedReadBytes",
                "estimatedWriteBytes",
                "estimatedTotalBytes",
                "totalGpuMicroseconds",
                "averageGpuMicroseconds",
                "totalWallMicroseconds",
                "averageWallMicroseconds",
            ],
            groups: aggregateEntries(entries: entries) { entry in
                [entry.kernelName, entry.category]
            }
        )
    }

    var layerCSVString: String {
        aggregateCSVString(
            headers: [
                "layerIndex",
                "category",
                "entryCount",
                "estimatedDispatchCount",
                "estimatedReadBytes",
                "estimatedWriteBytes",
                "estimatedTotalBytes",
                "totalGpuMicroseconds",
                "averageGpuMicroseconds",
                "totalWallMicroseconds",
                "averageWallMicroseconds",
            ],
            groups: aggregateEntries(entries: entries) { entry in
                [effectiveLayerIndex(entry).map(String.init) ?? "", entry.category]
            }
        )
    }

    var weightRoleCSVString: String {
        aggregateCSVString(
            headers: [
                "weightRole",
                "category",
                "entryCount",
                "estimatedDispatchCount",
                "estimatedReadBytes",
                "estimatedWriteBytes",
                "estimatedTotalBytes",
                "totalGpuMicroseconds",
                "averageGpuMicroseconds",
                "totalWallMicroseconds",
                "averageWallMicroseconds",
            ],
            groups: aggregateEntries(entries: entries) { entry in
                [weightRoleSummary(entry.weightTensorName), entry.category]
            }
        )
    }

    var blockCSVString: String {
        aggregateCSVString(
            headers: [
                "layerIndex",
                "semanticBlock",
                "category",
                "entryCount",
                "estimatedDispatchCount",
                "estimatedReadBytes",
                "estimatedWriteBytes",
                "estimatedTotalBytes",
                "totalGpuMicroseconds",
                "averageGpuMicroseconds",
                "totalWallMicroseconds",
                "averageWallMicroseconds",
            ],
            groups: aggregateEntries(entries: entries) { entry in
                [
                    effectiveLayerIndex(entry).map(String.init) ?? "",
                    semanticBlockSummary(entry),
                    entry.category,
                ]
            }
        )
    }

    var recurrentBlockWindowCSVString: String {
        var lines: [String] = [
            [
                "layerIndex",
                "rangeStart",
                "rangeEnd",
                "inputProjectionStepIndex",
                "recurrenceStepIndex",
                "bridgeStepIndices",
                "outputProjectionStepIndex",
                "outputProjectionStepIndices",
                "inputProjectionKernelName",
                "recurrenceKernelName",
                "outputProjectionKernelName",
                "outputProjectionKernelNames",
                "windowEntryCount",
                "totalGpuMicroseconds",
                "inputProjectionGpuMicroseconds",
                "recurrenceGpuMicroseconds",
                "bridgeGpuMicroseconds",
                "outputProjectionGpuMicroseconds",
                "estimatedTotalBytes",
            ].joined(separator: ","),
        ]
        for window in RecurrentBlockFusionWindowScanner.linearAttentionWindows(in: entries) {
            let timing = recurrentBlockWindowTiming(for: window)
            lines.append([
                String(window.layerIndex),
                String(window.rangeStart),
                String(window.rangeEnd),
                String(window.inputProjectionStepIndex),
                String(window.recurrenceStepIndex),
                csvEscape(window.bridgeStepIndices.map(String.init).joined(separator: ";")),
                String(window.outputProjectionStepIndex),
                csvEscape(window.outputProjectionStepIndices.map(String.init).joined(separator: ";")),
                csvEscape(window.inputProjectionKernelName),
                csvEscape(window.recurrenceKernelName),
                csvEscape(window.outputProjectionKernelName),
                csvEscape(window.outputProjectionKernelNames.joined(separator: ";")),
                String(timing.entryCount),
                String(format: "%.3f", timing.totalGpuMicroseconds),
                String(format: "%.3f", timing.inputProjectionGpuMicroseconds),
                String(format: "%.3f", timing.recurrenceGpuMicroseconds),
                String(format: "%.3f", timing.bridgeGpuMicroseconds),
                String(format: "%.3f", timing.outputProjectionGpuMicroseconds),
                String(timing.estimatedTotalBytes),
            ].joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    var mlpFusionWindowCSVString: String {
        var lines: [String] = [
            [
                "layerIndex",
                "rangeStart",
                "rangeEnd",
                "gateUpProjectionStepIndex",
                "activationStepIndex",
                "downProjectionStepIndex",
                "gateUpProjectionKernelName",
                "activationKernelName",
                "downProjectionKernelName",
                "route",
                "windowEntryCount",
                "totalGpuMicroseconds",
                "gateUpProjectionGpuMicroseconds",
                "activationGpuMicroseconds",
                "downProjectionGpuMicroseconds",
                "estimatedTotalBytes",
            ].joined(separator: ","),
        ]
        for window in MLPFusionWindowScanner.swigluDownWindows(in: entries) {
            let timing = mlpFusionWindowTiming(for: window)
            lines.append([
                String(window.layerIndex),
                String(window.rangeStart),
                String(window.rangeEnd),
                String(window.gateUpProjectionStepIndex),
                window.activationStepIndex.map(String.init) ?? "",
                String(window.downProjectionStepIndex),
                csvEscape(window.gateUpProjectionKernelName),
                csvEscape(window.activationKernelName ?? ""),
                csvEscape(window.downProjectionKernelName),
                window.route.rawValue,
                String(timing.entryCount),
                String(format: "%.3f", timing.totalGpuMicroseconds),
                String(format: "%.3f", timing.gateUpProjectionGpuMicroseconds),
                String(format: "%.3f", timing.activationGpuMicroseconds),
                String(format: "%.3f", timing.downProjectionGpuMicroseconds),
                String(timing.estimatedTotalBytes),
            ].joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private struct RecurrentBlockWindowTiming {
        let entryCount: Int
        let totalGpuMicroseconds: Double
        let inputProjectionGpuMicroseconds: Double
        let recurrenceGpuMicroseconds: Double
        let bridgeGpuMicroseconds: Double
        let outputProjectionGpuMicroseconds: Double
        let estimatedTotalBytes: Int
    }

    private func recurrentBlockWindowTiming(
        for window: RecurrentBlockFusionWindow
    ) -> RecurrentBlockWindowTiming {
        let windowEntries = entries.filter { entry in
            entry.rangeStart >= window.rangeStart && entry.rangeEnd <= window.rangeEnd
        }
        let inputProjectionGpuMicroseconds = windowEntries
            .filter { $0.index == window.inputProjectionStepIndex }
            .reduce(0) { $0 + $1.totalGpuMicroseconds }
        let recurrenceGpuMicroseconds = windowEntries
            .filter { $0.index == window.recurrenceStepIndex }
            .reduce(0) { $0 + $1.totalGpuMicroseconds }
        let outputProjectionStepIndices = Set(window.outputProjectionStepIndices)
        let outputProjectionGpuMicroseconds = windowEntries
            .filter { outputProjectionStepIndices.contains($0.index) }
            .reduce(0) { $0 + $1.totalGpuMicroseconds }
        let totalGpuMicroseconds = windowEntries.reduce(0) { $0 + $1.totalGpuMicroseconds }
        let bridgeGpuMicroseconds = totalGpuMicroseconds
            - inputProjectionGpuMicroseconds
            - recurrenceGpuMicroseconds
            - outputProjectionGpuMicroseconds
        let estimatedTotalBytes = windowEntries.reduce(0) { $0 + $1.estimatedTotalBytes }
        return RecurrentBlockWindowTiming(
            entryCount: windowEntries.count,
            totalGpuMicroseconds: totalGpuMicroseconds,
            inputProjectionGpuMicroseconds: inputProjectionGpuMicroseconds,
            recurrenceGpuMicroseconds: recurrenceGpuMicroseconds,
            bridgeGpuMicroseconds: bridgeGpuMicroseconds,
            outputProjectionGpuMicroseconds: outputProjectionGpuMicroseconds,
            estimatedTotalBytes: estimatedTotalBytes
        )
    }

    private struct MLPFusionWindowTiming {
        let entryCount: Int
        let totalGpuMicroseconds: Double
        let gateUpProjectionGpuMicroseconds: Double
        let activationGpuMicroseconds: Double
        let downProjectionGpuMicroseconds: Double
        let estimatedTotalBytes: Int
    }

    private func mlpFusionWindowTiming(
        for window: MLPFusionWindow
    ) -> MLPFusionWindowTiming {
        let windowEntries = entries.filter { entry in
            entry.rangeStart >= window.rangeStart && entry.rangeEnd <= window.rangeEnd
        }
        let gateUpProjectionGpuMicroseconds = windowEntries
            .filter { $0.index == window.gateUpProjectionStepIndex }
            .reduce(0) { $0 + $1.totalGpuMicroseconds }
        let activationGpuMicroseconds = windowEntries
            .filter { entry in
                guard let activationStepIndex = window.activationStepIndex else {
                    return false
                }
                return entry.index == activationStepIndex
            }
            .reduce(0) { $0 + $1.totalGpuMicroseconds }
        let downProjectionGpuMicroseconds = windowEntries
            .filter { $0.index == window.downProjectionStepIndex }
            .reduce(0) { $0 + $1.totalGpuMicroseconds }
        return MLPFusionWindowTiming(
            entryCount: windowEntries.count,
            totalGpuMicroseconds: windowEntries.reduce(0) { $0 + $1.totalGpuMicroseconds },
            gateUpProjectionGpuMicroseconds: gateUpProjectionGpuMicroseconds,
            activationGpuMicroseconds: activationGpuMicroseconds,
            downProjectionGpuMicroseconds: downProjectionGpuMicroseconds,
            estimatedTotalBytes: windowEntries.reduce(0) { $0 + $1.estimatedTotalBytes }
        )
    }

    func writeArtifacts(directory: URL, basename: String) throws -> [URL] {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)

        let jsonURL = directory.appendingPathComponent("\(basename).json")
        let csvURL = directory.appendingPathComponent("\(basename).csv")
        let categoryCSVURL = directory.appendingPathComponent("\(basename)-categories.csv")
        let kernelCSVURL = directory.appendingPathComponent("\(basename)-kernels.csv")
        let layerCSVURL = directory.appendingPathComponent("\(basename)-layers.csv")
        let weightCSVURL = directory.appendingPathComponent("\(basename)-weights.csv")
        let blockCSVURL = directory.appendingPathComponent("\(basename)-blocks.csv")
        let recurrentWindowsCSVURL = directory.appendingPathComponent("\(basename)-recurrent-windows.csv")
        let mlpWindowsCSVURL = directory.appendingPathComponent("\(basename)-mlp-windows.csv")

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let jsonData = try encoder.encode(self)
        try jsonData.write(to: jsonURL, options: .atomic)
        try Data(csvString.utf8).write(to: csvURL, options: .atomic)
        try Data(categoryCSVString.utf8).write(to: categoryCSVURL, options: .atomic)
        try Data(kernelCSVString.utf8).write(to: kernelCSVURL, options: .atomic)
        try Data(layerCSVString.utf8).write(to: layerCSVURL, options: .atomic)
        try Data(weightRoleCSVString.utf8).write(to: weightCSVURL, options: .atomic)
        try Data(blockCSVString.utf8).write(to: blockCSVURL, options: .atomic)
        try Data(recurrentBlockWindowCSVString.utf8).write(to: recurrentWindowsCSVURL, options: .atomic)
        try Data(mlpFusionWindowCSVString.utf8).write(to: mlpWindowsCSVURL, options: .atomic)
        return [
            jsonURL,
            csvURL,
            categoryCSVURL,
            kernelCSVURL,
            layerCSVURL,
            weightCSVURL,
            blockCSVURL,
            recurrentWindowsCSVURL,
            mlpWindowsCSVURL,
        ]
    }

    private static func makeSummary(entries: [Entry]) -> Summary {
        struct Accumulator {
            var count: Int = 0
            var gpu: Double = 0
            var wall: Double = 0
        }
        var byCategory: [String: Accumulator] = [:]
        var totalGpu: Double = 0
        var totalWall: Double = 0
        for entry in entries {
            totalGpu += entry.averageGpuMicroseconds
            totalWall += entry.averageWallMicroseconds
            var accumulator = byCategory[entry.category] ?? Accumulator()
            accumulator.count += 1
            accumulator.gpu += entry.averageGpuMicroseconds
            accumulator.wall += entry.averageWallMicroseconds
            byCategory[entry.category] = accumulator
        }
        let categories = byCategory
            .map { key, value in
                Summary.Category(
                    name: key,
                    entryCount: value.count,
                    totalGpuMicroseconds: value.gpu,
                    totalWallMicroseconds: value.wall,
                    percentageOfGpu: totalGpu > 0 ? value.gpu / totalGpu * 100 : 0
                )
            }
            .sorted {
                if $0.totalGpuMicroseconds == $1.totalGpuMicroseconds {
                    return $0.name < $1.name
                }
                return $0.totalGpuMicroseconds > $1.totalGpuMicroseconds
            }
        return Summary(
            totalGpuMicroseconds: totalGpu,
            totalWallMicroseconds: totalWall,
            entriesByCategory: categories
        )
    }
}

private struct PrefillProfileAggregate: Sendable {
    let keys: [String]
    var entryCount: Int
    var estimatedDispatchCount: Int
    var estimatedReadBytes: Int
    var estimatedWriteBytes: Int
    var estimatedTotalBytes: Int
    var totalGpuMicroseconds: Double
    var totalWallMicroseconds: Double

    var averageGpuMicroseconds: Double {
        totalGpuMicroseconds / Double(max(entryCount, 1))
    }

    var averageWallMicroseconds: Double {
        totalWallMicroseconds / Double(max(entryCount, 1))
    }
}

private func aggregateEntries(
    entries: [MetalPrefillProfile.Entry],
    key: (MetalPrefillProfile.Entry) -> [String]
) -> [PrefillProfileAggregate] {
    var aggregatesByKey: [String: PrefillProfileAggregate] = [:]
    for entry in entries {
        let keys = key(entry)
        let joinedKey = keys.map { "\($0.count):\($0)" }.joined(separator: "|")
        var aggregate = aggregatesByKey[joinedKey] ?? PrefillProfileAggregate(
            keys: keys,
            entryCount: 0,
            estimatedDispatchCount: 0,
            estimatedReadBytes: 0,
            estimatedWriteBytes: 0,
            estimatedTotalBytes: 0,
            totalGpuMicroseconds: 0,
            totalWallMicroseconds: 0
        )
        aggregate.entryCount += 1
        aggregate.estimatedDispatchCount += entry.estimatedDispatchCount
        aggregate.estimatedReadBytes += entry.estimatedReadBytes
        aggregate.estimatedWriteBytes += entry.estimatedWriteBytes
        aggregate.estimatedTotalBytes += entry.estimatedTotalBytes
        aggregate.totalGpuMicroseconds += entry.averageGpuMicroseconds
        aggregate.totalWallMicroseconds += entry.averageWallMicroseconds
        aggregatesByKey[joinedKey] = aggregate
    }
    return aggregatesByKey.values.sorted {
        if $0.totalGpuMicroseconds == $1.totalGpuMicroseconds {
            return $0.keys.lexicographicallyPrecedes($1.keys)
        }
        return $0.totalGpuMicroseconds > $1.totalGpuMicroseconds
    }
}

private func aggregateCSVString(
    headers: [String],
    groups: [PrefillProfileAggregate]
) -> String {
    var lines = [headers.joined(separator: ",")]
    for group in groups {
        var row = group.keys.map(csvEscape)
        row.append(String(group.entryCount))
        row.append(String(group.estimatedDispatchCount))
        row.append(String(group.estimatedReadBytes))
        row.append(String(group.estimatedWriteBytes))
        row.append(String(group.estimatedTotalBytes))
        row.append(String(format: "%.3f", group.totalGpuMicroseconds))
        row.append(String(format: "%.3f", group.averageGpuMicroseconds))
        row.append(String(format: "%.3f", group.totalWallMicroseconds))
        row.append(String(format: "%.3f", group.averageWallMicroseconds))
        lines.append(row.joined(separator: ","))
    }
    return lines.joined(separator: "\n") + "\n"
}

private func weightRoleSummary(_ tensorName: String?) -> String {
    guard let tensorName, !tensorName.isEmpty else {
        return ""
    }
    let roles = tensorName
        .split(separator: ";")
        .map(String.init)
        .map(singleWeightRoleSummary)
        .filter { !$0.isEmpty }
    return roles.joined(separator: "+")
}

private func semanticBlockSummary(_ entry: MetalPrefillProfile.Entry) -> String {
    let role = weightRoleSummary(entry.weightTensorName)
    if role.contains("linear_attn.") || entry.category == "ssm_recurrence" {
        return "linear_attn"
    }
    if role.contains("self_attn.") || entry.category == "attention" {
        return "self_attn"
    }
    if role.contains("mlp.") {
        return "mlp"
    }
    if role.contains("embed_tokens") || entry.category == "embedding" {
        return "embedding"
    }
    if entry.category == "reduction" {
        return "normalization"
    }
    return role.isEmpty ? entry.category : role
}

private func singleWeightRoleSummary(_ tensorName: String) -> String {
    var components = tensorName.split(separator: ".").map(String.init)
    guard !components.isEmpty else { return "" }
    if components.last == "weight" {
        components.removeLast()
    }
    if let layerIndex = components.firstIndex(of: "layers"),
       layerIndex + 2 < components.count {
        return components[(layerIndex + 2)...].joined(separator: ".")
    }
    return components.suffix(3).joined(separator: ".")
}

private func effectiveLayerIndex(_ entry: MetalPrefillProfile.Entry) -> Int? {
    if let layerIndex = entry.layerIndex {
        return layerIndex
    }
    return layerIndex(from: entry.weightTensorName)
}

private func layerIndex(from tensorName: String?) -> Int? {
    let firstTensorName = tensorName?.split(separator: ";").first.map(String.init)
    guard let components = firstTensorName?.split(separator: ".").map(String.init),
          let layerTokenIndex = components.firstIndex(of: "layers"),
          layerTokenIndex + 1 < components.count else {
        return nil
    }
    return Int(components[layerTokenIndex + 1])
}

struct MetalPrefillProfileHarness: Sendable {
    func profileSteps(
        plan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        sequenceLength: Int,
        iterations: Int,
        warmupIterations: Int = 1,
        basePosition: Int = 0,
        tokens: [Int32]? = nil,
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)]? = nil,
        ephemeralResidency: MetalResidencyLease = .empty
    ) throws -> MetalPrefillProfile {
        try validate(plan: plan, sequenceLength: sequenceLength, iterations: iterations, warmupIterations: warmupIterations)
        populateInputs(
            plan: plan,
            basePosition: basePosition,
            sequenceLength: sequenceLength,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxesByTokenIndex
        )
        writeRuntimeConstants(plan: plan, basePosition: basePosition, sequenceLength: sequenceLength)

        let activeSteps = plan.steps.enumerated().filter {
            $0.element.shouldExecute(sequenceLength: sequenceLength)
        }

        for _ in 0..<warmupIterations {
            for (_, step) in activeSteps {
                try encodeTimedStep(
                    step,
                    plan: plan,
                    submission: &submission,
                    sequenceLength: sequenceLength,
                    ephemeralResidency: ephemeralResidency
                )
            }
        }

        var entries: [MetalPrefillProfile.Entry] = []
        entries.reserveCapacity(activeSteps.count)
        for (index, step) in activeSteps {
            let timing = try measure(iterations: iterations) {
                try encodeTimedStep(
                    step,
                    plan: plan,
                    submission: &submission,
                    sequenceLength: sequenceLength,
                    ephemeralResidency: ephemeralResidency
                )
            }
            entries.append(
                makeStepEntry(
                    step: step,
                    index: index,
                    sequenceLength: sequenceLength,
                    timing: timing,
                    iterations: iterations
                )
            )
        }
        return MetalPrefillProfile(
            profileKind: "step",
            sequenceLength: sequenceLength,
            maximumSequenceLength: plan.maximumSequenceLength,
            iterations: iterations,
            warmupIterations: warmupIterations,
            stepCount: activeSteps.count,
            entries: entries
        )
    }

    func profilePasses(
        plan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        sequenceLength: Int,
        iterations: Int,
        warmupIterations: Int = 1,
        basePosition: Int = 0,
        tokens: [Int32]? = nil,
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)]? = nil,
        ephemeralResidency: MetalResidencyLease = .empty
    ) throws -> MetalPrefillProfile {
        try validate(plan: plan, sequenceLength: sequenceLength, iterations: iterations, warmupIterations: warmupIterations)
        populateInputs(
            plan: plan,
            basePosition: basePosition,
            sequenceLength: sequenceLength,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxesByTokenIndex
        )
        writeRuntimeConstants(plan: plan, basePosition: basePosition, sequenceLength: sequenceLength)

        let ranges = prefillPassRanges(for: plan.steps, within: 0..<plan.steps.count)
            .filter { range in
                plan.steps[range].contains {
                    $0.shouldExecute(sequenceLength: sequenceLength)
                }
            }
        for _ in 0..<warmupIterations {
            for range in ranges {
                try encodeTimedRange(
                    range,
                    plan: plan,
                    submission: &submission,
                    sequenceLength: sequenceLength,
                    ephemeralResidency: ephemeralResidency
                )
            }
        }

        var entries: [MetalPrefillProfile.Entry] = []
        entries.reserveCapacity(ranges.count)
        for (index, range) in ranges.enumerated() {
            let timing = try measure(iterations: iterations) {
                try encodeTimedRange(
                    range,
                    plan: plan,
                    submission: &submission,
                    sequenceLength: sequenceLength,
                    ephemeralResidency: ephemeralResidency
                )
            }
            entries.append(
                makePassEntry(
                    range: range,
                    index: index,
                    plan: plan,
                    sequenceLength: sequenceLength,
                    timing: timing,
                    iterations: iterations
                )
            )
        }
        return MetalPrefillProfile(
            profileKind: "pass",
            sequenceLength: sequenceLength,
            maximumSequenceLength: plan.maximumSequenceLength,
            iterations: iterations,
            warmupIterations: warmupIterations,
            stepCount: ranges.reduce(0) { partial, range in
                partial + plan.steps[range].filter {
                    $0.shouldExecute(sequenceLength: sequenceLength)
                }.count
            },
            entries: entries
        )
    }

    private func validate(
        plan: MetalPrefillPlan,
        sequenceLength: Int,
        iterations: Int,
        warmupIterations: Int
    ) throws {
        guard !plan.steps.isEmpty else {
            throw MetalCompilerError.deviceSetupFailed("Prefill profile requires at least one prefill step")
        }
        guard sequenceLength > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Prefill profile sequence length must be positive")
        }
        guard sequenceLength <= plan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed(
                "Prefill profile sequence length \(sequenceLength) exceeds maximum \(plan.maximumSequenceLength)"
            )
        }
        guard iterations > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Prefill profile iterations must be positive")
        }
        guard warmupIterations >= 0 else {
            throw MetalCompilerError.deviceSetupFailed("Prefill profile warmup iterations must be non-negative")
        }
    }

    private func populateInputs(
        plan: MetalPrefillPlan,
        basePosition: Int,
        sequenceLength: Int,
        tokens: [Int32]?,
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)]?
    ) {
        let tokenValues = tokens ?? (0..<sequenceLength).map { Int32($0 + 1) }
        precondition(tokenValues.count == sequenceLength, "Prefill profile token count mismatch")
        if let ropePositionAxesByTokenIndex {
            precondition(ropePositionAxesByTokenIndex.count == sequenceLength, "Prefill profile RoPE axis count mismatch")
        }

        let tokenPointer = plan.buffers.tokenIDs.contents()
            .bindMemory(to: Int32.self, capacity: sequenceLength)
        let positionPointer = plan.buffers.positions.contents()
            .bindMemory(to: UInt32.self, capacity: sequenceLength)
        let ropeAxesPointer = plan.buffers.ropePositionAxes.contents()
            .bindMemory(to: UInt32.self, capacity: sequenceLength * 3)

        for index in 0..<sequenceLength {
            tokenPointer[index] = tokenValues[index]
            let position = UInt32(basePosition + index)
            positionPointer[index] = position
            let axes = ropePositionAxesByTokenIndex?[index] ?? (position, position, position)
            ropeAxesPointer[index * 3] = axes.0
            ropeAxesPointer[index * 3 + 1] = axes.1
            ropeAxesPointer[index * 3 + 2] = axes.2
        }
    }

    private func writeRuntimeConstants(
        plan: MetalPrefillPlan,
        basePosition: Int,
        sequenceLength: Int
    ) {
        let pointer = plan.buffers.runtimeConstantBuffer.contents()
        pointer.storeBytes(of: UInt32(sequenceLength), toByteOffset: PrefillBufferSet.sequenceLengthOffset, as: UInt32.self)
        pointer.storeBytes(of: UInt32(0), toByteOffset: PrefillBufferSet.hiddenConversionCountOffset, as: UInt32.self)
        for index in 0..<sequenceLength {
            pointer.storeBytes(
                of: UInt32(basePosition + index),
                toByteOffset: PrefillBufferSet.positionOffset(at: index),
                as: UInt32.self
            )
        }
    }

    private func measure(
        iterations: Int,
        _ body: () throws -> (gpuStartTime: CFTimeInterval, gpuEndTime: CFTimeInterval, wallMicroseconds: Double)
    ) throws -> (gpuMicroseconds: Double, wallMicroseconds: Double) {
        var totalGPU: Double = 0
        var totalWall: Double = 0
        for _ in 0..<iterations {
            let timing = try body()
            totalGPU += (timing.gpuEndTime - timing.gpuStartTime) * 1_000_000
            totalWall += timing.wallMicroseconds
        }
        return (totalGPU, totalWall)
    }

    @discardableResult
    private func encodeTimedStep(
        _ step: MetalPrefillStep,
        plan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        sequenceLength: Int,
        ephemeralResidency: MetalResidencyLease
    ) throws -> (gpuStartTime: CFTimeInterval, gpuEndTime: CFTimeInterval, wallMicroseconds: Double) {
        let wallStart = CFAbsoluteTimeGetCurrent()
        let timing = try submission.withComputeTimed(ephemeralResidency: ephemeralResidency) { encoder, argumentTable in
            guard step.shouldExecute(sequenceLength: sequenceLength) else { return }
            encodeStep(
                step,
                encoder: encoder,
                argumentTable: argumentTable,
                runtimeConstantBuffer: plan.buffers.runtimeConstantBuffer,
                sequenceLength: sequenceLength
            )
        }
        let wallEnd = CFAbsoluteTimeGetCurrent()
        return (timing.gpuStartTime, timing.gpuEndTime, (wallEnd - wallStart) * 1_000_000)
    }

    @discardableResult
    private func encodeTimedRange(
        _ range: Range<Int>,
        plan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        sequenceLength: Int,
        ephemeralResidency: MetalResidencyLease
    ) throws -> (gpuStartTime: CFTimeInterval, gpuEndTime: CFTimeInterval, wallMicroseconds: Double) {
        let wallStart = CFAbsoluteTimeGetCurrent()
        let timing = try submission.withComputeTimed(ephemeralResidency: ephemeralResidency) { encoder, argumentTable in
            for step in plan.steps[range] {
                guard step.shouldExecute(sequenceLength: sequenceLength) else {
                    continue
                }
                encodeStep(
                    step,
                    encoder: encoder,
                    argumentTable: argumentTable,
                    runtimeConstantBuffer: plan.buffers.runtimeConstantBuffer,
                    sequenceLength: sequenceLength
                )
            }
        }
        let wallEnd = CFAbsoluteTimeGetCurrent()
        return (timing.gpuStartTime, timing.gpuEndTime, (wallEnd - wallStart) * 1_000_000)
    }

    private func encodeStep(
        _ step: MetalPrefillStep,
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        runtimeConstantBuffer: MTLBuffer,
        sequenceLength: Int
    ) {
        guard step.shouldExecute(sequenceLength: sequenceLength) else {
            return
        }
        switch step.mode {
        case .batch:
            step.bindings.bind(to: argumentTable)
            step.bindRuntimeArguments(
                argumentTable: argumentTable,
                runtimeConstantBuffer: runtimeConstantBuffer,
                sequenceLengthOffset: PrefillBufferSet.sequenceLengthOffset
            )
            let gridSize = step.resolvedGridSize(sequenceLength: sequenceLength)
            let descriptor = step.resolvedDescriptor(sequenceLength: sequenceLength)
            descriptor.encode(on: encoder, argumentTable: argumentTable, gridSize: gridSize)
        case .lastToken:
            let lastPosition = sequenceLength - 1
            step.bindStaticArguments(argumentTable: argumentTable, position: lastPosition)
            step.descriptor.encode(on: encoder, argumentTable: argumentTable)
        case .perPosition:
            for positionOffset in 0..<sequenceLength {
                step.bindStaticArguments(argumentTable: argumentTable, position: positionOffset)
                if let positionBufferIndex = step.positionBufferIndex {
                    argumentTable.setAddress(
                        runtimeConstantBuffer.gpuAddress
                            + UInt64(PrefillBufferSet.positionOffset(at: positionOffset)),
                        index: positionBufferIndex
                    )
                }
                step.descriptor.encode(on: encoder, argumentTable: argumentTable)
            }
        }
    }

    private func makeStepEntry(
        step: MetalPrefillStep,
        index: Int,
        sequenceLength: Int,
        timing: (gpuMicroseconds: Double, wallMicroseconds: Double),
        iterations: Int
    ) -> MetalPrefillProfile.Entry {
        let grid = step.resolvedGridSize(sequenceLength: sequenceLength)
        let byteTraffic = estimatedByteTraffic(for: step, sequenceLength: sequenceLength)
        return MetalPrefillProfile.Entry(
            scope: "step",
            index: index,
            rangeStart: index,
            rangeEnd: index + 1,
            kernelName: kernelName(for: step),
            category: classify(kernelName(for: step)),
            mode: describe(mode: step.mode),
            layerIndex: step.metadata.layerIndex,
            entryIndex: step.metadata.entryIndex,
            weightTensorName: step.metadata.weightTensorName,
            gridWidth: grid.width,
            gridHeight: grid.height,
            gridDepth: grid.depth,
            threadgroupWidth: step.threadgroupSize.width,
            threadgroupHeight: step.threadgroupSize.height,
            threadgroupDepth: step.threadgroupSize.depth,
            threadgroupMemoryBytes: step.threadgroupMemoryLength,
            bufferBindingCount: step.bindings.buffers.count,
            inlineConstantBytes: inlineConstantBytes(for: step),
            uniqueBoundBufferBytes: uniqueBoundBufferBytes(for: step),
            estimatedReadBytes: byteTraffic.readBytes,
            estimatedWriteBytes: byteTraffic.writeBytes,
            estimatedTotalBytes: byteTraffic.totalBytes,
            estimatedDispatchCount: estimatedDispatchCount(for: step, sequenceLength: sequenceLength),
            totalGpuMicroseconds: timing.gpuMicroseconds,
            averageGpuMicroseconds: timing.gpuMicroseconds / Double(iterations),
            totalWallMicroseconds: timing.wallMicroseconds,
            averageWallMicroseconds: timing.wallMicroseconds / Double(iterations)
        )
    }

    private func makePassEntry(
        range: Range<Int>,
        index: Int,
        plan: MetalPrefillPlan,
        sequenceLength: Int,
        timing: (gpuMicroseconds: Double, wallMicroseconds: Double),
        iterations: Int
    ) -> MetalPrefillProfile.Entry {
        let steps = plan.steps[range].filter {
            $0.shouldExecute(sequenceLength: sequenceLength)
        }
        let bindingCount = steps.reduce(0) { $0 + $1.bindings.buffers.count }
        let inlineBytes = steps.reduce(0) { $0 + inlineConstantBytes(for: $1) }
        let uniqueBytes = uniqueBoundBufferBytes(for: Array(steps))
        let byteTraffic = steps.reduce(PrefillByteTraffic.zero) { partial, step in
            partial + estimatedByteTraffic(for: step, sequenceLength: sequenceLength)
        }
        let dispatchCount = steps.reduce(0) {
            $0 + estimatedDispatchCount(for: $1, sequenceLength: sequenceLength)
        }
        let categories = Array(Set(steps.map { classify(kernelName(for: $0)) })).sorted()
        return MetalPrefillProfile.Entry(
            scope: "pass",
            index: index,
            rangeStart: range.lowerBound,
            rangeEnd: range.upperBound,
            kernelName: "prefill_pass[\(range.lowerBound)..<\(range.upperBound)]",
            category: categories.count == 1 ? categories[0] : "mixed",
            mode: "mixed",
            layerIndex: commonLayerIndex(for: steps),
            entryIndex: nil,
            weightTensorName: nil,
            gridWidth: 0,
            gridHeight: 0,
            gridDepth: 0,
            threadgroupWidth: 0,
            threadgroupHeight: 0,
            threadgroupDepth: 0,
            threadgroupMemoryBytes: steps.reduce(0) { max($0, $1.threadgroupMemoryLength) },
            bufferBindingCount: bindingCount,
            inlineConstantBytes: inlineBytes,
            uniqueBoundBufferBytes: uniqueBytes,
            estimatedReadBytes: byteTraffic.readBytes,
            estimatedWriteBytes: byteTraffic.writeBytes,
            estimatedTotalBytes: byteTraffic.totalBytes,
            estimatedDispatchCount: dispatchCount,
            totalGpuMicroseconds: timing.gpuMicroseconds,
            averageGpuMicroseconds: timing.gpuMicroseconds / Double(iterations),
            totalWallMicroseconds: timing.wallMicroseconds,
            averageWallMicroseconds: timing.wallMicroseconds / Double(iterations)
        )
    }
}

private struct PrefillByteTraffic: Sendable {
    static let zero = PrefillByteTraffic(readBytes: 0, writeBytes: 0)

    let readBytes: Int
    let writeBytes: Int

    var totalBytes: Int {
        readBytes + writeBytes
    }

    static func + (lhs: PrefillByteTraffic, rhs: PrefillByteTraffic) -> PrefillByteTraffic {
        PrefillByteTraffic(
            readBytes: lhs.readBytes + rhs.readBytes,
            writeBytes: lhs.writeBytes + rhs.writeBytes
        )
    }
}

private func kernelName(for step: MetalPrefillStep) -> String {
    step.metadata.kernelName ?? step.pipeline.label ?? "(unlabeled)"
}

private func classify(_ kernelName: String) -> String {
    let name = kernelName.lowercased()
    if name.hasPrefix("embedding_lookup") || name.contains("gather") { return "embedding" }
    if name.hasPrefix("mlp_fused_swiglu_down") { return "projection" }
    if name.hasPrefix("recurrent_block_partial_projection")
        || name.hasPrefix("recurrent_block_partial_reduce") {
        return "projection"
    }
    if name.hasPrefix("gemm_") || name.hasPrefix("gemv_")
        || name.contains("_gemm") || name.contains("_gemv") || name.contains("mpp") {
        return "projection"
    }
    if name.contains("ssm") || name.contains("delta") || name.contains("recurrence") { return "ssm_recurrence" }
    if name.hasPrefix("rms_norm") || name.contains("qk_rms_norm") || name.contains("_norm_") { return "reduction" }
    if name.hasPrefix("flash_attn") || name.hasPrefix("sdpa") { return "attention" }
    if name.hasPrefix("conv1d") || name.hasPrefix("conv_") { return "conv1d" }
    if name.contains("rope") { return "rope" }
    if name.contains("swiglu") || name.contains("silu") || name.contains("sigmoid") { return "elementwise" }
    if name.hasPrefix("copy_") || name.hasPrefix("add_") || name.hasPrefix("residual_")
        || name.hasPrefix("fused_") || name.hasPrefix("kv_cache_") {
        return "structural"
    }
    return "other"
}

private func describe(mode: PrefillStepMode) -> String {
    switch mode {
    case .batch:
        return "batch"
    case .perPosition:
        return "perPosition"
    case .lastToken:
        return "lastToken"
    }
}

private func inlineConstantBytes(for step: MetalPrefillStep) -> Int {
    step.bindings.constantBindings.inlineBindings.reduce(0) { $0 + $1.value.count }
}

private func uniqueBoundBufferBytes(for step: MetalPrefillStep) -> Int {
    uniqueBoundBufferBytes(for: [step])
}

private func uniqueBoundBufferBytes(for steps: [MetalPrefillStep]) -> Int {
    var lengthsByAddress: [UInt64: Int] = [:]
    for step in steps {
        for binding in step.bindings.buffers {
            lengthsByAddress[binding.buffer.gpuAddress] = binding.buffer.length
        }
    }
    return lengthsByAddress.values.reduce(0, +)
}

private func estimatedByteTraffic(
    for step: MetalPrefillStep,
    sequenceLength: Int
) -> PrefillByteTraffic {
    let name = kernelName(for: step).lowercased()
    if let projectionTraffic = estimatedProjectionByteTraffic(
        for: step,
        kernelName: name,
        sequenceLength: sequenceLength
    ) {
        return projectionTraffic
    }
    if name.contains("ssm_recurrence_seq") {
        return estimatedSSMRecurrenceByteTraffic(for: step, sequenceLength: sequenceLength)
    }
    return estimatedBindingByteTraffic(for: step)
}

private func estimatedProjectionByteTraffic(
    for step: MetalPrefillStep,
    kernelName: String,
    sequenceLength: Int
) -> PrefillByteTraffic? {
    let inputBytesPerElement = projectionActivationBytes(for: kernelName)
    let outputBytesPerElement = projectionActivationBytes(for: kernelName)
    let weightBytesPerElement = projectionWeightBytes(for: kernelName)

    if kernelName.hasPrefix("mlp_fused_swiglu_down"),
       let intermediateDimension = uint32Constant(step, index: 4).map(Int.init),
       let outputDimension = uint32Constant(step, index: 5).map(Int.init) {
        let gateBytes = sequenceLength * intermediateDimension * inputBytesPerElement
        let upBytes = sequenceLength * intermediateDimension * inputBytesPerElement
        let weightBytes = outputDimension * intermediateDimension * weightBytesPerElement
        let outputBytes = sequenceLength * outputDimension * outputBytesPerElement
        return PrefillByteTraffic(readBytes: gateBytes + upBytes + weightBytes, writeBytes: outputBytes)
    }

    if kernelName.hasPrefix("batched_gemv"),
       let projectionCount = projectionCount(fromBatchedKernelName: kernelName),
       let inputDimension = uint32Constant(step, index: 1 + 2 * projectionCount).map(Int.init) {
        var outputDimensions: [Int] = []
        outputDimensions.reserveCapacity(projectionCount)
        for projectionIndex in 0..<projectionCount {
            guard let dimension = uint32Constant(
                step,
                index: 2 + 2 * projectionCount + projectionIndex
            ).map(Int.init) else {
                return nil
            }
            outputDimensions.append(dimension)
        }
        let outputDimensionSum = outputDimensions.reduce(0, +)
        let inputBytes = sequenceLength * inputDimension * inputBytesPerElement
        let weightBytes = outputDimensionSum * inputDimension * weightBytesPerElement
        let outputBytes = sequenceLength * outputDimensionSum * outputBytesPerElement
        return PrefillByteTraffic(readBytes: inputBytes + weightBytes, writeBytes: outputBytes)
    }

    if (kernelName.hasPrefix("gemv_") || kernelName.hasPrefix("gemm_") || kernelName.contains("_gemv"))
        && step.bindings.buffers.count >= 3,
       let inputDimension = uint32Constant(step, index: 3).map(Int.init),
       let outputDimension = uint32Constant(step, index: 4).map(Int.init) {
        let inputBytes = sequenceLength * inputDimension * inputBytesPerElement
        let weightBytes = outputDimension * inputDimension * weightBytesPerElement
        let outputBytes = sequenceLength * outputDimension * outputBytesPerElement
        return PrefillByteTraffic(readBytes: inputBytes + weightBytes, writeBytes: outputBytes)
    }

    return nil
}

private func estimatedSSMRecurrenceByteTraffic(
    for step: MetalPrefillStep,
    sequenceLength: Int
) -> PrefillByteTraffic {
    let activationBytes = 4
    let weightBytes = projectionWeightBytes(for: kernelName(for: step).lowercased())
    guard let headCount = uint32Constant(step, index: 11).map(Int.init),
          let groupCount = uint32Constant(step, index: 12).map(Int.init),
          let keyDimension = uint32Constant(step, index: 13).map(Int.init),
          let valueDimension = uint32Constant(step, index: 14).map(Int.init),
          let convKernelSize = uint32Constant(step, index: 15).map(Int.init),
          let activationRowStride = uint32Constant(step, index: 17).map(Int.init) else {
        return estimatedBindingByteTraffic(for: step)
    }

    let keyGroupDimension = groupCount * keyDimension
    let convDimension = 2 * keyGroupDimension + headCount * valueDimension
    let projectedQKVBytes = sequenceLength * activationRowStride * activationBytes
    let projectedZBytes = sequenceLength * activationRowStride * activationBytes
    let projectedBetaBytes = sequenceLength * activationRowStride * activationBytes
    let projectedAlphaBytes = sequenceLength * activationRowStride * activationBytes
    let convWeightBytes = convDimension * convKernelSize * weightBytes
    let normWeightBytes = valueDimension * MemoryLayout<Float>.stride
    let dtAndALogBytes = headCount * MemoryLayout<Float>.stride * 2
    let recurrentStateBytesPerPass = sequenceLength
        * headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride
    let convStateBytesPerPass = sequenceLength
        * convKernelSize * convDimension * weightBytes
    let outputBytes = sequenceLength * activationRowStride * activationBytes

    return PrefillByteTraffic(
        readBytes: projectedQKVBytes + projectedZBytes + projectedBetaBytes + projectedAlphaBytes
            + convWeightBytes + normWeightBytes + dtAndALogBytes
            + recurrentStateBytesPerPass + convStateBytesPerPass,
        writeBytes: recurrentStateBytesPerPass + convStateBytesPerPass + outputBytes
    )
}

private func estimatedBindingByteTraffic(for step: MetalPrefillStep) -> PrefillByteTraffic {
    guard let pattern = step.metadata.bufferAccessPattern else {
        let bytes = step.bindings.buffers.reduce(0) { total, binding in
            total + max(0, binding.buffer.length - binding.offset)
        }
        return PrefillByteTraffic(readBytes: bytes, writeBytes: 0)
    }
    var readBytes = 0
    var writeBytes = 0
    for binding in step.bindings.buffers {
        let bytes = max(0, binding.buffer.length - binding.offset)
        if pattern.readIndices.contains(binding.index) {
            readBytes += bytes
        }
        if pattern.writeIndices.contains(binding.index) {
            writeBytes += bytes
        }
    }
    return PrefillByteTraffic(readBytes: readBytes, writeBytes: writeBytes)
}

private func uint32Constant(_ step: MetalPrefillStep, index: Int) -> UInt32? {
    for binding in step.bindings.constants where binding.index == index {
        switch binding {
        case .inline(let bytes):
            guard bytes.value.count >= MemoryLayout<UInt32>.size else {
                return nil
            }
            return bytes.value.withUnsafeBytes { rawBuffer in
                rawBuffer.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
            }
        case .buffer(let buffer):
            guard buffer.length >= MemoryLayout<UInt32>.size,
                  buffer.offset + MemoryLayout<UInt32>.size <= buffer.buffer.length,
                  buffer.buffer.storageMode != .private else {
                return nil
            }
            return buffer.buffer.contents()
                .advanced(by: buffer.offset)
                .loadUnaligned(as: UInt32.self)
        }
    }
    return nil
}

private func projectionCount(fromBatchedKernelName kernelName: String) -> Int? {
    for prefix in ["batched_gemv", "batched_gemm"] {
        guard let range = kernelName.range(of: prefix) else {
            continue
        }
        let suffix = kernelName[range.upperBound...]
        let digits = String(suffix.prefix { $0.isNumber })
        if let count = Int(digits) {
            return count
        }
    }
    return nil
}

private func projectionActivationBytes(for kernelName: String) -> Int {
    kernelName.contains("_f32") || kernelName.contains("f32s") ? 4 : 2
}

private func projectionWeightBytes(for kernelName: String) -> Int {
    if kernelName.contains("_q3") {
        return 1
    }
    if kernelName.contains("_q4") {
        return 1
    }
    if kernelName.contains("_q8") {
        return 1
    }
    if kernelName.contains("bf16") || kernelName.contains("f16") {
        return 2
    }
    return 4
}

private func estimatedDispatchCount(for step: MetalPrefillStep, sequenceLength: Int) -> Int {
    switch step.mode {
    case .batch, .lastToken:
        return 1
    case .perPosition:
        return sequenceLength
    }
}

private func commonLayerIndex<S: Sequence>(for steps: S) -> Int? where S.Element == MetalPrefillStep {
    let layerIndices = Set(steps.compactMap(\.metadata.layerIndex))
    return layerIndices.count == 1 ? layerIndices.first : nil
}

private func csvEscape(_ value: String) -> String {
    guard value.contains(",") || value.contains("\"") || value.contains("\n") else {
        return value
    }
    return "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
}
