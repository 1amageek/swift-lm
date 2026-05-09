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
        self.schemaVersion = 1
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
                "estimatedDispatchCount",
                "averageGpuMicroseconds",
                "averageWallMicroseconds",
            ].joined(separator: ","),
        ]
        for entry in entries {
            var row: [String] = []
            row.reserveCapacity(21)
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
            row.append(String(entry.estimatedDispatchCount))
            row.append(String(format: "%.3f", entry.averageGpuMicroseconds))
            row.append(String(format: "%.3f", entry.averageWallMicroseconds))
            lines.append(row.joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    func writeArtifacts(directory: URL, basename: String) throws -> [URL] {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)

        let jsonURL = directory.appendingPathComponent("\(basename).json")
        let csvURL = directory.appendingPathComponent("\(basename).csv")

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let jsonData = try encoder.encode(self)
        try jsonData.write(to: jsonURL, options: .atomic)
        try Data(csvString.utf8).write(to: csvURL, options: .atomic)
        return [jsonURL, csvURL]
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

        for _ in 0..<warmupIterations {
            for step in plan.steps {
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
        entries.reserveCapacity(plan.steps.count)
        for (index, step) in plan.steps.enumerated() {
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
            stepCount: plan.steps.count,
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
            stepCount: plan.steps.count,
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
        let steps = plan.steps[range]
        let bindingCount = steps.reduce(0) { $0 + $1.bindings.buffers.count }
        let inlineBytes = steps.reduce(0) { $0 + inlineConstantBytes(for: $1) }
        let uniqueBytes = uniqueBoundBufferBytes(for: Array(steps))
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
            estimatedDispatchCount: dispatchCount,
            totalGpuMicroseconds: timing.gpuMicroseconds,
            averageGpuMicroseconds: timing.gpuMicroseconds / Double(iterations),
            totalWallMicroseconds: timing.wallMicroseconds,
            averageWallMicroseconds: timing.wallMicroseconds / Double(iterations)
        )
    }
}

private func kernelName(for step: MetalPrefillStep) -> String {
    step.metadata.kernelName ?? step.pipeline.label ?? "(unlabeled)"
}

private func classify(_ kernelName: String) -> String {
    let name = kernelName.lowercased()
    if name.hasPrefix("embedding_lookup") || name.contains("gather") { return "embedding" }
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

private func estimatedDispatchCount(for step: MetalPrefillStep, sequenceLength: Int) -> Int {
    switch step.mode {
    case .batch, .lastToken:
        return 1
    case .perPosition:
        return sequenceLength
    }
}

private func commonLayerIndex(for steps: ArraySlice<MetalPrefillStep>) -> Int? {
    let layerIndices = Set(steps.compactMap(\.metadata.layerIndex))
    return layerIndices.count == 1 ? layerIndices.first : nil
}

private func csvEscape(_ value: String) -> String {
    guard value.contains(",") || value.contains("\"") || value.contains("\n") else {
        return value
    }
    return "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
}
