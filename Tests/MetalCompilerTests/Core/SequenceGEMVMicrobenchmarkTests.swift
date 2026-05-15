import Foundation
import Metal
import Testing
@testable import MetalCompiler

@Suite("Sequence GEMV Microbenchmark", .serialized)
struct SequenceGEMVMicrobenchmarkTests {
    private static let sequenceLengths = [16, 64, 128]
    private static let iterations = 5
    private static let warmupIterations = 1

    @Test("BF16 single sequence GEMV real-shape microbench")
    func bf16SingleSequenceGEMVRealShapeMicrobench() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try MicrobenchmarkHarness(device: device)
        let shapes = [
            Shape(role: "attn_or_ssm.out_proj", inputDimension: 2048, outputDimension: 1024),
            Shape(role: "mlp.down_proj", inputDimension: 3584, outputDimension: 1024),
        ]
        let variants = [
            Variant(name: "base", kernelName: "bench_gemv_seq_bf16_f32s", sequenceTile: 1, rowsPerSimdgroup: 1),
            Variant(name: "row2", kernelName: "bench_gemv_seq_bf16_f32s_rps2", sequenceTile: 1, rowsPerSimdgroup: 2),
            Variant(name: "tile2", kernelName: "bench_gemv_seq_bf16_f32s_tile2", sequenceTile: 2, rowsPerSimdgroup: 1),
            Variant(name: "tile4", kernelName: "bench_gemv_seq_bf16_f32s_tile4", sequenceTile: 4, rowsPerSimdgroup: 1),
        ]

        var rows: [ResultRow] = []
        for shape in shapes {
            for sequenceLength in Self.sequenceLengths {
                for variant in variants {
                    let measurement = try harness.measure(
                        shape: shape,
                        variant: variant,
                        sequenceLength: sequenceLength,
                        iterations: Self.iterations,
                        warmupIterations: Self.warmupIterations
                    )
                    rows.append(measurement)
                }
            }
        }

        let routePromotionRows = summarizeSingleRoutePromotions(rows: rows)
        let artifact = try writeCSV(rows: rows)
        let routePromotionArtifact = try writeSingleRoutePromotionCSV(rows: routePromotionRows)
        printReport(rows: rows, routePromotionRows: routePromotionRows, artifact: artifact, routePromotionArtifact: routePromotionArtifact)
        #expect(rows.count == shapes.count * Self.sequenceLengths.count * variants.count)
        #expect(routePromotionRows.count == shapes.count * (variants.count - 1))
    }

    @Test("BF16 batched sequence GEMV real-shape microbench")
    func bf16BatchedSequenceGEMVRealShapeMicrobench() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try MicrobenchmarkHarness(device: device)
        let shapes = [
            BatchedShape(
                role: "mlp.gate_up",
                inputDimension: 2048,
                outputDimensions: [3584, 3584]
            ),
            BatchedShape(
                role: "self_attn.qkv",
                inputDimension: 2048,
                outputDimensions: [4096, 512, 512]
            ),
            BatchedShape(
                role: "linear_attn.in_proj",
                inputDimension: 2048,
                outputDimensions: [6144, 2048, 16, 16]
            ),
        ]
        let variants = [
            BatchedVariant(name: "base", sequenceTile: 1),
            BatchedVariant(name: "tile2", sequenceTile: 2),
            BatchedVariant(name: "tile4", sequenceTile: 4),
        ]

        var rows: [BatchedResultRow] = []
        for shape in shapes {
            for sequenceLength in Self.sequenceLengths {
                for variant in variants {
                    let measurement = try harness.measureBatched(
                        shape: shape,
                        variant: variant,
                        sequenceLength: sequenceLength,
                        iterations: Self.iterations,
                        warmupIterations: Self.warmupIterations
                    )
                    rows.append(measurement)
                }
            }
        }

        let artifact = try writeBatchedCSV(rows: rows)
        printBatchedReport(rows: rows, artifact: artifact)
        #expect(rows.count == shapes.count * Self.sequenceLengths.count * variants.count)
    }

    @Test("BF16 fused SwiGLU down rows-per-threadgroup microbench")
    func bf16FusedSwigluDownRowsMicrobench() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try MicrobenchmarkHarness(device: device)
        let shape = Shape(role: "mlp.down_proj", inputDimension: 3584, outputDimension: 1024)
        let variants = [
            FusedRowsVariant(
                name: "rows2",
                kernelName: "bench_mlp_fused_swiglu_down_seq_bf16_f32s",
                simdgroupsPerThreadgroup: 2,
                rowsPerSimdgroup: 1
            ),
            FusedRowsVariant(
                name: "rows4",
                kernelName: "bench_mlp_fused_swiglu_down_seq_bf16_f32s",
                simdgroupsPerThreadgroup: 4,
                rowsPerSimdgroup: 1
            ),
            FusedRowsVariant(
                name: "rows8",
                kernelName: "bench_mlp_fused_swiglu_down_seq_bf16_f32s",
                simdgroupsPerThreadgroup: 8,
                rowsPerSimdgroup: 1
            ),
            FusedRowsVariant(
                name: "rows16_rps2",
                kernelName: "bench_mlp_fused_swiglu_down_seq_bf16_f32s_rps2",
                simdgroupsPerThreadgroup: 8,
                rowsPerSimdgroup: 2
            ),
        ]

        var rows: [FusedRowsResultRow] = []
        for sequenceLength in Self.sequenceLengths {
            for variant in variants {
                let measurement = try harness.measureFusedSwigluDown(
                    shape: shape,
                    variant: variant,
                    sequenceLength: sequenceLength,
                    iterations: Self.iterations,
                    warmupIterations: Self.warmupIterations
                )
                rows.append(measurement)
            }
        }

        let artifact = try writeFusedRowsCSV(rows: rows)
        printFusedRowsReport(rows: rows, artifact: artifact)
        #expect(rows.count == Self.sequenceLengths.count * variants.count)
    }

    @Test("BF16 single sequence GEMV route admissions require production sequence wins")
    func bf16SingleSequenceGEMVRouteAdmissionsRequireProductionSequenceWins() {
        let rows = [
            makeResultFixture(role: "dependent", sequenceLength: 16, variant: "base", averageGpuMicroseconds: 100.0),
            makeResultFixture(role: "dependent", sequenceLength: 16, variant: "tile2", averageGpuMicroseconds: 50.0),
            makeResultFixture(role: "dependent", sequenceLength: 64, variant: "base", averageGpuMicroseconds: 100.0),
            makeResultFixture(role: "dependent", sequenceLength: 64, variant: "tile2", averageGpuMicroseconds: 90.0),
            makeResultFixture(role: "dependent", sequenceLength: 64, variant: "row2", averageGpuMicroseconds: 90.0),
            makeResultFixture(role: "dependent", sequenceLength: 128, variant: "base", averageGpuMicroseconds: 100.0),
            makeResultFixture(role: "dependent", sequenceLength: 128, variant: "tile2", averageGpuMicroseconds: 90.0),
            makeResultFixture(role: "dependent", sequenceLength: 128, variant: "row2", averageGpuMicroseconds: 99.0),
        ]

        let routeRows = summarizeSingleRoutePromotions(rows: rows)
        let tile2 = routeRows.first { $0.variant == "tile2" }
        let row2 = routeRows.first { $0.variant == "row2" }
        #expect(tile2?.routePromotionAdmission == "candidate-single-gemv-default-route")
        #expect(tile2?.failingSequenceLengths == [])
        #expect(row2?.routePromotionAdmission == "reject-cross-sequence-threshold")
        #expect(row2?.failingSequenceLengths == [128])
        #expect(row2?.thresholdShortfallPercent == 2.0)
    }

    private func printReport(
        rows: [ResultRow],
        routePromotionRows: [SingleGEMVRoutePromotionRow],
        artifact: URL,
        routePromotionArtifact: URL
    ) {
        print()
        print("=== BF16 single sequence GEMV real-shape microbench ===")
        print("artifact: \(artifact.path)")
        print("route promotion artifact: \(routePromotionArtifact.path)")
        print("role                    seq  variant  avg_us  us/output  grid      tg")
        for row in rows.sorted(by: rowSort) {
            let role = row.role.padding(toLength: 23, withPad: " ", startingAt: 0)
            let variant = row.variant.padding(toLength: 7, withPad: " ", startingAt: 0)
            let grid = "\(row.gridWidth)x\(row.gridHeight)".padding(toLength: 9, withPad: " ", startingAt: 0)
            print("  \(role) \(String(format: "%3d", row.sequenceLength))  \(variant) \(String(format: "%7.1f", row.averageGpuMicroseconds))  \(String(format: "%8.4f", row.microsecondsPerOutput))  \(grid) \(row.threadgroupWidth)")
        }
        print()
        print("=== BF16 single sequence GEMV route promotion admissions ===")
        print("role                    variant  pass/required  min_speedup  shortfall  failing_seq  admission")
        for row in routePromotionRows.sorted(by: singleRoutePromotionSort) {
            let role = row.role.padding(toLength: 23, withPad: " ", startingAt: 0)
            let variant = row.variant.padding(toLength: 7, withPad: " ", startingAt: 0)
            let failingSequences = row.failingSequenceLengths.map(String.init).joined(separator: "|")
            print("  \(role) \(variant)  \(row.passingSequenceCount)/\(row.requiredSequenceCount)          \(String(format: "%7.2f", row.minimumSpeedupPercent))%  \(String(format: "%8.2f", row.thresholdShortfallPercent))%  \(failingSequences.padding(toLength: 11, withPad: " ", startingAt: 0))  \(row.routePromotionAdmission)")
        }
    }

    private func writeCSV(rows: [ResultRow]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/sequence-gemv-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("qwen35-bf16-single-sequence-gemv.csv")
        var lines = [
            [
                "role",
                "inputDimension",
                "outputDimension",
                "sequenceLength",
                "variant",
                "sequenceTile",
                "gridWidth",
                "gridHeight",
                "threadgroupWidth",
                "averageGpuMicroseconds",
                "microsecondsPerOutput",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: rowSort) {
            lines.append([
                row.role,
                String(row.inputDimension),
                String(row.outputDimension),
                String(row.sequenceLength),
                row.variant,
                String(row.sequenceTile),
                String(row.gridWidth),
                String(row.gridHeight),
                String(row.threadgroupWidth),
                String(format: "%.3f", row.averageGpuMicroseconds),
                String(format: "%.6f", row.microsecondsPerOutput),
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func writeSingleRoutePromotionCSV(rows: [SingleGEMVRoutePromotionRow]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/sequence-gemv-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("qwen35-bf16-single-sequence-gemv-route-promotions.csv")
        var lines = [
            [
                "role",
                "variant",
                "productionSequenceLengths",
                "speedupPercents",
                "passingSequenceCount",
                "requiredSequenceCount",
                "minimumSpeedupPercent",
                "thresholdShortfallPercent",
                "failingSequenceLengths",
                "routePromotionAdmission",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: singleRoutePromotionSort) {
            lines.append([
                row.role,
                row.variant,
                row.productionSequenceLengths.map(String.init).joined(separator: "|"),
                row.speedupPercents.map { String(format: "%.3f", $0) }.joined(separator: "|"),
                String(row.passingSequenceCount),
                String(row.requiredSequenceCount),
                String(format: "%.3f", row.minimumSpeedupPercent),
                String(format: "%.3f", row.thresholdShortfallPercent),
                row.failingSequenceLengths.map(String.init).joined(separator: "|"),
                row.routePromotionAdmission,
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func printBatchedReport(rows: [BatchedResultRow], artifact: URL) {
        print()
        print("=== BF16 batched sequence GEMV real-shape microbench ===")
        print("artifact: \(artifact.path)")
        print("role                 seq  variant  count  avg_us  us/output  grid      tg")
        for row in rows.sorted(by: batchedRowSort) {
            let role = row.role.padding(toLength: 20, withPad: " ", startingAt: 0)
            let variant = row.variant.padding(toLength: 7, withPad: " ", startingAt: 0)
            let grid = "\(row.gridWidth)x\(row.gridHeight)".padding(toLength: 9, withPad: " ", startingAt: 0)
            print("  \(role) \(String(format: "%3d", row.sequenceLength))  \(variant) \(String(format: "%5d", row.projectionCount))  \(String(format: "%7.1f", row.averageGpuMicroseconds))  \(String(format: "%8.4f", row.microsecondsPerOutput))  \(grid) \(row.threadgroupWidth)")
        }
    }

    private func writeBatchedCSV(rows: [BatchedResultRow]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/sequence-gemv-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("qwen35-bf16-batched-sequence-gemv.csv")
        var lines = [
            [
                "role",
                "inputDimension",
                "outputDimensions",
                "totalOutputDimension",
                "sequenceLength",
                "variant",
                "sequenceTile",
                "projectionCount",
                "gridWidth",
                "gridHeight",
                "threadgroupWidth",
                "averageGpuMicroseconds",
                "microsecondsPerOutput",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: batchedRowSort) {
            lines.append([
                row.role,
                String(row.inputDimension),
                row.outputDimensions.map(String.init).joined(separator: "+"),
                String(row.totalOutputDimension),
                String(row.sequenceLength),
                row.variant,
                String(row.sequenceTile),
                String(row.projectionCount),
                String(row.gridWidth),
                String(row.gridHeight),
                String(row.threadgroupWidth),
                String(format: "%.3f", row.averageGpuMicroseconds),
                String(format: "%.6f", row.microsecondsPerOutput),
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func printFusedRowsReport(rows: [FusedRowsResultRow], artifact: URL) {
        print()
        print("=== BF16 fused SwiGLU down rows-per-threadgroup microbench ===")
        print("artifact: \(artifact.path)")
        print("role           seq  variant      avg_us  us/output  grid      tg")
        for row in rows.sorted(by: fusedRowsSort) {
            let role = row.role.padding(toLength: 14, withPad: " ", startingAt: 0)
            let variant = row.variant.padding(toLength: 12, withPad: " ", startingAt: 0)
            let grid = "\(row.gridWidth)x\(row.gridHeight)".padding(toLength: 9, withPad: " ", startingAt: 0)
            print("  \(role) \(String(format: "%3d", row.sequenceLength))  \(variant) \(String(format: "%7.1f", row.averageGpuMicroseconds))  \(String(format: "%8.4f", row.microsecondsPerOutput))  \(grid) \(row.threadgroupWidth)")
        }
    }

    private func writeFusedRowsCSV(rows: [FusedRowsResultRow]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/sequence-gemv-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("qwen35-bf16-fused-swiglu-down-rows.csv")
        var lines = [
            [
                "role",
                "inputDimension",
                "outputDimension",
                "sequenceLength",
                "variant",
                "rowsPerThreadgroup",
                "simdgroupsPerThreadgroup",
                "rowsPerSimdgroup",
                "gridWidth",
                "gridHeight",
                "threadgroupWidth",
                "averageGpuMicroseconds",
                "microsecondsPerOutput",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: fusedRowsSort) {
            lines.append([
                row.role,
                String(row.inputDimension),
                String(row.outputDimension),
                String(row.sequenceLength),
                row.variant,
                String(row.rowsPerThreadgroup),
                String(row.simdgroupsPerThreadgroup),
                String(row.rowsPerSimdgroup),
                String(row.gridWidth),
                String(row.gridHeight),
                String(row.threadgroupWidth),
                String(format: "%.3f", row.averageGpuMicroseconds),
                String(format: "%.6f", row.microsecondsPerOutput),
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func rowSort(_ lhs: ResultRow, _ rhs: ResultRow) -> Bool {
        if lhs.role != rhs.role { return lhs.role < rhs.role }
        if lhs.sequenceLength != rhs.sequenceLength { return lhs.sequenceLength < rhs.sequenceLength }
        return lhs.sequenceTile < rhs.sequenceTile
    }

    private func batchedRowSort(_ lhs: BatchedResultRow, _ rhs: BatchedResultRow) -> Bool {
        if lhs.role != rhs.role { return lhs.role < rhs.role }
        if lhs.sequenceLength != rhs.sequenceLength { return lhs.sequenceLength < rhs.sequenceLength }
        return lhs.sequenceTile < rhs.sequenceTile
    }

    private func fusedRowsSort(_ lhs: FusedRowsResultRow, _ rhs: FusedRowsResultRow) -> Bool {
        if lhs.role != rhs.role { return lhs.role < rhs.role }
        if lhs.sequenceLength != rhs.sequenceLength { return lhs.sequenceLength < rhs.sequenceLength }
        if lhs.rowsPerThreadgroup != rhs.rowsPerThreadgroup {
            return lhs.rowsPerThreadgroup < rhs.rowsPerThreadgroup
        }
        return lhs.rowsPerSimdgroup < rhs.rowsPerSimdgroup
    }

    private func singleRoutePromotionSort(
        _ lhs: SingleGEMVRoutePromotionRow,
        _ rhs: SingleGEMVRoutePromotionRow
    ) -> Bool {
        if lhs.role != rhs.role { return lhs.role < rhs.role }
        return lhs.variant < rhs.variant
    }

    private func summarizeSingleRoutePromotions(rows: [ResultRow]) -> [SingleGEMVRoutePromotionRow] {
        let productionSequenceLengths = Self.sequenceLengths.filter {
            $0 >= SingleGEMVRoutePromotionFactory.minimumPromotionSequenceLength
        }
        let roles = Array(Set(rows.map(\.role))).sorted()
        let variants = Array(Set(rows.map(\.variant))).filter { $0 != "base" }.sorted()
        return roles.flatMap { role in
            variants.map { variant in
                SingleGEMVRoutePromotionFactory.make(
                    role: role,
                    variant: variant,
                    productionSequenceLengths: productionSequenceLengths,
                    rows: rows
                )
            }
        }
    }

    private func makeResultFixture(
        role: String,
        sequenceLength: Int,
        variant: String,
        averageGpuMicroseconds: Double
    ) -> ResultRow {
        ResultRow(
            role: role,
            inputDimension: 2048,
            outputDimension: 1024,
            sequenceLength: sequenceLength,
            variant: variant,
            sequenceTile: variant == "base" ? 1 : 2,
            gridWidth: 1024,
            gridHeight: sequenceLength,
            threadgroupWidth: 64,
            averageGpuMicroseconds: averageGpuMicroseconds
        )
    }
}

private struct Shape {
    let role: String
    let inputDimension: Int
    let outputDimension: Int
}

private struct BatchedShape {
    let role: String
    let inputDimension: Int
    let outputDimensions: [Int]

    var projectionCount: Int {
        outputDimensions.count
    }

    var totalOutputDimension: Int {
        outputDimensions.reduce(0, +)
    }

    var maximumOutputDimension: Int {
        outputDimensions.max() ?? 0
    }
}

private struct Variant {
    let name: String
    let kernelName: String
    let sequenceTile: Int
    let rowsPerSimdgroup: Int
}

private struct BatchedVariant {
    let name: String
    let sequenceTile: Int
}

private struct FusedRowsVariant {
    let name: String
    let kernelName: String
    let simdgroupsPerThreadgroup: Int
    let rowsPerSimdgroup: Int

    var rowsPerThreadgroup: Int {
        simdgroupsPerThreadgroup * rowsPerSimdgroup
    }
}

private struct ResultRow {
    let role: String
    let inputDimension: Int
    let outputDimension: Int
    let sequenceLength: Int
    let variant: String
    let sequenceTile: Int
    let gridWidth: Int
    let gridHeight: Int
    let threadgroupWidth: Int
    let averageGpuMicroseconds: Double

    var microsecondsPerOutput: Double {
        averageGpuMicroseconds / Double(sequenceLength * outputDimension)
    }
}

private struct SingleGEMVRoutePromotionRow {
    let role: String
    let variant: String
    let productionSequenceLengths: [Int]
    let speedupPercents: [Double]
    let passingSequenceCount: Int
    let requiredSequenceCount: Int
    let failingSequenceLengths: [Int]
    let routePromotionAdmission: String

    var minimumSpeedupPercent: Double {
        speedupPercents.min() ?? .nan
    }

    var thresholdShortfallPercent: Double {
        guard minimumSpeedupPercent.isFinite else { return .infinity }
        return max(0.0, SingleGEMVRoutePromotionFactory.promotionSpeedupThresholdPercent - minimumSpeedupPercent)
    }
}

private enum SingleGEMVRoutePromotionFactory {
    static let promotionSpeedupThresholdPercent = 3.0
    static let minimumPromotionSequenceLength = 64

    static func make(
        role: String,
        variant: String,
        productionSequenceLengths: [Int],
        rows: [ResultRow]
    ) -> SingleGEMVRoutePromotionRow {
        var speedupPercents: [Double] = []

        for sequenceLength in productionSequenceLengths {
            let base = rows.first {
                $0.role == role && $0.sequenceLength == sequenceLength && $0.variant == "base"
            }
            let candidate = rows.first {
                $0.role == role && $0.sequenceLength == sequenceLength && $0.variant == variant
            }
            guard let base, let candidate else {
                speedupPercents.append(-Double.infinity)
                continue
            }
            let speedup = (base.averageGpuMicroseconds - candidate.averageGpuMicroseconds)
                / base.averageGpuMicroseconds * 100.0
            speedupPercents.append(speedup)
        }

        let passingCount = speedupPercents.filter { $0 >= promotionSpeedupThresholdPercent }.count
        let failingSequenceLengths = zip(productionSequenceLengths, speedupPercents).compactMap { sequenceLength, speedup in
            speedup >= promotionSpeedupThresholdPercent ? nil : sequenceLength
        }
        let admission = passingCount == productionSequenceLengths.count
            ? "candidate-single-gemv-default-route"
            : "reject-cross-sequence-threshold"

        return SingleGEMVRoutePromotionRow(
            role: role,
            variant: variant,
            productionSequenceLengths: productionSequenceLengths,
            speedupPercents: speedupPercents,
            passingSequenceCount: passingCount,
            requiredSequenceCount: productionSequenceLengths.count,
            failingSequenceLengths: failingSequenceLengths,
            routePromotionAdmission: admission
        )
    }
}

private struct BatchedResultRow {
    let role: String
    let inputDimension: Int
    let outputDimensions: [Int]
    let sequenceLength: Int
    let variant: String
    let sequenceTile: Int
    let projectionCount: Int
    let gridWidth: Int
    let gridHeight: Int
    let threadgroupWidth: Int
    let averageGpuMicroseconds: Double

    var totalOutputDimension: Int {
        outputDimensions.reduce(0, +)
    }

    var microsecondsPerOutput: Double {
        averageGpuMicroseconds / Double(sequenceLength * totalOutputDimension)
    }
}

private struct FusedRowsResultRow {
    let role: String
    let inputDimension: Int
    let outputDimension: Int
    let sequenceLength: Int
    let variant: String
    let rowsPerThreadgroup: Int
    let simdgroupsPerThreadgroup: Int
    let rowsPerSimdgroup: Int
    let gridWidth: Int
    let gridHeight: Int
    let threadgroupWidth: Int
    let averageGpuMicroseconds: Double

    var microsecondsPerOutput: Double {
        averageGpuMicroseconds / Double(sequenceLength * outputDimension)
    }
}

private struct MicrobenchmarkHarness {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipelines: [String: MTLComputePipelineState]

    init(device: MTLDevice) throws {
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create command queue")
        }
        self.device = device
        self.queue = queue

        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateSequenceGEMV(
                name: "bench_gemv_seq_bf16_f32s",
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateTiledSequenceGEMV(
                name: "bench_gemv_seq_bf16_f32s_tile2",
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: 2
            ),
            MetalSourceGenerator.generateSequenceGEMV(
                name: "bench_gemv_seq_bf16_f32s_rps2",
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                rowsPerSimdgroup: 2
            ),
            MetalSourceGenerator.generateTiledSequenceGEMV(
                name: "bench_gemv_seq_bf16_f32s_tile4",
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: 4
            ),
            MetalSourceGenerator.generateBatchedSequenceGEMV(
                name: "bench_batched_gemv2_seq_bf16_f32s",
                count: 2,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateTiledBatchedSequenceGEMV(
                name: "bench_batched_gemv2_seq_bf16_f32s_tile2",
                count: 2,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: 2
            ),
            MetalSourceGenerator.generateTiledBatchedSequenceGEMV(
                name: "bench_batched_gemv2_seq_bf16_f32s_tile4",
                count: 2,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: 4
            ),
            MetalSourceGenerator.generateBatchedSequenceGEMV(
                name: "bench_batched_gemv3_seq_bf16_f32s",
                count: 3,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateTiledBatchedSequenceGEMV(
                name: "bench_batched_gemv3_seq_bf16_f32s_tile2",
                count: 3,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: 2
            ),
            MetalSourceGenerator.generateTiledBatchedSequenceGEMV(
                name: "bench_batched_gemv3_seq_bf16_f32s_tile4",
                count: 3,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: 4
            ),
            MetalSourceGenerator.generateBatchedSequenceGEMV(
                name: "bench_batched_gemv4_seq_bf16_f32s",
                count: 4,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateTiledBatchedSequenceGEMV(
                name: "bench_batched_gemv4_seq_bf16_f32s_tile2",
                count: 4,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: 2
            ),
            MetalSourceGenerator.generateTiledBatchedSequenceGEMV(
                name: "bench_batched_gemv4_seq_bf16_f32s_tile4",
                count: 4,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: 4
            ),
            MetalSourceGenerator.generateFusedSwigluDownSequenceGEMV(
                name: "bench_mlp_fused_swiglu_down_seq_bf16_f32s",
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateFusedSwigluDownSequenceGEMV(
                name: "bench_mlp_fused_swiglu_down_seq_bf16_f32s_rps2",
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                rowsPerSimdgroup: 2
            ),
        ].joined(separator: "\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let names = [
            "bench_gemv_seq_bf16_f32s",
            "bench_gemv_seq_bf16_f32s_tile2",
            "bench_gemv_seq_bf16_f32s_rps2",
            "bench_gemv_seq_bf16_f32s_tile4",
            "bench_batched_gemv2_seq_bf16_f32s",
            "bench_batched_gemv2_seq_bf16_f32s_tile2",
            "bench_batched_gemv2_seq_bf16_f32s_tile4",
            "bench_batched_gemv3_seq_bf16_f32s",
            "bench_batched_gemv3_seq_bf16_f32s_tile2",
            "bench_batched_gemv3_seq_bf16_f32s_tile4",
            "bench_batched_gemv4_seq_bf16_f32s",
            "bench_batched_gemv4_seq_bf16_f32s_tile2",
            "bench_batched_gemv4_seq_bf16_f32s_tile4",
            "bench_mlp_fused_swiglu_down_seq_bf16_f32s",
            "bench_mlp_fused_swiglu_down_seq_bf16_f32s_rps2",
        ]
        var compiled: [String: MTLComputePipelineState] = [:]
        for name in names {
            guard let function = library.makeFunction(name: name) else {
                throw MetalCompilerError.kernelNotFound(name)
            }
            compiled[name] = try device.makeComputePipelineState(function: function)
        }
        self.pipelines = compiled
    }

    func measure(
        shape: Shape,
        variant: Variant,
        sequenceLength: Int,
        iterations: Int,
        warmupIterations: Int
    ) throws -> ResultRow {
        guard let pipeline = pipelines[variant.kernelName] else {
            throw MetalCompilerError.kernelNotFound(variant.kernelName)
        }
        let inputValues = makeInputValues(count: sequenceLength * shape.inputDimension)
        let weights = makeWeights(count: shape.outputDimension * shape.inputDimension)
        let inputBuffer = try makeSharedBuffer(values: inputValues)
        let weightBuffer = try makeSharedBuffer(values: weights)
        let outputBuffer = try makeZeroedSharedBuffer(
            byteLength: sequenceLength * shape.outputDimension * MemoryLayout<Float>.stride
        )
        let geometry = dispatchGeometry(
            pipeline: pipeline,
            outputDimension: shape.outputDimension,
            sequenceLength: sequenceLength,
            sequenceTile: variant.sequenceTile,
            rowsPerSimdgroup: variant.rowsPerSimdgroup
        )

        for _ in 0..<warmupIterations {
            _ = try execute(
                pipeline: pipeline,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                shape: shape,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        var totalMicroseconds = 0.0
        for _ in 0..<iterations {
            totalMicroseconds += try execute(
                pipeline: pipeline,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                shape: shape,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        return ResultRow(
            role: shape.role,
            inputDimension: shape.inputDimension,
            outputDimension: shape.outputDimension,
            sequenceLength: sequenceLength,
            variant: variant.name,
            sequenceTile: variant.sequenceTile,
            gridWidth: geometry.grid.width,
            gridHeight: geometry.grid.height,
            threadgroupWidth: geometry.threadgroup.width,
            averageGpuMicroseconds: totalMicroseconds / Double(iterations)
        )
    }

    func measureBatched(
        shape: BatchedShape,
        variant: BatchedVariant,
        sequenceLength: Int,
        iterations: Int,
        warmupIterations: Int
    ) throws -> BatchedResultRow {
        let kernelName = batchedKernelName(count: shape.projectionCount, sequenceTile: variant.sequenceTile)
        guard let pipeline = pipelines[kernelName] else {
            throw MetalCompilerError.kernelNotFound(kernelName)
        }
        let inputValues = makeInputValues(count: sequenceLength * shape.inputDimension)
        let weights = shape.outputDimensions.map { outputDimension in
            makeWeights(count: outputDimension * shape.inputDimension)
        }
        let inputBuffer = try makeSharedBuffer(values: inputValues)
        let weightBuffers = try weights.map(makeSharedBuffer)
        let outputByteLength = sequenceLength * shape.maximumOutputDimension * MemoryLayout<Float>.stride
        let outputBuffers = try shape.outputDimensions.map { _ in
            try makeZeroedSharedBuffer(byteLength: outputByteLength)
        }
        let geometry = dispatchGeometry(
            pipeline: pipeline,
            outputDimension: shape.totalOutputDimension,
            sequenceLength: sequenceLength,
            sequenceTile: variant.sequenceTile
        )

        for _ in 0..<warmupIterations {
            _ = try executeBatched(
                pipeline: pipeline,
                inputBuffer: inputBuffer,
                weightBuffers: weightBuffers,
                outputBuffers: outputBuffers,
                shape: shape,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        var totalMicroseconds = 0.0
        for _ in 0..<iterations {
            totalMicroseconds += try executeBatched(
                pipeline: pipeline,
                inputBuffer: inputBuffer,
                weightBuffers: weightBuffers,
                outputBuffers: outputBuffers,
                shape: shape,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        return BatchedResultRow(
            role: shape.role,
            inputDimension: shape.inputDimension,
            outputDimensions: shape.outputDimensions,
            sequenceLength: sequenceLength,
            variant: variant.name,
            sequenceTile: variant.sequenceTile,
            projectionCount: shape.projectionCount,
            gridWidth: geometry.grid.width,
            gridHeight: geometry.grid.height,
            threadgroupWidth: geometry.threadgroup.width,
            averageGpuMicroseconds: totalMicroseconds / Double(iterations)
        )
    }

    func measureFusedSwigluDown(
        shape: Shape,
        variant: FusedRowsVariant,
        sequenceLength: Int,
        iterations: Int,
        warmupIterations: Int
    ) throws -> FusedRowsResultRow {
        guard let pipeline = pipelines[variant.kernelName] else {
            throw MetalCompilerError.kernelNotFound(variant.kernelName)
        }
        let gateValues = makeInputValues(count: sequenceLength * shape.inputDimension)
        let upValues = makeUpValues(count: sequenceLength * shape.inputDimension)
        let weights = makeWeights(count: shape.outputDimension * shape.inputDimension)
        let gateBuffer = try makeSharedBuffer(values: gateValues)
        let upBuffer = try makeSharedBuffer(values: upValues)
        let weightBuffer = try makeSharedBuffer(values: weights)
        let outputBuffer = try makeZeroedSharedBuffer(
            byteLength: sequenceLength * shape.outputDimension * MemoryLayout<Float>.stride
        )
        let geometry = fusedDispatchGeometry(
            pipeline: pipeline,
            outputDimension: shape.outputDimension,
            sequenceLength: sequenceLength,
            simdgroupsPerThreadgroup: variant.simdgroupsPerThreadgroup,
            rowsPerSimdgroup: variant.rowsPerSimdgroup
        )

        for _ in 0..<warmupIterations {
            _ = try executeFusedSwigluDown(
                pipeline: pipeline,
                gateBuffer: gateBuffer,
                upBuffer: upBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                shape: shape,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        var totalMicroseconds = 0.0
        for _ in 0..<iterations {
            totalMicroseconds += try executeFusedSwigluDown(
                pipeline: pipeline,
                gateBuffer: gateBuffer,
                upBuffer: upBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                shape: shape,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        return FusedRowsResultRow(
            role: shape.role,
            inputDimension: shape.inputDimension,
            outputDimension: shape.outputDimension,
            sequenceLength: sequenceLength,
            variant: variant.name,
            rowsPerThreadgroup: variant.rowsPerThreadgroup,
            simdgroupsPerThreadgroup: variant.simdgroupsPerThreadgroup,
            rowsPerSimdgroup: variant.rowsPerSimdgroup,
            gridWidth: geometry.grid.width,
            gridHeight: geometry.grid.height,
            threadgroupWidth: geometry.threadgroup.width,
            averageGpuMicroseconds: totalMicroseconds / Double(iterations)
        )
    }

    private func execute(
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        shape: Shape,
        sequenceLength: Int,
        geometry: (grid: MTLSize, threadgroup: MTLSize)
    ) throws -> Double {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create microbenchmark command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inputDimension = UInt32(shape.inputDimension)
        var outputDimension = UInt32(shape.outputDimension)
        var sequenceLengthValue = UInt32(sequenceLength)
        var inputRowStride = UInt32(shape.inputDimension)
        var outputRowStride = UInt32(shape.outputDimension)
        encoder.setBytes(&inputDimension, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&outputDimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&sequenceLengthValue, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&outputRowStride, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(geometry.grid, threadsPerThreadgroup: geometry.threadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("Microbenchmark command failed: \(error.localizedDescription)")
        }
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000
    }

    private func executeBatched(
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffers: [MTLBuffer],
        outputBuffers: [MTLBuffer],
        shape: BatchedShape,
        sequenceLength: Int,
        geometry: (grid: MTLSize, threadgroup: MTLSize)
    ) throws -> Double {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create batched microbenchmark command buffer")
        }
        let count = shape.projectionCount
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        for (index, buffer) in weightBuffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: 1 + index)
        }
        for (index, buffer) in outputBuffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: 1 + count + index)
        }
        let dimBase = 1 + 2 * count
        var inputDimension = UInt32(shape.inputDimension)
        encoder.setBytes(&inputDimension, length: MemoryLayout<UInt32>.stride, index: dimBase)
        var outputDimensions = shape.outputDimensions.map(UInt32.init)
        for index in outputDimensions.indices {
            encoder.setBytes(&outputDimensions[index], length: MemoryLayout<UInt32>.stride, index: dimBase + 1 + index)
        }
        var sequenceLengthValue = UInt32(sequenceLength)
        var inputRowStride = UInt32(shape.inputDimension)
        var outputRowStride = UInt32(shape.maximumOutputDimension)
        encoder.setBytes(&sequenceLengthValue, length: MemoryLayout<UInt32>.stride, index: dimBase + 1 + count)
        encoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: dimBase + 2 + count)
        encoder.setBytes(&outputRowStride, length: MemoryLayout<UInt32>.stride, index: dimBase + 3 + count)
        encoder.dispatchThreadgroups(geometry.grid, threadsPerThreadgroup: geometry.threadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("Batched microbenchmark command failed: \(error.localizedDescription)")
        }
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000
    }

    private func executeFusedSwigluDown(
        pipeline: MTLComputePipelineState,
        gateBuffer: MTLBuffer,
        upBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        shape: Shape,
        sequenceLength: Int,
        geometry: (grid: MTLSize, threadgroup: MTLSize)
    ) throws -> Double {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create fused microbenchmark command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        var intermediateDimension = UInt32(shape.inputDimension)
        var outputDimension = UInt32(shape.outputDimension)
        var sequenceLengthValue = UInt32(sequenceLength)
        var inputRowStride = UInt32(shape.inputDimension)
        var outputRowStride = UInt32(shape.outputDimension)
        encoder.setBytes(&intermediateDimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&outputDimension, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&sequenceLengthValue, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&outputRowStride, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(geometry.grid, threadsPerThreadgroup: geometry.threadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("Fused microbenchmark command failed: \(error.localizedDescription)")
        }
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000
    }

    private func dispatchGeometry(
        pipeline: MTLComputePipelineState,
        outputDimension: Int,
        sequenceLength: Int,
        sequenceTile: Int,
        rowsPerSimdgroup: Int = 1
    ) -> (grid: MTLSize, threadgroup: MTLSize) {
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let simdgroupsPerThreadgroup = 2
        let threads = min(
            simdWidth * simdgroupsPerThreadgroup * sequenceTile,
            pipeline.maxTotalThreadsPerThreadgroup
        )
        let actualSimdgroupsPerThreadgroup = max(1, (threads / simdWidth) / sequenceTile)
        let actualRowsPerThreadgroup = actualSimdgroupsPerThreadgroup * rowsPerSimdgroup
        let grid = MTLSize(
            width: (outputDimension + actualRowsPerThreadgroup - 1) / actualRowsPerThreadgroup,
            height: (sequenceLength + sequenceTile - 1) / sequenceTile,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)
        return (grid, threadgroup)
    }

    private func fusedDispatchGeometry(
        pipeline: MTLComputePipelineState,
        outputDimension: Int,
        sequenceLength: Int,
        simdgroupsPerThreadgroup: Int,
        rowsPerSimdgroup: Int
    ) -> (grid: MTLSize, threadgroup: MTLSize) {
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let threads = min(simdWidth * simdgroupsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup)
        let actualSimdgroupsPerThreadgroup = max(1, threads / simdWidth)
        let rowsPerThreadgroup = actualSimdgroupsPerThreadgroup * rowsPerSimdgroup
        let grid = MTLSize(
            width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: sequenceLength,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)
        return (grid, threadgroup)
    }

    private func makeInputValues(count: Int) -> [Float] {
        (0..<count).map { index in
            Float((index * 17) % 61 - 30) * 0.0078125
        }
    }

    private func makeUpValues(count: Int) -> [Float] {
        (0..<count).map { index in
            Float((index * 19) % 71 - 35) * 0.00625
        }
    }

    private func makeWeights(count: Int) -> [BFloat16] {
        (0..<count).map { index in
            BFloat16(Float((index * 13) % 67 - 33) * 0.00390625)
        }
    }

    private func batchedKernelName(count: Int, sequenceTile: Int) -> String {
        if sequenceTile == 1 {
            return "bench_batched_gemv\(count)_seq_bf16_f32s"
        }
        return "bench_batched_gemv\(count)_seq_bf16_f32s_tile\(sequenceTile)"
    }

    private func makeSharedBuffer<T>(values: [T]) throws -> MTLBuffer {
        let byteLength = values.count * MemoryLayout<T>.stride
        guard let buffer = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate microbenchmark buffer")
        }
        try values.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else {
                throw MetalCompilerError.deviceSetupFailed("Cannot access microbenchmark source bytes")
            }
            buffer.contents().copyMemory(from: baseAddress, byteCount: byteLength)
        }
        return buffer
    }

    private func makeZeroedSharedBuffer(byteLength: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate zeroed microbenchmark buffer")
        }
        memset(buffer.contents(), 0, byteLength)
        return buffer
    }
}
