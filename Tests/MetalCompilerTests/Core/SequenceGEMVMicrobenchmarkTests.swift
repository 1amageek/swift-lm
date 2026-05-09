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
            Variant(name: "base", kernelName: "bench_gemv_seq_bf16_f32s", sequenceTile: 1),
            Variant(name: "tile2", kernelName: "bench_gemv_seq_bf16_f32s_tile2", sequenceTile: 2),
            Variant(name: "tile4", kernelName: "bench_gemv_seq_bf16_f32s_tile4", sequenceTile: 4),
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

        let artifact = try writeCSV(rows: rows)
        printReport(rows: rows, artifact: artifact)
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

    private func printReport(rows: [ResultRow], artifact: URL) {
        print()
        print("=== BF16 single sequence GEMV real-shape microbench ===")
        print("artifact: \(artifact.path)")
        print("role                    seq  variant  avg_us  us/output  grid      tg")
        for row in rows.sorted(by: rowSort) {
            let role = row.role.padding(toLength: 23, withPad: " ", startingAt: 0)
            let variant = row.variant.padding(toLength: 7, withPad: " ", startingAt: 0)
            let grid = "\(row.gridWidth)x\(row.gridHeight)".padding(toLength: 9, withPad: " ", startingAt: 0)
            print("  \(role) \(String(format: "%3d", row.sequenceLength))  \(variant) \(String(format: "%7.1f", row.averageGpuMicroseconds))  \(String(format: "%8.4f", row.microsecondsPerOutput))  \(grid) \(row.threadgroupWidth)")
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

    private func fusedRowsSort(_ lhs: FusedRowsResultRow, _ rhs: FusedRowsResultRow) -> Bool {
        if lhs.role != rhs.role { return lhs.role < rhs.role }
        if lhs.sequenceLength != rhs.sequenceLength { return lhs.sequenceLength < rhs.sequenceLength }
        if lhs.rowsPerThreadgroup != rhs.rowsPerThreadgroup {
            return lhs.rowsPerThreadgroup < rhs.rowsPerThreadgroup
        }
        return lhs.rowsPerSimdgroup < rhs.rowsPerSimdgroup
    }
}

private struct Shape {
    let role: String
    let inputDimension: Int
    let outputDimension: Int
}

private struct Variant {
    let name: String
    let kernelName: String
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
            MetalSourceGenerator.generateTiledSequenceGEMV(
                name: "bench_gemv_seq_bf16_f32s_tile4",
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
            "bench_gemv_seq_bf16_f32s_tile4",
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
            sequenceTile: variant.sequenceTile
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
        sequenceTile: Int
    ) -> (grid: MTLSize, threadgroup: MTLSize) {
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let rowsPerThreadgroup = 2
        let threads = min(
            simdWidth * rowsPerThreadgroup * sequenceTile,
            pipeline.maxTotalThreadsPerThreadgroup
        )
        let actualRowsPerThreadgroup = max(1, (threads / simdWidth) / sequenceTile)
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
