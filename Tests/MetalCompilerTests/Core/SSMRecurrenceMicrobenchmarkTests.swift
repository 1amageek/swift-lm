import Foundation
import Metal
import Testing
@testable import MetalCompiler

@Suite("SSM Recurrence Microbenchmark", .serialized)
struct SSMRecurrenceMicrobenchmarkTests {
    private static let sequenceLengths = [16, 64, 128]
    private static let iterations = 5
    private static let warmupIterations = 1

    @Test("BF16 SSM recurrence real-shape microbench")
    func bf16SSMRecurrenceRealShapeMicrobench() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try SSMRecurrenceMicrobenchmarkHarness(device: device)
        let variants = [
            SSMVariant(name: "base_tg128", kernelName: "bench_ssm_recurrence_seq_bf16_f32", threadgroupWidth: 128),
            SSMVariant(name: "base_tg256", kernelName: "bench_ssm_recurrence_seq_bf16_f32", threadgroupWidth: 256),
            SSMVariant(name: "base_tg384", kernelName: "bench_ssm_recurrence_seq_bf16_f32", threadgroupWidth: 384),
            SSMVariant(name: "shared_tg128", kernelName: "bench_ssm_recurrence_seq_bf16_f32_shared_rms", threadgroupWidth: 128),
            SSMVariant(name: "shared_tg256", kernelName: "bench_ssm_recurrence_seq_bf16_f32_shared_rms", threadgroupWidth: 256),
            SSMVariant(name: "shared_tg384", kernelName: "bench_ssm_recurrence_seq_bf16_f32_shared_rms", threadgroupWidth: 384),
            SSMVariant(name: "prewrite_tg128", kernelName: "bench_ssm_recurrence_seq_bf16_f32_prewrite_decay", threadgroupWidth: 128),
            SSMVariant(name: "prewrite_tg256", kernelName: "bench_ssm_recurrence_seq_bf16_f32_prewrite_decay", threadgroupWidth: 256),
            SSMVariant(name: "prewrite_tg384", kernelName: "bench_ssm_recurrence_seq_bf16_f32_prewrite_decay", threadgroupWidth: 384),
        ]

        var rows: [SSMResultRow] = []
        for sequenceLength in Self.sequenceLengths {
            for variant in variants {
                let row = try harness.measure(
                    variant: variant,
                    sequenceLength: sequenceLength,
                    iterations: Self.iterations,
                    warmupIterations: Self.warmupIterations
                )
                rows.append(row)
            }
        }

        let artifact = try writeCSV(rows: rows)
        printReport(rows: rows, artifact: artifact)
        #expect(rows.count == Self.sequenceLengths.count * variants.count)
    }

    private func printReport(rows: [SSMResultRow], artifact: URL) {
        print()
        print("=== BF16 SSM recurrence real-shape microbench ===")
        print("artifact: \(artifact.path)")
        print("seq  variant       avg_us  us/token  grid   tg")
        for row in rows.sorted(by: rowSort) {
            let variant = row.variant.padding(toLength: 13, withPad: " ", startingAt: 0)
            let grid = "\(row.gridWidth)x\(row.gridHeight)".padding(toLength: 6, withPad: " ", startingAt: 0)
            print("  \(String(format: "%3d", row.sequenceLength))  \(variant) \(String(format: "%7.1f", row.averageGpuMicroseconds))  \(String(format: "%8.3f", row.microsecondsPerToken))  \(grid) \(row.threadgroupWidth)")
        }
    }

    private func writeCSV(rows: [SSMResultRow]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/ssm-recurrence-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("qwen35-bf16-ssm-recurrence.csv")
        var lines = [
            [
                "sequenceLength",
                "variant",
                "headCount",
                "groupCount",
                "keyDimension",
                "valueDimension",
                "convKernelSize",
                "gridWidth",
                "gridHeight",
                "threadgroupWidth",
                "requestedThreadgroupWidth",
                "averageGpuMicroseconds",
                "microsecondsPerToken",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: rowSort) {
            lines.append([
                String(row.sequenceLength),
                row.variant,
                String(row.headCount),
                String(row.groupCount),
                String(row.keyDimension),
                String(row.valueDimension),
                String(row.convKernelSize),
                String(row.gridWidth),
                String(row.gridHeight),
                String(row.threadgroupWidth),
                String(row.requestedThreadgroupWidth),
                String(format: "%.3f", row.averageGpuMicroseconds),
                String(format: "%.6f", row.microsecondsPerToken),
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

    private func rowSort(_ lhs: SSMResultRow, _ rhs: SSMResultRow) -> Bool {
        if lhs.sequenceLength != rhs.sequenceLength {
            return lhs.sequenceLength < rhs.sequenceLength
        }
        if lhs.requestedThreadgroupWidth != rhs.requestedThreadgroupWidth {
            return lhs.requestedThreadgroupWidth < rhs.requestedThreadgroupWidth
        }
        return lhs.variant < rhs.variant
    }
}

private struct SSMVariant {
    let name: String
    let kernelName: String
    let threadgroupWidth: Int
}

private struct SSMResultRow {
    let sequenceLength: Int
    let variant: String
    let headCount: Int
    let groupCount: Int
    let keyDimension: Int
    let valueDimension: Int
    let convKernelSize: Int
    let gridWidth: Int
    let gridHeight: Int
    let threadgroupWidth: Int
    let requestedThreadgroupWidth: Int
    let averageGpuMicroseconds: Double

    var microsecondsPerToken: Double {
        averageGpuMicroseconds / Double(sequenceLength)
    }
}

private struct SSMRecurrenceMicrobenchmarkHarness {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipelines: [String: MTLComputePipelineState]

    private let headCount = 16
    private let groupCount = 16
    private let keyDimension = 128
    private let valueDimension = 128
    private let convKernelSize = 4

    private var keyGroupDimension: Int { groupCount * keyDimension }
    private var convDimension: Int { 2 * keyGroupDimension + headCount * valueDimension }
    private var outputDimension: Int { headCount * valueDimension }
    private var activationRowStride: Int { max(convDimension, outputDimension, headCount) }

    init(device: MTLDevice) throws {
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create SSM microbenchmark command queue")
        }
        self.device = device
        self.queue = queue

        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateSSMWeightIndependentHelpers(),
            MetalSourceGenerator.generateSSMConvSiluHelper(weightFormat: .bfloat16),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: "bench_ssm_recurrence_seq_bf16_f32",
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: 2 * 16 * 128 + 16 * 128,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: 16,
                groupCount: 16,
                keyHeadDimension: 128,
                valueHeadDimension: 128
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: "bench_ssm_recurrence_seq_bf16_f32_shared_rms",
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: 2 * 16 * 128 + 16 * 128,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: 16,
                groupCount: 16,
                keyHeadDimension: 128,
                valueHeadDimension: 128,
                shareRMSScale: true
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: "bench_ssm_recurrence_seq_bf16_f32_prewrite_decay",
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: 2 * 16 * 128 + 16 * 128,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: 16,
                groupCount: 16,
                keyHeadDimension: 128,
                valueHeadDimension: 128,
                prewriteDecayedState: true
            ),
        ].joined(separator: "\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let names = [
            "bench_ssm_recurrence_seq_bf16_f32",
            "bench_ssm_recurrence_seq_bf16_f32_shared_rms",
            "bench_ssm_recurrence_seq_bf16_f32_prewrite_decay",
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
        variant: SSMVariant,
        sequenceLength: Int,
        iterations: Int,
        warmupIterations: Int
    ) throws -> SSMResultRow {
        guard let pipeline = pipelines[variant.kernelName] else {
            throw MetalCompilerError.kernelNotFound(variant.kernelName)
        }
        let inputs = try makeInputs(sequenceLength: sequenceLength)
        let recurrentState = try makeZeroedSharedBuffer(
            byteLength: headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride
        )
        let convState = try makeZeroedSharedBuffer(
            byteLength: convKernelSize * convDimension * MemoryLayout<BFloat16>.stride
        )
        let output = try makeZeroedSharedBuffer(
            byteLength: sequenceLength * activationRowStride * MemoryLayout<Float>.stride
        )
        let geometry = dispatchGeometry(pipeline: pipeline, requestedThreadgroupWidth: variant.threadgroupWidth)

        for _ in 0..<warmupIterations {
            reset(recurrentState: recurrentState, convState: convState, output: output, sequenceLength: sequenceLength)
            _ = try execute(
                pipeline: pipeline,
                inputs: inputs,
                recurrentState: recurrentState,
                convState: convState,
                output: output,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        var totalMicroseconds = 0.0
        for _ in 0..<iterations {
            reset(recurrentState: recurrentState, convState: convState, output: output, sequenceLength: sequenceLength)
            totalMicroseconds += try execute(
                pipeline: pipeline,
                inputs: inputs,
                recurrentState: recurrentState,
                convState: convState,
                output: output,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        return SSMResultRow(
            sequenceLength: sequenceLength,
            variant: variant.name,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            gridWidth: geometry.grid.width,
            gridHeight: geometry.grid.height,
            threadgroupWidth: geometry.threadgroup.width,
            requestedThreadgroupWidth: variant.threadgroupWidth,
            averageGpuMicroseconds: totalMicroseconds / Double(iterations)
        )
    }

    private func makeInputs(sequenceLength: Int) throws -> SSMInputs {
        let qkv = try makeSharedBuffer(values: paddedRows(
            makeFloatValues(count: sequenceLength * convDimension, multiplier: 13, modulus: 23, scale: 0.125),
            rowCount: sequenceLength,
            logicalWidth: convDimension
        ))
        let z = try makeSharedBuffer(values: paddedRows(
            makeFloatValues(count: sequenceLength * outputDimension, multiplier: 17, modulus: 19, scale: 0.125),
            rowCount: sequenceLength,
            logicalWidth: outputDimension
        ))
        let beta = try makeSharedBuffer(values: paddedRows(
            makeFloatValues(count: sequenceLength * headCount, multiplier: 7, modulus: 11, scale: 0.125),
            rowCount: sequenceLength,
            logicalWidth: headCount
        ))
        let alpha = try makeSharedBuffer(values: paddedRows(
            makeFloatValues(count: sequenceLength * headCount, multiplier: 5, modulus: 13, scale: 0.125),
            rowCount: sequenceLength,
            logicalWidth: headCount
        ))
        let convWeight = try makeSharedBuffer(values: (0..<(convDimension * convKernelSize)).map { index in
            BFloat16(Float((index * 11) % 17 - 8) * 0.03125)
        })
        let normWeight = try makeSharedBuffer(values: (0..<valueDimension).map { index in
            0.75 + Float(index) * 0.0625
        })
        let dtBias = try makeSharedBuffer(values: (0..<headCount).map { index in
            BFloat16(Float(index - 1) * 0.03125)
        })
        let aLog = try makeSharedBuffer(values: (0..<headCount).map { index in
            Float(index) * 0.0625 - 0.125
        })
        return SSMInputs(
            qkv: qkv,
            z: z,
            beta: beta,
            alpha: alpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog
        )
    }

    private func execute(
        pipeline: MTLComputePipelineState,
        inputs: SSMInputs,
        recurrentState: MTLBuffer,
        convState: MTLBuffer,
        output: MTLBuffer,
        sequenceLength: Int,
        geometry: (grid: MTLSize, threadgroup: MTLSize)
    ) throws -> Double {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create SSM microbenchmark command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs.qkv, offset: 0, index: 0)
        encoder.setBuffer(inputs.z, offset: 0, index: 1)
        encoder.setBuffer(inputs.beta, offset: 0, index: 2)
        encoder.setBuffer(inputs.alpha, offset: 0, index: 3)
        encoder.setBuffer(inputs.convWeight, offset: 0, index: 4)
        encoder.setBuffer(inputs.normWeight, offset: 0, index: 5)
        encoder.setBuffer(inputs.dtBias, offset: 0, index: 6)
        encoder.setBuffer(inputs.aLog, offset: 0, index: 7)
        encoder.setBuffer(recurrentState, offset: 0, index: 8)
        encoder.setBuffer(convState, offset: 0, index: 9)
        encoder.setBuffer(output, offset: 0, index: 10)
        encoder.setBuffer(output, offset: 0, index: 18)
        setConstants(encoder: encoder, sequenceLength: sequenceLength)
        encoder.dispatchThreadgroups(geometry.grid, threadsPerThreadgroup: geometry.threadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("SSM microbenchmark command failed: \(error.localizedDescription)")
        }
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000
    }

    private func setConstants(encoder: MTLComputeCommandEncoder, sequenceLength: Int) {
        var heads = UInt32(headCount)
        var groups = UInt32(groupCount)
        var keyDim = UInt32(keyDimension)
        var valueDim = UInt32(valueDimension)
        var kernel = UInt32(convKernelSize)
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(activationRowStride)
        var debugEnabled: UInt32 = 0
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&groups, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 15)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 17)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 19)
        encoder.setBytes(&debugEnabled, length: MemoryLayout<UInt32>.stride, index: 20)
    }

    private func reset(recurrentState: MTLBuffer, convState: MTLBuffer, output: MTLBuffer, sequenceLength: Int) {
        memset(recurrentState.contents(), 0, recurrentState.length)
        memset(convState.contents(), 0, convState.length)
        memset(output.contents(), 0, sequenceLength * activationRowStride * MemoryLayout<Float>.stride)
    }

    private func dispatchGeometry(
        pipeline: MTLComputePipelineState,
        requestedThreadgroupWidth: Int
    ) -> (grid: MTLSize, threadgroup: MTLSize) {
        let safeGroupCount = max(groupCount, 1)
        let headsPerGroup = max(1, headCount / safeGroupCount)
        let localDimension = 2 * keyDimension + headsPerGroup * valueDimension
        let phase2Threads = headsPerGroup * min(valueDimension, 256)
        let desiredThreads = max(localDimension, phase2Threads)
        let defaultThreads = min(
            min(SSMRecurrenceFragment.maxThreadgroupSize, desiredThreads),
            pipeline.maxTotalThreadsPerThreadgroup
        )
        let threads = min(max(requestedThreadgroupWidth, 1), defaultThreads)
        return (
            MTLSize(width: safeGroupCount, height: 1, depth: 1),
            MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func paddedRows(_ values: [Float], rowCount: Int, logicalWidth: Int) -> [Float] {
        var padded = [Float](repeating: .zero, count: rowCount * activationRowStride)
        for row in 0..<rowCount {
            let sourceStart = row * logicalWidth
            let destinationStart = row * activationRowStride
            padded.replaceSubrange(
                destinationStart..<(destinationStart + logicalWidth),
                with: values[sourceStart..<(sourceStart + logicalWidth)]
            )
        }
        return padded
    }

    private func makeFloatValues(count: Int, multiplier: Int, modulus: Int, scale: Float) -> [Float] {
        (0..<count).map { index in
            Float(BFloat16(Float((index * multiplier) % modulus - modulus / 2) * scale))
        }
    }

    private func makeSharedBuffer<T>(values: [T]) throws -> MTLBuffer {
        guard !values.isEmpty else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create an empty SSM microbenchmark buffer")
        }
        var copy = values
        let byteLength = copy.count * MemoryLayout<T>.stride
        return try copy.withUnsafeMutableBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                throw MetalCompilerError.deviceSetupFailed("Cannot access SSM microbenchmark buffer bytes")
            }
            guard let buffer = device.makeBuffer(bytes: baseAddress, length: byteLength, options: .storageModeShared) else {
                throw MetalCompilerError.deviceSetupFailed("Cannot allocate SSM microbenchmark buffer")
            }
            return buffer
        }
    }

    private func makeZeroedSharedBuffer(byteLength: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate zeroed SSM microbenchmark buffer")
        }
        memset(buffer.contents(), 0, byteLength)
        return buffer
    }
}

private struct SSMInputs {
    let qkv: MTLBuffer
    let z: MTLBuffer
    let beta: MTLBuffer
    let alpha: MTLBuffer
    let convWeight: MTLBuffer
    let normWeight: MTLBuffer
    let dtBias: MTLBuffer
    let aLog: MTLBuffer
}
