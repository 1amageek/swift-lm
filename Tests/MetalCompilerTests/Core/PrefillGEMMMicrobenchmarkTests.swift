import Foundation
import Metal
import Testing
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
@Suite("Prefill GEMM Microbench", .serialized)
struct PrefillGEMMMicrobenchmarkTests {
    private struct CompactBridgeResult {
        let sequenceLength: Int
        let compactMPPMilliseconds: Double
        let packPlusMPPMilliseconds: Double
        let packOverheadPercent: Double
    }

    @Test("MPP GEMM tile variant smoke benchmark")
    func mppGEMMTileVariantSmokeBenchmark() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 2048
        let outputDimension = 8192
        let maximumSequenceLength = 128
        let sequenceLengths = [16, 32, 64, 96, 128]
        let iterations = 3
        let tileSizes = MetalSourceGenerator.mppGEMMTileSizes
        let pipelines = try makePipelines(device: device, tileSizes: tileSizes)

        let input = (0..<(inputDimension * maximumSequenceLength)).map {
            Float(($0 % 17) - 8) * 0.015625
        }
        let weight = (0..<(inputDimension * outputDimension)).map {
            Float16(Float(($0 % 23) - 11) * 0.0078125)
        }
        let outputByteCount = outputDimension * maximumSequenceLength * MemoryLayout<Float>.stride

        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            length: outputByteCount,
            options: .storageModeShared
        ))
        let queue = try #require(device.makeCommandQueue())

        print("")
        print("=== Prefill MPP GEMM tile variant smoke benchmark ===")
        print("shape: seq<=\(maximumSequenceLength), in=\(inputDimension), out=\(outputDimension), iterations=\(iterations)")
        print("seq  selected  selected_ms  base64_ms  delta")

        var resultCount = 0
        for sequenceLength in sequenceLengths {
            let selectedTile = selectedTileSize(for: sequenceLength, tileSizes: tileSizes)
            let selectedPipeline = try #require(pipelines[selectedTile])
            let basePipeline = try #require(pipelines[MetalSourceGenerator.mppGEMMDefaultTileSize])

            _ = try dispatchOnce(
                pipeline: selectedPipeline,
                tileSize: selectedTile,
                sequenceLength: sequenceLength,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                queue: queue
            )
            _ = try dispatchOnce(
                pipeline: basePipeline,
                tileSize: MetalSourceGenerator.mppGEMMDefaultTileSize,
                sequenceLength: sequenceLength,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                queue: queue
            )

            let selectedMS = try measureMedianMilliseconds(
                iterations: iterations,
                pipeline: selectedPipeline,
                tileSize: selectedTile,
                sequenceLength: sequenceLength,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                queue: queue
            )
            let baseMS = try measureMedianMilliseconds(
                iterations: iterations,
                pipeline: basePipeline,
                tileSize: MetalSourceGenerator.mppGEMMDefaultTileSize,
                sequenceLength: sequenceLength,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                queue: queue
            )
            let delta = baseMS > 0 ? (selectedMS - baseMS) / baseMS * 100 : 0
            print(String(
                format: "%3d  mtile%-3d  %10.3f  %9.3f  %+6.1f%%",
                sequenceLength,
                selectedTile,
                selectedMS,
                baseMS,
                delta
            ))
            resultCount += 1
        }

        #expect(resultCount == sequenceLengths.count)
    }

    @Test("MPP GEMM compact-input bridge smoke benchmark")
    func mppGEMMCompactInputBridgeSmokeBenchmark() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 3584
        let inputRowStride = 8192
        let outputDimension = 1024
        let maximumSequenceLength = 128
        let sequenceLengths = [64, 128]
        let iterations = 3
        let tileSize = 64
        let packKernelName = "prefill_pack_strided_f32_to_compact"
        let mppKernelName = "prefill_gemm_compact_bridge_mtile64"
        let pipelines = try makeCompactBridgePipelines(
            device: device,
            packKernelName: packKernelName,
            mppKernelName: mppKernelName,
            tileSize: tileSize
        )

        let stridedInput = (0..<(maximumSequenceLength * inputRowStride)).map {
            Float(($0 % 19) - 9) * 0.015625
        }
        let compactInput = (0..<(maximumSequenceLength * inputDimension)).map {
            Float(($0 % 19) - 9) * 0.015625
        }
        let weight = (0..<(inputDimension * outputDimension)).map {
            BFloat16(Float(($0 % 23) - 11) * 0.0078125)
        }
        let paddedSequenceLength = ((maximumSequenceLength + tileSize - 1) / tileSize) * tileSize
        let outputByteCount = outputDimension * paddedSequenceLength * MemoryLayout<Float>.stride

        let stridedInputBuffer = try #require(device.makeBuffer(
            bytes: stridedInput,
            length: stridedInput.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let compactInputBuffer = try #require(device.makeBuffer(
            bytes: compactInput,
            length: compactInput.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let bridgeCompactInputBuffer = try #require(device.makeBuffer(
            length: compactInput.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<BFloat16>.stride,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            length: outputByteCount,
            options: .storageModeShared
        ))
        let queue = try #require(device.makeCommandQueue())

        print("")
        print("=== Prefill MPP compact-input bridge smoke benchmark ===")
        print("shape: seq<=\(maximumSequenceLength), in=\(inputDimension), stride=\(inputRowStride), out=\(outputDimension), iterations=\(iterations)")
        print("seq  compact_mpp_ms  pack_plus_mpp_ms  pack_overhead")

        var rows: [CompactBridgeResult] = []
        var resultCount = 0
        for sequenceLength in sequenceLengths {
            _ = try dispatchOnce(
                pipeline: pipelines.mpp,
                tileSize: tileSize,
                sequenceLength: sequenceLength,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                inputBuffer: compactInputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                queue: queue
            )
            _ = try dispatchPackThenMPPOnce(
                packPipeline: pipelines.pack,
                mppPipeline: pipelines.mpp,
                tileSize: tileSize,
                sequenceLength: sequenceLength,
                inputDimension: inputDimension,
                inputRowStride: inputRowStride,
                outputDimension: outputDimension,
                stridedInputBuffer: stridedInputBuffer,
                compactInputBuffer: bridgeCompactInputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                queue: queue
            )

            let compactMS = try measureMedianMilliseconds(
                iterations: iterations,
                pipeline: pipelines.mpp,
                tileSize: tileSize,
                sequenceLength: sequenceLength,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                inputBuffer: compactInputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                queue: queue
            )
            let bridgeMS = try measureMedianBridgeMilliseconds(
                iterations: iterations,
                packPipeline: pipelines.pack,
                mppPipeline: pipelines.mpp,
                tileSize: tileSize,
                sequenceLength: sequenceLength,
                inputDimension: inputDimension,
                inputRowStride: inputRowStride,
                outputDimension: outputDimension,
                stridedInputBuffer: stridedInputBuffer,
                compactInputBuffer: bridgeCompactInputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                queue: queue
            )
            let overhead = compactMS > 0 ? (bridgeMS - compactMS) / compactMS * 100.0 : 0.0
            print(String(
                format: "%3d  %14.3f  %16.3f  %+6.1f%%",
                sequenceLength,
                compactMS,
                bridgeMS,
                overhead
            ))
            rows.append(CompactBridgeResult(
                sequenceLength: sequenceLength,
                compactMPPMilliseconds: compactMS,
                packPlusMPPMilliseconds: bridgeMS,
                packOverheadPercent: overhead
            ))
            resultCount += 1
        }

        let artifact = try writeCompactBridgeCSV(rows: rows)
        print("compact bridge artifact: \(artifact.path)")
        #expect(resultCount == sequenceLengths.count)
    }

    private func makePipelines(
        device: MTLDevice,
        tileSizes: [Int]
    ) throws -> [Int: MTLComputePipelineState] {
        let source = tileSizes.map { tileSize in
            MetalSourceGenerator.generateMPPGEMM(
                name: kernelName(tileSize: tileSize),
                bufferPrecision: .float32,
                weightFormat: .float16,
                mTile: tileSize
            )
        }.joined(separator: "\n\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)

        var pipelines: [Int: MTLComputePipelineState] = [:]
        for tileSize in tileSizes {
            let function = try #require(library.makeFunction(name: kernelName(tileSize: tileSize)))
            pipelines[tileSize] = try device.makeComputePipelineState(function: function)
        }
        return pipelines
    }

    private func makeCompactBridgePipelines(
        device: MTLDevice,
        packKernelName: String,
        mppKernelName: String,
        tileSize: Int
    ) throws -> (pack: MTLComputePipelineState, mpp: MTLComputePipelineState) {
        let source = [
            Self.generatePackStridedF32ToCompact(name: packKernelName),
            MetalSourceGenerator.generateMPPGEMM(
                name: mppKernelName,
                bufferPrecision: .float32,
                weightFormat: WeightFormats.bfloat16,
                mTile: tileSize
            ),
        ].joined(separator: "\n\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let packFunction = try #require(library.makeFunction(name: packKernelName))
        let mppFunction = try #require(library.makeFunction(name: mppKernelName))
        return (
            try device.makeComputePipelineState(function: packFunction),
            try device.makeComputePipelineState(function: mppFunction)
        )
    }

    private func measureMedianMilliseconds(
        iterations: Int,
        pipeline: MTLComputePipelineState,
        tileSize: Int,
        sequenceLength: Int,
        inputDimension: Int,
        outputDimension: Int,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        queue: MTLCommandQueue
    ) throws -> Double {
        var values: [Double] = []
        values.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let milliseconds = try dispatchOnce(
                pipeline: pipeline,
                tileSize: tileSize,
                sequenceLength: sequenceLength,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                queue: queue
            )
            values.append(milliseconds)
        }
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    private func measureMedianBridgeMilliseconds(
        iterations: Int,
        packPipeline: MTLComputePipelineState,
        mppPipeline: MTLComputePipelineState,
        tileSize: Int,
        sequenceLength: Int,
        inputDimension: Int,
        inputRowStride: Int,
        outputDimension: Int,
        stridedInputBuffer: MTLBuffer,
        compactInputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        queue: MTLCommandQueue
    ) throws -> Double {
        var values: [Double] = []
        values.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let milliseconds = try dispatchPackThenMPPOnce(
                packPipeline: packPipeline,
                mppPipeline: mppPipeline,
                tileSize: tileSize,
                sequenceLength: sequenceLength,
                inputDimension: inputDimension,
                inputRowStride: inputRowStride,
                outputDimension: outputDimension,
                stridedInputBuffer: stridedInputBuffer,
                compactInputBuffer: compactInputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                queue: queue
            )
            values.append(milliseconds)
        }
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    private func dispatchOnce(
        pipeline: MTLComputePipelineState,
        tileSize: Int,
        sequenceLength: Int,
        inputDimension: Int,
        outputDimension: Int,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        queue: MTLCommandQueue
    ) throws -> Double {
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(inputDimension)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + 31) / 32,
                height: (sequenceLength + tileSize - 1) / tileSize,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: min(pipeline.threadExecutionWidth * 4, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()

        let start = CFAbsoluteTimeGetCurrent()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let elapsedMS = (CFAbsoluteTimeGetCurrent() - start) * 1000
        if let error = commandBuffer.error {
            throw error
        }
        let gpuMS = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000
        return gpuMS > 0 ? gpuMS : elapsedMS
    }

    private func dispatchPackThenMPPOnce(
        packPipeline: MTLComputePipelineState,
        mppPipeline: MTLComputePipelineState,
        tileSize: Int,
        sequenceLength: Int,
        inputDimension: Int,
        inputRowStride: Int,
        outputDimension: Int,
        stridedInputBuffer: MTLBuffer,
        compactInputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        queue: MTLCommandQueue
    ) throws -> Double {
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let packEncoder = try #require(commandBuffer.makeComputeCommandEncoder())
        packEncoder.setComputePipelineState(packPipeline)
        packEncoder.setBuffer(stridedInputBuffer, offset: 0, index: 0)
        packEncoder.setBuffer(compactInputBuffer, offset: 0, index: 1)
        var inputDim = UInt32(inputDimension)
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(inputRowStride)
        packEncoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 2)
        packEncoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 3)
        packEncoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 4)
        packEncoder.dispatchThreadgroups(
            MTLSize(
                width: (inputDimension + 255) / 256,
                height: sequenceLength,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        packEncoder.endEncoding()

        let mppEncoder = try #require(commandBuffer.makeComputeCommandEncoder())
        mppEncoder.setComputePipelineState(mppPipeline)
        mppEncoder.setBuffer(compactInputBuffer, offset: 0, index: 0)
        mppEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
        mppEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var outDim = UInt32(outputDimension)
        var compactRowStride = UInt32(inputDimension)
        mppEncoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 3)
        mppEncoder.setBytes(&outDim, length: MemoryLayout<UInt32>.stride, index: 4)
        mppEncoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
        mppEncoder.setBytes(&compactRowStride, length: MemoryLayout<UInt32>.stride, index: 6)
        mppEncoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + 31) / 32,
                height: (sequenceLength + tileSize - 1) / tileSize,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: min(mppPipeline.threadExecutionWidth * 4, mppPipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1
            )
        )
        mppEncoder.endEncoding()

        let start = CFAbsoluteTimeGetCurrent()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let elapsedMS = (CFAbsoluteTimeGetCurrent() - start) * 1000
        if let error = commandBuffer.error {
            throw error
        }
        let gpuMS = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000
        return gpuMS > 0 ? gpuMS : elapsedMS
    }

    private func selectedTileSize(for sequenceLength: Int, tileSizes: [Int]) -> Int {
        for tileSize in tileSizes where tileSize >= sequenceLength {
            return tileSize
        }
        return tileSizes.last ?? MetalSourceGenerator.mppGEMMDefaultTileSize
    }

    private func kernelName(tileSize: Int) -> String {
        "prefill_gemm_microbench_mtile\(tileSize)"
    }

    private func writeCompactBridgeCSV(rows: [CompactBridgeResult]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/prefill-gemm-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("prefill-gemm-compact-input-bridge.csv")
        var lines = [
            [
                "sequenceLength",
                "compactMPPMilliseconds",
                "packPlusMPPMilliseconds",
                "packOverheadPercent",
            ].joined(separator: ","),
        ]
        for row in rows {
            lines.append([
                String(row.sequenceLength),
                String(format: "%.3f", row.compactMPPMilliseconds),
                String(format: "%.3f", row.packPlusMPPMilliseconds),
                String(format: "%.3f", row.packOverheadPercent),
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

    private static func generatePackStridedF32ToCompact(name: String) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void \(name)(
            device const float* input        [[buffer(0)]],
            device float* output             [[buffer(1)]],
            constant uint& inputDimension    [[buffer(2)]],
            constant uint& sequenceLength    [[buffer(3)]],
            constant uint& inputRowStride    [[buffer(4)]],
            uint2 gid                        [[threadgroup_position_in_grid]],
            uint tid                         [[thread_index_in_threadgroup]]
        ) {
            const uint col = gid.x * 256 + tid;
            const uint seq = gid.y;
            if (col >= inputDimension || seq >= sequenceLength) {
                return;
            }
            output[seq * inputDimension + col] = input[seq * inputRowStride + col];
        }
        """
    }
}
#endif
