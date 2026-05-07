import Foundation
import Metal
import Testing
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
@Suite("Prefill GEMM Microbench", .serialized)
struct PrefillGEMMMicrobenchmarkTests {
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

    private func selectedTileSize(for sequenceLength: Int, tileSizes: [Int]) -> Int {
        for tileSize in tileSizes where tileSize >= sequenceLength {
            return tileSize
        }
        return tileSizes.last ?? MetalSourceGenerator.mppGEMMDefaultTileSize
    }

    private func kernelName(tileSize: Int) -> String {
        "prefill_gemm_microbench_mtile\(tileSize)"
    }
}
#endif
