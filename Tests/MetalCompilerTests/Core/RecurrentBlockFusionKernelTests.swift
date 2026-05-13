import Metal
import Testing
@testable import MetalCompiler

@Suite("Recurrent Block Fusion Kernels", .serialized)
struct RecurrentBlockFusionKernelTests {
    @Test("Partial reduce kernel matches CPU reference with padded strides")
    func partialReduceKernelMatchesCPUReferenceWithPaddedStrides() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let kernelName = "test_recurrent_block_partial_reduce_f32"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateRecurrentBlockPartialReduce(
                name: kernelName,
                bufferPrecision: .float32
            ),
        ].joined(separator: "\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: kernelName))
        let pipeline = try device.makeComputePipelineState(function: function)
        let commandQueue = try #require(device.makeCommandQueue())

        let groupCount = 4
        let outputDimension = 9
        let sequenceLength = 3
        let partialRowStride = 13
        let outputRowStride = 12

        var partial = [Float](repeating: -777.0, count: sequenceLength * groupCount * partialRowStride)
        for seq in 0..<sequenceLength {
            for group in 0..<groupCount {
                for row in 0..<outputDimension {
                    let index = seq * groupCount * partialRowStride + group * partialRowStride + row
                    partial[index] = Float((seq + 1) * 100 + group * 10 + row) * 0.03125
                }
            }
        }
        let output = [Float](repeating: -999.0, count: sequenceLength * outputRowStride)
        let expected = expectedPartialReduce(
            partial: partial,
            groupCount: groupCount,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength,
            partialRowStride: partialRowStride,
            outputRowStride: outputRowStride,
            sentinel: -999.0
        )

        let partialBuffer = try #require(device.makeBuffer(
            bytes: partial,
            length: partial.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            bytes: output,
            length: output.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))

        let commandBuffer = try #require(commandQueue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(partialBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes([UInt32(groupCount)], length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes([UInt32(outputDimension)], length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes([UInt32(sequenceLength)], length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes([UInt32(partialRowStride)], length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes([UInt32(outputRowStride)], length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: outputDimension, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        #expect(commandBuffer.error == nil)
        let pointer = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        let actual = (0..<output.count).map { pointer[$0] }
        let maxError = zip(expected, actual).reduce(Float.zero) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
        #expect(maxError == 0.0, "partial reduce maxError=\(maxError)")
        #expect(actual == expected)
    }

    private func expectedPartialReduce(
        partial: [Float],
        groupCount: Int,
        outputDimension: Int,
        sequenceLength: Int,
        partialRowStride: Int,
        outputRowStride: Int,
        sentinel: Float
    ) -> [Float] {
        var output = [Float](repeating: sentinel, count: sequenceLength * outputRowStride)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var sum: Float = 0
                for group in 0..<groupCount {
                    let index = seq * groupCount * partialRowStride + group * partialRowStride + row
                    sum += partial[index]
                }
                output[seq * outputRowStride + row] = sum
            }
        }
        return output
    }
}
