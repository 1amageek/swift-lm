import Metal
import Testing
@testable import MetalCompiler

@Suite("Recurrent Block Fusion Kernels", .serialized)
struct RecurrentBlockFusionKernelTests {
    @Test("Partial projection plus reduce matches CPU reference")
    func partialProjectionPlusReduceMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let projectionKernelName = "test_recurrent_block_partial_projection_f32_bf16"
        let reduceKernelName = "test_recurrent_block_partial_reduce_f32_pipeline"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateRecurrentBlockPartialProjection(
                name: projectionKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                tileElements: 4
            ),
            MetalSourceGenerator.generateRecurrentBlockPartialReduce(
                name: reduceKernelName,
                bufferPrecision: .float32
            ),
        ].joined(separator: "\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let projectionFunction = try #require(library.makeFunction(name: projectionKernelName))
        let reduceFunction = try #require(library.makeFunction(name: reduceKernelName))
        let projectionPipeline = try device.makeComputePipelineState(function: projectionFunction)
        let reducePipeline = try device.makeComputePipelineState(function: reduceFunction)
        let commandQueue = try #require(device.makeCommandQueue())

        let groupCount = 4
        let partitionInputDimension = 5
        let inputDimension = groupCount * partitionInputDimension
        let outputDimension = 9
        let sequenceLength = 3
        let inputRowStride = 23
        let partialRowStride = 13
        let outputRowStride = 12

        var input = [Float](repeating: -333.0, count: sequenceLength * inputRowStride)
        for seq in 0..<sequenceLength {
            for column in 0..<inputDimension {
                input[seq * inputRowStride + column] = Float((seq + 1) * 17 + column * 3 - 21) * 0.03125
            }
        }
        let weights = (0..<(outputDimension * inputDimension)).map { index in
            BFloat16(Float((index * 7) % 29 - 14) * 0.015625)
        }
        let partial = [Float](repeating: -777.0, count: groupCount * sequenceLength * partialRowStride)
        let output = [Float](repeating: -999.0, count: sequenceLength * outputRowStride)
        let expectedPartial = expectedPartialProjection(
            input: input,
            weights: weights,
            groupCount: groupCount,
            partitionInputDimension: partitionInputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            partialRowStride: partialRowStride,
            sentinel: -777.0
        )
        let expected = expectedPartialProjectionAndReduce(
            input: input,
            weights: weights,
            groupCount: groupCount,
            partitionInputDimension: partitionInputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            outputRowStride: outputRowStride,
            sentinel: -999.0
        )

        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<BFloat16>.stride,
            options: .storageModeShared
        ))
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
        let projectionEncoder = try #require(commandBuffer.makeComputeCommandEncoder())
        projectionEncoder.setComputePipelineState(projectionPipeline)
        projectionEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        projectionEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
        projectionEncoder.setBuffer(partialBuffer, offset: 0, index: 2)
        projectionEncoder.setBytes([UInt32(partitionInputDimension)], length: MemoryLayout<UInt32>.stride, index: 3)
        projectionEncoder.setBytes([UInt32(outputDimension)], length: MemoryLayout<UInt32>.stride, index: 4)
        projectionEncoder.setBytes([UInt32(groupCount)], length: MemoryLayout<UInt32>.stride, index: 5)
        projectionEncoder.setBytes([UInt32(sequenceLength)], length: MemoryLayout<UInt32>.stride, index: 6)
        projectionEncoder.setBytes([UInt32(inputRowStride)], length: MemoryLayout<UInt32>.stride, index: 7)
        projectionEncoder.setBytes([UInt32(partialRowStride)], length: MemoryLayout<UInt32>.stride, index: 8)
        let simdWidth = max(projectionPipeline.threadExecutionWidth, 1)
        let projectionRowsPerThreadgroup = 2
        projectionEncoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + projectionRowsPerThreadgroup - 1) / projectionRowsPerThreadgroup,
                height: sequenceLength,
                depth: groupCount
            ),
            threadsPerThreadgroup: MTLSize(width: simdWidth * projectionRowsPerThreadgroup, height: 1, depth: 1)
        )
        projectionEncoder.endEncoding()

        let reduceEncoder = try #require(commandBuffer.makeComputeCommandEncoder())
        reduceEncoder.setComputePipelineState(reducePipeline)
        reduceEncoder.setBuffer(partialBuffer, offset: 0, index: 0)
        reduceEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        reduceEncoder.setBytes([UInt32(groupCount)], length: MemoryLayout<UInt32>.stride, index: 2)
        reduceEncoder.setBytes([UInt32(outputDimension)], length: MemoryLayout<UInt32>.stride, index: 3)
        reduceEncoder.setBytes([UInt32(sequenceLength)], length: MemoryLayout<UInt32>.stride, index: 4)
        reduceEncoder.setBytes([UInt32(partialRowStride)], length: MemoryLayout<UInt32>.stride, index: 5)
        reduceEncoder.setBytes([UInt32(outputRowStride)], length: MemoryLayout<UInt32>.stride, index: 6)
        reduceEncoder.dispatchThreads(
            MTLSize(width: outputDimension, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
        )
        reduceEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        #expect(commandBuffer.error == nil)
        let partialPointer = partialBuffer.contents().assumingMemoryBound(to: Float.self)
        let actualPartial = (0..<partial.count).map { partialPointer[$0] }
        let partialMaxError = zip(expectedPartial, actualPartial).reduce(Float.zero) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
        #expect(partialMaxError <= 0.000_01, "partial projection buffer maxError=\(partialMaxError)")
        for group in 0..<groupCount {
            for seq in 0..<sequenceLength {
                for row in outputDimension..<partialRowStride {
                    let index = group * sequenceLength * partialRowStride + seq * partialRowStride + row
                    #expect(actualPartial[index] == -777.0)
                }
            }
        }

        let pointer = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        let actual = (0..<output.count).map { pointer[$0] }
        let maxError = zip(expected, actual).reduce(Float.zero) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
        #expect(maxError <= 0.000_01, "partial projection pipeline maxError=\(maxError)")
        for seq in 0..<sequenceLength {
            for row in outputDimension..<outputRowStride {
                #expect(actual[seq * outputRowStride + row] == -999.0)
            }
        }
    }

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

        var partial = [Float](repeating: -777.0, count: groupCount * sequenceLength * partialRowStride)
        for seq in 0..<sequenceLength {
            for group in 0..<groupCount {
                for row in 0..<outputDimension {
                    let index = group * sequenceLength * partialRowStride + seq * partialRowStride + row
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
                    let index = group * sequenceLength * partialRowStride + seq * partialRowStride + row
                    sum += partial[index]
                }
                output[seq * outputRowStride + row] = sum
            }
        }
        return output
    }

    private func expectedPartialProjection(
        input: [Float],
        weights: [BFloat16],
        groupCount: Int,
        partitionInputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        inputRowStride: Int,
        partialRowStride: Int,
        sentinel: Float
    ) -> [Float] {
        let inputDimension = groupCount * partitionInputDimension
        var partial = [Float](repeating: sentinel, count: groupCount * sequenceLength * partialRowStride)
        for group in 0..<groupCount {
            let groupInputBase = group * partitionInputDimension
            for seq in 0..<sequenceLength {
                for row in 0..<outputDimension {
                    var sum: Float = 0
                    for column in 0..<partitionInputDimension {
                        let inputValue = input[seq * inputRowStride + groupInputBase + column]
                        let weightValue = Float(weights[row * inputDimension + groupInputBase + column])
                        sum += inputValue * weightValue
                    }
                    partial[group * sequenceLength * partialRowStride + seq * partialRowStride + row] = sum
                }
            }
        }
        return partial
    }

    private func expectedPartialProjectionAndReduce(
        input: [Float],
        weights: [BFloat16],
        groupCount: Int,
        partitionInputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        inputRowStride: Int,
        outputRowStride: Int,
        sentinel: Float
    ) -> [Float] {
        let inputDimension = groupCount * partitionInputDimension
        var output = [Float](repeating: sentinel, count: sequenceLength * outputRowStride)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var total: Float = 0
                for group in 0..<groupCount {
                    var partial: Float = 0
                    let groupInputBase = group * partitionInputDimension
                    for column in 0..<partitionInputDimension {
                        let inputValue = input[seq * inputRowStride + groupInputBase + column]
                        let weightValue = Float(weights[row * inputDimension + groupInputBase + column])
                        partial += inputValue * weightValue
                    }
                    total += partial
                }
                output[seq * outputRowStride + row] = total
            }
        }
        return output
    }
}
