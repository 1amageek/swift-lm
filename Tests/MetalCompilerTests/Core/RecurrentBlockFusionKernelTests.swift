import Foundation
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

    @Test("Group-owned partial projection plus reduce matches CPU reference")
    func groupOwnedPartialProjectionPlusReduceMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let projectionKernelName = "test_recurrent_block_group_owned_partial_projection"
        let reduceKernelName = "test_recurrent_block_group_owned_partial_reduce"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateRecurrentBlockGroupOwnedPartialProjection(
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
                input[seq * inputRowStride + column] = Float((seq + 2) * 11 + column * 5 - 19) * 0.03125
            }
        }
        let weights = (0..<(outputDimension * inputDimension)).map { index in
            BFloat16(Float((index * 11) % 31 - 15) * 0.015625)
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
        let projectionSimdWidth = max(projectionPipeline.threadExecutionWidth, 1)
        let projectionSimdgroupsPerThreadgroup = 2
        projectionEncoder.dispatchThreadgroups(
            MTLSize(width: groupCount, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: projectionSimdWidth * projectionSimdgroupsPerThreadgroup,
                height: 1,
                depth: 1
            )
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
        #expect(partialMaxError <= 0.000_01, "group-owned partial projection maxError=\(partialMaxError)")
        for group in 0..<groupCount {
            for seq in 0..<sequenceLength {
                for row in outputDimension..<partialRowStride {
                    let index = group * sequenceLength * partialRowStride + seq * partialRowStride + row
                    #expect(actualPartial[index] == -777.0)
                }
            }
        }

        let outputPointer = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        let actual = (0..<output.count).map { outputPointer[$0] }
        let maxError = zip(expected, actual).reduce(Float.zero) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
        #expect(maxError <= 0.000_01, "group-owned partial pipeline maxError=\(maxError)")
        for seq in 0..<sequenceLength {
            for row in outputDimension..<outputRowStride {
                #expect(actual[seq * outputRowStride + row] == -999.0)
            }
        }
    }

    @Test("Row-grid fan-in projection matches CPU reference")
    func rowGridFanInProjectionMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let kernelName = "test_recurrent_block_row_grid_fan_in_projection"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateRecurrentBlockRowGridFanInProjection(
                name: kernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                tileElements: 4
            ),
        ].joined(separator: "\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: kernelName))
        let pipeline = try device.makeComputePipelineState(function: function)
        let commandQueue = try #require(device.makeCommandQueue())

        let groupCount = 4
        let partitionInputDimension = 5
        let inputDimension = groupCount * partitionInputDimension
        let outputDimension = 9
        let sequenceLength = 3
        let inputRowStride = 23
        let outputRowStride = 12

        var input = [Float](repeating: -333.0, count: sequenceLength * inputRowStride)
        for seq in 0..<sequenceLength {
            for column in 0..<inputDimension {
                input[seq * inputRowStride + column] = Float((seq + 3) * 13 + column * 7 - 23) * 0.03125
            }
        }
        let weights = (0..<(outputDimension * inputDimension)).map { index in
            BFloat16(Float((index * 13) % 37 - 18) * 0.015625)
        }
        let output = [Float](repeating: -999.0, count: sequenceLength * outputRowStride)
        let expected = expectedRowGridFanInProjection(
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
        let outputBuffer = try #require(device.makeBuffer(
            bytes: output,
            length: output.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))

        let commandBuffer = try #require(commandQueue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes([UInt32(partitionInputDimension)], length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes([UInt32(outputDimension)], length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes([UInt32(groupCount)], length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes([UInt32(sequenceLength)], length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes([UInt32(inputRowStride)], length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes([UInt32(outputRowStride)], length: MemoryLayout<UInt32>.stride, index: 8)
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let rowsPerThreadgroup = 2
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                height: sequenceLength,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: simdWidth * rowsPerThreadgroup, height: 1, depth: 1)
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
        #expect(maxError <= 0.000_01, "row-grid fan-in projection maxError=\(maxError)")
        for seq in 0..<sequenceLength {
            for row in outputDimension..<outputRowStride {
                #expect(actual[seq * outputRowStride + row] == -999.0)
            }
        }
    }

    @Test("Row-grid fan-in projection without inline rounding matches CPU reference")
    func rowGridFanInProjectionWithoutInlineRoundingMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let kernelName = "test_recurrent_block_row_grid_fan_in_projection_raw_input"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateRecurrentBlockRowGridFanInProjection(
                name: kernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                tileElements: 4,
                roundInputForDecodeEquivalentStorage: false
            ),
        ].joined(separator: "\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: kernelName))
        let pipeline = try device.makeComputePipelineState(function: function)
        let commandQueue = try #require(device.makeCommandQueue())

        let groupCount = 4
        let partitionInputDimension = 5
        let inputDimension = groupCount * partitionInputDimension
        let outputDimension = 9
        let sequenceLength = 3
        let inputRowStride = 23
        let outputRowStride = 12

        var input = [Float](repeating: -333.0, count: sequenceLength * inputRowStride)
        for seq in 0..<sequenceLength {
            for column in 0..<inputDimension {
                input[seq * inputRowStride + column] = Float((seq + 5) * 17 + column * 11 - 29) * 0.03125
            }
        }
        let weights = (0..<(outputDimension * inputDimension)).map { index in
            BFloat16(Float((index * 17) % 41 - 20) * 0.015625)
        }
        let output = [Float](repeating: -999.0, count: sequenceLength * outputRowStride)
        let expected = expectedRowGridFanInProjection(
            input: input,
            weights: weights,
            groupCount: groupCount,
            partitionInputDimension: partitionInputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            outputRowStride: outputRowStride,
            sentinel: -999.0,
            roundsInput: false
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
        let outputBuffer = try #require(device.makeBuffer(
            bytes: output,
            length: output.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))

        let commandBuffer = try #require(commandQueue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encodeRowGridFanInProjection(
            encoder: encoder,
            pipeline: pipeline,
            inputBuffer: inputBuffer,
            weightBuffer: weightBuffer,
            outputBuffer: outputBuffer,
            groupCount: groupCount,
            partitionInputDimension: partitionInputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            outputRowStride: outputRowStride
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        #expect(commandBuffer.error == nil)
        let pointer = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        let actual = (0..<output.count).map { pointer[$0] }
        let maxError = zip(expected, actual).reduce(Float.zero) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
        #expect(maxError <= 0.000_01, "row-grid raw-input fan-in projection maxError=\(maxError)")
        for seq in 0..<sequenceLength {
            for row in outputDimension..<outputRowStride {
                #expect(actual[seq * outputRowStride + row] == -999.0)
            }
        }
    }

    @Test("Row-grid fan-in inline rounding cost probe")
    func rowGridFanInInlineRoundingCostProbe() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let commandQueue = try #require(device.makeCommandQueue())
        let gemvKernelName = "bench_linear_attn_out_proj_gemv_seq_bf16_f32s"
        let roundedKernelName = "bench_recurrent_block_row_grid_fan_in_seq_bf16_f32_rounded"
        let rawInputKernelName = "bench_recurrent_block_row_grid_fan_in_seq_bf16_f32_raw_input"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateSequenceGEMV(
                name: gemvKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            ),
            MetalSourceGenerator.generateRecurrentBlockRowGridFanInProjection(
                name: roundedKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            ),
            MetalSourceGenerator.generateRecurrentBlockRowGridFanInProjection(
                name: rawInputKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                roundInputForDecodeEquivalentStorage: false
            ),
        ].joined(separator: "\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let gemvPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: gemvKernelName))
        )
        let roundedPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: roundedKernelName))
        )
        let rawInputPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: rawInputKernelName))
        )

        let groupCount = 4
        let partitionInputDimension = 64
        let inputDimension = groupCount * partitionInputDimension
        let outputDimension = 1024
        let inputRowStride = 2048
        let outputRowStride = 2048
        let sequenceLengths = [16, 64, 128]
        let iterations = 5
        let warmupIterations = 1
        let variants = [
            RowGridFanInVariant(name: "decodeRoundedInput", pipeline: roundedPipeline),
            RowGridFanInVariant(name: "rawInputDiagnostic", pipeline: rawInputPipeline),
        ]

        let weights = makeRowGridWeights(count: outputDimension * inputDimension)
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<BFloat16>.stride,
            options: .storageModeShared
        ))

        var rows: [RowGridFanInRoundingCostRow] = []
        for sequenceLength in sequenceLengths {
            let input = makeRowGridInputValues(count: sequenceLength * inputRowStride)
            let preRoundedInput = input.map { Float(BFloat16($0)) }
            let inputBuffer = try #require(device.makeBuffer(
                bytes: input,
                length: input.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ))
            let preRoundedInputBuffer = try #require(device.makeBuffer(
                bytes: preRoundedInput,
                length: preRoundedInput.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ))

            let gemvOutputByteLength = sequenceLength * outputRowStride * MemoryLayout<Float>.stride
            let gemvOutputBuffer = try #require(device.makeBuffer(
                length: gemvOutputByteLength,
                options: .storageModeShared
            ))
            for _ in 0..<warmupIterations {
                _ = try executeSequenceGEMVProjection(
                    commandQueue: commandQueue,
                    pipeline: gemvPipeline,
                    inputBuffer: preRoundedInputBuffer,
                    weightBuffer: weightBuffer,
                    outputBuffer: gemvOutputBuffer,
                    inputDimension: inputDimension,
                    outputDimension: outputDimension,
                    sequenceLength: sequenceLength,
                    inputRowStride: inputRowStride,
                    outputRowStride: outputRowStride
                )
            }

            var gemvTotalMicroseconds = 0.0
            for _ in 0..<iterations {
                gemvTotalMicroseconds += try executeSequenceGEMVProjection(
                    commandQueue: commandQueue,
                    pipeline: gemvPipeline,
                    inputBuffer: preRoundedInputBuffer,
                    weightBuffer: weightBuffer,
                    outputBuffer: gemvOutputBuffer,
                    inputDimension: inputDimension,
                    outputDimension: outputDimension,
                    sequenceLength: sequenceLength,
                    inputRowStride: inputRowStride,
                    outputRowStride: outputRowStride
                )
            }
            let gemvGeometry = sequenceGEMVGeometry(
                pipeline: gemvPipeline,
                outputDimension: outputDimension,
                sequenceLength: sequenceLength
            )
            rows.append(RowGridFanInRoundingCostRow(
                sequenceLength: sequenceLength,
                variant: "gemvSeqPreRoundedInput",
                groupCount: groupCount,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                inputRowStride: inputRowStride,
                outputRowStride: outputRowStride,
                gridWidth: gemvGeometry.grid.width,
                gridHeight: gemvGeometry.grid.height,
                threadgroupWidth: gemvGeometry.threadgroup.width,
                averageGpuMicroseconds: gemvTotalMicroseconds / Double(iterations)
            ))

            for variant in variants {
                let outputByteLength = sequenceLength * outputRowStride * MemoryLayout<Float>.stride
                let outputBuffer = try #require(device.makeBuffer(
                    length: outputByteLength,
                    options: .storageModeShared
                ))

                for _ in 0..<warmupIterations {
                    _ = try executeRowGridFanInProjection(
                        commandQueue: commandQueue,
                        pipeline: variant.pipeline,
                        inputBuffer: inputBuffer,
                        weightBuffer: weightBuffer,
                        outputBuffer: outputBuffer,
                        groupCount: groupCount,
                        partitionInputDimension: partitionInputDimension,
                        outputDimension: outputDimension,
                        sequenceLength: sequenceLength,
                        inputRowStride: inputRowStride,
                        outputRowStride: outputRowStride
                    )
                }

                var totalMicroseconds = 0.0
                for _ in 0..<iterations {
                    totalMicroseconds += try executeRowGridFanInProjection(
                        commandQueue: commandQueue,
                        pipeline: variant.pipeline,
                        inputBuffer: inputBuffer,
                        weightBuffer: weightBuffer,
                        outputBuffer: outputBuffer,
                        groupCount: groupCount,
                        partitionInputDimension: partitionInputDimension,
                        outputDimension: outputDimension,
                        sequenceLength: sequenceLength,
                        inputRowStride: inputRowStride,
                        outputRowStride: outputRowStride
                    )
                }

                let geometry = rowGridFanInGeometry(
                    pipeline: variant.pipeline,
                    outputDimension: outputDimension,
                    sequenceLength: sequenceLength
                )
                rows.append(RowGridFanInRoundingCostRow(
                    sequenceLength: sequenceLength,
                    variant: variant.name,
                    groupCount: groupCount,
                    inputDimension: inputDimension,
                    outputDimension: outputDimension,
                    inputRowStride: inputRowStride,
                    outputRowStride: outputRowStride,
                    gridWidth: geometry.grid.width,
                    gridHeight: geometry.grid.height,
                    threadgroupWidth: geometry.threadgroup.width,
                    averageGpuMicroseconds: totalMicroseconds / Double(iterations)
                ))
            }
        }

        let artifact = try writeRowGridFanInRoundingCostCSV(rows: rows)
        printRowGridFanInRoundingCostReport(rows: rows, artifact: artifact)
        #expect(rows.count == sequenceLengths.count * (variants.count + 1))
        #expect(rows.allSatisfy { $0.averageGpuMicroseconds.isFinite && $0.averageGpuMicroseconds > 0 })
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

    private func expectedRowGridFanInProjection(
        input: [Float],
        weights: [BFloat16],
        groupCount: Int,
        partitionInputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        inputRowStride: Int,
        outputRowStride: Int,
        sentinel: Float,
        roundsInput: Bool = true
    ) -> [Float] {
        let inputDimension = groupCount * partitionInputDimension
        var output = [Float](repeating: sentinel, count: sequenceLength * outputRowStride)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var total: Float = 0
                for group in 0..<groupCount {
                    let groupInputBase = group * partitionInputDimension
                    for column in 0..<partitionInputDimension {
                        let rawInputValue = input[seq * inputRowStride + groupInputBase + column]
                        let inputValue = roundsInput ? Float(BFloat16(rawInputValue)) : rawInputValue
                        let weightValue = Float(weights[row * inputDimension + groupInputBase + column])
                        total += inputValue * weightValue
                    }
                }
                output[seq * outputRowStride + row] = total
            }
        }
        return output
    }

    private func encodeRowGridFanInProjection(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        groupCount: Int,
        partitionInputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        inputRowStride: Int,
        outputRowStride: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes([UInt32(partitionInputDimension)], length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes([UInt32(outputDimension)], length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes([UInt32(groupCount)], length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes([UInt32(sequenceLength)], length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes([UInt32(inputRowStride)], length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes([UInt32(outputRowStride)], length: MemoryLayout<UInt32>.stride, index: 8)
        let geometry = rowGridFanInGeometry(
            pipeline: pipeline,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )
        encoder.dispatchThreadgroups(geometry.grid, threadsPerThreadgroup: geometry.threadgroup)
        encoder.endEncoding()
    }

    private func executeRowGridFanInProjection(
        commandQueue: MTLCommandQueue,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        groupCount: Int,
        partitionInputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        inputRowStride: Int,
        outputRowStride: Int
    ) throws -> Double {
        let commandBuffer = try #require(commandQueue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encodeRowGridFanInProjection(
            encoder: encoder,
            pipeline: pipeline,
            inputBuffer: inputBuffer,
            weightBuffer: weightBuffer,
            outputBuffer: outputBuffer,
            groupCount: groupCount,
            partitionInputDimension: partitionInputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            outputRowStride: outputRowStride
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("Row-grid fan-in command failed: \(error.localizedDescription)")
        }
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000
    }

    private func executeSequenceGEMVProjection(
        commandQueue: MTLCommandQueue,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        inputRowStride: Int,
        outputRowStride: Int
    ) throws -> Double {
        let commandBuffer = try #require(commandQueue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes([UInt32(inputDimension)], length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes([UInt32(outputDimension)], length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes([UInt32(sequenceLength)], length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes([UInt32(inputRowStride)], length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes([UInt32(outputRowStride)], length: MemoryLayout<UInt32>.stride, index: 7)
        let geometry = sequenceGEMVGeometry(
            pipeline: pipeline,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )
        encoder.dispatchThreadgroups(geometry.grid, threadsPerThreadgroup: geometry.threadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("Sequence GEMV command failed: \(error.localizedDescription)")
        }
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000
    }

    private func rowGridFanInGeometry(
        pipeline: MTLComputePipelineState,
        outputDimension: Int,
        sequenceLength: Int
    ) -> (grid: MTLSize, threadgroup: MTLSize) {
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let simdgroupsPerThreadgroup = 2
        let threads = min(simdWidth * simdgroupsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / simdWidth)
        return (
            grid: MTLSize(
                width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                height: sequenceLength,
                depth: 1
            ),
            threadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func sequenceGEMVGeometry(
        pipeline: MTLComputePipelineState,
        outputDimension: Int,
        sequenceLength: Int
    ) -> (grid: MTLSize, threadgroup: MTLSize) {
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let simdgroupsPerThreadgroup = 2
        let threads = min(simdWidth * simdgroupsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup)
        let actualSimdgroupsPerThreadgroup = max(1, threads / simdWidth)
        return (
            grid: MTLSize(
                width: (outputDimension + actualSimdgroupsPerThreadgroup - 1) / actualSimdgroupsPerThreadgroup,
                height: sequenceLength,
                depth: 1
            ),
            threadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func makeRowGridInputValues(count: Int) -> [Float] {
        (0..<count).map { index in
            Float((index * 17) % 257 - 128) * 0.0078125
        }
    }

    private func makeRowGridWeights(count: Int) -> [BFloat16] {
        (0..<count).map { index in
            BFloat16(Float((index * 13) % 251 - 125) * 0.00390625)
        }
    }

    private func writeRowGridFanInRoundingCostCSV(rows: [RowGridFanInRoundingCostRow]) throws -> URL {
        let directory = URL(fileURLWithPath: ".test-artifacts/recurrent-block-fusion", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let artifact = directory.appendingPathComponent("row-grid-fan-in-rounding-cost.csv")
        var csv = [
            [
                "sequenceLength",
                "variant",
                "groupCount",
                "inputDimension",
                "outputDimension",
                "inputRowStride",
                "outputRowStride",
                "gridWidth",
                "gridHeight",
                "threadgroupWidth",
                "averageGpuMicroseconds",
                "relativeToDecodeRounded",
                "relativeToGEMVPreRounded",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: rowGridFanInRoundingCostSort) {
            let rounded = rows.first {
                $0.sequenceLength == row.sequenceLength && $0.variant == "decodeRoundedInput"
            }
            let gemv = rows.first {
                $0.sequenceLength == row.sequenceLength && $0.variant == "gemvSeqPreRoundedInput"
            }
            let relativeToRounded = rounded.map { row.averageGpuMicroseconds / $0.averageGpuMicroseconds } ?? .nan
            let relativeToGEMV = gemv.map { row.averageGpuMicroseconds / $0.averageGpuMicroseconds } ?? .nan
            csv.append([
                "\(row.sequenceLength)",
                row.variant,
                "\(row.groupCount)",
                "\(row.inputDimension)",
                "\(row.outputDimension)",
                "\(row.inputRowStride)",
                "\(row.outputRowStride)",
                "\(row.gridWidth)",
                "\(row.gridHeight)",
                "\(row.threadgroupWidth)",
                String(format: "%.3f", row.averageGpuMicroseconds),
                String(format: "%.4f", relativeToRounded),
                String(format: "%.4f", relativeToGEMV),
            ].joined(separator: ","))
        }
        try csv.joined(separator: "\n").write(to: artifact, atomically: true, encoding: .utf8)
        return artifact
    }

    private func printRowGridFanInRoundingCostReport(rows: [RowGridFanInRoundingCostRow], artifact: URL) {
        print("row-grid fan-in inline rounding cost artifact: \(artifact.path)")
        for row in rows.sorted(by: rowGridFanInRoundingCostSort) {
            print(
                "seqLen=\(row.sequenceLength) variant=\(row.variant) avg_us="
                + String(format: "%.3f", row.averageGpuMicroseconds)
                + " grid=\(row.gridWidth)x\(row.gridHeight) tg=\(row.threadgroupWidth)"
            )
        }
    }

    private func rowGridFanInRoundingCostSort(
        lhs: RowGridFanInRoundingCostRow,
        rhs: RowGridFanInRoundingCostRow
    ) -> Bool {
        if lhs.sequenceLength != rhs.sequenceLength {
            return lhs.sequenceLength < rhs.sequenceLength
        }
        return lhs.variant < rhs.variant
    }

    private struct RowGridFanInVariant {
        let name: String
        let pipeline: MTLComputePipelineState
    }

    private struct RowGridFanInRoundingCostRow {
        let sequenceLength: Int
        let variant: String
        let groupCount: Int
        let inputDimension: Int
        let outputDimension: Int
        let inputRowStride: Int
        let outputRowStride: Int
        let gridWidth: Int
        let gridHeight: Int
        let threadgroupWidth: Int
        let averageGpuMicroseconds: Double
    }
}
