import Metal
import Testing
@testable import MetalCompiler

@Suite("Sequence Projection Equivalence", .serialized)
struct SequenceProjectionEquivalenceTests {
    @Test("BF16 batched sequence GEMV matches decode GEMV with padded scratch slots")
    func bf16BatchedSequenceGEMVMatchesDecodeGEMVWithPaddedScratchSlots() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 1024
        let slotDimension = 6144
        let sequenceLength = 5
        let outputDimensions = [6144, 2048, 16, 16]
        let decodeKernelName = "batched_gemv4_bf16_padded_decode_equivalence"
        let sequenceKernelName = "batched_gemv4_bf16_padded_sequence_equivalence"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateBatchedGEMV4(
                name: decodeKernelName,
                bufferPrecision: BufferPrecision.bfloat16,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateBatchedSequenceGEMV(
                name: sequenceKernelName,
                count: outputDimensions.count,
                bufferPrecision: BufferPrecision.float32,
                weightFormat: WeightFormats.bfloat16
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let decodePipeline = try harness.pipeline(named: decodeKernelName)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)

        let packedInput = (0..<(sequenceLength * inputDimension)).map { index in
            Float(BFloat16(Float((index * 37) % 29 - 14) * 0.03125))
        }
        let paddedInput = paddedRows(
            packedInput,
            rowCount: sequenceLength,
            logicalWidth: inputDimension,
            rowStride: slotDimension
        )
        let weights = outputDimensions.enumerated().map { projection, outputDimension in
            (0..<(outputDimension * inputDimension)).map { index in
                BFloat16(Float((index * (projection + 3) + projection * 11) % 31 - 15) * 0.015625)
            }
        }

        let expected = try runDecodeProjectionTraceInScratch(
            harness: harness,
            pipeline: decodePipeline,
            inputValues: paddedInput,
            weights: weights,
            inputDimension: inputDimension,
            inputRowStride: slotDimension,
            slotDimension: slotDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions
        )
        let actual = try runSequenceProjectionTraceInScratch(
            harness: harness,
            pipeline: sequencePipeline,
            inputValues: paddedInput,
            weights: weights,
            inputDimension: inputDimension,
            inputRowStride: slotDimension,
            outputRowStride: slotDimension,
            slotDimension: slotDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions
        )

        for projection in outputDimensions.indices {
            let mismatch = harness.firstMismatch(
                expected: expected[projection],
                actual: actual[projection],
                tolerance: 0.000_001
            )
            #expect(
                mismatch == nil,
                "padded projection \(projection) drifted: \(String(describing: mismatch)), maxError=\(harness.maxAbsoluteError(expected: expected[projection], actual: actual[projection]))"
            )
        }
    }

    @Test("BF16 batched sequence GEMV matches repeated decode GEMV")
    func bf16BatchedSequenceGEMVMatchesRepeatedDecodeGEMV() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 64
        let sequenceLength = 5
        let outputDimensions = [17, 9, 13, 7]
        let decodeKernelName = "batched_gemv4_bf16_decode_equivalence"
        let sequenceKernelName = "batched_gemv4_bf16_sequence_equivalence"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateBatchedGEMV4(
                name: decodeKernelName,
                bufferPrecision: BufferPrecision.bfloat16,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateBatchedSequenceGEMV(
                name: sequenceKernelName,
                count: outputDimensions.count,
                bufferPrecision: BufferPrecision.float32,
                weightFormat: WeightFormats.bfloat16
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let decodePipeline = try harness.pipeline(named: decodeKernelName)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)

        let inputValues = (0..<(sequenceLength * inputDimension)).map { index in
            Float(BFloat16(Float((index * 37) % 29 - 14) * 0.03125))
        }
        let weights = outputDimensions.enumerated().map { projection, outputDimension in
            (0..<(outputDimension * inputDimension)).map { index in
                BFloat16(Float((index * (projection + 3) + projection * 11) % 31 - 15) * 0.015625)
            }
        }

        let expected = try runDecodeProjectionTrace(
            harness: harness,
            pipeline: decodePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions
        )
        let actual = try runSequenceProjectionTrace(
            harness: harness,
            pipeline: sequencePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions
        )

        for projection in outputDimensions.indices {
            let mismatch = harness.firstMismatch(
                expected: expected[projection],
                actual: actual[projection],
                tolerance: 0.000_001
            )
            #expect(
                mismatch == nil,
                "projection \(projection) drifted: \(String(describing: mismatch)), maxError=\(harness.maxAbsoluteError(expected: expected[projection], actual: actual[projection]))"
            )
        }
    }

    @Test("BF16 tiled batched sequence GEMV matches repeated decode GEMV")
    func bf16TiledBatchedSequenceGEMVMatchesRepeatedDecodeGEMV() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 64
        let sequenceLength = 7
        let sequenceTile = 4
        let outputDimensions = [17, 9, 13, 7]
        let decodeKernelName = "batched_gemv4_bf16_tiled_decode_equivalence"
        let sequenceKernelName = "batched_gemv4_bf16_tiled_sequence_equivalence"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateBatchedGEMV4(
                name: decodeKernelName,
                bufferPrecision: BufferPrecision.bfloat16,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateTiledBatchedSequenceGEMV(
                name: sequenceKernelName,
                count: outputDimensions.count,
                bufferPrecision: BufferPrecision.float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: sequenceTile
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let decodePipeline = try harness.pipeline(named: decodeKernelName)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)

        let inputValues = (0..<(sequenceLength * inputDimension)).map { index in
            Float(BFloat16(Float((index * 37) % 29 - 14) * 0.03125))
        }
        let weights = outputDimensions.enumerated().map { projection, outputDimension in
            (0..<(outputDimension * inputDimension)).map { index in
                BFloat16(Float((index * (projection + 3) + projection * 11) % 31 - 15) * 0.015625)
            }
        }

        let expected = try runDecodeProjectionTrace(
            harness: harness,
            pipeline: decodePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions
        )
        let actual = try runSequenceProjectionTrace(
            harness: harness,
            pipeline: sequencePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions,
            sequenceTile: sequenceTile
        )

        for projection in outputDimensions.indices {
            let mismatch = harness.firstMismatch(
                expected: expected[projection],
                actual: actual[projection],
                tolerance: 0.000_001
            )
            #expect(
                mismatch == nil,
                "tiled projection \(projection) drifted: \(String(describing: mismatch)), maxError=\(harness.maxAbsoluteError(expected: expected[projection], actual: actual[projection]))"
            )
        }
    }

    @Test("BF16 batched MPP GEMM matches batched sequence GEMV within MPP precision")
    func bf16BatchedMPPGEMMMatchesBatchedSequenceGEMVWithinMPPPrecision() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try assertBatchedMPPMatchesSequence(
            device: device,
            count: 2,
            inputDimension: 64,
            sequenceLength: 5,
            outputDimensions: [96, 96]
        )
        try assertBatchedMPPMatchesSequence(
            device: device,
            count: 3,
            inputDimension: 64,
            sequenceLength: 5,
            outputDimensions: [96, 96, 96]
        )
    }

    @Test("BF16 tiled single sequence GEMV matches repeated decode GEMV")
    func bf16TiledSingleSequenceGEMVMatchesRepeatedDecodeGEMV() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 73
        let outputDimension = 20
        let sequenceLength = 7
        let sequenceTile = 4
        let decodeKernelName = "gemv_bf16_single_decode_equivalence"
        let sequenceKernelName = "gemv_bf16_single_tiled_sequence_equivalence"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateGEMV(
                name: decodeKernelName,
                bufferPrecision: BufferPrecision.bfloat16,
                weightFormat: WeightFormats.bfloat16,
                tileElements: 256
            ),
            MetalSourceGenerator.generateTiledSequenceGEMV(
                name: sequenceKernelName,
                bufferPrecision: BufferPrecision.float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: sequenceTile
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let decodePipeline = try harness.pipeline(named: decodeKernelName)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)

        let inputValues = (0..<(sequenceLength * inputDimension)).map { index in
            Float(BFloat16(Float((index * 19) % 37 - 18) * 0.03125))
        }
        let weights = (0..<(outputDimension * inputDimension)).map { index in
            BFloat16(Float((index * 7) % 43 - 21) * 0.015625)
        }

        let expected = try runDecodeSingleProjectionTrace(
            harness: harness,
            pipeline: decodePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )
        let actual = try runSequenceSingleProjectionTrace(
            harness: harness,
            pipeline: sequencePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength,
            sequenceTile: sequenceTile
        )

        let mismatch = harness.firstMismatch(
            expected: expected,
            actual: actual,
            tolerance: 0.000_001
        )
        #expect(
            mismatch == nil,
            "single tiled projection drifted: \(String(describing: mismatch)), maxError=\(harness.maxAbsoluteError(expected: expected, actual: actual))"
        )
    }

    @Test("BF16 tile2 single sequence GEMV matches repeated decode GEMV")
    func bf16Tile2SingleSequenceGEMVMatchesRepeatedDecodeGEMV() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        // Use an odd sequence length to exercise the tail sequence path
        // (sequenceLength % sequenceTile != 0). Output dimension is kept
        // aligned to rowsPerThreadgroup so the existing single-projection
        // helper geometry stays valid for the row dimension.
        let inputDimension = 73
        let outputDimension = 20
        let sequenceLength = 9
        let sequenceTile = 2
        let decodeKernelName = "gemv_bf16_single_decode_equivalence_tile2"
        let sequenceKernelName = "gemv_bf16_single_tile2_sequence_equivalence"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateGEMV(
                name: decodeKernelName,
                bufferPrecision: BufferPrecision.bfloat16,
                weightFormat: WeightFormats.bfloat16,
                tileElements: 256
            ),
            MetalSourceGenerator.generateTiledSequenceGEMV(
                name: sequenceKernelName,
                bufferPrecision: BufferPrecision.float32,
                weightFormat: WeightFormats.bfloat16,
                sequenceTile: sequenceTile
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let decodePipeline = try harness.pipeline(named: decodeKernelName)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)

        let inputValues = (0..<(sequenceLength * inputDimension)).map { index in
            Float(BFloat16(Float((index * 19) % 37 - 18) * 0.03125))
        }
        let weights = (0..<(outputDimension * inputDimension)).map { index in
            BFloat16(Float((index * 7) % 43 - 21) * 0.015625)
        }

        let expected = try runDecodeSingleProjectionTrace(
            harness: harness,
            pipeline: decodePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )
        let actual = try runSequenceSingleProjectionTrace(
            harness: harness,
            pipeline: sequencePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength,
            sequenceTile: sequenceTile
        )

        let mismatch = harness.firstMismatch(
            expected: expected,
            actual: actual,
            tolerance: 0.000_001
        )
        #expect(
            mismatch == nil,
            "single tile2 projection drifted: \(String(describing: mismatch)), maxError=\(harness.maxAbsoluteError(expected: expected, actual: actual))"
        )
    }

    private func runDecodeProjectionTraceInScratch(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [[BFloat16]],
        inputDimension: Int,
        inputRowStride: Int,
        slotDimension: Int,
        sequenceLength: Int,
        outputDimensions: [Int],
        sequenceTile: Int = 1
    ) throws -> [[Float]] {
        let weightBuffers = try weights.map { try harness.makeSharedBuffer(values: $0) }
        var traces = outputDimensions.map { [Float](repeating: .zero, count: sequenceLength * $0) }
        let simdWidth = 32
        let threads = min(simdWidth * 2, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / simdWidth)
        let totalRows = outputDimensions.reduce(0, +)
        let grid = MTLSize(
            width: (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: 1,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        for position in 0..<sequenceLength {
            let inputStart = position * inputRowStride
            let tokenInput = inputValues[inputStart..<(inputStart + inputDimension)].map { BFloat16($0) }
            let scratch = try harness.makeZeroedSharedBuffer(
                byteLength: 5 * slotDimension * MemoryLayout<BFloat16>.stride
            )
            scratch.contents()
                .bindMemory(to: BFloat16.self, capacity: 5 * slotDimension)
                .update(from: Array(tokenInput), count: inputDimension)

            let (commandBuffer, encoder) = try harness.makeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(scratch, offset: 0, index: 0)
            for index in 0..<4 {
                encoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
                encoder.setBuffer(
                    scratch,
                    offset: (index + 1) * slotDimension * MemoryLayout<BFloat16>.stride,
                    index: 5 + index
                )
            }
            var inputDim = UInt32(inputDimension)
            var outputDim0 = UInt32(outputDimensions[0])
            var outputDim1 = UInt32(outputDimensions[1])
            var outputDim2 = UInt32(outputDimensions[2])
            var outputDim3 = UInt32(outputDimensions[3])
            encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 9)
            encoder.setBytes(&outputDim0, length: MemoryLayout<UInt32>.stride, index: 10)
            encoder.setBytes(&outputDim1, length: MemoryLayout<UInt32>.stride, index: 11)
            encoder.setBytes(&outputDim2, length: MemoryLayout<UInt32>.stride, index: 12)
            encoder.setBytes(&outputDim3, length: MemoryLayout<UInt32>.stride, index: 13)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
            encoder.endEncoding()
            try harness.complete(commandBuffer)

            let scratchValues = harness.readBFloat16AsFloat(
                scratch,
                count: 5 * slotDimension
            )
            for projection in outputDimensions.indices {
                let sourceOffset = (projection + 1) * slotDimension
                let destinationOffset = position * outputDimensions[projection]
                traces[projection].replaceSubrange(
                    destinationOffset..<(destinationOffset + outputDimensions[projection]),
                    with: scratchValues[sourceOffset..<(sourceOffset + outputDimensions[projection])]
                )
            }
        }
        return traces
    }

    private func runDecodeSingleProjectionTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [BFloat16],
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int
    ) throws -> [Float] {
        let weightBuffer = try harness.makeSharedBuffer(values: weights)
        var trace = [Float](repeating: .zero, count: sequenceLength * outputDimension)
        let simdWidth = 32
        let threads = min(simdWidth * 2, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / simdWidth)
        let grid = MTLSize(
            width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: 1,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        for position in 0..<sequenceLength {
            let inputStart = position * inputDimension
            let tokenInput = inputValues[inputStart..<(inputStart + inputDimension)].map { BFloat16($0) }
            let inputBuffer = try harness.makeSharedBuffer(values: Array(tokenInput))
            let outputBuffer = try harness.makeZeroedSharedBuffer(
                byteLength: outputDimension * MemoryLayout<BFloat16>.stride
            )

            let (commandBuffer, encoder) = try harness.makeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(weightBuffer, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
            var inputDim = UInt32(inputDimension)
            var outputDim = UInt32(outputDimension)
            encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&outputDim, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
            encoder.endEncoding()
            try harness.complete(commandBuffer)

            let values = harness.readBFloat16AsFloat(outputBuffer, count: outputDimension)
            let offset = position * outputDimension
            trace.replaceSubrange(offset..<(offset + outputDimension), with: values)
        }
        return trace
    }

    private func runSequenceSingleProjectionTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [BFloat16],
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        sequenceTile: Int
    ) throws -> [Float] {
        let inputBuffer = try harness.makeSharedBuffer(values: inputValues)
        let weightBuffer = try harness.makeSharedBuffer(values: weights)
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: sequenceLength * outputDimension * MemoryLayout<Float>.stride
        )
        let simdWidth = 32
        let threads = min(simdWidth * 2 * sequenceTile, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, (threads / simdWidth) / sequenceTile)
        let grid = MTLSize(
            width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: (sequenceLength + sequenceTile - 1) / sequenceTile,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inputDim = UInt32(inputDimension)
        var outputDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        var inputRowStride = UInt32(inputDimension)
        var outputRowStride = UInt32(outputDimension)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&outputDim, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&outputRowStride, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        return harness.readFloat32(outputBuffer, count: sequenceLength * outputDimension)
    }

    private func runSequenceProjectionTraceInScratch(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [[BFloat16]],
        inputDimension: Int,
        inputRowStride: Int,
        outputRowStride: Int,
        slotDimension: Int,
        sequenceLength: Int,
        outputDimensions: [Int],
        sequenceTile: Int = 1
    ) throws -> [[Float]] {
        let weightBuffers = try weights.map { try harness.makeSharedBuffer(values: $0) }
        let scratch = try harness.makeZeroedSharedBuffer(
            byteLength: 5 * sequenceLength * slotDimension * MemoryLayout<Float>.stride
        )
        scratch.contents()
            .bindMemory(to: Float.self, capacity: 5 * sequenceLength * slotDimension)
            .update(from: inputValues, count: inputValues.count)

        let simdWidth = 32
        let threads = min(simdWidth * 2 * sequenceTile, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, (threads / simdWidth) / sequenceTile)
        let totalRows = outputDimensions.reduce(0, +)
        let grid = MTLSize(
            width: (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: (sequenceLength + sequenceTile - 1) / sequenceTile,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(scratch, offset: 0, index: 0)
        for index in 0..<4 {
            encoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
            encoder.setBuffer(
                scratch,
                offset: (index + 1) * sequenceLength * slotDimension * MemoryLayout<Float>.stride,
                index: 5 + index
            )
        }
        var inputDim = UInt32(inputDimension)
        var outputDim0 = UInt32(outputDimensions[0])
        var outputDim1 = UInt32(outputDimensions[1])
        var outputDim2 = UInt32(outputDimensions[2])
        var outputDim3 = UInt32(outputDimensions[3])
        var seqLen = UInt32(sequenceLength)
        var inputStride = UInt32(inputRowStride)
        var outputStride = UInt32(outputRowStride)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&outputDim0, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&outputDim1, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&outputDim2, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&outputDim3, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&inputStride, length: MemoryLayout<UInt32>.stride, index: 15)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        let scratchValues = harness.readFloat32(
            scratch,
            count: 5 * sequenceLength * slotDimension
        )
        return outputDimensions.indices.map { projection in
            var packed: [Float] = []
            packed.reserveCapacity(sequenceLength * outputDimensions[projection])
            let slotOffset = (projection + 1) * sequenceLength * slotDimension
            for position in 0..<sequenceLength {
                let sourceOffset = slotOffset + position * outputRowStride
                packed.append(contentsOf: scratchValues[sourceOffset..<(sourceOffset + outputDimensions[projection])])
            }
            return packed
        }
    }

    private func assertBatchedMPPMatchesSequence(
        device: MTLDevice,
        count: Int,
        inputDimension: Int,
        sequenceLength: Int,
        outputDimensions: [Int]
    ) throws {
        let sequenceKernelName = "batched_gemv\(count)_bf16_sequence_reference"
        let mppKernelName = "batched_gemm_bf16_mpp_equivalence_\(count)"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateBatchedSequenceGEMV(
                name: sequenceKernelName,
                count: count,
                bufferPrecision: BufferPrecision.float32,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateBatchedMPPGEMM(
                name: mppKernelName,
                count: count,
                bufferPrecision: BufferPrecision.float32,
                weightFormat: WeightFormats.bfloat16,
                mTile: 16
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)
        let mppPipeline = try harness.pipeline(named: mppKernelName)

        let inputValues = (0..<(sequenceLength * inputDimension)).map { index in
            Float(BFloat16(Float((index * 37) % 29 - 14) * 0.03125))
        }
        let weights = outputDimensions.enumerated().map { projection, outputDimension in
            (0..<(outputDimension * inputDimension)).map { index in
                BFloat16(Float((index * (projection + 3) + projection * 11) % 31 - 15) * 0.015625)
            }
        }

        let expected = try runSequenceProjectionTrace(
            harness: harness,
            pipeline: sequencePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions
        )
        let actual = try runBatchedMPPProjectionTrace(
            harness: harness,
            pipeline: mppPipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions,
            mTile: 16
        )

        for projection in outputDimensions.indices {
            let mismatch = harness.firstMismatch(
                expected: expected[projection],
                actual: actual[projection],
                tolerance: 0.002
            )
            #expect(
                mismatch == nil,
                "batched MPP count \(count) projection \(projection) drifted: \(String(describing: mismatch)), maxError=\(harness.maxAbsoluteError(expected: expected[projection], actual: actual[projection]))"
            )
        }
    }

    private func runBatchedMPPProjectionTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [[BFloat16]],
        inputDimension: Int,
        sequenceLength: Int,
        outputDimensions: [Int],
        mTile: Int
    ) throws -> [[Float]] {
        let inputBuffer = try harness.makeSharedBuffer(values: inputValues)
        let weightBuffers = try weights.map { try harness.makeSharedBuffer(values: $0) }
        let paddedSequenceLength = ((sequenceLength + mTile - 1) / mTile) * mTile
        let outputBuffers = try outputDimensions.map { outputDimension in
            try harness.makeZeroedSharedBuffer(
                byteLength: paddedSequenceLength * outputDimension * MemoryLayout<Float>.stride
            )
        }
        let totalNTiles = outputDimensions.reduce(0) { $0 + (($1 + 31) / 32) }
        let grid = MTLSize(
            width: totalNTiles,
            height: (sequenceLength + mTile - 1) / mTile,
            depth: 1
        )
        let threadgroup = MTLSize(
            width: min(pipeline.threadExecutionWidth * 4, pipeline.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1
        )

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        for index in outputDimensions.indices {
            encoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
            encoder.setBuffer(outputBuffers[index], offset: 0, index: 1 + outputDimensions.count + index)
        }
        let dimBase = 1 + 2 * outputDimensions.count
        var inputDim = UInt32(inputDimension)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: dimBase)
        var outputDims = outputDimensions.map(UInt32.init)
        for index in outputDims.indices {
            encoder.setBytes(&outputDims[index], length: MemoryLayout<UInt32>.stride, index: dimBase + 1 + index)
        }
        var seqLen = UInt32(sequenceLength)
        var inputRowStride = UInt32(inputDimension)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: dimBase + 1 + outputDimensions.count)
        encoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: dimBase + 2 + outputDimensions.count)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        return outputDimensions.indices.map { projection in
            let padded = harness.readFloat32(
                outputBuffers[projection],
                count: paddedSequenceLength * outputDimensions[projection]
            )
            var packed: [Float] = []
            packed.reserveCapacity(sequenceLength * outputDimensions[projection])
            for position in 0..<sequenceLength {
                let start = position * outputDimensions[projection]
                packed.append(contentsOf: padded[start..<(start + outputDimensions[projection])])
            }
            return packed
        }
    }

    private func runDecodeProjectionTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [[BFloat16]],
        inputDimension: Int,
        sequenceLength: Int,
        outputDimensions: [Int],
        sequenceTile: Int = 1
    ) throws -> [[Float]] {
        let weightBuffers = try weights.map { try harness.makeSharedBuffer(values: $0) }
        var traces = outputDimensions.map { [Float](repeating: .zero, count: sequenceLength * $0) }
        let simdWidth = 32
        let threads = min(simdWidth * 2, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / simdWidth)
        let totalRows = outputDimensions.reduce(0, +)
        let grid = MTLSize(
            width: (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: 1,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        for position in 0..<sequenceLength {
            let inputStart = position * inputDimension
            let tokenInput = inputValues[inputStart..<(inputStart + inputDimension)].map { BFloat16($0) }
            let inputBuffer = try harness.makeSharedBuffer(values: Array(tokenInput))
            let outputBuffers = try outputDimensions.map {
                try harness.makeZeroedSharedBuffer(byteLength: $0 * MemoryLayout<BFloat16>.stride)
            }

            let (commandBuffer, encoder) = try harness.makeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            for index in 0..<4 {
                encoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
                encoder.setBuffer(outputBuffers[index], offset: 0, index: 5 + index)
            }
            var inputDim = UInt32(inputDimension)
            var outputDim0 = UInt32(outputDimensions[0])
            var outputDim1 = UInt32(outputDimensions[1])
            var outputDim2 = UInt32(outputDimensions[2])
            var outputDim3 = UInt32(outputDimensions[3])
            encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 9)
            encoder.setBytes(&outputDim0, length: MemoryLayout<UInt32>.stride, index: 10)
            encoder.setBytes(&outputDim1, length: MemoryLayout<UInt32>.stride, index: 11)
            encoder.setBytes(&outputDim2, length: MemoryLayout<UInt32>.stride, index: 12)
            encoder.setBytes(&outputDim3, length: MemoryLayout<UInt32>.stride, index: 13)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
            encoder.endEncoding()
            try harness.complete(commandBuffer)

            for projection in outputDimensions.indices {
                let values = harness.readBFloat16AsFloat(
                    outputBuffers[projection],
                    count: outputDimensions[projection]
                )
                let offset = position * outputDimensions[projection]
                traces[projection].replaceSubrange(offset..<(offset + outputDimensions[projection]), with: values)
            }
        }
        return traces
    }

    private func runSequenceProjectionTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [[BFloat16]],
        inputDimension: Int,
        sequenceLength: Int,
        outputDimensions: [Int],
        sequenceTile: Int = 1
    ) throws -> [[Float]] {
        let inputBuffer = try harness.makeSharedBuffer(values: inputValues)
        let weightBuffers = try weights.map { try harness.makeSharedBuffer(values: $0) }
        let outputRowStride = outputDimensions.max() ?? 0
        let outputBuffers = try outputDimensions.map { _ in
            try harness.makeZeroedSharedBuffer(byteLength: sequenceLength * outputRowStride * MemoryLayout<Float>.stride)
        }
        let simdWidth = 32
        let threads = min(simdWidth * 2 * sequenceTile, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, (threads / simdWidth) / sequenceTile)
        let totalRows = outputDimensions.reduce(0, +)
        let grid = MTLSize(
            width: (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: (sequenceLength + sequenceTile - 1) / sequenceTile,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        for index in outputDimensions.indices {
            encoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
            encoder.setBuffer(outputBuffers[index], offset: 0, index: 1 + outputDimensions.count + index)
        }
        let dimBase = 1 + 2 * outputDimensions.count
        var inputDim = UInt32(inputDimension)
        var seqLen = UInt32(sequenceLength)
        var inputRowStride = UInt32(inputDimension)
        var outputStride = UInt32(outputRowStride)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: dimBase)
        var outputDims = outputDimensions.map(UInt32.init)
        for index in outputDims.indices {
            encoder.setBytes(&outputDims[index], length: MemoryLayout<UInt32>.stride, index: dimBase + 1 + index)
        }
        let seqLenIndex = dimBase + 1 + outputDimensions.count
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: seqLenIndex)
        encoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: seqLenIndex + 1)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.stride, index: seqLenIndex + 2)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        return outputDimensions.indices.map { projection in
            let padded = harness.readFloat32(
                outputBuffers[projection],
                count: sequenceLength * outputRowStride
            )
            var packed: [Float] = []
            packed.reserveCapacity(sequenceLength * outputDimensions[projection])
            for position in 0..<sequenceLength {
                let start = position * outputRowStride
                packed.append(contentsOf: padded[start..<(start + outputDimensions[projection])])
            }
            return packed
        }
    }

    private func paddedRows(
        _ values: [Float],
        rowCount: Int,
        logicalWidth: Int,
        rowStride: Int
    ) -> [Float] {
        var padded = [Float](repeating: .zero, count: rowCount * rowStride)
        for row in 0..<rowCount {
            let sourceStart = row * logicalWidth
            let destinationStart = row * rowStride
            padded.replaceSubrange(
                destinationStart..<(destinationStart + logicalWidth),
                with: values[sourceStart..<(sourceStart + logicalWidth)]
            )
        }
        return padded
    }
}
