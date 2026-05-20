import Testing
import Metal
@testable import MetalCompiler

/// Verify MetalSourceGenerator produces valid, compilable MSL
/// that computes the same results as the hardcoded kernels.
@Suite("Metal Source Generator", .serialized)
struct MetalSourceGeneratorTests {

    @Test("Generated RMSNorm compiles for all precision × weight format combinations")
    func rmsNormCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let precisions: [(MetalSourceGenerator.BufferPrecision, String)] = [
            (.float16, "f16"),
            (.float32, "f32"),
        ]
        let weightFormats: [(MetalSourceGenerator.WeightFormat, String)] = [
            (.float16, "fp16"),
            (.bfloat16, "bf16"),
        ]

        for (precision, precisionLabel) in precisions {
            for (weightFormat, weightLabel) in weightFormats {
                let name = "rms_norm_\(precisionLabel)_\(weightLabel)"
                let source = MetalSourceGenerator.commonHeader + "\n\n"
                    + MetalSourceGenerator.generateReduction(
                        name: name, dimension: 2048, epsilon: 1e-5,
                        bufferPrecision: precision, weightFormat: weightFormat)

                let options = MTLCompileOptions()
                options.languageVersion = .version4_0
                let library = try device.makeLibrary(source: source, options: options)
                let function = library.makeFunction(name: name)
                #expect(function != nil, "Failed to compile \(name)")
            }
        }
    }

    @Test("Generated SwiGLU compiles for both precisions")
    func swigluCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
            let name = "swiglu_\(precision)"
            let source = MetalSourceGenerator.commonHeader + "\n\n"
                + MetalSourceGenerator.generateSwiGLU(name: name, bufferPrecision: precision)

            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: source, options: options)
            #expect(library.makeFunction(name: name) != nil, "Failed to compile \(name)")
        }
    }

    @Test("Generated GEMM compiles for all weight formats")
    func gemmCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let formats: [(MetalSourceGenerator.WeightFormat, String)] = [
            (.float16, "fp16"), (.bfloat16, "bf16")
        ]

        for (format, label) in formats {
            for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
                let name = "gemm_\(label)_\(precision)"
                let source = MetalSourceGenerator.commonHeader + "\n\n"
                    + MetalSourceGenerator.generateGEMM(
                        name: name, bufferPrecision: precision, weightFormat: format)

                let options = MTLCompileOptions()
                options.languageVersion = .version4_0
                let library = try device.makeLibrary(source: source, options: options)
                #expect(library.makeFunction(name: name) != nil, "Failed to compile \(name)")
            }
        }
    }

    @Test("MPP GEMM matches CPU reference for BF16 prefill projection")
    func mppGEMMMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 7
        let outputDimension = 5
        let sequenceLength = 4

        var input: [Float] = (0..<(inputDimension * sequenceLength)).map {
            Float(($0 % 11) - 5) * 0.25
        }
        var weight: [BFloat16] = (0..<(outputDimension * inputDimension)).map {
            BFloat16(Float((($0 * 3) % 13) - 6) * 0.125)
        }
        var output = [Float](repeating: .zero, count: outputDimension * sequenceLength)

        let inputBuffer = try #require(device.makeBuffer(
            bytes: &input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: &weight,
            length: weight.count * MemoryLayout<BFloat16>.size,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateMPPGEMM(
                name: "test_mpp_gemm_bf16_f32s",
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_mpp_gemm_bf16_f32s"))
        )

        let queue = try #require(device.makeCommandQueue())
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
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + 31) / 32,
                height: (sequenceLength + 63) / 64,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: min(pipeline.threadExecutionWidth * 4, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let result = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: output.count
        )

        var expected = [Float](repeating: .zero, count: output.count)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var sum: Float = 0
                for column in 0..<inputDimension {
                    sum += input[seq * inputDimension + column] * Float(weight[row * inputDimension + column])
                }
                expected[seq * outputDimension + row] = sum
            }
        }

        let actual = (0..<output.count).map { result[$0] }
        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(maxError < 0.01, "MPP GEMM drifted: maxError=\(maxError)")
    }

    @Test("Q3 dequant then MPP GEMM matches CPU reference for all group sizes")
    func q3DequantThenMPPGEMMMatchesCPUReferenceForAllGroupSizes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try runQ3DequantThenMPPGEMMReferenceTest(device: device, format: AffineQ3Group16Format())
        try runQ3DequantThenMPPGEMMReferenceTest(device: device, format: AffineQ3Group32Format())
        try runQ3DequantThenMPPGEMMReferenceTest(device: device, format: AffineQ3Group64Format())
    }

    private func runQ3DequantThenMPPGEMMReferenceTest(
        device: MTLDevice,
        format: any QuantizationFormat
    ) throws {
        let inputDimension = 128
        let outputDimension = 5
        let sequenceLength = 4
        let blocksPerRow = inputDimension / format.groupSize
        let suffix = q3KernelSuffix(for: format.schemeIdentifier)

        var packedWeights: [UInt8] = []
        var dequantizedWeights: [Float] = []
        for row in 0..<outputDimension {
            for block in 0..<blocksPerRow {
                let scale = 0.03125 * Float(block + 1) + 0.0078125 * Float(row)
                let zero = -0.125 + 0.0625 * Float(row) - 0.015625 * Float(block)
                let weights = (0..<format.groupSize).map {
                    UInt32(($0 + block * 3 + row * 5) % 8)
                }
                packedWeights.append(contentsOf: makeQuantizedBlock(
                    weights: weights,
                    bits: format.bits,
                    scale: scale,
                    zero: zero,
                    payloadByteCount: format.bytesPerBlock - 4
                ))
                dequantizedWeights.append(contentsOf: weights.map { scale * Float($0) + zero })
            }
        }

        var input: [Float] = (0..<(inputDimension * sequenceLength)).map {
            Float(($0 % 17) - 8) * 0.0625
        }
        var output = [Float](repeating: .zero, count: outputDimension * sequenceLength)

        let packedBuffer = try #require(device.makeBuffer(
            bytes: packedWeights,
            length: packedWeights.count,
            options: .storageModeShared
        ))
        let scratchBuffer = try #require(device.makeBuffer(
            length: outputDimension * inputDimension * MemoryLayout<UInt16>.stride,
            options: .storageModePrivate
        ))
        let inputBuffer = try #require(device.makeBuffer(
            bytes: &input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))

        let dequantName = "test_dequant_\(suffix)_bf16"
        let gemmName = "test_mpp_\(suffix)_bf16_f32s"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateUnifiedDequantToBFloat(
                name: dequantName,
                format: format
            ) + "\n\n"
            + MetalSourceGenerator.generateMPPGEMM(
                name: gemmName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let dequantPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: dequantName))
        )
        let gemmPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: gemmName))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(inputDimension)

        encoder.setComputePipelineState(dequantPipeline)
        encoder.setBuffer(packedBuffer, offset: 0, index: 0)
        encoder.setBuffer(scratchBuffer, offset: 0, index: 1)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: outputDimension, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.memoryBarrier(resources: [scratchBuffer])

        encoder.setComputePipelineState(gemmPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(scratchBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + 31) / 32,
                height: (sequenceLength + 63) / 64,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: min(gemmPipeline.threadExecutionWidth * 4, gemmPipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let actualPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: output.count)
        let actual = (0..<output.count).map { actualPointer[$0] }
        var expected = [Float](repeating: .zero, count: output.count)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var sum: Float = 0
                for column in 0..<inputDimension {
                    sum += input[seq * inputDimension + column]
                        * dequantizedWeights[row * inputDimension + column]
                }
                expected[seq * outputDimension + row] = sum
            }
        }

        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(maxError < 0.02, "\(suffix) dequant to MPP GEMM drifted: maxError=\(maxError)")
    }

    private func q3KernelSuffix(for scheme: QuantizationSchemeIdentifier) -> String {
        switch scheme {
        case .q3Group16ScaleF16:
            return "q3_g16"
        case .q3Group32ScaleF16:
            return "q3_g32"
        case .q3Group64ScaleF16:
            return "q3_g64"
        default:
            preconditionFailure("Unexpected Q3 scheme: \(scheme)")
        }
    }

    @Test("Q3 sequence GEMV matches decode-rounded CPU reference for all group sizes")
    func q3SequenceGEMVMatchesDecodeRoundedCPUReferenceForAllGroupSizes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try runQ3SequenceGEMVReferenceTest(device: device, format: AffineQ3Group16Format())
        try runQ3SequenceGEMVReferenceTest(device: device, format: AffineQ3Group32Format())
        try runQ3SequenceGEMVReferenceTest(device: device, format: AffineQ3Group64Format())
    }

    @Test("Q3 tiled sequence GEMV matches decode-rounded CPU reference for all group sizes")
    func q3TiledSequenceGEMVMatchesDecodeRoundedCPUReferenceForAllGroupSizes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try runQ3SequenceGEMVReferenceTest(device: device, format: AffineQ3Group16Format(), tiled: true)
        try runQ3SequenceGEMVReferenceTest(device: device, format: AffineQ3Group32Format(), tiled: true)
        try runQ3SequenceGEMVReferenceTest(device: device, format: AffineQ3Group64Format(), tiled: true)
    }

    @Test("Q3 batched sequence GEMV matches decode-rounded CPU reference for all counts and group sizes")
    func q3BatchedSequenceGEMVMatchesDecodeRoundedCPUReferenceForAllCountsAndGroupSizes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        for count in 2...4 {
            try runQ3BatchedSequenceGEMVReferenceTest(
                device: device,
                count: count,
                format: AffineQ3Group16Format()
            )
            try runQ3BatchedSequenceGEMVReferenceTest(
                device: device,
                count: count,
                format: AffineQ3Group32Format()
            )
            try runQ3BatchedSequenceGEMVReferenceTest(
                device: device,
                count: count,
                format: AffineQ3Group64Format()
            )
        }
    }

    private func runQ3SequenceGEMVReferenceTest(
        device: MTLDevice,
        format: any QuantizationFormat,
        tiled: Bool = false
    ) throws {
        let inputDimension = 128
        let outputDimension = 7
        let sequenceLength = 5
        let outputRowStride = 16
        let blocksPerRow = inputDimension / format.groupSize
        let suffix = q3KernelSuffix(for: format.schemeIdentifier)

        var packedWeights: [UInt8] = []
        var dequantizedWeights: [Float] = []
        for row in 0..<outputDimension {
            for block in 0..<blocksPerRow {
                let scale = 0.03125 * Float(block + 1) + 0.0078125 * Float(row)
                let zero = -0.125 + 0.0625 * Float(row) - 0.015625 * Float(block)
                let weights = (0..<format.groupSize).map {
                    UInt32(($0 + block * 3 + row * 5) % 8)
                }
                packedWeights.append(contentsOf: makeQuantizedBlock(
                    weights: weights,
                    bits: format.bits,
                    scale: scale,
                    zero: zero,
                    payloadByteCount: format.bytesPerBlock - 4
                ))
                dequantizedWeights.append(contentsOf: weights.map { scale * Float($0) + zero })
            }
        }

        var input: [Float] = (0..<(inputDimension * sequenceLength)).map {
            Float(($0 % 19) - 9) * 0.03125
        }
        var output = [Float](repeating: .nan, count: outputRowStride * sequenceLength)

        let inputBuffer = try #require(device.makeBuffer(
            bytes: &input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let packedBuffer = try #require(device.makeBuffer(
            bytes: packedWeights,
            length: packedWeights.count,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))

        let kernelName = tiled ? "test_gemv_seq_\(suffix)_f32s_tile4" : "test_gemv_seq_\(suffix)_f32s"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + (tiled ? MetalSourceGenerator.generateTiledQuantizedSequenceGEMV(
                name: kernelName,
                format: format,
                bufferPrecision: .float32,
                sequenceTile: 4
            ) : MetalSourceGenerator.generateUnifiedQuantizedSequenceGEMV(
                name: kernelName,
                format: format,
                bufferPrecision: .float32
            ))
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: kernelName))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        var inputRowStride = UInt32(inputDimension)
        var rowStride = UInt32(outputRowStride)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(packedBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + 1) / 2,
                height: tiled ? (sequenceLength + 3) / 4 : sequenceLength,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: min(
                    pipeline.threadExecutionWidth * (tiled ? 8 : 2),
                    pipeline.maxTotalThreadsPerThreadgroup
                ),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let actualPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: output.count)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var sum: Float = 0
                for column in 0..<inputDimension {
                    sum += input[seq * inputDimension + column]
                        * dequantizedWeights[row * inputDimension + column]
                }
                let expected = Float(Float16(sum))
                let actual = actualPointer[seq * outputRowStride + row]
                #expect(
                    abs(actual - expected) < 0.0001,
                    "\(suffix) \(tiled ? "tiled " : "")sequence GEMV mismatch seq=\(seq) row=\(row): actual=\(actual) expected=\(expected)"
                )
            }
        }
    }

    private func runQ3BatchedSequenceGEMVReferenceTest(
        device: MTLDevice,
        count: Int,
        format: any QuantizationFormat
    ) throws {
        let inputDimension = 128
        let sequenceLength = 4
        let outputRowStride = 16
        let outputDimensions = Array([3, 5, 7, 9].prefix(count))
        let blocksPerRow = inputDimension / format.groupSize
        let suffix = q3KernelSuffix(for: format.schemeIdentifier)

        var packedByProjection: [[UInt8]] = []
        var dequantizedByProjection: [[Float]] = []
        for projection in 0..<count {
            let outputDimension = outputDimensions[projection]
            var packedWeights: [UInt8] = []
            var dequantizedWeights: [Float] = []
            for row in 0..<outputDimension {
                for block in 0..<blocksPerRow {
                    let scale = 0.0234375 * Float(block + 1)
                        + 0.005859375 * Float(row + projection)
                    let zero = -0.1875
                        + 0.03125 * Float(row)
                        - 0.01171875 * Float(block + projection)
                    let weights = (0..<format.groupSize).map {
                        UInt32(($0 + block * 3 + row * 5 + projection * 7) % 8)
                    }
                    packedWeights.append(contentsOf: makeQuantizedBlock(
                        weights: weights,
                        bits: format.bits,
                        scale: scale,
                        zero: zero,
                        payloadByteCount: format.bytesPerBlock - 4
                    ))
                    dequantizedWeights.append(contentsOf: weights.map { scale * Float($0) + zero })
                }
            }
            packedByProjection.append(packedWeights)
            dequantizedByProjection.append(dequantizedWeights)
        }

        var input: [Float] = (0..<(inputDimension * sequenceLength)).map {
            Float(($0 % 23) - 11) * 0.02734375
        }
        let inputBuffer = try #require(device.makeBuffer(
            bytes: &input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        var packedBuffers: [MTLBuffer] = []
        for packed in packedByProjection {
            let buffer = try #require(device.makeBuffer(
                bytes: packed,
                length: packed.count,
                options: .storageModeShared
            ))
            packedBuffers.append(buffer)
        }
        var outputBuffers: [MTLBuffer] = []
        for _ in 0..<count {
            var output = [Float](repeating: .nan, count: outputRowStride * sequenceLength)
            let buffer = try #require(device.makeBuffer(
                bytes: &output,
                length: output.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ))
            outputBuffers.append(buffer)
        }

        let kernelName = "test_batched_gemv\(count)_seq_\(suffix)_f32s"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateBatchedQuantizedSequenceGEMV(
                name: kernelName,
                count: count,
                format: format,
                bufferPrecision: .float32
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: kernelName))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        for i in 0..<count {
            encoder.setBuffer(packedBuffers[i], offset: 0, index: 1 + i)
            encoder.setBuffer(outputBuffers[i], offset: 0, index: 1 + count + i)
        }
        var inDim = UInt32(inputDimension)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.stride, index: 1 + 2 * count)
        for i in 0..<count {
            var outDim = UInt32(outputDimensions[i])
            encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.stride, index: 2 + 2 * count + i)
        }
        var seqLen = UInt32(sequenceLength)
        var inputRowStride = UInt32(inputDimension)
        var rowStride = UInt32(outputRowStride)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 2 + 3 * count)
        encoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: 3 + 3 * count)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 4 + 3 * count)

        let totalRows = outputDimensions.reduce(0, +)
        encoder.dispatchThreadgroups(
            MTLSize(width: (totalRows + 1) / 2, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(pipeline.threadExecutionWidth * 2, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        for projection in 0..<count {
            let outputDimension = outputDimensions[projection]
            let actualPointer = outputBuffers[projection]
                .contents()
                .bindMemory(to: Float.self, capacity: outputRowStride * sequenceLength)
            let dequantizedWeights = dequantizedByProjection[projection]
            for seq in 0..<sequenceLength {
                for row in 0..<outputDimension {
                    var sum: Float = 0
                    for column in 0..<inputDimension {
                        sum += input[seq * inputDimension + column]
                            * dequantizedWeights[row * inputDimension + column]
                    }
                    let expected = Float(Float16(sum))
                    let actual = actualPointer[seq * outputRowStride + row]
                    #expect(
                        abs(actual - expected) < 0.0001,
                        "\(suffix) batched sequence GEMV mismatch count=\(count) projection=\(projection) seq=\(seq) row=\(row): actual=\(actual) expected=\(expected)"
                    )
                }
            }
        }
    }

    @Test("MPP GEMM matches CPU reference for FP16 prefill projection")
    func mppFP16GEMMMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 7
        let outputDimension = 5
        let sequenceLength = 4

        var input: [Float] = (0..<(inputDimension * sequenceLength)).map {
            Float(($0 % 11) - 5) * 0.25
        }
        var weight: [Float16] = (0..<(outputDimension * inputDimension)).map {
            Float16(Float((($0 * 3) % 13) - 6) * 0.125)
        }
        var output = [Float](repeating: .zero, count: outputDimension * sequenceLength)

        let inputBuffer = try #require(device.makeBuffer(
            bytes: &input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: &weight,
            length: weight.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateMPPGEMM(
                name: "test_mpp_gemm_f32s",
                bufferPrecision: .float32,
                weightFormat: .float16
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_mpp_gemm_f32s"))
        )

        let queue = try #require(device.makeCommandQueue())
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
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + 31) / 32,
                height: (sequenceLength + 63) / 64,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: min(pipeline.threadExecutionWidth * 4, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let result = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: output.count
        )

        var expected = [Float](repeating: .zero, count: output.count)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var sum: Float = 0
                for column in 0..<inputDimension {
                    sum += input[seq * inputDimension + column] * Float(weight[row * inputDimension + column])
                }
                expected[seq * outputDimension + row] = sum
            }
        }

        let actual = (0..<output.count).map { result[$0] }
        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(maxError < 0.01, "MPP FP16 GEMM drifted: maxError=\(maxError)")
    }

    @Test("Quantized Q4 GEMM matches CPU reference with padded scratch input and output stride")
    func quantizedQ4GEMMMatchesCPUReferenceWithPaddedScratchInputAndOutputStride() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 64
        let outputDimension = 4
        let sequenceLength = 4
        let inputRowStride = 256
        let outputRowStride = 8

        var input = [Float](repeating: 9_999, count: sequenceLength * inputRowStride)
        for seq in 0..<sequenceLength {
            for column in 0..<inputDimension {
                input[seq * inputRowStride + column] = Float(((seq + 3) * ((column % 7) - 3))) * 0.125
            }
        }

        var weightBytes: [UInt8] = []
        weightBytes.reserveCapacity(outputDimension * 36)
        func appendBytes<T>(_ value: T) {
            withUnsafeBytes(of: value) { weightBytes.append(contentsOf: $0) }
        }
        for row in 0..<outputDimension {
            let scale = Float16(0.25)
            let zero = Float16(0)
            appendBytes(scale)
            appendBytes(zero)
            let nibble = UInt8(row + 1)
            let packed = nibble | (nibble << 4)
            weightBytes.append(contentsOf: repeatElement(packed, count: inputDimension / 2))
        }

        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weightBytes,
            length: weightBytes.count,
            options: .storageModeShared
        ))
        var output = [Float](repeating: .nan, count: outputRowStride * sequenceLength)
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateQuantizedGEMM_Q4(
                name: "test_quantized_gemm_q4_g64_f32s",
                bufferPrecision: .float32,
                groupSize: 64
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_quantized_gemm_q4_g64_f32s"))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        var inputStride = UInt32(inputRowStride)
        var outputStride = UInt32(outputRowStride)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&inputStride, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.size, index: 7)
        let simdWidth = pipeline.threadExecutionWidth
        let threads = min(2 * simdWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (outputDimension + 1) / 2, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actualPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: output.count
        )
        var actual: [Float] = []
        actual.reserveCapacity(outputDimension * sequenceLength)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                actual.append(actualPointer[seq * outputRowStride + row])
            }
            for row in outputDimension..<outputRowStride {
                #expect(actualPointer[seq * outputRowStride + row].isNaN)
            }
        }

        var expected = [Float](repeating: .zero, count: outputDimension * sequenceLength)
        for seq in 0..<sequenceLength {
            let inputSum = (0..<inputDimension).reduce(Float.zero) { partial, column in
                partial + input[seq * inputRowStride + column]
            }
            for row in 0..<outputDimension {
                expected[seq * outputDimension + row] = inputSum * (0.25 * Float(row + 1))
            }
        }

        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(
            maxError < 0.001,
            """
            Quantized Q4 GEMM drifted with padded scratch input/output stride
            maxError=\(maxError)
            actualPrefix=\(actual.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            expectedPrefix=\(expected.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("Decode GEMV matches CPU reference with odd output tail")
    func decodeGEMVMatchesCPUReferenceWithOddOutputTail() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 65
        let outputDimension = 5
        let outputCapacity = 8
        let input = makeDecodeFloatInput(inputDimension: inputDimension)
        let weights = makeDecodeFloat16Weights(
            outputDimension: outputDimension,
            inputDimension: inputDimension,
            projectionSeed: 3
        )
        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ))
        var output = [Float](repeating: .nan, count: outputCapacity)
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateGEMV(
                name: "test_decode_gemv_odd_tail",
                bufferPrecision: .float32,
                weightFormat: .float16,
                tileElements: 128
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_decode_gemv_odd_tail"))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inputDim = UInt32(inputDimension)
        var outputDim = UInt32(outputDimension)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outputDim, length: MemoryLayout<UInt32>.size, index: 4)
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / pipeline.threadExecutionWidth)
        encoder.dispatchThreadgroups(
            MTLSize(width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actualPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: output.count
        )
        let expected = decodeCPUReference(
            input: input,
            weights: weights,
            inputDimension: inputDimension,
            outputDimension: outputDimension
        )
        var actual: [Float] = []
        actual.reserveCapacity(outputDimension)
        for row in 0..<outputDimension {
            actual.append(actualPointer[row])
        }
        for row in outputDimension..<outputCapacity {
            #expect(actualPointer[row].isNaN)
        }

        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(
            maxError < 0.001,
            "Decode GEMV odd-tail drifted: maxError=\(maxError)"
        )
    }

    @Test("Batched decode GEMV matches CPU reference with odd total row tail")
    func batchedDecodeGEMVMatchesCPUReferenceWithOddTotalRowTail() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try assertBatchedDecodeGEMVOddTail(
            device: device,
            count: 2,
            outputDimensions: [3, 4]
        )
        try assertBatchedDecodeGEMVOddTail(
            device: device,
            count: 3,
            outputDimensions: [2, 2, 3]
        )
        try assertBatchedDecodeGEMVOddTail(
            device: device,
            count: 4,
            outputDimensions: [1, 2, 1, 3]
        )
    }

    @Test("Decode GEMV argument table matches CPU reference with odd output tail")
    func decodeGEMVArgumentTableMatchesCPUReferenceWithOddOutputTail() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let argumentBufferIndex = 30
        let inputDimension = 65
        let outputDimension = 5
        let outputCapacity = 8
        let input = makeDecodeFloatInput(inputDimension: inputDimension)
        let weights = makeDecodeFloat16Weights(
            outputDimension: outputDimension,
            inputDimension: inputDimension,
            projectionSeed: 3
        )
        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ))
        var output = [Float](repeating: .nan, count: outputCapacity)
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateGEMVArgumentTableVariant(
                name: "test_decode_gemv_argbuf_odd_tail",
                argumentBufferIndex: argumentBufferIndex,
                bufferPrecision: .float32,
                weightFormat: .float16,
                tileElements: 128
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: "test_decode_gemv_argbuf_odd_tail"))
        let pipeline = try device.makeComputePipelineState(function: function)
        let argumentEncoder = function.makeArgumentEncoder(bufferIndex: argumentBufferIndex)
        let argumentBuffer = try #require(device.makeBuffer(
            length: argumentEncoder.encodedLength,
            options: .storageModeShared
        ))
        argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
        argumentEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        argumentEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
        argumentEncoder.setBuffer(outputBuffer, offset: 0, index: 2)

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(argumentBuffer, offset: 0, index: argumentBufferIndex)
        encoder.useResource(argumentBuffer, usage: .read)
        encoder.useResource(inputBuffer, usage: .read)
        encoder.useResource(weightBuffer, usage: .read)
        encoder.useResource(outputBuffer, usage: .write)
        var inputDim = UInt32(inputDimension)
        var outputDim = UInt32(outputDimension)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outputDim, length: MemoryLayout<UInt32>.size, index: 4)
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / pipeline.threadExecutionWidth)
        encoder.dispatchThreadgroups(
            MTLSize(width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        assertDecodeOutput(
            outputBuffer: outputBuffer,
            outputCapacity: outputCapacity,
            input: input,
            weights: weights,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            label: "Decode GEMV argument table odd-tail"
        )
    }

    @Test("Batched decode GEMV argument table matches CPU reference with odd total row tail")
    func batchedDecodeGEMVArgumentTableMatchesCPUReferenceWithOddTotalRowTail() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try assertBatchedDecodeGEMVOddTail(
            device: device,
            count: 2,
            outputDimensions: [3, 4],
            usesArgumentTable: true
        )
        try assertBatchedDecodeGEMVOddTail(
            device: device,
            count: 3,
            outputDimensions: [2, 2, 3],
            usesArgumentTable: true
        )
        try assertBatchedDecodeGEMVOddTail(
            device: device,
            count: 4,
            outputDimensions: [1, 2, 1, 3],
            usesArgumentTable: true
        )
    }

    @Test("Specialized decode GEMV matches CPU reference with odd output tail")
    func specializedDecodeGEMVMatchesCPUReferenceWithOddOutputTail() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try assertSingleDecodeGEMVOddTail(
            device: device,
            name: "test_input2048_decode_gemv_odd_tail",
            inputDimension: 2_048,
            source: MetalSourceGenerator.generateInput2048GEMV(
                name: "test_input2048_decode_gemv_odd_tail",
                bufferPrecision: .float32,
                weightFormat: .float16,
                fixedOutputDimension: 5
            )
        )
        try assertSingleDecodeGEMVOddTail(
            device: device,
            name: "test_input8192_decode_gemv_odd_tail",
            inputDimension: 8_192,
            source: MetalSourceGenerator.generateInput8192TiledGEMV(
                name: "test_input8192_decode_gemv_odd_tail",
                bufferPrecision: .float32,
                weightFormat: .float16,
                fixedOutputDimension: 5,
                tileElements: 1_024
            )
        )
    }

    @Test("Specialized decode GEMV argument table matches CPU reference with odd output tail")
    func specializedDecodeGEMVArgumentTableMatchesCPUReferenceWithOddOutputTail() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let argumentBufferIndex = 30
        try assertSingleDecodeGEMVOddTail(
            device: device,
            name: "test_vocab_decode_gemv_argbuf_odd_tail",
            inputDimension: 2_048,
            source: MetalSourceGenerator.generateVocabGEMVArgumentTableVariant(
                name: "test_vocab_decode_gemv_argbuf_odd_tail",
                argumentBufferIndex: argumentBufferIndex,
                bufferPrecision: .float32,
                weightFormat: .float16
            ),
            argumentBufferIndex: argumentBufferIndex
        )
        try assertSingleDecodeGEMVOddTail(
            device: device,
            name: "test_input2048_decode_gemv_argbuf_odd_tail",
            inputDimension: 2_048,
            source: MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_input2048_decode_gemv_argbuf_odd_tail",
                argumentBufferIndex: argumentBufferIndex,
                bufferPrecision: .float32,
                weightFormat: .float16,
                fixedOutputDimension: 5
            ),
            argumentBufferIndex: argumentBufferIndex
        )
        try assertSingleDecodeGEMVOddTail(
            device: device,
            name: "test_input8192_decode_gemv_argbuf_odd_tail",
            inputDimension: 8_192,
            source: MetalSourceGenerator.generateInput8192TiledGEMVArgumentTableVariant(
                name: "test_input8192_decode_gemv_argbuf_odd_tail",
                argumentBufferIndex: argumentBufferIndex,
                bufferPrecision: .float32,
                weightFormat: .float16,
                fixedOutputDimension: 5,
                tileElements: 1_024
            ),
            argumentBufferIndex: argumentBufferIndex
        )
    }

    @Test("Quantized Q8 GEMM matches CPU reference with padded scratch input and output stride")
    func quantizedQ8GEMMMatchesCPUReferenceWithPaddedScratchInputAndOutputStride() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 64
        let outputDimension = 5
        let sequenceLength = 3
        let inputRowStride = 96
        let outputRowStride = 8

        var input = [Float](repeating: 9_999, count: sequenceLength * inputRowStride)
        for seq in 0..<sequenceLength {
            for column in 0..<inputDimension {
                input[seq * inputRowStride + column] = Float(((seq + 2) * ((column % 11) - 5))) * 0.0625
            }
        }

        var weightBytes: [UInt8] = []
        weightBytes.reserveCapacity(outputDimension * 68)
        func appendBytes<T>(_ value: T) {
            withUnsafeBytes(of: value) { weightBytes.append(contentsOf: $0) }
        }
        for row in 0..<outputDimension {
            let scale = Float16(0.125)
            let zero = Float16(0)
            appendBytes(scale)
            appendBytes(zero)
            weightBytes.append(contentsOf: repeatElement(UInt8(row + 2), count: inputDimension))
        }

        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weightBytes,
            length: weightBytes.count,
            options: .storageModeShared
        ))
        var output = [Float](repeating: .nan, count: outputRowStride * sequenceLength)
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateQuantizedGEMM_Q8(
                name: "test_quantized_gemm_q8_g64_f32s",
                bufferPrecision: .float32,
                groupSize: 64
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_quantized_gemm_q8_g64_f32s"))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        var inputStride = UInt32(inputRowStride)
        var outputStride = UInt32(outputRowStride)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&inputStride, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.size, index: 7)
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (outputDimension + 1) / 2, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actualPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: output.count
        )
        var actual: [Float] = []
        actual.reserveCapacity(outputDimension * sequenceLength)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                actual.append(actualPointer[seq * outputRowStride + row])
            }
            for row in outputDimension..<outputRowStride {
                #expect(actualPointer[seq * outputRowStride + row].isNaN)
            }
        }

        var expected = [Float](repeating: .zero, count: outputDimension * sequenceLength)
        for seq in 0..<sequenceLength {
            let inputSum = (0..<inputDimension).reduce(Float.zero) { partial, column in
                partial + input[seq * inputRowStride + column]
            }
            for row in 0..<outputDimension {
                expected[seq * outputDimension + row] = inputSum * (0.125 * Float(row + 2))
            }
        }

        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(
            maxError < 0.001,
            """
            Quantized Q8 GEMM drifted with padded scratch input/output stride
            maxError=\(maxError)
            actualPrefix=\(actual.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            expectedPrefix=\(expected.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("Unified quantized GEMV matches CPU reference with odd output tail")
    func unifiedQuantizedGEMVMatchesCPUReferenceWithOddOutputTail() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 64
        let outputDimension = 5
        let outputCapacity = 8
        let scale: Float = 0.25
        let quantizedValueBase: UInt8 = 2
        let input = makeDecodeFloatInput(inputDimension: inputDimension)
        let weights = makeConstantQ4Weights(
            outputDimension: outputDimension,
            inputDimension: inputDimension,
            scale: scale,
            quantizedValueBase: quantizedValueBase
        )
        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weights,
            length: weights.count,
            options: .storageModeShared
        ))
        var output = [Float](repeating: .nan, count: outputCapacity)
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let kernelName = "test_unified_q4_gemv_odd_tail"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: kernelName,
                format: AffineQ4Group64Format(),
                bufferPrecision: .float32,
                tileElements: inputDimension
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: kernelName))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inputDim = UInt32(inputDimension)
        var outputDim = UInt32(outputDimension)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outputDim, length: MemoryLayout<UInt32>.size, index: 4)
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / pipeline.threadExecutionWidth)
        encoder.dispatchThreadgroups(
            MTLSize(width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        assertConstantQ4DecodeOutput(
            outputBuffer,
            input: input,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            outputCapacity: outputCapacity,
            scale: scale,
            quantizedValueBase: quantizedValueBase
        )
    }

    @Test("Batched quantized Q4 GEMM count 2 matches CPU reference with padded output stride")
    func batchedQuantizedQ4GEMM2MatchesCPUReferenceWithPaddedScratchOutputStride() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 64
        let outputDim0 = 3
        let outputDim1 = 5
        let sequenceLength = 3
        let inputRowStride = 96
        let outputRowStride = 8

        let input = makePaddedFloatInput(
            sequenceLength: sequenceLength,
            inputDimension: inputDimension,
            inputRowStride: inputRowStride
        )
        let weight0 = makeConstantQ4Weights(
            outputDimension: outputDim0,
            inputDimension: inputDimension,
            scale: 0.25,
            quantizedValueBase: 1
        )
        let weight1 = makeConstantQ4Weights(
            outputDimension: outputDim1,
            inputDimension: inputDimension,
            scale: 0.25,
            quantizedValueBase: 5
        )

        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer0 = try #require(device.makeBuffer(
            bytes: weight0,
            length: weight0.count,
            options: .storageModeShared
        ))
        let weightBuffer1 = try #require(device.makeBuffer(
            bytes: weight1,
            length: weight1.count,
            options: .storageModeShared
        ))
        var output0 = [Float](repeating: .nan, count: outputRowStride * sequenceLength)
        var output1 = [Float](repeating: .nan, count: outputRowStride * sequenceLength)
        let outputBuffer0 = try #require(device.makeBuffer(
            bytes: &output0,
            length: output0.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let outputBuffer1 = try #require(device.makeBuffer(
            bytes: &output1,
            length: output1.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateBatchedQuantizedGEMM_Q4_2(
                name: "test_batched_quantized_gemm_q4_2_g64_f32s",
                bufferPrecision: .float32,
                groupSize: 64
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_batched_quantized_gemm_q4_2_g64_f32s"))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer0, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer1, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer0, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer1, offset: 0, index: 4)
        var inDim = UInt32(inputDimension)
        var outDim0 = UInt32(outputDim0)
        var outDim1 = UInt32(outputDim1)
        var seqLen = UInt32(sequenceLength)
        var inputStride = UInt32(inputRowStride)
        var outputStride = UInt32(outputRowStride)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&outDim0, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&outDim1, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&inputStride, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.size, index: 10)
        let totalRows = outputDim0 + outputDim1
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (totalRows + 1) / 2, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        assertConstantQ4Output(
            outputBuffer0,
            input: input,
            sequenceLength: sequenceLength,
            inputDimension: inputDimension,
            inputRowStride: inputRowStride,
            outputDimension: outputDim0,
            outputRowStride: outputRowStride,
            scale: 0.25,
            quantizedValueBase: 1
        )
        assertConstantQ4Output(
            outputBuffer1,
            input: input,
            sequenceLength: sequenceLength,
            inputDimension: inputDimension,
            inputRowStride: inputRowStride,
            outputDimension: outputDim1,
            outputRowStride: outputRowStride,
            scale: 0.25,
            quantizedValueBase: 5
        )
    }

    @Test("Batched quantized Q4 GEMM count 3 matches CPU reference with padded output stride")
    func batchedQuantizedQ4GEMM3MatchesCPUReferenceWithPaddedScratchOutputStride() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 64
        let outputDim0 = 2
        let outputDim1 = 4
        let outputDim2 = 5
        let sequenceLength = 3
        let inputRowStride = 96
        let outputRowStride = 8

        let input = makePaddedFloatInput(
            sequenceLength: sequenceLength,
            inputDimension: inputDimension,
            inputRowStride: inputRowStride
        )
        let weight0 = makeConstantQ4Weights(
            outputDimension: outputDim0,
            inputDimension: inputDimension,
            scale: 0.25,
            quantizedValueBase: 1
        )
        let weight1 = makeConstantQ4Weights(
            outputDimension: outputDim1,
            inputDimension: inputDimension,
            scale: 0.25,
            quantizedValueBase: 4
        )
        let weight2 = makeConstantQ4Weights(
            outputDimension: outputDim2,
            inputDimension: inputDimension,
            scale: 0.25,
            quantizedValueBase: 8
        )

        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer0 = try #require(device.makeBuffer(
            bytes: weight0,
            length: weight0.count,
            options: .storageModeShared
        ))
        let weightBuffer1 = try #require(device.makeBuffer(
            bytes: weight1,
            length: weight1.count,
            options: .storageModeShared
        ))
        let weightBuffer2 = try #require(device.makeBuffer(
            bytes: weight2,
            length: weight2.count,
            options: .storageModeShared
        ))
        var output0 = [Float](repeating: .nan, count: outputRowStride * sequenceLength)
        var output1 = [Float](repeating: .nan, count: outputRowStride * sequenceLength)
        var output2 = [Float](repeating: .nan, count: outputRowStride * sequenceLength)
        let outputBuffer0 = try #require(device.makeBuffer(
            bytes: &output0,
            length: output0.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let outputBuffer1 = try #require(device.makeBuffer(
            bytes: &output1,
            length: output1.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let outputBuffer2 = try #require(device.makeBuffer(
            bytes: &output2,
            length: output2.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateBatchedQuantizedGEMM_Q4_3(
                name: "test_batched_quantized_gemm_q4_3_g64_f32s",
                bufferPrecision: .float32,
                groupSize: 64
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_batched_quantized_gemm_q4_3_g64_f32s"))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer0, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer1, offset: 0, index: 2)
        encoder.setBuffer(weightBuffer2, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer0, offset: 0, index: 4)
        encoder.setBuffer(outputBuffer1, offset: 0, index: 5)
        encoder.setBuffer(outputBuffer2, offset: 0, index: 6)
        var inDim = UInt32(inputDimension)
        var outDim0 = UInt32(outputDim0)
        var outDim1 = UInt32(outputDim1)
        var outDim2 = UInt32(outputDim2)
        var seqLen = UInt32(sequenceLength)
        var inputStride = UInt32(inputRowStride)
        var outputStride = UInt32(outputRowStride)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&outDim0, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&outDim1, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBytes(&outDim2, length: MemoryLayout<UInt32>.size, index: 10)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 11)
        encoder.setBytes(&inputStride, length: MemoryLayout<UInt32>.size, index: 12)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.size, index: 13)
        let totalRows = outputDim0 + outputDim1 + outputDim2
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (totalRows + 1) / 2, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        assertConstantQ4Output(
            outputBuffer0,
            input: input,
            sequenceLength: sequenceLength,
            inputDimension: inputDimension,
            inputRowStride: inputRowStride,
            outputDimension: outputDim0,
            outputRowStride: outputRowStride,
            scale: 0.25,
            quantizedValueBase: 1
        )
        assertConstantQ4Output(
            outputBuffer1,
            input: input,
            sequenceLength: sequenceLength,
            inputDimension: inputDimension,
            inputRowStride: inputRowStride,
            outputDimension: outputDim1,
            outputRowStride: outputRowStride,
            scale: 0.25,
            quantizedValueBase: 4
        )
        assertConstantQ4Output(
            outputBuffer2,
            input: input,
            sequenceLength: sequenceLength,
            inputDimension: inputDimension,
            inputRowStride: inputRowStride,
            outputDimension: outputDim2,
            outputRowStride: outputRowStride,
            scale: 0.25,
            quantizedValueBase: 8
        )
    }

    @Test("Generated structural kernels compile")
    func structuralCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
            var allSource = MetalSourceGenerator.commonHeader + "\n\n"
            allSource += MetalSourceGenerator.generateCopy(
                name: "copy_\(precision)", bufferPrecision: precision) + "\n\n"
            allSource += MetalSourceGenerator.generateResidualAdd(
                name: "add_\(precision)", bufferPrecision: precision) + "\n\n"
            allSource += MetalSourceGenerator.generateArgmax(
                name: "argmax_\(precision)", bufferPrecision: precision) + "\n\n"
            allSource += MetalSourceGenerator.generateEmbeddingLookup(
                name: "emb_\(precision)", bufferPrecision: precision, weightFormat: .bfloat16)

            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: allSource, options: options)
            #expect(library.makeFunction(name: "copy_\(precision)") != nil)
            #expect(library.makeFunction(name: "add_\(precision)") != nil)
            #expect(library.makeFunction(name: "argmax_\(precision)") != nil)
            #expect(library.makeFunction(name: "emb_\(precision)") != nil)
        }
    }

    @Test("All precision × weight format combinations produce unique kernels")
    func noDuplicateVariants() {
        // Same computation, different precision/format → different source
        let a = MetalSourceGenerator.generateReduction(
            name: "norm", dimension: 2048, epsilon: 1e-5,
            bufferPrecision: .float16, weightFormat: .float16)
        let b = MetalSourceGenerator.generateReduction(
            name: "norm", dimension: 2048, epsilon: 1e-5,
            bufferPrecision: .float32, weightFormat: .bfloat16)
        #expect(a != b, "Different precision/format should produce different source")

        // Verify BF16 source uses bf16_to_float
        #expect(b.contains("bf16_to_float"))
        #expect(!a.contains("bf16_to_float"))
    }

    @Test("SSM recurrence resolves distinct BF16 kernel variants")
    func ssmRecurrenceKernelNamesIncludeWeightFormat() {
        let fragment = SSMRecurrenceFragment(
            headCount: 8,
            groupCount: 8,
            keyHeadDimension: 64,
            valueHeadDimension: 64,
            convKernelSize: 4
        )

        #expect(fragment.kernelName(context: KernelContext(bufferPrecision: .float16, weightFormat: .float16)) == "ssm_recurrence")
        #expect(fragment.kernelName(context: KernelContext(bufferPrecision: .float16, weightFormat: .bfloat16)) == "ssm_recurrence_bf16")
        #expect(fragment.kernelName(context: KernelContext(bufferPrecision: .float32, weightFormat: .float16)) == "ssm_recurrence_f32")
        #expect(fragment.kernelName(context: KernelContext(bufferPrecision: .float32, weightFormat: .bfloat16)) == "ssm_recurrence_bf16_f32")
        #expect(SSMRecurrenceFragment.sequenceKernelName(bufferPrecision: .float32, weightFormat: .bfloat16) == "ssm_recurrence_seq_bf16_f32")
        #expect(SSMRecurrenceFragment.prewriteDecaySequenceKernelName(bufferPrecision: .float32, weightFormat: .bfloat16) == "ssm_recurrence_seq_bf16_f32_prewrite_decay")
        #expect(SSMRecurrenceFragment.qkParallelSequenceKernelName(bufferPrecision: .float32, weightFormat: .bfloat16) == "ssm_recurrence_seq_bf16_f32_qkpar")
        #expect(SSMRecurrenceFragment.cachedParametersSequenceKernelName(bufferPrecision: .float32, weightFormat: .bfloat16) == "ssm_recurrence_seq_bf16_f32_cached_params")
        #expect(SSMRecurrenceFragment.parallelStateSequenceKernelName(bufferPrecision: .float32, weightFormat: .bfloat16) == "ssm_recurrence_seq_bf16_f32_parallel_state")
        #expect(SSMRecurrenceFragment.groupOwnedPartialProjectionSequenceKernelName(bufferPrecision: .float32, weightFormat: .bfloat16) == "ssm_recurrence_seq_bf16_f32_group_owned_partial")
        #expect(SSMRecurrenceFragment.partitionOwnedPartialProjectionSequenceKernelName(bufferPrecision: .float32, weightFormat: .bfloat16) == "ssm_recurrence_seq_bf16_f32_partition_owned_partial")
    }

    @Test("SSM recurrence reads Qwen gated norm weight directly")
    func ssmRecurrenceUsesDirectNormWeight() {
        let decodeSource = MetalSourceGenerator.generateSSMRecurrence(
            name: "ssm_recurrence_bf16",
            bufferPrecision: .float16,
            weightFormat: .bfloat16,
            convDimension: 4096,
            maxThreadgroupSize: 1024,
            headCount: 16,
            groupCount: 16,
            keyHeadDimension: 128,
            valueHeadDimension: 128
        )
        let sequenceSource = MetalSourceGenerator.generateSSMRecurrenceSequence(
            name: "ssm_recurrence_seq_bf16_f32",
            bufferPrecision: .float32,
            weightFormat: .bfloat16,
            convDimension: 4096,
            maxThreadgroupSize: 1024,
            headCount: 16,
            groupCount: 16,
            keyHeadDimension: 128,
            valueHeadDimension: 128
        )

        for source in [decodeSource, sequenceSource] {
            #expect(source.contains("device const float* normWeight [[buffer(5)]]"))
            #expect(source.contains("* rmsScale * normWeight[d]"))
            #expect(!source.contains("1.0f + normWeight[d]"))
            #expect(!source.contains("bf16_to_float(normWeight[d])"))
        }
    }

    @Test("SSM recurrence prewrite decay variant materializes decayed state")
    func ssmRecurrencePrewriteDecayVariantMaterializesDecayedState() {
        let source = MetalSourceGenerator.generateSSMRecurrenceSequence(
            name: "ssm_recurrence_seq_bf16_f32_prewrite_decay",
            bufferPrecision: .float32,
            weightFormat: .bfloat16,
            convDimension: 4096,
            maxThreadgroupSize: 1024,
            headCount: 16,
            groupCount: 16,
            keyHeadDimension: 128,
            valueHeadDimension: 128,
            prewriteDecayedState: true
        )

        #expect(source.contains("state[j * dv + d] = s;"))
        #expect(source.contains("state[j * dv + d] = state[j * dv + d] + convSiluCache[kBase + j] * kInvDelta;"))
    }

    @Test("Batched QK norm decode applies unit-offset weight bias")
    func batchedQKNormDecodeAppliesWeightBias() {
        let source = MetalSourceGenerator.generateBatchedPerHead2(
            name: "batched_qk_rms_norm_bf16_2",
            bufferPrecision: .float16,
            weightFormat: .bfloat16
        )

        #expect(source.contains("constant float& weightBias     [[buffer(8)]]"))
        #expect(source.contains("float affine = bf16_to_float(weight[i]) + weightBias;"))
        #expect(source.contains("scale * affine"))
    }

    @Test("QK norm decode overdispatch updates only valid heads")
    func qkNormDecodeOverdispatchUpdatesOnlyValidHeads() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try assertQKNormDecodeOverdispatch(
            device: device,
            name: "test_qk_norm_overdispatch",
            source: MetalSourceGenerator.generateQKNorm(
                name: "test_qk_norm_overdispatch",
                bufferPrecision: .float32,
                weightFormat: .float16
            )
        )
        try assertQKNormDecodeOverdispatch(
            device: device,
            name: "test_qk_norm_argbuf_overdispatch",
            source: MetalSourceGenerator.generateQKNormArgumentTableVariant(
                name: "test_qk_norm_argbuf_overdispatch",
                argumentBufferIndex: 30,
                bufferPrecision: .float32,
                weightFormat: .float16
            ),
            argumentBufferIndex: 30
        )
    }

    @Test("Batched QK norm decode overdispatch updates only valid heads")
    func batchedQKNormDecodeOverdispatchUpdatesOnlyValidHeads() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try assertBatchedQKNormDecodeOverdispatch(
            device: device,
            name: "test_batched_qk_norm_overdispatch",
            source: MetalSourceGenerator.generateBatchedPerHead2(
                name: "test_batched_qk_norm_overdispatch",
                bufferPrecision: .float32,
                weightFormat: .float16
            )
        )
        try assertBatchedQKNormDecodeOverdispatch(
            device: device,
            name: "test_batched_qk_norm_argbuf_overdispatch",
            source: MetalSourceGenerator.generateBatchedPerHead2ArgumentTableVariant(
                name: "test_batched_qk_norm_argbuf_overdispatch",
                argumentBufferIndex: 30,
                bufferPrecision: .float32,
                weightFormat: .float16
            ),
            argumentBufferIndex: 30
        )
    }

    @Test("RoPE decode overdispatch updates only valid heads and lanes")
    func ropeDecodeOverdispatchUpdatesOnlyValidHeadsAndLanes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try assertRoPEDecodeOverdispatch(
            device: device,
            name: "test_rope_overdispatch",
            source: MetalSourceGenerator.generateRoPE(
                name: "test_rope_overdispatch",
                bufferPrecision: .float32
            )
        )
        try assertRoPEDecodeOverdispatch(
            device: device,
            name: "test_rope_argbuf_overdispatch",
            source: MetalSourceGenerator.generateRoPEArgumentTableVariant(
                name: "test_rope_argbuf_overdispatch",
                argumentBufferIndex: 30,
                bufferPrecision: .float32
            ),
            argumentBufferIndex: 30
        )
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q6 bit extraction")
    func unifiedQuantizedGEMVQ6EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ6Group16Format(),
            AffineQ6Group32Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // MLX extract_bits<6>: 4 weights span 3 bytes. Each weight slot is
            // selected by a ternary chain keyed on `k & 3`.
            #expect(source.contains("qs[(((k)) >> 2) * 3 + 0] & 0x3f"),
                "Missing w[0] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 2) * 3 + 0] >> 6) & 0x03) | ((qs[(((k)) >> 2) * 3 + 1] & 0x0f) << 2)"),
                "Missing w[1] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 2) * 3 + 1] >> 4) & 0x0f) | ((qs[(((k)) >> 2) * 3 + 2] & 0x03) << 4)"),
                "Missing w[2] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("qs[(((k)) >> 2) * 3 + 2] >> 2"),
                "Missing w[3] extraction for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q2 bit extraction")
    func unifiedQuantizedGEMVQ2EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ2Group16Format(),
            AffineQ2Group32Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // Aligned Q2: 4 weights per byte. perWeightExpression inlined in per-k loop.
            // `qs[(k) >> 2]` selects the byte; `((k) & 3) * 2` shifts to low bits; `& 0x3` masks.
            #expect(source.contains("qs[(k) >> 2]"),
                "Missing Q2 byte index for \(format.schemeIdentifier)")
            #expect(source.contains("((k) & 3) * 2"),
                "Missing Q2 sub-byte shift for \(format.schemeIdentifier)")
            #expect(source.contains("& 0x3)"),
                "Missing Q2 2-bit mask for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q4 bit extraction")
    func unifiedQuantizedGEMVQ4EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ4Group64Format(),
            AffineQ4Group128Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // Aligned Q4: 2 weights per byte.
            // `qs[(k) >> 1]` selects the byte; `((k) & 1) * 4` shifts to low nibble; `& 0xF` masks.
            #expect(source.contains("qs[(k) >> 1]"),
                "Missing Q4 byte index for \(format.schemeIdentifier)")
            #expect(source.contains("((k) & 1) * 4"),
                "Missing Q4 nibble shift for \(format.schemeIdentifier)")
            #expect(source.contains("& 0xF)"),
                "Missing Q4 4-bit mask for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q8 byte read")
    func unifiedQuantizedGEMVQ8EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ8Group32Format(),
            AffineQ8Group64Format(),
            AffineQ8Group128Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // Aligned Q8: 1 weight per byte, direct read.
            #expect(source.contains("float(qs[k])"),
                "Missing Q8 direct byte read for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q3 bit extraction")
    func unifiedQuantizedGEMVQ3EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ3Group16Format(),
            AffineQ3Group32Format(),
            AffineQ3Group64Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // MLX extract_bits<3>: 8 weights span 3 bytes.
            #expect(source.contains("qs[(((k)) >> 3) * 3 + 0] & 0x07"),
                "Missing Q3 w[0] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 3 + 0] >> 3) & 0x07"),
                "Missing Q3 w[1] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 3 + 0] >> 6) & 0x03) | ((qs[(((k)) >> 3) * 3 + 1] & 0x01) << 2)"),
                "Missing Q3 w[2] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 3 + 1] >> 1) & 0x07"),
                "Missing Q3 w[3] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 3 + 1] >> 4) & 0x07"),
                "Missing Q3 w[4] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 3 + 1] >> 7) & 0x01) | ((qs[(((k)) >> 3) * 3 + 2] & 0x03) << 1)"),
                "Missing Q3 w[5] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 3 + 2] >> 2) & 0x07"),
                "Missing Q3 w[6] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 3 + 2] >> 5) & 0x07"),
                "Missing Q3 w[7] extraction for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q5 bit extraction")
    func unifiedQuantizedGEMVQ5EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ5Group32Format(),
            AffineQ5Group64Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // MLX extract_bits<5>: 8 weights span 5 bytes.
            #expect(source.contains("qs[(((k)) >> 3) * 5 + 0] & 0x1f"),
                "Missing Q5 w[0] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 5 + 0] >> 5) & 0x07) | ((qs[(((k)) >> 3) * 5 + 1] & 0x03) << 3)"),
                "Missing Q5 w[1] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 5 + 1] >> 2) & 0x1f"),
                "Missing Q5 w[2] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 5 + 1] >> 7) & 0x01) | ((qs[(((k)) >> 3) * 5 + 2] & 0x0f) << 1)"),
                "Missing Q5 w[3] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 5 + 2] >> 4) & 0x0f) | ((qs[(((k)) >> 3) * 5 + 3] & 0x01) << 4)"),
                "Missing Q5 w[4] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 5 + 3] >> 1) & 0x1f"),
                "Missing Q5 w[5] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 5 + 3] >> 6) & 0x03) | ((qs[(((k)) >> 3) * 5 + 4] & 0x07) << 2)"),
                "Missing Q5 w[6] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 5 + 4] >> 3) & 0x1f"),
                "Missing Q5 w[7] extraction for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV compiles for Q6 group16 and group32")
    func unifiedQuantizedGEMVQ6Compiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let formats: [any QuantizationFormat] = [
            AffineQ6Group16Format(),
            AffineQ6Group32Format(),
        ]
        for format in formats {
            for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
                let name = "test_q6_g\(format.groupSize)_\(precision)"
                let source = MetalSourceGenerator.commonHeader + "\n\n"
                    + MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                        name: name,
                        format: format,
                        bufferPrecision: precision
                    )
                let options = MTLCompileOptions()
                options.languageVersion = .version4_0
                let library = try device.makeLibrary(source: source, options: options)
                #expect(library.makeFunction(name: name) != nil,
                    "Failed to compile \(name)")
            }
        }
    }

    @Test("Q6 group dequant matches CPU reference for all packed values")
    func q6GroupDequantMatchesReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let formats: [any QuantizationFormat] = [
            AffineQ6Group16Format(),
            AffineQ6Group32Format(),
        ]

        for format in formats {
            // Sweep quant values spanning the full 6-bit range [0, 63] by
            // using a scale=1.0, zero=0.0 block whose packed weights cycle
            // through 0..63 so every 4-weight group exercises one bit offset.
            let wpb = format.weightsPerBlock
            let scale: Float16 = 1.0
            let zero: Float16 = 0.0

            // Build expected weight values [0, 1, 2, ..., wpb-1] — all
            // distinct values < 64 so none overflow 6 bits.
            let expectedWeights: [UInt8] = (0..<wpb).map { UInt8($0) }
            precondition(expectedWeights.allSatisfy { $0 < 64 })

            // Pack into Q6 layout: 4 weights share 3 bytes per MLX spec.
            let packedBytes = (wpb / 4) * 3
            var blockBytes = Data(count: 4 + packedBytes)
            blockBytes.withUnsafeMutableBytes { raw in
                var scaleCopy = scale
                memcpy(raw.baseAddress!, &scaleCopy, 2)
                var zeroCopy = zero
                memcpy(raw.baseAddress! + 2, &zeroCopy, 2)
                let qs = raw.baseAddress!.advanced(by: 4)
                    .assumingMemoryBound(to: UInt8.self)
                for g in 0..<(wpb / 4) {
                    let w0 = expectedWeights[g * 4 + 0]
                    let w1 = expectedWeights[g * 4 + 1]
                    let w2 = expectedWeights[g * 4 + 2]
                    let w3 = expectedWeights[g * 4 + 3]
                    qs[g * 3 + 0] = (w0 & 0x3f) | ((w1 & 0x03) << 6)
                    qs[g * 3 + 1] = ((w1 >> 2) & 0x0f) | ((w2 & 0x0f) << 4)
                    qs[g * 3 + 2] = ((w2 >> 4) & 0x03) | ((w3 & 0x3f) << 2)
                }
            }

            // Compile a dequant-only kernel that wraps perWeightReadExpression.
            let name = "dequant_q6_g\(format.groupSize)_test"
            let readExpression = format.perWeightReadExpression(
                blocksVar: "qs",
                weightIndexVar: "k"
            )!
            let source = """
            #include <metal_stdlib>
            using namespace metal;

            kernel void \(name)(
                device const uchar* block [[buffer(0)]],
                device float* output      [[buffer(1)]]
            ) {
                float scale = float(*(device const half*)(block));
                float zero  = float(*(device const half*)(block + 2));
                device const uchar* qs = block + 4;
                for (uint k = 0; k < \(wpb); k++) output[k] = \(readExpression);
            }
            """
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: source, options: options)
            let pipeline = try device.makeComputePipelineState(
                function: try #require(library.makeFunction(name: name))
            )

            let blockBuffer = try #require(device.makeBuffer(
                bytes: [UInt8](blockBytes),
                length: blockBytes.count,
                options: .storageModeShared
            ))
            let outputBuffer = try #require(device.makeBuffer(
                length: wpb * MemoryLayout<Float>.size,
                options: .storageModeShared
            ))

            let queue = try #require(device.makeCommandQueue())
            let commandBuffer = try #require(queue.makeCommandBuffer())
            let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(blockBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let result = outputBuffer.contents().bindMemory(
                to: Float.self, capacity: wpb
            )
            for k in 0..<wpb {
                let expected = Float(expectedWeights[k])
                let actual = result[k]
                #expect(
                    abs(actual - expected) < 1e-5,
                    "\(format.schemeIdentifier) dequant mismatch at k=\(k): expected=\(expected) actual=\(actual)"
                )
            }
        }
    }

    @Test("Generated complete library includes BF16 SSM kernels")
    func completeLibraryIncludesBF16SSMVariants() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let source = MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16)
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)

        for name in [
            "ssm_recurrence_bf16",
            "ssm_recurrence_bf16_f32",
            "ssm_recurrence_seq_bf16",
            "ssm_recurrence_seq_bf16_f32",
            "ssm_recurrence_seq_bf16_f32_cached_params",
            "ssm_recurrence_seq_bf16_f32_parallel_state",
            "ssm_recurrence_seq_bf16_f32_group_owned_partial",
            "ssm_recurrence_seq_bf16_f32_partition_owned_partial",
        ] {
            #expect(library.makeFunction(name: name) != nil, "Missing: \(name)")
        }
    }

    private func makeQuantizedBlock(
        weights: [UInt32],
        bits: Int,
        scale: Float,
        zero: Float,
        payloadByteCount: Int
    ) -> [UInt8] {
        var bytes = [UInt8]()
        bytes.reserveCapacity(4 + payloadByteCount)
        let scaleBits = Float16(scale).bitPattern
        let zeroBits = Float16(zero).bitPattern
        bytes.append(UInt8(scaleBits & 0x00FF))
        bytes.append(UInt8((scaleBits >> 8) & 0x00FF))
        bytes.append(UInt8(zeroBits & 0x00FF))
        bytes.append(UInt8((zeroBits >> 8) & 0x00FF))
        bytes.append(contentsOf: packLSBFirstBitStream(weights: weights, bits: bits))
        #expect(bytes.count == 4 + payloadByteCount)
        return bytes
    }

    private func makePaddedFloatInput(
        sequenceLength: Int,
        inputDimension: Int,
        inputRowStride: Int
    ) -> [Float] {
        var input = [Float](repeating: 9_999, count: sequenceLength * inputRowStride)
        for seq in 0..<sequenceLength {
            for column in 0..<inputDimension {
                input[seq * inputRowStride + column] = Float(((seq + 3) * ((column % 9) - 4))) * 0.125
            }
        }
        return input
    }

    private func makeDecodeFloatInput(inputDimension: Int) -> [Float] {
        (0..<inputDimension).map { index in
            Float((index * 13) % 41 - 20) * 0.03125
        }
    }

    private func makeDecodeFloat16Weights(
        outputDimension: Int,
        inputDimension: Int,
        projectionSeed: Int
    ) -> [Float16] {
        (0..<(outputDimension * inputDimension)).map { index in
            Float16(Float((index * projectionSeed) % 37 - 18) * 0.015625)
        }
    }

    private func decodeCPUReference(
        input: [Float],
        weights: [Float16],
        inputDimension: Int,
        outputDimension: Int
    ) -> [Float] {
        var expected = [Float](repeating: .zero, count: outputDimension)
        for row in 0..<outputDimension {
            var sum: Float = 0
            for column in 0..<inputDimension {
                sum += Float(weights[row * inputDimension + column]) * input[column]
            }
            expected[row] = sum
        }
        return expected
    }

    private func assertSingleDecodeGEMVOddTail(
        device: MTLDevice,
        name: String,
        inputDimension: Int,
        outputDimension: Int = 5,
        outputCapacity: Int = 8,
        source: String,
        argumentBufferIndex: Int? = nil
    ) throws {
        let input = makeDecodeFloatInput(inputDimension: inputDimension)
        let weights = makeDecodeFloat16Weights(
            outputDimension: outputDimension,
            inputDimension: inputDimension,
            projectionSeed: 3
        )
        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ))
        var output = [Float](repeating: .nan, count: outputCapacity)
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.commonHeader + "\n\n" + source,
            options: options
        )
        let function = try #require(library.makeFunction(name: name))
        let pipeline = try device.makeComputePipelineState(function: function)

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        if let argumentBufferIndex {
            let argumentEncoder = function.makeArgumentEncoder(bufferIndex: argumentBufferIndex)
            let argumentBuffer = try #require(device.makeBuffer(
                length: argumentEncoder.encodedLength,
                options: .storageModeShared
            ))
            argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
            argumentEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
            argumentEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
            argumentEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
            encoder.setBuffer(argumentBuffer, offset: 0, index: argumentBufferIndex)
            encoder.useResource(argumentBuffer, usage: .read)
        } else {
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(weightBuffer, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        }
        encoder.useResource(inputBuffer, usage: .read)
        encoder.useResource(weightBuffer, usage: .read)
        encoder.useResource(outputBuffer, usage: .write)
        var inputDim = UInt32(inputDimension)
        var outputDim = UInt32(outputDimension)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outputDim, length: MemoryLayout<UInt32>.size, index: 4)
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / pipeline.threadExecutionWidth)
        encoder.dispatchThreadgroups(
            MTLSize(width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        assertDecodeOutput(
            outputBuffer: outputBuffer,
            outputCapacity: outputCapacity,
            input: input,
            weights: weights,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            label: "\(name) odd-tail"
        )
    }

    private func assertBatchedDecodeGEMVOddTail(
        device: MTLDevice,
        count: Int,
        outputDimensions: [Int],
        usesArgumentTable: Bool = false
    ) throws {
        let argumentBufferIndex = 30
        let inputDimension = 65
        let outputCapacity = 8
        let kernelName = usesArgumentTable
            ? "test_batched_decode_gemv\(count)_argbuf_odd_tail"
            : "test_batched_decode_gemv\(count)_odd_tail"
        let input = makeDecodeFloatInput(inputDimension: inputDimension)
        let weights = outputDimensions.enumerated().map { projection, outputDimension in
            makeDecodeFloat16Weights(
                outputDimension: outputDimension,
                inputDimension: inputDimension,
                projectionSeed: projection + 3
            )
        }
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + batchedDecodeGEMVSource(
                name: kernelName,
                count: count,
                argumentBufferIndex: argumentBufferIndex,
                usesArgumentTable: usesArgumentTable
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: kernelName))
        let pipeline = try device.makeComputePipelineState(function: function)

        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffers = try weights.map { weight in
            try #require(device.makeBuffer(
                bytes: weight,
                length: weight.count * MemoryLayout<Float16>.size,
                options: .storageModeShared
            ))
        }
        var outputValues = outputDimensions.map { _ in
            [Float](repeating: .nan, count: outputCapacity)
        }
        let outputBuffers = try outputValues.indices.map { index in
            try #require(device.makeBuffer(
                bytes: &outputValues[index],
                length: outputValues[index].count * MemoryLayout<Float>.size,
                options: .storageModeShared
            ))
        }

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        if usesArgumentTable {
            let argumentEncoder = function.makeArgumentEncoder(bufferIndex: argumentBufferIndex)
            let argumentBuffer = try #require(device.makeBuffer(
                length: argumentEncoder.encodedLength,
                options: .storageModeShared
            ))
            argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
            argumentEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
            for index in 0..<count {
                argumentEncoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
                argumentEncoder.setBuffer(outputBuffers[index], offset: 0, index: 1 + count + index)
            }
            encoder.setBuffer(argumentBuffer, offset: 0, index: argumentBufferIndex)
            encoder.useResource(argumentBuffer, usage: .read)
        } else {
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            for index in 0..<count {
                encoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
                encoder.setBuffer(outputBuffers[index], offset: 0, index: 1 + count + index)
            }
        }
        encoder.useResource(inputBuffer, usage: .read)
        for index in 0..<count {
            encoder.useResource(weightBuffers[index], usage: .read)
            encoder.useResource(outputBuffers[index], usage: .write)
        }
        let dimBase = 1 + 2 * count
        var inputDim = UInt32(inputDimension)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.size, index: dimBase)
        var outputDims = outputDimensions.map(UInt32.init)
        for index in outputDims.indices {
            encoder.setBytes(&outputDims[index], length: MemoryLayout<UInt32>.size, index: dimBase + 1 + index)
        }
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / pipeline.threadExecutionWidth)
        let totalRows = outputDimensions.reduce(0, +)
        encoder.dispatchThreadgroups(
            MTLSize(width: (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        for projection in 0..<count {
            assertDecodeOutput(
                outputBuffer: outputBuffers[projection],
                outputCapacity: outputCapacity,
                input: input,
                weights: weights[projection],
                inputDimension: inputDimension,
                outputDimension: outputDimensions[projection],
                label: "Batched decode GEMV\(count) odd-tail projection \(projection)"
            )
        }
    }

    private func assertDecodeOutput(
        outputBuffer: MTLBuffer,
        outputCapacity: Int,
        input: [Float],
        weights: [Float16],
        inputDimension: Int,
        outputDimension: Int,
        label: String
    ) {
        let actualPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: outputCapacity
        )
        let expected = decodeCPUReference(
            input: input,
            weights: weights,
            inputDimension: inputDimension,
            outputDimension: outputDimension
        )
        var actual: [Float] = []
        actual.reserveCapacity(outputDimension)
        for row in 0..<outputDimension {
            actual.append(actualPointer[row])
        }
        for row in outputDimension..<outputCapacity {
            #expect(actualPointer[row].isNaN)
        }
        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(
            maxError < 0.001,
            "\(label) drifted: maxError=\(maxError)"
        )
    }

    private func batchedDecodeGEMVSource(
        name: String,
        count: Int,
        argumentBufferIndex: Int = 30,
        usesArgumentTable: Bool = false
    ) -> String {
        switch count {
        case 2:
            usesArgumentTable ? MetalSourceGenerator.generateBatchedGEMV2ArgumentTableVariant(
                name: name,
                argumentBufferIndex: argumentBufferIndex,
                bufferPrecision: .float32,
                weightFormat: .float16
            ) : MetalSourceGenerator.generateBatchedGEMV2(
                name: name,
                bufferPrecision: .float32,
                weightFormat: .float16
            )
        case 3:
            usesArgumentTable ? MetalSourceGenerator.generateBatchedGEMV3ArgumentTableVariant(
                name: name,
                argumentBufferIndex: argumentBufferIndex,
                bufferPrecision: .float32,
                weightFormat: .float16
            ) : MetalSourceGenerator.generateBatchedGEMV3(
                name: name,
                bufferPrecision: .float32,
                weightFormat: .float16
            )
        case 4:
            usesArgumentTable ? MetalSourceGenerator.generateBatchedGEMV4ArgumentTableVariant(
                name: name,
                argumentBufferIndex: argumentBufferIndex,
                bufferPrecision: .float32,
                weightFormat: .float16
            ) : MetalSourceGenerator.generateBatchedGEMV4(
                name: name,
                bufferPrecision: .float32,
                weightFormat: .float16
            )
        default:
            fatalError("Unsupported batched decode GEMV count \(count)")
        }
    }

    private func makeConstantQ4Weights(
        outputDimension: Int,
        inputDimension: Int,
        scale: Float,
        quantizedValueBase: UInt8
    ) -> [UInt8] {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(outputDimension * (4 + inputDimension / 2))
        func appendBytes<T>(_ value: T) {
            withUnsafeBytes(of: value) { bytes.append(contentsOf: $0) }
        }
        for row in 0..<outputDimension {
            appendBytes(Float16(scale))
            appendBytes(Float16(0))
            let quantized = quantizedValueBase + UInt8(row)
            let packed = quantized | (quantized << 4)
            bytes.append(contentsOf: repeatElement(packed, count: inputDimension / 2))
        }
        return bytes
    }

    private func assertConstantQ4Output(
        _ outputBuffer: MTLBuffer,
        input: [Float],
        sequenceLength: Int,
        inputDimension: Int,
        inputRowStride: Int,
        outputDimension: Int,
        outputRowStride: Int,
        scale: Float,
        quantizedValueBase: UInt8
    ) {
        let actualPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: outputRowStride * sequenceLength
        )
        var actual: [Float] = []
        actual.reserveCapacity(outputDimension * sequenceLength)
        var expected = [Float](repeating: .zero, count: outputDimension * sequenceLength)
        for seq in 0..<sequenceLength {
            let inputSum = (0..<inputDimension).reduce(Float.zero) { partial, column in
                partial + input[seq * inputRowStride + column]
            }
            for row in 0..<outputDimension {
                actual.append(actualPointer[seq * outputRowStride + row])
                expected[seq * outputDimension + row] =
                    inputSum * scale * Float(quantizedValueBase + UInt8(row))
            }
            for row in outputDimension..<outputRowStride {
                #expect(actualPointer[seq * outputRowStride + row].isNaN)
            }
        }

        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(
            maxError < 0.001,
            """
            Batched quantized Q4 GEMM drifted with padded output stride
            maxError=\(maxError)
            actualPrefix=\(actual.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            expectedPrefix=\(expected.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    private func assertConstantQ4DecodeOutput(
        _ outputBuffer: MTLBuffer,
        input: [Float],
        inputDimension: Int,
        outputDimension: Int,
        outputCapacity: Int,
        scale: Float,
        quantizedValueBase: UInt8
    ) {
        let actualPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: outputCapacity
        )
        let inputSum = (0..<inputDimension).reduce(Float.zero) { partial, column in
            partial + input[column]
        }
        var maxError: Float = 0
        for row in 0..<outputDimension {
            let expected = inputSum * scale * Float(quantizedValueBase + UInt8(row))
            maxError = max(maxError, abs(actualPointer[row] - expected))
        }
        for row in outputDimension..<outputCapacity {
            #expect(actualPointer[row].isNaN)
        }
        #expect(
            maxError < 0.001,
            "Unified Q4 decode GEMV odd-tail drifted: maxError=\(maxError)"
        )
    }

    private func assertQKNormDecodeOverdispatch(
        device: MTLDevice,
        name: String,
        source: String,
        argumentBufferIndex: Int? = nil
    ) throws {
        let headCount = 3
        let headDimension = 8
        let headCapacity = 4
        let epsilon: Float = 1e-5
        let weightBias: Float = 0
        var data = (0..<(headCapacity * headDimension)).map { index in
            Float((index % 13) - 6) * 0.125
        }
        let tailStart = headCount * headDimension
        for index in tailStart..<data.count {
            data[index] = 1234.5
        }
        let originalData = data
        let weight = (0..<headDimension).map { index in
            Float16(0.75 + Float(index) * 0.03125)
        }
        let dataBuffer = try #require(device.makeBuffer(
            bytes: &data,
            length: data.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ))

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.commonHeader + "\n\n" + source,
            options: options
        )
        let function = try #require(library.makeFunction(name: name))
        let pipeline = try device.makeComputePipelineState(function: function)

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        if let argumentBufferIndex {
            let argumentEncoder = function.makeArgumentEncoder(bufferIndex: argumentBufferIndex)
            let argumentBuffer = try #require(device.makeBuffer(
                length: argumentEncoder.encodedLength,
                options: .storageModeShared
            ))
            argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
            argumentEncoder.setBuffer(dataBuffer, offset: 0, index: 0)
            argumentEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
            encoder.setBuffer(argumentBuffer, offset: 0, index: argumentBufferIndex)
            encoder.useResource(argumentBuffer, usage: .read)
        } else {
            encoder.setBuffer(dataBuffer, offset: 0, index: 0)
            encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        }
        encoder.useResource(dataBuffer, usage: [.read, .write])
        encoder.useResource(weightBuffer, usage: .read)
        var headCountValue = UInt32(headCount)
        var headDimValue = UInt32(headDimension)
        var epsilonValue = epsilon
        var weightBiasValue = weightBias
        encoder.setBytes(&headCountValue, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&headDimValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&epsilonValue, length: MemoryLayout<Float>.size, index: 4)
        encoder.setBytes(&weightBiasValue, length: MemoryLayout<Float>.size, index: 5)
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: headCapacity, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actualPointer = dataBuffer.contents().bindMemory(
            to: Float.self,
            capacity: headCapacity * headDimension
        )
        for head in 0..<headCount {
            let base = head * headDimension
            let sumSquares = (0..<headDimension).reduce(Float.zero) { partial, index in
                let value = originalData[base + index]
                return partial + value * value
            }
            let scale = 1 / sqrt(sumSquares / Float(headDimension) + epsilon)
            for index in 0..<headDimension {
                let expected = originalData[base + index] * scale * (Float(weight[index]) + weightBias)
                #expect(
                    abs(actualPointer[base + index] - expected) < 0.001,
                    "\(name) drifted at head=\(head) index=\(index)"
                )
            }
        }
        for index in tailStart..<data.count {
            #expect(actualPointer[index] == originalData[index])
        }
    }

    private func assertBatchedQKNormDecodeOverdispatch(
        device: MTLDevice,
        name: String,
        source: String,
        argumentBufferIndex: Int? = nil
    ) throws {
        let count0 = 2
        let count1 = 1
        let headDimension = 8
        let epsilon: Float = 1e-5
        let weightBias: Float = 0
        var data0 = (0..<(count0 * headDimension)).map { index in
            Float((index % 11) - 5) * 0.125
        }
        var data1 = (0..<(count1 * headDimension)).map { index in
            Float((index % 7) - 3) * 0.25
        }
        let originalData0 = data0
        let originalData1 = data1
        let weight0 = (0..<headDimension).map { index in
            Float16(0.5 + Float(index) * 0.0625)
        }
        let weight1 = (0..<headDimension).map { index in
            Float16(0.75 + Float(index) * 0.03125)
        }
        let data0Buffer = try #require(device.makeBuffer(
            bytes: &data0,
            length: data0.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let data1Buffer = try #require(device.makeBuffer(
            bytes: &data1,
            length: data1.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weight0Buffer = try #require(device.makeBuffer(
            bytes: weight0,
            length: weight0.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ))
        let weight1Buffer = try #require(device.makeBuffer(
            bytes: weight1,
            length: weight1.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ))

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.commonHeader + "\n\n" + source,
            options: options
        )
        let function = try #require(library.makeFunction(name: name))
        let pipeline = try device.makeComputePipelineState(function: function)

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        if let argumentBufferIndex {
            let argumentEncoder = function.makeArgumentEncoder(bufferIndex: argumentBufferIndex)
            let argumentBuffer = try #require(device.makeBuffer(
                length: argumentEncoder.encodedLength,
                options: .storageModeShared
            ))
            argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
            argumentEncoder.setBuffer(data0Buffer, offset: 0, index: 0)
            argumentEncoder.setBuffer(data1Buffer, offset: 0, index: 1)
            argumentEncoder.setBuffer(weight0Buffer, offset: 0, index: 2)
            argumentEncoder.setBuffer(weight1Buffer, offset: 0, index: 3)
            encoder.setBuffer(argumentBuffer, offset: 0, index: argumentBufferIndex)
            encoder.useResource(argumentBuffer, usage: .read)
        } else {
            encoder.setBuffer(data0Buffer, offset: 0, index: 0)
            encoder.setBuffer(data1Buffer, offset: 0, index: 1)
            encoder.setBuffer(weight0Buffer, offset: 0, index: 2)
            encoder.setBuffer(weight1Buffer, offset: 0, index: 3)
        }
        encoder.useResource(data0Buffer, usage: [.read, .write])
        encoder.useResource(data1Buffer, usage: [.read, .write])
        encoder.useResource(weight0Buffer, usage: .read)
        encoder.useResource(weight1Buffer, usage: .read)
        var count0Value = UInt32(count0)
        var count1Value = UInt32(count1)
        var headDimValue = UInt32(headDimension)
        var epsilonValue = epsilon
        var weightBiasValue = weightBias
        encoder.setBytes(&count0Value, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&count1Value, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&headDimValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&epsilonValue, length: MemoryLayout<Float>.size, index: 7)
        encoder.setBytes(&weightBiasValue, length: MemoryLayout<Float>.size, index: 8)
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: count0 + count1 + 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        assertQKNormValues(
            buffer: data0Buffer,
            original: originalData0,
            weight: weight0,
            headCount: count0,
            headDimension: headDimension,
            epsilon: epsilon,
            weightBias: weightBias,
            label: "\(name).data0"
        )
        assertQKNormValues(
            buffer: data1Buffer,
            original: originalData1,
            weight: weight1,
            headCount: count1,
            headDimension: headDimension,
            epsilon: epsilon,
            weightBias: weightBias,
            label: "\(name).data1"
        )
    }

    private func assertQKNormValues(
        buffer: MTLBuffer,
        original: [Float],
        weight: [Float16],
        headCount: Int,
        headDimension: Int,
        epsilon: Float,
        weightBias: Float,
        label: String
    ) {
        let actualPointer = buffer.contents().bindMemory(
            to: Float.self,
            capacity: original.count
        )
        for head in 0..<headCount {
            let base = head * headDimension
            let sumSquares = (0..<headDimension).reduce(Float.zero) { partial, index in
                let value = original[base + index]
                return partial + value * value
            }
            let scale = 1 / sqrt(sumSquares / Float(headDimension) + epsilon)
            for index in 0..<headDimension {
                let expected = original[base + index] * scale * (Float(weight[index]) + weightBias)
                #expect(
                    abs(actualPointer[base + index] - expected) < 0.001,
                    "\(label) drifted at head=\(head) index=\(index)"
                )
            }
        }
    }

    private func assertRoPEDecodeOverdispatch(
        device: MTLDevice,
        name: String,
        source: String,
        argumentBufferIndex: Int? = nil
    ) throws {
        let headCount = 2
        let kvHeadCount = 1
        let headCapacity = 3
        let headDimension = 8
        let ropeDimension = 8
        let pairCount = ropeDimension / 2
        let ropeBase: Float = 10_000
        let position: UInt32 = 3
        var query = (0..<(headCapacity * headDimension)).map { index in
            Float((index % 17) - 8) * 0.125
        }
        var key = (0..<(headCapacity * headDimension)).map { index in
            Float((index % 19) - 9) * 0.0625
        }
        let queryTailStart = headCount * headDimension
        let keyTailStart = kvHeadCount * headDimension
        for index in queryTailStart..<query.count {
            query[index] = 2000 + Float(index)
        }
        for index in keyTailStart..<key.count {
            key[index] = 3000 + Float(index)
        }
        let originalQuery = query
        let originalKey = key
        var positionAxes = [position, position, position]
        let queryBuffer = try #require(device.makeBuffer(
            bytes: &query,
            length: query.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let keyBuffer = try #require(device.makeBuffer(
            bytes: &key,
            length: key.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let positionBuffer = try #require(device.makeBuffer(
            bytes: &positionAxes,
            length: positionAxes.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ))

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.commonHeader + "\n\n" + source,
            options: options
        )
        let function = try #require(library.makeFunction(name: name))
        let pipeline = try device.makeComputePipelineState(function: function)

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        if let argumentBufferIndex {
            let argumentEncoder = function.makeArgumentEncoder(bufferIndex: argumentBufferIndex)
            let argumentBuffer = try #require(device.makeBuffer(
                length: argumentEncoder.encodedLength,
                options: .storageModeShared
            ))
            argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
            argumentEncoder.setBuffer(queryBuffer, offset: 0, index: 0)
            argumentEncoder.setBuffer(keyBuffer, offset: 0, index: 1)
            argumentEncoder.setBuffer(positionBuffer, offset: 0, index: 2)
            encoder.setBuffer(argumentBuffer, offset: 0, index: argumentBufferIndex)
            encoder.useResource(argumentBuffer, usage: .read)
        } else {
            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(keyBuffer, offset: 0, index: 1)
            encoder.setBuffer(positionBuffer, offset: 0, index: 2)
        }
        encoder.useResource(queryBuffer, usage: [.read, .write])
        encoder.useResource(keyBuffer, usage: [.read, .write])
        encoder.useResource(positionBuffer, usage: .read)
        var headCountValue = UInt32(headCount)
        var kvHeadCountValue = UInt32(kvHeadCount)
        var headDimValue = UInt32(headDimension)
        var ropeDimValue = UInt32(ropeDimension)
        var ropeBaseValue = ropeBase
        var zero: UInt32 = 0
        encoder.setBytes(&headCountValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&kvHeadCountValue, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&headDimValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&ropeDimValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&ropeBaseValue, length: MemoryLayout<Float>.size, index: 7)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.size, index: 10)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.size, index: 11)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.size, index: 12)
        let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: headCapacity, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        assertRoPEValues(
            buffer: queryBuffer,
            original: originalQuery,
            activeHeadCount: headCount,
            headCapacity: headCapacity,
            headDimension: headDimension,
            pairCount: pairCount,
            ropeDimension: ropeDimension,
            position: Float(position),
            ropeBase: ropeBase,
            label: "\(name).query"
        )
        assertRoPEValues(
            buffer: keyBuffer,
            original: originalKey,
            activeHeadCount: kvHeadCount,
            headCapacity: headCapacity,
            headDimension: headDimension,
            pairCount: pairCount,
            ropeDimension: ropeDimension,
            position: Float(position),
            ropeBase: ropeBase,
            label: "\(name).key"
        )
    }

    private func assertRoPEValues(
        buffer: MTLBuffer,
        original: [Float],
        activeHeadCount: Int,
        headCapacity: Int,
        headDimension: Int,
        pairCount: Int,
        ropeDimension: Int,
        position: Float,
        ropeBase: Float,
        label: String
    ) {
        let actualPointer = buffer.contents().bindMemory(
            to: Float.self,
            capacity: headCapacity * headDimension
        )
        for head in 0..<activeHeadCount {
            let base = head * headDimension
            for pair in 0..<pairCount {
                let theta = position * pow(ropeBase, -2 * Float(pair) / Float(ropeDimension))
                let cosTheta = cos(theta)
                let sinTheta = sin(theta)
                let x0 = original[base + pair]
                let x1 = original[base + pairCount + pair]
                let expected0 = x0 * cosTheta - x1 * sinTheta
                let expected1 = x0 * sinTheta + x1 * cosTheta
                #expect(abs(actualPointer[base + pair] - expected0) < 0.0001, "\(label) low pair \(pair) drifted")
                #expect(abs(actualPointer[base + pairCount + pair] - expected1) < 0.0001, "\(label) high pair \(pair) drifted")
            }
        }
        for index in (activeHeadCount * headDimension)..<original.count {
            #expect(actualPointer[index] == original[index])
        }
    }

    private func packLSBFirstBitStream(weights: [UInt32], bits: Int) -> [UInt8] {
        let totalBits = weights.count * bits
        let byteCount = (totalBits + 7) / 8
        var result = [UInt8](repeating: 0, count: byteCount)
        let mask = (UInt64(1) << bits) - 1
        for (index, weight) in weights.enumerated() {
            let value = UInt64(weight) & mask
            let bitOffset = index * bits
            let byteIndex = bitOffset / 8
            let bitIndex = bitOffset % 8
            let shifted = value << bitIndex
            let spannedBytes = (bitIndex + bits + 7) / 8
            for offset in 0..<spannedBytes {
                result[byteIndex + offset] |= UInt8((shifted >> (offset * 8)) & 0xFF)
            }
        }
        return result
    }
}
