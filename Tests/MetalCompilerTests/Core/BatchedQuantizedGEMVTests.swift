import Metal
import Testing
@testable import MetalCompiler

@Suite("Batched Quantized GEMV Correctness", .serialized)
struct BatchedQuantizedGEMVTests {

    @Test("Q4G64 batched GEMV matches independent dot products")
    func q4Group64BatchedGEMV() throws {
        try runBatchedGEMVTest(format: AffineQ4Group64Format(), projectionCount: 3)
    }

    @Test("Q3G64 batched GEMV matches independent dot products")
    func q3Group64BatchedGEMV() throws {
        try runBatchedGEMVTest(format: AffineQ3Group64Format(), projectionCount: 2)
    }

    private func runBatchedGEMVTest(
        format: any QuantizationFormat,
        projectionCount: Int,
        numBlocksPerRow: Int = 3,
        rowsPerProjection: Int = 4
    ) throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let weightsPerBlock = format.weightsPerBlock
        let inputDimension = weightsPerBlock * numBlocksPerRow
        let bitRange = UInt32(1) << format.bits
        let kernelName = "test_batched_quant_gemv_\(format.schemeIdentifier.rawValue)_\(projectionCount)"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateBatchedQuantizedGEMV(
                name: kernelName,
                count: projectionCount,
                format: format,
                bufferPrecision: .float16
            )
        let pipeline = try makePipeline(device: device, source: source, functionName: kernelName)

        let inputValues: [Float16] = (0..<inputDimension).map { index in
            Float16(Float(index % 17) * 0.03125 - 0.25)
        }
        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputValues,
            length: inputValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ))

        var projectionWeights: [[[[UInt32]]]] = []
        var projectionScales: [[[Float]]] = []
        var projectionZeros: [[[Float]]] = []
        var weightBuffers: [MTLBuffer] = []
        var outputBuffers: [MTLBuffer] = []

        for projection in 0..<projectionCount {
            var packedWeightBytes: [UInt8] = []
            var rowWeights: [[[UInt32]]] = []
            var rowScales: [[Float]] = []
            var rowZeros: [[Float]] = []

            for row in 0..<rowsPerProjection {
                var blockWeights: [[UInt32]] = []
                var blockScales: [Float] = []
                var blockZeros: [Float] = []
                for block in 0..<numBlocksPerRow {
                    let scale = 0.03125 * Float(block + 1) + 0.0078125 * Float(projection + row)
                    let zero = -0.25 + 0.0625 * Float(row) - 0.015625 * Float(projection)
                    let weights = (0..<weightsPerBlock).map { k in
                        UInt32((k + block * 5 + row * 7 + projection * 11) % Int(bitRange))
                    }
                    packedWeightBytes.append(contentsOf: packSingleBlock(
                        format: format,
                        weights: weights,
                        scale: scale,
                        zero: zero
                    ))
                    blockWeights.append(weights)
                    blockScales.append(scale)
                    blockZeros.append(zero)
                }
                rowWeights.append(blockWeights)
                rowScales.append(blockScales)
                rowZeros.append(blockZeros)
            }

            projectionWeights.append(rowWeights)
            projectionScales.append(rowScales)
            projectionZeros.append(rowZeros)
            weightBuffers.append(try #require(device.makeBuffer(
                bytes: packedWeightBytes,
                length: packedWeightBytes.count,
                options: .storageModeShared
            )))
            outputBuffers.append(try #require(device.makeBuffer(
                length: rowsPerProjection * MemoryLayout<Float16>.stride,
                options: .storageModeShared
            )))
        }

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        for projection in 0..<projectionCount {
            encoder.setBuffer(weightBuffers[projection], offset: 0, index: 1 + projection)
            encoder.setBuffer(outputBuffers[projection], offset: 0, index: 1 + projectionCount + projection)
        }
        var inputDim = UInt32(inputDimension)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 1 + 2 * projectionCount)
        for projection in 0..<projectionCount {
            var outputDim = UInt32(rowsPerProjection)
            encoder.setBytes(&outputDim, length: MemoryLayout<UInt32>.stride, index: 2 + 2 * projectionCount + projection)
        }
        let threadgroupWidth = 256
        let rowsPerThreadgroup = threadgroupWidth / 32
        let groupCount = (projectionCount * rowsPerProjection + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        encoder.dispatchThreadgroups(
            MTLSize(width: groupCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let absoluteTolerance: Float = 0.04
        let relativeTolerance: Float = 0.04
        for projection in 0..<projectionCount {
            let outputPointer = outputBuffers[projection].contents()
                .bindMemory(to: Float16.self, capacity: rowsPerProjection)
            for row in 0..<rowsPerProjection {
                var expected: Float = 0
                for block in 0..<numBlocksPerRow {
                    let scale = projectionScales[projection][row][block]
                    let zero = projectionZeros[projection][row][block]
                    for k in 0..<weightsPerBlock {
                        let inputIndex = block * weightsPerBlock + k
                        let weight = scale * Float(projectionWeights[projection][row][block][k]) + zero
                        expected += weight * Float(inputValues[inputIndex])
                    }
                }
                let actual = Float(outputPointer[row])
                let tolerance = max(absoluteTolerance, relativeTolerance * abs(expected))
                #expect(
                    abs(actual - expected) < tolerance,
                    """
                    \(format.schemeIdentifier) batched GEMV drift projection=\(projection) row=\(row)
                    actual=\(actual), expected=\(expected), diff=\(actual - expected), tolerance=\(tolerance)
                    """
                )
            }
        }
    }

    private func packSingleBlock(
        format: any QuantizationFormat,
        weights: [UInt32],
        scale: Float,
        zero: Float
    ) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: format.bytesPerBlock)
        let scaleBits = Float16(scale).bitPattern
        let zeroBits = Float16(zero).bitPattern
        bytes[0] = UInt8(scaleBits & 0xFF)
        bytes[1] = UInt8((scaleBits >> 8) & 0xFF)
        bytes[2] = UInt8(zeroBits & 0xFF)
        bytes[3] = UInt8((zeroBits >> 8) & 0xFF)

        let packed = packLSBFirstBitStream(weights: weights, bits: format.bits)
        for (index, byte) in packed.enumerated() {
            bytes[4 + index] = byte
        }
        return bytes
    }

    private func packLSBFirstBitStream(weights: [UInt32], bits: Int) -> [UInt8] {
        let totalBits = weights.count * bits
        let byteCount = (totalBits + 7) / 8
        var result = [UInt8](repeating: 0, count: byteCount)
        let mask = (UInt64(1) << bits) - 1
        for (k, weight) in weights.enumerated() {
            let value = UInt64(weight) & mask
            let bitOffset = k * bits
            let byteIndex = bitOffset / 8
            let bitIndex = bitOffset % 8
            let shifted = value << bitIndex
            let spannedBytes = (bitIndex + bits + 7) / 8
            for offset in 0..<spannedBytes {
                let byte = UInt8((shifted >> (offset * 8)) & 0xFF)
                result[byteIndex + offset] |= byte
            }
        }
        return result
    }

    private func makePipeline(
        device: MTLDevice,
        source: String,
        functionName: String
    ) throws -> MTLComputePipelineState {
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: functionName))
        return try device.makeComputePipelineState(function: function)
    }
}
