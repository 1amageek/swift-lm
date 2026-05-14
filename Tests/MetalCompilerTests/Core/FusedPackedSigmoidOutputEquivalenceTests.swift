import Metal
import Testing
@testable import MetalCompiler

/// Admission tests for the opt-in packed-sigmoid + attention output projection
/// fusion.
///
/// The fused kernel is intentionally non-default today because full-model Qwen
/// profiling showed a regression. These tests keep its numerical contract pinned
/// so future recurrent-block fusion work can reuse or replace it deliberately.
@Suite("Fused Packed Sigmoid Output Equivalence", .serialized)
struct FusedPackedSigmoidOutputEquivalenceTests {
    @Test("Fused packed sigmoid output BF16 matches Swift reference")
    func fusedPackedSigmoidOutputMatchesSwiftReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let shape = Shape()
        let fusedKernel = "test_attn_fused_sigmoid_o_seq_bf16_f32s"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateFusedPackedSigmoidGateOutputSequenceGEMV(
                name: fusedKernel,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let pipeline = try harness.pipeline(named: fusedKernel)

        let input = makeInput(shape: shape)
        let packed = makePacked(shape: shape)
        let weight = makeWeight(shape: shape)
        let actual = try runFusedKernel(
            harness: harness,
            pipeline: pipeline,
            input: input,
            packed: packed,
            weight: weight,
            shape: shape
        )
        let expected = swiftReference(
            input: input,
            packed: packed,
            weight: weight,
            shape: shape
        )

        let mismatch = harness.firstMismatch(expected: expected, actual: actual, tolerance: 0.001)
        let maxError = harness.maxAbsoluteError(expected: expected, actual: actual)
        #expect(
            mismatch == nil,
            "fused packed sigmoid output drifted from Swift reference: \(String(describing: mismatch)), maxError=\(maxError)"
        )
    }

    @Test("Fused packed sigmoid output BF16 matches unfused two-kernel path")
    func fusedPackedSigmoidOutputMatchesUnfusedPath() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let shape = Shape()
        let fusedKernel = "test_unfused_compare_attn_fused_sigmoid_o"
        let gateKernel = "test_unfused_packed_sigmoid_gate_seq_f32"
        let gemvKernel = "test_unfused_attn_o_gemv_seq_bf16_f32s"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generatePackedSigmoidGate(
                name: gateKernel,
                bufferPrecision: .float32,
                isSequence: true
            ),
            MetalSourceGenerator.generateSequenceGEMV(
                name: gemvKernel,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            ),
            MetalSourceGenerator.generateFusedPackedSigmoidGateOutputSequenceGEMV(
                name: fusedKernel,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let gatePipeline = try harness.pipeline(named: gateKernel)
        let gemvPipeline = try harness.pipeline(named: gemvKernel)
        let fusedPipeline = try harness.pipeline(named: fusedKernel)

        let input = makeInput(shape: shape)
        let packed = makePacked(shape: shape)
        let weight = makeWeight(shape: shape)
        let unfused = try runUnfusedPath(
            harness: harness,
            gatePipeline: gatePipeline,
            gemvPipeline: gemvPipeline,
            input: input,
            packed: packed,
            weight: weight,
            shape: shape
        )
        let fused = try runFusedKernel(
            harness: harness,
            pipeline: fusedPipeline,
            input: input,
            packed: packed,
            weight: weight,
            shape: shape
        )

        let mismatch = harness.firstMismatch(expected: unfused, actual: fused, tolerance: 0.0)
        let maxError = harness.maxAbsoluteError(expected: unfused, actual: fused)
        #expect(
            mismatch == nil,
            "fused packed sigmoid output drifted from unfused path: \(String(describing: mismatch)), maxError=\(maxError)"
        )
    }

    private struct Shape {
        let inputDimension = 256
        let outputDimension = 192
        let headDimension = 64
        let packedHeadStride = 192
        let gateHeadOffset = 128
        let sequenceLength = 8
        let inputRowStride = 384
        let packedRowStride = 832
        let outputRowStride = 256
    }

    private func makeInput(shape: Shape) -> [Float] {
        var values = [Float](repeating: 0, count: shape.sequenceLength * shape.inputRowStride)
        for seq in 0..<shape.sequenceLength {
            for index in 0..<shape.inputDimension {
                values[seq * shape.inputRowStride + index] =
                    Float((seq * 17 + index * 5) % 37 - 18) * 0.03125
            }
        }
        return values
    }

    private func makePacked(shape: Shape) -> [Float] {
        var values = [Float](repeating: 0, count: shape.sequenceLength * shape.packedRowStride)
        let headCount = shape.inputDimension / shape.headDimension
        for seq in 0..<shape.sequenceLength {
            for head in 0..<headCount {
                for lane in 0..<shape.headDimension {
                    let index = seq * shape.packedRowStride
                        + head * shape.packedHeadStride
                        + shape.gateHeadOffset
                        + lane
                    values[index] = Float((seq * 11 + head * 13 + lane * 7) % 29 - 14) * 0.0625
                }
            }
        }
        return values
    }

    private func makeWeight(shape: Shape) -> [BFloat16] {
        (0..<(shape.outputDimension * shape.inputDimension)).map { index in
            BFloat16(Float((index * 19 + 3) % 31 - 15) * 0.015625)
        }
    }

    private func swiftReference(
        input: [Float],
        packed: [Float],
        weight: [BFloat16],
        shape: Shape
    ) -> [Float] {
        var output = [Float](repeating: 0, count: shape.sequenceLength * shape.outputRowStride)
        let tileElements = 256
        for seq in 0..<shape.sequenceLength {
            for row in 0..<shape.outputDimension {
                var sum: Float = 0
                var base = 0
                while base < shape.inputDimension {
                    let tileCount = min(tileElements, shape.inputDimension - base)
                    for tileIndex in 0..<tileCount {
                        let inputIndex = base + tileIndex
                        let headIndex = inputIndex / shape.headDimension
                        let lane = inputIndex % shape.headDimension
                        let gateIndex = seq * shape.packedRowStride
                            + headIndex * shape.packedHeadStride
                            + shape.gateHeadOffset
                            + lane
                        let x = input[seq * shape.inputRowStride + inputIndex]
                        let g = packed[gateIndex]
                        let gated = x * (1.0 / (1.0 + expf(-g)))
                        let w = Float(weight[row * shape.inputDimension + inputIndex])
                        sum += w * gated
                    }
                    base += tileElements
                }
                output[seq * shape.outputRowStride + row] = Float(BFloat16(sum))
            }
        }
        return output
    }

    private func runFusedKernel(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        input: [Float],
        packed: [Float],
        weight: [BFloat16],
        shape: Shape
    ) throws -> [Float] {
        let inputBuffer = try harness.makeSharedBuffer(values: input)
        let packedBuffer = try harness.makeSharedBuffer(values: packed)
        let weightBuffer = try harness.makeSharedBuffer(values: weight)
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: shape.sequenceLength * shape.outputRowStride * MemoryLayout<Float>.stride
        )
        let simdWidth = 32
        let rowsPerThreadgroup = 2
        let threads = min(simdWidth * rowsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup)
        let grid = MTLSize(
            width: (shape.outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: shape.sequenceLength,
            depth: 1
        )

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(packedBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        bindFusedConstants(encoder, shape: shape)
        encoder.dispatchThreadgroups(
            grid,
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        return harness.readFloat32(outputBuffer, count: shape.sequenceLength * shape.outputRowStride)
    }

    private func runUnfusedPath(
        harness: SequenceKernelEquivalenceHarness,
        gatePipeline: MTLComputePipelineState,
        gemvPipeline: MTLComputePipelineState,
        input: [Float],
        packed: [Float],
        weight: [BFloat16],
        shape: Shape
    ) throws -> [Float] {
        let inputBuffer = try harness.makeSharedBuffer(values: input)
        let packedBuffer = try harness.makeSharedBuffer(values: packed)
        let weightBuffer = try harness.makeSharedBuffer(values: weight)
        let gatedBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: shape.sequenceLength * shape.inputRowStride * MemoryLayout<Float>.stride
        )
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: shape.sequenceLength * shape.outputRowStride * MemoryLayout<Float>.stride
        )

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(gatePipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(packedBuffer, offset: 0, index: 1)
        encoder.setBuffer(gatedBuffer, offset: 0, index: 2)
        bindGateConstants(encoder, shape: shape)
        let gateThreads = min(256, gatePipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (shape.inputDimension + gateThreads - 1) / gateThreads,
                height: shape.sequenceLength,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: gateThreads, height: 1, depth: 1)
        )

        encoder.setComputePipelineState(gemvPipeline)
        encoder.setBuffer(gatedBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        bindGEMVConstants(encoder, shape: shape)
        let simdWidth = 32
        let gemvRowsPerThreadgroup = 2
        let gemvThreads = min(simdWidth * gemvRowsPerThreadgroup, gemvPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (shape.outputDimension + gemvRowsPerThreadgroup - 1) / gemvRowsPerThreadgroup,
                height: shape.sequenceLength,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: gemvThreads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        return harness.readFloat32(outputBuffer, count: shape.sequenceLength * shape.outputRowStride)
    }

    private func bindFusedConstants(_ encoder: MTLComputeCommandEncoder, shape: Shape) {
        bindUInt32(encoder, shape.inputDimension, 4)
        bindUInt32(encoder, shape.outputDimension, 5)
        bindUInt32(encoder, shape.headDimension, 6)
        bindUInt32(encoder, shape.packedHeadStride, 7)
        bindUInt32(encoder, shape.gateHeadOffset, 8)
        bindUInt32(encoder, shape.sequenceLength, 9)
        bindUInt32(encoder, shape.inputRowStride, 10)
        bindUInt32(encoder, shape.packedRowStride, 11)
        bindUInt32(encoder, shape.outputRowStride, 12)
    }

    private func bindGateConstants(_ encoder: MTLComputeCommandEncoder, shape: Shape) {
        bindUInt32(encoder, shape.inputDimension, 3)
        bindUInt32(encoder, shape.headDimension, 4)
        bindUInt32(encoder, shape.packedHeadStride, 5)
        bindUInt32(encoder, shape.gateHeadOffset, 6)
        bindUInt32(encoder, shape.packedRowStride, 7)
        bindUInt32(encoder, shape.inputRowStride, 8)
        bindUInt32(encoder, shape.sequenceLength, 9)
    }

    private func bindGEMVConstants(_ encoder: MTLComputeCommandEncoder, shape: Shape) {
        bindUInt32(encoder, shape.inputDimension, 3)
        bindUInt32(encoder, shape.outputDimension, 4)
        bindUInt32(encoder, shape.sequenceLength, 5)
        bindUInt32(encoder, shape.inputRowStride, 6)
        bindUInt32(encoder, shape.outputRowStride, 7)
    }

    private func bindUInt32(_ encoder: MTLComputeCommandEncoder, _ value: Int, _ index: Int) {
        var constant = UInt32(value)
        encoder.setBytes(&constant, length: MemoryLayout<UInt32>.stride, index: index)
    }
}
