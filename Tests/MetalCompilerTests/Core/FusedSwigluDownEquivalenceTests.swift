import Foundation
import Metal
import Testing
@testable import MetalCompiler

/// Admission tests for the fused SwiGLU + down-projection kernel.
///
/// These tests pin down the numerical contract of
/// `mlp_fused_swiglu_down_seq_bf16_f32s` before any prefill routing is wired up.
///
/// The fused kernel intentionally diverges from the unfused `swiglu_seq_f32 +
/// gemv_seq_bf16_f32s` pair: the contract requires the SwiGLU intermediate
/// `silu(g) * up` to be BF16-rounded before participating in the GEMV reduction,
/// while the unfused two-kernel path keeps the intermediate in F32. The reference
/// computed here applies the rounding explicitly so the admission test directly
/// validates the contract.
@Suite("Fused SwiGLU+Down Equivalence", .serialized)
struct FusedSwigluDownEquivalenceTests {
    @Test("Fused SwiGLU+down BF16 matches Swift reference with explicit BF16 rounding")
    func fusedSwigluDownMatchesSwiftReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        // Representative shape with non-trivial input row stride to exercise
        // scratch slot padding (the production prefill scratch buffer uses
        // slotDimension >= intermediateDim).
        let intermediateDim = 1024
        let outputDim = 256
        let sequenceLength = 16
        let inputRowStride = 1280
        let outputRowStride = outputDim
        let fusedKernel = "test_mlp_fused_swiglu_down_seq_bf16_f32s"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateFusedSwigluDownSequenceGEMV(
                name: fusedKernel,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let pipeline = try harness.pipeline(named: fusedKernel)

        let gateValues = makeGateValues(
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            intermediateDim: intermediateDim
        )
        let upValues = makeUpValues(
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            intermediateDim: intermediateDim
        )
        let weightValues = makeWeightValues(
            outputDim: outputDim,
            intermediateDim: intermediateDim
        )

        let actual = try runFusedKernel(
            harness: harness,
            pipeline: pipeline,
            gate: gateValues,
            up: upValues,
            weight: weightValues,
            intermediateDim: intermediateDim,
            outputDim: outputDim,
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            outputRowStride: outputRowStride
        )
        let expected = swiftReference(
            gate: gateValues,
            up: upValues,
            weight: weightValues,
            intermediateDim: intermediateDim,
            outputDim: outputDim,
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            outputRowStride: outputRowStride
        )

        let mismatch = harness.firstMismatch(
            expected: expected,
            actual: actual,
            tolerance: 0.0
        )
        let maxError = harness.maxAbsoluteError(expected: expected, actual: actual)
        #expect(
            mismatch == nil,
            "fused SwiGLU+down drifted from BF16-rounded Swift reference: \(String(describing: mismatch)), maxError=\(maxError)"
        )
    }

    @Test("Fused SwiGLU+down BF16 quantifies divergence from unfused two-kernel path")
    func fusedSwigluDownDivergesFromUnfusedPath() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let intermediateDim = 1024
        let outputDim = 256
        let sequenceLength = 16
        let inputRowStride = 1280
        let outputRowStride = outputDim
        let fusedKernel = "test_unfused_compare_fused_swiglu_down"
        let swigluKernel = "test_unfused_swiglu_seq_f32"
        let gemvKernel = "test_unfused_gemv_seq_bf16_f32s"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateSwiGLU(
                name: swigluKernel,
                bufferPrecision: .float32
            ),
            MetalSourceGenerator.generateSequenceGEMV(
                name: gemvKernel,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            ),
            MetalSourceGenerator.generateFusedSwigluDownSequenceGEMV(
                name: fusedKernel,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let swigluPipeline = try harness.pipeline(named: swigluKernel)
        let gemvPipeline = try harness.pipeline(named: gemvKernel)
        let fusedPipeline = try harness.pipeline(named: fusedKernel)

        let gateValues = makeGateValues(
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            intermediateDim: intermediateDim
        )
        let upValues = makeUpValues(
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            intermediateDim: intermediateDim
        )
        let weightValues = makeWeightValues(
            outputDim: outputDim,
            intermediateDim: intermediateDim
        )

        let unfused = try runUnfusedPath(
            harness: harness,
            swigluPipeline: swigluPipeline,
            gemvPipeline: gemvPipeline,
            gate: gateValues,
            up: upValues,
            weight: weightValues,
            intermediateDim: intermediateDim,
            outputDim: outputDim,
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            outputRowStride: outputRowStride
        )
        let fused = try runFusedKernel(
            harness: harness,
            pipeline: fusedPipeline,
            gate: gateValues,
            up: upValues,
            weight: weightValues,
            intermediateDim: intermediateDim,
            outputDim: outputDim,
            sequenceLength: sequenceLength,
            inputRowStride: inputRowStride,
            outputRowStride: outputRowStride
        )

        let maxAbsError = harness.maxAbsoluteError(expected: unfused, actual: fused)
        let unfusedMagnitude = unfused.reduce(0.0 as Float) { partial, value in
            max(partial, abs(value))
        }
        let relative = unfusedMagnitude > 0 ? maxAbsError / unfusedMagnitude : maxAbsError
        // The contract permits per-element BF16 rounding noise on the SwiGLU
        // intermediate, so the divergence accumulates over `intermediateDim`
        // multiply-adds. A loose envelope (1% relative) catches gross routing
        // bugs without overconstraining the documented numerical drift.
        #expect(
            relative < 0.01,
            "fused vs unfused divergence exceeds rounding envelope: maxAbsError=\(maxAbsError), unfusedMagnitude=\(unfusedMagnitude), relative=\(relative)"
        )
    }

    // MARK: - Inputs

    private func makeGateValues(
        sequenceLength: Int,
        inputRowStride: Int,
        intermediateDim: Int
    ) -> [Float] {
        var values = [Float](repeating: 0, count: sequenceLength * inputRowStride)
        for seq in 0..<sequenceLength {
            for j in 0..<intermediateDim {
                let raw = Float((seq * 13 + j * 7) % 31 - 15) * 0.0625
                values[seq * inputRowStride + j] = raw
            }
        }
        return values
    }

    private func makeUpValues(
        sequenceLength: Int,
        inputRowStride: Int,
        intermediateDim: Int
    ) -> [Float] {
        var values = [Float](repeating: 0, count: sequenceLength * inputRowStride)
        for seq in 0..<sequenceLength {
            for j in 0..<intermediateDim {
                let raw = Float((seq * 19 + j * 11) % 29 - 14) * 0.05
                values[seq * inputRowStride + j] = raw
            }
        }
        return values
    }

    private func makeWeightValues(outputDim: Int, intermediateDim: Int) -> [BFloat16] {
        (0..<(outputDim * intermediateDim)).map { index in
            BFloat16(Float((index * 17 + 5) % 23 - 11) * 0.03125)
        }
    }

    // MARK: - Swift reference

    private func swiftReference(
        gate: [Float],
        up: [Float],
        weight: [BFloat16],
        intermediateDim: Int,
        outputDim: Int,
        sequenceLength: Int,
        inputRowStride: Int,
        outputRowStride: Int
    ) -> [Float] {
        var output = [Float](repeating: 0, count: sequenceLength * outputRowStride)
        // The fused kernel processes the GEMV in `tileElements`-wide tiles and
        // accumulates `tile[j] * weight[j]` inside each tile sequentially before
        // moving to the next tile. The Swift reference must mirror this tile
        // structure exactly because float addition is non-associative — a flat
        // 0..<intermediateDim accumulation would diverge by ULP-level noise.
        let tileElements = 256
        for seq in 0..<sequenceLength {
            for row in 0..<outputDim {
                var sum: Float = 0
                var base = 0
                while base < intermediateDim {
                    let tileCount = min(tileElements, intermediateDim - base)
                    for j in 0..<tileCount {
                        let idx = base + j
                        let g = gate[seq * inputRowStride + idx]
                        let u = up[seq * inputRowStride + idx]
                        let siluUp = g * (1.0 / (1.0 + Foundation_exp(-g))) * u
                        let rounded = Float(BFloat16(siluUp))
                        let w = Float(weight[row * intermediateDim + idx])
                        sum += w * rounded
                    }
                    base += tileElements
                }
                let storedSum = Float(BFloat16(sum))
                output[seq * outputRowStride + row] = storedSum
            }
        }
        return output
    }

    // MARK: - Kernel dispatch helpers

    private func runFusedKernel(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        gate: [Float],
        up: [Float],
        weight: [BFloat16],
        intermediateDim: Int,
        outputDim: Int,
        sequenceLength: Int,
        inputRowStride: Int,
        outputRowStride: Int
    ) throws -> [Float] {
        let gateBuffer = try harness.makeSharedBuffer(values: gate)
        let upBuffer = try harness.makeSharedBuffer(values: up)
        let weightBuffer = try harness.makeSharedBuffer(values: weight)
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: sequenceLength * outputRowStride * MemoryLayout<Float>.stride
        )
        let simdWidth = 32
        let threads = min(simdWidth * 2, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / simdWidth)
        let grid = MTLSize(
            width: (outputDim + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: sequenceLength,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        var intermediate = UInt32(intermediateDim)
        var output = UInt32(outputDim)
        var seqLen = UInt32(sequenceLength)
        var inputStride = UInt32(inputRowStride)
        var outputStride = UInt32(outputRowStride)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&output, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&inputStride, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        return harness.readFloat32(outputBuffer, count: sequenceLength * outputRowStride)
    }

    private func runUnfusedPath(
        harness: SequenceKernelEquivalenceHarness,
        swigluPipeline: MTLComputePipelineState,
        gemvPipeline: MTLComputePipelineState,
        gate: [Float],
        up: [Float],
        weight: [BFloat16],
        intermediateDim: Int,
        outputDim: Int,
        sequenceLength: Int,
        inputRowStride: Int,
        outputRowStride: Int
    ) throws -> [Float] {
        let gateBuffer = try harness.makeSharedBuffer(values: gate)
        let upBuffer = try harness.makeSharedBuffer(values: up)
        let weightBuffer = try harness.makeSharedBuffer(values: weight)
        let scratchBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: sequenceLength * inputRowStride * MemoryLayout<Float>.stride
        )
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: sequenceLength * outputRowStride * MemoryLayout<Float>.stride
        )

        // SwiGLU: 2D grid (intermediate, sequenceLength).
        let swigluThreads = min(256, swigluPipeline.maxTotalThreadsPerThreadgroup)
        let swigluGrid = MTLSize(
            width: (intermediateDim + swigluThreads - 1) / swigluThreads,
            height: sequenceLength,
            depth: 1
        )
        let swigluThreadgroup = MTLSize(width: swigluThreads, height: 1, depth: 1)

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(swigluPipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(scratchBuffer, offset: 0, index: 2)
        var swigluDimension = UInt32(intermediateDim)
        var swigluSeqLen = UInt32(sequenceLength)
        var swigluRowStride = UInt32(inputRowStride)
        encoder.setBytes(&swigluDimension, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&swigluSeqLen, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&swigluRowStride, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.dispatchThreadgroups(swigluGrid, threadsPerThreadgroup: swigluThreadgroup)

        // GEMV: SIMD-row layout matching `runFusedKernel`.
        let simdWidth = 32
        let gemvThreads = min(simdWidth * 2, gemvPipeline.maxTotalThreadsPerThreadgroup)
        let gemvRowsPerThreadgroup = max(1, gemvThreads / simdWidth)
        let gemvGrid = MTLSize(
            width: (outputDim + gemvRowsPerThreadgroup - 1) / gemvRowsPerThreadgroup,
            height: sequenceLength,
            depth: 1
        )
        let gemvThreadgroup = MTLSize(width: gemvThreads, height: 1, depth: 1)
        encoder.setComputePipelineState(gemvPipeline)
        encoder.setBuffer(scratchBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var gemvInputDim = UInt32(intermediateDim)
        var gemvOutputDim = UInt32(outputDim)
        var gemvSeqLen = UInt32(sequenceLength)
        var gemvInputStride = UInt32(inputRowStride)
        var gemvOutputStride = UInt32(outputRowStride)
        encoder.setBytes(&gemvInputDim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&gemvOutputDim, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&gemvSeqLen, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&gemvInputStride, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&gemvOutputStride, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(gemvGrid, threadsPerThreadgroup: gemvThreadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        return harness.readFloat32(outputBuffer, count: sequenceLength * outputRowStride)
    }
}

/// Float exponential helper that matches the Metal `exp` semantics closely
/// enough for the BF16-rounded reference. We use `expf` directly so the call
/// site can stay in `Float` without intermediate `Double` widening.
@inline(__always)
private func Foundation_exp(_ value: Float) -> Float {
    expf(value)
}
