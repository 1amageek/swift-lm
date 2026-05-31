import Metal
import Testing
import LMIR
@testable import MetalCompiler

@Suite("Conv1d Sequence Equivalence", .serialized)
struct Conv1dSequenceEquivalenceTests {
    @Test("BF16 fused ShortConv decode matches unfused projection plus state update")
    func bf16FusedShortConvDecodeMatchesUnfusedProjectionAndStateUpdate() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let source = [
            MetalSourceGenerator.commonHeader,
            Self.shortConvInProjKernelSource(name: "test_in_proj_bf16"),
            MetalSourceGenerator.generateConvStateUpdate(
                name: "test_conv_state_update_bf16",
                bufferPrecision: .bfloat16,
                weightFormat: .bfloat16
            ),
            MetalSourceGenerator.generateShortConvInProjUpdateBF16(
                name: "test_shortconv_inproj_update_bf16"
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let inProjPipeline = try harness.pipeline(named: "test_in_proj_bf16")
        let convPipeline = try harness.pipeline(named: "test_conv_state_update_bf16")
        let fusedPipeline = try harness.pipeline(named: "test_shortconv_inproj_update_bf16")

        let dimension = 2_048
        let projectedDimension = dimension * 3
        let kernelSize = 3
        let inputValues = (0..<dimension).map { index in
            BFloat16(Float((index * 17) % 31 - 15) * 0.0078125)
        }
        let projectionWeights = (0..<projectedDimension * dimension).map { index in
            BFloat16(Float((index * 13) % 29 - 14) * 0.00390625)
        }
        let convWeights = (0..<dimension * kernelSize).map { index in
            BFloat16(Float((index * 7) % 19 - 9) * 0.015625)
        }
        let initialState = (0..<dimension * kernelSize).map { index in
            BFloat16(Float((index * 11) % 23 - 11) * 0.00390625)
        }

        let inputBuffer = try harness.makeSharedBuffer(values: inputValues)
        let projectionWeightBuffer = try harness.makeSharedBuffer(values: projectionWeights)
        let convWeightBuffer = try harness.makeSharedBuffer(values: convWeights)
        let unfusedStateBuffer = try harness.makeSharedBuffer(values: initialState)
        let fusedStateBuffer = try harness.makeSharedBuffer(values: initialState)
        let projectedBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: projectedDimension * MemoryLayout<BFloat16>.stride
        )
        let unfusedOutputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: dimension * MemoryLayout<BFloat16>.stride
        )
        let fusedOutputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: dimension * MemoryLayout<BFloat16>.stride
        )

        try runInProj(
            harness: harness,
            pipeline: inProjPipeline,
            inputBuffer: inputBuffer,
            weightBuffer: projectionWeightBuffer,
            outputBuffer: projectedBuffer,
            inputDimension: dimension,
            outputDimension: projectedDimension
        )
        try runConvUpdate(
            harness: harness,
            pipeline: convPipeline,
            convState: unfusedStateBuffer,
            inputBuffer: projectedBuffer,
            weightBuffer: convWeightBuffer,
            outputBuffer: unfusedOutputBuffer,
            dimension: dimension,
            kernelSize: kernelSize
        )
        try runFusedShortConv(
            harness: harness,
            pipeline: fusedPipeline,
            inputBuffer: inputBuffer,
            projectionWeightBuffer: projectionWeightBuffer,
            convState: fusedStateBuffer,
            convWeightBuffer: convWeightBuffer,
            outputBuffer: fusedOutputBuffer,
            dimension: dimension,
            kernelSize: kernelSize
        )

        let unfusedOutputBits = harness.readBFloat16Bits(unfusedOutputBuffer, count: dimension)
        let fusedOutputBits = harness.readBFloat16Bits(fusedOutputBuffer, count: dimension)
        let unfusedStateBits = harness.readBFloat16Bits(unfusedStateBuffer, count: dimension * kernelSize)
        let fusedStateBits = harness.readBFloat16Bits(fusedStateBuffer, count: dimension * kernelSize)

        #expect(
            fusedOutputBits == unfusedOutputBits,
            "fused ShortConv output drifted from unfused projection+conv"
        )
        #expect(
            fusedStateBits == unfusedStateBits,
            "fused ShortConv state drifted from unfused projection+conv"
        )
    }

    @Test("BF16 ShortConv fusion admission rejects cross-composite and non-row-major weights")
    func bf16ShortConvFusionAdmissionRejectsUnsafeRoutes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try SequenceKernelEquivalenceHarness(device: device)
        let pipelineCache = try makeShortConvAdmissionPipelineCache(harness: harness)
        let rowMajorStore = try makeShortConvAdmissionWeightStore(device: device, blockedInProj: false)
        let blockedStore = try makeShortConvAdmissionWeightStore(device: device, blockedInProj: true)
        let rowMajorResolver = ProjectionWeightAccessPolicyResolver()
        let blockedResolver = ProjectionWeightAccessPolicyResolver(
            override: ProjectionWeightAccessPolicyOverride { context in
                guard context.role == "in_proj" else { return nil }
                return .optimized(.blockedRows8Tiles128)
            }
        )

        let admitted = try makeShortConvAdmissionPlan(
            device: device,
            pipelineCache: pipelineCache,
            inProjCompositeID: 7,
            convCompositeID: 7,
            weightStore: rowMajorStore,
            accessPolicyResolver: rowMajorResolver
        )
        #expect(admitted.kernelNames == ["shortconv_inproj_update_bf16"])

        let crossed = try makeShortConvAdmissionPlan(
            device: device,
            pipelineCache: pipelineCache,
            inProjCompositeID: 7,
            convCompositeID: 8,
            weightStore: rowMajorStore,
            accessPolicyResolver: rowMajorResolver
        )
        #expect(!crossed.kernelNames.contains("shortconv_inproj_update_bf16"))
        #expect(crossed.kernelNames.contains("gemv_2048_6144_bf16"))
        #expect(crossed.kernelNames.contains("conv_state_update_bf16"))

        let blocked = try makeShortConvAdmissionPlan(
            device: device,
            pipelineCache: pipelineCache,
            inProjCompositeID: 7,
            convCompositeID: 7,
            weightStore: blockedStore,
            accessPolicyResolver: blockedResolver
        )
        #expect(!blocked.kernelNames.contains("shortconv_inproj_update_bf16"))
        #expect(blocked.kernelNames.contains("gemv_2048_6144_bf16"))
        #expect(blocked.kernelNames.contains("conv_state_update_bf16"))
        #expect(
            blocked.plan.quantizationPlan.entries.contains { entry in
                entry.tensorName == Self.inProjTensorName && entry.layout == .blockedRows8Tiles128
            }
        )
    }

    @Test("BF16 conv sequence kernels match repeated decode state updates")
    func bf16ConvSequenceKernelsMatchRepeatedDecodeStateUpdates() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try SequenceKernelEquivalenceHarness(device: device)
        let decodePipeline = try harness.pipeline(named: "conv_state_update_bf16")
        let sequencePipeline = try harness.pipeline(named: "conv1d_causal_seq_f32")
        let extractPipeline = try harness.pipeline(named: "extract_conv_state_f32")

        let convDimension = 8
        let inputProjectionDimension = convDimension * 3
        let kernelSize = 4
        let sequenceLength = 6
        let inputValues = (0..<(sequenceLength * inputProjectionDimension)).map { index in
            Float(Float16(Float((index * 19) % 23 - 11) * 0.0625))
        }
        let weights = (0..<(convDimension * kernelSize)).map { index in
            BFloat16(Float((index * 7) % 17 - 8) * 0.03125)
        }

        let decode = try runDecodeConvTrace(
            harness: harness,
            pipeline: decodePipeline,
            inputValues: inputValues,
            weights: weights,
            convDimension: convDimension,
            inputProjectionDimension: inputProjectionDimension,
            kernelSize: kernelSize,
            sequenceLength: sequenceLength
        )
        let sequence = try runSequenceConvTrace(
            harness: harness,
            sequencePipeline: sequencePipeline,
            extractPipeline: extractPipeline,
            inputValues: inputValues,
            weights: weights,
            convDimension: convDimension,
            inputProjectionDimension: inputProjectionDimension,
            kernelSize: kernelSize,
            sequenceLength: sequenceLength
        )

        let roundedSequenceOutput = sequence.output.map { Float(Float16($0)) }
        let mismatch = harness.firstMismatch(
            expected: decode.output,
            actual: roundedSequenceOutput,
            tolerance: 0.000_001
        )
        #expect(
            mismatch == nil,
            "conv output drifted: \(String(describing: mismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.output, actual: roundedSequenceOutput))"
        )
        #expect(
            decode.convStateBits == sequence.convStateBits,
            "conv state drifted: decode=\(decode.convStateBits), sequence=\(sequence.convStateBits)"
        )
    }

    private func runDecodeConvTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [BFloat16],
        convDimension: Int,
        inputProjectionDimension: Int,
        kernelSize: Int,
        sequenceLength: Int
    ) throws -> (output: [Float], convStateBits: [UInt16]) {
        let convState = try harness.makeZeroedSharedBuffer(
            byteLength: convDimension * kernelSize * MemoryLayout<BFloat16>.stride
        )
        let weightBuffer = try harness.makeSharedBuffer(values: weights)
        var trace = [Float](repeating: .zero, count: sequenceLength * convDimension)
        let threads = min(max(pipeline.threadExecutionWidth, 32), pipeline.maxTotalThreadsPerThreadgroup)
        let grid = MTLSize(
            width: (convDimension + threads - 1) / threads,
            height: 1,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        for position in 0..<sequenceLength {
            let inputStart = position * inputProjectionDimension
            let tokenInput = inputValues[inputStart..<(inputStart + inputProjectionDimension)].map {
                Float16($0)
            }
            let inputBuffer = try harness.makeSharedBuffer(values: Array(tokenInput))
            let outputBuffer = try harness.makeZeroedSharedBuffer(
                byteLength: convDimension * MemoryLayout<Float16>.stride
            )
            let (commandBuffer, encoder) = try harness.makeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(convState, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputBuffer, offset: 0, index: 3)
            var dimension = UInt32(convDimension)
            var kernel = UInt32(kernelSize)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
            encoder.endEncoding()
            try harness.complete(commandBuffer)

            let output = harness.readFloat16AsFloat(outputBuffer, count: convDimension)
            let offset = position * convDimension
            trace.replaceSubrange(offset..<(offset + convDimension), with: output)
        }
        return (
            output: trace,
            convStateBits: harness.readBFloat16Bits(
                convState,
                count: convDimension * kernelSize
            )
        )
    }

    private func runSequenceConvTrace(
        harness: SequenceKernelEquivalenceHarness,
        sequencePipeline: MTLComputePipelineState,
        extractPipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [BFloat16],
        convDimension: Int,
        inputProjectionDimension: Int,
        kernelSize: Int,
        sequenceLength: Int
    ) throws -> (output: [Float], convStateBits: [UInt16]) {
        let inputBuffer = try harness.makeSharedBuffer(values: inputValues)
        let weightBuffer = try harness.makeSharedBuffer(values: weights)
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: sequenceLength * convDimension * MemoryLayout<Float>.stride
        )
        let convState = try harness.makeZeroedSharedBuffer(
            byteLength: convDimension * kernelSize * MemoryLayout<BFloat16>.stride
        )
        var convDim = UInt32(convDimension)
        var inputProjDim = UInt32(inputProjectionDimension)
        var kernel = UInt32(kernelSize)
        var seqLen = UInt32(sequenceLength)
        // Native packed layout for the unit test: input rows stride at
        // inputProjectionDimension, output rows stride at convDimension.
        var inputRowStride = UInt32(inputProjectionDimension)
        var outputRowStride = UInt32(convDimension)

        let sequenceThreads = MTLSize(width: 8, height: 1, depth: 1)
        let sequenceGrid = MTLSize(
            width: (convDimension + sequenceThreads.width - 1) / sequenceThreads.width,
            height: sequenceLength,
            depth: 1
        )
        let (sequenceCommandBuffer, sequenceEncoder) = try harness.makeCommandEncoder()
        sequenceEncoder.setComputePipelineState(sequencePipeline)
        sequenceEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        sequenceEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
        sequenceEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
        sequenceEncoder.setBytes(&convDim, length: MemoryLayout<UInt32>.stride, index: 3)
        sequenceEncoder.setBytes(&inputProjDim, length: MemoryLayout<UInt32>.stride, index: 4)
        sequenceEncoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 5)
        sequenceEncoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
        sequenceEncoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: 7)
        sequenceEncoder.setBytes(&outputRowStride, length: MemoryLayout<UInt32>.stride, index: 8)
        sequenceEncoder.dispatchThreadgroups(sequenceGrid, threadsPerThreadgroup: sequenceThreads)
        sequenceEncoder.endEncoding()
        try harness.complete(sequenceCommandBuffer)

        let extractThreads = MTLSize(width: 8, height: 1, depth: 1)
        let extractGrid = MTLSize(
            width: (convDimension + extractThreads.width - 1) / extractThreads.width,
            height: kernelSize,
            depth: 1
        )
        let (extractCommandBuffer, extractEncoder) = try harness.makeCommandEncoder()
        extractEncoder.setComputePipelineState(extractPipeline)
        extractEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        extractEncoder.setBuffer(convState, offset: 0, index: 1)
        extractEncoder.setBytes(&convDim, length: MemoryLayout<UInt32>.stride, index: 2)
        extractEncoder.setBytes(&inputProjDim, length: MemoryLayout<UInt32>.stride, index: 3)
        extractEncoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 4)
        extractEncoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
        extractEncoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: 6)
        extractEncoder.dispatchThreadgroups(extractGrid, threadsPerThreadgroup: extractThreads)
        extractEncoder.endEncoding()
        try harness.complete(extractCommandBuffer)

        return (
            output: harness.readFloat32(
                outputBuffer,
                count: sequenceLength * convDimension
            ),
            convStateBits: harness.readBFloat16Bits(
                convState,
                count: convDimension * kernelSize
            )
        )
    }

    private static let inProjTensorName = "model.layers.0.conv.in_proj.weight"
    private static let convTensorName = "model.layers.0.conv.conv_weight"

    private static func shortConvInProjKernelSource(name: String) -> String {
        let policy = Input2048GEMVSourcePolicy.expanded6144(weightFormat: .bfloat16)
        return MetalSourceGenerator.generateInput2048GEMV(
            name: name,
            bufferPrecision: .bfloat16,
            weightFormat: .bfloat16,
            fixedOutputDimension: policy.fixedOutputDimension,
            fixedRowsPerThreadgroup: policy.fixedRowsPerThreadgroup,
            fixedSimdgroups: policy.fixedSimdgroups,
            stagesInputAsFloat: policy.stagesInputAsFloat,
            weightLayoutPolicy: policy.weightLayoutPolicy,
            unrollFactor: policy.unrollFactor
        )
    }

    private func runInProj(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        inputDimension: Int,
        outputDimension: Int
    ) throws {
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let simdgroups = 8
        let threadgroup = MTLSize(width: simdWidth * simdgroups, height: 1, depth: 1)
        let grid = MTLSize(width: (outputDimension + 7) / 8, height: 1, depth: 1)
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
    }

    private func runConvUpdate(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        convState: MTLBuffer,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        dimension: Int,
        kernelSize: Int
    ) throws {
        let threads = min(max(pipeline.threadExecutionWidth, 32), pipeline.maxTotalThreadsPerThreadgroup)
        let grid = MTLSize(width: (dimension + threads - 1) / threads, height: 1, depth: 1)
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)
        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(convState, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        var convDim = UInt32(dimension)
        var kernel = UInt32(kernelSize)
        encoder.setBytes(&convDim, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)
    }

    private func runFusedShortConv(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        projectionWeightBuffer: MTLBuffer,
        convState: MTLBuffer,
        convWeightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        dimension: Int,
        kernelSize: Int
    ) throws {
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let simdgroups = max(1, min(16, pipeline.maxTotalThreadsPerThreadgroup / simdWidth))
        let threads = simdgroups * simdWidth
        let grid = MTLSize(width: (dimension + simdgroups - 1) / simdgroups, height: 1, depth: 1)
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)
        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(projectionWeightBuffer, offset: 0, index: 1)
        encoder.setBuffer(convState, offset: 0, index: 2)
        encoder.setBuffer(convWeightBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer, offset: 0, index: 4)
        var convDim = UInt32(dimension)
        var kernel = UInt32(kernelSize)
        encoder.setBytes(&convDim, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)
    }

    private func makeShortConvAdmissionPipelineCache(
        harness: SequenceKernelEquivalenceHarness
    ) throws -> [String: MTLComputePipelineState] {
        let names = [
            "gemv_2048_6144_bf16",
            "conv_state_update_bf16",
            "shortconv_inproj_update_bf16",
        ]
        var pipelines: [String: MTLComputePipelineState] = [:]
        for name in names {
            pipelines[name] = try harness.pipeline(named: name)
        }
        return pipelines
    }

    private func makeShortConvAdmissionPlan(
        device: MTLDevice,
        pipelineCache: [String: MTLComputePipelineState],
        inProjCompositeID: Int,
        convCompositeID: Int,
        weightStore: STAFWeightStore,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    ) throws -> (plan: MetalDispatchPlan, kernelNames: [String]) {
        let builder = MetalDispatchStepBuilder()
        let compileContext = CompileContext(
            graph: ModelGraph(rootRegion: Region()),
            hiddenSize: 2_048,
            intermediateSize: 0,
            vocabSize: 1,
            inferencePolicy: .default,
            stafWeightStore: weightStore,
            device: device,
            weightFormat: WeightFormats.bfloat16,
            decodeBufferPrecision: .bfloat16,
            accessPolicyResolver: accessPolicyResolver
        )
        let planBuildContext = PlanBuildContext(
            compileContext: compileContext,
            kernelContext: KernelContext(bufferPrecision: .bfloat16, weightFormat: WeightFormats.bfloat16),
            pipelineCache: pipelineCache,
            quantizationCapabilities: .none,
            dispatchHeuristics: DispatchHeuristics()
        )
        let entries = [
            DispatchEntry(
                index: 0,
                fragment: LinearFragment(field: "in_proj", inputDimension: 2_048, outputDimension: 6_144),
                parameterBindings: [ParameterBinding(role: "in_proj", tensorName: Self.inProjTensorName)],
                layerIndex: 0,
                compositeID: inProjCompositeID
            ),
            DispatchEntry(
                index: 1,
                fragment: Conv1dFragment(dimension: 2_048, kernelSize: 3),
                parameterBindings: [ParameterBinding(role: "conv_weight", tensorName: Self.convTensorName)],
                layerIndex: 0,
                compositeID: convCompositeID
            ),
        ]
        let plan = try builder.buildDecodePlan(
            fusedEntries: entries,
            unfusedCount: entries.count,
            bufferSet: try makeShortConvAdmissionBufferSet(device: device),
            slotDimension: 6_144,
            stafWeightStore: weightStore,
            hiddenSize: 2_048,
            accessPolicyResolver: accessPolicyResolver,
            planBuildContext: planBuildContext,
            argumentEncoders: [:],
            resolveDispatch: { entry in
                let name: String
                if entry.fragment is LinearFragment {
                    name = "gemv_2048_6144_bf16"
                } else if entry.fragment is Conv1dFragment {
                    name = "conv_state_update_bf16"
                } else {
                    throw MetalCompilerError.kernelNotFound("unexpected admission test fragment")
                }
                let pipeline = try #require(pipelineCache[name])
                let threads = min(max(pipeline.threadExecutionWidth, 1), pipeline.maxTotalThreadsPerThreadgroup)
                return (
                    name,
                    pipeline,
                    (
                        grid: MTLSize(width: 1, height: 1, depth: 1),
                        threadgroup: MTLSize(width: threads, height: 1, depth: 1),
                        sharedMemoryBytes: 0
                    )
                )
            }
        )
        return (
            plan,
            plan.steps.map { $0.metadata.kernelName ?? $0.pipeline.label ?? "(unlabeled)" }
        )
    }

    private func makeShortConvAdmissionBufferSet(device: MTLDevice) throws -> MetalBufferSet {
        let scalarByteSize = MemoryLayout<BFloat16>.stride
        let hidden = try makeBuffer(device: device, byteLength: 2_048 * scalarByteSize)
        let residual = try makeBuffer(device: device, byteLength: 2_048 * scalarByteSize)
        let scratch = try makeBuffer(device: device, byteLength: 2 * 6_144 * scalarByteSize)
        let convState = try makeBuffer(device: device, byteLength: 2_048 * 3 * scalarByteSize)
        let logits = try makeBuffer(device: device, byteLength: scalarByteSize)
        let uint32Bytes = MemoryLayout<UInt32>.stride
        return MetalBufferSet(
            bufferPrecision: .bfloat16,
            hidden: hidden,
            residual: residual,
            scratch: scratch,
            moeScratch: nil,
            weights: [],
            kvCache: nil,
            convState: convState,
            recurrentState: nil,
            convStateDimension: 2_048,
            convStateKernelSize: 3,
            recurrentStateBytesPerLayer: 0,
            perLayerInputs: nil,
            perLayerInputDimension: 0,
            perLayerInputLayerCount: 0,
            logits: logits,
            position: try makeBuffer(device: device, byteLength: uint32Bytes),
            ropePositionAxes: try makeBuffer(device: device, byteLength: 4 * uint32Bytes),
            tokenIn: try makeBuffer(device: device, byteLength: uint32Bytes),
            tokenOut: try makeBuffer(device: device, byteLength: uint32Bytes)
        )
    }

    private func makeShortConvAdmissionWeightStore(
        device: MTLDevice,
        blockedInProj: Bool
    ) throws -> STAFWeightStore {
        let rowMajor = try makeBuffer(device: device, byteLength: 1)
        let blocked = try makeBuffer(device: device, byteLength: 1)
        var specialized: [STAFSpecializedWeightKey: STAFWeightBufferAccess] = [:]
        if blockedInProj {
            specialized[STAFSpecializedWeightKey(
                tensorName: Self.inProjTensorName,
                layout: .blockedRows8Tiles128
            )] = STAFWeightBufferAccess(
                buffer: blocked,
                offset: 0,
                size: blocked.length,
                layout: .blockedRows8Tiles128
            )
        }
        return STAFWeightStore(
            buffer: rowMajor,
            entries: [
                Self.inProjTensorName: STAFTensorEntry(
                    name: Self.inProjTensorName,
                    payloadOffset: 0,
                    payloadSize: rowMajor.length,
                    schemeIdentifier: .bf16RowMajor,
                    semanticRole: .other,
                    shape: [6_144, 2_048],
                    blockSize: 0,
                    groupSize: 0,
                    bufferOffset: 0
                ),
                Self.convTensorName: STAFTensorEntry(
                    name: Self.convTensorName,
                    payloadOffset: 0,
                    payloadSize: rowMajor.length,
                    schemeIdentifier: .bf16RowMajor,
                    semanticRole: .other,
                    shape: [2_048, 3],
                    blockSize: 0,
                    groupSize: 0,
                    bufferOffset: 0
                ),
            ],
            metadata: .empty,
            specializedBufferAccesses: specialized
        )
    }

    private func makeBuffer(device: MTLDevice, byteLength: Int) throws -> MTLBuffer {
        try #require(device.makeBuffer(length: byteLength, options: .storageModeShared))
    }
}
