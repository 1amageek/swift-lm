import Testing
import Metal
@testable import MetalCompiler
import LMArchitecture
import LMIR

@Suite("Fragment Protocol")
struct FragmentProtocolTests {

    @Test
    func embeddingFragmentIsGather() {
        let a = TokenEmbeddingAttributes(vocabSize: 32000, embeddingSize: 2048)
        let frag = a.fragment(context: KernelContext(bufferPrecision: .float16, weightFormat: .float16))
        #expect(frag is GatherFragment)
        if case .gather(let count) = frag.dispatchDimension {
            #expect(count == 2048)
        }
    }

    @Test
    func rmsNormFragmentIsReduction() {
        let a = RMSNormAttributes(dimension: 2048, epsilon: 1e-5)
        let frag = a.fragment(context: KernelContext(bufferPrecision: .float16, weightFormat: .float16))
        #expect(frag is Reduction)
        if case .reduction(let dim) = frag.dispatchDimension {
            #expect(dim == 2048)
        }
    }

    @Test
    func structuralOpsReturnNilFragment() {
        let r = OperationKind.residual(strategy: .add, body: Region())
        let op = Operation(key: OperationKey(rawValue: 0), kind: r, operands: [], results: [])
        #expect(op.kernelFragment == nil)
    }
}

@Suite
struct DispatchPlanCompilationTests {

    @Test
    func tinyModelCompilesToMultiDispatch() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let graph = try ModelGraph(TinyTestModel(hiddenSize: 64, vocabSize: 100))
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(graph: graph, hiddenSize: 64, vocabSize: 100, device: device)

        // Multiple dispatch steps (not 1)
        #expect(plan.steps.count > 1, "Expected multiple dispatches, got \(plan.steps.count)")
    }

    @Test
    func transformerCompilesToManyDispatches() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let graph = try ModelGraph(TinyTransformer(hiddenSize: 64, layers: 2, vocabSize: 100))
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(graph: graph, hiddenSize: 64, vocabSize: 100, device: device)

        // 2 layers x (norm + 5 attn + residual + norm + 4 mlp + residual) + embed + final_norm + output + argmax
        #expect(plan.steps.count > 20, "Expected many dispatches for 2-layer transformer, got \(plan.steps.count)")
    }

    @Test
    func standardOptimizerFusesMlpFrontHalfInDispatchDump() throws {
        let graph = try ModelGraph(TinyTransformer(hiddenSize: 64, layers: 1, vocabSize: 100))
        let compiler = MetalInferenceCompiler()
        let dump = compiler.dumpDispatchEntries(graph: graph, hiddenSize: 64)

        #expect(dump.contains("fusedSwiGLUProjection("), "Standard optimizer should fuse MLP front half:\n\(dump)")
    }

    @Test
    func standardOptimizerDoesNotBatchAttentionProjections() throws {
        let graph = try ModelGraph(TinyTransformer(hiddenSize: 64, layers: 1, vocabSize: 100))
        let compiler = MetalInferenceCompiler()
        let dump = compiler.dumpDispatchEntries(graph: graph, hiddenSize: 64)

        #expect(!dump.contains("batchedProjection("), "Standard optimizer should not batch projections:\n\(dump)")
    }

    @Test
    func aggressiveOptimizerBatchesAttentionProjectionsInDispatchDump() throws {
        let graph = try ModelGraph(TinyTransformer(hiddenSize: 64, layers: 1, vocabSize: 100))
        let compiler = MetalInferenceCompiler(optimizer: AggressiveOptimizer())
        let dump = compiler.dumpDispatchEntries(graph: graph, hiddenSize: 64)

        #expect(dump.contains("batchedProjection(q_proj,k_proj,v_proj)"), "Aggressive optimizer should batch attention projections:\n\(dump)")
    }

    @Test
    func threadgroupSizesRespectPipelineLimits() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let graph = try ModelGraph(TinyTestModel(hiddenSize: 64, vocabSize: 100))
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(graph: graph, hiddenSize: 64, vocabSize: 100, device: device)

        for (i, step) in plan.steps.enumerated() {
            let maxThreads = step.pipeline.maxTotalThreadsPerThreadgroup
            let tgWidth = step.threadgroupSize.width * step.threadgroupSize.height * step.threadgroupSize.depth
            #expect(tgWidth <= maxThreads,
                "Step \(i) threadgroup \(tgWidth) exceeds pipeline max \(maxThreads)")

            let warpWidth = step.pipeline.threadExecutionWidth
            #expect(step.threadgroupSize.width % warpWidth == 0 || step.threadgroupSize.width < warpWidth,
                "Step \(i) threadgroup width \(step.threadgroupSize.width) not aligned to warp \(warpWidth)")
        }
    }

    @Test
    func specializedFusedSwiGLUArgumentTableRetainsOutputDimensionConstant() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let model = SpecializedSwiGLUTestModel(
            hiddenSize: 2048,
            intermediateSize: 6144,
            vocabSize: 128
        )
        let graph = try ModelGraph(model)
        let compiler = MetalInferenceCompiler(optimizer: AggressiveOptimizer())
        let plan = try compiler.compile(
            graph: graph,
            hiddenSize: 2048,
            intermediateSize: 6144,
            vocabSize: 128,
            device: device
        )

        let step = try #require(plan.steps.first { step in
            (step.pipeline.label ?? "").hasPrefix("fused_swiglu_projection_2048")
        })
        #expect(step.bindings.argumentPolicy == .argumentTable)

        let outputDimensionBinding = try #require(
            step.bindings.constants.first(where: { $0.index == 4 })
        )
        #expect(decodeUInt32(outputDimensionBinding) == 6144)
    }

    @Test
    func fusedSwiGLUDecodeRoutingKeepsFollowingProjectionOutOfPlace() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let model = SpecializedSwiGLUTestModel(
            hiddenSize: 2048,
            intermediateSize: 6144,
            vocabSize: 128
        )
        let graph = try ModelGraph(model)
        let compiler = MetalInferenceCompiler(optimizer: AggressiveOptimizer())
        let plan = try compiler.compile(
            graph: graph,
            hiddenSize: 2048,
            intermediateSize: 6144,
            vocabSize: 128,
            device: device
        )

        let fusedIndex = try #require(plan.steps.firstIndex { step in
            let name = step.metadata.kernelName ?? step.pipeline.label ?? ""
            return name.hasPrefix("fused_swiglu_projection_2048")
        })
        let fusedStep = plan.steps[fusedIndex]
        let fusedOutput = try #require(fusedStep.bindings.buffers.first(where: { $0.index == 3 }))

        let nextStepIndex = fusedIndex + 1
        #expect(nextStepIndex < plan.steps.count)
        let downProjection = plan.steps[nextStepIndex]
        let nextKernelName = downProjection.metadata.kernelName ?? downProjection.pipeline.label ?? ""
        #expect(
            nextKernelName.hasPrefix("gemv"),
            "Expected the step after fused SwiGLU to be the following projection, got \(nextKernelName)"
        )
        let downInput = try #require(downProjection.bindings.buffers.first(where: { $0.index == 0 }))
        let downOutput = try #require(downProjection.bindings.buffers.first(where: { $0.index == 2 }))

        #expect(downInput.buffer === fusedOutput.buffer)
        #expect(downInput.offset == fusedOutput.offset)
        #expect(
            !(downOutput.buffer === downInput.buffer && downOutput.offset == downInput.offset),
            "Decode down projection must not alias fused SwiGLU output in place"
        )
    }
}

@Suite
struct SafetensorsTests {
    @Test
    func dtypeParsing() {
        #expect(SafetensorsDType(rawValue: "F16") == .float16)
        #expect(SafetensorsDType(rawValue: "F32") == .float32)
        #expect(SafetensorsDType.float16.elementSize == 2)
    }
}

// MARK: - Test Models

struct TinyTestModel: ModelComponent {
    let hiddenSize: Int
    let vocabSize: Int
    var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
        RMSNorm(dimension: hiddenSize, epsilon: 1e-5)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}

struct TinyTransformer: ModelComponent {
    let hiddenSize: Int
    let layers: Int
    let vocabSize: Int
    var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
        Repeat(count: layers) {
            Residual {
                RMSNorm(dimension: hiddenSize, epsilon: 1e-5)
                Attention(
                    hiddenSize: hiddenSize, headCount: 4, kvHeadCount: 2,
                    headDimension: hiddenSize / 4,
                    rope: RoPEAttributes(dimension: hiddenSize / 4, base: 10000.0))
            }
            Residual {
                RMSNorm(dimension: hiddenSize, epsilon: 1e-5)
                MLP(inputSize: hiddenSize, intermediateSize: hiddenSize * 4)
            }
        }
        RMSNorm(dimension: hiddenSize, epsilon: 1e-5)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}

struct SpecializedSwiGLUTestModel: ModelComponent {
    let hiddenSize: Int
    let intermediateSize: Int
    let vocabSize: Int

    var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
        Residual {
            RMSNorm(dimension: hiddenSize, epsilon: 1e-5)
            MLP(inputSize: hiddenSize, intermediateSize: intermediateSize)
        }
        RMSNorm(dimension: hiddenSize, epsilon: 1e-5)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}

private func decodeUInt32(_ binding: MetalConstantBinding) -> UInt32 {
    switch binding {
    case .inline(let bytes):
        return decodeUInt32(bytes.value)
    case .buffer(let bytes):
        let start = bytes.offset
        let end = start + bytes.length
        guard bytes.length == 4 else {
            return 0
        }
        let contents = bytes.buffer.contents()
        let pointer = contents.bindMemory(to: UInt8.self, capacity: end)
        let value = Array(UnsafeBufferPointer(start: pointer.advanced(by: start), count: bytes.length))
        return decodeUInt32(value)
    }
}

private func decodeUInt32(_ bytes: [UInt8]) -> UInt32 {
    guard bytes.count == 4 else { return 0 }
    return UInt32(bytes[0])
        | (UInt32(bytes[1]) << 8)
        | (UInt32(bytes[2]) << 16)
        | (UInt32(bytes[3]) << 24)
}

// MARK: - Kernel Completeness Tests

@Suite("Kernel Completeness")
struct KernelCompletenessTests {

    @Test("All QuantizationFormat kernel names exist in MSL")
    func quantizationFormatKernelsExist() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16), options: options)
        let available = Set(library.functionNames)

        let formats: [any QuantizationFormat] = [
            Float16Format(),
            BFloat16Format(),
            AffineQ4Group64Format(),
            AffineQ4Group128Format(),
            AffineQ8Group32Format(),
            AffineQ8Group64Format(),
        ]
        for format in formats {
            #expect(available.contains(format.gemvKernelName),
                "Missing kernel '\(format.gemvKernelName)' for \(type(of: format))")
        }
    }

    @Test("All primitive fragment kernels can be generated")
    func primitiveFragmentKernelsGenerate() throws {
        let fragments: [any PrimitiveMetalKernelFragment] = [
            Reduction(dimension: 128, epsilon: 1e-6),
            ElementwiseFragment(count: 128),
            GatherFragment(vocabularySize: 1000, embeddingDimension: 128),
            ArgmaxFragment(vocabularySize: 1000),
            FlashAttentionFragment(headCount: 4, kvHeadCount: 4, headDimension: 64),
            RoPEFragment(headCount: 4, kvHeadCount: 4, headDimension: 64, ropeDimension: 64, base: 10000),
            QKNormFragment(headCount: 4, headDimension: 64, epsilon: 1e-6, weightRole: "q_layernorm"),
            Conv1dFragment(dimension: 128, kernelSize: 3),
            SigmoidGateFragment(dimension: 128),
        ]
        for frag in fragments {
            let src = frag.kernelSource(
                name: "test_\(type(of: frag))",
                bufferPrecision: .float16, weightFormat: .bfloat16)
            #expect(!src.isEmpty, "Failed to generate kernel for \(type(of: frag))")
        }
    }

    @Test("Structural kernels exist in MSL")
    func structuralKernelsExist() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16), options: options)
        let available = Set(library.functionNames)

        #expect(available.contains("copy_buffer"))
        #expect(available.contains("residual_add"))
        #expect(available.contains("quantize_kv_q8"))
        #expect(available.contains("dequantize_kv_q8"))
    }
}
