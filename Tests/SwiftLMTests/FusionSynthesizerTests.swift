import Testing
@testable import MetalCompiler

@Suite("FusionSynthesizer")
struct FusionSynthesizerTests {

    // MARK: - Register Intermediate (perElement + perElement)

    @Test("Two perElement fragments fuse with register intermediate")
    func registerFusion() throws {
        let fragA = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let contractA = try #require(fragA.fusionContract)
        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let fragB = ScalarMultiplyFragment(count: 896, weightRole: "scale_b")
        let contractB = try #require(fragB.fusionContract)
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        // Verify intermediate storage is register
        let storage = contractA.intermediateStorage(to: contractB)
        #expect(storage == .register)

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["scale_b": .float16]),
        ])

        // Register variable declared
        #expect(result.body.contains("float _fused_0;"))

        // Producer output subscript stripped: output[idx] → _fused_0
        #expect(result.body.contains("_fused_0 ="))
        #expect(!result.body.contains("_fused_0["))

        // Consumer input subscript stripped: data[idx] → _fused_0
        // Final output still writes to output[idx]
        #expect(result.body.contains("output[idx]"))
        #expect(result.body.contains("_fused_0 *"))
    }

    @Test("Register fusion merged contract eliminates internal ports")
    func registerFusionContract() throws {
        let fragA = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let contractA = try #require(fragA.fusionContract)
        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let fragB = ScalarMultiplyFragment(count: 896, weightRole: "scale_b")
        let contractB = try #require(fragB.fusionContract)
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["scale_b": .float16]),
        ])

        let mc = result.contract

        // A has 3 ports (data, weight, output), B has 3 ports (data, weight, output)
        // Internal: A.output → B.data (eliminated)
        // External: A.data (input), A.weight (input), B.weight (input), B.output (output)
        #expect(mc.ports.count == 4)

        // Inputs: A's data, A's weight, B's weight
        let inputs = mc.ports.filter { $0.direction == .input }
        #expect(inputs.count == 3)

        // Output: B's output only
        let outputs = mc.ports.filter { $0.direction == .output }
        #expect(outputs.count == 1)
        #expect(outputs[0].name == "output")

        // Parallelism preserved
        #expect(mc.parallelism == .perElement(count: 896))

        // No SIMD reduction needed
        #expect(mc.requiresSIMDReduction == false)

        // No threadgroup memory needed (register intermediate)
        #expect(mc.threadgroupMemoryBytes == 0)
    }

    @Test("Register fusion produces valid MSL via KernelScaffold")
    func registerFusionFullKernel() throws {
        let fragA = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let contractA = try #require(fragA.fusionContract)
        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let fragB = ScalarMultiplyFragment(count: 896, weightRole: "scale_b")
        let contractB = try #require(fragB.fusionContract)
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["scale_b": .float16]),
        ])

        // Generate complete MSL kernel
        let msl = KernelScaffold.generate(
            name: "fused_scalar_multiply_test",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float16,
            weightFormats: result.weightFormats,
            isSequence: false
        )

        // Kernel function declaration
        #expect(msl.contains("kernel void fused_scalar_multiply_test"))

        // Buffer bindings for external ports
        #expect(msl.contains("[[buffer(0)]]"))  // A.data
        #expect(msl.contains("[[buffer(1)]]"))  // A.weight (scale_a)
        #expect(msl.contains("[[buffer(2)]]"))  // B.weight (scale_b)
        #expect(msl.contains("[[buffer(3)]]"))  // B.output

        // Dimension parameter
        #expect(msl.contains("dimension"))

        // Register intermediate variable
        #expect(msl.contains("float _fused_0;"))

        // No threadgroup barrier (register intermediate)
        #expect(!msl.contains("threadgroup_barrier"))
    }

    // MARK: - Threadgroup Memory Intermediate (perRow + perElement)

    @Test("Reduction + ScalarMultiply intermediate is threadgroup memory")
    func threadgroupIntermediateStorage() throws {
        let reduction = Reduction(dimension: 896, epsilon: 1e-6, weightRole: "scale")
        let contractA = try #require(reduction.fusionContract)

        let scalar = ScalarMultiplyFragment(count: 896, weightRole: "layer_scalar")
        let contractB = try #require(scalar.fusionContract)

        // perRow + perElement with singlePass → threadgroup memory
        // (perRow resolved, singlePass but perRow uses loop — TG memory needed)
        let storage = contractA.intermediateStorage(to: contractB)
        if case .threadgroupMemory(let dim) = storage {
            #expect(dim == 896)
        } else {
            Issue.record("Expected threadgroupMemory, got \(storage)")
        }

        // Resolved parallelism: perRow wins
        let resolved = contractA.parallelism.resolved(with: contractB.parallelism)
        #expect(resolved == .perRow(dimension: 896))
    }

    @Test("Reduction + ScalarMultiply fuse into single kernel with TG intermediate")
    func perRowPerElementFusion() throws {
        let reduction = Reduction(dimension: 896, epsilon: 1e-6, weightRole: "norm_scale", weightBias: 1.0)
        let contractA = try #require(reduction.fusionContract)
        let bodyA = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let scalar = ScalarMultiplyFragment(count: 896, weightRole: "layer_scalar")
        let contractB = try #require(scalar.fusionContract)
        let bodyB = try #require(scalar.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["norm_scale": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["layer_scalar": .float16]),
        ])

        // Resolved parallelism: perRow wins
        #expect(result.contract.parallelism == .perRow(dimension: 896))

        // SIMD reduction from Reduction
        #expect(result.contract.requiresSIMDReduction == true)

        // Threadgroup memory: base 128 (32*4 from Reduction) + intermediate 896*4 = 3712
        #expect(result.contract.threadgroupMemoryBytes == 128 + 896 * MemoryLayout<Float>.size)

        // TG intermediate declared
        #expect(result.body.contains("threadgroup float _tg_fused_0[896];"))

        // Barrier between Reduction output and ScalarMultiply input
        #expect(result.body.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"))

        // Reduction body writes to TG intermediate instead of output
        #expect(result.body.contains("_tg_fused_0[i]"))

        // ScalarMultiply body wrapped in perRow loop
        #expect(result.body.contains("for (uint i = tid; i < dimension; i += threadgroupSize)"))

        // ScalarMultiply reads from TG intermediate
        // (data renamed to _tg_fused_0, original body: data[idx] → _tg_fused_0[i] after wrap+rename)
        #expect(result.body.contains("_tg_fused_0[i]"))

        // Scalar constants from both fragments merged
        #expect(result.contract.scalarConstants.count == 2)  // epsilon, weightBias from Reduction
        #expect(result.contract.scalarConstants[0].name == "epsilon")
        #expect(result.contract.scalarConstants[1].name == "weightBias")
    }

    @Test("Reduction + ScalarMultiply produces valid MSL kernel")
    func perRowPerElementFullKernel() throws {
        let reduction = Reduction(dimension: 256, epsilon: 1e-5, weightRole: "norm_scale")
        let contractA = try #require(reduction.fusionContract)
        let bodyA = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let scalar = ScalarMultiplyFragment(count: 256, weightRole: "layer_scalar")
        let contractB = try #require(scalar.fusionContract)
        let bodyB = try #require(scalar.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["norm_scale": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["layer_scalar": .float16]),
        ])

        let msl = KernelScaffold.generate(
            name: "fused_norm_layerscale",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float32,
            weightFormats: result.weightFormats,
            isSequence: true
        )

        // Kernel function
        #expect(msl.contains("kernel void fused_norm_layerscale"))

        // perRow scaffold markers
        #expect(msl.contains("tid"))
        #expect(msl.contains("threadgroupSize"))
        #expect(msl.contains("threadgroup float shared[32]"))

        // External buffer ports: data(input), norm_weight(input), scalar_weight(input), output
        // + dimension + scalar constants + sequenceLength
        #expect(msl.contains("data_base"))
        #expect(msl.contains("output_base"))
        #expect(msl.contains("sequenceLength"))

        // RMS norm computation from Reduction body
        #expect(msl.contains("simd_sum(sumSquared)"))
        #expect(msl.contains("rsqrt(total / float(dimension) + epsilon)"))

        // TG barrier between phases
        #expect(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"))

        // ScalarMultiply in perRow loop
        #expect(msl.contains("for (uint i = tid; i < dimension; i += threadgroupSize)"))
    }

    @Test("wrapPerElementBodyForPerRow adds cooperative loop")
    func wrapPerElementBody() {
        let body = """
        output[idx] = data[idx] * scale;
        """
        let wrapped = FusionSynthesizer.wrapPerElementBodyForPerRow(body)

        #expect(wrapped.contains("for (uint i = tid; i < dimension; i += threadgroupSize)"))
        #expect(wrapped.contains("uint idx = i;"))
        #expect(wrapped.contains("output[idx] = data[idx] * scale;"))
    }

    // MARK: - Merged Contract Properties

    @Test("Three-way fusion merges contracts correctly")
    func threeWayFusionContract() throws {
        let fragA = ScalarMultiplyFragment(count: 512, weightRole: "w_a")
        let fragB = ScalarMultiplyFragment(count: 512, weightRole: "w_b")
        let fragC = ScalarMultiplyFragment(count: 512, weightRole: "w_c")

        let contractA = try #require(fragA.fusionContract)
        let contractB = try #require(fragB.fusionContract)
        let contractC = try #require(fragC.fusionContract)

        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float32, weightFormat: .float16))
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float32, weightFormat: .float16))
        let bodyC = try #require(fragC.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["w_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["w_b": .float16]),
            .init(contract: contractC, body: bodyC, weightFormats: ["w_c": .float16]),
        ])

        let mc = result.contract

        // A(data,weight_a,output) → B(data,weight_b,output) → C(data,weight_c,output)
        // Internal: A.output→B.data, B.output→C.data
        // External: A.data, A.weight, B.weight, C.weight, C.output
        #expect(mc.ports.count == 5)

        let inputs = mc.ports.filter { $0.direction == .input }
        #expect(inputs.count == 4)  // A.data + 3 weights

        let outputs = mc.ports.filter { $0.direction == .output }
        #expect(outputs.count == 1)

        // Two register intermediates
        #expect(result.body.contains("float _fused_0;"))
        #expect(result.body.contains("float _fused_1;"))

        // Combined weight formats
        #expect(result.weightFormats.count == 3)
    }

    // MARK: - Weight Format Preservation

    @Test("Weight formats from all entries are preserved in result")
    func weightFormatPreservation() throws {
        let fragA = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let contractA = try #require(fragA.fusionContract)
        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let fragB = ScalarMultiplyFragment(count: 896, weightRole: "scale_b")
        let contractB = try #require(fragB.fusionContract)
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["scale_b": .bfloat16]),
        ])

        #expect(result.weightFormats["scale_a"] == .float16)
        #expect(result.weightFormats["scale_b"] == .bfloat16)
    }

    // MARK: - Error Cases

    @Test("Insufficient entries throws error")
    func insufficientEntries() throws {
        let frag = ScalarMultiplyFragment(count: 896, weightRole: "scale")
        let contract = try #require(frag.fusionContract)
        let body = try #require(frag.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        #expect(throws: FusionSynthesizer.SynthesisError.self) {
            _ = try FusionSynthesizer.synthesize([
                .init(contract: contract, body: body)
            ])
        }
    }

    @Test("Incompatible parallelism throws error")
    func incompatibleParallelism() throws {
        let frag896 = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let frag2048 = ScalarMultiplyFragment(count: 2048, weightRole: "scale_b")

        let contractA = try #require(frag896.fusionContract)
        let contractB = try #require(frag2048.fusionContract)
        let bodyA = try #require(frag896.kernelBody(bufferPrecision: .float16, weightFormat: .float16))
        let bodyB = try #require(frag2048.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        #expect(throws: FusionSynthesizer.SynthesisError.self) {
            _ = try FusionSynthesizer.synthesize([
                .init(contract: contractA, body: bodyA),
                .init(contract: contractB, body: bodyB),
            ])
        }
    }

    // MARK: - Variable Renaming Correctness

    @Test("replaceArrayAccessWithScalar strips subscripts correctly")
    func arrayAccessReplacement() {
        // Basic subscript stripping
        let result1 = FusionSynthesizer.replaceArrayAccessWithScalar(
            in: "output[idx] = data[idx] * scale;",
            arrayName: "output",
            scalarName: "_fused_0"
        )
        #expect(result1 == "_fused_0 = data[idx] * scale;")

        // Multiple occurrences
        let result2 = FusionSynthesizer.replaceArrayAccessWithScalar(
            in: "x = output[i] + output[j];",
            arrayName: "output",
            scalarName: "_v"
        )
        #expect(result2 == "x = _v + _v;")

        // Does not replace substrings
        let result3 = FusionSynthesizer.replaceArrayAccessWithScalar(
            in: "output_base[idx] = 0;",
            arrayName: "output",
            scalarName: "_fused"
        )
        #expect(result3 == "output_base[idx] = 0;")

        // Bare name without subscript also replaced
        let result4 = FusionSynthesizer.replaceArrayAccessWithScalar(
            in: "float v = data;",
            arrayName: "data",
            scalarName: "_fused"
        )
        #expect(result4 == "float v = _fused;")
    }

    // MARK: - Sequence Mode Fusion

    @Test("Register fusion works in sequence mode (F32 prefill)")
    func registerFusionSequenceMode() throws {
        let fragA = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let contractA = try #require(fragA.fusionContract)
        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let fragB = ScalarMultiplyFragment(count: 896, weightRole: "scale_b")
        let contractB = try #require(fragB.fusionContract)
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["scale_b": .float16]),
        ])

        let msl = KernelScaffold.generate(
            name: "fused_seq_test",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float32,
            weightFormats: result.weightFormats,
            isSequence: true
        )

        // Sequence mode markers
        #expect(msl.contains("sequenceLength"))
        #expect(msl.contains("seqPos"))
        #expect(msl.contains("if (i >= dimension || seqPos >= sequenceLength) return;"))

        // Kernel function
        #expect(msl.contains("kernel void fused_seq_test"))
    }
}
