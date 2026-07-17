import CoreAIExport
import LMArchitecture
import ModelDeclarations
import Testing

@Suite("Core AI export")
struct CoreAIExportTests {
    @Test("Transformer export is deterministic and preserves weight bindings")
    func transformerExport() throws {
        let config = ModelConfig(
            hiddenSize: 8,
            layerCount: 1,
            intermediateSize: 16,
            vocabSize: 32,
            attentionHeads: 2,
            kvHeads: 2,
            headDim: 4,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-5,
            normKind: .rmsNorm,
            ropeTheta: 10_000,
            ropeDimension: 4,
            ropeScaling: nil,
            tiedEmbeddings: true,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: nil,
            ssmNumHeads: nil,
            ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil,
            convKernelSize: nil,
            partialRotaryFactor: nil,
            slidingWindow: nil
        )
        let configuration = try CoreAIExportConfiguration(
            name: "tiny-transformer",
            modelType: "llama",
            target: .macOSDynamic,
            maxContextLength: 128,
            vocabSize: config.vocabSize
        )
        let exporter = CoreAIModelExporter()

        let first = try exporter.makeDocument(
            component: Transformer(config: config),
            namingConvention: LlamaFamilyNaming(),
            configuration: configuration
        )
        let second = try exporter.makeDocument(
            component: Transformer(config: config),
            namingConvention: LlamaFamilyNaming(),
            configuration: configuration
        )

        #expect(first == second)
        #expect(first.formatVersion == CoreAIExportDocument.currentFormatVersion)
        #expect(first.rootRegion.operations.isEmpty == false)
        #expect(first.rootRegion.operations.contains { !$0.parameterBindings.isEmpty })
    }

    @Test("Canonicalization preserves parameter bindings")
    func canonicalizationPreservesBindings() throws {
        let operation = Operation(
            key: OperationKey(rawValue: 4),
            kind: .primitive(RMSNormAttributes(dimension: 8)),
            operands: [],
            results: [OperationResult(id: ValueID(rawValue: 8))],
            parameterBindings: [ParameterBinding(role: "weight", tensorName: "norm.weight")]
        )
        let graph = ModelGraph(
            rootRegion: Region(
                operations: [operation],
                results: [ValueUse(value: ValueID(rawValue: 8))]
            )
        )

        let canonical = canonicalize(graph)
        #expect(canonical.rootRegion.operations[0].parameterBindings == operation.parameterBindings)
    }

    @Test("JSON attributes preserve integer and boolean scalar types")
    func jsonScalarTypes() throws {
        let attributes = AttentionAttributes(
            hiddenSize: 8,
            headCount: 2,
            kvHeadCount: 1,
            headDimension: 4
        )
        let operation = Operation(
            key: OperationKey(rawValue: 0),
            kind: .primitive(attributes),
            operands: [],
            results: [OperationResult(id: ValueID(rawValue: 0))]
        )
        let document = try CoreAIModelExporter().makeDocument(
            graph: ModelGraph(
                rootRegion: Region(
                    operations: [operation],
                    results: [ValueUse(value: ValueID(rawValue: 0))]
                )
            ),
            configuration: CoreAIExportConfiguration(
                name: "scalar-types",
                modelType: "llama",
                target: .macOSDynamic,
                maxContextLength: 32,
                vocabSize: 16
            )
        )

        guard case .primitive(let primitive) = document.rootRegion.operations[0].kind,
              case .object(let payload) = primitive.attributes else {
            Issue.record("Expected an encoded attention primitive")
            return
        }
        #expect(payload["kvHeadCount"] == .number(1))
        #expect(payload["causal"] == .bool(true))
    }

    @Test("Canonicalization preserves RMSNorm scale policy")
    func rmsNormScalePolicy() throws {
        let operation = Operation(
            key: OperationKey(rawValue: 0),
            kind: .primitive(
                RMSNormAttributes(
                    dimension: 8,
                    epsilon: 1e-5,
                    weightBias: 1,
                    withScale: false
                )
            ),
            operands: [],
            results: [OperationResult(id: ValueID(rawValue: 0))]
        )
        let graph = canonicalize(
            ModelGraph(
                rootRegion: Region(
                    operations: [operation],
                    results: [ValueUse(value: ValueID(rawValue: 0))]
                )
            )
        )

        guard case .primitive(let rawAttributes) = graph.rootRegion.operations[0].kind,
              let attributes = rawAttributes as? RMSNormAttributes else {
            Issue.record("Expected RMSNorm attributes")
            return
        }
        #expect(attributes.withScale == false)
        #expect(attributes.weightBias == 1)
    }

    @Test("Transformer export binds every layer independently")
    func transformerLayerBindings() throws {
        let config = ModelConfig(
            hiddenSize: 8,
            layerCount: 2,
            intermediateSize: 16,
            vocabSize: 32,
            attentionHeads: 2,
            kvHeads: 1,
            headDim: 4,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-5,
            normKind: .rmsNorm,
            ropeTheta: 10_000,
            ropeDimension: 4,
            ropeScaling: nil,
            tiedEmbeddings: false,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: nil,
            ssmNumHeads: nil,
            ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil,
            convKernelSize: nil,
            partialRotaryFactor: nil,
            slidingWindow: nil
        )
        let document = try CoreAIModelExporter().makeDocument(
            component: Transformer(config: config),
            namingConvention: LlamaFamilyNaming(),
            configuration: CoreAIExportConfiguration(
                name: "two-layer-transformer",
                modelType: "llama",
                target: .macOSDynamic,
                maxContextLength: 32,
                vocabSize: config.vocabSize
            )
        )
        let names = allTensorNames(in: document.rootRegion)

        #expect(names.contains("model.layers.0.self_attn.q_proj.weight"))
        #expect(names.contains("model.layers.1.self_attn.q_proj.weight"))
        #expect(document.rootRegion.operations.contains { operation in
            if case .repeating = operation.kind { return true }
            return false
        } == false)
    }

    @Test("Stateful export derives persistent state from Swift IR")
    func statefulProgramContract() throws {
        let attention = Operation(
            key: OperationKey(rawValue: 10),
            kind: .primitive(
                AttentionAttributes(
                    hiddenSize: 8,
                    headCount: 2,
                    kvHeadCount: 1,
                    headDimension: 4
                )
            ),
            operands: [],
            results: [OperationResult(id: ValueID(rawValue: 10))]
        )
        let convolution = Operation(
            key: OperationKey(rawValue: 11),
            kind: .primitive(ShortConvAttributes(hiddenSize: 8, kernelSize: 3)),
            operands: [],
            results: [OperationResult(id: ValueID(rawValue: 11))]
        )
        let document = try CoreAIModelExporter().makeDocument(
            graph: ModelGraph(
                rootRegion: Region(
                    operations: [attention, convolution],
                    results: [ValueUse(value: ValueID(rawValue: 11))]
                )
            ),
            configuration: CoreAIExportConfiguration(
                name: "stateful-hybrid",
                modelType: "lfm2",
                target: .macOSDynamic,
                maxContextLength: 64,
                vocabSize: 32,
                execution: .stateful
            )
        )

        #expect(document.formatVersion == 2)
        #expect(document.program.source == .swiftLMIR)
        #expect(document.program.execution == .stateful)
        #expect(document.program.functions.count == 1)
        let function = try #require(document.program.functions.first)
        #expect(function.states.map(\.name) == ["keyCache", "valueCache", "convCache"])
        #expect(function.states[0].dimensions[0] == .fixed(1))
        #expect(function.states[0].dimensions[3] == .dynamic("cache_length", minimum: 1, maximum: 64))
        #expect(document.rootRegion.operations[0].stateBindings == [
            .init(role: "key_cache", state: "keyCache", axisIndex: 0),
            .init(role: "value_cache", state: "valueCache", axisIndex: 0),
        ])
        #expect(document.rootRegion.operations[1].stateBindings == [
            .init(role: "conv_cache", state: "convCache", axisIndex: 0)
        ])
    }

    @Test("Stateful export rejects static iOS target")
    func statefulStaticTargetFailure() throws {
        let graph = ModelGraph(rootRegion: Region())
        #expect(throws: CoreAIExportError.self) {
            try CoreAIModelExporter().makeDocument(
                graph: graph,
                configuration: CoreAIExportConfiguration(
                    name: "invalid-stateful",
                    modelType: "lfm2",
                    target: .iOSStatic,
                    maxContextLength: 64,
                    vocabSize: 32,
                    execution: .stateful
                )
            )
        }
    }

    @Test("Stateful export requires mutable graph state")
    func statefulWithoutStateFailure() throws {
        let operation = Operation(
            key: OperationKey(rawValue: 0),
            kind: .primitive(RMSNormAttributes(dimension: 8)),
            operands: [],
            results: [OperationResult(id: ValueID(rawValue: 0))]
        )
        #expect(throws: CoreAIExportError.self) {
            try CoreAIModelExporter().makeDocument(
                graph: ModelGraph(
                    rootRegion: Region(
                        operations: [operation],
                        results: [ValueUse(value: ValueID(rawValue: 0))]
                    )
                ),
                configuration: CoreAIExportConfiguration(
                    name: "invalid-stateful",
                    modelType: "test",
                    target: .macOSDynamic,
                    maxContextLength: 64,
                    vocabSize: 32,
                    execution: .stateful
                )
            )
        }
    }
}

private func allTensorNames(in region: CoreAIExportDocument.Region) -> Set<String> {
    var names = Set(region.operations.flatMap { $0.parameterBindings.map(\.tensorName) })
    for operation in region.operations {
        switch operation.kind {
        case .primitive:
            break
        case .residual(_, let body), .repeating(_, let body):
            names.formUnion(allTensorNames(in: body))
        case .parallel(_, let branches):
            for branch in branches {
                names.formUnion(allTensorNames(in: branch))
            }
        case .conditional(_, let thenRegion, let elseRegion):
            names.formUnion(allTensorNames(in: thenRegion))
            names.formUnion(allTensorNames(in: elseRegion))
        }
    }
    return names
}
