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
}
