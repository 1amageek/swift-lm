import Foundation
import Testing
@testable import LMIR
@testable import ModelDeclarations

@Suite("Model Family Contract Tests", .tags(.unit))
struct ModelFamilyContractTests {
    @Test("Shared decoder normalizes nested text configuration")
    func decodesNestedTextConfiguration() throws {
        let data = Data(
            """
            {
              "model_type": "qwen3_vl",
              "max_position_embeddings": 256,
              "text_config": {
                "model_type": "qwen3_5",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "vocab_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rms_norm_eps": 0.00001,
                "rope_theta": 10000
              }
            }
            """.utf8
        )

        let configuration = try HuggingFaceConfigDecoder().decode(from: data)

        #expect(configuration.modelType == "qwen3_vl")
        #expect(configuration.maxContextLength == 256)
        #expect(configuration.modelConfig.hiddenSize == 16)
        #expect(configuration.modelConfig.layerCount == 2)
        #expect(configuration.modelConfig.kvHeads == 1)
        #expect(configuration.modelConfig.headDim == 8)
        #expect(configuration.modelConfig.ropeTheta == 10_000)
    }

    @Test("Shared decoder applies LFM2 block FFN adjustment")
    func decodesLFM2BlockAdjustedIntermediateSize() throws {
        let data = Data(
            """
            {
              "model_type": "lfm2",
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 128,
              "block_auto_adjust_ff_dim": true,
              "block_multiple_of": 256,
              "vocab_size": 32,
              "num_attention_heads": 2,
              "num_key_value_heads": 2,
              "head_dim": 4,
              "layer_types": ["conv"]
            }
            """.utf8
        )

        let configuration = try HuggingFaceConfigDecoder().decode(from: data)

        #expect(configuration.modelConfig.intermediateSize == 256)
    }

    @Test("Registry resolves LFM2 graph and family aliases")
    func resolvesLFM2Graph() throws {
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
            qkNorm: true,
            fullAttentionInterval: nil,
            ssmNumHeads: nil,
            ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil,
            convKernelSize: 3,
            convLCache: 2,
            partialRotaryFactor: nil,
            slidingWindow: nil,
            layerTypes: ["conv"]
        )

        let graph = try ModelFamilyRegistry.resolveModelGraph(
            modelType: "lfm2",
            config: config
        )

        #expect(ModelFamilyRegistry.family(for: "lfm2_moe") == .lfm2)
        #expect(ModelFamilyRegistry.family(for: "qwen3_vl") == .qwen35)
        #expect(graph.rootRegion.operations.isEmpty == false)
    }

    @Test("Unknown model families fail explicitly")
    func rejectsUnknownFamily() throws {
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
            convLCache: nil,
            partialRotaryFactor: nil,
            slidingWindow: nil
        )

        do {
            _ = try ModelFamilyRegistry.resolveModelGraph(
                modelType: "nemotron_h",
                config: config
            )
            Issue.record("Expected unsupported model family to fail")
        } catch let error as ModelGraphBuildError {
            #expect(error.description.contains("Unsupported model_type"))
        }

        do {
            _ = try ModelFamilyRegistry.namingConvention(for: "nemotron_h")
            Issue.record("Expected unsupported naming convention to fail")
        } catch let error as ModelGraphBuildError {
            #expect(error.description.contains("Unsupported model_type"))
        }
    }
}
