import SwiftLM

/// Best-effort model for unknown model_type values.
///
/// Inspects ModelConfig fields to construct a reasonable IR:
/// - Has layer_types -> heterogeneous LayerStack
/// - No layer_types -> uniform Repeat with attention blocks
/// - Has expertCount -> MoE instead of MLP
/// - Has convLCache -> Short convolution layers (via StateSpace)
/// - Has ssmNumHeads -> DeltaNet layers
///
/// No guarantee of correctness. Errors on missing required data.
public struct AnyModel: ModelComponent {

    public let config: ModelConfig

    public init(config: ModelConfig) {
        self.config = config
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)

        if let layerTypes = config.layerTypes {
            ForEach(layerTypes.enumerated().map { IndexedLayerType(index: $0.offset, type: $0.element) }) { item in
                AnyDecoderLayer(config: config, layerType: item.type)
            }
        } else {
            Repeat(count: config.layerCount, label: "layers") {
                AnyDecoderLayer(config: config, layerType: "full_attention")
            }
        }

        makeNorm()
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            tiedToEmbedding: config.tiedEmbeddings
        )
    }

    @ModelComponentBuilder
    private func makeNorm() -> some ModelComponent {
        switch config.normKind {
        case .rmsNorm:
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        case .layerNorm:
            LayerNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        }
    }
}

/// Helper for indexed iteration with ForEach.
private struct IndexedLayerType: Sendable {
    let index: Int
    let type: String
}

/// Single decoder layer for AnyModel.
/// Selects the operator component based on the layer_type string.
struct AnyDecoderLayer: ModelComponent {

    let config: ModelConfig
    let layerType: String

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            makeNorm()
            makeOperator()
        }
        Residual {
            makeNorm()
            makeFeedForward()
        }
    }

    @ModelComponentBuilder
    private func makeNorm() -> some ModelComponent {
        switch config.normKind {
        case .rmsNorm:
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        case .layerNorm:
            LayerNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        }
    }

    @ModelComponentBuilder
    private func makeOperator() -> some ModelComponent {
        switch layerType {
        case "conv":
            StateSpace(
                hiddenSize: config.hiddenSize,
                numHeads: 1,
                keyHeadDim: config.convLCache ?? 3,
                valueHeadDim: config.convLCache ?? 3,
                variant: "short_conv"
            )
        case "linear_attention":
            StateSpace(
                hiddenSize: config.hiddenSize,
                numHeads: config.ssmNumHeads ?? config.attentionHeads,
                groupCount: config.ssmGroupCount ?? config.ssmNumHeads ?? config.attentionHeads,
                keyHeadDim: config.ssmKeyHeadDim ?? 128,
                valueHeadDim: config.ssmValueHeadDim ?? 128,
                variant: DeltaNet.Variant.gated.rawValue
            )
        default:
            Attention(
                hiddenSize: config.hiddenSize,
                headCount: config.attentionHeads,
                kvHeadCount: config.kvHeads,
                headDimension: config.headDim,
                bias: config.attentionBias,
                rope: RoPEAttributes(
                    dimension: config.ropeDimension,
                    base: config.ropeTheta,
                    scaling: config.ropeScaling,
                    mropeAxes: config.mropeAxes
                ),
                qkNorm: config.qkNorm ? .rmsNorm : nil,
                window: config.slidingWindow.map { AttentionWindow(left: $0, right: $0) }
            )
        }
    }

    @ModelComponentBuilder
    private func makeFeedForward() -> some ModelComponent {
        if let expertCount = config.expertCount, let expertsPerToken = config.expertsPerToken {
            MoE(
                expertCount: expertCount,
                expertsPerToken: expertsPerToken,
                expertInputSize: config.hiddenSize,
                expertOutputSize: config.hiddenSize,
                expertIntermediateSize: config.intermediateSize,
                expertActivation: .silu,
                expertGating: .swiglu,
                expertBias: config.mlpBias
            )
        } else {
            MLP(
                inputSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                activation: .silu,
                gating: .swiglu,
                bias: config.mlpBias
            )
        }
    }
}
