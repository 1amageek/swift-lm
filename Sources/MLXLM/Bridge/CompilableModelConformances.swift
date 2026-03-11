import GGUFParser
import MLX
import MLXCompiler
import SwiftLM

// MARK: - TransformerModel + GGUFCompilableModel

extension TransformerModel: GGUFCompilableModel {
    package static func makeModelDeclaration(from file: GGUFFile) throws -> any ModelComponent {
        let config = try TransformerConfiguration(from: file)

        var moe: Transformer.MoEConfig? = nil
        if config.isMoE, let expertCount = config.expertCount,
           let expertsPerToken = config.expertUsedCount
        {
            moe = Transformer.MoEConfig(
                expertCount: expertCount,
                expertsPerToken: expertsPerToken
            )
        }

        return Transformer(config: .init(
            hiddenSize: config.hiddenSize,
            hiddenLayers: config.hiddenLayers,
            intermediateSize: config.intermediateSize,
            attentionHeads: config.attentionHeads,
            kvHeads: config.kvHeads,
            headDimension: config.headDimensions,
            vocabularySize: config.vocabularySize,
            normEps: config.normEps,
            attentionBias: config.attentionBias,
            mlpBias: config.mlpBias,
            ropeTheta: config.ropeTheta,
            moe: moe,
            tieWordEmbeddings: config.tieWordEmbeddings
        ))
    }
}

// MARK: - CohereModel + GGUFCompilableModel

extension CohereModel: GGUFCompilableModel {
    package static func makeModelDeclaration(from file: GGUFFile) throws -> any ModelComponent {
        let config = try CohereConfiguration(from: file)
        return CohereTransformer(config: .init(
            hiddenSize: config.hiddenSize,
            hiddenLayers: config.hiddenLayers,
            intermediateSize: config.intermediateSize,
            attentionHeads: config.attentionHeads,
            kvHeads: config.kvHeads,
            headDimension: config.headDimensions,
            vocabularySize: config.vocabularySize,
            normEps: config.layerNormEps,
            ropeTheta: config.ropeTheta,
            tieWordEmbeddings: config.tieWordEmbeddings,
            useQKNorm: config.useQKNorm
        ))
    }
}

// MARK: - Qwen35Model + GGUFCompilableModel

extension Qwen35Model: GGUFCompilableModel {
    package static func makeModelDeclaration(from file: GGUFFile) throws -> any ModelComponent {
        let config = try Qwen35Configuration(from: file)
        return Qwen35HybridTransformer(config: .init(
            hiddenSize: config.hiddenSize,
            hiddenLayers: config.hiddenLayers,
            intermediateSize: config.intermediateSize,
            vocabularySize: config.vocabularySize,
            normEps: config.normEps,
            attentionHeads: config.attentionHeads,
            kvHeads: config.kvHeads,
            headDimension: config.headDim,
            ropeTheta: config.ropeTheta,
            partialRotaryFactor: config.partialRotaryFactor,
            linearKeyHeads: config.linearKeyHeads,
            linearValueHeads: config.linearValueHeads,
            linearKeyHeadDim: config.linearKeyHeadDim,
            linearValueHeadDim: config.linearValueHeadDim,
            convKernelSize: config.convKernelSize,
            fullAttentionInterval: config.fullAttentionInterval,
            tieWordEmbeddings: config.tieWordEmbeddings
        ))
    }

    package static func sanitizeCompiledWeights(_ weights: [String: TensorData]) -> [String: TensorData] {
        var result = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        for key in Array(result.keys) where key.contains("conv1d.weight") {
            guard let td = result[key],
                  let storage = td.storage as? MLXTensorStorage
            else { continue }

            // Extract dense array — conv1d weights are small tensors where
            // quantization provides no benefit. If quantized, dequantize here
            // since LoweredDeltaNet requires dense conv1d weights.
            let array: MLXArray
            switch storage {
            case .dense(let a):
                array = a
            case .affineQuantized(let qt):
                array = dequantized(
                    qt.packedWeight, scales: qt.scales, biases: qt.zeroBiases,
                    groupSize: qt.groupSize, bits: qt.bits
                )
            }

            let reshaped: MLXArray
            if array.ndim == 2 {
                // GGUF [C, K] → MLX Conv1d [C, K, 1]
                reshaped = array.expandedDimensions(axis: -1)
            } else if array.ndim == 3 {
                // PyTorch [O, I/G, K] → MLX [O, K, I/G]
                reshaped = array.transposed(0, 2, 1)
            } else {
                continue
            }

            let newShape = (0..<reshaped.ndim).map { reshaped.dim($0) }
            result[key] = TensorData(
                shape: newShape,
                dtype: td.dtype,
                storage: MLXTensorStorage.dense(reshaped)
            )
        }
        return result
    }
}
