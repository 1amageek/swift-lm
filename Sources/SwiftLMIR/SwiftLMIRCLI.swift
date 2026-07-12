import CoreAIExport
import Foundation
import LMIR
import ModelDeclarations

@main
struct SwiftLMIRCLI {
    static func main() {
        do {
            let arguments = try Arguments(commandLine: Array(CommandLine.arguments.dropFirst()))
            let configuration = try ConfigDecoder().decode(from: arguments.configURL)
            let exportConfiguration = try CoreAIExportConfiguration(
                name: arguments.name,
                modelType: configuration.modelType,
                target: arguments.target,
                maxContextLength: arguments.maxContextLength ?? configuration.maxContextLength,
                vocabSize: configuration.modelConfig.vocabSize
            )
            let exporter = CoreAIModelExporter()
            let document: CoreAIExportDocument
            switch configuration.modelType {
            case "llama", "qwen2", "qwen3", "mistral":
                document = try exporter.makeDocument(
                    component: Transformer(config: configuration.modelConfig),
                    namingConvention: LlamaFamilyNaming(),
                    configuration: exportConfiguration
                )
            case "lfm2", "lfm2_moe":
                document = try exporter.makeDocument(
                    component: LFM2(config: configuration.modelConfig),
                    namingConvention: LFM2FamilyNaming(),
                    configuration: exportConfiguration
                )
            default:
                throw CLIError.unsupportedModelType(configuration.modelType)
            }
            try exporter.write(document, to: arguments.outputURL)
            print(arguments.outputURL.path)
        } catch {
            FileHandle.standardError.write(Data("swiftlm-ir: \(error)\n".utf8))
            Foundation.exit(1)
        }
    }

    private struct Arguments {
        let configURL: URL
        let outputURL: URL
        let name: String
        let target: CoreAIExportDocument.Target
        let maxContextLength: Int?

        init(commandLine: [String]) throws {
            var values = commandLine
            configURL = try Self.requiredURL(flag: "--config", values: &values)
            outputURL = try Self.requiredURL(flag: "--output", values: &values)
            name = try Self.requiredString(flag: "--name", values: &values)
            let targetValue = try Self.requiredString(flag: "--target", values: &values)
            switch targetValue {
            case "macos":
                target = .macOSDynamic
            case "ios":
                target = .iOSStatic
            default:
                throw CLIError.invalidArgument("--target must be macos or ios")
            }
            if let value = Self.optionalString(flag: "--max-context-length", values: &values) {
                guard let length = Int(value), length > 0 else {
                    throw CLIError.invalidArgument("--max-context-length must be positive")
                }
                maxContextLength = length
            } else {
                maxContextLength = nil
            }
            guard values.isEmpty else {
                throw CLIError.invalidArgument("unknown arguments: \(values.joined(separator: " "))")
            }
        }

        private static func requiredURL(flag: String, values: inout [String]) throws -> URL {
            URL(fileURLWithPath: try requiredString(flag: flag, values: &values))
        }

        private static func requiredString(flag: String, values: inout [String]) throws -> String {
            guard let index = values.firstIndex(of: flag) else {
                throw CLIError.invalidArgument("missing \(flag)")
            }
            let valueIndex = values.index(after: index)
            guard valueIndex < values.endIndex else {
                throw CLIError.invalidArgument("missing value for \(flag)")
            }
            let value = values[valueIndex]
            values.removeSubrange(index...valueIndex)
            return value
        }

        private static func optionalString(flag: String, values: inout [String]) -> String? {
            guard let index = values.firstIndex(of: flag) else { return nil }
            let valueIndex = values.index(after: index)
            guard valueIndex < values.endIndex else { return nil }
            let value = values[valueIndex]
            values.removeSubrange(index...valueIndex)
            return value
        }
    }

    private struct DecodedConfiguration {
        let modelType: String
        let modelConfig: ModelConfig
        let maxContextLength: Int
    }

    private struct ConfigDecoder {
        func decode(from url: URL) throws -> DecodedConfiguration {
            let data: Data
            do {
                data = try Data(contentsOf: url)
            } catch {
                throw CLIError.configReadFailed(url, String(describing: error))
            }
            let raw: [String: Any]
            do {
                guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    throw CLIError.invalidConfig("config.json must be an object")
                }
                raw = object
            } catch let error as CLIError {
                throw error
            } catch {
                throw CLIError.invalidConfig(String(describing: error))
            }

            var json = raw
            if let textConfig = raw["text_config"] as? [String: Any] {
                json.merge(textConfig) { _, nested in nested }
            }
            guard let modelType = json["model_type"] as? String else {
                throw CLIError.invalidConfig("model_type is required")
            }
            let hiddenSize = try requiredInt("hidden_size", json)
            let layerCount = try requiredInt("num_hidden_layers", json)
            let vocabSize = try requiredInt("vocab_size", json)
            let attentionHeads = try requiredInt("num_attention_heads", json)
            let intermediateSize = try jsonInt("intermediate_size", json)
                ?? jsonInt("block_ff_dim", json)
                ?? { throw CLIError.invalidConfig("intermediate_size or block_ff_dim is required") }()
            let kvHeads = jsonInt("num_key_value_heads", json) ?? attentionHeads
            let headDim = jsonInt("head_dim", json) ?? (hiddenSize / attentionHeads)
            let normEps = jsonDouble("rms_norm_eps", json)
                ?? jsonDouble("layer_norm_eps", json)
                ?? jsonDouble("norm_eps", json)
                ?? 1e-6
            let ropeParameters = json["rope_parameters"] as? [String: Any]
            let ropeScaling = json["rope_scaling"] as? [String: Any]
            let ropeTheta = jsonDouble("rope_theta", json)
                ?? ropeParameters.flatMap { jsonDouble("rope_theta", $0) }
                ?? ropeScaling.flatMap { jsonDouble("rope_theta", $0) }
                ?? 10_000
            let modelConfig = ModelConfig(
                hiddenSize: hiddenSize,
                layerCount: layerCount,
                intermediateSize: intermediateSize,
                vocabSize: vocabSize,
                attentionHeads: attentionHeads,
                kvHeads: kvHeads,
                headDim: headDim,
                attentionBias: jsonBool("attention_bias", json) ?? false,
                mlpBias: jsonBool("mlp_bias", json) ?? false,
                normEps: Float(normEps),
                normKind: modelType == "cohere" ? .layerNorm : .rmsNorm,
                ropeTheta: Float(ropeTheta),
                ropeDimension: jsonInt("rope_dim", json) ?? headDim,
                ropeScaling: nil,
                tiedEmbeddings: jsonBool("tie_word_embeddings", json) ?? false,
                expertCount: jsonInt("num_local_experts", json) ?? jsonInt("num_experts", json),
                expertsPerToken: jsonInt("num_experts_per_tok", json),
                moeIntermediateSize: jsonInt("moe_intermediate_size", json),
                moeNormalizeRoutingWeights: jsonBool("norm_topk_prob", json) ?? false,
                moeRoutedScalingFactor: Float(jsonDouble("routed_scaling_factor", json) ?? 1.0),
                moeUseExpertBias: jsonBool("use_expert_bias", json) ?? false,
                qkNorm: jsonBool("qk_norm", json) ?? ["lfm2", "lfm2_moe"].contains(modelType),
                fullAttentionInterval: jsonInt("full_attention_interval", json),
                ssmNumHeads: jsonInt("ssm_num_heads", json) ?? jsonInt("linear_num_value_heads", json),
                ssmGroupCount: jsonInt("linear_num_key_heads", json),
                ssmKeyHeadDim: jsonInt("ssm_state_size", json) ?? jsonInt("linear_key_head_dim", json),
                ssmValueHeadDim: jsonInt("ssm_state_size", json) ?? jsonInt("linear_value_head_dim", json),
                convKernelSize: jsonInt("conv_kernel_size", json) ?? jsonInt("linear_conv_kernel_dim", json),
                convLCache: jsonInt("conv_L_cache", json),
                partialRotaryFactor: jsonDouble("partial_rotary_factor", json).map(Float.init),
                slidingWindow: jsonInt("sliding_window", json),
                layerTypes: json["layer_types"] as? [String],
                finalLogitSoftcapping: jsonDouble("final_logit_softcapping", json).map(Float.init),
                numDenseLayers: jsonInt("num_dense_layers", json) ?? 0
            )
            return DecodedConfiguration(
                modelType: modelType.lowercased(),
                modelConfig: modelConfig,
                maxContextLength: jsonInt("max_position_embeddings", json) ?? 2048
            )
        }

        private func requiredInt(_ key: String, _ json: [String: Any]) throws -> Int {
            guard let value = jsonInt(key, json) else {
                throw CLIError.invalidConfig("\(key) is required")
            }
            return value
        }

        private func jsonInt(_ key: String, _ json: [String: Any]) -> Int? {
            if let value = json[key] as? Int { return value }
            if let value = json[key] as? NSNumber { return value.intValue }
            return nil
        }

        private func jsonDouble(_ key: String, _ json: [String: Any]) -> Double? {
            if let value = json[key] as? Double { return value }
            if let value = json[key] as? NSNumber { return value.doubleValue }
            return nil
        }

        private func jsonBool(_ key: String, _ json: [String: Any]) -> Bool? {
            json[key] as? Bool
        }
    }

    private enum CLIError: Error, CustomStringConvertible {
        case invalidArgument(String)
        case invalidConfig(String)
        case configReadFailed(URL, String)
        case unsupportedModelType(String)

        var description: String {
            switch self {
            case .invalidArgument(let message): return message
            case .invalidConfig(let message): return "invalid config: \(message)"
            case .configReadFailed(let url, let message): return "could not read \(url.path): \(message)"
            case .unsupportedModelType(let type): return "unsupported model_type: \(type)"
            }
        }
    }
}
