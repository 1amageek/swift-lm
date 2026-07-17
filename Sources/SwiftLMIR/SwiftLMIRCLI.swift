import CoreAIExport
import Foundation
import LMIR
import ModelDeclarations

@main
struct SwiftLMIRCLI {
    static func main() {
        do {
            let arguments = try Arguments(commandLine: Array(CommandLine.arguments.dropFirst()))
            let configuration: HuggingFaceModelConfiguration
            do {
                configuration = try HuggingFaceConfigDecoder().decode(from: arguments.configURL)
            } catch let error as HuggingFaceConfigError {
                throw CLIError.invalidConfig(error.description)
            }
            let exportConfiguration = try CoreAIExportConfiguration(
                name: arguments.name,
                modelType: configuration.modelType,
                target: arguments.target,
                maxContextLength: arguments.maxContextLength ?? configuration.maxContextLength,
                vocabSize: configuration.modelConfig.vocabSize,
                execution: arguments.execution
            )
            let exporter = CoreAIModelExporter()
            let graph = try ModelFamilyRegistry.resolveModelGraph(
                modelType: configuration.modelType,
                config: configuration.modelConfig
            )
            let resolvedGraph = ParameterResolver().resolve(
                graph: graph,
                convention: try ModelFamilyRegistry.namingConvention(for: configuration.modelType)
            )
            let document = try exporter.makeDocument(
                graph: resolvedGraph,
                configuration: exportConfiguration
            )
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
        let execution: CoreAIProgramContract.Execution

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
            if let index = values.firstIndex(of: "--stateful") {
                values.remove(at: index)
                execution = .stateful
            } else {
                execution = .stateless
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

    private enum CLIError: Error, CustomStringConvertible {
        case invalidArgument(String)
        case invalidConfig(String)

        var description: String {
            switch self {
            case .invalidArgument(let message): return message
            case .invalidConfig(let message): return "invalid config: \(message)"
            }
        }
    }
}
