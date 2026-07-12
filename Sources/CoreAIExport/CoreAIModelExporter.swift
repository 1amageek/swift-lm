import Foundation
import LMArchitecture
import LMIR

/// Converts declarative model graphs into the stable Core AI export document.
public struct CoreAIModelExporter: Sendable {

    public init() {}

    public func makeDocument(
        graph: ModelGraph,
        configuration: CoreAIExportConfiguration
    ) throws -> CoreAIExportDocument {
        do {
            try GraphValidator.validate(graph)
            try DimensionValidator.validate(graph)
        } catch {
            throw CoreAIExportError.invalidGraph(String(describing: error))
        }

        let canonicalGraph = canonicalize(graph)
        return CoreAIExportDocument(
            metadata: configuration.metadata,
            rootRegion: try encodeRegion(canonicalGraph.rootRegion)
        )
    }

    public func makeDocument<Component: ModelComponent>(
        component: Component,
        namingConvention: any WeightNamingConvention,
        configuration: CoreAIExportConfiguration
    ) throws -> CoreAIExportDocument {
        let normalized = try normalize(component)
        let resolvedGraph = ParameterResolver().resolve(
            graph: normalized.graph,
            convention: namingConvention
        )
        return try makeDocument(graph: resolvedGraph, configuration: configuration)
    }

    public func write(
        _ document: CoreAIExportDocument,
        to url: URL
    ) throws {
        guard document.formatVersion == CoreAIExportDocument.currentFormatVersion else {
            throw CoreAIExportError.unsupportedFormatVersion(document.formatVersion)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            let data = try encoder.encode(document)
            try data.write(to: url, options: [.atomic])
        } catch {
            throw CoreAIExportError.serializationFailed(String(describing: error))
        }
    }

    private func encodeRegion(_ region: Region) throws -> CoreAIExportDocument.Region {
        CoreAIExportDocument.Region(
            parameters: region.parameters.map { .init(rawValue: $0.id.rawValue) },
            operations: try region.operations.map(encodeOperation),
            results: region.results.map { .init(rawValue: $0.value.rawValue) }
        )
    }

    private func encodeOperation(_ operation: LMIR.Operation) throws -> CoreAIExportDocument.Operation {
        CoreAIExportDocument.Operation(
            key: operation.key.rawValue,
            operands: operation.operands.map { .init(rawValue: $0.value.rawValue) },
            results: operation.results.map { .init(rawValue: $0.id.rawValue) },
            parameterBindings: operation.parameterBindings.map {
                CoreAIExportDocument.ParameterBinding(
                    role: $0.role,
                    tensorName: $0.tensorName
                )
            },
            kind: try encodeKind(operation.kind)
        )
    }

    private func encodeKind(
        _ kind: OperationKind
    ) throws -> CoreAIExportDocument.OperationKind {
        switch kind {
        case .primitive(let attributes):
            return .primitive(try encodePrimitive(attributes))
        case .residual(let strategy, let body):
            return .residual(strategy: strategy, body: try encodeRegion(body))
        case .parallel(let merge, let branches):
            return .parallel(
                merge: merge,
                branches: try branches.map(encodeRegion)
            )
        case .repeating(let count, let body):
            guard count >= 0 else {
                throw CoreAIExportError.invalidGraph("repeat count must not be negative")
            }
            return .repeating(count: count, body: try encodeRegion(body))
        case .conditional(let condition, let then, let `else`):
            return .conditional(
                condition: condition,
                then: try encodeRegion(then),
                else: try encodeRegion(`else`)
            )
        }
    }

    private func encodePrimitive(
        _ attributes: any OperationAttributes
    ) throws -> CoreAIExportDocument.Primitive {
        let (opcode, encodable): (String, any Encodable)
        switch attributes {
        case let value as AttentionAttributes:
            opcode = "attention"
            encodable = value
        case let value as LayerScaleAttributes:
            opcode = "layer_scale"
            encodable = value
        case let value as LinearAttributes:
            opcode = "linear"
            encodable = value
        case let value as MLPAttributes:
            opcode = "mlp"
            encodable = value
        case let value as MoEAttributes:
            opcode = "moe"
            encodable = value
        case let value as RMSNormAttributes:
            opcode = "rms_norm"
            encodable = value
        case let value as LayerNormAttributes:
            opcode = "layer_norm"
            encodable = value
        case let value as OutputHeadAttributes:
            opcode = "output_head"
            encodable = value
        case let value as PatchEmbeddingAttributes:
            opcode = "patch_embedding"
            encodable = value
        case let value as PerLayerInputAttributes:
            opcode = "per_layer_input"
            encodable = value
        case let value as PoolingAttributes:
            opcode = "pooling"
            encodable = value
        case let value as PositionEmbeddingAttributes:
            opcode = "position_embedding"
            encodable = value
        case let value as RoPEAttributes:
            opcode = "rope"
            encodable = value
        case let value as ShortConvAttributes:
            opcode = "short_conv"
            encodable = value
        case let value as StandardizeAttributes:
            opcode = "standardize"
            encodable = value
        case let value as StateSpaceAttributes:
            opcode = "state_space"
            encodable = value
        case let value as TokenEmbeddingAttributes:
            opcode = "token_embedding"
            encodable = value
        default:
            throw CoreAIExportError.unsupportedPrimitive(
                String(reflecting: type(of: attributes))
            )
        }

        return CoreAIExportDocument.Primitive(
            opcode: opcode,
            attributes: try CoreAIExportDocument.JSONValue(encodable: encodable)
        )
    }
}
