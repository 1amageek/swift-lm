import Foundation
import LMIR

/// A versioned, backend-neutral document consumed by the Core AI exporter.
public struct CoreAIExportDocument: Codable, Equatable, Sendable {

    public static let currentFormatVersion = 2

    public let formatVersion: Int
    public let metadata: Metadata
    public let program: CoreAIProgramContract
    public let rootRegion: Region

    public init(
        formatVersion: Int = CoreAIExportDocument.currentFormatVersion,
        metadata: Metadata,
        program: CoreAIProgramContract,
        rootRegion: Region
    ) {
        self.formatVersion = formatVersion
        self.metadata = metadata
        self.program = program
        self.rootRegion = rootRegion
    }

    public struct Metadata: Codable, Equatable, Sendable {
        public let name: String
        public let modelType: String
        public let target: Target
        public let maxContextLength: Int
        public let vocabSize: Int

        public init(
            name: String,
            modelType: String,
            target: Target,
            maxContextLength: Int,
            vocabSize: Int
        ) {
            self.name = name
            self.modelType = modelType
            self.target = target
            self.maxContextLength = maxContextLength
            self.vocabSize = vocabSize
        }
    }

    public enum Target: String, Codable, Equatable, Sendable {
        case macOSDynamic = "macos_dynamic"
        case iOSStatic = "ios_static"
    }

    public struct Region: Codable, Equatable, Sendable {
        public let parameters: [ValueID]
        public let operations: [Operation]
        public let results: [ValueID]

        public init(
            parameters: [ValueID],
            operations: [Operation],
            results: [ValueID]
        ) {
            self.parameters = parameters
            self.operations = operations
            self.results = results
        }
    }

    public struct Operation: Codable, Equatable, Sendable {
        public let key: Int
        public let operands: [ValueID]
        public let results: [ValueID]
        public let parameterBindings: [ParameterBinding]
        public let stateBindings: [StateBinding]
        public let kind: OperationKind

        public init(
            key: Int,
            operands: [ValueID],
            results: [ValueID],
            parameterBindings: [ParameterBinding],
            stateBindings: [StateBinding] = [],
            kind: OperationKind
        ) {
            self.key = key
            self.operands = operands
            self.results = results
            self.parameterBindings = parameterBindings
            self.stateBindings = stateBindings
            self.kind = kind
        }
    }

    public struct ValueID: Codable, Equatable, Hashable, Sendable {
        public let rawValue: Int

        public init(rawValue: Int) {
            self.rawValue = rawValue
        }

        public init(from decoder: Decoder) throws {
            rawValue = try decoder.singleValueContainer().decode(Int.self)
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            try container.encode(rawValue)
        }
    }

    public struct ParameterBinding: Codable, Equatable, Sendable {
        public let role: String
        public let tensorName: String

        public init(role: String, tensorName: String) {
            self.role = role
            self.tensorName = tensorName
        }
    }

    public struct StateBinding: Codable, Equatable, Sendable {
        public let role: String
        public let state: String
        public let axisIndex: Int

        public init(role: String, state: String, axisIndex: Int) {
            self.role = role
            self.state = state
            self.axisIndex = axisIndex
        }
    }

    public indirect enum OperationKind: Codable, Equatable, Sendable {
        case primitive(Primitive)
        case residual(strategy: ResidualStrategy, body: Region)
        case parallel(merge: ParallelMergeStrategy, branches: [Region])
        case repeating(count: Int, body: Region)
        case conditional(condition: ConditionKind, then: Region, else: Region)

        private enum CodingKeys: String, CodingKey {
            case tag
            case primitive
            case strategy
            case body
            case merge
            case branches
            case count
            case condition
            case then
            case `else`
        }

        private enum Tag: String, Codable {
            case primitive
            case residual
            case parallel
            case repeating
            case conditional
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            switch try container.decode(Tag.self, forKey: .tag) {
            case .primitive:
                self = .primitive(try container.decode(Primitive.self, forKey: .primitive))
            case .residual:
                self = .residual(
                    strategy: try container.decode(ResidualStrategy.self, forKey: .strategy),
                    body: try container.decode(Region.self, forKey: .body)
                )
            case .parallel:
                self = .parallel(
                    merge: try container.decode(ParallelMergeStrategy.self, forKey: .merge),
                    branches: try container.decode([Region].self, forKey: .branches)
                )
            case .repeating:
                self = .repeating(
                    count: try container.decode(Int.self, forKey: .count),
                    body: try container.decode(Region.self, forKey: .body)
                )
            case .conditional:
                self = .conditional(
                    condition: try container.decode(ConditionKind.self, forKey: .condition),
                    then: try container.decode(Region.self, forKey: .then),
                    else: try container.decode(Region.self, forKey: .else)
                )
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .primitive(let primitive):
                try container.encode(Tag.primitive, forKey: .tag)
                try container.encode(primitive, forKey: .primitive)
            case .residual(let strategy, let body):
                try container.encode(Tag.residual, forKey: .tag)
                try container.encode(strategy, forKey: .strategy)
                try container.encode(body, forKey: .body)
            case .parallel(let merge, let branches):
                try container.encode(Tag.parallel, forKey: .tag)
                try container.encode(merge, forKey: .merge)
                try container.encode(branches, forKey: .branches)
            case .repeating(let count, let body):
                try container.encode(Tag.repeating, forKey: .tag)
                try container.encode(count, forKey: .count)
                try container.encode(body, forKey: .body)
            case .conditional(let condition, let then, let `else`):
                try container.encode(Tag.conditional, forKey: .tag)
                try container.encode(condition, forKey: .condition)
                try container.encode(then, forKey: .then)
                try container.encode(`else`, forKey: .else)
            }
        }
    }

    public struct Primitive: Codable, Equatable, Sendable {
        public let opcode: String
        public let attributes: JSONValue

        public init(opcode: String, attributes: JSONValue) {
            self.opcode = opcode
            self.attributes = attributes
        }
    }

    /// JSON data used for stable primitive attribute payloads.
    public indirect enum JSONValue: Codable, Equatable, Sendable {
        case null
        case bool(Bool)
        case number(Double)
        case string(String)
        case array([JSONValue])
        case object([String: JSONValue])

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if container.decodeNil() {
                self = .null
            } else {
                do {
                    self = .bool(try container.decode(Bool.self))
                } catch {
                    do {
                        self = .number(try container.decode(Double.self))
                    } catch {
                        do {
                            self = .string(try container.decode(String.self))
                        } catch {
                            do {
                                self = .array(try container.decode([JSONValue].self))
                            } catch {
                                self = .object(try container.decode([String: JSONValue].self))
                            }
                        }
                    }
                }
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .null:
                try container.encodeNil()
            case .bool(let value):
                try container.encode(value)
            case .number(let value):
                try container.encode(value)
            case .string(let value):
                try container.encode(value)
            case .array(let value):
                try container.encode(value)
            case .object(let value):
                try container.encode(value)
            }
        }

        init(encodable value: any Encodable) throws {
            let data = try JSONEncoder().encode(EncodableBox(value: value))
            let object = try JSONSerialization.jsonObject(with: data)
            self = try Self(jsonObject: object)
        }

        private init(jsonObject object: Any) throws {
            switch object {
            case is NSNull:
                self = .null
            case let value as NSNumber:
                if CFGetTypeID(value) == CFBooleanGetTypeID() {
                    self = .bool(value.boolValue)
                } else {
                    self = .number(value.doubleValue)
                }
            case let value as String:
                self = .string(value)
            case let value as [Any]:
                self = .array(try value.map(Self.init(jsonObject:)))
            case let value as [String: Any]:
                self = .object(try value.mapValues(Self.init(jsonObject:)))
            default:
                throw CoreAIExportError.invalidAttributePayload(String(describing: type(of: object)))
            }
        }
    }
}

private struct EncodableBox: Encodable {
    let value: any Encodable

    func encode(to encoder: Encoder) throws {
        try value.encode(to: encoder)
    }
}
