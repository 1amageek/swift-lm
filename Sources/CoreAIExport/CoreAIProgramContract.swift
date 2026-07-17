/// Core AI function signatures and persistent-state layout derived from Swift LMIR.
public struct CoreAIProgramContract: Codable, Equatable, Sendable {
    public let source: Source
    public let execution: Execution
    public let functions: [Function]

    public init(
        source: Source = .swiftLMIR,
        execution: Execution,
        functions: [Function]
    ) {
        self.source = source
        self.execution = execution
        self.functions = functions
    }

    public enum Source: String, Codable, Equatable, Sendable {
        case swiftLMIR = "swift_lmir"
    }

    public enum Execution: String, Codable, Equatable, Sendable {
        case stateless
        case stateful
    }

    public struct Function: Codable, Equatable, Sendable {
        public let name: String
        public let inputs: [Tensor]
        public let outputs: [Tensor]
        public let states: [Tensor]

        public init(
            name: String,
            inputs: [Tensor],
            outputs: [Tensor],
            states: [Tensor]
        ) {
            self.name = name
            self.inputs = inputs
            self.outputs = outputs
            self.states = states
        }
    }

    public struct Tensor: Codable, Equatable, Sendable {
        public let name: String
        public let dataType: DataType
        public let dimensions: [Dimension]

        public init(name: String, dataType: DataType, dimensions: [Dimension]) {
            self.name = name
            self.dataType = dataType
            self.dimensions = dimensions
        }
    }

    public enum DataType: String, Codable, Equatable, Sendable {
        case int32
        case float16
        case bfloat16
        case float32
    }

    public struct Dimension: Codable, Equatable, Sendable {
        public let kind: Kind
        public let size: Int?
        public let symbol: String?
        public let minimum: Int?
        public let maximum: Int?

        public static func fixed(_ size: Int) -> Dimension {
            Dimension(kind: .fixed, size: size)
        }

        public static func dynamic(
            _ symbol: String,
            minimum: Int,
            maximum: Int
        ) -> Dimension {
            Dimension(
                kind: .dynamic,
                symbol: symbol,
                minimum: minimum,
                maximum: maximum
            )
        }

        private init(
            kind: Kind,
            size: Int? = nil,
            symbol: String? = nil,
            minimum: Int? = nil,
            maximum: Int? = nil
        ) {
            self.kind = kind
            self.size = size
            self.symbol = symbol
            self.minimum = minimum
            self.maximum = maximum
        }

        public enum Kind: String, Codable, Equatable, Sendable {
            case fixed
            case dynamic
        }
    }
}
