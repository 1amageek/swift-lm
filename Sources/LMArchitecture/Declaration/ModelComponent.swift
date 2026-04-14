import LMIR

/// A declarative structural building block for model topology.
///
/// `ModelComponent` is the user-facing abstraction for defining model structure,
/// analogous to SwiftUI's `View` protocol. It is open for extension — users can
/// define custom composite components that compose existing ones.
///
/// Every component declares two associated types:
///
/// - **`Attributes`**: The IR node type this component produces.
///   Primitive components specify a concrete `OperationAttributes` type.
///   Composite components leave it as the default `Never`.
///
/// - **`Body`**: The composed component body.
///   Primitive components leave it as the default `Never`.
///   Composite components define `body` to compose other components.
///
/// ```swift
/// // Primitive: Attributes = concrete, Body = Never
/// struct RMSNorm: ModelComponent {
///     typealias Attributes = RMSNormAttributes
///     let dimension: Int
///     var attributes: RMSNormAttributes {
///         RMSNormAttributes(dimension: dimension)
///     }
/// }
///
/// // Composite: Attributes = Never, Body = inferred
/// struct TransformerBlock: ModelComponent {
///     let hiddenSize: Int
///     var body: some ModelComponent {
///         Residual {
///             RMSNorm(dimension: hiddenSize)
///             Attention(...)
///         }
///     }
/// }
/// ```
public protocol ModelComponent: Sendable {

    /// The IR node type this component produces.
    ///
    /// Primitive components specify a concrete `OperationAttributes` type
    /// (e.g., `RMSNormAttributes`). Composite components use the default `Never`.
    associatedtype Attributes: OperationAttributes, Sendable, Codable = Never

    /// The type of the composed body. Primitive components use the default `Never`.
    associatedtype Body: ModelComponent = Never

    /// The concrete attributes instance for this component.
    ///
    /// Primitive components (where `Attributes != Never`) provide this to
    /// produce the IR `OperationKind.primitive(attributes)`.
    /// Composite components use the default (which traps).
    var attributes: Attributes { get }

    /// The arity signature for this component.
    ///
    /// Default: 1 operand → 1 result (unary). Override for special cases
    /// (e.g., TokenEmbedding: 0→1).
    var operationSignature: OperationSignature { get }

    /// Declarative structural body composed from other components.
    ///
    /// Composite components define this to compose child components.
    /// Primitive components (where `Body == Never`) get a default
    /// implementation that traps — they are handled directly by
    /// the normalizer.
    @ModelComponentBuilder var body: Body { get }
}

// MARK: - Defaults

extension ModelComponent where Attributes == Never {

    /// Composite components have no attributes — accessing them is a programming error.
    public var attributes: Never {
        fatalError("\(type(of: self)) is a composite ModelComponent and has no attributes")
    }
}

extension ModelComponent {

    /// Default signature: unary primitive (1 operand, 1 result).
    public var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}

extension ModelComponent where Body == Never {

    /// Primitive components have no body — accessing it is a programming error.
    public var body: Never {
        fatalError("\(type(of: self)) is a primitive ModelComponent and has no body")
    }
}

// MARK: - Never Conformance

extension Never: ModelComponent {
    public typealias Body = Never
    public typealias Attributes = Never
}
