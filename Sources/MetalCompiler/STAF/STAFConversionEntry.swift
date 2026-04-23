import Foundation

struct STAFConversionEntry: Sendable {
    /// Canonical tensor name written to STAF metadata and observed by
    /// downstream consumers (STAFLoader, ParameterResolver, runtime lookup).
    let name: String
    /// Original tensor name in the source safetensors shard. Retained so
    /// payload repacking can still resolve companion tensors (`.scales`,
    /// `.biases`) against their on-disk names after canonicalization rewrites
    /// `name`.
    let sourceName: String
    let info: SafetensorsTensorInfo
    let shardIndex: Int
    let shardURL: URL
    let schemeIdentifier: QuantizationSchemeIdentifier
    let semanticRole: SemanticRole
    let originalDType: OriginalDType
}
