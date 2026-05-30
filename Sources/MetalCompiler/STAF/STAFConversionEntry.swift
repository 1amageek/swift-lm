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
    let packedMoE: STAFPackedMoEEntry?

    init(
        name: String,
        sourceName: String,
        info: SafetensorsTensorInfo,
        shardIndex: Int,
        shardURL: URL,
        schemeIdentifier: QuantizationSchemeIdentifier,
        semanticRole: SemanticRole,
        originalDType: OriginalDType,
        packedMoE: STAFPackedMoEEntry? = nil
    ) {
        self.name = name
        self.sourceName = sourceName
        self.info = info
        self.shardIndex = shardIndex
        self.shardURL = shardURL
        self.schemeIdentifier = schemeIdentifier
        self.semanticRole = semanticRole
        self.originalDType = originalDType
        self.packedMoE = packedMoE
    }
}

struct STAFPackedMoEEntry: Sendable {
    enum Kind: Sendable {
        case gateUp
        case down
    }

    let kind: Kind
    let experts: [STAFPackedMoEExpertSources]
    let bulk: STAFPackedMoEBulkSources?

    init(
        kind: Kind,
        experts: [STAFPackedMoEExpertSources] = [],
        bulk: STAFPackedMoEBulkSources? = nil
    ) {
        self.kind = kind
        self.experts = experts
        self.bulk = bulk
    }

    var consumedTensorNames: [String] {
        if let bulk {
            return bulk.consumedTensorNames
        }
        return experts.flatMap { [$0.gate.name, $0.up.name, $0.down.name] }
    }
}

struct STAFPackedMoEExpertSources: Sendable {
    let gate: STAFPackedMoETensorSource
    let up: STAFPackedMoETensorSource
    let down: STAFPackedMoETensorSource
}

struct STAFPackedMoETensorSource: Sendable {
    let name: String
    let shardURL: URL
    let info: SafetensorsTensorInfo?
    let schemeIdentifier: QuantizationSchemeIdentifier?

    init(
        name: String,
        shardURL: URL,
        info: SafetensorsTensorInfo? = nil,
        schemeIdentifier: QuantizationSchemeIdentifier? = nil
    ) {
        self.name = name
        self.shardURL = shardURL
        self.info = info
        self.schemeIdentifier = schemeIdentifier
    }
}

struct STAFPackedMoEBulkSources: Sendable {
    let gate: STAFPackedMoETensorSource
    let up: STAFPackedMoETensorSource
    let down: STAFPackedMoETensorSource
    let expertCount: Int
    let intermediateDimension: Int
    let outputDimension: Int

    var consumedTensorNames: [String] {
        [gate, up, down].flatMap { source in
            let modulePath = source.name.hasSuffix(".weight")
                ? String(source.name.dropLast(".weight".count))
                : source.name
            return [
                source.name,
                modulePath + ".scales",
                modulePath + ".biases",
            ]
        }
    }
}
