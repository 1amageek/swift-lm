import Foundation

struct STAFConversionEntry: Sendable {
    let name: String
    let info: SafetensorsTensorInfo
    let shardIndex: Int
    let shardURL: URL
    let schemeIdentifier: QuantizationSchemeIdentifier
    let semanticRole: SemanticRole
    let originalDType: OriginalDType
}
