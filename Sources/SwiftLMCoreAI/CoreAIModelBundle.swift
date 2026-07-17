import CoreAI
import CoreAIExport
import CryptoKit
import Foundation

/// A Core AI language bundle whose executable contract originates in Swift LMIR.
@available(macOS 27.0, iOS 27.0, *)
public struct CoreAIModelBundle: Sendable {
    public let url: URL
    public let name: String
    public let maxContextLength: Int
    public let asset: CoreAIModelAsset
    public let document: CoreAIExportDocument

    public init(contentsOf url: URL) throws {
        let metadataURL = url.appendingPathComponent("metadata.json")
        let metadata: Metadata
        do {
            metadata = try JSONDecoder().decode(
                Metadata.self,
                from: Data(contentsOf: metadataURL)
            )
        } catch {
            throw CoreAIModelAssetError.invalidBundle(
                url,
                "metadata.json could not be decoded: \(error)"
            )
        }
        guard metadata.metadataVersion == "0.2",
              metadata.kind == "llm",
              metadata.source.modelDefinition == "swift_lmir",
              metadata.source.formatVersion == CoreAIExportDocument.currentFormatVersion else {
            throw CoreAIModelAssetError.invalidBundle(
                url,
                "bundle metadata does not declare the Swift LMIR v2 contract"
            )
        }

        let assetURL = try Self.resolve(metadata.assets.main, inside: url)
        let contractURL = try Self.resolve(metadata.assets.contract, inside: url)
        let contractData: Data
        do {
            contractData = try Data(contentsOf: contractURL)
        } catch {
            throw CoreAIModelAssetError.invalidBundle(
                url,
                "Swift LMIR contract could not be read: \(error)"
            )
        }
        let contractHash = SHA256.hash(data: contractData)
            .map { String(format: "%02x", $0) }
            .joined()
        guard contractHash == metadata.assets.contractSHA256 else {
            throw CoreAIModelAssetError.invalidBundle(
                url,
                "Swift LMIR contract hash does not match metadata"
            )
        }
        let document: CoreAIExportDocument
        do {
            document = try JSONDecoder().decode(
                CoreAIExportDocument.self,
                from: contractData
            )
        } catch {
            throw CoreAIModelAssetError.invalidBundle(
                url,
                "Swift LMIR contract could not be decoded: \(error)"
            )
        }
        guard document.formatVersion == CoreAIExportDocument.currentFormatVersion,
              document.program.source == .swiftLMIR,
              document.metadata.name == metadata.name,
              document.metadata.maxContextLength == metadata.language.maxContextLength,
              document.metadata.vocabSize == metadata.language.vocabSize else {
            throw CoreAIModelAssetError.invalidBundle(
                url,
                "bundle metadata and Swift LMIR contract disagree"
            )
        }

        let asset = try CoreAIModelAsset(contentsOf: assetURL)
        try Self.validate(document: document, asset: asset)
        self.url = url
        self.name = metadata.name
        self.maxContextLength = metadata.language.maxContextLength
        self.asset = asset
        self.document = document
    }

    public func makeStateSession(
        functionName: String = "main",
        maxContextLength: Int? = nil,
        options: SpecializationOptions = .default,
        cache: AIModelCache = .default,
        cachePolicy: AIModelCache.Policy = .default
    ) async throws -> CoreAIStateSession {
        guard document.program.execution == .stateful else {
            throw CoreAIModelAssetError.invalidBundle(
                url,
                "a state session requires a stateful Swift LMIR contract"
            )
        }
        let function = try contractFunction(named: functionName)
        let resolvedContextLength = maxContextLength ?? self.maxContextLength
        guard resolvedContextLength > 0,
              resolvedContextLength <= self.maxContextLength else {
            throw CoreAIModelAssetError.invalidBundle(
                url,
                "requested context length is outside the bundle contract"
            )
        }
        let model = try await asset.specialize(
            options: options,
            cache: cache,
            cachePolicy: cachePolicy
        )
        return try CoreAIStateSession(
            model: model,
            functionName: functionName,
            stateShapes: Dictionary(uniqueKeysWithValues: function.states.map { state in
                (state.name, state.dimensions.map { dimension in
                    switch dimension.kind {
                    case .fixed:
                        return dimension.size ?? 0
                    case .dynamic:
                        return resolvedContextLength
                    }
                })
            })
        )
    }

    public func makeStatelessSession(
        functionName: String = "main",
        options: SpecializationOptions = .default,
        cache: AIModelCache = .default,
        cachePolicy: AIModelCache.Policy = .default
    ) async throws -> CoreAIStatelessSession {
        guard document.program.execution == .stateless else {
            throw CoreAIModelAssetError.invalidBundle(
                url,
                "a stateless session requires a stateless Swift LMIR contract"
            )
        }
        _ = try contractFunction(named: functionName)
        let model = try await asset.specialize(
            options: options,
            cache: cache,
            cachePolicy: cachePolicy
        )
        return try CoreAIStatelessSession(
            model: model,
            functionName: functionName
        )
    }

    private func contractFunction(named name: String) throws -> CoreAIProgramContract.Function {
        guard let function = document.program.functions.first(where: { $0.name == name }) else {
            throw CoreAIModelAssetError.functionNotFound(name)
        }
        return function
    }

    private static func validate(
        document: CoreAIExportDocument,
        asset: CoreAIModelAsset
    ) throws {
        for function in document.program.functions {
            let descriptor = try asset.function(named: function.name)
            try validate(
                descriptor.inputs,
                contract: function.inputs,
                function: function.name,
                category: "inputs"
            )
            try validate(
                descriptor.states,
                contract: function.states,
                function: function.name,
                category: "states"
            )
            try validate(
                descriptor.outputs,
                contract: function.outputs,
                function: function.name,
                category: "outputs"
            )
        }
    }

    private static func validate(
        _ descriptors: [AIModelAsset.ValueDescriptor],
        contract: [CoreAIProgramContract.Tensor],
        function: String,
        category: String
    ) throws {
        guard descriptors.count == contract.count else {
            throw CoreAIModelAssetError.contractMismatch(
                function: function,
                message: "\(category) count is \(descriptors.count), expected \(contract.count)"
            )
        }
        for (descriptor, tensor) in zip(descriptors, contract) {
            guard descriptor.name == tensor.name else {
                throw CoreAIModelAssetError.contractMismatch(
                    function: function,
                    message: "\(category) contains '\(descriptor.name)', expected '\(tensor.name)'"
                )
            }
            let expectedType = typeName(for: tensor)
            guard descriptor.typeName == expectedType else {
                throw CoreAIModelAssetError.contractMismatch(
                    function: function,
                    message: "\(category).\(tensor.name) is \(descriptor.typeName), expected \(expectedType)"
                )
            }
        }
    }

    private static func typeName(for tensor: CoreAIProgramContract.Tensor) -> String {
        let dataType: String
        switch tensor.dataType {
        case .int32:
            dataType = "Int32"
        case .float16:
            dataType = "Float16"
        case .bfloat16:
            dataType = "BFloat16"
        case .float32:
            dataType = "Float32"
        }
        let dimensions = tensor.dimensions.map { dimension in
            switch dimension.kind {
            case .fixed:
                return String(dimension.size ?? 0)
            case .dynamic:
                return "?"
            }
        }
        return "NDArray (\(dataType), \(dimensions.joined(separator: " \u{00D7} ")))"
    }

    private static func resolve(_ path: String, inside bundle: URL) throws -> URL {
        let bundlePath = bundle.standardizedFileURL.path
        let resolved = bundle.appendingPathComponent(path).standardizedFileURL
        guard resolved.path == bundlePath || resolved.path.hasPrefix(bundlePath + "/") else {
            throw CoreAIModelAssetError.invalidBundle(bundle, "asset path escapes the bundle")
        }
        return resolved
    }
}

@available(macOS 27.0, iOS 27.0, *)
private extension CoreAIModelBundle {
    struct Metadata: Decodable {
        let metadataVersion: String
        let kind: String
        let name: String
        let assets: Assets
        let language: Language
        let source: Source

        enum CodingKeys: String, CodingKey {
            case metadataVersion = "metadata_version"
            case kind
            case name
            case assets
            case language
            case source
        }
    }

    struct Assets: Decodable {
        let main: String
        let contract: String
        let contractSHA256: String

        enum CodingKeys: String, CodingKey {
            case main
            case contract
            case contractSHA256 = "contract_sha256"
        }
    }

    struct Language: Decodable {
        let maxContextLength: Int
        let vocabSize: Int

        enum CodingKeys: String, CodingKey {
            case maxContextLength = "max_context_length"
            case vocabSize = "vocab_size"
        }
    }

    struct Source: Decodable {
        let modelDefinition: String
        let formatVersion: Int

        enum CodingKeys: String, CodingKey {
            case modelDefinition = "model_definition"
            case formatVersion = "format_version"
        }
    }
}
