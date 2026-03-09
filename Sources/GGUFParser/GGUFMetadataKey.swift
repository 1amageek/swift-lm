/// Type-safe key for accessing GGUF metadata values.
///
/// Keys encode both the metadata path and the expected value type,
/// eliminating stringly-typed metadata access.
///
/// ```swift
/// let embed = try file.require(.embeddingLength)  // Int
/// let theta = file[.ropeFreqBase]                 // Float?
/// ```
package struct GGUFMetadataKey<T: Sendable>: Sendable {

    /// Determines how the key path is resolved to a full metadata key.
    package enum Scope: Sendable {
        /// `"general.{path}"` — model-level metadata.
        case global
        /// `"{arch}.{path}"` — architecture-specific, resolved at lookup time.
        case architecture
        /// `"tokenizer.ggml.{path}"` — tokenizer metadata.
        case tokenizer
        /// Path used as-is without prefix.
        case raw
    }

    package let path: String
    package let scope: Scope
    package let extract: @Sendable (GGUFMetadataValue) -> T?

    package init(path: String, scope: Scope, extract: @Sendable @escaping (GGUFMetadataValue) -> T?) {
        self.path = path
        self.scope = scope
        self.extract = extract
    }
}

// MARK: - GGUFFile subscript + require

extension GGUFFile {

    /// Resolve a key's full metadata path based on its scope.
    private func resolvedKey<T>(for key: GGUFMetadataKey<T>) -> String {
        switch key.scope {
        case .global:
            return "general.\(key.path)"
        case .architecture:
            let arch = architecture ?? ""
            return "\(arch).\(key.path)"
        case .tokenizer:
            return "tokenizer.ggml.\(key.path)"
        case .raw:
            return key.path
        }
    }

    /// Look up a typed metadata value. Returns `nil` if the key is missing
    /// or the value cannot be extracted as the expected type.
    package subscript<T>(key: GGUFMetadataKey<T>) -> T? {
        let fullKey = resolvedKey(for: key)
        guard let value = metadata[fullKey] else { return nil }
        return key.extract(value)
    }

    /// Look up a required metadata value. Throws if missing or wrong type.
    package func require<T>(_ key: GGUFMetadataKey<T>) throws -> T {
        guard let value = self[key] else {
            let fullKey = resolvedKey(for: key)
            throw GGUFError.metadataKeyNotFound(fullKey)
        }
        return value
    }
}
