import Foundation

/// Canonicalizes tensor names from source-specific conventions (e.g. MLX VLM)
/// into the HuggingFace form expected by `WeightNamingConvention` implementations.
///
/// Without this, MLX-community VLM bundles (Qwen3.5-VL, Gemma4-VL, ...) fail
/// to bind: MLX packages the text backbone under `language_model.model.*`
/// while HuggingFace uses `model.language_model.*`. The mismatch is per-source
/// convention, not per-model, so a single rewrite rule fixes every current and
/// future MLX VLM bundle.
///
/// The canonicalizer runs once at STAF conversion time. Downstream consumers
/// (STAFLoader, ParameterResolver, runtime weight lookup) see only canonical
/// names.
public struct TensorNameCanonicalizer: Sendable {

    /// A prefix-substitution rule: if a tensor name starts with `from`,
    /// replace that prefix with `to`.
    public struct Rule: Sendable {
        public let from: String
        public let to: String

        public init(from: String, to: String) {
            self.from = from
            self.to = to
        }
    }

    public let rules: [Rule]

    public init(rules: [Rule]) {
        self.rules = rules
    }

    /// Apply the first matching prefix rule, or return `name` unchanged.
    public func canonicalize(_ name: String) -> String {
        for rule in rules {
            if name.hasPrefix(rule.from) {
                return rule.to + name.dropFirst(rule.from.count)
            }
        }
        return name
    }
}

extension TensorNameCanonicalizer {

    /// No-op canonicalizer for bundles already in HuggingFace form.
    public static let identity = TensorNameCanonicalizer(rules: [])

    /// MLX VLM → HuggingFace VLM convention.
    ///
    /// Rewrites the text-backbone prefix:
    /// - `language_model.model.*` → `model.language_model.*`
    ///
    /// Vision-tower tensors (`visual.*`) are left untouched; vision bundles
    /// already align across conventions for the Qwen3.5 / Gemma4 families.
    public static let mlxVLMToHuggingFace = TensorNameCanonicalizer(
        rules: [
            Rule(from: "language_model.model.", to: "model.language_model."),
        ]
    )

    /// Detect the source convention from observed tensor names.
    ///
    /// Returns `mlxVLMToHuggingFace` if any name carries the MLX-style
    /// `language_model.model.` prefix; `identity` otherwise. The two
    /// conventions are mutually exclusive, so a single probe is sufficient.
    public static func detect<S: Sequence>(from names: S) -> TensorNameCanonicalizer
    where S.Element == String {
        for name in names {
            if name.hasPrefix("language_model.model.") {
                return .mlxVLMToHuggingFace
            }
        }
        return .identity
    }
}
