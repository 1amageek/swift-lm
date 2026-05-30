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
    public enum RewriteMode: Sendable {
        case prefixOnly
        case lfm2MLX
    }

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
    public let mode: RewriteMode

    public init(rules: [Rule], mode: RewriteMode = .prefixOnly) {
        self.rules = rules
        self.mode = mode
    }

    /// Apply the first matching prefix rule, or return `name` unchanged.
    public func canonicalize(_ name: String) -> String {
        let name = rewriteByMode(name)
        for rule in rules {
            if name.hasPrefix(rule.from) {
                return rule.to + name.dropFirst(rule.from.count)
            }
        }
        return name
    }

    private func rewriteByMode(_ name: String) -> String {
        switch mode {
        case .prefixOnly:
            return name
        case .lfm2MLX:
            return Self.rewriteLFM2MLXName(name)
        }
    }

    private static func rewriteLFM2MLXName(_ name: String) -> String {
        name
            .replacingOccurrences(of: ".feed_forward.gate_proj.", with: ".feed_forward.w1.")
            .replacingOccurrences(of: ".feed_forward.up_proj.", with: ".feed_forward.w3.")
            .replacingOccurrences(of: ".feed_forward.down_proj.", with: ".feed_forward.w2.")
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

    /// LFM2 MLX text convention → HuggingFace LFM2 convention.
    ///
    /// LiquidAI's MLX 8-bit A1B bundle stores the dense FFN projections as
    /// `gate_proj` / `up_proj` / `down_proj`; the HF declaration used by
    /// swift-lm expects `w1` / `w3` / `w2`.
    public static let lfm2MLXToHuggingFace = TensorNameCanonicalizer(
        rules: [],
        mode: .lfm2MLX
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
            if name.contains(".feed_forward.switch_mlp.") {
                return .lfm2MLXToHuggingFace
            }
        }
        return .identity
    }
}
