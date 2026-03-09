import Foundation

/// Diagnostic report capturing facts from each stage of the GGUF loading pipeline.
///
/// Assembled imperatively during `GGUFModelLoader.loadContext()` as each stage
/// produces facts. Stored on `ModelContext.loadReport` when loaded via the
/// standard pipeline; `nil` when `ModelContext` is constructed manually.
///
/// ```swift
/// let container = try GGUFModelLoader().load(url: ggufURL)
/// try container.perform { context in
///     if let report = context.loadReport {
///         print(report.summary)
///     }
/// }
/// ```
public struct LoadReport: Sendable {

    /// Which model type was selected from the candidate list.
    public let modelResolution: ModelResolution

    /// Tensor mapping statistics from GGUF → MLX weight paths.
    public let weightLoading: WeightLoading

    /// Quantization configuration applied to the model.
    public let quantization: QuantizationApplied

    // MARK: - Model Resolution

    /// Records the outcome of data-driven model type selection.
    public struct ModelResolution: Sendable {

        /// Swift type name of the selected model (e.g., `"CohereModel"`).
        public let selectedType: String

        /// Number of candidates evaluated before a match was found.
        ///
        /// `1` means the first candidate matched. Equal to the total candidate
        /// count means the universal fallback was used.
        public let candidatesEvaluated: Int

        /// Total number of registered model types.
        public let totalCandidates: Int
    }

    // MARK: - Weight Loading

    /// Facts about tensor conversion from GGUF to MLX weight paths.
    public struct WeightLoading: Sendable {

        /// Number of GGUF tensors successfully mapped to MLX weight paths.
        public let mappedCount: Int

        /// Number of GGUF tensors skipped (no mapper match).
        public let skippedCount: Int

        /// GGUF tensor names that were skipped.
        ///
        /// Useful for diagnosing incomplete model support or mapper gaps.
        public let skippedTensors: [String]

        /// Embedded LoRA/DoRA adapter details, if detected in the GGUF file.
        public let embeddedAdapter: EmbeddedAdapter?
    }

    /// Embedded LoRA/DoRA adapter detected within the GGUF file's tensors.
    public struct EmbeddedAdapter: Sendable {

        /// Fine-tuning method used.
        public enum AdapterType: String, Sendable {
            case lora
            case dora
        }

        /// Fine-tuning method (LoRA or DoRA).
        public let type: AdapterType

        /// Rank of the low-rank matrices.
        public let rank: Int

        /// Number of transformer layers containing adapter weights.
        public let layerCount: Int

        /// Total number of adapter tensors (lora_a + lora_b + optional magnitude).
        public let tensorCount: Int
    }

    // MARK: - Quantization

    /// Records how quantization was resolved and applied.
    public struct QuantizationApplied: Sendable {

        /// How the quantization configuration was determined.
        public enum Source: String, Sendable {
            /// Caller passed an explicit `QuantizationConfiguration`.
            case userSpecified
            /// Auto-detected from predominant GGUF tensor quantization type.
            case autoDetected
            /// Quantization was skipped (F16 weights retained).
            case disabled
        }

        /// How this configuration was determined.
        public let source: Source

        /// Bits per weight element. `0` when disabled.
        public let bits: Int

        /// Elements per quantization group. `0` when disabled.
        public let groupSize: Int
    }

    // MARK: - Summary

    /// Human-readable summary of the loading pipeline.
    public var summary: String {
        var lines: [String] = []

        // Model resolution
        let fallback = modelResolution.candidatesEvaluated == modelResolution.totalCandidates
        lines.append(
            "Model: \(modelResolution.selectedType)"
            + (fallback ? " (fallback)" : "")
        )

        // Weight loading
        lines.append(
            "Tensors: \(weightLoading.mappedCount) mapped, "
            + "\(weightLoading.skippedCount) skipped"
        )

        // LoRA
        if let adapter = weightLoading.embeddedAdapter {
            lines.append(
                "Adapter: \(adapter.type.rawValue.uppercased()) "
                + "rank=\(adapter.rank) layers=\(adapter.layerCount) "
                + "tensors=\(adapter.tensorCount)"
            )
        }

        // Quantization
        switch quantization.source {
        case .disabled:
            lines.append("Quantization: disabled (F16)")
        case .autoDetected:
            lines.append(
                "Quantization: \(quantization.bits)-bit "
                + "group=\(quantization.groupSize) (auto-detected)"
            )
        case .userSpecified:
            lines.append(
                "Quantization: \(quantization.bits)-bit "
                + "group=\(quantization.groupSize) (user-specified)"
            )
        }

        return lines.joined(separator: "\n")
    }
}
