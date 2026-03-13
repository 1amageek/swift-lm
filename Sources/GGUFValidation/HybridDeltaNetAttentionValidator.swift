import GGUFParser
import GGUFToolingCore
import MLXLM

public struct HybridDeltaNetAttentionValidator: GGUFFamilyValidator {
    public let family: DetectedArchitecture = .hybridDeltaNetAttention

    public init() {}

    public func validate(context: ValidationContext) -> [GGUFValidationIssue] {
        var issues: [GGUFValidationIssue] = []
        let file = context.file

        if file.architectureMetadata("attention.key_length") == nil {
            let detail = attentionKeyLengthEvidence(file: file, architecture: context.architecture)
            issues.append(
                GGUFValidationIssue(
                    severity: .error,
                    kind: .missingMetadata,
                    metadataKey: fullyQualifiedKey("attention.key_length", architecture: context.architecture),
                    message: detail.message,
                    evidence: detail.evidence
                )
            )
        }

        // ssm.time_step_rank is the canonical DeltaNet head count.
        // For asymmetric models (group_count != time_step_rank), this must be present.
        if file.architectureMetadata("ssm.time_step_rank") == nil {
            let detail = timeStepRankEvidence(file: file, architecture: context.architecture)
            issues.append(
                GGUFValidationIssue(
                    severity: .error,
                    kind: .missingMetadata,
                    metadataKey: fullyQualifiedKey("ssm.time_step_rank", architecture: context.architecture),
                    message: detail.message,
                    evidence: detail.evidence,
                    suggestedValue: detail.suggestedValue
                )
            )
        }

        for key in ["ssm.group_count", "ssm.state_size", "ssm.conv_kernel", "full_attention_interval"] {
            if file.architectureMetadata(key) == nil {
                issues.append(
                    GGUFValidationIssue(
                        severity: .error,
                        kind: .missingMetadata,
                        metadataKey: fullyQualifiedKey(key, architecture: context.architecture),
                        message: "Missing required GGUF metadata: \(key)",
                        evidence: ["expected key: \(fullyQualifiedKey(key, architecture: context.architecture))"]
                    )
                )
            }
        }

        if file.partialRotaryFactor == nil {
            let detail = partialRotaryFactorEvidence(file: file, architecture: context.architecture)
            issues.append(
                GGUFValidationIssue(
                    severity: .error,
                    kind: .missingMetadata,
                    metadataKey: fullyQualifiedKey("rope.partial_rotary_factor", architecture: context.architecture),
                    message: detail.message,
                    evidence: detail.evidence,
                    suggestedValue: detail.suggestedValue
                )
            )
        }

        return issues
    }

    public func repairActions(
        context: ValidationContext,
        mode: RepairPlanningMode
    ) -> [GGUFRepairAction] {
        guard mode == .includeInferredRepairs else {
            return []
        }
        let file = context.file
        var actions: [GGUFRepairAction] = []

        // Repair ssm.time_step_rank from ssm.inner_size / ssm.state_size
        if file.architectureMetadata("ssm.time_step_rank") == nil {
            let detail = timeStepRankEvidence(file: file, architecture: context.architecture)
            if let suggestedValue = detail.suggestedValue {
                actions.append(
                    .addMetadata(
                        key: fullyQualifiedKey("ssm.time_step_rank", architecture: context.architecture),
                        value: suggestedValue,
                        rationale: detail.rationale
                    )
                )
            }
        }

        // Repair rope.partial_rotary_factor from rope.dimension_count / attention.key_length
        if file.partialRotaryFactor == nil {
            let detail = partialRotaryFactorEvidence(file: file, architecture: context.architecture)
            if let suggestedValue = detail.suggestedValue {
                actions.append(
                    .addMetadata(
                        key: fullyQualifiedKey("rope.partial_rotary_factor", architecture: context.architecture),
                        value: suggestedValue,
                        rationale: detail.rationale
                    )
                )
            }
        }

        return actions
    }

    // MARK: - Evidence helpers

    private func timeStepRankEvidence(
        file: GGUFFile,
        architecture: String?
    ) -> (message: String, evidence: [String], suggestedValue: GGUFMetadataValue?, rationale: String) {
        let key = fullyQualifiedKey("ssm.time_step_rank", architecture: architecture)
        var evidence = ["expected key: \(key)"]

        guard let innerSize = file.ssmInnerSize else {
            return (
                "Missing required GGUF metadata: ssm.time_step_rank (number of DeltaNet value heads)",
                evidence,
                nil,
                "No deterministic inference rule was available."
            )
        }
        evidence.append("\(fullyQualifiedKey("ssm.inner_size", architecture: architecture))=\(innerSize)")

        guard let stateSize = file.ssmStateSize, stateSize > 0 else {
            return (
                "Missing required GGUF metadata: ssm.time_step_rank (number of DeltaNet value heads)",
                evidence,
                nil,
                "No deterministic inference rule was available."
            )
        }
        evidence.append("\(fullyQualifiedKey("ssm.state_size", architecture: architecture))=\(stateSize)")

        guard innerSize % stateSize == 0 else {
            return (
                "Missing required GGUF metadata: ssm.time_step_rank (ssm.inner_size is not divisible by ssm.state_size)",
                evidence,
                nil,
                "No deterministic inference rule was available."
            )
        }

        let inferred = innerSize / stateSize
        let inferredValue = GGUFMetadataValue.uint32(UInt32(inferred))
        return (
            "Missing required GGUF metadata: ssm.time_step_rank (inferred value would be \(inferred), but strict loading requires explicit metadata)",
            evidence,
            inferredValue,
            "Inferred from \(fullyQualifiedKey("ssm.inner_size", architecture: architecture)) / \(fullyQualifiedKey("ssm.state_size", architecture: architecture)) = \(innerSize) / \(stateSize)."
        )
    }

    private func partialRotaryFactorEvidence(
        file: GGUFFile,
        architecture: String?
    ) -> (message: String, evidence: [String], suggestedValue: GGUFMetadataValue?, rationale: String) {
        let key = fullyQualifiedKey("rope.partial_rotary_factor", architecture: architecture)
        var evidence = ["expected key: \(key)"]

        guard let ropeDimension = file.ropeDimensionCount else {
            return (
                "Missing required GGUF metadata: rope.partial_rotary_factor",
                evidence,
                nil,
                "No deterministic inference rule was available."
            )
        }
        evidence.append("\(fullyQualifiedKey("rope.dimension_count", architecture: architecture))=\(ropeDimension)")

        guard let attentionKeyLength = file.attentionKeyLength, attentionKeyLength > 0 else {
            return (
                "Missing required GGUF metadata: rope.partial_rotary_factor",
                evidence,
                nil,
                "No deterministic inference rule was available."
            )
        }
        evidence.append("\(fullyQualifiedKey("attention.key_length", architecture: architecture))=\(attentionKeyLength)")

        let inferred = Float(ropeDimension) / Float(attentionKeyLength)
        let inferredValue = GGUFMetadataValue.float32(inferred)
        let inferredString = inferredValue.displayString
        return (
            "Missing required GGUF metadata: rope.partial_rotary_factor (inferred factor would be \(inferredString), but strict loading requires explicit metadata)",
            evidence,
            inferredValue,
            "Inferred from \(fullyQualifiedKey("rope.dimension_count", architecture: architecture)) / \(fullyQualifiedKey("attention.key_length", architecture: architecture)) = \(ropeDimension) / \(attentionKeyLength)."
        )
    }

    private func attentionKeyLengthEvidence(
        file: GGUFFile,
        architecture: String?
    ) -> (message: String, evidence: [String]) {
        let key = fullyQualifiedKey("attention.key_length", architecture: architecture)
        var evidence = ["expected key: \(key)"]

        if let embeddingLength = file.embeddingLength {
            evidence.append("\(fullyQualifiedKey("embedding_length", architecture: architecture))=\(embeddingLength)")
        }
        if let headCount = file.headCount {
            evidence.append("\(fullyQualifiedKey("attention.head_count", architecture: architecture))=\(headCount)")
        }
        if let embeddingLength = file.embeddingLength, let headCount = file.headCount, headCount > 0 {
            let inferred = embeddingLength / headCount
            return (
                "Missing required GGUF metadata: attention.key_length (inferred head dimension would be \(inferred), but strict loading requires explicit metadata)",
                evidence
            )
        }

        return ("Missing required GGUF metadata: attention.key_length", evidence)
    }

    private func fullyQualifiedKey(_ key: String, architecture: String?) -> String {
        guard let architecture, !architecture.isEmpty else {
            return key
        }
        return "\(architecture).\(key)"
    }
}
