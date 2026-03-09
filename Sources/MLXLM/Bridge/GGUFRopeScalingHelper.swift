import GGUFParser

/// Extract RoPE scaling configuration from GGUF metadata.
///
/// Shared helper used by per-model Configuration `init(from:)`.
func extractRopeScaling(from file: GGUFFile) -> [String: StringOrNumber]? {
    guard let ropeType = file[.ropeScalingType] else { return nil }

    var config: [String: StringOrNumber] = [
        "type": .string(ropeType)
    ]

    if let factor = file[.ropeScalingFactor] {
        config["factor"] = .float(factor)
    }
    if let origMax = file[.ropeScalingOriginalMaxPositionEmbeddings] {
        config["original_max_position_embeddings"] = .int(origMax)
    }

    // Llama 3 specific
    if let lowFreq = file[.ropeScalingLowFreqFactor] {
        config["low_freq_factor"] = .float(lowFreq)
    }
    if let highFreq = file[.ropeScalingHighFreqFactor] {
        config["high_freq_factor"] = .float(highFreq)
    }

    // Su/LongRoPE specific
    if let attnFactor = file[.ropeScalingAttnFactor] {
        config["attn_factor"] = .float(attnFactor)
    }
    if let shortFactor = file[.ropeScalingShortFactor] {
        config["short_factor"] = .floats(shortFactor)
    }
    if let longFactor = file[.ropeScalingLongFactor] {
        config["long_factor"] = .floats(longFactor)
    }

    return config
}

/// Detect whether word embeddings are tied (no separate output.weight tensor).
func detectTieWordEmbeddings(from file: GGUFFile) -> Bool {
    !file.tensors.contains { $0.name == "output.weight" }
}
