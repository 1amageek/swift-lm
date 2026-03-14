/// Validates dimensional consistency of a `ModelGraph`.
///
/// `DimensionValidator` performs two levels of validation:
///
/// 1. **Attribute invariants**: each operation's attributes satisfy internal
///    dimensional constraints (e.g., `headCount * headDimension == hiddenSize`).
///
/// 2. **Hidden dimension propagation**: tracks the "current hidden dimension"
///    through the graph and verifies that each operation's expected input
///    dimension matches what the preceding operation produces.
///
/// These checks catch misconfigurations that `GraphValidator` (structural
/// arity) cannot detect — e.g., a residual body whose norm dimension
/// doesn't match the attention hidden size, or a stateSpace block where
/// `numHeads * valueHeadDim != hiddenSize`.
public enum DimensionValidator {

    /// Validate dimensional consistency of a model graph.
    ///
    /// - Throws: `DimensionValidationError` if any dimensional invariant is violated.
    public static func validate(_ graph: ModelGraph) throws {
        _ = try validateRegion(graph.rootRegion, inputDim: nil)
    }
}

// MARK: - Errors

/// Errors from dimensional validation.
public enum DimensionValidationError: Error, Sendable, CustomStringConvertible {

    /// A positive integer was expected but got zero or negative.
    case nonPositiveDimension(field: String, value: Int, operationKey: OperationKey?)

    /// An internal attribute invariant is violated.
    case attributeInvariant(message: String, operationKey: OperationKey?)

    /// An operation's expected input dimension doesn't match the current hidden dimension.
    case dimensionMismatch(
        expected: Int,
        actual: Int,
        field: String,
        operationKey: OperationKey?
    )

    /// A structural operation's body produces a different dimension than required.
    case bodyDimensionMismatch(
        expected: Int,
        actual: Int,
        context: String,
        operationKey: OperationKey?
    )

    public var description: String {
        switch self {
        case .nonPositiveDimension(let field, let value, let key):
            return "Non-positive dimension: \(field) = \(value) (operation \(keyDesc(key)))"
        case .attributeInvariant(let message, let key):
            return "Attribute invariant: \(message) (operation \(keyDesc(key)))"
        case .dimensionMismatch(let expected, let actual, let field, let key):
            return "Dimension mismatch: \(field) expected \(expected) but got \(actual) (operation \(keyDesc(key)))"
        case .bodyDimensionMismatch(let expected, let actual, let context, let key):
            return "Body dimension mismatch: \(context) expected \(expected) but got \(actual) (operation \(keyDesc(key)))"
        }
    }
}

private func keyDesc(_ key: OperationKey?) -> String {
    key.map { "#\($0.rawValue)" } ?? "root"
}

// MARK: - Attribute Invariant Validation

private func validateAttentionInvariants(
    _ attrs: AttentionAttributes,
    key: OperationKey
) throws {
    try requirePositive("attention.hiddenSize", attrs.hiddenSize, key: key)
    try requirePositive("attention.headCount", attrs.headCount, key: key)
    try requirePositive("attention.kvHeadCount", attrs.kvHeadCount, key: key)
    try requirePositive("attention.headDimension", attrs.headDimension, key: key)

    // Q projection output must reconstruct hidden size for residual compatibility
    if attrs.headCount * attrs.headDimension != attrs.hiddenSize {
        throw DimensionValidationError.attributeInvariant(
            message: "headCount(\(attrs.headCount)) * headDimension(\(attrs.headDimension)) = \(attrs.headCount * attrs.headDimension) != hiddenSize(\(attrs.hiddenSize))",
            operationKey: key
        )
    }

    // GQA: kvHeadCount must divide headCount evenly
    if attrs.kvHeadCount > attrs.headCount {
        throw DimensionValidationError.attributeInvariant(
            message: "kvHeadCount(\(attrs.kvHeadCount)) > headCount(\(attrs.headCount))",
            operationKey: key
        )
    }
    if attrs.headCount % attrs.kvHeadCount != 0 {
        throw DimensionValidationError.attributeInvariant(
            message: "headCount(\(attrs.headCount)) not divisible by kvHeadCount(\(attrs.kvHeadCount))",
            operationKey: key
        )
    }

    // RoPE dimension constraints
    if let rope = attrs.rope {
        try validateRoPEInvariants(rope, headDimension: attrs.headDimension, key: key)
    }
}

private func validateStateSpaceInvariants(
    _ attrs: StateSpaceAttributes,
    key: OperationKey
) throws {
    try requirePositive("stateSpace.hiddenSize", attrs.hiddenSize, key: key)
    try requirePositive("stateSpace.numHeads", attrs.numHeads, key: key)
    try requirePositive("stateSpace.groupCount", attrs.groupCount, key: key)
    try requirePositive("stateSpace.keyHeadDim", attrs.keyHeadDim, key: key)
    try requirePositive("stateSpace.valueHeadDim", attrs.valueHeadDim, key: key)

    // DeltaNet: output projection is numHeads * valueHeadDim → hiddenSize (matmul),
    // so the product must equal hiddenSize for residual compatibility.
    // Mamba uses a different architecture where state dimension is independent.
    let isDeltaNet = attrs.variant.contains("deltanet") || attrs.variant.contains("delta_net")
    if isDeltaNet && attrs.numHeads * attrs.valueHeadDim != attrs.hiddenSize {
        throw DimensionValidationError.attributeInvariant(
            message: "numHeads(\(attrs.numHeads)) * valueHeadDim(\(attrs.valueHeadDim)) = \(attrs.numHeads * attrs.valueHeadDim) != hiddenSize(\(attrs.hiddenSize))",
            operationKey: key
        )
    }

    // DeltaNet: groupCount is the key/query head count, expanded to match numHeads.
    // groupCount must divide numHeads evenly for clean expansion.
    if isDeltaNet {
        if attrs.groupCount > attrs.numHeads {
            throw DimensionValidationError.attributeInvariant(
                message: "groupCount(\(attrs.groupCount)) > numHeads(\(attrs.numHeads))",
                operationKey: key
            )
        }
        if attrs.numHeads % attrs.groupCount != 0 {
            throw DimensionValidationError.attributeInvariant(
                message: "numHeads(\(attrs.numHeads)) not divisible by groupCount(\(attrs.groupCount))",
                operationKey: key
            )
        }
    }
}

private func validateShortConvInvariants(
    _ attrs: ShortConvAttributes,
    key: OperationKey
) throws {
    try requirePositive("shortConv.hiddenSize", attrs.hiddenSize, key: key)
    try requirePositive("shortConv.kernelSize", attrs.kernelSize, key: key)
}

private func validateMLPInvariants(
    _ attrs: MLPAttributes,
    key: OperationKey
) throws {
    try requirePositive("mlp.inputSize", attrs.inputSize, key: key)
    try requirePositive("mlp.outputSize", attrs.outputSize, key: key)
    try requirePositive("mlp.intermediateSize", attrs.intermediateSize, key: key)
}

private func validateMoEInvariants(
    _ attrs: MoEAttributes,
    key: OperationKey
) throws {
    try requirePositive("moe.expertCount", attrs.expertCount, key: key)
    try requirePositive("moe.expertsPerToken", attrs.expertsPerToken, key: key)

    if attrs.expertsPerToken > attrs.expertCount {
        throw DimensionValidationError.attributeInvariant(
            message: "expertsPerToken(\(attrs.expertsPerToken)) > expertCount(\(attrs.expertCount))",
            operationKey: key
        )
    }

    try validateMLPInvariants(attrs.expertMLP, key: key)
}

private func validateRoPEInvariants(
    _ attrs: RoPEAttributes,
    headDimension: Int?,
    key: OperationKey
) throws {
    try requirePositive("rope.dimension", attrs.dimension, key: key)

    // RoPE operates on pairs of dimensions
    if attrs.dimension % 2 != 0 {
        throw DimensionValidationError.attributeInvariant(
            message: "rope.dimension(\(attrs.dimension)) must be even (rotation applied to pairs)",
            operationKey: key
        )
    }

    // RoPE dimension must not exceed head dimension (partial RoPE allowed)
    if let headDim = headDimension, attrs.dimension > headDim {
        throw DimensionValidationError.attributeInvariant(
            message: "rope.dimension(\(attrs.dimension)) > headDimension(\(headDim))",
            operationKey: key
        )
    }

    if attrs.base <= 0 {
        throw DimensionValidationError.attributeInvariant(
            message: "rope.base(\(attrs.base)) must be positive",
            operationKey: key
        )
    }
}

private func validateNormInvariants(
    dimension: Int,
    epsilon: Float,
    label: String,
    key: OperationKey
) throws {
    try requirePositive("\(label).dimension", dimension, key: key)
    if epsilon <= 0 {
        throw DimensionValidationError.attributeInvariant(
            message: "\(label).epsilon(\(epsilon)) must be positive",
            operationKey: key
        )
    }
}

private func requirePositive(
    _ field: String,
    _ value: Int,
    key: OperationKey
) throws {
    if value <= 0 {
        throw DimensionValidationError.nonPositiveDimension(
            field: field, value: value, operationKey: key
        )
    }
}

// MARK: - Hidden Dimension Propagation

/// Validate a region and return the output hidden dimension.
///
/// - Parameter inputDim: The hidden dimension flowing into this region
///   (nil for root region, which starts from a source operation).
/// - Returns: The hidden dimension produced by this region.
@discardableResult
private func validateRegion(
    _ region: Region,
    inputDim: Int?
) throws -> Int? {
    var currentDim = inputDim

    for op in region.operations {
        currentDim = try validateOperation(op, currentDim: currentDim)
    }

    return currentDim
}

/// Validate a single operation and return the output hidden dimension.
private func validateOperation(
    _ op: Operation,
    currentDim: Int?
) throws -> Int? {
    switch op.kind {

    // MARK: Source operations (produce initial dimension)

    case .tokenEmbedding(let attrs):
        try requirePositive("tokenEmbedding.vocabSize", attrs.vocabSize, key: op.key)
        try requirePositive("tokenEmbedding.embeddingSize", attrs.embeddingSize, key: op.key)
        return attrs.embeddingSize

    // MARK: Dimension-preserving primitives

    case .attention(let attrs):
        try validateAttentionInvariants(attrs, key: op.key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: attrs.hiddenSize, field: "attention.hiddenSize", key: op.key)
        }
        return attrs.hiddenSize

    case .stateSpace(let attrs):
        try validateStateSpaceInvariants(attrs, key: op.key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: attrs.hiddenSize, field: "stateSpace.hiddenSize", key: op.key)
        }
        return attrs.hiddenSize

    case .shortConv(let attrs):
        try validateShortConvInvariants(attrs, key: op.key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: attrs.hiddenSize, field: "shortConv.hiddenSize", key: op.key)
        }
        return attrs.hiddenSize

    case .rmsNorm(let attrs):
        try validateNormInvariants(dimension: attrs.dimension, epsilon: attrs.epsilon, label: "rmsNorm", key: op.key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: attrs.dimension, field: "rmsNorm.dimension", key: op.key)
        }
        return attrs.dimension

    case .layerNorm(let attrs):
        try validateNormInvariants(dimension: attrs.dimension, epsilon: attrs.epsilon, label: "layerNorm", key: op.key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: attrs.dimension, field: "layerNorm.dimension", key: op.key)
        }
        return attrs.dimension

    // MARK: Dimension-transforming primitives

    case .mlp(let attrs):
        try validateMLPInvariants(attrs, key: op.key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: attrs.inputSize, field: "mlp.inputSize", key: op.key)
        }
        return attrs.outputSize

    case .moe(let attrs):
        try validateMoEInvariants(attrs, key: op.key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: attrs.expertMLP.inputSize, field: "moe.expertMLP.inputSize", key: op.key)
        }
        return attrs.expertMLP.outputSize

    case .linear(let attrs):
        try requirePositive("linear.inputSize", attrs.inputSize, key: op.key)
        try requirePositive("linear.outputSize", attrs.outputSize, key: op.key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: attrs.inputSize, field: "linear.inputSize", key: op.key)
        }
        return attrs.outputSize

    case .outputHead(let attrs):
        try requirePositive("outputHead.inputSize", attrs.inputSize, key: op.key)
        try requirePositive("outputHead.vocabSize", attrs.vocabSize, key: op.key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: attrs.inputSize, field: "outputHead.inputSize", key: op.key)
        }
        return attrs.vocabSize

    // MARK: Pass-through primitives

    case .rope(let attrs):
        try validateRoPEInvariants(attrs, headDimension: nil, key: op.key)
        return currentDim

    case .positionalEmbedding:
        return currentDim

    // MARK: Structural operations

    case .residual(let strategy, let body):
        let bodyOutputDim = try validateRegion(body, inputDim: currentDim)
        // For `.add` strategy, body output must match input dimension
        if case .add = strategy, let inDim = currentDim, let outDim = bodyOutputDim {
            if inDim != outDim {
                throw DimensionValidationError.bodyDimensionMismatch(
                    expected: inDim,
                    actual: outDim,
                    context: "residual(.add) body output",
                    operationKey: op.key
                )
            }
        }
        return currentDim

    case .parallel(let merge, let branches):
        var branchDims: [Int] = []
        for branch in branches {
            if let dim = try validateRegion(branch, inputDim: currentDim) {
                branchDims.append(dim)
            }
        }

        switch merge {
        case .add:
            // All branches must produce the same dimension as input
            if let inDim = currentDim {
                for (i, dim) in branchDims.enumerated() {
                    if dim != inDim {
                        throw DimensionValidationError.bodyDimensionMismatch(
                            expected: inDim,
                            actual: dim,
                            context: "parallel(.add) branch \(i)",
                            operationKey: op.key
                        )
                    }
                }
            }
            return currentDim

        case .concat:
            // Output is concatenation of all branch dimensions
            return branchDims.isEmpty ? currentDim : branchDims.reduce(0, +)

        case .stack:
            // Stack adds a new axis; hidden dimension is preserved
            return currentDim

        case .custom:
            // No dimensional constraint for custom merge
            return currentDim
        }

    case .repeating(_, let body):
        let bodyOutputDim = try validateRegion(body, inputDim: currentDim)
        // Loop-carried: body output must match input (feeds back as next iteration input)
        if let inDim = currentDim, let outDim = bodyOutputDim {
            if inDim != outDim {
                throw DimensionValidationError.bodyDimensionMismatch(
                    expected: inDim,
                    actual: outDim,
                    context: "repeating body output",
                    operationKey: op.key
                )
            }
        }
        return currentDim

    case .layerStack(let layers):
        for (i, layer) in layers.enumerated() {
            let layerOutputDim = try validateRegion(layer, inputDim: currentDim)
            // Each layer must preserve dimension (loop-carried)
            if let inDim = currentDim, let outDim = layerOutputDim {
                if inDim != outDim {
                    throw DimensionValidationError.bodyDimensionMismatch(
                        expected: inDim,
                        actual: outDim,
                        context: "layerStack layer \(i) output",
                        operationKey: op.key
                    )
                }
            }
        }
        return currentDim

    // MARK: Escape hatch

    case .custom:
        // Cannot validate custom operations dimensionally
        return currentDim
    }
}

private func checkDimensionMatch(
    expected: Int,
    actual: Int,
    field: String,
    key: OperationKey
) throws {
    if expected != actual {
        throw DimensionValidationError.dimensionMismatch(
            expected: expected,
            actual: actual,
            field: field,
            operationKey: key
        )
    }
}
