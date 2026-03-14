import Foundation
import MLX
import MLXNN
import MLXFast

/// LanguageModel implementation backed by MPSGraph.
///
/// All layers execute in a single MPSGraph dispatch with kernel fusion.
/// KV cache is not yet supported — each forward pass processes the full
/// input sequence. This means generation currently re-processes all
/// previous tokens on each step.
///
/// To be replaced with a stateful KV cache implementation in a future phase.
public final class MPSGraphLanguageModel: Module, LanguageModel, @unchecked Sendable {

    private let engine: MPSGraphInferenceEngine
    private let _layerCount: Int
    private let _kvHeads: [Int]

    public init(engine: MPSGraphInferenceEngine) {
        self.engine = engine
        self._layerCount = engine.config.layerCount
        self._kvHeads = Array(repeating: engine.config.kvHeadCount, count: engine.config.layerCount)
        super.init()
    }

    // MARK: - LanguageModel Protocol

    public var layerCount: Int { _layerCount }
    public var kvHeads: [Int] { _kvHeads }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // MPSGraph currently processes full sequence each call.
        // Return a simple cache that just tracks token history.
        [MPSGraphKVCache()]
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        guard let mpsCache = cache?.first as? MPSGraphKVCache else {
            // No cache — single forward pass
            let tokenIDs: [Int32] = input.tokens.flattened().asArray(Int32.self)
            let logits = engine.forward(tokenIDs: tokenIDs)
            return LMOutput(logits: logits)
        }

        // Append new tokens to history
        let newTokens: [Int32] = input.tokens.flattened().asArray(Int32.self)
        mpsCache.appendTokens(newTokens)

        // Forward pass with full token history
        let logits = engine.forward(tokenIDs: mpsCache.tokenHistory)
        return LMOutput(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        let output = callAsFunction(input.text, cache: cache, state: nil)
        return .logits(output)
    }
}

// MARK: - MPSGraphKVCache

/// Simple KV cache for MPSGraph backend.
///
/// Since MPSGraph doesn't have built-in stateful KV cache,
/// this tracks token history and replays the full sequence each step.
/// The graph compiler optimizes the full-sequence forward pass.
final class MPSGraphKVCache: KVCache, @unchecked Sendable {

    private(set) var tokenHistory: [Int32] = []

    var offset: Int { tokenHistory.count }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(n, tokenHistory.count)
        tokenHistory.removeLast(trimmed)
        return trimmed
    }

    var state: [MLXArray] {
        get { [] }
        set { }
    }

    var metaState: [String] {
        get { [String(tokenHistory.count)] }
        set {
            if let first = newValue.first, let val = Int(first) {
                tokenHistory = Array(tokenHistory.prefix(val))
            }
        }
    }

    func innerState() -> [MLXArray] { [] }

    func makeMask(queryLength: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        queryLength > 1 ? .causal : .none
    }

    func appendTokens(_ tokens: [Int32]) {
        tokenHistory.append(contentsOf: tokens)
    }

    func reset() {
        tokenHistory.removeAll()
    }
}
