import CoreML
import MLX
import MLXFast

/// KVCache wrapper around CoreML's MLState.
///
/// CoreML stateful models manage KV cache internally via state tensors.
/// This wrapper adapts the CoreML state to the `KVCache` protocol used
/// by `TokenIterator` and `ModelContainer`.
public final class CoreMLKVCache: KVCache, @unchecked Sendable {

    /// CoreML state object (contains all layer KV caches).
    public let mlState: MLState

    /// Number of tokens currently cached.
    private var _offset: Int = 0

    /// Number of layers in the model.
    public let layerCount: Int

    public init(mlState: MLState, layerCount: Int) {
        self.mlState = mlState
        self.layerCount = layerCount
    }

    // MARK: - KVCache Protocol

    public var offset: Int { _offset }

    public var maxSize: Int? { nil }

    public var isTrimmable: Bool { true }

    /// CoreML manages cache internally — this is not used for CoreML models.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        // CoreML handles KV cache updates inside the stateful model.
        // Return empty arrays; the actual cache is in MLState.
        return (keys, values)
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = min(n, _offset)
        _offset -= trimmed
        return trimmed
    }

    /// State serialization — not fully supported for CoreML (MLState is opaque).
    public var state: [MLXArray] {
        get { [] }
        set { /* CoreML state is managed internally */ }
    }

    /// Metadata state for prompt cache snapshots.
    public var metaState: [String] {
        get { [String(_offset)] }
        set {
            if let first = newValue.first, let val = Int(first) {
                _offset = val
            }
        }
    }

    /// CoreML state is opaque — no MLXArrays to evaluate.
    public func innerState() -> [MLXArray] { [] }

    /// Attention mask for CoreML — handled internally by the model.
    public func makeMask(queryLength: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        queryLength > 1 ? .causal : .none
    }

    // MARK: - CoreML-specific

    /// Advance the offset after a prediction.
    public func advanceOffset(by tokenCount: Int) {
        _offset += tokenCount
    }

    /// Reset the cache for a new conversation.
    public func reset() {
        _offset = 0
    }
}
