/// Structured representation of a parsed GGUF tensor name.
///
/// GGUF tensors follow the naming convention `blk.{layerIndex}.{suffix}`
/// for per-layer tensors and plain names for global tensors.
/// This type parses the name once and provides typed access to the components.
///
/// ```swift
/// let ref = GGUFTensorRef("blk.0.attn_q.weight")
/// ref.layerIndex  // 0
/// ref.suffix       // "attn_q.weight"
/// ref.isBlock      // true
///
/// let global = GGUFTensorRef("token_embd.weight")
/// global.layerIndex  // nil
/// global.suffix       // "token_embd.weight"
/// global.isBlock      // false
/// ```
package struct GGUFTensorRef: Sendable, Equatable {

    /// Layer index for per-block tensors, `nil` for global tensors.
    package let layerIndex: Int?

    /// Tensor suffix after `blk.{i}.`, or the full name for global tensors.
    package let suffix: String

    /// Original tensor name.
    package let name: String

    /// Whether this tensor belongs to a transformer block.
    package var isBlock: Bool { layerIndex != nil }

    package init(_ name: String) {
        self.name = name

        if name.hasPrefix("blk.") {
            let parts = name.split(separator: ".", maxSplits: 2)
            if parts.count == 3,
               parts[0] == "blk",
               let index = Int(parts[1]) {
                self.layerIndex = index
                self.suffix = String(parts[2])
                return
            }
        }

        self.layerIndex = nil
        self.suffix = name
    }
}
