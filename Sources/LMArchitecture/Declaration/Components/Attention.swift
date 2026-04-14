/// Multi-head attention component.
///
/// Composes the full attention operation: Q/K/V projections, optional QK normalization,
/// optional rotary position encoding, scaled dot-product attention, and output projection.
///
/// Properties directly determine the fragment tree structure:
/// - `headCount`, `kvHeadCount`, `headDimension` → projection sizing and GQA ratio
/// - `rope` → rotary position encoding (RoPE) when present
/// - `qkNorm` → QK normalization before attention scores when present
/// - `causal`, `window` → attention masking behavior
///
/// ```swift
/// Attention(
///     hiddenSize: 4096,
///     headCount: 32,
///     kvHeadCount: 8,
///     rope: RoPEAttributes(dimension: 128, base: 500_000)
/// )
/// ```
public struct Attention: ModelComponent {

    public typealias Attributes = AttentionAttributes

    // MARK: - Projection geometry

    /// Model hidden dimension. Determines Q and O projection input/output sizes.
    public let hiddenSize: Int

    /// Number of query attention heads. Each head computes an independent attention pattern.
    public let headCount: Int

    /// Number of key/value heads. When less than `headCount`, enables grouped-query attention (GQA).
    public let kvHeadCount: Int

    /// Dimension of each attention head. Determines the dot-product space.
    public let headDimension: Int

    /// Whether Q/K/V/O projections include bias terms.
    public let bias: Bool

    // MARK: - Attention computation

    /// Override for attention score scaling. Default: `1 / sqrt(headDimension)`.
    public let attentionScale: Float?

    /// Whether attention is causal (autoregressive). Masks future positions.
    public let causal: Bool

    /// Sliding window configuration. Limits the attention range per position.
    public let window: AttentionWindow?

    // MARK: - Position encoding

    /// Rotary position embedding configuration. When present, RoPE is applied to Q and K.
    public let rope: RoPEAttributes?

    // MARK: - Normalization

    /// QK normalization strategy applied before attention score computation.
    public let qkNorm: QKNormKind?

    public init(
        hiddenSize: Int,
        headCount: Int,
        kvHeadCount: Int,
        headDimension: Int? = nil,
        attentionScale: Float? = nil,
        bias: Bool = false,
        causal: Bool = true,
        rope: RoPEAttributes? = nil,
        qkNorm: QKNormKind? = nil,
        window: AttentionWindow? = nil
    ) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(headCount > 0, "headCount must be positive")
        precondition(kvHeadCount > 0, "kvHeadCount must be positive")
        precondition(kvHeadCount <= headCount, "kvHeadCount must not exceed headCount")
        if let headDimension {
            precondition(headDimension > 0, "headDimension must be positive")
        } else {
            precondition(hiddenSize % headCount == 0,
                "hiddenSize must be divisible by headCount when headDimension is not specified")
        }
        self.hiddenSize = hiddenSize
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension ?? (hiddenSize / headCount)
        self.attentionScale = attentionScale
        self.bias = bias
        self.causal = causal
        self.rope = rope
        self.qkNorm = qkNorm
        self.window = window
    }

    public var attributes: AttentionAttributes {
        AttentionAttributes(
            hiddenSize: hiddenSize,
            headCount: headCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            attentionScale: attentionScale,
            bias: bias,
            causal: causal,
            rope: rope,
            qkNorm: qkNorm
        )
    }
}
