/// Parameters controlling text generation.
public struct GenerationParameters: Sendable {
    /// Maximum visible (answer) tokens to generate. nil uses the runtime default cap.
    public var maxTokens: Int?
    /// Hard cap on tokens generated inside the reasoning channel before the
    /// model must close `</think>` (or equivalent). Combined with `maxTokens`
    /// it bounds total raw generation as `maxTokens + maxReasoningTokens`.
    /// When nil and the model exposes a thinking-tag policy, the runtime
    /// falls back to the historical generous multiplier
    /// (`max(maxTokens * 16, maxTokens + 256)`) — set this explicitly to make
    /// runaway reasoning loops fail fast instead of running to that cap.
    public var maxReasoningTokens: Int?
    /// Maximum number of tokens to coalesce into one streamed text chunk.
    public var streamChunkTokenCount: Int
    /// Sampling temperature. 0 = greedy.
    public var temperature: Float
    /// Top-p (nucleus) sampling threshold.
    public var topP: Float
    /// Limit sampling to the highest-probability K tokens. nil = disabled.
    public var topK: Int?
    /// Minimum probability threshold relative to the best candidate. 0 = disabled.
    public var minP: Float
    /// Repetition penalty factor. nil = disabled.
    public var repetitionPenalty: Float?
    /// Penalize tokens that have already appeared in the recent context.
    public var presencePenalty: Float?
    /// Number of recent tokens to consider for repetition penalty.
    public var repetitionContextSize: Int
    /// Request-level controls for how reasoning content is surfaced.
    public var reasoning: ReasoningOptions

    public init(
        maxTokens: Int? = nil,
        maxReasoningTokens: Int? = nil,
        streamChunkTokenCount: Int = 8,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int? = nil,
        minP: Float = 0,
        repetitionPenalty: Float? = nil,
        presencePenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        reasoning: ReasoningOptions = .hidden
    ) {
        self.maxTokens = maxTokens
        self.maxReasoningTokens = maxReasoningTokens
        self.streamChunkTokenCount = streamChunkTokenCount
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.repetitionContextSize = repetitionContextSize
        self.reasoning = reasoning
    }
}
