import Metal
import LMIR

/// Mutable runtime for token-by-token inference using a compiled Metal model.
public struct MetalInferenceModel: @unchecked Sendable {

    public private(set) var compiledModel: MetalCompiledModel
    public let commandQueue: MTLCommandQueue
    public var position: Int = 0

    private let submission: MetalSubmissionContext
    private let decodeExecutor = MetalDecodeExecutor()
    private let prefillExecutor = MetalPrefillExecutor()
    private let promptStateStore = MetalPromptStateStore()
    private var pendingCommandBuffer: MTLCommandBuffer?
    private var hasPendingResult: Bool = false

    /// Backward-compatible decode plan view.
    ///
    /// Prefer ``decodePlan`` or ``compiledModel`` for new call sites.
    public var plan: MetalDispatchPlan { compiledModel.decodePlan }

    /// Decode-time dispatch plan extracted from the compiled model.
    public var decodePlan: MetalDispatchPlan { compiledModel.decodePlan }

    /// Optional sequence-oriented prefill plan paired with the decode plan.
    public var prefillPlan: MetalPrefillPlan? {
        get { compiledModel.prefillPlan }
        set { compiledModel = compiledModel.withPrefillPlan(newValue) }
    }

    /// Shared runtime buffers used by decode execution.
    public var buffers: MetalBufferSet { decodePlan.buffers }

    public init(plan: MetalDispatchPlan, device: MTLDevice) throws {
        self.compiledModel = MetalCompiledModel(decodePlan: plan)
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create command queue")
        }
        self.commandQueue = queue
        self.submission = MetalSubmissionContext(commandQueue: queue)
        try Self.zeroStateBuffers(plan.buffers, submission: submission)
    }

    public init(compiledModel: MetalCompiledModel, device: MTLDevice) throws {
        self.compiledModel = compiledModel
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create command queue")
        }
        self.commandQueue = queue
        self.submission = MetalSubmissionContext(commandQueue: queue)
        try Self.zeroStateBuffers(compiledModel.decodePlan.buffers, submission: submission)
    }

    public init(plan: MetalCompiledModel, device: MTLDevice) throws {
        try self.init(compiledModel: plan, device: device)
    }

    private static func zeroStateBuffers(_ buffers: MetalBufferSet, submission: MetalSubmissionContext) throws {
        _ = try submission.withTransaction(label: "state.zero") { transaction in
            try transaction.withBlitEncoder { blit in
                blit.fill(buffer: buffers.hidden, range: 0..<buffers.hidden.length, value: 0)
                blit.fill(buffer: buffers.residual, range: 0..<buffers.residual.length, value: 0)
                blit.fill(buffer: buffers.scratch, range: 0..<buffers.scratch.length, value: 0)
                blit.fill(buffer: buffers.logits, range: 0..<buffers.logits.length, value: 0)
                if let kv = buffers.kvCache {
                    blit.fill(buffer: kv.keys, range: 0..<kv.keys.length, value: 0)
                    blit.fill(buffer: kv.values, range: 0..<kv.values.length, value: 0)
                }
                if let convState = buffers.convState {
                    blit.fill(buffer: convState, range: 0..<convState.length, value: 0)
                }
            }
        }
    }

    public mutating func decode(tokenID: Int32) -> Int32 {
        decodeExecutor.decode(
            plan: decodePlan,
            submission: submission,
            position: &position,
            pendingCommandBuffer: &pendingCommandBuffer,
            hasPendingResult: &hasPendingResult,
            tokenID: tokenID
        )
    }

    public mutating func decodeSync(tokenID: Int32) -> Int32 {
        decodeExecutor.decodeSync(
            plan: decodePlan,
            submission: submission,
            position: &position,
            tokenID: tokenID
        )
    }

    // MARK: - Prefill

    /// Prefill the KV cache with prompt tokens and return the first predicted token.
    ///
    /// Returns the argmax of the prefill logits (the model's first generated token).
    /// The caller should output this token and feed it to the first decode step.
    @discardableResult
    public mutating func prefill(tokens: [Int32]) -> Int32 {
        guard let prefillPlan else {
            var lastOutput: Int32 = -1
            for token in tokens {
                lastOutput = decodeSync(tokenID: token)
            }
            return lastOutput
        }

        // Use sequence-parallel prefill for short prompts (verified correct for ≤8 tokens).
        // Longer prompts fall back to sequential decode to avoid a known
        // cross-threadgroup KV cache visibility issue in perPosition flash attention.
        // TODO: Replace with a proper batch flash attention kernel.
        if tokens.count > 8 {
            var lastOutput: Int32 = -1
            for token in tokens {
                lastOutput = decodeSync(tokenID: token)
            }
            return lastOutput
        }

        return prefillExecutor.prefill(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            submission: submission,
            position: &position,
            tokens: tokens
        )
    }

    // MARK: - Lifecycle

    public mutating func flush() -> Int32 {
        decodeExecutor.flush(
            plan: decodePlan,
            submission: submission,
            pendingCommandBuffer: &pendingCommandBuffer,
            hasPendingResult: &hasPendingResult
        )
    }

    public func makePromptState(firstToken: Int32) throws -> MetalPromptState {
        try promptStateStore.makePromptState(
            plan: decodePlan,
            submission: submission,
            position: position,
            firstToken: firstToken
        )
    }

    public mutating func restore(promptState: MetalPromptState) throws {
        pendingCommandBuffer = nil
        hasPendingResult = false
        try promptStateStore.restore(plan: decodePlan, submission: submission, promptState: promptState)
        position = promptState.position
    }

    public mutating func resetCaches() {
        if let pendingCommandBuffer {
            do {
                try submission.waitUntilCompleted(pendingCommandBuffer)
            } catch {
                print("[MetalInference] Pending decode failed during reset: \(error)")
            }
        }
        pendingCommandBuffer = nil
        hasPendingResult = false
        position = 0
        do {
            try Self.zeroStateBuffers(decodePlan.buffers, submission: submission)
        } catch {
            print("[MetalInference] Failed to reset GPU state: \(error)")
        }
    }
}
