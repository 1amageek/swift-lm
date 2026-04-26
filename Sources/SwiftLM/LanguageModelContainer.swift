import Foundation

/// Immutable, shareable container for a compiled language model bundle.
///
/// A container owns the loaded model assets, tokenizer, templates, and compile
/// products. This is the primary public entry point for most application code.
/// Initialize ``LanguageModelContext`` with it only when you need isolated
/// mutable inference state, explicit prompt staging, or prompt snapshot reuse.
///
/// All prewarm cost — STAF mmap, tokenizer load, IR resolve, kernel compile,
/// MTLResidencySet promotion, and the initial GPU sync that flushes weight
/// buffers into wired memory — is paid synchronously inside ``ModelBundleLoader/load(repo:revision:configuration:inferencePolicy:)``.
/// When that call returns, the container is ready and the next ``generate(_:parameters:)``
/// reflects pure inference cost. There is no separate `prewarm` step to invoke.
///
/// The convenience ``generate(_:parameters:)`` and ``generate(from:parameters:)``
/// reuse the prewarmed prototype context whenever it is idle, paying only the
/// cost of clearing decode state (KV cache, scratch). When a prior generation
/// is still streaming, the next call falls back to cloning a fresh context so
/// concurrent generations stay isolated. Callers that want guaranteed
/// parallelism should hold their own ``LanguageModelContext`` instances.
public final class LanguageModelContainer: @unchecked Sendable {
    let prototypeContext: LanguageModelContext

    private let prototypeLock = NSLock()
    private var prototypeInUse = false

    init(prototypeContext: LanguageModelContext) {
        self.prototypeContext = prototypeContext
    }

    /// Model configuration (name, EOS tokens, capabilities).
    public var configuration: ModelConfiguration {
        prototypeContext.configuration
    }

    /// Prepare user-facing input into rendered text, tokens, and prompt metadata.
    ///
    /// Most callers can skip this and use ``generate(_:parameters:)`` directly.
    public func prepare(_ input: ModelInput) async throws -> PreparedPrompt {
        try await prototypeContext.prepare(input)
    }

    /// Decode token IDs to text.
    public func decode(_ tokenIDs: [Int], skipSpecialTokens: Bool = true) -> String {
        prototypeContext.decode(tokenIDs, skipSpecialTokens: skipSpecialTokens)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        prototypeContext.encode(text, addSpecialTokens: addSpecialTokens)
    }

    /// Convenience one-shot generation from an executable prompt.
    ///
    /// Reuses the prewarmed prototype context when it is idle, otherwise
    /// clones a fresh isolated context so concurrent generations do not
    /// share decode-time mutable state.
    public func generate(
        from prompt: ExecutablePrompt,
        parameters: GenerationParameters = GenerationParameters()
    ) throws -> AsyncStream<GenerationEvent> {
        let lease = try acquireWorkingContext()
        let stream = try lease.context.generate(from: prompt, parameters: parameters)
        return wrapStream(stream, lease: lease)
    }

    /// Convenience one-shot generation from user input.
    ///
    /// Reuses the prewarmed prototype context when it is idle, otherwise
    /// clones a fresh isolated context so concurrent generations do not
    /// share decode-time mutable state. This is the recommended high-level
    /// entry point for most applications.
    public func generate(
        _ input: ModelInput,
        parameters: GenerationParameters = GenerationParameters()
    ) async throws -> AsyncStream<GenerationEvent> {
        let lease = try acquireWorkingContext()
        let stream = try await lease.context.generate(input, parameters: parameters)
        return wrapStream(stream, lease: lease)
    }

    private struct WorkingContextLease {
        let context: LanguageModelContext
        let release: () -> Void
    }

    private func acquireWorkingContext() throws -> WorkingContextLease {
        prototypeLock.lock()
        if !prototypeInUse {
            prototypeInUse = true
            prototypeLock.unlock()
            prototypeContext.resetState()
            return WorkingContextLease(
                context: prototypeContext,
                release: { [weak self] in self?.releasePrototype() }
            )
        }
        prototypeLock.unlock()
        let context = try LanguageModelContext(self)
        return WorkingContextLease(context: context, release: {})
    }

    private func releasePrototype() {
        prototypeLock.lock()
        prototypeInUse = false
        prototypeLock.unlock()
    }

    private func wrapStream(
        _ source: AsyncStream<GenerationEvent>,
        lease: WorkingContextLease
    ) -> AsyncStream<GenerationEvent> {
        AsyncStream { continuation in
            let releaseBox = OnceBox(release: lease.release)
            let pump = Task {
                for await event in source {
                    if Task.isCancelled { break }
                    continuation.yield(event)
                }
                releaseBox.fire()
                continuation.finish()
            }
            continuation.onTermination = { _ in
                pump.cancel()
                releaseBox.fire()
            }
        }
    }
}

private final class OnceBox: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false
    private let release: () -> Void

    init(release: @escaping () -> Void) {
        self.release = release
    }

    func fire() {
        lock.lock()
        let shouldFire = !fired
        fired = true
        lock.unlock()
        if shouldFire {
            release()
        }
    }
}
