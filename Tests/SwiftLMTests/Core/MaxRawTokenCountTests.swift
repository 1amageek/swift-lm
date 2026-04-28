import Testing
@testable import SwiftLM

/// Pins the contract of `LanguageModelContext.maxRawTokenCount`.
///
/// This function bounds raw generation length when reasoning is split off
/// from the visible answer. Historically it returned a generous multiplier
/// (`max(visibleLimit * 16, visibleLimit + 256)`), which let degenerate
/// reasoning loops run far past the caller's intended `maxTokens`. Adding
/// `maxReasoningTokens` lets the caller bound that escape hatch explicitly.
@Suite("MaxRawTokenCount")
struct MaxRawTokenCountTests {
    private let policy = ThinkingTagPolicy(
        openTag: "<think>",
        closeTag: "</think>",
        openTagTokenID: nil,
        closeTagTokenID: nil
    )

    // MARK: - Without thinking-tag policy: cap is the visible limit

    @Test("Without a thinking policy, raw cap equals visible limit (multiplier never applies)")
    func withoutPolicyCapEqualsVisible() {
        let cap = LanguageModelContext.maxRawTokenCount(
            forVisibleLimit: 128,
            maxReasoningTokens: nil,
            visibilityPolicy: nil,
            reasoningVisibility: .separate
        )
        #expect(cap == 128)
    }

    @Test("Without a thinking policy, maxReasoningTokens is ignored (cap stays at visible limit)")
    func withoutPolicyReasoningCapIgnored() {
        let cap = LanguageModelContext.maxRawTokenCount(
            forVisibleLimit: 128,
            maxReasoningTokens: 64,
            visibilityPolicy: nil,
            reasoningVisibility: .separate
        )
        #expect(cap == 128)
    }

    // MARK: - Inline reasoning: cap is the visible limit

    @Test("With .inline reasoning the multiplier never applies")
    func inlineReasoningCapEqualsVisible() {
        let cap = LanguageModelContext.maxRawTokenCount(
            forVisibleLimit: 128,
            maxReasoningTokens: nil,
            visibilityPolicy: policy,
            reasoningVisibility: .inline
        )
        #expect(cap == 128)
    }

    // MARK: - Separate reasoning, no explicit cap: historical multiplier

    @Test("Separate reasoning + nil maxReasoningTokens uses historical 16x multiplier")
    func separateReasoningWithoutCapUsesMultiplier() {
        let cap = LanguageModelContext.maxRawTokenCount(
            forVisibleLimit: 128,
            maxReasoningTokens: nil,
            visibilityPolicy: policy,
            reasoningVisibility: .separate
        )
        // max(128 * 16, 128 + 256) = max(2048, 384) = 2048
        #expect(cap == 2048)
    }

    @Test("Multiplier floor of +256 wins for very small visible limits")
    func multiplierFloorAppliesForSmallVisibleLimit() {
        let cap = LanguageModelContext.maxRawTokenCount(
            forVisibleLimit: 8,
            maxReasoningTokens: nil,
            visibilityPolicy: policy,
            reasoningVisibility: .separate
        )
        // max(8 * 16, 8 + 256) = max(128, 264) = 264
        #expect(cap == 264)
    }

    // MARK: - Separate reasoning, explicit cap: deterministic visible + reasoning

    @Test("Separate reasoning + explicit maxReasoningTokens caps at visible + reasoning")
    func separateReasoningHonorsExplicitCap() {
        let cap = LanguageModelContext.maxRawTokenCount(
            forVisibleLimit: 128,
            maxReasoningTokens: 256,
            visibilityPolicy: policy,
            reasoningVisibility: .separate
        )
        #expect(cap == 128 + 256)
    }

    @Test("Explicit maxReasoningTokens of 0 collapses raw cap to visible limit")
    func zeroReasoningCapCollapsesToVisibleLimit() {
        let cap = LanguageModelContext.maxRawTokenCount(
            forVisibleLimit: 128,
            maxReasoningTokens: 0,
            visibilityPolicy: policy,
            reasoningVisibility: .separate
        )
        #expect(cap == 128)
    }

    @Test("Explicit cap is honored even when smaller than the historical multiplier floor")
    func explicitCapWinsOverMultiplierFloor() {
        // Historical multiplier for visibleLimit=8 would be 264 (floor).
        // Explicit cap of 4 should still win, not be raised to 264.
        let cap = LanguageModelContext.maxRawTokenCount(
            forVisibleLimit: 8,
            maxReasoningTokens: 4,
            visibilityPolicy: policy,
            reasoningVisibility: .separate
        )
        #expect(cap == 12)
    }

    @Test("Explicit cap is honored even when larger than the historical multiplier")
    func explicitCapWinsOverMultiplierWhenLarger() {
        // Historical multiplier for visibleLimit=128 would be 2048.
        // Explicit cap of 4096 should win — caller knows best.
        let cap = LanguageModelContext.maxRawTokenCount(
            forVisibleLimit: 128,
            maxReasoningTokens: 4096,
            visibilityPolicy: policy,
            reasoningVisibility: .separate
        )
        #expect(cap == 128 + 4096)
    }

    // MARK: - GenerationParameters.maxReasoningTokens default is nil

    @Test("GenerationParameters.maxReasoningTokens defaults to nil so existing callers preserve historical behavior")
    func generationParametersDefaultIsNil() {
        let params = GenerationParameters()
        #expect(params.maxReasoningTokens == nil)
    }
}
