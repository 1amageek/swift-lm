import Foundation

enum ReleaseSmokeTestSupport {
    /// Resolves the LFM2.5-1.2B-Thinking bundle from the HuggingFace cache
    /// (`~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Thinking/...`).
    /// Returns `nil` when the bundle has not been downloaded so callers can
    /// skip — never substitute a project-local path per CLAUDE.md.
    static func readableLocalModelDirectoryOrSkip() -> URL? {
        guard let snapshot = HFCacheLocator.resolveSnapshotPath(
            repoDirectoryName: "models--LiquidAI--LFM2.5-1.2B-Thinking"
        ) else {
            print("[Skip] LFM2.5-1.2B-Thinking not cached. Run `huggingface-cli download LiquidAI/LFM2.5-1.2B-Thinking`.")
            return nil
        }
        return URL(fileURLWithPath: snapshot)
    }
}
