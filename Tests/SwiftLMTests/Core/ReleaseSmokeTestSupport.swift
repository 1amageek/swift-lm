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

    /// Resolves the LFM2.5-8B-A1B bundle from the HuggingFace cache.
    ///
    /// This model is substantially larger than the default release-smoke
    /// LFM2.5-1.2B bundle, so tests must never download it implicitly.
    static func readableLFM25A1BModelDirectoryOrSkip() -> URL? {
        guard let snapshot = HFCacheLocator.resolveSnapshotPath(
            repoDirectoryName: "models--LiquidAI--LFM2.5-8B-A1B"
        ) else {
            print("[Skip] LFM2.5-8B-A1B not cached. Run `huggingface-cli download LiquidAI/LFM2.5-8B-A1B`.")
            return nil
        }
        return URL(fileURLWithPath: snapshot)
    }

    /// Resolves the LFM2.5-8B-A1B MLX 8-bit bundle from the HuggingFace cache.
    ///
    /// This route exercises packed Q8 MoE weights. Tests must skip when the
    /// optional quantized bundle is absent rather than downloading it implicitly.
    static func readableLFM25A1BMLX8BitModelDirectoryOrSkip() -> URL? {
        guard let snapshot = HFCacheLocator.resolveSnapshotPath(
            repoDirectoryName: "models--LiquidAI--LFM2.5-8B-A1B-MLX-8bit"
        ) else {
            print("[Skip] LFM2.5-8B-A1B-MLX-8bit not cached. Run `hf download LiquidAI/LFM2.5-8B-A1B-MLX-8bit --cache-dir ~/.cache/huggingface/hub`.")
            return nil
        }
        let snapshotURL = URL(fileURLWithPath: snapshot)
        guard snapshotHasReadableSafetensors(snapshotURL) else {
            print("[Skip] LFM2.5-8B-A1B-MLX-8bit snapshot is incomplete. Finish `hf download LiquidAI/LFM2.5-8B-A1B-MLX-8bit --cache-dir ~/.cache/huggingface/hub`.")
            return nil
        }
        return snapshotURL
    }

    private static func snapshotHasReadableSafetensors(_ snapshotURL: URL) -> Bool {
        let indexURL = snapshotURL.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            return snapshotHasReadableIndexedSafetensors(snapshotURL: snapshotURL, indexURL: indexURL)
        }

        let files: [URL]
        do {
            files = try FileManager.default.contentsOfDirectory(
                at: snapshotURL,
                includingPropertiesForKeys: nil
            )
        } catch {
            return false
        }
        let safetensors = files.filter { $0.pathExtension == "safetensors" }
        guard !safetensors.isEmpty else {
            return false
        }
        return safetensors.allSatisfy { FileManager.default.fileExists(atPath: $0.path) }
    }

    private static func snapshotHasReadableIndexedSafetensors(snapshotURL: URL, indexURL: URL) -> Bool {
        do {
            let data = try Data(contentsOf: indexURL)
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weightMap = root["weight_map"] as? [String: String] else {
                return false
            }
            let shardNames = Set(weightMap.values)
            guard !shardNames.isEmpty else {
                return false
            }
            return shardNames.allSatisfy { shardName in
                FileManager.default.fileExists(
                    atPath: snapshotURL.appendingPathComponent(shardName).path
                )
            }
        } catch {
            return false
        }
    }
}
