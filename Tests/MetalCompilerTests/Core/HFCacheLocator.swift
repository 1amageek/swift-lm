import Foundation

/// Resolve HuggingFace Hub cache snapshot directories.
///
/// Model bundles live under `~/.cache/huggingface/hub/` in the Python-compatible
/// layout used by `huggingface-cli download` and `huggingface_hub.snapshot_download`:
///
/// ```
/// ~/.cache/huggingface/hub/
///   models--<org>--<name>/
///     snapshots/<commit-sha>/
///       config.json   (symlink → ../../blobs/<etag>)
///       *.safetensors (symlink → ../../blobs/<etag>)
///       ...
/// ```
///
/// Tests reference bundles by their HF cache directory name (e.g.
/// `models--LiquidAI--LFM2.5-1.2B-Thinking`). When the bundle is absent the
/// resolver returns `nil` so the test can skip gracefully — never silently
/// fall back to a project-local path.
enum HFCacheLocator {

    /// Root of the HuggingFace Hub cache. Always `~/.cache/huggingface/hub/`
    /// per CLAUDE.md ("HuggingFace モデルは ~/.cache/huggingface/hub/ に統一する").
    static var hubRoot: String {
        NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
    }

    /// Resolve a snapshot directory containing `config.json` for the given
    /// HF cache repository directory name (e.g. `models--Qwen--Qwen3.5-0.8B`).
    ///
    /// Returns the first snapshot whose `config.json` exists. Returns `nil`
    /// if the repo directory is missing or no snapshot has a `config.json`.
    /// Callers must treat `nil` as "not cached" and skip — never substitute
    /// a project-local path.
    static func resolveSnapshotPath(repoDirectoryName: String) -> String? {
        let snapshotsDir = "\(hubRoot)/\(repoDirectoryName)/snapshots"
        guard FileManager.default.fileExists(atPath: snapshotsDir) else {
            return nil
        }
        let entries: [String]
        do {
            entries = try FileManager.default.contentsOfDirectory(atPath: snapshotsDir).sorted()
        } catch {
            return nil
        }
        for entry in entries {
            let candidate = "\(snapshotsDir)/\(entry)"
            if FileManager.default.fileExists(atPath: "\(candidate)/config.json") {
                return candidate
            }
        }
        return nil
    }
}
