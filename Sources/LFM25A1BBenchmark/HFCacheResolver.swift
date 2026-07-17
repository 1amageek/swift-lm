import Foundation

enum HFCacheResolver {
    static func resolveSnapshot(repoDirectoryName: String) throws -> URL {
#if os(macOS) && !targetEnvironment(macCatalyst)
        let home = FileManager.default.homeDirectoryForCurrentUser
        let cacheRoot = home
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
#else
        guard let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            throw BenchmarkError.modelNotFound("No writable user cache directory available")
        }
        let cacheRoot = caches
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
#endif
        let snapshots = cacheRoot
            .appendingPathComponent(repoDirectoryName, isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
        let contents = try FileManager.default.contentsOfDirectory(
            at: snapshots,
            includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey]
        )
        let candidates = contents.filter { url in
            let resourceValues: URLResourceValues
            do {
                resourceValues = try url.resourceValues(forKeys: [.isDirectoryKey])
            } catch {
                return false
            }
            guard resourceValues.isDirectory == true else { return false }
            return FileManager.default.fileExists(
                atPath: url.appendingPathComponent("config.json").path
            )
        }
        guard let newest = candidates.max(by: { lhs, rhs in
            modificationDate(lhs) < modificationDate(rhs)
        }) else {
            throw BenchmarkError.modelNotFound(snapshots.path)
        }
        return newest
    }

    private static func modificationDate(_ url: URL) -> Date {
        do {
            return try url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? .distantPast
        } catch {
            return .distantPast
        }
    }
}
