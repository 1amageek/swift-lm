import Foundation

enum HFCacheResolver {
    static func resolveSnapshot(repoDirectoryName: String) throws -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let snapshots = home
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
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
