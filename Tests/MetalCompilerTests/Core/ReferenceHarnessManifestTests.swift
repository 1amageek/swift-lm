import Foundation
import Testing

@Suite("Reference Harness Manifest")
struct ReferenceHarnessManifestTests {
    @Test("M2 reference harness manifest keeps ready and missing oracles explicit")
    func manifestKeepsReadyAndMissingOraclesExplicit() throws {
        let root = try repositoryRoot()
        let manifest = ReferenceHarnessManifest.entries(repositoryRoot: root)

        #expect(manifest.contains { $0.modelFamily == "LFM2" && $0.status == .ready })
        #expect(manifest.contains { $0.modelFamily == "Qwen3.5" && $0.status == .ready })
        #expect(manifest.contains {
            $0.modelFamily == "Qwen3.5"
                && $0.command.contains("--linear-block-ordinals")
        })

        for entry in manifest where entry.status == .ready {
            for path in entry.requiredPaths {
                #expect(
                    FileManager.default.fileExists(atPath: path.path),
                    "ready reference harness path is missing: \(path.path)"
                )
            }
        }

        for entry in manifest where entry.status == .missing {
            #expect(!entry.blocker.isEmpty, "missing reference harness requires an explicit blocker")
        }
    }

    private func repositoryRoot() throws -> URL {
        var directory = URL(fileURLWithPath: #filePath)
        for _ in 0..<12 {
            directory.deleteLastPathComponent()
            let packagePath = directory.appendingPathComponent("Package.swift").path
            if FileManager.default.fileExists(atPath: packagePath) {
                return directory
            }
        }
        throw ReferenceHarnessManifestError.repositoryRootNotFound
    }
}

private enum ReferenceHarnessManifestError: Error {
    case repositoryRootNotFound
}

private enum ReferenceHarnessStatus: String, Sendable {
    case ready
    case missing
}

private struct ReferenceHarnessEntry: Sendable {
    let modelFamily: String
    let status: ReferenceHarnessStatus
    let requiredPaths: [URL]
    let command: String
    let blocker: String
}

private enum ReferenceHarnessManifest {
    static func entries(repositoryRoot: URL) -> [ReferenceHarnessEntry] {
        [
            ReferenceHarnessEntry(
                modelFamily: "LFM2",
                status: .ready,
                requiredPaths: [
                    repositoryRoot.appendingPathComponent("scripts/hf/dump_lfm2_reference.py"),
                    repositoryRoot.appendingPathComponent("Tests/MetalCompilerTests/Models/LFM2/ReferenceComparisonTests.swift"),
                ],
                command: "python3 scripts/hf/dump_lfm2_reference.py",
                blocker: ""
            ),
            ReferenceHarnessEntry(
                modelFamily: "Qwen3.5",
                status: .ready,
                requiredPaths: [
                    repositoryRoot.appendingPathComponent("scripts/hf/dump_qwen35_reference.py"),
                    repositoryRoot.appendingPathComponent("Tests/MetalCompilerTests/Models/Qwen35/Qwen35ReferenceComparisonTests.swift"),
                ],
                command: "python3 scripts/hf/dump_qwen35_reference.py --output TestData/qwen35_reference.safetensors --decode-steps 2 --linear-block-ordinals 0",
                blocker: ""
            ),
        ]
    }
}
