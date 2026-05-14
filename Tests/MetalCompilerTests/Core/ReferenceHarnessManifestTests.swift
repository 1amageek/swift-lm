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
                && $0.expectedSchemaVersion == 6
        })

        for entry in manifest where entry.status == .ready {
            for path in entry.requiredPaths {
                #expect(
                    FileManager.default.fileExists(atPath: path.path),
                    "ready reference harness path is missing: \(path.path)"
                )
            }
            try entry.validateReferenceSnapshotIfNeeded()
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
    let referenceSnapshot: URL?
    let expectedSchemaVersion: Int32?
    let requiredTensorNames: [String]

    func validateReferenceSnapshotIfNeeded() throws {
        guard let referenceSnapshot else { return }
        let header = try SafetensorsHeader.load(from: referenceSnapshot)
        for tensorName in requiredTensorNames {
            #expect(
                header.containsTensor(named: tensorName),
                "reference snapshot for \(modelFamily) is missing tensor: \(tensorName)"
            )
        }
        if let expectedSchemaVersion {
            let actual = try header.readInt32Tensor(named: "ref.meta.schema_version", from: referenceSnapshot)
            #expect(
                actual == expectedSchemaVersion,
                "reference snapshot for \(modelFamily) has schema \(actual), expected \(expectedSchemaVersion)"
            )
        }
    }
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
                blocker: "",
                referenceSnapshot: nil,
                expectedSchemaVersion: nil,
                requiredTensorNames: []
            ),
            ReferenceHarnessEntry(
                modelFamily: "Qwen3.5",
                status: .ready,
                requiredPaths: [
                    repositoryRoot.appendingPathComponent("scripts/hf/dump_qwen35_reference.py"),
                    repositoryRoot.appendingPathComponent("Tests/MetalCompilerTests/Models/Qwen35/Qwen35ReferenceComparisonTests.swift"),
                    repositoryRoot.appendingPathComponent("TestData/qwen35_reference.safetensors"),
                ],
                command: "python3 scripts/hf/dump_qwen35_reference.py --output TestData/qwen35_reference.safetensors --decode-steps 2 --linear-block-ordinals 0,9,17",
                blocker: "",
                referenceSnapshot: repositoryRoot.appendingPathComponent("TestData/qwen35_reference.safetensors"),
                expectedSchemaVersion: 6,
                requiredTensorNames: [
                    "ref.meta.schema_version",
                    "ref.meta.linear_block_ordinals",
                    "ref.case_0.prefill.linear_ordinal_0.block.out_projection_partials",
                    "ref.case_0.prefill.linear_ordinal_0.block.out_projection_reduced",
                    "ref.case_0.prefill.linear_ordinal_9.block.out_projection_partials",
                    "ref.case_0.prefill.linear_ordinal_9.block.out_projection_reduced",
                    "ref.case_0.prefill.linear_ordinal_17.block.out_projection_partials",
                    "ref.case_0.prefill.linear_ordinal_17.block.out_projection_reduced",
                ]
            ),
        ]
    }
}

private enum SafetensorsHeaderError: Error {
    case invalidPrefix
    case invalidHeader
    case tensorMissing(String)
    case invalidTensorMetadata(String)
    case unsupportedDType(String, String)
    case invalidTensorLength(String, Int)
    case closeFailed(Error)
}

private struct SafetensorsHeader {
    struct Tensor {
        let dtype: String
        let shape: [Int]
        let dataOffsets: [Int]
    }

    let headerLength: Int
    let tensors: [String: Tensor]

    static func load(from url: URL) throws -> SafetensorsHeader {
        let handle = try FileHandle(forReadingFrom: url)
        do {
            guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
                throw SafetensorsHeaderError.invalidPrefix
            }
            let headerLength = Int(littleEndianUInt64(prefix))
            guard let headerData = try handle.read(upToCount: headerLength),
                  headerData.count == headerLength else {
                throw SafetensorsHeaderError.invalidHeader
            }
            let object = try JSONSerialization.jsonObject(with: headerData)
            guard let dictionary = object as? [String: Any] else {
                throw SafetensorsHeaderError.invalidHeader
            }

            var tensors: [String: Tensor] = [:]
            for (name, value) in dictionary where name != "__metadata__" {
                guard let metadata = value as? [String: Any],
                      let dtype = metadata["dtype"] as? String,
                      let shape = metadata["shape"] as? [Int],
                      let dataOffsets = metadata["data_offsets"] as? [Int],
                      dataOffsets.count == 2 else {
                    throw SafetensorsHeaderError.invalidTensorMetadata(name)
                }
                tensors[name] = Tensor(dtype: dtype, shape: shape, dataOffsets: dataOffsets)
            }

            try handle.close()
            return SafetensorsHeader(headerLength: headerLength, tensors: tensors)
        } catch {
            try close(handle)
            throw error
        }
    }

    func containsTensor(named name: String) -> Bool {
        tensors[name] != nil
    }

    func readInt32Tensor(named name: String, from url: URL) throws -> Int32 {
        guard let tensor = tensors[name] else {
            throw SafetensorsHeaderError.tensorMissing(name)
        }
        guard tensor.dtype == "I32" else {
            throw SafetensorsHeaderError.unsupportedDType(name, tensor.dtype)
        }
        let elementCount = tensor.shape.reduce(1, *)
        guard elementCount == 1 else {
            throw SafetensorsHeaderError.invalidTensorLength(name, elementCount)
        }
        let byteCount = tensor.dataOffsets[1] - tensor.dataOffsets[0]
        guard byteCount == 4 else {
            throw SafetensorsHeaderError.invalidTensorLength(name, byteCount)
        }

        let handle = try FileHandle(forReadingFrom: url)
        do {
            let offset = UInt64(8 + headerLength + tensor.dataOffsets[0])
            try handle.seek(toOffset: offset)
            guard let data = try handle.read(upToCount: 4), data.count == 4 else {
                throw SafetensorsHeaderError.invalidTensorLength(name, 0)
            }
            try handle.close()
            return Int32(bitPattern: UInt32(Self.littleEndianUInt32(data)))
        } catch {
            try Self.close(handle)
            throw error
        }
    }

    private static func close(_ handle: FileHandle) throws {
        do {
            try handle.close()
        } catch {
            throw SafetensorsHeaderError.closeFailed(error)
        }
    }

    private static func littleEndianUInt64(_ data: Data) -> UInt64 {
        data.enumerated().reduce(UInt64(0)) { value, element in
            value | (UInt64(element.element) << UInt64(element.offset * 8))
        }
    }

    private static func littleEndianUInt32(_ data: Data) -> UInt32 {
        data.enumerated().reduce(UInt32(0)) { value, element in
            value | (UInt32(element.element) << UInt32(element.offset * 8))
        }
    }
}
