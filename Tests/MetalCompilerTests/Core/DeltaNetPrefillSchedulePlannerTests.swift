import Foundation
import Testing
@testable import MetalCompiler

@Suite("DeltaNet Prefill Schedule Planner")
struct DeltaNetPrefillSchedulePlannerTests {
    @Test("planner admits block streaming statically but blocks promotion without gates")
    func plannerAdmitsBlockStreamingStaticallyButBlocksPromotionWithoutGates() {
        let rows = DeltaNetPrefillSchedulePlanner.decisions(
            shape: .qwen35Default(sequenceLength: 128)
        )
        let bySchedule = Dictionary(uniqueKeysWithValues: rows.map { ($0.scheduleID, $0) })
        let blockStreaming = bySchedule[.blockStreaming]

        #expect(blockStreaming?.staticFeasibility == "pass")
        #expect(blockStreaming?.supportsNonzeroInitialState == true)
        #expect(blockStreaming?.staticThreadgroupBytes == 17_932)
        let reduction = blockStreaming?.estimatedStateTrafficReductionPercent ?? .nan
        #expect(abs(reduction - 83.333_333_333_333_34) < 0.000_001)
        #expect(blockStreaming?.routePromotion == "reject-missing-correctness-gate")
    }

    @Test("planner rejects scan schedules that cannot preserve restored state")
    func plannerRejectsScanSchedulesThatCannotPreserveRestoredState() {
        let rows = DeltaNetPrefillSchedulePlanner.decisions(
            shape: .qwen35Default(sequenceLength: 128)
        )
        let blockScan = rows.first { $0.scheduleID == .blockScan }

        #expect(blockScan?.supportsNonzeroInitialState == false)
        #expect(blockScan?.staticFeasibility == "reject-prefill-state-contract")
        #expect(blockScan?.routePromotion == "reject-static-feasibility")
    }

    @Test("production decision requires correctness reference and profile gates")
    func productionDecisionRequiresCorrectnessReferenceAndProfileGates() {
        let shape = DeltaNetPrefillScheduleShape.qwen35Default(sequenceLength: 128)
        let missingGateDecision = DeltaNetPrefillSchedulePlanner.productionDecision(
            shape: shape,
            evidenceBySchedule: [
                .blockStreaming: DeltaNetPrefillScheduleEvidence(
                    correctnessGate: .pass,
                    referenceGate: .pass,
                    profileGate: .missing
                ),
            ]
        )
        #expect(missingGateDecision.scheduleID == .parallelState)
        #expect(missingGateDecision.routePromotion == "current-production-route")

        let passingGateDecision = DeltaNetPrefillSchedulePlanner.productionDecision(
            shape: shape,
            evidenceBySchedule: [
                .blockStreaming: DeltaNetPrefillScheduleEvidence(
                    correctnessGate: .pass,
                    referenceGate: .pass,
                    profileGate: .pass
                ),
            ]
        )
        #expect(passingGateDecision.scheduleID == .blockStreaming)
        #expect(passingGateDecision.routePromotion == "candidate-production-route")
    }

    @Test("experiment requested schedule does not become production route")
    func experimentRequestedScheduleDoesNotBecomeProductionRoute() {
        let shape = DeltaNetPrefillScheduleShape.qwen35Default(sequenceLength: 128)
        let decision = DeltaNetPrefillSchedulePlanner.decisions(
            shape: shape,
            evidenceBySchedule: [
                .blockStreaming: DeltaNetPrefillScheduleEvidence(
                    correctnessGate: .pass,
                    referenceGate: .pass,
                    profileGate: .pass,
                    experimentRequested: true
                ),
            ]
        )
            .first { $0.scheduleID == .blockStreaming }

        #expect(decision?.routePromotion == "experiment-only")
        #expect(DeltaNetPrefillSchedulePlanner.productionDecision(
            shape: shape,
            evidenceBySchedule: [
                .blockStreaming: DeltaNetPrefillScheduleEvidence(
                    correctnessGate: .pass,
                    referenceGate: .pass,
                    profileGate: .pass,
                    experimentRequested: true
                ),
            ]
        ).scheduleID == .parallelState)
    }

    @Test("schedule artifact can be written and reconstructed")
    func scheduleArtifactCanBeWrittenAndReconstructed() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-lm-deltanet-schedule-\(UUID().uuidString)", isDirectory: true)
        defer {
            tryCleanup(directory)
        }
        let rows = DeltaNetPrefillSchedulePlanner.decisions(
            shape: .qwen35Default(sequenceLength: 128),
            evidenceBySchedule: [
                .blockStreaming: DeltaNetPrefillScheduleEvidence(
                    correctnessGate: .pass,
                    referenceGate: .pass,
                    profileGate: .pass
                ),
            ]
        )
        let artifact = try DeltaNetPrefillScheduleArtifact.write(rows: rows, directory: directory)
        let reconstructed = try DeltaNetPrefillScheduleArtifact.read(artifact)
        let bySchedule = Dictionary(uniqueKeysWithValues: reconstructed.map { ($0.scheduleID, $0) })

        #expect(reconstructed.count == DeltaNetPrefillScheduleID.allCases.count)
        #expect(bySchedule[.blockStreaming]?.routePromotion == "candidate-production-route")
        #expect(bySchedule[.blockScan]?.staticFeasibility == "reject-prefill-state-contract")
    }

    @Test("current schedule artifact records unpromoted block streaming route")
    func currentScheduleArtifactRecordsUnpromotedBlockStreamingRoute() throws {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/ssm-recurrence-microbench", isDirectory: true)
        let rows = DeltaNetPrefillSchedulePlanner.decisions(
            shape: .qwen35Default(sequenceLength: 128)
        )
        let artifact = try DeltaNetPrefillScheduleArtifact.write(rows: rows, directory: directory)
        let reconstructed = try DeltaNetPrefillScheduleArtifact.read(artifact)
        let bySchedule = Dictionary(uniqueKeysWithValues: reconstructed.map { ($0.scheduleID, $0) })

        #expect(bySchedule[.parallelState]?.routePromotion == "current-production-route")
        #expect(bySchedule[.blockStreaming]?.staticFeasibility == "pass")
        #expect(bySchedule[.blockStreaming]?.routePromotion == "reject-missing-correctness-gate")
    }

    @Test("current schedule artifact can be reconstructed when requested")
    func currentScheduleArtifactCanBeReconstructedWhenRequested() throws {
        guard ProcessInfo.processInfo.environment["SWIFTLM_VALIDATE_DELTANET_PREFILL_SCHEDULE_ARTIFACTS"] == "1" else {
            return
        }
        let artifact = repositoryRoot()
            .appendingPathComponent(".test-artifacts/ssm-recurrence-microbench", isDirectory: true)
            .appendingPathComponent(DeltaNetPrefillScheduleArtifact.fileName)
        let rows = try DeltaNetPrefillScheduleArtifact.read(artifact)
        let bySchedule = Dictionary(uniqueKeysWithValues: rows.map { ($0.scheduleID, $0) })

        #expect(rows.count == DeltaNetPrefillScheduleID.allCases.count)
        #expect(bySchedule[.parallelState]?.routePromotion == "current-production-route")
        #expect(bySchedule[.blockStreaming]?.staticFeasibility == "pass")
    }
}

private func tryCleanup(_ url: URL) {
    do {
        try FileManager.default.removeItem(at: url)
    } catch {
        Issue.record("Failed to clean temporary directory \(url.path): \(error)")
    }
}

private enum DeltaNetPrefillScheduleArtifact {
    static let fileName = "qwen35-bf16-deltanet-prefill-schedules.csv"

    static func write(rows: [DeltaNetPrefillScheduleDecision], directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(fileName)
        var lines = [
            [
                "scheduleID",
                "scheduleClass",
                "supportsNonzeroInitialState",
                "staticThreadgroupBytes",
                "threadgroupMemoryLimitBytes",
                "estimatedStateGlobalBytesPerToken",
                "baselineStateGlobalBytesPerToken",
                "estimatedStateTrafficReductionPercent",
                "staticFeasibility",
                "staticRejectReason",
                "correctnessGate",
                "referenceGate",
                "profileGate",
                "routePromotion",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: { $0.scheduleID.rawValue < $1.scheduleID.rawValue }) {
            lines.append([
                row.scheduleID.rawValue,
                row.scheduleClass,
                String(row.supportsNonzeroInitialState),
                String(row.staticThreadgroupBytes),
                String(row.threadgroupMemoryLimitBytes),
                String(row.estimatedStateGlobalBytesPerToken),
                String(row.baselineStateGlobalBytesPerToken),
                String(format: "%.6f", row.estimatedStateTrafficReductionPercent),
                row.staticFeasibility,
                row.staticRejectReason,
                row.correctnessGate.rawValue,
                row.referenceGate.rawValue,
                row.profileGate.rawValue,
                row.routePromotion,
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    static func read(_ url: URL) throws -> [DeltaNetPrefillScheduleDecision] {
        try parseCSV(url).map { row in
            let scheduleName = try requiredValue("scheduleID", in: row, artifact: url)
            guard let scheduleID = DeltaNetPrefillScheduleID(rawValue: scheduleName) else {
                throw DeltaNetPrefillScheduleArtifactError.invalidValue(url.path, "scheduleID", scheduleName)
            }
            let correctness = try gateValue("correctnessGate", in: row, artifact: url)
            let reference = try gateValue("referenceGate", in: row, artifact: url)
            let profile = try gateValue("profileGate", in: row, artifact: url)
            return DeltaNetPrefillScheduleDecision(
                scheduleID: scheduleID,
                scheduleClass: try requiredValue("scheduleClass", in: row, artifact: url),
                supportsNonzeroInitialState: try boolValue("supportsNonzeroInitialState", in: row, artifact: url),
                staticThreadgroupBytes: try intValue("staticThreadgroupBytes", in: row, artifact: url),
                threadgroupMemoryLimitBytes: try intValue("threadgroupMemoryLimitBytes", in: row, artifact: url),
                estimatedStateGlobalBytesPerToken: try intValue("estimatedStateGlobalBytesPerToken", in: row, artifact: url),
                baselineStateGlobalBytesPerToken: try intValue("baselineStateGlobalBytesPerToken", in: row, artifact: url),
                estimatedStateTrafficReductionPercent: try doubleValue("estimatedStateTrafficReductionPercent", in: row, artifact: url),
                staticFeasibility: try requiredValue("staticFeasibility", in: row, artifact: url),
                staticRejectReason: try requiredValue("staticRejectReason", in: row, artifact: url),
                correctnessGate: correctness,
                referenceGate: reference,
                profileGate: profile,
                routePromotion: try requiredValue("routePromotion", in: row, artifact: url)
            )
        }
    }

    private static func parseCSV(_ url: URL) throws -> [[String: String]] {
        let content = try String(contentsOf: url, encoding: .utf8)
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let lines = content.split(whereSeparator: \.isNewline).map(String.init)
        guard let header = lines.first else {
            throw DeltaNetPrefillScheduleArtifactError.emptyCSV(url.path)
        }
        let columns = header.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        return try lines.dropFirst().map { line in
            let values = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            guard values.count == columns.count else {
                throw DeltaNetPrefillScheduleArtifactError.rowWidthMismatch(url.path, columns.count, values.count)
            }
            return Dictionary(uniqueKeysWithValues: zip(columns, values))
        }
    }

    private static func requiredValue(_ key: String, in row: [String: String], artifact: URL) throws -> String {
        guard let value = row[key] else {
            throw DeltaNetPrefillScheduleArtifactError.missingColumn(artifact.path, key)
        }
        return value
    }

    private static func intValue(_ key: String, in row: [String: String], artifact: URL) throws -> Int {
        let value = try requiredValue(key, in: row, artifact: artifact)
        guard let intValue = Int(value) else {
            throw DeltaNetPrefillScheduleArtifactError.invalidValue(artifact.path, key, value)
        }
        return intValue
    }

    private static func doubleValue(_ key: String, in row: [String: String], artifact: URL) throws -> Double {
        let value = try requiredValue(key, in: row, artifact: artifact)
        guard let doubleValue = Double(value) else {
            throw DeltaNetPrefillScheduleArtifactError.invalidValue(artifact.path, key, value)
        }
        return doubleValue
    }

    private static func boolValue(_ key: String, in row: [String: String], artifact: URL) throws -> Bool {
        let value = try requiredValue(key, in: row, artifact: artifact)
        if value == "true" { return true }
        if value == "false" { return false }
        throw DeltaNetPrefillScheduleArtifactError.invalidValue(artifact.path, key, value)
    }

    private static func gateValue(_ key: String, in row: [String: String], artifact: URL) throws -> DeltaNetPrefillGate {
        let value = try requiredValue(key, in: row, artifact: artifact)
        guard let gate = DeltaNetPrefillGate(rawValue: value) else {
            throw DeltaNetPrefillScheduleArtifactError.invalidValue(artifact.path, key, value)
        }
        return gate
    }
}

private enum DeltaNetPrefillScheduleArtifactError: Error, CustomStringConvertible {
    case emptyCSV(String)
    case invalidValue(String, String, String)
    case missingColumn(String, String)
    case rowWidthMismatch(String, Int, Int)

    var description: String {
        switch self {
        case .emptyCSV(let path):
            return "CSV artifact is empty: \(path)"
        case .invalidValue(let path, let column, let value):
            return "CSV artifact \(path) has invalid value \(value) in column \(column)"
        case .missingColumn(let path, let column):
            return "CSV artifact \(path) is missing required column \(column)"
        case .rowWidthMismatch(let path, let expected, let actual):
            return "CSV row width mismatch in \(path): expected \(expected), got \(actual)"
        }
    }
}

private func repositoryRoot() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}
