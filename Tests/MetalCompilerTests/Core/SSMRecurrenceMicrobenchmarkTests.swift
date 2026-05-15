import Foundation
import Metal
import Testing
@testable import MetalCompiler

@Suite("SSM Recurrence Microbenchmark", .serialized)
struct SSMRecurrenceMicrobenchmarkTests {
    private static let sequenceLengths = [16, 64, 128]
    private static let iterations = 5
    private static let warmupIterations = 1
    private static let stabilitySamples = 3
    private static let stabilityIterations = 3

    @Test("BF16 SSM recurrence real-shape microbench")
    func bf16SSMRecurrenceRealShapeMicrobench() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try SSMRecurrenceMicrobenchmarkHarness(device: device)
        let variants = [
            SSMVariant(name: "base_tg128", kernelName: "bench_ssm_recurrence_seq_bf16_f32", threadgroupWidth: 128),
            SSMVariant(name: "base_tg256", kernelName: "bench_ssm_recurrence_seq_bf16_f32", threadgroupWidth: 256),
            SSMVariant(name: "base_tg384", kernelName: "bench_ssm_recurrence_seq_bf16_f32", threadgroupWidth: 384),
            SSMVariant(name: "shared_tg128", kernelName: "bench_ssm_recurrence_seq_bf16_f32_shared_rms", threadgroupWidth: 128),
            SSMVariant(name: "shared_tg256", kernelName: "bench_ssm_recurrence_seq_bf16_f32_shared_rms", threadgroupWidth: 256),
            SSMVariant(name: "shared_tg384", kernelName: "bench_ssm_recurrence_seq_bf16_f32_shared_rms", threadgroupWidth: 384),
            SSMVariant(name: "prewrite_tg128", kernelName: "bench_ssm_recurrence_seq_bf16_f32_prewrite_decay", threadgroupWidth: 128),
            SSMVariant(name: "prewrite_tg256", kernelName: "bench_ssm_recurrence_seq_bf16_f32_prewrite_decay", threadgroupWidth: 256),
            SSMVariant(name: "prewrite_tg384", kernelName: "bench_ssm_recurrence_seq_bf16_f32_prewrite_decay", threadgroupWidth: 384),
            SSMVariant(name: "qkpar_tg128", kernelName: "bench_ssm_recurrence_seq_bf16_f32_qkpar", threadgroupWidth: 128),
            SSMVariant(name: "qkpar_tg256", kernelName: "bench_ssm_recurrence_seq_bf16_f32_qkpar", threadgroupWidth: 256),
            SSMVariant(name: "qkpar_tg384", kernelName: "bench_ssm_recurrence_seq_bf16_f32_qkpar", threadgroupWidth: 384),
        ]

        var rows: [SSMResultRow] = []
        for sequenceLength in Self.sequenceLengths {
            for variant in variants {
                let row = try harness.measure(
                    variant: variant,
                    sequenceLength: sequenceLength,
                    iterations: Self.iterations,
                    warmupIterations: Self.warmupIterations
                )
                rows.append(row)
            }
        }

        let summaryRows = summarize(rows: rows)
        let routePromotionRows = summarizeRoutePromotions(rows: rows)
        let artifact = try writeCSV(rows: rows)
        let summaryArtifact = try writeSummaryCSV(rows: summaryRows)
        let routePromotionArtifact = try writeRoutePromotionCSV(rows: routePromotionRows)
        printReport(
            rows: rows,
            summaryRows: summaryRows,
            routePromotionRows: routePromotionRows,
            artifact: artifact,
            summaryArtifact: summaryArtifact,
            routePromotionArtifact: routePromotionArtifact
        )
        #expect(rows.count == Self.sequenceLengths.count * variants.count)
        #expect(summaryRows.count == Self.sequenceLengths.count)
        #expect(routePromotionRows.count == SSMRoutePromotionCandidate.allCases.count)
        #expect(routePromotionRows.allSatisfy { $0.requiredSequenceCount == 2 })
    }

    @Test("BF16 SSM recurrence phase-isolation microbench")
    func bf16SSMRecurrencePhaseIsolationMicrobench() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try SSMRecurrenceMicrobenchmarkHarness(device: device)
        let phases = [
            SSMPhaseVariant(name: "conv_silu", kernelName: "bench_ssm_phase_conv_silu_bf16_f32", threadgroupWidth: 384),
            SSMPhaseVariant(name: "state_recurrence", kernelName: "bench_ssm_phase_state_recurrence_f32", threadgroupWidth: 384),
            SSMPhaseVariant(name: "state_recurrence_d2", kernelName: "bench_ssm_phase_state_recurrence_d2_f32", threadgroupWidth: 384),
            SSMPhaseVariant(name: "state_recurrence_qkpar", kernelName: "bench_ssm_phase_state_recurrence_qkpar_f32", threadgroupWidth: 384),
            SSMPhaseVariant(name: "state_recurrence_cache32", kernelName: "bench_ssm_phase_state_recurrence_cache32_f32", threadgroupWidth: 384),
            SSMPhaseVariant(name: "rms_gate", kernelName: "bench_ssm_phase_rms_gate_f32", threadgroupWidth: 384),
        ]

        var rows: [SSMPhaseResultRow] = []
        for sequenceLength in Self.sequenceLengths {
            let fullBase = try harness.measure(
                variant: SSMVariant(
                    name: "base_tg384",
                    kernelName: "bench_ssm_recurrence_seq_bf16_f32",
                    threadgroupWidth: 384
                ),
                sequenceLength: sequenceLength,
                iterations: Self.iterations,
                warmupIterations: Self.warmupIterations
            )
            for phase in phases {
                let row = try harness.measurePhase(
                    phase: phase,
                    sequenceLength: sequenceLength,
                    fullBaseAverageGpuMicroseconds: fullBase.averageGpuMicroseconds,
                    iterations: Self.iterations,
                    warmupIterations: Self.warmupIterations
                )
                rows.append(row)
            }
        }

        let artifact = try writePhaseCSV(rows: rows)
        printPhaseReport(rows: rows, artifact: artifact)
        #expect(rows.count == Self.sequenceLengths.count * phases.count)
        #expect(rows.allSatisfy { $0.outputChecksum.isFinite && $0.outputChecksum > 0 })
        #expect(rows.filter { $0.phase == "state_recurrence_cache32" }.allSatisfy {
            $0.phasePromotionAdmission == "reject-lane-parallelism-lost"
        })
        #expect(rows.filter { $0.phase == "state_recurrence_d2" }.allSatisfy {
            $0.phasePromotionAdmission == "reject-serial-value-lanes"
        })
        #expect(rows.filter { $0.phase == "state_recurrence_qkpar" }.allSatisfy {
            $0.phasePromotionAdmission == "eligible-for-full-kernel-check"
        })
    }

    @Test("BF16 SSM state recurrence phase matches CPU reference")
    func bf16SSMStateRecurrencePhaseMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try SSMRecurrenceMicrobenchmarkHarness(device: device)
        for kernelName in [
            "bench_ssm_phase_state_recurrence_f32",
            "bench_ssm_phase_state_recurrence_d2_f32",
            "bench_ssm_phase_state_recurrence_qkpar_f32",
            "bench_ssm_phase_state_recurrence_cache32_f32",
        ] {
            let validation = try harness.validateStateRecurrencePhase(kernelName: kernelName, sequenceLength: 5)
            #expect(
                validation.outputMaxError <= 0.000_5,
                "\(kernelName) output drifted: maxError=\(validation.outputMaxError)"
            )
            #expect(
                validation.recurrentStateMaxError <= 0.000_5,
                "\(kernelName) recurrent state drifted: maxError=\(validation.recurrentStateMaxError)"
            )
        }
    }

    @Test("BF16 SSM state recurrence candidate stability microbench")
    func bf16SSMStateRecurrenceCandidateStabilityMicrobench() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try SSMRecurrenceMicrobenchmarkHarness(device: device)
        let sequenceLength = 128
        let baselinePhase = SSMPhaseVariant(
            name: "state_recurrence",
            kernelName: "bench_ssm_phase_state_recurrence_f32",
            threadgroupWidth: 384
        )
        let candidatePhases = [
            SSMPhaseVariant(
                name: "state_recurrence_d2",
                kernelName: "bench_ssm_phase_state_recurrence_d2_f32",
                threadgroupWidth: 384
            ),
            SSMPhaseVariant(
                name: "state_recurrence_qkpar",
                kernelName: "bench_ssm_phase_state_recurrence_qkpar_f32",
                threadgroupWidth: 384
            ),
            SSMPhaseVariant(
                name: "state_recurrence_cache32",
                kernelName: "bench_ssm_phase_state_recurrence_cache32_f32",
                threadgroupWidth: 384
            ),
        ]

        var rows: [SSMPhaseStabilityRow] = []
        for sampleIndex in 0..<Self.stabilitySamples {
            let baseline = try harness.measurePhase(
                phase: baselinePhase,
                sequenceLength: sequenceLength,
                fullBaseAverageGpuMicroseconds: 1.0,
                iterations: Self.stabilityIterations,
                warmupIterations: Self.warmupIterations
            )
            for candidate in candidatePhases {
                let candidateRow = try harness.measurePhase(
                    phase: candidate,
                    sequenceLength: sequenceLength,
                    fullBaseAverageGpuMicroseconds: 1.0,
                    iterations: Self.stabilityIterations,
                    warmupIterations: Self.warmupIterations
                )
                rows.append(SSMPhaseStabilityRow(sampleIndex: sampleIndex, baseline: baseline, candidate: candidateRow))
            }
        }

        let artifact = try writePhaseStabilityCSV(rows: rows)
        printPhaseStabilityReport(rows: rows, artifact: artifact)
        #expect(rows.count == Self.stabilitySamples * candidatePhases.count)
        #expect(rows.allSatisfy { $0.candidateOutputChecksum.isFinite && $0.candidateOutputChecksum > 0 })
    }

    @Test("SSM summary promotion admissions classify candidates")
    func ssmSummaryPromotionAdmissionsClassifyCandidates() {
        let base = makeSummaryFixture(sequenceLength: 128, variant: "base_tg384", averageGpuMicroseconds: 100.0)

        let shared = SSMSummaryFactory.make(
            sequenceLength: 128,
            best: makeSummaryFixture(sequenceLength: 128, variant: "shared_tg384", averageGpuMicroseconds: 95.0),
            bestBase: base,
            speedupVsBestBasePercent: 5.0
        )
        #expect(shared.decision == "candidate-shared-rms")
        #expect(shared.promotionAdmission == "candidate-shared-rms")

        let shortQKParallel = SSMSummaryFactory.make(
            sequenceLength: 16,
            best: makeSummaryFixture(sequenceLength: 16, variant: "qkpar_tg384", averageGpuMicroseconds: 70.0),
            bestBase: makeSummaryFixture(sequenceLength: 16, variant: "base_tg384", averageGpuMicroseconds: 100.0),
            speedupVsBestBasePercent: 30.0
        )
        #expect(shortQKParallel.decision == "keep-default")
        #expect(shortQKParallel.promotionAdmission == "reject-short-sequence-only")

        let noisyQKParallel = SSMSummaryFactory.make(
            sequenceLength: 128,
            best: makeSummaryFixture(sequenceLength: 128, variant: "qkpar_tg384", averageGpuMicroseconds: 98.0),
            bestBase: base,
            speedupVsBestBasePercent: 2.0
        )
        #expect(noisyQKParallel.decision == "keep-default")
        #expect(noisyQKParallel.promotionAdmission == "reject-speedup-below-threshold")

        let fastQKParallel = SSMSummaryFactory.make(
            sequenceLength: 128,
            best: makeSummaryFixture(sequenceLength: 128, variant: "qkpar_tg384", averageGpuMicroseconds: 90.0),
            bestBase: base,
            speedupVsBestBasePercent: 10.0
        )
        #expect(fastQKParallel.decision == "candidate-qkpar-full-kernel")
        #expect(fastQKParallel.promotionAdmission == "candidate-qkpar-full-kernel")
    }

    @Test("SSM route promotion admissions require production sequence wins")
    func ssmRoutePromotionAdmissionsRequireProductionSequenceWins() {
        let rows = [
            makeSummaryFixture(sequenceLength: 16, variant: "base_tg384", averageGpuMicroseconds: 100.0),
            makeSummaryFixture(sequenceLength: 16, variant: "shared_tg384", averageGpuMicroseconds: 50.0),
            makeSummaryFixture(sequenceLength: 16, variant: "qkpar_tg384", averageGpuMicroseconds: 50.0),
            makeSummaryFixture(sequenceLength: 64, variant: "base_tg384", averageGpuMicroseconds: 100.0),
            makeSummaryFixture(sequenceLength: 64, variant: "shared_tg384", averageGpuMicroseconds: 94.0),
            makeSummaryFixture(sequenceLength: 64, variant: "qkpar_tg384", averageGpuMicroseconds: 90.0),
            makeSummaryFixture(sequenceLength: 128, variant: "base_tg384", averageGpuMicroseconds: 100.0),
            makeSummaryFixture(sequenceLength: 128, variant: "shared_tg384", averageGpuMicroseconds: 99.0),
            makeSummaryFixture(sequenceLength: 128, variant: "qkpar_tg384", averageGpuMicroseconds: 90.0),
        ]
        let routeRows = summarizeRoutePromotions(rows: rows)
        let shared = routeRows.first { $0.candidate == .sharedRMS }
        let qkParallel = routeRows.first { $0.candidate == .qkParallel }
        #expect(shared?.routePromotionAdmission == "reject-cross-sequence-threshold")
        #expect(shared?.failingSequenceLengths == [128])
        #expect(shared?.thresholdShortfallPercent == 2.0)
        #expect(qkParallel?.routePromotionAdmission == "candidate-qkpar-default-route")
        #expect(qkParallel?.failingSequenceLengths == [])
        #expect(qkParallel?.thresholdShortfallPercent == 0.0)
    }

    private func printReport(
        rows: [SSMResultRow],
        summaryRows: [SSMSummaryRow],
        routePromotionRows: [SSMRoutePromotionRow],
        artifact: URL,
        summaryArtifact: URL,
        routePromotionArtifact: URL
    ) {
        print()
        print("=== BF16 SSM recurrence real-shape microbench ===")
        print("artifact: \(artifact.path)")
        print("summary artifact: \(summaryArtifact.path)")
        print("route promotion artifact: \(routePromotionArtifact.path)")
        print("seq  variant       avg_us  us/token  grid   tg")
        for row in rows.sorted(by: rowSort) {
            let variant = row.variant.padding(toLength: 16, withPad: " ", startingAt: 0)
            let grid = "\(row.gridWidth)x\(row.gridHeight)".padding(toLength: 6, withPad: " ", startingAt: 0)
            print("  \(String(format: "%3d", row.sequenceLength))  \(variant) \(String(format: "%7.1f", row.averageGpuMicroseconds))  \(String(format: "%8.3f", row.microsecondsPerToken))  \(grid) \(row.threadgroupWidth)")
        }
        print()
        print("=== BF16 SSM recurrence promotion decisions ===")
        print("seq  best             best_us  state_mb/tok  base             base_us  speedup  decision                     admission")
        for row in summaryRows.sorted(by: { $0.sequenceLength < $1.sequenceLength }) {
            let bestVariant = row.bestVariant.padding(toLength: 16, withPad: " ", startingAt: 0)
            let baseVariant = row.bestBaseVariant.padding(toLength: 16, withPad: " ", startingAt: 0)
            let decision = row.decision.padding(toLength: 28, withPad: " ", startingAt: 0)
            let stateMegabytes = Double(row.bestEstimatedStateTotalBytesPerToken) / 1_048_576.0
            print("  \(String(format: "%3d", row.sequenceLength))  \(bestVariant) \(String(format: "%7.1f", row.bestAverageGpuMicroseconds))  \(String(format: "%12.3f", stateMegabytes))  \(baseVariant) \(String(format: "%7.1f", row.bestBaseAverageGpuMicroseconds))  \(String(format: "%6.2f", row.speedupVsBestBasePercent))%  \(decision) \(row.promotionAdmission)")
        }
        print()
        print("=== BF16 SSM recurrence route promotion admissions ===")
        print("candidate       pass/required  min_speedup  shortfall  failing_seq  admission")
        for row in routePromotionRows.sorted(by: { $0.candidate.rawValue < $1.candidate.rawValue }) {
            let candidate = row.candidate.rawValue.padding(toLength: 15, withPad: " ", startingAt: 0)
            let failingSequences = row.failingSequenceLengths.map(String.init).joined(separator: "|")
            print("  \(candidate) \(row.passingSequenceCount)/\(row.requiredSequenceCount)          \(String(format: "%7.2f", row.minimumSpeedupPercent))%  \(String(format: "%8.2f", row.thresholdShortfallPercent))%  \(failingSequences.padding(toLength: 11, withPad: " ", startingAt: 0))  \(row.routePromotionAdmission)")
        }
    }

    private func writeCSV(rows: [SSMResultRow]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/ssm-recurrence-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("qwen35-bf16-ssm-recurrence.csv")
        var lines = [
            [
                "sequenceLength",
                "variant",
                "headCount",
                "groupCount",
                "keyDimension",
                "valueDimension",
                "convKernelSize",
                "gridWidth",
                "gridHeight",
                "threadgroupWidth",
                "requestedThreadgroupWidth",
                "averageGpuMicroseconds",
                "microsecondsPerToken",
                "stateElementsPerToken",
                "estimatedStateReadBytesPerToken",
                "estimatedStateWriteBytesPerToken",
                "estimatedStateTotalBytesPerToken",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: rowSort) {
            lines.append([
                String(row.sequenceLength),
                row.variant,
                String(row.headCount),
                String(row.groupCount),
                String(row.keyDimension),
                String(row.valueDimension),
                String(row.convKernelSize),
                String(row.gridWidth),
                String(row.gridHeight),
                String(row.threadgroupWidth),
                String(row.requestedThreadgroupWidth),
                String(format: "%.3f", row.averageGpuMicroseconds),
                String(format: "%.6f", row.microsecondsPerToken),
                String(row.stateElementsPerToken),
                String(row.estimatedStateReadBytesPerToken),
                String(row.estimatedStateWriteBytesPerToken),
                String(row.estimatedStateTotalBytesPerToken),
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func writeSummaryCSV(rows: [SSMSummaryRow]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/ssm-recurrence-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("qwen35-bf16-ssm-recurrence-summary.csv")
        var lines = [
            [
                "sequenceLength",
                "bestVariant",
                "bestThreadgroupWidth",
                "bestAverageGpuMicroseconds",
                "bestMicrosecondsPerToken",
                "bestEstimatedStateTotalBytesPerToken",
                "bestBaseVariant",
                "bestBaseThreadgroupWidth",
                "bestBaseAverageGpuMicroseconds",
                "bestBaseEstimatedStateTotalBytesPerToken",
                "speedupVsBestBasePercent",
                "decision",
                "promotionAdmission",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: { $0.sequenceLength < $1.sequenceLength }) {
            lines.append([
                String(row.sequenceLength),
                row.bestVariant,
                String(row.bestThreadgroupWidth),
                String(format: "%.3f", row.bestAverageGpuMicroseconds),
                String(format: "%.6f", row.bestMicrosecondsPerToken),
                String(row.bestEstimatedStateTotalBytesPerToken),
                row.bestBaseVariant,
                String(row.bestBaseThreadgroupWidth),
                String(format: "%.3f", row.bestBaseAverageGpuMicroseconds),
                String(row.bestBaseEstimatedStateTotalBytesPerToken),
                String(format: "%.3f", row.speedupVsBestBasePercent),
                row.decision,
                row.promotionAdmission,
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func writeRoutePromotionCSV(rows: [SSMRoutePromotionRow]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/ssm-recurrence-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("qwen35-bf16-ssm-recurrence-route-promotions.csv")
        var lines = [
            [
                "candidate",
                "productionSequenceLengths",
                "bestVariants",
                "speedupPercents",
                "passingSequenceCount",
                "requiredSequenceCount",
                "minimumSpeedupPercent",
                "thresholdShortfallPercent",
                "failingSequenceLengths",
                "routePromotionAdmission",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: { $0.candidate.rawValue < $1.candidate.rawValue }) {
            lines.append([
                row.candidate.rawValue,
                row.productionSequenceLengths.map(String.init).joined(separator: "|"),
                row.bestVariants.joined(separator: "|"),
                row.speedupPercents.map { String(format: "%.3f", $0) }.joined(separator: "|"),
                String(row.passingSequenceCount),
                String(row.requiredSequenceCount),
                String(format: "%.3f", row.minimumSpeedupPercent),
                String(format: "%.3f", row.thresholdShortfallPercent),
                row.failingSequenceLengths.map(String.init).joined(separator: "|"),
                row.routePromotionAdmission,
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func writePhaseCSV(rows: [SSMPhaseResultRow]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/ssm-recurrence-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("qwen35-bf16-ssm-recurrence-phases.csv")
        var lines = [
            [
                "sequenceLength",
                "phase",
                "gridWidth",
                "gridHeight",
                "threadgroupWidth",
                "requestedThreadgroupWidth",
                "averageGpuMicroseconds",
                "microsecondsPerToken",
                "fullBaseAverageGpuMicroseconds",
                "relativeToFullBasePercent",
                "activeThreadsPerThreadgroup",
                "valueLanesPerThread",
                "laneParallelismPreserved",
                "stateInnerStrideElements",
                "coalescedValueLanesPerStateRow",
                "serialStateLanesPerThread",
                "estimatedStateTotalBytesPerToken",
                "phasePromotionAdmission",
                "outputChecksum",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: phaseRowSort) {
            lines.append([
                String(row.sequenceLength),
                row.phase,
                String(row.gridWidth),
                String(row.gridHeight),
                String(row.threadgroupWidth),
                String(row.requestedThreadgroupWidth),
                String(format: "%.3f", row.averageGpuMicroseconds),
                String(format: "%.6f", row.microsecondsPerToken),
                String(format: "%.3f", row.fullBaseAverageGpuMicroseconds),
                String(format: "%.3f", row.relativeToFullBasePercent),
                String(row.activeThreadsPerThreadgroup),
                String(row.valueLanesPerThread),
                String(row.laneParallelismPreserved),
                String(row.stateInnerStrideElements),
                String(row.coalescedValueLanesPerStateRow),
                String(row.serialStateLanesPerThread),
                String(row.estimatedStateTotalBytesPerToken),
                row.phasePromotionAdmission,
                String(format: "%.6f", row.outputChecksum),
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func writePhaseStabilityCSV(rows: [SSMPhaseStabilityRow]) throws -> URL {
        let directory = repositoryRoot()
            .appendingPathComponent(".test-artifacts/ssm-recurrence-microbench", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("qwen35-bf16-ssm-recurrence-phase-stability.csv")
        var lines = [
            [
                "sampleIndex",
                "sequenceLength",
                "candidatePhase",
                "baselineAverageGpuMicroseconds",
                "candidateAverageGpuMicroseconds",
                "candidateDeltaPercent",
                "candidateWins",
                "activeThreadsPerThreadgroup",
                "valueLanesPerThread",
                "laneParallelismPreserved",
                "stateInnerStrideElements",
                "coalescedValueLanesPerStateRow",
                "serialStateLanesPerThread",
                "candidatePromotionAdmission",
                "candidateOutputChecksum",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: phaseStabilityRowSort) {
            lines.append([
                String(row.sampleIndex),
                String(row.sequenceLength),
                row.candidatePhase,
                String(format: "%.3f", row.baselineAverageGpuMicroseconds),
                String(format: "%.3f", row.candidateAverageGpuMicroseconds),
                String(format: "%.3f", row.candidateDeltaPercent),
                String(row.candidateWins),
                String(row.activeThreadsPerThreadgroup),
                String(row.valueLanesPerThread),
                String(row.laneParallelismPreserved),
                String(row.stateInnerStrideElements),
                String(row.coalescedValueLanesPerStateRow),
                String(row.serialStateLanesPerThread),
                row.candidatePromotionAdmission,
                String(format: "%.6f", row.candidateOutputChecksum),
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func printPhaseReport(rows: [SSMPhaseResultRow], artifact: URL) {
        print()
        print("=== BF16 SSM recurrence phase-isolation microbench ===")
        print("artifact: \(artifact.path)")
        print("seq  phase                      avg_us  us/token  full_%  active  lanes  stride  coalesced  state_mb/tok")
        for row in rows.sorted(by: phaseRowSort) {
            let phase = row.phase.padding(toLength: 25, withPad: " ", startingAt: 0)
            let stateMegabytes = Double(row.estimatedStateTotalBytesPerToken) / 1_048_576.0
            print("  \(String(format: "%3d", row.sequenceLength))  \(phase) \(String(format: "%7.1f", row.averageGpuMicroseconds))  \(String(format: "%8.3f", row.microsecondsPerToken))  \(String(format: "%6.2f", row.relativeToFullBasePercent))  \(String(format: "%6d", row.activeThreadsPerThreadgroup))  \(String(format: "%5d", row.valueLanesPerThread))  \(String(format: "%6d", row.stateInnerStrideElements))  \(String(format: "%9d", row.coalescedValueLanesPerStateRow))  \(String(format: "%12.3f", stateMegabytes))")
        }
    }

    private func printPhaseStabilityReport(rows: [SSMPhaseStabilityRow], artifact: URL) {
        print()
        print("=== BF16 SSM state recurrence candidate stability ===")
        print("artifact: \(artifact.path)")
        print("sample  candidate                 base_us  cand_us  delta_%  wins")
        for row in rows.sorted(by: phaseStabilityRowSort) {
            let candidate = row.candidatePhase.padding(toLength: 25, withPad: " ", startingAt: 0)
            print("  \(String(format: "%3d", row.sampleIndex))   \(candidate) \(String(format: "%7.1f", row.baselineAverageGpuMicroseconds))  \(String(format: "%7.1f", row.candidateAverageGpuMicroseconds))  \(String(format: "%7.2f", row.candidateDeltaPercent))  \(row.candidateWins)")
        }
    }

    private func summarize(rows: [SSMResultRow]) -> [SSMSummaryRow] {
        Self.sequenceLengths.compactMap { sequenceLength in
            let sequenceRows = rows.filter { $0.sequenceLength == sequenceLength }
            let baseRows = sequenceRows.filter { $0.variant.hasPrefix("base_") }
            guard let best = sequenceRows.min(by: averageSort),
                  let bestBase = baseRows.min(by: averageSort) else {
                return nil
            }
            let speedup = (bestBase.averageGpuMicroseconds - best.averageGpuMicroseconds)
                / bestBase.averageGpuMicroseconds * 100.0
            return SSMSummaryFactory.make(
                sequenceLength: sequenceLength,
                best: best,
                bestBase: bestBase,
                speedupVsBestBasePercent: speedup
            )
        }
    }

    private func summarizeRoutePromotions(rows: [SSMResultRow]) -> [SSMRoutePromotionRow] {
        let productionSequenceLengths = Self.sequenceLengths.filter { $0 >= SSMSummaryFactory.minimumPromotionSequenceLength }
        return SSMRoutePromotionCandidate.allCases.map { candidate in
            SSMRoutePromotionFactory.make(
                candidate: candidate,
                productionSequenceLengths: productionSequenceLengths,
                rows: rows
            )
        }
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func rowSort(_ lhs: SSMResultRow, _ rhs: SSMResultRow) -> Bool {
        if lhs.sequenceLength != rhs.sequenceLength {
            return lhs.sequenceLength < rhs.sequenceLength
        }
        if lhs.requestedThreadgroupWidth != rhs.requestedThreadgroupWidth {
            return lhs.requestedThreadgroupWidth < rhs.requestedThreadgroupWidth
        }
        return lhs.variant < rhs.variant
    }

    private func averageSort(_ lhs: SSMResultRow, _ rhs: SSMResultRow) -> Bool {
        if lhs.averageGpuMicroseconds != rhs.averageGpuMicroseconds {
            return lhs.averageGpuMicroseconds < rhs.averageGpuMicroseconds
        }
        if lhs.requestedThreadgroupWidth != rhs.requestedThreadgroupWidth {
            return lhs.requestedThreadgroupWidth < rhs.requestedThreadgroupWidth
        }
        return lhs.variant < rhs.variant
    }

    private func phaseRowSort(_ lhs: SSMPhaseResultRow, _ rhs: SSMPhaseResultRow) -> Bool {
        if lhs.sequenceLength != rhs.sequenceLength {
            return lhs.sequenceLength < rhs.sequenceLength
        }
        return lhs.phase < rhs.phase
    }

    private func phaseStabilityRowSort(_ lhs: SSMPhaseStabilityRow, _ rhs: SSMPhaseStabilityRow) -> Bool {
        if lhs.sampleIndex != rhs.sampleIndex {
            return lhs.sampleIndex < rhs.sampleIndex
        }
        return lhs.candidatePhase < rhs.candidatePhase
    }

    private func makeSummaryFixture(
        sequenceLength: Int,
        variant: String,
        averageGpuMicroseconds: Double
    ) -> SSMResultRow {
        SSMResultRow(
            sequenceLength: sequenceLength,
            variant: variant,
            headCount: 16,
            groupCount: 16,
            keyDimension: 128,
            valueDimension: 128,
            convKernelSize: 4,
            gridWidth: 16,
            gridHeight: 1,
            threadgroupWidth: 384,
            requestedThreadgroupWidth: 384,
            averageGpuMicroseconds: averageGpuMicroseconds
        )
    }
}

private struct SSMVariant {
    let name: String
    let kernelName: String
    let threadgroupWidth: Int
}

private struct SSMPhaseVariant {
    let name: String
    let kernelName: String
    let threadgroupWidth: Int
}

private struct SSMResultRow {
    let sequenceLength: Int
    let variant: String
    let headCount: Int
    let groupCount: Int
    let keyDimension: Int
    let valueDimension: Int
    let convKernelSize: Int
    let gridWidth: Int
    let gridHeight: Int
    let threadgroupWidth: Int
    let requestedThreadgroupWidth: Int
    let averageGpuMicroseconds: Double

    var microsecondsPerToken: Double {
        averageGpuMicroseconds / Double(sequenceLength)
    }

    var stateElementsPerToken: Int {
        headCount * keyDimension * valueDimension
    }

    var estimatedStateReadBytesPerToken: Int {
        stateElementsPerToken * MemoryLayout<Float>.stride * 2
    }

    var estimatedStateWriteBytesPerToken: Int {
        stateElementsPerToken * MemoryLayout<Float>.stride * stateWritePassesPerElement
    }

    var estimatedStateTotalBytesPerToken: Int {
        estimatedStateReadBytesPerToken + estimatedStateWriteBytesPerToken
    }

    private var stateWritePassesPerElement: Int {
        variant.hasPrefix("prewrite_") ? 2 : 1
    }
}

private struct SSMPhaseResultRow {
    let sequenceLength: Int
    let phase: String
    let headCount: Int
    let keyDimension: Int
    let valueDimension: Int
    let gridWidth: Int
    let gridHeight: Int
    let threadgroupWidth: Int
    let requestedThreadgroupWidth: Int
    let averageGpuMicroseconds: Double
    let fullBaseAverageGpuMicroseconds: Double
    let outputChecksum: Double

    var microsecondsPerToken: Double {
        averageGpuMicroseconds / Double(sequenceLength)
    }

    var relativeToFullBasePercent: Double {
        averageGpuMicroseconds / fullBaseAverageGpuMicroseconds * 100.0
    }

    var estimatedStateTotalBytesPerToken: Int {
        guard phase.hasPrefix("state_recurrence") else { return 0 }
        let devicePasses = stateCacheTileWidth == nil ? 3 : 2
        return headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride * devicePasses
    }

    var valueLanesPerThread: Int {
        phase == "state_recurrence_d2" ? 2 : 1
    }

    var activeThreadsPerThreadgroup: Int {
        if phase == "conv_silu" {
            return threadgroupWidth
        }
        if let stateCacheTileWidth {
            return min(threadgroupWidth, stateCacheTileWidth)
        }
        let valueThreadCount = (valueDimension + valueLanesPerThread - 1) / valueLanesPerThread
        return min(threadgroupWidth, valueThreadCount)
    }

    var laneParallelismPreserved: Bool {
        valueLanesPerThread == 1 && coalescedValueLanesPerStateRow == valueDimension
    }

    var stateInnerStrideElements: Int {
        phase.hasPrefix("state_recurrence") ? valueDimension : 0
    }

    var coalescedValueLanesPerStateRow: Int {
        phase.hasPrefix("state_recurrence") ? activeThreadsPerThreadgroup * valueLanesPerThread : 0
    }

    var serialStateLanesPerThread: Int {
        if let stateCacheTileWidth {
            return (valueDimension + stateCacheTileWidth - 1) / stateCacheTileWidth
        }
        return phase.hasPrefix("state_recurrence") ? valueLanesPerThread : 0
    }

    var phasePromotionAdmission: String {
        guard phase.hasPrefix("state_recurrence") else {
            return "not-state-recurrence"
        }
        if phase == "state_recurrence" {
            return "baseline"
        }
        if coalescedValueLanesPerStateRow < valueDimension {
            return "reject-lane-parallelism-lost"
        }
        if valueLanesPerThread != 1 {
            return "reject-serial-value-lanes"
        }
        return "eligible-for-full-kernel-check"
    }

    private var stateCacheTileWidth: Int? {
        if phase == "state_recurrence_cache32" {
            return 32
        }
        return nil
    }
}

private struct SSMPhaseStabilityRow {
    let sampleIndex: Int
    let sequenceLength: Int
    let candidatePhase: String
    let baselineAverageGpuMicroseconds: Double
    let candidateAverageGpuMicroseconds: Double
    let candidateOutputChecksum: Double
    let activeThreadsPerThreadgroup: Int
    let valueLanesPerThread: Int
    let laneParallelismPreserved: Bool
    let stateInnerStrideElements: Int
    let coalescedValueLanesPerStateRow: Int
    let serialStateLanesPerThread: Int
    let candidatePromotionAdmission: String

    init(sampleIndex: Int, baseline: SSMPhaseResultRow, candidate: SSMPhaseResultRow) {
        self.sampleIndex = sampleIndex
        self.sequenceLength = candidate.sequenceLength
        self.candidatePhase = candidate.phase
        self.baselineAverageGpuMicroseconds = baseline.averageGpuMicroseconds
        self.candidateAverageGpuMicroseconds = candidate.averageGpuMicroseconds
        self.candidateOutputChecksum = candidate.outputChecksum
        self.activeThreadsPerThreadgroup = candidate.activeThreadsPerThreadgroup
        self.valueLanesPerThread = candidate.valueLanesPerThread
        self.laneParallelismPreserved = candidate.laneParallelismPreserved
        self.stateInnerStrideElements = candidate.stateInnerStrideElements
        self.coalescedValueLanesPerStateRow = candidate.coalescedValueLanesPerStateRow
        self.serialStateLanesPerThread = candidate.serialStateLanesPerThread
        self.candidatePromotionAdmission = candidate.phasePromotionAdmission
    }

    var candidateDeltaPercent: Double {
        (candidateAverageGpuMicroseconds - baselineAverageGpuMicroseconds)
            / baselineAverageGpuMicroseconds * 100.0
    }

    var candidateWins: Bool {
        candidateAverageGpuMicroseconds < baselineAverageGpuMicroseconds
    }
}

private struct SSMStatePhaseValidation {
    let outputMaxError: Float
    let recurrentStateMaxError: Float
}

private struct SSMSummaryRow {
    let sequenceLength: Int
    let bestVariant: String
    let bestThreadgroupWidth: Int
    let bestAverageGpuMicroseconds: Double
    let bestMicrosecondsPerToken: Double
    let bestEstimatedStateTotalBytesPerToken: Int
    let bestBaseVariant: String
    let bestBaseThreadgroupWidth: Int
    let bestBaseAverageGpuMicroseconds: Double
    let bestBaseEstimatedStateTotalBytesPerToken: Int
    let speedupVsBestBasePercent: Double
    let decision: String
    let promotionAdmission: String
}

private enum SSMRoutePromotionCandidate: String, CaseIterable {
    case prewriteDecay = "prewrite_decay"
    case qkParallel = "qkpar"
    case sharedRMS = "shared_rms"

    var variantPrefix: String {
        switch self {
        case .prewriteDecay:
            return "prewrite_"
        case .qkParallel:
            return "qkpar_"
        case .sharedRMS:
            return "shared_"
        }
    }

    var routePromotionAdmission: String {
        switch self {
        case .prewriteDecay:
            return "candidate-prewrite-decay-default-route"
        case .qkParallel:
            return "candidate-qkpar-default-route"
        case .sharedRMS:
            return "candidate-shared-rms-default-route"
        }
    }
}

private struct SSMRoutePromotionRow {
    let candidate: SSMRoutePromotionCandidate
    let productionSequenceLengths: [Int]
    let bestVariants: [String]
    let speedupPercents: [Double]
    let passingSequenceCount: Int
    let requiredSequenceCount: Int
    let failingSequenceLengths: [Int]
    let routePromotionAdmission: String

    var minimumSpeedupPercent: Double {
        speedupPercents.min() ?? .nan
    }

    var thresholdShortfallPercent: Double {
        guard minimumSpeedupPercent.isFinite else { return .infinity }
        return max(0.0, SSMSummaryFactory.promotionSpeedupThresholdPercent - minimumSpeedupPercent)
    }
}

private enum SSMSummaryFactory {
    static let promotionSpeedupThresholdPercent = 3.0
    static let minimumPromotionSequenceLength = 64

    static func make(
        sequenceLength: Int,
        best: SSMResultRow,
        bestBase: SSMResultRow,
        speedupVsBestBasePercent: Double
    ) -> SSMSummaryRow {
        let admission = promotionAdmission(
            sequenceLength: sequenceLength,
            bestVariant: best.variant,
            speedupVsBestBasePercent: speedupVsBestBasePercent
        )
        return SSMSummaryRow(
            sequenceLength: sequenceLength,
            bestVariant: best.variant,
            bestThreadgroupWidth: best.threadgroupWidth,
            bestAverageGpuMicroseconds: best.averageGpuMicroseconds,
            bestMicrosecondsPerToken: best.microsecondsPerToken,
            bestEstimatedStateTotalBytesPerToken: best.estimatedStateTotalBytesPerToken,
            bestBaseVariant: bestBase.variant,
            bestBaseThreadgroupWidth: bestBase.threadgroupWidth,
            bestBaseAverageGpuMicroseconds: bestBase.averageGpuMicroseconds,
            bestBaseEstimatedStateTotalBytesPerToken: bestBase.estimatedStateTotalBytesPerToken,
            speedupVsBestBasePercent: speedupVsBestBasePercent,
            decision: decision(promotionAdmission: admission),
            promotionAdmission: admission
        )
    }

    private static func decision(promotionAdmission: String) -> String {
        promotionAdmission.hasPrefix("candidate-") ? promotionAdmission : "keep-default"
    }

    private static func promotionAdmission(
        sequenceLength: Int,
        bestVariant: String,
        speedupVsBestBasePercent: Double
    ) -> String {
        guard !bestVariant.hasPrefix("base_") else { return "baseline-best" }
        guard sequenceLength >= minimumPromotionSequenceLength else { return "reject-short-sequence-only" }
        guard speedupVsBestBasePercent >= promotionSpeedupThresholdPercent else { return "reject-speedup-below-threshold" }
        if bestVariant.hasPrefix("shared_") {
            return "candidate-shared-rms"
        }
        if bestVariant.hasPrefix("prewrite_") {
            return "candidate-prewrite-decay"
        }
        if bestVariant.hasPrefix("qkpar_") {
            return "candidate-qkpar-full-kernel"
        }
        return "reject-unknown-variant"
    }
}

private enum SSMRoutePromotionFactory {
    static func make(
        candidate: SSMRoutePromotionCandidate,
        productionSequenceLengths: [Int],
        rows: [SSMResultRow]
    ) -> SSMRoutePromotionRow {
        var bestVariants: [String] = []
        var speedupPercents: [Double] = []

        for sequenceLength in productionSequenceLengths {
            let sequenceRows = rows.filter { $0.sequenceLength == sequenceLength }
            let baseRows = sequenceRows.filter { $0.variant.hasPrefix("base_") }
            let candidateRows = sequenceRows.filter { $0.variant.hasPrefix(candidate.variantPrefix) }
            guard let bestBase = baseRows.min(by: averageSort),
                  let bestCandidate = candidateRows.min(by: averageSort) else {
                bestVariants.append("missing")
                speedupPercents.append(-Double.infinity)
                continue
            }
            bestVariants.append(bestCandidate.variant)
            let speedup = (bestBase.averageGpuMicroseconds - bestCandidate.averageGpuMicroseconds)
                / bestBase.averageGpuMicroseconds * 100.0
            speedupPercents.append(speedup)
        }

        let passingCount = speedupPercents.filter {
            $0 >= SSMSummaryFactory.promotionSpeedupThresholdPercent
        }.count
        let admission = passingCount == productionSequenceLengths.count
            ? candidate.routePromotionAdmission
            : "reject-cross-sequence-threshold"
        let failingSequenceLengths = zip(productionSequenceLengths, speedupPercents).compactMap { sequenceLength, speedup in
            speedup >= SSMSummaryFactory.promotionSpeedupThresholdPercent ? nil : sequenceLength
        }

        return SSMRoutePromotionRow(
            candidate: candidate,
            productionSequenceLengths: productionSequenceLengths,
            bestVariants: bestVariants,
            speedupPercents: speedupPercents,
            passingSequenceCount: passingCount,
            requiredSequenceCount: productionSequenceLengths.count,
            failingSequenceLengths: failingSequenceLengths,
            routePromotionAdmission: admission
        )
    }

    private static func averageSort(_ lhs: SSMResultRow, _ rhs: SSMResultRow) -> Bool {
        if lhs.averageGpuMicroseconds != rhs.averageGpuMicroseconds {
            return lhs.averageGpuMicroseconds < rhs.averageGpuMicroseconds
        }
        if lhs.requestedThreadgroupWidth != rhs.requestedThreadgroupWidth {
            return lhs.requestedThreadgroupWidth < rhs.requestedThreadgroupWidth
        }
        return lhs.variant < rhs.variant
    }
}

private struct SSMRecurrenceMicrobenchmarkHarness {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipelines: [String: MTLComputePipelineState]

    private let headCount = 16
    private let groupCount = 16
    private let keyDimension = 128
    private let valueDimension = 128
    private let convKernelSize = 4

    private var keyGroupDimension: Int { groupCount * keyDimension }
    private var convDimension: Int { 2 * keyGroupDimension + headCount * valueDimension }
    private var outputDimension: Int { headCount * valueDimension }
    private var activationRowStride: Int { max(convDimension, outputDimension, headCount) }

    init(device: MTLDevice) throws {
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create SSM microbenchmark command queue")
        }
        self.device = device
        self.queue = queue

        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateSSMWeightIndependentHelpers(),
            MetalSourceGenerator.generateSSMConvSiluHelper(weightFormat: .bfloat16),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: "bench_ssm_recurrence_seq_bf16_f32",
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: 2 * 16 * 128 + 16 * 128,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: 16,
                groupCount: 16,
                keyHeadDimension: 128,
                valueHeadDimension: 128
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: "bench_ssm_recurrence_seq_bf16_f32_shared_rms",
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: 2 * 16 * 128 + 16 * 128,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: 16,
                groupCount: 16,
                keyHeadDimension: 128,
                valueHeadDimension: 128,
                shareRMSScale: true
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: "bench_ssm_recurrence_seq_bf16_f32_prewrite_decay",
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: 2 * 16 * 128 + 16 * 128,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: 16,
                groupCount: 16,
                keyHeadDimension: 128,
                valueHeadDimension: 128,
                prewriteDecayedState: true
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: "bench_ssm_recurrence_seq_bf16_f32_qkpar",
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: 2 * 16 * 128 + 16 * 128,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: 16,
                groupCount: 16,
                keyHeadDimension: 128,
                valueHeadDimension: 128,
                parallelQKReduction: true
            ),
            Self.generatePhaseConvSiluKernel(),
            Self.generatePhaseStateRecurrenceKernel(),
            Self.generatePhaseStateRecurrenceKernel(
                name: "bench_ssm_phase_state_recurrence_d2_f32",
                valueLanesPerThread: 2
            ),
            Self.generatePhaseStateRecurrenceKernel(
                name: "bench_ssm_phase_state_recurrence_qkpar_f32",
                parallelQKReduction: true
            ),
            Self.generatePhaseStateRecurrenceKernel(
                name: "bench_ssm_phase_state_recurrence_cache32_f32",
                stateCacheTileWidth: 32
            ),
            Self.generatePhaseRMSGateKernel(),
        ].joined(separator: "\n")
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let names = [
            "bench_ssm_recurrence_seq_bf16_f32",
            "bench_ssm_recurrence_seq_bf16_f32_shared_rms",
            "bench_ssm_recurrence_seq_bf16_f32_prewrite_decay",
            "bench_ssm_recurrence_seq_bf16_f32_qkpar",
            "bench_ssm_phase_conv_silu_bf16_f32",
            "bench_ssm_phase_state_recurrence_f32",
            "bench_ssm_phase_state_recurrence_d2_f32",
            "bench_ssm_phase_state_recurrence_qkpar_f32",
            "bench_ssm_phase_state_recurrence_cache32_f32",
            "bench_ssm_phase_rms_gate_f32",
        ]
        var compiled: [String: MTLComputePipelineState] = [:]
        for name in names {
            guard let function = library.makeFunction(name: name) else {
                throw MetalCompilerError.kernelNotFound(name)
            }
            compiled[name] = try device.makeComputePipelineState(function: function)
        }
        self.pipelines = compiled
    }

    func measure(
        variant: SSMVariant,
        sequenceLength: Int,
        iterations: Int,
        warmupIterations: Int
    ) throws -> SSMResultRow {
        guard let pipeline = pipelines[variant.kernelName] else {
            throw MetalCompilerError.kernelNotFound(variant.kernelName)
        }
        let inputs = try makeInputs(sequenceLength: sequenceLength)
        let recurrentState = try makeZeroedSharedBuffer(
            byteLength: headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride
        )
        let convState = try makeZeroedSharedBuffer(
            byteLength: convKernelSize * convDimension * MemoryLayout<BFloat16>.stride
        )
        let output = try makeZeroedSharedBuffer(
            byteLength: sequenceLength * activationRowStride * MemoryLayout<Float>.stride
        )
        let geometry = dispatchGeometry(pipeline: pipeline, requestedThreadgroupWidth: variant.threadgroupWidth)

        for _ in 0..<warmupIterations {
            reset(recurrentState: recurrentState, convState: convState, output: output, sequenceLength: sequenceLength)
            _ = try execute(
                pipeline: pipeline,
                inputs: inputs,
                recurrentState: recurrentState,
                convState: convState,
                output: output,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        var totalMicroseconds = 0.0
        for _ in 0..<iterations {
            reset(recurrentState: recurrentState, convState: convState, output: output, sequenceLength: sequenceLength)
            totalMicroseconds += try execute(
                pipeline: pipeline,
                inputs: inputs,
                recurrentState: recurrentState,
                convState: convState,
                output: output,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        return SSMResultRow(
            sequenceLength: sequenceLength,
            variant: variant.name,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            gridWidth: geometry.grid.width,
            gridHeight: geometry.grid.height,
            threadgroupWidth: geometry.threadgroup.width,
            requestedThreadgroupWidth: variant.threadgroupWidth,
            averageGpuMicroseconds: totalMicroseconds / Double(iterations)
        )
    }

    func measurePhase(
        phase: SSMPhaseVariant,
        sequenceLength: Int,
        fullBaseAverageGpuMicroseconds: Double,
        iterations: Int,
        warmupIterations: Int
    ) throws -> SSMPhaseResultRow {
        guard let pipeline = pipelines[phase.kernelName] else {
            throw MetalCompilerError.kernelNotFound(phase.kernelName)
        }
        let inputs = try makeInputs(sequenceLength: sequenceLength)
        let recurrentState = try makeZeroedSharedBuffer(
            byteLength: headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride
        )
        let convState = try makeZeroedSharedBuffer(
            byteLength: convKernelSize * convDimension * MemoryLayout<BFloat16>.stride
        )
        let output = try makeZeroedSharedBuffer(
            byteLength: sequenceLength * activationRowStride * MemoryLayout<Float>.stride
        )
        let geometry = dispatchGeometry(pipeline: pipeline, requestedThreadgroupWidth: phase.threadgroupWidth)

        for _ in 0..<warmupIterations {
            reset(recurrentState: recurrentState, convState: convState, output: output, sequenceLength: sequenceLength)
            _ = try execute(
                pipeline: pipeline,
                inputs: inputs,
                recurrentState: recurrentState,
                convState: convState,
                output: output,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }

        var totalMicroseconds = 0.0
        for _ in 0..<iterations {
            reset(recurrentState: recurrentState, convState: convState, output: output, sequenceLength: sequenceLength)
            totalMicroseconds += try execute(
                pipeline: pipeline,
                inputs: inputs,
                recurrentState: recurrentState,
                convState: convState,
                output: output,
                sequenceLength: sequenceLength,
                geometry: geometry
            )
        }
        let outputChecksum = checksum(output: output, sequenceLength: sequenceLength)

        return SSMPhaseResultRow(
            sequenceLength: sequenceLength,
            phase: phase.name,
            headCount: headCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            gridWidth: geometry.grid.width,
            gridHeight: geometry.grid.height,
            threadgroupWidth: geometry.threadgroup.width,
            requestedThreadgroupWidth: phase.threadgroupWidth,
            averageGpuMicroseconds: totalMicroseconds / Double(iterations),
            fullBaseAverageGpuMicroseconds: fullBaseAverageGpuMicroseconds,
            outputChecksum: outputChecksum
        )
    }

    func validateStateRecurrencePhase(kernelName: String, sequenceLength: Int) throws -> SSMStatePhaseValidation {
        guard let pipeline = pipelines[kernelName] else {
            throw MetalCompilerError.kernelNotFound(kernelName)
        }
        let inputs = try makeInputs(sequenceLength: sequenceLength)
        let recurrentState = try makeZeroedSharedBuffer(
            byteLength: headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride
        )
        let convState = try makeZeroedSharedBuffer(
            byteLength: convKernelSize * convDimension * MemoryLayout<BFloat16>.stride
        )
        let output = try makeZeroedSharedBuffer(
            byteLength: sequenceLength * activationRowStride * MemoryLayout<Float>.stride
        )
        let geometry = dispatchGeometry(pipeline: pipeline, requestedThreadgroupWidth: 384)
        reset(recurrentState: recurrentState, convState: convState, output: output, sequenceLength: sequenceLength)
        _ = try execute(
            pipeline: pipeline,
            inputs: inputs,
            recurrentState: recurrentState,
            convState: convState,
            output: output,
            sequenceLength: sequenceLength,
            geometry: geometry
        )

        let actualOutput = readFloatBuffer(output, count: sequenceLength * activationRowStride)
        let actualState = readFloatBuffer(recurrentState, count: headCount * keyDimension * valueDimension)
        let reference = cpuStateRecurrenceReference(inputs: inputs, sequenceLength: sequenceLength)
        return SSMStatePhaseValidation(
            outputMaxError: maxAbsoluteError(actualOutput, reference.output),
            recurrentStateMaxError: maxAbsoluteError(actualState, reference.recurrentState)
        )
    }

    private static func generatePhaseConvSiluKernel() -> String {
        """
        kernel void bench_ssm_phase_conv_silu_bf16_f32(
            device const float* projectedQKV [[buffer(0)]],
            device const float* projectedZ [[buffer(1)]],
            device const float* projectedBeta [[buffer(2)]],
            device const float* projectedAlpha [[buffer(3)]],
            device const uint16_t* convWeight [[buffer(4)]],
            device const float* normWeight [[buffer(5)]],
            device const uint16_t* dtBias [[buffer(6)]],
            device const float* aLog [[buffer(7)]],
            device float* recurrentState [[buffer(8)]],
            device uint16_t* convState [[buffer(9)]],
            device float* output [[buffer(10)]],
            constant uint& numHeads [[buffer(11)]],
            constant uint& groupCount [[buffer(12)]],
            constant uint& keyDimension [[buffer(13)]],
            constant uint& valueDimension [[buffer(14)]],
            constant uint& convKernelSize [[buffer(15)]],
            constant uint& sequenceLength [[buffer(16)]],
            constant uint& activationRowStride [[buffer(17)]],
            device float* debugConvSilu [[buffer(18)]],
            constant uint& debugConvRowStride [[buffer(19)]],
            constant uint& debugConvEnabled [[buffer(20)]],
            uint tid [[thread_index_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]],
            uint tgid [[threadgroup_position_in_grid]]
        ) {
            const uint dk = keyDimension;
            const uint dv = valueDimension;
            const uint safeGroupCount = max(groupCount, 1u);
            const uint headsPerGroup = max(1u, numHeads / safeGroupCount);
            const uint keyGroupDim = safeGroupCount * dk;
            const uint convDim = 2u * keyGroupDim + numHeads * dv;
            const uint localDim = 2u * dk + headsPerGroup * dv;
            const uint groupIndex = tgid;
            if (groupIndex >= safeGroupCount) { return; }
            const uint headStart = groupIndex * headsPerGroup;
            const uint qBaseGlobal = groupIndex * dk;
            const uint kBaseGlobal = keyGroupDim + groupIndex * dk;
            const uint vBaseGlobal = 2u * keyGroupDim + headStart * dv;

            for (uint pos = 0; pos < sequenceLength; ++pos) {
                device const float* projectedQKVPos = projectedQKV + pos * activationRowStride;
                device float* outputPos = output + pos * activationRowStride;
                for (uint localCh = tid; localCh < localDim; localCh += tgSize) {
                    uint globalCh;
                    if (localCh < dk) {
                        globalCh = qBaseGlobal + localCh;
                    } else if (localCh < 2u * dk) {
                        globalCh = kBaseGlobal + (localCh - dk);
                    } else {
                        globalCh = vBaseGlobal + (localCh - 2u * dk);
                    }
                    float sum = 0.0f;
                    for (uint k = 0; k + 1 < convKernelSize; ++k) {
                        float val = bf16_to_float(convState[(k + 1) * convDim + globalCh]);
                        convState[k * convDim + globalCh] = float_to_bf16(val);
                        sum += val * bf16_to_float(convWeight[globalCh * convKernelSize + k]);
                    }
                    float newVal = projectedQKVPos[globalCh];
                    convState[(convKernelSize - 1) * convDim + globalCh] = float_to_bf16(newVal);
                    sum += newVal * bf16_to_float(convWeight[globalCh * convKernelSize + convKernelSize - 1]);
                    outputPos[globalCh] = sum * stable_sigmoid(sum);
                }
                if (pos + 1 < sequenceLength) {
                    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                }
            }
        }
        """
    }

    private static func generatePhaseStateRecurrenceKernel(
        name: String = "bench_ssm_phase_state_recurrence_f32",
        valueLanesPerThread: Int = 1,
        parallelQKReduction: Bool = false,
        stateCacheTileWidth: Int? = nil
    ) -> String {
        if let stateCacheTileWidth {
            return generatePhaseStateRecurrenceCachedKernel(name: name, stateCacheTileWidth: stateCacheTileWidth)
        }
        let valuesPerThread = max(1, valueLanesPerThread)
        let threadPlan = valuesPerThread == 1
            ? "const uint threadsPerHead = min(tgSize / max(headsPerGroup, 1u), dv);"
            : """
                const uint valuesPerThread = \(valuesPerThread)u;
                const uint valueThreadCount = (dv + valuesPerThread - 1u) / valuesPerThread;
                const uint threadsPerHead = min(tgSize / max(headsPerGroup, 1u), valueThreadCount);
            """
        let valueRange = valuesPerThread == 1
            ? """
                    const uint dChunk = dv / threadsPerHead;
                    const uint dStart = localTid * dChunk;
                    const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;
            """
            : """
                    const uint dStart = localTid * valuesPerThread;
                    const uint dEnd = min(dStart + valuesPerThread, dv);
            """
        let qkPartialStorage = parallelQKReduction
            ? """
            threadgroup float qNormSqPartials[128];
            threadgroup float kNormSqPartials[128];
            threadgroup float kqSumPartials[128];
            """
            : ""
        let qkReductionPhase = parallelQKReduction
            ? """
                if (tid < activeThreads) {
                    const uint localTid = tid % threadsPerHead;
                    if (localTid < dk) {
                        float q = convSiluCache[localTid];
                        float k = convSiluCache[dk + localTid];
                        qNormSqPartials[localTid] = q * q;
                        kNormSqPartials[localTid] = k * k;
                        kqSumPartials[localTid] = q * k;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < activeThreads) {
                    const uint localTid = tid % threadsPerHead;
                    if (localTid == 0) {
                        float qNormSq = 0.0f;
                        float kNormSq = 0.0f;
                        float kqSum = 0.0f;
                        for (uint j = 0; j < dk; ++j) {
                            qNormSq += qNormSqPartials[j];
                            kNormSq += kNormSqPartials[j];
                            kqSum += kqSumPartials[j];
                        }
                        qInvCache[0] = rsqrt(qNormSq + 1e-6f) * rsqrt(float(dk));
                        kInvCache[0] = rsqrt(kNormSq + 1e-6f);
                        kqSumCache[0] = kqSum;
                    }
                }
            """
            : """
                if (tid < activeThreads) {
                    const uint localTid = tid % threadsPerHead;
                    if (localTid == 0) {
                        float qNormSq = 0.0f;
                        float kNormSq = 0.0f;
                        float kqSum = 0.0f;
                        for (uint j = 0; j < dk; ++j) {
                            float q = convSiluCache[j];
                            float k = convSiluCache[dk + j];
                            qNormSq += q * q;
                            kNormSq += k * k;
                            kqSum += q * k;
                        }
                        qInvCache[0] = rsqrt(qNormSq + 1e-6f) * rsqrt(float(dk));
                        kInvCache[0] = rsqrt(kNormSq + 1e-6f);
                        kqSumCache[0] = kqSum;
                    }
                }
            """
        return """
        kernel void \(name)(
            device const float* convSilu [[buffer(0)]],
            device const float* projectedZ [[buffer(1)]],
            device const float* projectedBeta [[buffer(2)]],
            device const float* projectedAlpha [[buffer(3)]],
            device const uint16_t* convWeight [[buffer(4)]],
            device const float* normWeight [[buffer(5)]],
            device const uint16_t* dtBias [[buffer(6)]],
            device const float* aLog [[buffer(7)]],
            device float* recurrentState [[buffer(8)]],
            device uint16_t* convState [[buffer(9)]],
            device float* output [[buffer(10)]],
            constant uint& numHeads [[buffer(11)]],
            constant uint& groupCount [[buffer(12)]],
            constant uint& keyDimension [[buffer(13)]],
            constant uint& valueDimension [[buffer(14)]],
            constant uint& convKernelSize [[buffer(15)]],
            constant uint& sequenceLength [[buffer(16)]],
            constant uint& activationRowStride [[buffer(17)]],
            device float* debugConvSilu [[buffer(18)]],
            constant uint& debugConvRowStride [[buffer(19)]],
            constant uint& debugConvEnabled [[buffer(20)]],
            uint tid [[thread_index_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]],
            uint tgid [[threadgroup_position_in_grid]]
        ) {
            const uint dk = keyDimension;
            const uint dv = valueDimension;
            const uint safeGroupCount = max(groupCount, 1u);
            const uint headsPerGroup = max(1u, numHeads / safeGroupCount);
            const uint keyGroupDim = safeGroupCount * dk;
            const uint localDim = 2u * dk + headsPerGroup * dv;
            const uint groupIndex = tgid;
            if (groupIndex >= safeGroupCount) { return; }
            const uint headStart = groupIndex * headsPerGroup;

            threadgroup float convSiluCache[384];
            threadgroup float qInvCache[1];
            threadgroup float kInvCache[1];
            threadgroup float kqSumCache[1];
            \(qkPartialStorage)

            for (uint pos = 0; pos < sequenceLength; ++pos) {
                device const float* convSiluPos = convSilu + pos * activationRowStride;
                device const float* betaPos = projectedBeta + pos * activationRowStride;
                device const float* alphaPos = projectedAlpha + pos * activationRowStride;
                device float* outputPos = output + pos * activationRowStride;

                for (uint localCh = tid; localCh < localDim; localCh += tgSize) {
                    uint globalCh;
                    if (localCh < dk) {
                        globalCh = groupIndex * dk + localCh;
                    } else if (localCh < 2u * dk) {
                        globalCh = keyGroupDim + groupIndex * dk + (localCh - dk);
                    } else {
                        globalCh = 2u * keyGroupDim + headStart * dv + (localCh - 2u * dk);
                    }
                    convSiluCache[localCh] = convSiluPos[globalCh];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                \(threadPlan)
                const uint activeThreads = headsPerGroup * threadsPerHead;
                \(qkReductionPhase)
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid < activeThreads) {
                    const uint headIndex = headStart;
                    const uint localTid = tid % threadsPerHead;
                    \(valueRange)
                    float alpha = alphaPos[headIndex];
                    float betaInput = betaPos[headIndex];
                    float decay = exp(-exp(aLog[headIndex]) * stable_softplus(alpha + bf16_to_float(dtBias[headIndex])));
                    float beta = stable_sigmoid(betaInput);
                    device float* state = recurrentState + headIndex * dk * dv;
                    float qInv = qInvCache[0];
                    float kInv = kInvCache[0];
                    float kqSum = kqSumCache[0];
                    for (uint d = dStart; d < dEnd; ++d) {
                        float kvmemRaw = 0.0f;
                        float sqSum = 0.0f;
                        for (uint j = 0; j < dk; ++j) {
                            float s = state[j * dv + d] * decay;
                            kvmemRaw += s * convSiluCache[dk + j];
                            sqSum += s * convSiluCache[j];
                        }
                        float delta = beta * (convSiluCache[2u * dk + d] - kvmemRaw * kInv);
                        float kInvDelta = kInv * delta;
                        float dot = (sqSum + kInvDelta * kqSum) * qInv;
                        for (uint j = 0; j < dk; ++j) {
                            state[j * dv + d] = state[j * dv + d] * decay + convSiluCache[dk + j] * kInvDelta;
                        }
                        outputPos[headIndex * dv + d] = dot;
                    }
                }
                if (pos + 1 < sequenceLength) {
                    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                }
            }
        }
        """
    }

    private static func generatePhaseStateRecurrenceCachedKernel(
        name: String,
        stateCacheTileWidth: Int
    ) -> String {
        let tileWidth = max(1, stateCacheTileWidth)
        return """
        kernel void \(name)(
            device const float* convSilu [[buffer(0)]],
            device const float* projectedZ [[buffer(1)]],
            device const float* projectedBeta [[buffer(2)]],
            device const float* projectedAlpha [[buffer(3)]],
            device const uint16_t* convWeight [[buffer(4)]],
            device const float* normWeight [[buffer(5)]],
            device const uint16_t* dtBias [[buffer(6)]],
            device const float* aLog [[buffer(7)]],
            device float* recurrentState [[buffer(8)]],
            device uint16_t* convState [[buffer(9)]],
            device float* output [[buffer(10)]],
            constant uint& numHeads [[buffer(11)]],
            constant uint& groupCount [[buffer(12)]],
            constant uint& keyDimension [[buffer(13)]],
            constant uint& valueDimension [[buffer(14)]],
            constant uint& convKernelSize [[buffer(15)]],
            constant uint& sequenceLength [[buffer(16)]],
            constant uint& activationRowStride [[buffer(17)]],
            device float* debugConvSilu [[buffer(18)]],
            constant uint& debugConvRowStride [[buffer(19)]],
            constant uint& debugConvEnabled [[buffer(20)]],
            uint tid [[thread_index_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]],
            uint tgid [[threadgroup_position_in_grid]]
        ) {
            const uint dk = keyDimension;
            const uint dv = valueDimension;
            const uint safeGroupCount = max(groupCount, 1u);
            const uint headsPerGroup = max(1u, numHeads / safeGroupCount);
            const uint keyGroupDim = safeGroupCount * dk;
            const uint localDim = 2u * dk + headsPerGroup * dv;
            const uint groupIndex = tgid;
            if (groupIndex >= safeGroupCount) { return; }
            const uint headStart = groupIndex * headsPerGroup;

            threadgroup float convSiluCache[384];
            threadgroup float qInvCache[1];
            threadgroup float kInvCache[1];
            threadgroup float kqSumCache[1];
            threadgroup float decayedStateCache[128 * \(tileWidth)];

            for (uint pos = 0; pos < sequenceLength; ++pos) {
                device const float* convSiluPos = convSilu + pos * activationRowStride;
                device const float* betaPos = projectedBeta + pos * activationRowStride;
                device const float* alphaPos = projectedAlpha + pos * activationRowStride;
                device float* outputPos = output + pos * activationRowStride;

                for (uint localCh = tid; localCh < localDim; localCh += tgSize) {
                    uint globalCh;
                    if (localCh < dk) {
                        globalCh = groupIndex * dk + localCh;
                    } else if (localCh < 2u * dk) {
                        globalCh = keyGroupDim + groupIndex * dk + (localCh - dk);
                    } else {
                        globalCh = 2u * keyGroupDim + headStart * dv + (localCh - 2u * dk);
                    }
                    convSiluCache[localCh] = convSiluPos[globalCh];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileWidth = \(tileWidth)u;
                const uint threadsPerHead = min(tgSize / max(headsPerGroup, 1u), tileWidth);
                const uint activeThreads = headsPerGroup * threadsPerHead;
                if (tid < activeThreads) {
                    const uint localTid = tid % threadsPerHead;
                    if (localTid == 0) {
                        float qNormSq = 0.0f;
                        float kNormSq = 0.0f;
                        float kqSum = 0.0f;
                        for (uint j = 0; j < dk; ++j) {
                            float q = convSiluCache[j];
                            float k = convSiluCache[dk + j];
                            qNormSq += q * q;
                            kNormSq += k * k;
                            kqSum += q * k;
                        }
                        qInvCache[0] = rsqrt(qNormSq + 1e-6f) * rsqrt(float(dk));
                        kInvCache[0] = rsqrt(kNormSq + 1e-6f);
                        kqSumCache[0] = kqSum;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint tileBase = 0; tileBase < dv; tileBase += tileWidth) {
                    if (tid < activeThreads) {
                        const uint headIndex = headStart;
                        const uint localTid = tid % threadsPerHead;
                        const uint d = tileBase + localTid;
                        if (d < dv) {
                            float alpha = alphaPos[headIndex];
                            float betaInput = betaPos[headIndex];
                            float decay = exp(-exp(aLog[headIndex]) * stable_softplus(alpha + bf16_to_float(dtBias[headIndex])));
                            float beta = stable_sigmoid(betaInput);
                            device float* state = recurrentState + headIndex * dk * dv;
                            float qInv = qInvCache[0];
                            float kInv = kInvCache[0];
                            float kqSum = kqSumCache[0];
                            float kvmemRaw = 0.0f;
                            float sqSum = 0.0f;
                            for (uint j = 0; j < dk; ++j) {
                                float s = state[j * dv + d] * decay;
                                decayedStateCache[j * tileWidth + localTid] = s;
                                kvmemRaw += s * convSiluCache[dk + j];
                                sqSum += s * convSiluCache[j];
                            }
                            float delta = beta * (convSiluCache[2u * dk + d] - kvmemRaw * kInv);
                            float kInvDelta = kInv * delta;
                            float dot = (sqSum + kInvDelta * kqSum) * qInv;
                            for (uint j = 0; j < dk; ++j) {
                                state[j * dv + d] = decayedStateCache[j * tileWidth + localTid]
                                    + convSiluCache[dk + j] * kInvDelta;
                            }
                            outputPos[headIndex * dv + d] = dot;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (pos + 1 < sequenceLength) {
                    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                }
            }
        }
        """
    }

    private static func generatePhaseRMSGateKernel() -> String {
        """
        kernel void bench_ssm_phase_rms_gate_f32(
            device const float* dotInput [[buffer(0)]],
            device const float* projectedZ [[buffer(1)]],
            device const float* projectedBeta [[buffer(2)]],
            device const float* projectedAlpha [[buffer(3)]],
            device const uint16_t* convWeight [[buffer(4)]],
            device const float* normWeight [[buffer(5)]],
            device const uint16_t* dtBias [[buffer(6)]],
            device const float* aLog [[buffer(7)]],
            device float* recurrentState [[buffer(8)]],
            device uint16_t* convState [[buffer(9)]],
            device float* output [[buffer(10)]],
            constant uint& numHeads [[buffer(11)]],
            constant uint& groupCount [[buffer(12)]],
            constant uint& keyDimension [[buffer(13)]],
            constant uint& valueDimension [[buffer(14)]],
            constant uint& convKernelSize [[buffer(15)]],
            constant uint& sequenceLength [[buffer(16)]],
            constant uint& activationRowStride [[buffer(17)]],
            device float* debugConvSilu [[buffer(18)]],
            constant uint& debugConvRowStride [[buffer(19)]],
            constant uint& debugConvEnabled [[buffer(20)]],
            uint tid [[thread_index_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]],
            uint tgid [[threadgroup_position_in_grid]]
        ) {
            const uint dv = valueDimension;
            const uint safeGroupCount = max(groupCount, 1u);
            const uint headsPerGroup = max(1u, numHeads / safeGroupCount);
            const uint groupIndex = tgid;
            if (groupIndex >= safeGroupCount) { return; }
            const uint headStart = groupIndex * headsPerGroup;
            const uint threadsPerHead = min(tgSize / max(headsPerGroup, 1u), dv);
            const uint activeThreads = headsPerGroup * threadsPerHead;
            threadgroup float normPartials[384];
            threadgroup float rmsScaleCache[1];

            for (uint pos = 0; pos < sequenceLength; ++pos) {
                device const float* dotPos = dotInput + pos * activationRowStride;
                device const float* zPos = projectedZ + pos * activationRowStride;
                device float* outputPos = output + pos * activationRowStride;
                if (tid < activeThreads) {
                    const uint localTid = tid % threadsPerHead;
                    const uint headIndex = headStart;
                    const uint dChunk = dv / threadsPerHead;
                    const uint dStart = localTid * dChunk;
                    const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;
                    float localNormSq = 0.0f;
                    for (uint d = dStart; d < dEnd; ++d) {
                        float value = dotPos[headIndex * dv + d];
                        localNormSq += value * value;
                    }
                    normPartials[localTid] = localNormSq;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < activeThreads) {
                    const uint localTid = tid % threadsPerHead;
                    if (localTid == 0) {
                        float totalNormSq = 0.0f;
                        for (uint t = 0; t < threadsPerHead; ++t) {
                            totalNormSq += normPartials[t];
                        }
                        rmsScaleCache[0] = rsqrt(totalNormSq / float(dv) + 1e-6f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < activeThreads) {
                    const uint localTid = tid % threadsPerHead;
                    const uint headIndex = headStart;
                    const uint dChunk = dv / threadsPerHead;
                    const uint dStart = localTid * dChunk;
                    const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;
                    float rmsScale = rmsScaleCache[0];
                    for (uint d = dStart; d < dEnd; ++d) {
                        float normed = dotPos[headIndex * dv + d] * rmsScale * normWeight[d];
                        float z = zPos[headIndex * dv + d];
                        outputPos[headIndex * dv + d] = normed * z * stable_sigmoid(z);
                    }
                }
                if (pos + 1 < sequenceLength) {
                    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                }
            }
        }
        """
    }

    private func makeInputs(sequenceLength: Int) throws -> SSMInputs {
        let qkvValues = paddedRows(
            makeFloatValues(count: sequenceLength * convDimension, multiplier: 13, modulus: 23, scale: 0.125),
            rowCount: sequenceLength,
            logicalWidth: convDimension
        )
        let zValues = paddedRows(
            makeFloatValues(count: sequenceLength * outputDimension, multiplier: 17, modulus: 19, scale: 0.125),
            rowCount: sequenceLength,
            logicalWidth: outputDimension
        )
        let betaValues = paddedRows(
            makeFloatValues(count: sequenceLength * headCount, multiplier: 7, modulus: 11, scale: 0.125),
            rowCount: sequenceLength,
            logicalWidth: headCount
        )
        let alphaValues = paddedRows(
            makeFloatValues(count: sequenceLength * headCount, multiplier: 5, modulus: 13, scale: 0.125),
            rowCount: sequenceLength,
            logicalWidth: headCount
        )
        let convWeightValues = (0..<(convDimension * convKernelSize)).map { index in
            BFloat16(Float((index * 11) % 17 - 8) * 0.03125)
        }
        let normWeightValues = (0..<valueDimension).map { index in
            0.75 + Float(index) * 0.0625
        }
        let dtBiasValues = (0..<headCount).map { index in
            BFloat16(Float(index - 1) * 0.03125)
        }
        let aLogValues = (0..<headCount).map { index in
            Float(index) * 0.0625 - 0.125
        }
        return SSMInputs(
            qkv: try makeSharedBuffer(values: qkvValues),
            z: try makeSharedBuffer(values: zValues),
            beta: try makeSharedBuffer(values: betaValues),
            alpha: try makeSharedBuffer(values: alphaValues),
            convWeight: try makeSharedBuffer(values: convWeightValues),
            normWeight: try makeSharedBuffer(values: normWeightValues),
            dtBias: try makeSharedBuffer(values: dtBiasValues),
            aLog: try makeSharedBuffer(values: aLogValues),
            qkvValues: qkvValues,
            betaValues: betaValues,
            alphaValues: alphaValues,
            dtBiasValues: dtBiasValues,
            aLogValues: aLogValues
        )
    }

    private func execute(
        pipeline: MTLComputePipelineState,
        inputs: SSMInputs,
        recurrentState: MTLBuffer,
        convState: MTLBuffer,
        output: MTLBuffer,
        sequenceLength: Int,
        geometry: (grid: MTLSize, threadgroup: MTLSize)
    ) throws -> Double {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create SSM microbenchmark command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs.qkv, offset: 0, index: 0)
        encoder.setBuffer(inputs.z, offset: 0, index: 1)
        encoder.setBuffer(inputs.beta, offset: 0, index: 2)
        encoder.setBuffer(inputs.alpha, offset: 0, index: 3)
        encoder.setBuffer(inputs.convWeight, offset: 0, index: 4)
        encoder.setBuffer(inputs.normWeight, offset: 0, index: 5)
        encoder.setBuffer(inputs.dtBias, offset: 0, index: 6)
        encoder.setBuffer(inputs.aLog, offset: 0, index: 7)
        encoder.setBuffer(recurrentState, offset: 0, index: 8)
        encoder.setBuffer(convState, offset: 0, index: 9)
        encoder.setBuffer(output, offset: 0, index: 10)
        encoder.setBuffer(output, offset: 0, index: 18)
        setConstants(encoder: encoder, sequenceLength: sequenceLength)
        encoder.dispatchThreadgroups(geometry.grid, threadsPerThreadgroup: geometry.threadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("SSM microbenchmark command failed: \(error.localizedDescription)")
        }
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000
    }

    private func setConstants(encoder: MTLComputeCommandEncoder, sequenceLength: Int) {
        var heads = UInt32(headCount)
        var groups = UInt32(groupCount)
        var keyDim = UInt32(keyDimension)
        var valueDim = UInt32(valueDimension)
        var kernel = UInt32(convKernelSize)
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(activationRowStride)
        var debugEnabled: UInt32 = 0
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&groups, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 15)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 17)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 19)
        encoder.setBytes(&debugEnabled, length: MemoryLayout<UInt32>.stride, index: 20)
    }

    private func reset(recurrentState: MTLBuffer, convState: MTLBuffer, output: MTLBuffer, sequenceLength: Int) {
        memset(recurrentState.contents(), 0, recurrentState.length)
        memset(convState.contents(), 0, convState.length)
        memset(output.contents(), 0, sequenceLength * activationRowStride * MemoryLayout<Float>.stride)
    }

    private func checksum(output: MTLBuffer, sequenceLength: Int) -> Double {
        let elementCount = sequenceLength * activationRowStride
        let values = output.contents().bindMemory(to: Float.self, capacity: elementCount)
        var checksum = 0.0
        for index in 0..<elementCount {
            let value = values[index]
            guard value.isFinite else {
                return .nan
            }
            checksum += Double(abs(value))
        }
        return checksum
    }

    private func readFloatBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let values = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { values[$0] }
    }

    private func cpuStateRecurrenceReference(
        inputs: SSMInputs,
        sequenceLength: Int
    ) -> (output: [Float], recurrentState: [Float]) {
        let headsPerGroup = max(1, headCount / max(groupCount, 1))
        var recurrentState = [Float](repeating: .zero, count: headCount * keyDimension * valueDimension)
        var output = [Float](repeating: .zero, count: sequenceLength * activationRowStride)

        for pos in 0..<sequenceLength {
            let rowBase = pos * activationRowStride
            for groupIndex in 0..<groupCount {
                let headStart = groupIndex * headsPerGroup
                let qBaseGlobal = groupIndex * keyDimension
                let kBaseGlobal = keyGroupDimension + groupIndex * keyDimension
                let vBaseGlobal = 2 * keyGroupDimension + headStart * valueDimension
                for localHead in 0..<headsPerGroup {
                    let headIndex = headStart + localHead
                    let qBase = rowBase + qBaseGlobal
                    let kBase = rowBase + kBaseGlobal
                    let vBase = rowBase + vBaseGlobal + localHead * valueDimension
                    var qNormSq: Float = 0
                    var kNormSq: Float = 0
                    var kqSum: Float = 0
                    for j in 0..<keyDimension {
                        let q = inputs.qkvValues[qBase + j]
                        let k = inputs.qkvValues[kBase + j]
                        qNormSq += q * q
                        kNormSq += k * k
                        kqSum += q * k
                    }
                    let qInv = rsqrt(qNormSq + 1e-6) * rsqrt(Float(keyDimension))
                    let kInv = rsqrt(kNormSq + 1e-6)
                    let alpha = inputs.alphaValues[rowBase + headIndex]
                    let betaInput = inputs.betaValues[rowBase + headIndex]
                    let decay = exp(-exp(inputs.aLogValues[headIndex]) * stableSoftplus(alpha + Float(inputs.dtBiasValues[headIndex])))
                    let beta = stableSigmoid(betaInput)
                    let stateHeadBase = headIndex * keyDimension * valueDimension
                    for d in 0..<valueDimension {
                        var kvmemRaw: Float = 0
                        var sqSum: Float = 0
                        for j in 0..<keyDimension {
                            let stateIndex = stateHeadBase + j * valueDimension + d
                            let s = recurrentState[stateIndex] * decay
                            kvmemRaw += s * inputs.qkvValues[kBase + j]
                            sqSum += s * inputs.qkvValues[qBase + j]
                        }
                        let delta = beta * (inputs.qkvValues[vBase + d] - kvmemRaw * kInv)
                        let kInvDelta = kInv * delta
                        let dot = (sqSum + kInvDelta * kqSum) * qInv
                        for j in 0..<keyDimension {
                            let stateIndex = stateHeadBase + j * valueDimension + d
                            recurrentState[stateIndex] = recurrentState[stateIndex] * decay
                                + inputs.qkvValues[kBase + j] * kInvDelta
                        }
                        output[rowBase + headIndex * valueDimension + d] = dot
                    }
                }
            }
        }
        return (output, recurrentState)
    }

    private func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        var result: Float = 0
        for index in lhs.indices {
            result = max(result, abs(lhs[index] - rhs[index]))
        }
        return result
    }

    private func stableSigmoid(_ value: Float) -> Float {
        1.0 / (1.0 + Float(Foundation.exp(Double(-value))))
    }

    private func stableSoftplus(_ value: Float) -> Float {
        max(value, 0) + Float(Foundation.log1p(Foundation.exp(Double(-abs(value)))))
    }

    private func rsqrt(_ value: Float) -> Float {
        1.0 / Foundation.sqrt(value)
    }

    private func dispatchGeometry(
        pipeline: MTLComputePipelineState,
        requestedThreadgroupWidth: Int
    ) -> (grid: MTLSize, threadgroup: MTLSize) {
        let safeGroupCount = max(groupCount, 1)
        let headsPerGroup = max(1, headCount / safeGroupCount)
        let localDimension = 2 * keyDimension + headsPerGroup * valueDimension
        let phase2Threads = headsPerGroup * min(valueDimension, 256)
        let desiredThreads = max(localDimension, phase2Threads)
        let defaultThreads = min(
            min(SSMRecurrenceFragment.maxThreadgroupSize, desiredThreads),
            pipeline.maxTotalThreadsPerThreadgroup
        )
        let threads = min(max(requestedThreadgroupWidth, 1), defaultThreads)
        return (
            MTLSize(width: safeGroupCount, height: 1, depth: 1),
            MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func paddedRows(_ values: [Float], rowCount: Int, logicalWidth: Int) -> [Float] {
        var padded = [Float](repeating: .zero, count: rowCount * activationRowStride)
        for row in 0..<rowCount {
            let sourceStart = row * logicalWidth
            let destinationStart = row * activationRowStride
            padded.replaceSubrange(
                destinationStart..<(destinationStart + logicalWidth),
                with: values[sourceStart..<(sourceStart + logicalWidth)]
            )
        }
        return padded
    }

    private func makeFloatValues(count: Int, multiplier: Int, modulus: Int, scale: Float) -> [Float] {
        (0..<count).map { index in
            Float(BFloat16(Float((index * multiplier) % modulus - modulus / 2) * scale))
        }
    }

    private func makeSharedBuffer<T>(values: [T]) throws -> MTLBuffer {
        guard !values.isEmpty else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create an empty SSM microbenchmark buffer")
        }
        var copy = values
        let byteLength = copy.count * MemoryLayout<T>.stride
        return try copy.withUnsafeMutableBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                throw MetalCompilerError.deviceSetupFailed("Cannot access SSM microbenchmark buffer bytes")
            }
            guard let buffer = device.makeBuffer(bytes: baseAddress, length: byteLength, options: .storageModeShared) else {
                throw MetalCompilerError.deviceSetupFailed("Cannot allocate SSM microbenchmark buffer")
            }
            return buffer
        }
    }

    private func makeZeroedSharedBuffer(byteLength: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate zeroed SSM microbenchmark buffer")
        }
        memset(buffer.contents(), 0, byteLength)
        return buffer
    }
}

private struct SSMInputs {
    let qkv: MTLBuffer
    let z: MTLBuffer
    let beta: MTLBuffer
    let alpha: MTLBuffer
    let convWeight: MTLBuffer
    let normWeight: MTLBuffer
    let dtBias: MTLBuffer
    let aLog: MTLBuffer
    let qkvValues: [Float]
    let betaValues: [Float]
    let alphaValues: [Float]
    let dtBiasValues: [BFloat16]
    let aLogValues: [Float]
}
