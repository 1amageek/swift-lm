import Foundation
import Metal
import Testing
@testable import MetalCompiler

/// Per-step GPU profiling for Qwen3.5-0.8B prefill.
///
/// Purpose: identify which prefill steps scale with sequence length. MLX keeps prefill
/// at a constant ~41-43 ms for sequence length 16→128, while swift-lm grows linearly
/// at ~2.2 ms/token. This test times each step individually to find the scaling culprit.
#if ENABLE_METAL_PROBES
@Suite("Qwen35 Prefill Profile", .serialized)
struct Qwen35PrefillProfileTests {

    static let sequenceLengths = [16, 64, 128]
    static let iterations = 5

    @Test("Per-step prefill timing at seqLen 16/64/128")
    func perStepPrefillTimingByLength() throws {
        guard let bundlePath = try resolveBundlePath() else {
            Issue.record("Qwen3.5-0.8B bundle not found.")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: bundlePath,
            maximumPrefillLength: 128
        )
        guard let plan = model.prefillPlan else {
            Issue.record("No prefill plan")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        // Structural summary
        print("=== Qwen3.5-0.8B Prefill Plan ===")
        print("total steps: \(plan.steps.count)")
        printStepSummary(plan: plan)

        // Clone plan for isolated profiling
        let isolatedPlan = try plan.makeRuntimeIsolatedCopy(device: device)

        let residency = try MetalResidencyLease.combined(
            label: "qwen35.profile",
            leases: [
                MetalResidencyLease.required(
                    device: device,
                    label: "qwen35.profile.runtime",
                    buffers: isolatedPlan.buffers.runtimeResidencyBuffers
                ),
                MetalResidencyLease.required(
                    device: device,
                    label: "qwen35.profile.weights",
                    buffers: isolatedPlan.buffers.weightResidencyBuffers
                ),
                MetalResidencyLease.required(
                    device: device,
                    label: "qwen35.profile.supplemental",
                    buffers: isolatedPlan.supplementalResidencyBuffers
                ),
            ]
        )
        var submission = try MetalSubmissionContext(device: device)

        let harness = MetalPrefillProfileHarness()
        let artifactDirectory = try artifactDirectory()

        // Profile at each sequence length and print category breakdown.
        var profilesByLength: [Int: [MetalPrefillProfile.Entry]] = [:]
        for seqLen in Self.sequenceLengths {
            let profile = try harness.profileSteps(
                plan: isolatedPlan,
                submission: &submission,
                sequenceLength: seqLen,
                iterations: Self.iterations,
                warmupIterations: 1,
                ephemeralResidency: residency
            )
            let stepArtifacts = try profile.writeArtifacts(
                directory: artifactDirectory,
                basename: "qwen35-prefill-steps-seq\(seqLen)"
            )
            let passProfile = try harness.profilePasses(
                plan: isolatedPlan,
                submission: &submission,
                sequenceLength: seqLen,
                iterations: max(1, min(Self.iterations, 3)),
                warmupIterations: 1,
                ephemeralResidency: residency
            )
            let passArtifacts = try passProfile.writeArtifacts(
                directory: artifactDirectory,
                basename: "qwen35-prefill-passes-seq\(seqLen)"
            )
            let profiles = profile.entries
            profilesByLength[seqLen] = profiles
            printCategoryBreakdown(profiles: profiles, iterations: Self.iterations, seqLen: seqLen)
            print("  artifacts: \(stepArtifacts.map(\.path).joined(separator: ", "))")
            print("  pass artifacts: \(passArtifacts.map(\.path).joined(separator: ", "))")
        }

        // Scaling report: for each step, show time at 16 / 64 / 128 and ratio 128/16
        printScalingReport(profilesByLength: profilesByLength, iterations: Self.iterations)

        #expect(!profilesByLength.isEmpty)
    }

    // MARK: - Bundle resolution

    private func resolveBundlePath() throws -> String? {
        if let override = ProcessInfo.processInfo.environment["SWIFTLM_QWEN35_BUNDLE"],
           !override.trimmingCharacters(in: .whitespaces).isEmpty {
            return NSString(string: override).expandingTildeInPath
        }
        let hubRoot = NSString(string: "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots").expandingTildeInPath
        guard FileManager.default.fileExists(atPath: hubRoot) else { return nil }
        let entries = try FileManager.default.contentsOfDirectory(atPath: hubRoot).sorted()
        for entry in entries {
            let candidate = "\(hubRoot)/\(entry)"
            let cfg = "\(candidate)/config.json"
            if FileManager.default.fileExists(atPath: cfg) {
                return candidate
            }
        }
        return nil
    }

    private func artifactDirectory() throws -> URL {
        let root = repositoryRoot()
            .appendingPathComponent(".test-artifacts/prefill-profile", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    // MARK: - Step summary

    private func printStepSummary(plan: MetalPrefillPlan) {
        var kindCounts: [String: Int] = [:]
        for step in plan.steps {
            let name = step.metadata.kernelName ?? step.pipeline.label ?? "(unknown)"
            kindCounts[name, default: 0] += 1
        }
        let sorted = kindCounts.sorted { $0.value > $1.value }
        for (name, count) in sorted {
            print("  \(count)× \(name)")
        }
    }

    // MARK: - Reporting

    private func printCategoryBreakdown(profiles: [MetalPrefillProfile.Entry], iterations: Int, seqLen: Int) {
        struct Entry { var steps: Int = 0; var totalMicros: Double = 0 }
        var byCategory: [String: Entry] = [:]
        var total: Double = 0
        for p in profiles {
            let avg = p.averageGpuMicroseconds
            var e = byCategory[p.category] ?? Entry()
            e.steps += 1
            e.totalMicros += avg
            byCategory[p.category] = e
            total += avg
        }
        print("--- seqLen=\(seqLen) total=\(String(format: "%.3f", total / 1000.0))ms steps=\(profiles.count) ---")
        for (cat, entry) in byCategory.sorted(by: { $0.value.totalMicros > $1.value.totalMicros }) {
            let ms = entry.totalMicros / 1000.0
            let pct = total > 0 ? entry.totalMicros / total * 100.0 : 0
            let catPadded = cat.padding(toLength: 16, withPad: " ", startingAt: 0)
            let msStr = String(format: "%.3f", ms)
            let pctStr = String(format: "%.1f", pct)
            print("  \(catPadded) steps=\(entry.steps)  \(msStr) ms (\(pctStr)%)")
        }
    }

    private func printScalingReport(profilesByLength: [Int: [MetalPrefillProfile.Entry]], iterations: Int) {
        print()
        print("=== Per-kernel scaling (seqLen 16 → 128) ===")
        // Aggregate by kernel name across all steps
        struct Row {
            let kernelName: String
            let category: String
            var times: [Int: Double] = [:]  // seqLen -> avg micros (sum across all steps)
            var counts: [Int: Int] = [:]    // seqLen -> step count
            var firstGrid: (Int, Int, Int)? = nil
        }
        var rows: [String: Row] = [:]
        for (seqLen, profiles) in profilesByLength {
            for p in profiles {
                var row = rows[p.kernelName] ?? Row(kernelName: p.kernelName, category: p.category)
                let avg = p.averageGpuMicroseconds
                row.times[seqLen, default: 0] += avg
                row.counts[seqLen, default: 0] += 1
                if row.firstGrid == nil {
                    row.firstGrid = (p.gridWidth, p.gridHeight, p.threadgroupWidth)
                }
                rows[p.kernelName] = row
            }
        }
        let lengths = Self.sequenceLengths
        // Header
        var header = "  kernel                                 cat             n "
        for l in lengths { header += "  \(l)us    " }
        header += "  128/16  grid(w×h,tg)"
        print(header)
        let sorted = rows.values.sorted { ($0.times[128] ?? 0) > ($1.times[128] ?? 0) }
        for row in sorted {
            let count = row.counts[lengths[0]] ?? 0
            let knamePadded = row.kernelName.prefix(38).padding(toLength: 38, withPad: " ", startingAt: 0)
            let catPadded = row.category.padding(toLength: 14, withPad: " ", startingAt: 0)
            var line = "  \(knamePadded) \(catPadded) \(count) "
            for l in lengths {
                let us = row.times[l] ?? 0
                line += String(format: " %7.0f", us)
            }
            let t16 = row.times[16] ?? 0
            let t128 = row.times[128] ?? 0
            let ratio = t16 > 0 ? t128 / t16 : 0
            line += String(format: "  %6.2fx", ratio)
            if let g = row.firstGrid {
                line += "  (\(g.0)×\(g.1), tg=\(g.2))"
            }
            print(line)
        }
    }
}
#endif
