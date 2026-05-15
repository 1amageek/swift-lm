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

    private struct RoutePromotionCandidate {
        let routeFamily: String
        let role: String
        let variant: String
        let microbenchAdmission: String
        let readinessPrerequisite: String
        let requiredProfileRouteGate: String
    }

    private struct ProfileRouteGate {
        let routeFamily: String
        let role: String
        let routeGate: String
    }

    private struct RouteReadinessRow {
        let routeFamily: String
        let role: String
        let variant: String
        let microbenchAdmission: String
        let readinessPrerequisite: String
        let requiredProfileRouteGate: String
        let observedProfileRouteGate: String?
        let routeReadiness: String
    }

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
            let routeManifestArtifact = try writeRouteManifest(
                profiles: profile.entries,
                sequenceLength: seqLen,
                directory: artifactDirectory
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
            assertDefaultProjectionRoutes(profiles: profiles, sequenceLength: seqLen)
            printCategoryBreakdown(profiles: profiles, iterations: Self.iterations, seqLen: seqLen)
            if seqLen == 128 {
                printSingleProjectionRoleBreakdown(profiles: profiles)
                let windows = RecurrentBlockFusionWindowScanner.linearAttentionWindows(in: profiles)
                printLinearAttentionWindowSummary(windows: windows)
                let partialProjectionCount = profiles.filter {
                    $0.kernelName == "recurrent_block_partial_projection_seq_bf16_f32"
                }.count
                if partialProjectionCount > 0 {
                    #expect(partialProjectionCount == 18)
                } else {
                    #expect(windows.count == 18)
                }
            }
            print("  artifacts: \(stepArtifacts.map(\.path).joined(separator: ", "))")
            print("  route manifest: \(routeManifestArtifact.path)")
            print("  pass artifacts: \(passArtifacts.map(\.path).joined(separator: ", "))")
        }

        // Scaling report: for each step, show time at 16 / 64 / 128 and ratio 128/16
        printScalingReport(profilesByLength: profilesByLength, iterations: Self.iterations)
        let routeGateArtifact = try writeRouteGate(
            profilesByLength: profilesByLength,
            directory: artifactDirectory
        )
        print("route gate: \(routeGateArtifact.path)")

        #expect(!profilesByLength.isEmpty)
    }

    @Test("Route manifest classifies projection routes")
    func routeManifestClassifiesProjectionRoutes() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-lm-qwen-route-manifest-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let url = try writeRouteManifest(
            profiles: [
                syntheticProfileEntry(
                    index: 0,
                    kernelName: "batched_gemv4_seq_bf16_f32s",
                    weightTensorName: [
                        "model.layers.0.linear_attn.in_proj_q.weight",
                        "model.layers.0.linear_attn.in_proj_k.weight",
                        "model.layers.0.linear_attn.in_proj_v.weight",
                    ].joined(separator: ";"),
                    averageGpuMicroseconds: 100
                ),
                syntheticProfileEntry(
                    index: 1,
                    kernelName: "batched_gemv2_seq_bf16_f32s",
                    weightTensorName: [
                        "model.layers.0.mlp.gate_proj.weight",
                        "model.layers.0.mlp.up_proj.weight",
                    ].joined(separator: ";"),
                    averageGpuMicroseconds: 200
                ),
                syntheticProfileEntry(
                    index: 2,
                    kernelName: "batched_gemv3_seq_bf16_f32s",
                    weightTensorName: [
                        "model.layers.0.self_attn.q_proj.weight",
                        "model.layers.0.self_attn.k_proj.weight",
                        "model.layers.0.self_attn.v_proj.weight",
                    ].joined(separator: ";"),
                    averageGpuMicroseconds: 300
                ),
                syntheticProfileEntry(
                    index: 3,
                    kernelName: "mlp_fused_swiglu_down_seq_bf16_f32s",
                    weightTensorName: "model.layers.0.mlp.down_proj.weight",
                    averageGpuMicroseconds: 400
                ),
                syntheticProfileEntry(
                    index: 4,
                    kernelName: "gemv_seq_bf16_f32s_rps2",
                    weightTensorName: "model.layers.0.self_attn.o_proj.weight",
                    averageGpuMicroseconds: 500
                ),
            ],
            sequenceLength: 128,
            directory: directory
        )

        let csv = try String(contentsOf: url, encoding: .utf8)
        #expect(csv.contains("sequenceLength,routeFamily,role,kernelName,activeCount,totalGpuMicroseconds,averageGpuMicroseconds,routeObservation"))
        #expect(csv.contains("128,batched_projection,linear_attn.in_proj,batched_gemv4_seq_bf16_f32s,1,100.000,100.000,baseline-route-observed"))
        #expect(csv.contains("128,batched_projection,mlp.gate_up,batched_gemv2_seq_bf16_f32s,1,200.000,200.000,baseline-route-observed"))
        #expect(csv.contains("128,batched_projection,self_attn.qkv,batched_gemv3_seq_bf16_f32s,1,300.000,300.000,baseline-route-observed"))
        #expect(csv.contains("128,mlp_fused_down,mlp.down_proj,mlp_fused_swiglu_down_seq_bf16_f32s,1,400.000,400.000,default-runtime-gated-route"))
        #expect(csv.contains("128,single_projection,self_attn.o_proj,gemv_seq_bf16_f32s_rps2,1,500.000,500.000,experimental-route-observed"))
    }

    @Test("Route gate summarizes production sequence routes")
    func routeGateSummarizesProductionSequenceRoutes() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-lm-qwen-route-gate-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let url = try writeRouteGate(
            profilesByLength: [
                64: [
                    syntheticProfileEntry(
                        index: 0,
                        kernelName: "mlp_fused_swiglu_down_seq_bf16_f32s",
                        weightTensorName: "model.layers.0.mlp.down_proj.weight",
                        averageGpuMicroseconds: 100
                    ),
                    syntheticProfileEntry(
                        index: 1,
                        kernelName: "gemv_seq_bf16_f32s",
                        weightTensorName: "model.layers.0.linear_attn.out_proj.weight",
                        averageGpuMicroseconds: 200
                    ),
                    syntheticProfileEntry(
                        index: 2,
                        kernelName: "gemv_seq_bf16_f32s_rps2",
                        weightTensorName: "model.layers.0.self_attn.o_proj.weight",
                        averageGpuMicroseconds: 50
                    ),
                ],
                128: [
                    syntheticProfileEntry(
                        index: 3,
                        kernelName: "mlp_fused_swiglu_down_seq_bf16_f32s",
                        weightTensorName: "model.layers.0.mlp.down_proj.weight",
                        averageGpuMicroseconds: 300
                    ),
                    syntheticProfileEntry(
                        index: 4,
                        kernelName: "gemv_seq_bf16_f32s_rps2",
                        weightTensorName: "model.layers.0.self_attn.o_proj.weight",
                        averageGpuMicroseconds: 400
                    ),
                ],
            ],
            directory: directory
        )

        let csv = try String(contentsOf: url, encoding: .utf8)
        #expect(csv.contains("routeFamily,role,kernelName,productionSequenceLengths,activeCounts,totalGpuMicroseconds,routeObservations,missingSequenceLengths,routeGate"))
        #expect(csv.contains("mlp_fused_down,mlp.down_proj,mlp_fused_swiglu_down_seq_bf16_f32s,64|128,1|1,400.000,default-runtime-gated-route|default-runtime-gated-route,,default-runtime-gated-route-active"))
        #expect(csv.contains("single_projection,linear_attn.out_proj,gemv_seq_bf16_f32s,64,1,200.000,baseline-route-observed,128,missing-production-sequence"))
        #expect(csv.contains("single_projection,self_attn.o_proj,gemv_seq_bf16_f32s_rps2,64|128,1|1,450.000,experimental-route-observed|experimental-route-observed,,experimental-route-observed"))
    }

    @Test("Route readiness combines microbench and full profile gates")
    func routeReadinessCombinesMicrobenchAndFullProfileGates() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-lm-qwen-route-readiness-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let rows = routeReadinessRows(
            candidates: [
                RoutePromotionCandidate(
                    routeFamily: "single_projection",
                    role: "linear_attn.out_proj",
                    variant: "rps2",
                    microbenchAdmission: "candidate-single-gemv-default-route",
                    readinessPrerequisite: "requires-full-profile-route-gate",
                    requiredProfileRouteGate: "experimental-route-observed"
                ),
                RoutePromotionCandidate(
                    routeFamily: "single_projection",
                    role: "self_attn.o_proj",
                    variant: "tile2",
                    microbenchAdmission: "reject-cross-sequence-threshold",
                    readinessPrerequisite: "microbench-rejected",
                    requiredProfileRouteGate: "experimental-route-observed"
                ),
                RoutePromotionCandidate(
                    routeFamily: "batched_projection",
                    role: "mlp.gate_up",
                    variant: "tile2",
                    microbenchAdmission: "candidate-batched-gemv-default-route",
                    readinessPrerequisite: "requires-full-profile-route-gate",
                    requiredProfileRouteGate: "experimental-route-observed"
                ),
                RoutePromotionCandidate(
                    routeFamily: "batched_projection",
                    role: "self_attn.qkv",
                    variant: "tile4",
                    microbenchAdmission: "candidate-batched-gemv-default-route",
                    readinessPrerequisite: "requires-full-profile-route-gate",
                    requiredProfileRouteGate: "experimental-route-observed"
                ),
                RoutePromotionCandidate(
                    routeFamily: "single_projection",
                    role: "mlp.down_proj",
                    variant: "tile4",
                    microbenchAdmission: "candidate-single-gemv-default-route",
                    readinessPrerequisite: "microbench-rejected",
                    requiredProfileRouteGate: "experimental-route-observed"
                ),
            ],
            profileGates: [
                ProfileRouteGate(
                    routeFamily: "single_projection",
                    role: "linear_attn.out_proj",
                    routeGate: "experimental-route-observed"
                ),
                ProfileRouteGate(
                    routeFamily: "single_projection",
                    role: "self_attn.o_proj",
                    routeGate: "experimental-route-observed"
                ),
                ProfileRouteGate(
                    routeFamily: "batched_projection",
                    role: "mlp.gate_up",
                    routeGate: "baseline-route-preserved"
                ),
                ProfileRouteGate(
                    routeFamily: "batched_projection",
                    role: "self_attn.qkv",
                    routeGate: "missing-production-sequence"
                ),
                ProfileRouteGate(
                    routeFamily: "single_projection",
                    role: "mlp.down_proj",
                    routeGate: "experimental-route-observed"
                ),
            ]
        )
        let url = try writeRouteReadiness(rows: rows, directory: directory)
        let csv = try String(contentsOf: url, encoding: .utf8)

        #expect(csv.contains("routeFamily,role,variant,microbenchAdmission,readinessPrerequisite,requiredProfileRouteGate,observedProfileRouteGate,routeReadiness"))
        #expect(csv.contains("single_projection,linear_attn.out_proj,rps2,candidate-single-gemv-default-route,requires-full-profile-route-gate,experimental-route-observed,experimental-route-observed,candidate-production-route"))
        #expect(csv.contains("single_projection,self_attn.o_proj,tile2,reject-cross-sequence-threshold,microbench-rejected,experimental-route-observed,experimental-route-observed,reject-microbench"))
        #expect(csv.contains("batched_projection,mlp.gate_up,tile2,candidate-batched-gemv-default-route,requires-full-profile-route-gate,experimental-route-observed,baseline-route-preserved,reject-full-profile-route-not-observed"))
        #expect(csv.contains("batched_projection,self_attn.qkv,tile4,candidate-batched-gemv-default-route,requires-full-profile-route-gate,experimental-route-observed,missing-production-sequence,reject-full-profile-missing-production-sequence"))
        #expect(csv.contains("single_projection,mlp.down_proj,tile4,candidate-single-gemv-default-route,microbench-rejected,experimental-route-observed,experimental-route-observed,reject-microbench-prerequisite"))
    }

    @Test("Route readiness can be reconstructed from artifact CSVs")
    func routeReadinessCanBeReconstructedFromArtifactCSVs() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-lm-qwen-route-readiness-artifact-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let singleArtifact = directory.appendingPathComponent("qwen35-bf16-single-sequence-gemv-route-promotions.csv")
        try Data("""
        role,variant,productionSequenceLengths,speedupPercents,passingSequenceCount,requiredSequenceCount,minimumSpeedupPercent,thresholdShortfallPercent,failingSequenceLengths,routePromotionAdmission,requiredProfileRouteGate,readinessPrerequisite
        linear_attn.out_proj,rps2,64|128,10.000|10.000,2,2,10.000,0.000,,candidate-single-gemv-default-route,experimental-route-observed,requires-full-profile-route-gate
        self_attn.o_proj,tile2,64|128,1.000|10.000,1,2,1.000,2.000,64,reject-cross-sequence-threshold,,microbench-rejected
        mlp.down_proj,tile4,64|128,10.000|10.000,2,2,10.000,0.000,,candidate-single-gemv-default-route,experimental-route-observed,microbench-rejected
        """.utf8).write(to: singleArtifact, options: .atomic)

        let batchedArtifact = directory.appendingPathComponent("qwen35-bf16-batched-sequence-gemv-route-promotions.csv")
        try Data("""
        role,variant,productionSequenceLengths,speedupPercents,passingSequenceCount,requiredSequenceCount,minimumSpeedupPercent,thresholdShortfallPercent,failingSequenceLengths,routePromotionAdmission,requiredProfileRouteGate,readinessPrerequisite
        mlp.gate_up,tile2,64|128,10.000|10.000,2,2,10.000,0.000,,candidate-batched-gemv-default-route,experimental-route-observed,requires-full-profile-route-gate
        self_attn.qkv,tile4,64|128,10.000|10.000,2,2,10.000,0.000,,candidate-batched-gemv-default-route,experimental-route-observed,requires-full-profile-route-gate
        """.utf8).write(to: batchedArtifact, options: .atomic)

        let routeGateArtifact = directory.appendingPathComponent("qwen35-prefill-route-gate.csv")
        try Data("""
        routeFamily,role,kernelName,productionSequenceLengths,activeCounts,totalGpuMicroseconds,routeObservations,missingSequenceLengths,routeGate
        single_projection,linear_attn.out_proj,gemv_seq_bf16_f32s_rps2,64|128,1|1,200.000,experimental-route-observed|experimental-route-observed,,experimental-route-observed
        single_projection,self_attn.o_proj,gemv_seq_bf16_f32s_tile2,64|128,1|1,200.000,experimental-route-observed|experimental-route-observed,,experimental-route-observed
        single_projection,mlp.down_proj,gemv_seq_bf16_f32s_tile4,64|128,1|1,200.000,experimental-route-observed|experimental-route-observed,,experimental-route-observed
        batched_projection,mlp.gate_up,batched_gemv2_seq_bf16_f32s_tile2,64|128,1|1,200.000,baseline-route-observed|baseline-route-observed,,baseline-route-preserved
        batched_projection,self_attn.qkv,batched_gemv3_seq_bf16_f32s_tile4,64,1,100.000,experimental-route-observed,128,missing-production-sequence
        """.utf8).write(to: routeGateArtifact, options: .atomic)

        let rows = routeReadinessRows(
            candidates: try routePromotionCandidates(singleArtifact: singleArtifact, batchedArtifact: batchedArtifact),
            profileGates: try profileRouteGates(artifact: routeGateArtifact)
        )
        let url = try writeRouteReadiness(rows: rows, directory: directory)
        let csv = try String(contentsOf: url, encoding: .utf8)

        #expect(csv.contains("single_projection,linear_attn.out_proj,rps2,candidate-single-gemv-default-route,requires-full-profile-route-gate,experimental-route-observed,experimental-route-observed,candidate-production-route"))
        #expect(csv.contains("single_projection,self_attn.o_proj,tile2,reject-cross-sequence-threshold,microbench-rejected,,experimental-route-observed,reject-microbench"))
        #expect(csv.contains("single_projection,mlp.down_proj,tile4,candidate-single-gemv-default-route,microbench-rejected,experimental-route-observed,experimental-route-observed,reject-microbench-prerequisite"))
        #expect(csv.contains("batched_projection,mlp.gate_up,tile2,candidate-batched-gemv-default-route,requires-full-profile-route-gate,experimental-route-observed,baseline-route-preserved,reject-full-profile-route-not-observed"))
        #expect(csv.contains("batched_projection,self_attn.qkv,tile4,candidate-batched-gemv-default-route,requires-full-profile-route-gate,experimental-route-observed,missing-production-sequence,reject-full-profile-missing-production-sequence"))
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

    private func printSingleProjectionRoleBreakdown(profiles: [MetalPrefillProfile.Entry]) {
        struct RoleAggregate {
            var count: Int = 0
            var totalMicroseconds: Double = 0
        }

        var byRole: [String: RoleAggregate] = [:]
        for profile in profiles where profile.kernelName.hasPrefix("gemv_seq_bf16_f32s") {
            let role = profile.weightTensorName.map(weightRoleSummary) ?? "(unknown)"
            var aggregate = byRole[role] ?? RoleAggregate()
            aggregate.count += 1
            aggregate.totalMicroseconds += profile.averageGpuMicroseconds
            byRole[role] = aggregate
        }

        guard !byRole.isEmpty else { return }
        let total = byRole.values.reduce(0.0) { $0 + $1.totalMicroseconds }
        print()
        print("=== BF16 single sequence GEMV role breakdown (seqLen 128) ===")
        for (role, aggregate) in byRole.sorted(by: { $0.value.totalMicroseconds > $1.value.totalMicroseconds }) {
            let totalMs = aggregate.totalMicroseconds / 1000.0
            let averageUs = aggregate.totalMicroseconds / Double(max(aggregate.count, 1))
            let share = total > 0 ? aggregate.totalMicroseconds / total * 100.0 : 0
            print("  \(role.padding(toLength: 24, withPad: " ", startingAt: 0)) count=\(aggregate.count) total=\(String(format: "%.3f", totalMs)) ms avg=\(String(format: "%.1f", averageUs)) us share=\(String(format: "%.1f", share))%")
        }
    }

    private func printLinearAttentionWindowSummary(windows: [RecurrentBlockFusionWindow]) {
        guard !windows.isEmpty else { return }
        let first = windows.first
        let last = windows.last
        print()
        print("=== Linear attention recurrent block windows (seqLen 128) ===")
        print("  count=\(windows.count)")
        if let first {
            print("  first layer=\(first.layerIndex) range=\(first.rangeStart)..<\(first.rangeEnd)")
        }
        if let last {
            print("  last layer=\(last.layerIndex) range=\(last.rangeStart)..<\(last.rangeEnd)")
        }
    }

    private func weightRoleSummary(_ tensorName: String) -> String {
        var components = tensorName.split(separator: ".").map(String.init)
        if components.last == "weight" {
            components.removeLast()
        }
        if let layerIndex = components.firstIndex(of: "layers"),
           layerIndex + 2 < components.count {
            return components[(layerIndex + 2)...].joined(separator: ".")
        }
        return components.suffix(3).joined(separator: ".")
    }

    private func writeRouteManifest(
        profiles: [MetalPrefillProfile.Entry],
        sequenceLength: Int,
        directory: URL
    ) throws -> URL {
        let url = directory.appendingPathComponent("qwen35-prefill-route-manifest-seq\(sequenceLength).csv")
        struct Aggregate {
            var count: Int = 0
            var totalGpuMicroseconds: Double = 0
        }

        var groups: [String: Aggregate] = [:]
        for profile in profiles where isProjectionRouteManifestEntry(profile) {
            let routeFamily = projectionRouteFamily(profile.kernelName)
            let role = projectionManifestRole(profile)
            let key = [routeFamily, role, profile.kernelName].joined(separator: "\u{1F}")
            var aggregate = groups[key] ?? Aggregate()
            aggregate.count += 1
            aggregate.totalGpuMicroseconds += profile.averageGpuMicroseconds
            groups[key] = aggregate
        }

        var lines = [
            [
                "sequenceLength",
                "routeFamily",
                "role",
                "kernelName",
                "activeCount",
                "totalGpuMicroseconds",
                "averageGpuMicroseconds",
                "routeObservation",
            ].joined(separator: ","),
        ]
        for (key, aggregate) in groups.sorted(by: { $0.key < $1.key }) {
            let parts = key.split(separator: "\u{1F}", omittingEmptySubsequences: false).map(String.init)
            let routeFamily = parts[0]
            let role = parts[1]
            let kernelName = parts[2]
            lines.append([
                String(sequenceLength),
                routeFamily,
                csvEscape(role),
                csvEscape(kernelName),
                String(aggregate.count),
                String(format: "%.3f", aggregate.totalGpuMicroseconds),
                String(format: "%.3f", aggregate.totalGpuMicroseconds / Double(max(aggregate.count, 1))),
                routeObservation(kernelName: kernelName, sequenceLength: sequenceLength),
            ].joined(separator: ","))
        }

        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func writeRouteGate(
        profilesByLength: [Int: [MetalPrefillProfile.Entry]],
        directory: URL
    ) throws -> URL {
        let productionSequenceLengths = Self.sequenceLengths.filter { $0 >= 64 }
        let url = directory.appendingPathComponent("qwen35-prefill-route-gate.csv")

        struct Aggregate {
            var sequenceLengths: [Int] = []
            var activeCounts: [Int] = []
            var totalGpuMicroseconds: Double = 0
            var routeObservations: [String] = []
        }

        var groups: [String: Aggregate] = [:]
        for sequenceLength in productionSequenceLengths {
            guard let profiles = profilesByLength[sequenceLength] else { continue }
            var sequenceGroups: [String: (count: Int, totalGpuMicroseconds: Double, routeObservation: String)] = [:]
            for profile in profiles where isProjectionRouteManifestEntry(profile) {
                let routeFamily = projectionRouteFamily(profile.kernelName)
                let role = projectionManifestRole(profile)
                let key = [routeFamily, role, profile.kernelName].joined(separator: "\u{1F}")
                let observation = routeObservation(kernelName: profile.kernelName, sequenceLength: sequenceLength)
                var sequenceAggregate = sequenceGroups[key] ?? (count: 0, totalGpuMicroseconds: 0, routeObservation: observation)
                sequenceAggregate.count += 1
                sequenceAggregate.totalGpuMicroseconds += profile.averageGpuMicroseconds
                sequenceAggregate.routeObservation = observation
                sequenceGroups[key] = sequenceAggregate
            }
            for (key, sequenceAggregate) in sequenceGroups {
                var aggregate = groups[key] ?? Aggregate()
                aggregate.sequenceLengths.append(sequenceLength)
                aggregate.activeCounts.append(sequenceAggregate.count)
                aggregate.totalGpuMicroseconds += sequenceAggregate.totalGpuMicroseconds
                aggregate.routeObservations.append(sequenceAggregate.routeObservation)
                groups[key] = aggregate
            }
        }

        var lines = [
            [
                "routeFamily",
                "role",
                "kernelName",
                "productionSequenceLengths",
                "activeCounts",
                "totalGpuMicroseconds",
                "routeObservations",
                "missingSequenceLengths",
                "routeGate",
            ].joined(separator: ","),
        ]
        for (key, aggregate) in groups.sorted(by: { $0.key < $1.key }) {
            let parts = key.split(separator: "\u{1F}", omittingEmptySubsequences: false).map(String.init)
            let routeFamily = parts[0]
            let role = parts[1]
            let kernelName = parts[2]
            let missingSequenceLengths = productionSequenceLengths.filter {
                !aggregate.sequenceLengths.contains($0)
            }
            lines.append([
                routeFamily,
                csvEscape(role),
                csvEscape(kernelName),
                aggregate.sequenceLengths.map(String.init).joined(separator: "|"),
                aggregate.activeCounts.map(String.init).joined(separator: "|"),
                String(format: "%.3f", aggregate.totalGpuMicroseconds),
                aggregate.routeObservations.joined(separator: "|"),
                missingSequenceLengths.map(String.init).joined(separator: "|"),
                routeGate(
                    routeFamily: routeFamily,
                    routeObservations: aggregate.routeObservations,
                    missingSequenceLengths: missingSequenceLengths
                ),
            ].joined(separator: ","))
        }

        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func writeRouteReadiness(rows: [RouteReadinessRow], directory: URL) throws -> URL {
        let url = directory.appendingPathComponent("qwen35-prefill-route-readiness.csv")
        var lines = [
            [
                "routeFamily",
                "role",
                "variant",
                "microbenchAdmission",
                "readinessPrerequisite",
                "requiredProfileRouteGate",
                "observedProfileRouteGate",
                "routeReadiness",
            ].joined(separator: ","),
        ]
        for row in rows.sorted(by: routeReadinessSort) {
            lines.append([
                csvEscape(row.routeFamily),
                csvEscape(row.role),
                csvEscape(row.variant),
                csvEscape(row.microbenchAdmission),
                csvEscape(row.readinessPrerequisite),
                csvEscape(row.requiredProfileRouteGate),
                csvEscape(row.observedProfileRouteGate ?? ""),
                csvEscape(row.routeReadiness),
            ].joined(separator: ","))
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: url, options: .atomic)
        return url
    }

    private func routePromotionCandidates(singleArtifact: URL, batchedArtifact: URL) throws -> [RoutePromotionCandidate] {
        try routePromotionCandidates(artifact: singleArtifact, routeFamily: "single_projection")
            + routePromotionCandidates(artifact: batchedArtifact, routeFamily: "batched_projection")
    }

    private func routePromotionCandidates(artifact: URL, routeFamily: String) throws -> [RoutePromotionCandidate] {
        try parseCSV(artifact).map { row in
            RoutePromotionCandidate(
                routeFamily: routeFamily,
                role: try requiredCSVValue("role", in: row, artifact: artifact),
                variant: try requiredCSVValue("variant", in: row, artifact: artifact),
                microbenchAdmission: try requiredCSVValue("routePromotionAdmission", in: row, artifact: artifact),
                readinessPrerequisite: try requiredCSVValue("readinessPrerequisite", in: row, artifact: artifact),
                requiredProfileRouteGate: try requiredCSVValue("requiredProfileRouteGate", in: row, artifact: artifact)
            )
        }
    }

    private func profileRouteGates(artifact: URL) throws -> [ProfileRouteGate] {
        try parseCSV(artifact).map { row in
            ProfileRouteGate(
                routeFamily: try requiredCSVValue("routeFamily", in: row, artifact: artifact),
                role: try requiredCSVValue("role", in: row, artifact: artifact),
                routeGate: try requiredCSVValue("routeGate", in: row, artifact: artifact)
            )
        }
    }

    private func routeReadinessRows(
        candidates: [RoutePromotionCandidate],
        profileGates: [ProfileRouteGate]
    ) -> [RouteReadinessRow] {
        candidates.map { candidate in
            let observedGate = profileGates.first {
                $0.routeFamily == candidate.routeFamily && $0.role == candidate.role
            }?.routeGate
            return RouteReadinessRow(
                routeFamily: candidate.routeFamily,
                role: candidate.role,
                variant: candidate.variant,
                microbenchAdmission: candidate.microbenchAdmission,
                readinessPrerequisite: candidate.readinessPrerequisite,
                requiredProfileRouteGate: candidate.requiredProfileRouteGate,
                observedProfileRouteGate: observedGate,
                routeReadiness: routeReadiness(
                    microbenchAdmission: candidate.microbenchAdmission,
                    readinessPrerequisite: candidate.readinessPrerequisite,
                    requiredProfileRouteGate: candidate.requiredProfileRouteGate,
                    observedProfileRouteGate: observedGate
                )
            )
        }
    }

    private func routeReadiness(
        microbenchAdmission: String,
        readinessPrerequisite: String,
        requiredProfileRouteGate: String,
        observedProfileRouteGate: String?
    ) -> String {
        guard microbenchAdmission.hasPrefix("candidate-") else {
            return "reject-microbench"
        }
        guard readinessPrerequisite == "requires-full-profile-route-gate" else {
            return "reject-microbench-prerequisite"
        }
        guard let observedProfileRouteGate else {
            return "reject-missing-full-profile-route"
        }
        guard observedProfileRouteGate != "missing-production-sequence" else {
            return "reject-full-profile-missing-production-sequence"
        }
        guard observedProfileRouteGate == requiredProfileRouteGate else {
            return "reject-full-profile-route-not-observed"
        }
        return "candidate-production-route"
    }

    private func isProjectionRouteManifestEntry(_ entry: MetalPrefillProfile.Entry) -> Bool {
        entry.kernelName.hasPrefix("gemv_seq_bf16_f32s")
            || entry.kernelName.hasPrefix("batched_gemv")
            || entry.kernelName.hasPrefix("mlp_fused_swiglu_down")
    }

    private func projectionRouteFamily(_ kernelName: String) -> String {
        if kernelName.hasPrefix("mlp_fused_swiglu_down") {
            return "mlp_fused_down"
        }
        if kernelName.hasPrefix("batched_gemv") {
            return "batched_projection"
        }
        if kernelName.hasPrefix("gemv_seq_bf16_f32s") {
            return "single_projection"
        }
        return "other"
    }

    private func routeObservation(kernelName: String, sequenceLength: Int) -> String {
        if kernelName.hasSuffix("_tile2") || kernelName.hasSuffix("_tile4") || kernelName.hasSuffix("_rps2") {
            return "experimental-route-observed"
        }
        if kernelName.hasPrefix("mlp_fused_swiglu_down") {
            return sequenceLength >= 64 ? "default-runtime-gated-route" : "unexpected-short-sequence-route"
        }
        return "baseline-route-observed"
    }

    private func routeGate(
        routeFamily: String,
        routeObservations: [String],
        missingSequenceLengths: [Int]
    ) -> String {
        guard missingSequenceLengths.isEmpty else {
            return "missing-production-sequence"
        }
        if routeObservations.contains("experimental-route-observed") {
            return "experimental-route-observed"
        }
        if routeFamily == "mlp_fused_down",
           routeObservations.allSatisfy({ $0 == "default-runtime-gated-route" }) {
            return "default-runtime-gated-route-active"
        }
        if routeObservations.allSatisfy({ $0 == "baseline-route-observed" }) {
            return "baseline-route-preserved"
        }
        return "mixed-route-observed"
    }

    private func csvEscape(_ value: String) -> String {
        if value.contains(",") || value.contains("\"") || value.contains("\n") {
            return "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
        }
        return value
    }

    private func parseCSV(_ url: URL) throws -> [[String: String]] {
        let csv = try String(contentsOf: url, encoding: .utf8)
        let parsedRows = try csvRows(csv, artifact: url)
        guard let header = parsedRows.first, !header.isEmpty else {
            throw RouteReadinessArtifactError.emptyCSV(url.path)
        }
        return try parsedRows.dropFirst().map { fields in
            guard fields.count == header.count else {
                throw RouteReadinessArtifactError.rowWidthMismatch(
                    path: url.path,
                    expected: header.count,
                    actual: fields.count
                )
            }
            return Dictionary(uniqueKeysWithValues: zip(header, fields))
        }
    }

    private func csvRows(_ csv: String, artifact: URL) throws -> [[String]] {
        var rows: [[String]] = []
        var row: [String] = []
        var field = ""
        var inQuotes = false
        var index = csv.startIndex
        while index < csv.endIndex {
            let character = csv[index]
            if inQuotes {
                if character == "\"" {
                    let nextIndex = csv.index(after: index)
                    if nextIndex < csv.endIndex, csv[nextIndex] == "\"" {
                        field.append("\"")
                        index = nextIndex
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(character)
                }
            } else {
                switch character {
                case "\"":
                    inQuotes = true
                case ",":
                    row.append(field)
                    field.removeAll(keepingCapacity: true)
                case "\n":
                    row.append(field)
                    field.removeAll(keepingCapacity: true)
                    if !row.allSatisfy(\.isEmpty) {
                        rows.append(row)
                    }
                    row.removeAll(keepingCapacity: true)
                case "\r":
                    break
                default:
                    field.append(character)
                }
            }
            index = csv.index(after: index)
        }
        if inQuotes {
            throw RouteReadinessArtifactError.unclosedQuote(artifact.path)
        }
        row.append(field)
        if !row.allSatisfy(\.isEmpty) {
            rows.append(row)
        }
        return rows
    }

    private func requiredCSVValue(_ key: String, in row: [String: String], artifact: URL) throws -> String {
        guard let value = row[key] else {
            throw RouteReadinessArtifactError.missingColumn(path: artifact.path, column: key)
        }
        return value
    }

    private func projectionManifestRole(_ entry: MetalPrefillProfile.Entry) -> String {
        let tensorName = entry.weightTensorName ?? ""
        if entry.kernelName.hasPrefix("batched_gemv") {
            if tensorName.contains("linear_attn.in_proj") {
                return "linear_attn.in_proj"
            }
            if tensorName.contains("mlp.gate_proj") || tensorName.contains("mlp.up_proj") {
                return "mlp.gate_up"
            }
            if tensorName.contains("self_attn.q_proj")
                || tensorName.contains("self_attn.k_proj")
                || tensorName.contains("self_attn.v_proj") {
                return "self_attn.qkv"
            }
            return "batched_projection"
        }
        return entry.weightTensorName.map(weightRoleSummary) ?? "(unknown)"
    }

    private func routeReadinessSort(_ lhs: RouteReadinessRow, _ rhs: RouteReadinessRow) -> Bool {
        if lhs.routeFamily != rhs.routeFamily { return lhs.routeFamily < rhs.routeFamily }
        if lhs.role != rhs.role { return lhs.role < rhs.role }
        return lhs.variant < rhs.variant
    }

    private func assertDefaultProjectionRoutes(profiles: [MetalPrefillProfile.Entry], sequenceLength: Int) {
        guard shouldAssertDefaultProjectionRoutes() else { return }

        var counts: [String: Int] = [:]
        for profile in profiles where isProjectionRouteManifestEntry(profile) {
            let key = "\(projectionRouteFamily(profile.kernelName))/\(projectionManifestRole(profile))"
            counts[key, default: 0] += 1
        }

        #expect(counts["batched_projection/linear_attn.in_proj"] == 18)
        #expect(counts["batched_projection/mlp.gate_up"] == 24)
        #expect(counts["batched_projection/self_attn.qkv"] == 6)
        #expect(counts["single_projection/linear_attn.out_proj"] == 18)
        #expect(counts["single_projection/self_attn.o_proj"] == 6)

        if sequenceLength >= 64 {
            #expect(counts["mlp_fused_down/mlp.down_proj"] == 24)
            #expect(counts["single_projection/mlp.down_proj"] == nil)
        } else {
            #expect(counts["mlp_fused_down/mlp.down_proj"] == nil)
            #expect(counts["single_projection/mlp.down_proj"] == 24)
        }
    }

    private func shouldAssertDefaultProjectionRoutes() -> Bool {
        let environment = ProcessInfo.processInfo.environment
        let routeOverrideKeys = [
            "SWIFTLM_PREFILL_BF16_SINGLE_TILE2",
            "SWIFTLM_PREFILL_BF16_SINGLE_RPS2",
            "SWIFTLM_PREFILL_BF16_FUSED_MLP_DOWN",
            "SWIFTLM_PREFILL_BF16_FUSED_MLP_DOWN_ROWS",
            "SWIFTLM_PREFILL_BF16_FUSED_MLP_DOWN_ROWS_PER_SIMDGROUP",
            "SWIFTLM_PREFILL_BF16_FUSED_MLP_DOWN_MIN_SEQUENCE_LENGTH",
            "SWIFTLM_PREFILL_BF16_FUSED_ATTENTION_O",
            "SWIFTLM_PREFILL_BF16_RECURRENT_BLOCK_PARTIAL",
            "SWIFTLM_PREFILL_BF16_RECURRENT_BLOCK_FUSED_PARTIAL",
            "SWIFTLM_PREFILL_SSM_SHARED_RMS",
            "SWIFTLM_PREFILL_SSM_PREWRITE_DECAY",
            "SWIFTLM_PREFILL_SSM_QKPAR",
            "SWIFTLM_PREFILL_SSM_THREADGROUP_WIDTH",
        ]
        return !routeOverrideKeys.contains { environment[$0] != nil }
    }

    private func syntheticProfileEntry(
        index: Int,
        kernelName: String,
        weightTensorName: String,
        averageGpuMicroseconds: Double
    ) -> MetalPrefillProfile.Entry {
        MetalPrefillProfile.Entry(
            scope: "step",
            index: index,
            rangeStart: index,
            rangeEnd: index + 1,
            kernelName: kernelName,
            category: "projection",
            mode: "profile",
            layerIndex: 0,
            entryIndex: index,
            weightTensorName: weightTensorName,
            gridWidth: 1,
            gridHeight: 1,
            gridDepth: 1,
            threadgroupWidth: 1,
            threadgroupHeight: 1,
            threadgroupDepth: 1,
            threadgroupMemoryBytes: 0,
            bufferBindingCount: 0,
            inlineConstantBytes: 0,
            uniqueBoundBufferBytes: 0,
            estimatedReadBytes: 0,
            estimatedWriteBytes: 0,
            estimatedTotalBytes: 0,
            estimatedDispatchCount: 1,
            totalGpuMicroseconds: averageGpuMicroseconds,
            averageGpuMicroseconds: averageGpuMicroseconds,
            totalWallMicroseconds: averageGpuMicroseconds,
            averageWallMicroseconds: averageGpuMicroseconds
        )
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

private enum RouteReadinessArtifactError: Error, CustomStringConvertible {
    case emptyCSV(String)
    case rowWidthMismatch(path: String, expected: Int, actual: Int)
    case unclosedQuote(String)
    case missingColumn(path: String, column: String)

    var description: String {
        switch self {
        case .emptyCSV(let path):
            return "CSV artifact is empty: \(path)"
        case .rowWidthMismatch(let path, let expected, let actual):
            return "CSV row width mismatch in \(path): expected \(expected), got \(actual)"
        case .unclosedQuote(let path):
            return "CSV artifact has an unclosed quote: \(path)"
        case .missingColumn(let path, let column):
            return "CSV artifact \(path) is missing required column \(column)"
        }
    }
}
#endif
