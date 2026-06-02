import Foundation
import Metal
import Testing
@testable import MetalCompiler

@Suite("LFM2.5 8B-A1B Decode Profile", .serialized)
struct LFM25A1BDecodeProfileTests {
    @Test("A1B decode step profile identifies hot kernel families", .timeLimit(.minutes(10)))
    func a1bDecodeStepProfileIdentifiesHotKernelFamilies() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let bundlePath = HFCacheLocator.resolveSnapshotPath(
            repoDirectoryName: "models--LiquidAI--LFM2.5-8B-A1B"
        ) else {
            print("[Skip] LFM2.5-8B-A1B not cached. Run `huggingface-cli download LiquidAI/LFM2.5-8B-A1B`.")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        var model = try BenchmarkSupport.setupFromBundle(
            bundlePath: bundlePath,
            maximumPrefillLength: 128
        ).0
        let profiles = try BenchmarkSupport.profileDecodeSteps(
            model: &model,
            device: device,
            iterations: 3,
            filter: { _ in true }
        )

        struct Aggregate {
            var totalMicroseconds: Double = 0
            var count: Int = 0
        }
        var aggregates: [String: Aggregate] = [:]
        for profile in profiles {
            let family = Self.kernelFamily(profile.kernelName)
            aggregates[family, default: Aggregate()].totalMicroseconds += profile.totalMicroseconds / 3.0
            aggregates[family, default: Aggregate()].count += 1
        }
        let totalMicroseconds = aggregates.values.reduce(0) { $0 + $1.totalMicroseconds }
        let sorted = aggregates.sorted { lhs, rhs in
            lhs.value.totalMicroseconds > rhs.value.totalMicroseconds
        }
        print("[LFM2.5 8B-A1B decode profile total_us] \(String(format: "%.0f", totalMicroseconds))")
        var moeProjectionShare = 0.0
        var residualBoundaryShare = 0.0
        var routerShare = 0.0
        for (family, aggregate) in sorted.prefix(12) {
            let share = totalMicroseconds > 0 ? aggregate.totalMicroseconds / totalMicroseconds * 100 : 0
            if Self.isMoEProjectionFamily(family) {
                moeProjectionShare += share
            }
            if family == "synthesized_3way_residual" {
                residualBoundaryShare += share
            }
            if family == "sparse_moe_bf16_router_parallel"
                || family == "sparse_moe_bf16_router_parallel_staged_packed4"
                || family == "residual_rms_router_parallel_bf16_sigmoid" {
                routerShare += share
            }
            print(String(
                format: "[LFM2.5 8B-A1B decode profile] family=%@ count=%d total_us=%.0f share=%.1f%%",
                family,
                aggregate.count,
                aggregate.totalMicroseconds,
                share
            ))
            if let estimate = Self.workEstimate(for: family) {
                let seconds = aggregate.totalMicroseconds / 1_000_000
                let effectiveGBs = seconds > 0 ? estimate.bytes / seconds / 1_000_000_000 : 0
                let effectiveGFLOPs = seconds > 0 ? estimate.flops / seconds / 1_000_000_000 : 0
                print(String(
                    format: "[LFM2.5 8B-A1B MoE roofline] family=%@ active_weight_gb=%.3f work_gflop=%.3f time_us=%.0f effective_gb_s=%.1f effective_gflop_s=%.1f",
                    family,
                    estimate.bytes / 1_000_000_000,
                    estimate.flops / 1_000_000_000,
                    aggregate.totalMicroseconds,
                    effectiveGBs,
                    effectiveGFLOPs
                ))
                #expect(effectiveGBs > 0)
                #expect(effectiveGFLOPs > 0)
            }
        }
        print(String(
            format: "[LFM2.5 8B-A1B decode profile MoE projection share] %.1f%%",
            moeProjectionShare
        ))
        print(String(
            format: "[LFM2.5 8B-A1B decode profile primary route share] moe=%.1f%% residual=%.1f%% router=%.1f%% total=%.1f%%",
            moeProjectionShare,
            residualBoundaryShare,
            routerShare,
            moeProjectionShare + residualBoundaryShare + routerShare
        ))

        #expect(totalMicroseconds > 0)
        #expect(!sorted.isEmpty)
        #expect(moeProjectionShare > 30)
        #expect(moeProjectionShare + residualBoundaryShare + routerShare > 55)
    }

    private static func kernelFamily(_ kernelName: String) -> String {
        if kernelName.hasPrefix("sparse_moe") {
            return kernelName
        }
        if kernelName.hasPrefix("gemv_2048_sq") {
            return "gemv_2048_sq_bf16"
        }
        if kernelName.hasPrefix("gemv_2048_6144") {
            return "gemv_2048_6144_bf16"
        }
        if kernelName.hasPrefix("synthesized_3way") {
            return "synthesized_3way_residual"
        }
        if kernelName.hasPrefix("residual_rms_router_parallel_bf16") {
            return "residual_rms_router_parallel_bf16_sigmoid"
        }
        if kernelName.hasPrefix("shortconv_inproj_update_bf16") {
            return "shortconv_inproj_update_bf16"
        }
        if kernelName.hasPrefix("conv_state_update") {
            return "conv_state_update_bf16"
        }
        if kernelName.hasPrefix("batched_gemv3") {
            return "batched_gemv3_bf16"
        }
        if kernelName.hasPrefix("batched_gemv2") {
            return "batched_gemv2_bf16"
        }
        return kernelName
    }

    private static func workEstimate(for family: String) -> WorkEstimate? {
        switch family {
        case "sparse_moe_bf16_gate_up",
             "sparse_moe_bf16_gate_up_packed4",
             "sparse_moe_bf16_gate_up_staged_packed4":
            let operations = moeLayerCount
                * expertsPerToken
                * 2
                * intermediateDimension
                * hiddenDimension
            return WorkEstimate(
                bytes: Double(operations * bf16Bytes),
                flops: Double(operations * flopsPerFMA)
            )
        case "sparse_moe_bf16_down",
             "sparse_moe_bf16_down_packed4",
             "sparse_moe_bf16_down_blocked8x128_packed4",
             "sparse_moe_bf16_down_blocked8x128_staged_act":
            let operations = moeLayerCount
                * expertsPerToken
                * hiddenDimension
                * intermediateDimension
            return WorkEstimate(
                bytes: Double(operations * bf16Bytes),
                flops: Double(operations * flopsPerFMA)
            )
        default:
            return nil
        }
    }

    private static func isMoEProjectionFamily(_ family: String) -> Bool {
        family == "sparse_moe_bf16_gate_up"
            || family == "sparse_moe_bf16_gate_up_packed4"
            || family == "sparse_moe_bf16_gate_up_staged_packed4"
            || family == "sparse_moe_bf16_down"
            || family == "sparse_moe_bf16_down_packed4"
            || family == "sparse_moe_bf16_down_blocked8x128_packed4"
            || family == "sparse_moe_bf16_down_blocked8x128_staged_act"
    }

    private struct WorkEstimate {
        var bytes: Double
        var flops: Double
    }

    private static let hiddenDimension = 2_048
    private static let intermediateDimension = 1_792
    private static let expertsPerToken = 4
    private static let moeLayerCount = 22
    private static let bf16Bytes = 2
    private static let flopsPerFMA = 2
}
