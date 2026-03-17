import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Benchmark prefill and decode throughput with real model weights.
///
/// Reports tok/s for prefill (batch) and decode (autoregressive).
/// Run with:
///   xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
///       -only-testing 'MetalCompilerTests/BenchmarkTests' -parallel-testing-enabled NO
@Suite("Benchmark", .serialized)
struct BenchmarkTests {

    private static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"
    private static let outputPath = "/Users/1amageek/Desktop/swift-lm/TestData/benchmark.txt"

    // MARK: - Prefill Benchmark

    @Test("Prefill throughput (tok/s)")
    func prefillBenchmark() throws {
        let (model, _) = try setupOrSkip()
        var inferenceModel = model

        // Warm-up
        let warmupTokens: [Int32] = [1, 1, 6]
        _ = inferenceModel.prefill(tokens: warmupTokens)
        inferenceModel.resetCaches()

        // Benchmark with varying sequence lengths
        let lengths = [16, 32, 64]
        for length in lengths {
            inferenceModel.resetCaches()
            let tokens = [Int32](repeating: 1, count: length)

            let start = CFAbsoluteTimeGetCurrent()
            _ = inferenceModel.prefill(tokens: tokens)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            let tokPerSec = Double(length) / elapsed
            print("[Benchmark] prefill \(length) tokens: \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.3f", elapsed * 1000))ms)")
        }
    }

    // MARK: - Decode Benchmark

    @Test("Decode throughput (tok/s)")
    func decodeBenchmark() throws {
        let (model, _) = try setupOrSkip()
        var inferenceModel = model

        // Prefill with short prompt
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = inferenceModel.prefill(tokens: promptTokens)

        // Warm-up decode
        for _ in 0..<3 {
            currentToken = inferenceModel.decodeSync(tokenID: currentToken)
        }

        // Benchmark decode
        let decodeSteps = 50
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<decodeSteps {
            currentToken = inferenceModel.decodeSync(tokenID: currentToken)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let tokPerSec = Double(decodeSteps) / elapsed
        let msPerToken = elapsed / Double(decodeSteps) * 1000
        print("[Benchmark] decode \(decodeSteps) tokens: \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.2f", msPerToken)) ms/tok)")
    }

    // MARK: - Aggressive Optimizer Benchmark

    @Test("Aggressive optimizer: decode throughput")
    func aggressiveDecodeBenchmark() throws {
        let (model, _) = try setupOrSkip(optimizer: AggressiveOptimizer())
        var inferenceModel = model

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = inferenceModel.prefill(tokens: promptTokens)

        // Warm-up
        for _ in 0..<3 {
            currentToken = inferenceModel.decodeSync(tokenID: currentToken)
        }

        let decodeSteps = 50
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<decodeSteps {
            currentToken = inferenceModel.decodeSync(tokenID: currentToken)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let tokPerSec = Double(decodeSteps) / elapsed
        let msPerToken = elapsed / Double(decodeSteps) * 1000
        print("[Benchmark/aggressive] decode \(decodeSteps) tokens: \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.2f", msPerToken)) ms/tok)")
        print("[Benchmark/aggressive] dispatches: \(inferenceModel.plan.fusedEntryCount) (from \(inferenceModel.plan.unfusedEntryCount))")
    }

    // MARK: - End-to-End Benchmark

    @Test("End-to-end: prefill + decode with memory diagnostics")
    func endToEndBenchmark() throws {
        let (model, _) = try setupOrSkip()
        var inferenceModel = model

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let generateCount = 100

        // Buffer memory diagnostics
        let b = inferenceModel.plan.buffers
        let hiddenBytes = b.hidden.length
        let scratchBytes = b.scratch.length
        let residualBytes = b.residual.length
        let logitsBytes = b.logits.length
        let kvBytes = (b.kvCache?.keys.length ?? 0) + (b.kvCache?.values.length ?? 0)
        let convBytes = b.convState?.length ?? 0
        let weightBytes = b.weights.reduce(0) { $0 + $1.length }
        let totalGPUBytes = hiddenBytes + scratchBytes + residualBytes + logitsBytes + kvBytes + convBytes
        let totalWeightBytes = weightBytes

        func storageMode(_ buf: MTLBuffer) -> String {
            switch buf.storageMode {
            case .shared: return "shared"
            case .private: return "private"
            case .memoryless: return "memoryless"
            @unknown default: return "unknown"
            }
        }

        let totalStart = CFAbsoluteTimeGetCurrent()
        let prefillStart = CFAbsoluteTimeGetCurrent()
        var currentToken = inferenceModel.prefill(tokens: promptTokens)
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart

        let decodeStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<generateCount {
            currentToken = inferenceModel.decodeSync(tokenID: currentToken)
        }
        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        let totalTime = CFAbsoluteTimeGetCurrent() - totalStart
        let totalTokens = promptTokens.count + generateCount

        // Estimate memory bandwidth
        // Each decode step reads: hidden + weight (per GEMV) + KV cache (attention)
        // Approximate: 2 * model_size_bytes per token (read weights + write intermediates)
        let bytesPerToken = Double(totalWeightBytes) * 2.0  // rough estimate
        let decodeBandwidth = bytesPerToken * Double(generateCount) / decodeTime / 1e9

        var report = "=== Benchmark: LFM2.5-1.2B ===\n\n"
        report += "Memory Layout:\n"
        report += "  hidden:   \(String(format: "%6.1f", Double(hiddenBytes) / 1024)) KB (\(storageMode(b.hidden)))\n"
        report += "  scratch:  \(String(format: "%6.1f", Double(scratchBytes) / 1024)) KB (\(storageMode(b.scratch)))\n"
        report += "  residual: \(String(format: "%6.1f", Double(residualBytes) / 1024)) KB (\(storageMode(b.residual)))\n"
        report += "  logits:   \(String(format: "%6.1f", Double(logitsBytes) / 1024)) KB (\(storageMode(b.logits)))\n"
        report += "  KV cache: \(String(format: "%6.1f", Double(kvBytes) / 1024 / 1024)) MB (\(b.kvCache.map { storageMode($0.keys) } ?? "none"))\n"
        report += "  conv:     \(String(format: "%6.1f", Double(convBytes) / 1024)) KB (\(b.convState.map { storageMode($0) } ?? "none"))\n"
        report += "  weights:  \(String(format: "%6.1f", Double(weightBytes) / 1024 / 1024)) MB (\(b.weights.first.map { storageMode($0) } ?? "none"))\n"
        report += "  total intermediate: \(String(format: "%.1f", Double(totalGPUBytes) / 1024 / 1024)) MB\n"
        report += "  total weights:      \(String(format: "%.1f", Double(totalWeightBytes) / 1024 / 1024)) MB\n"
        report += "\nDispatch:\n"
        report += "  decode steps: \(inferenceModel.plan.steps.count)\n"
        report += "  fused entries: \(inferenceModel.plan.fusedEntryCount) (from \(inferenceModel.plan.unfusedEntryCount))\n"
        report += "\nThroughput:\n"
        report += "  prefill: \(promptTokens.count) tokens, \(String(format: "%.1f", Double(promptTokens.count) / prefillTime)) tok/s\n"
        report += "  decode:  \(generateCount) tokens, \(String(format: "%.1f", Double(generateCount) / decodeTime)) tok/s (\(String(format: "%.2f", decodeTime / Double(generateCount) * 1000)) ms/tok)\n"
        report += "  total:   \(totalTokens) tokens in \(String(format: "%.0f", totalTime * 1000))ms\n"
        report += "\nBandwidth (estimate):\n"
        report += "  decode: \(String(format: "%.1f", decodeBandwidth)) GB/s (2× weight read per token)\n"

        print(report)
        try report.write(toFile: Self.outputPath, atomically: true, encoding: .utf8)
    }

    // MARK: - Compilation Benchmark

    @Test("Compilation time (IR → dispatch plan)")
    func compilationBenchmark() throws {
        let (_, store) = try setupOrSkip()

        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)
        guard let device = MTLCreateSystemDefaultDevice() else { throw BenchError.noDevice }

        let start = CFAbsoluteTimeGetCurrent()
        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let compileTime = CFAbsoluteTimeGetCurrent() - start

        let prefillStart = CFAbsoluteTimeGetCurrent()
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, maximumSequenceLength: 4096,
            stafWeightStore: store, device: device)
        let prefillCompileTime = CFAbsoluteTimeGetCurrent() - prefillStart

        print("[Benchmark] compilation:")
        print("  decode plan: \(decodePlan.fusedEntryCount) dispatches, \(String(format: "%.0f", compileTime * 1000))ms")
        print("  prefill plan: \(prefillPlan.stepCount) steps, \(String(format: "%.0f", prefillCompileTime * 1000))ms")

        // Optimizer comparison: count dispatch entries with each strategy
        let optimizers: [any DispatchOptimizer] = [
            NoOptimizer(),
            StandardOptimizer(),
            AggressiveOptimizer(),
        ]
        print("\n[Benchmark] optimizer comparison:")
        for opt in optimizers {
            let comp = MetalInferenceCompiler(optimizer: opt)
            let report = comp.analyzeOptimization(graph: resolved, hiddenSize: 2048)
            report.printReport()
        }
    }

    // MARK: - Per-Step Profiling

    @Test("Per-step decode profiling")
    func perStepDecodeProfile() throws {
        let (model, _) = try setupOrSkip()
        var m = model
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { throw BenchError.noDevice }

        // Prefill + warm-up
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = m.prefill(tokens: promptTokens)
        for _ in 0..<3 { currentToken = m.decodeSync(tokenID: currentToken) }

        // Profile: run each step in its own command buffer
        let plan = m.plan
        let steps = plan.steps
        let iterations = 10

        // Set up input for a decode step
        plan.buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(m.position)
        plan.buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = currentToken

        struct StepProfile {
            let index: Int
            let kernelName: String
            let gridSize: MTLSize
            let threadgroupSize: MTLSize
            var totalMicroseconds: Double = 0
        }

        var profiles: [StepProfile] = steps.enumerated().map { (i, step) in
            StepProfile(
                index: i,
                kernelName: step.pipeline.label ?? "step_\(i)",
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize)
        }

        // Warm up all steps once
        for step in steps {
            guard let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { continue }
            if step.sync == .bufferBarrier { enc.memoryBarrier(scope: .buffers) }
            enc.setComputePipelineState(step.pipeline)
            for (index, buffer, offset) in step.bufferBindings {
                enc.setBuffer(buffer, offset: offset, index: index)
            }
            for (index, value) in step.bytesBindings {
                value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) }
            }
            if step.threadgroupMemoryLength > 0 {
                enc.setThreadgroupMemoryLength(step.threadgroupMemoryLength, index: 0)
            }
            enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        // Timed runs
        for _ in 0..<iterations {
            for (i, step) in steps.enumerated() {
                guard let cb = queue.makeCommandBuffer(),
                      let enc = cb.makeComputeCommandEncoder() else { continue }
                if step.sync == .bufferBarrier { enc.memoryBarrier(scope: .buffers) }
                enc.setComputePipelineState(step.pipeline)
                for (index, buffer, offset) in step.bufferBindings {
                    enc.setBuffer(buffer, offset: offset, index: index)
                }
                for (index, value) in step.bytesBindings {
                    value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) }
                }
                if step.threadgroupMemoryLength > 0 {
                    enc.setThreadgroupMemoryLength(step.threadgroupMemoryLength, index: 0)
                }
                enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()

                let gpuStart = cb.gpuStartTime
                let gpuEnd = cb.gpuEndTime
                profiles[i].totalMicroseconds += (gpuEnd - gpuStart) * 1_000_000
            }
        }

        // Aggregate by kernel name
        struct KernelAggregate {
            var totalMicroseconds: Double = 0
            var count: Int = 0
            var gridSample: MTLSize = MTLSize(width: 0, height: 0, depth: 0)
        }
        var aggregates: [String: KernelAggregate] = [:]
        let totalMicroseconds = profiles.reduce(0.0) { $0 + $1.totalMicroseconds }

        for p in profiles {
            let avgUs = p.totalMicroseconds / Double(iterations)
            aggregates[p.kernelName, default: KernelAggregate()].totalMicroseconds += avgUs
            aggregates[p.kernelName, default: KernelAggregate()].count += 1
            if aggregates[p.kernelName]?.gridSample.width == 0 {
                aggregates[p.kernelName]?.gridSample = p.gridSize
            }
        }

        let avgTotalUs = totalMicroseconds / Double(iterations)
        let sorted = aggregates.sorted { $0.value.totalMicroseconds > $1.value.totalMicroseconds }

        print("\n=== Per-Step Decode Profile: LFM2.5-1.2B (avg of \(iterations) runs) ===")
        print("Total: \(String(format: "%.0f", avgTotalUs)) us (\(String(format: "%.1f", avgTotalUs / 1000)) ms)")
        print("")
        let header = "Kernel".padding(toLength: 35, withPad: " ", startingAt: 0)
            + "Count Total us     %  Grid"
        print(header)
        print(String(repeating: "-", count: 80))

        for (name, agg) in sorted {
            let pct = agg.totalMicroseconds / avgTotalUs * 100
            let grid = "\(agg.gridSample.width)x\(agg.gridSample.height)x\(agg.gridSample.depth)"
            let pad = name.padding(toLength: 35, withPad: " ", startingAt: 0)
            print("\(pad)\(String(format: "%5d %8.0f %5.1f%%", agg.count, agg.totalMicroseconds, pct))  \(grid)")
        }

        // Per-step detail (top 20 by time)
        print("\n--- Top 20 individual steps ---")
        let topSteps = profiles.sorted { $0.totalMicroseconds > $1.totalMicroseconds }.prefix(20)
        for p in topSteps {
            let avgUs = p.totalMicroseconds / Double(iterations)
            let pct = avgUs / avgTotalUs * 100
            let grid = "\(p.gridSize.width)x\(p.gridSize.height)"
            let name = p.kernelName.padding(toLength: 30, withPad: " ", startingAt: 0)
            print("  [\(String(format: "%3d", p.index))] \(name) \(String(format: "%6.0f", avgUs)) us (\(String(format: "%4.1f", pct))%)  grid=\(grid)")
        }
    }

    // MARK: - Optimizer Comparison Benchmark

    @Test("Optimizer comparison: prefill + decode throughput")
    func optimizerComparisonBenchmark() throws {
        let optimizers: [(any DispatchOptimizer, String)] = [
            (NoOptimizer(), "none"),
            (StandardOptimizer(), "standard"),
            (AggressiveOptimizer(), "aggressive"),
        ]

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let prefillLength = 64
        let decodeSteps = 50

        let iterations = 5

        print("\n=== Optimizer Comparison: LFM2.5-1.2B (\(iterations) iterations) ===")
        print("Optimizer     Decode Pfill  Dec tok/s Pfl tok/s Dec ms/tk Pfl ms/tk")
        print(String(repeating: "-", count: 72))

        for (opt, name) in optimizers {
            let (model, _) = try setupOrSkip(optimizer: opt)
            var m = model

            var decResults: [Double] = []
            var pfResults: [Double] = []

            for _ in 0..<iterations {
                // Prefill benchmark
                m.resetCaches()
                let prefillTokens = [Int32](repeating: 1, count: prefillLength)
                let pfStart = CFAbsoluteTimeGetCurrent()
                _ = m.prefill(tokens: prefillTokens)
                let pfTime = CFAbsoluteTimeGetCurrent() - pfStart
                pfResults.append(pfTime)

                // Decode benchmark
                m.resetCaches()
                var currentToken = m.prefill(tokens: promptTokens)
                for _ in 0..<3 { currentToken = m.decodeSync(tokenID: currentToken) }

                let decStart = CFAbsoluteTimeGetCurrent()
                for _ in 0..<decodeSteps { currentToken = m.decodeSync(tokenID: currentToken) }
                let decTime = CFAbsoluteTimeGetCurrent() - decStart
                decResults.append(decTime)
            }

            // Median (more stable than mean for GPU benchmarks)
            let decMedian = decResults.sorted()[iterations / 2]
            let pfMedian = pfResults.sorted()[iterations / 2]

            let decTokPerSec = Double(decodeSteps) / decMedian
            let pfTokPerSec = Double(prefillLength) / pfMedian
            let decMsPerTok = decMedian / Double(decodeSteps) * 1000
            let pfMsPerTok = pfMedian / Double(prefillLength) * 1000
            let decDispatches = m.plan.fusedEntryCount
            let pfSteps = m.prefillPlan?.stepCount ?? 0

            let pad = name.padding(toLength: 12, withPad: " ", startingAt: 0)
            print("\(pad) \(String(format: "%6d %5d %9.1f %9.1f %9.2f %9.2f", decDispatches, pfSteps, decTokPerSec, pfTokPerSec, decMsPerTok, pfMsPerTok))")
        }
    }

    // MARK: - Setup

    private func setupOrSkip(
        optimizer: (any DispatchOptimizer)? = nil
    ) throws -> (MetalInferenceModel, STAFWeightStore) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw BenchError.noDevice }

        let stafURL = URL(fileURLWithPath: Self.stafPath)
        if !FileManager.default.fileExists(atPath: stafURL.path) {
            let safetensorsURL = stafURL.deletingLastPathComponent()
                .appendingPathComponent("model.safetensors")
            guard FileManager.default.fileExists(atPath: safetensorsURL.path) else {
                throw BenchError.noModel
            }
            try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        }

        let store = try STAFLoader().load(at: stafURL, device: device)

        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler(optimizer: optimizer)
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, maximumSequenceLength: 64,
            stafWeightStore: store,
            sharedKVCache: decodePlan.buffers.kvCache,
            sharedConvState: decodePlan.buffers.convState,
            sharedConvStateDimension: decodePlan.buffers.convStateDimension,
            sharedConvStateKernelSize: decodePlan.buffers.convStateKernelSize,
            device: device)

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan

        return (model, store)
    }

    private enum BenchError: Error {
        case noDevice
        case noModel
    }
}
