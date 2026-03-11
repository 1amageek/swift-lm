import Testing
import Foundation
import GGUFParser
@testable import MLXLM

@Suite("GGUFModelLoader Profiling", .tags(.diagnostic))
struct LoaderProfilingTests {

    private static let cachedModelPath: String = {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent("swift-mlx-lm/huggingface/unsloth--Qwen3.5-0.8B-GGUF/main/Qwen3.5-0.8B-Q4_K_M.gguf")
            .path
    }()

    /// Determine whether convertDirect bottleneck is I/O (mmap page fault) or CPU (pack).
    ///
    /// Strategy: call convertDirect twice per tensor.
    ///   - 1st call: cold (mmap pages not resident) = I/O + CPU
    ///   - 2nd call: warm (pages in page cache) = CPU only
    ///   - Difference = I/O cost
    @Test("IO vs CPU breakdown")
    func ioCpuBreakdown() throws {
        let url = URL(fileURLWithPath: Self.cachedModelPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("[SKIP] cached model not found at \(url.path)")
            return
        }

        let file = try GGUFFile.parse(url: url)
        let bridge = GGUFTensorBridge()

        // Pick tensors of varying sizes
        let targets: [(String, Int)] = file.tensors.compactMap { t in
            let elements = t.dimensions.reduce(1, *)
            return (t.name, elements)
        }.sorted { $0.1 > $1.1 }

        // Take top 5 largest + 5 smallest weight tensors
        let largest = Array(targets.filter { $0.0.hasSuffix(".weight") && $0.1 > 1 }.prefix(5))
        let smallest = Array(targets.filter { $0.0.hasSuffix(".weight") && $0.1 > 1 }.suffix(5))
        let selected = largest + smallest

        print("[profile] Selected \(selected.count) tensors for I/O vs CPU test")
        print("[profile] -------------------------------------------------------")

        var totalCold = 0.0
        var totalWarm = 0.0

        for (name, _) in selected {
            guard let tensor = file.tensors.first(where: { $0.name == name }) else { continue }
            let data = try file.tensorData(for: tensor)
            let elements = tensor.dimensions.reduce(1, *)
            let isWeight = name.hasSuffix(".weight")

            // Cold run (may trigger page faults if pages were evicted)
            let t0 = CFAbsoluteTimeGetCurrent()
            if isWeight {
                _ = try bridge.convertDirect(tensor: tensor, data: data)
            } else {
                _ = try bridge.convert(tensor: tensor, data: data)
            }
            let coldMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

            // Warm run (pages now in cache)
            let t1 = CFAbsoluteTimeGetCurrent()
            if isWeight {
                _ = try bridge.convertDirect(tensor: tensor, data: data)
            } else {
                _ = try bridge.convert(tensor: tensor, data: data)
            }
            let warmMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0

            let ioMs = max(0, coldMs - warmMs)
            totalCold += coldMs
            totalWarm += warmMs

            print("[profile] \(name)")
            print("[profile]   elements=\(elements) qtype=\(tensor.quantizationType)")
            print("[profile]   cold=\(String(format: "%.1f", coldMs))ms  warm=\(String(format: "%.1f", warmMs))ms  io=\(String(format: "%.1f", ioMs))ms")
        }

        let totalIo = max(0, totalCold - totalWarm)
        print("[profile] -------------------------------------------------------")
        print("[profile] TOTAL cold=\(String(format: "%.0f", totalCold))ms  warm=\(String(format: "%.0f", totalWarm))ms  io=\(String(format: "%.0f", totalIo))ms")
        print("[profile] CPU=\(String(format: "%.0f%%", totalWarm / totalCold * 100))  I/O=\(String(format: "%.0f%%", totalIo / totalCold * 100))")
    }
}
