import Foundation
import CoreML

/// Manages CoreML model compilation from SwiftLM IR via Python coremltools.
///
/// Handles:
/// - Invoking the Python compilation script
/// - Caching compiled .mlpackage files
/// - Loading compiled MLModel instances
///
/// The compilation script (`scripts/compile_full_model.py`) generates a stateful
/// CoreML model with KV cache from model configuration parameters.
public struct CoreMLCompilationManager: Sendable {

    /// Directory where compiled .mlpackage files are cached.
    private let cacheDir: URL

    /// Path to the Python compilation script.
    private let scriptPath: String

    public init(cacheDir: URL? = nil, scriptPath: String? = nil) {
        self.cacheDir = cacheDir ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-mlx-lm-coreml", isDirectory: true)
        self.scriptPath = scriptPath ?? Self.findScript()
    }

    /// Compile a CoreML model from model parameters, or load from cache.
    ///
    /// - Parameters:
    ///   - D: Hidden dimension
    ///   - H: Number of attention heads
    ///   - KVH: Number of KV heads
    ///   - hd: Head dimension
    ///   - I: Intermediate (FFN) dimension
    ///   - L: Number of layers
    ///   - maxSeqLen: Maximum sequence length for KV cache
    ///   - vocabSize: Vocabulary size
    ///   - computeUnits: CoreML compute units to use
    /// - Returns: Loaded MLModel ready for inference
    public func compileAndLoad(
        D: Int, H: Int, KVH: Int, hd: Int, I: Int, L: Int,
        maxSeqLen: Int = 512, vocabSize: Int = 32000,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) throws -> MLModel {
        let name = "full_D\(D)_L\(L)_seq\(maxSeqLen)"
        let packagePath = cacheDir.appendingPathComponent("\(name).mlpackage")

        // Check cache
        if !FileManager.default.fileExists(atPath: packagePath.path) {
            try compileModel(
                D: D, H: H, KVH: KVH, hd: hd, I: I, L: L,
                maxSeqLen: maxSeqLen, vocabSize: vocabSize,
                outputDir: cacheDir)
        }

        guard FileManager.default.fileExists(atPath: packagePath.path) else {
            throw CoreMLCompilationError.compilationFailed(
                "Model not found after compilation: \(packagePath.path)")
        }

        // Compile to mlmodelc and load
        let compiledURL = try MLModel.compileModel(at: packagePath)
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        return try MLModel(contentsOf: compiledURL, configuration: config)
    }

    /// Check if a compiled model exists in cache.
    public func isCached(D: Int, L: Int, maxSeqLen: Int = 512) -> Bool {
        let name = "full_D\(D)_L\(L)_seq\(maxSeqLen)"
        let path = cacheDir.appendingPathComponent("\(name).mlpackage")
        return FileManager.default.fileExists(atPath: path.path)
    }

    // MARK: - Private

    private func compileModel(
        D: Int, H: Int, KVH: Int, hd: Int, I: Int, L: Int,
        maxSeqLen: Int, vocabSize: Int, outputDir: URL
    ) throws {
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [
            scriptPath,
            "--output-dir", outputDir.path,
            "--D", String(D),
            "--H", String(H),
            "--KVH", String(KVH),
            "--hd", String(hd),
            "--I", String(I),
            "--layers", String(L),
            "--max-seq-len", String(maxSeqLen),
            "--vocab-size", String(vocabSize),
        ]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        print("[CoreMLCompilationManager] compiling D=\(D) L=\(L) ...")
        let startTime = CFAbsoluteTimeGetCurrent()

        try process.run()
        process.waitUntilExit()

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(),
                            encoding: .utf8) ?? ""

        guard process.terminationStatus == 0 else {
            throw CoreMLCompilationError.compilationFailed(
                "Python script failed (exit \(process.terminationStatus)):\n\(output)")
        }

        print("[CoreMLCompilationManager] compiled in \(String(format: "%.1f", elapsed))s")
    }

    private static func findScript() -> String {
        // Look for script relative to the package root
        let candidates = [
            "scripts/compile_full_model.py",
            "../scripts/compile_full_model.py",
            "../../scripts/compile_full_model.py",
        ]
        let fm = FileManager.default
        for candidate in candidates {
            let url = URL(fileURLWithPath: fm.currentDirectoryPath)
                .appendingPathComponent(candidate)
            if fm.fileExists(atPath: url.path) {
                return url.path
            }
        }
        // Fallback: assume it's in the PATH or specified explicitly
        return "scripts/compile_full_model.py"
    }
}

/// Errors from CoreML compilation.
public enum CoreMLCompilationError: Error, LocalizedError {
    case compilationFailed(String)
    case pythonNotFound
    case coremlToolsNotInstalled

    public var errorDescription: String? {
        switch self {
        case .compilationFailed(let msg): return "CoreML compilation failed: \(msg)"
        case .pythonNotFound: return "Python 3 not found. Install python3 to use CoreML backend."
        case .coremlToolsNotInstalled: return "coremltools not installed. Run: pip3 install coremltools"
        }
    }
}
