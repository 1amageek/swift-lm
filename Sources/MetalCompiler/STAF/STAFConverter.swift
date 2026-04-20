import Foundation

/// Converts safetensors files to STAF (SafeTensor Accelerated Format).
///
/// The converter performs a one-time offline transformation:
/// 1. Parse safetensors headers to discover tensors
/// 2. Repack quantized weights into interleaved block format
/// 3. Write STAF with 4KB-aligned payload for zero-copy GPU access
///
/// safetensors remain source of truth. STAF is a regenerable cache.
public struct STAFConverter: Sendable {

    private let planner = STAFConversionPlanner()
    private let writer = STAFWriter()
    private let validator = STAFValidator()

    public init() {}

    /// Convert one or more safetensors files into a single STAF file.
    ///
    /// - Parameters:
    ///   - safetensorsURLs: Paths to safetensors shards (sorted by filename).
    ///   - outputURL: Destination path for the .staf file.
    ///   - metadata: Optional file-level metadata to store in the STAF cache.
    /// - Throws: If parsing fails or the output file cannot be written.
    public func convert(
        safetensorsURLs: [URL],
        outputURL: URL,
        quantization: MLXQuantizationHint? = nil,
        metadata: STAFFileMetadata = .empty
    ) throws {
        let plan = try planner.plan(
            safetensorsURLs: safetensorsURLs,
            quantization: quantization
        )
        let fileMetadata = STAFFileMetadata.defaultCacheMetadata(sourceShardCount: plan.sortedURLs.count)
            .merged(with: metadata)
        try writer.write(plan: plan, outputURL: outputURL, metadata: fileMetadata)
    }

    /// Check if an existing STAF cache is still valid for the given bundle inputs.
    ///
    /// The cache remains valid only when:
    /// - the STAF header and tables are structurally valid
    /// - the source safetensors shards are not newer than the cache
    /// - optional expected metadata matches the file-level metadata stored in STAF
    public func isValid(
        stafURL: URL,
        safetensorsURLs: [URL],
        expectedMetadata: STAFFileMetadata? = nil
    ) throws -> Bool {
        try validator.isValid(
            stafURL: stafURL,
            safetensorsURLs: safetensorsURLs,
            expectedMetadata: expectedMetadata
        )
    }
}

// MARK: - Errors

public enum STAFConversionError: Error, CustomStringConvertible {
    case readFailed(String)
    case tensorNotFound(String)
    case unsupportedFormat(UInt8)
    case missingQuantizationHint(String)
    case unsupportedQuantization(bits: Int, groupSize: Int)
    case inconsistentQuantizationShape(name: String, reason: String)

    public var description: String {
        switch self {
        case .readFailed(let name): return "STAFConversionError: failed to read tensor '\(name)'"
        case .tensorNotFound(let name): return "STAFConversionError: tensor '\(name)' not found"
        case .unsupportedFormat(let id): return "STAFConversionError: unsupported format 0x\(String(id, radix: 16))"
        case .missingQuantizationHint(let name):
            return "STAFConversionError: quantized companion tensors exist for '\(name)' but no MLXQuantizationHint was provided. Supply config.json `quantization` metadata."
        case .unsupportedQuantization(let bits, let groupSize):
            return "STAFConversionError: unsupported MLX quantization (bits=\(bits), group_size=\(groupSize))"
        case .inconsistentQuantizationShape(let name, let reason):
            return "STAFConversionError: inconsistent quantization shape for '\(name)': \(reason)"
        }
    }
}
