import Foundation
import MLX
import GGUFParser

/// Converts GGUF tensor data to MLXArray values.
///
/// Dequantizes all formats to F16. Bit layouts match llama.cpp ggml-quants.c.
struct GGUFTensorBridge {

    init() {}

    /// Convert a GGUF tensor to an MLXArray.
    func convert(tensor: GGUFTensorInfo, data: Data) throws -> MLXArray {
        let qtype = tensor.quantizationType

        // GGUF dimensions: ne[0] is innermost, ne[1] is next, etc.
        // MLX uses row-major with last dim fastest, so reverse.
        let shape = tensor.dimensions.reversed().map { Int($0) }

        switch qtype {
        case .f32:
            return loadFloat32(data: data, shape: shape)
        case .f16:
            return loadFloat16(data: data, shape: shape)
        case .bf16:
            return loadBFloat16(data: data, shape: shape)
        case .q4_0:
            return try dequantizeQ4_0(data: data, shape: shape)
        case .q8_0:
            return try dequantizeQ8_0(data: data, shape: shape)
        case .q4_K:
            return try dequantizeQ4_K(data: data, shape: shape)
        case .q6_K:
            return try dequantizeQ6_K(data: data, shape: shape)
        case .q2_K:
            return try dequantizeQ2_K(data: data, shape: shape)
        case .q3_K:
            return try dequantizeQ3_K(data: data, shape: shape)
        case .q5_K:
            return try dequantizeQ5_K(data: data, shape: shape)
        default:
            throw GGUFLoadError.unsupportedQuantization(qtype.rawValue)
        }
    }

    // MARK: - Unquantized Loaders

    private func loadFloat32(data: Data, shape: [Int]) -> MLXArray {
        MLXArray(data, shape, type: Float.self).asType(.float16)
    }

    private func loadFloat16(data: Data, shape: [Int]) -> MLXArray {
        MLXArray(data, shape, type: Float16.self)
    }

    private func loadBFloat16(data: Data, shape: [Int]) -> MLXArray {
        MLXArray(data, shape, type: UInt16.self).view(dtype: .bfloat16).asType(.float16)
    }

    // MARK: - Q4_0 Dequantization

    /// Q4_0: Block of 32 elements = 2 bytes (f16 scale) + 16 bytes (packed 4-bit)
    private func dequantizeQ4_0(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 18
                let scale = readFloat16(bytes, offset)

                for j in 0..<16 {
                    let byte = bytes[offset + 2 + j]
                    result[block * 32 + j] = Float(Int(byte & 0x0F) - 8) * scale
                    result[block * 32 + j + 16] = Float(Int((byte >> 4) & 0x0F) - 8) * scale
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q8_0 Dequantization

    /// Q8_0: Block of 32 elements = 2 bytes (f16 scale) + 32 bytes (int8 values)
    private func dequantizeQ8_0(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 34
                let scale = readFloat16(bytes, offset)

                for j in 0..<32 {
                    let q = Int8(bitPattern: bytes[offset + 2 + j])
                    result[block * 32 + j] = Float(q) * scale
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q4_K Dequantization

    /// Q4_K: Super-block of 256 elements = 144 bytes
    /// Layout: d(2) + dmin(2) + scales(12) + qs(128)
    ///
    /// Scale encoding (12 bytes → 8 scales + 8 mins, each 6-bit):
    ///   bytes 0-3:  lower 6 bits of scales[0..3], bits 6-7 = upper 2 bits of scales[4..7]
    ///   bytes 4-7:  lower 6 bits of mins[0..3],   bits 6-7 = upper 2 bits of mins[4..7]
    ///   bytes 8-11: lower 4 bits = scales[4..7] lower 4, upper 4 bits = mins[4..7] lower 4
    private func dequantizeQ4_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 144
                let d = readFloat16(bytes, offset)
                let dmin = readFloat16(bytes, offset + 2)
                let q = offset + 4  // scales[12] start

                var scales = [Float](repeating: 0, count: 8)
                var mins = [Float](repeating: 0, count: 8)

                // Matches llama.cpp get_scale_min_k4
                for j in 0..<8 {
                    if j < 4 {
                        scales[j] = Float(bytes[q + j] & 63) * d
                        mins[j] = Float(bytes[q + j + 4] & 63) * dmin
                    } else {
                        // scale: lower 4 bits from q[j+4], upper 2 bits from q[j-4] bits 6-7
                        let sc = (bytes[q + j + 4] & 0x0F) | ((bytes[q + j - 4] >> 6) << 4)
                        // min: lower 4 bits from q[j+4] >> 4, upper 2 bits from q[j] bits 6-7
                        let mn = (bytes[q + j + 4] >> 4) | ((bytes[q + j] >> 6) << 4)
                        scales[j] = Float(sc) * d
                        mins[j] = Float(mn) * dmin
                    }
                }

                let qsOffset = offset + 16
                for subBlock in 0..<8 {
                    let sc = scales[subBlock]
                    let mn = mins[subBlock]
                    let baseIdx = block * 256 + subBlock * 32

                    for j in 0..<16 {
                        let byte = bytes[qsOffset + subBlock * 16 + j]
                        result[baseIdx + j] = sc * Float(byte & 0x0F) - mn
                        result[baseIdx + j + 16] = sc * Float((byte >> 4) & 0x0F) - mn
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q6_K Dequantization

    /// Q6_K: Super-block of 256 elements = 210 bytes
    /// Layout: ql(128) + qh(64) + scales(16) + d(2)
    ///
    /// Each element = 6-bit signed value (subtract 32 → range [-32, +31]).
    /// Lower 4 bits in ql[], upper 2 bits in qh[].
    /// qh mapping (per 128-element half, l=0..31):
    ///   qh[l] bits 0-1 → element l+0   (ql[l] low nibble)
    ///   qh[l] bits 2-3 → element l+32  (ql[l+32] low nibble)
    ///   qh[l] bits 4-5 → element l+64  (ql[l] high nibble)
    ///   qh[l] bits 6-7 → element l+96  (ql[l+32] high nibble)
    private func dequantizeQ6_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 210
                let scalesOffset = offset + 192
                let d = readFloat16(bytes, offset + 208)

                // Process two 128-element halves (matches llama.cpp outer loop)
                for half in 0..<2 {
                    let qlBase = offset + half * 64
                    let qhBase = offset + 128 + half * 32
                    let scBase = scalesOffset + half * 8
                    let outBase = block * 256 + half * 128

                    for l in 0..<32 {
                        let scIdx = l / 16

                        let q1ql = Int(bytes[qlBase + l] & 0x0F)
                        let q1qh = Int((bytes[qhBase + l] >> 0) & 3) << 4
                        let q1 = (q1ql | q1qh) - 32

                        let q2ql = Int(bytes[qlBase + l + 32] & 0x0F)
                        let q2qh = Int((bytes[qhBase + l] >> 2) & 3) << 4
                        let q2 = (q2ql | q2qh) - 32

                        let q3ql = Int(bytes[qlBase + l] >> 4)
                        let q3qh = Int((bytes[qhBase + l] >> 4) & 3) << 4
                        let q3 = (q3ql | q3qh) - 32

                        let q4ql = Int(bytes[qlBase + l + 32] >> 4)
                        let q4qh = Int((bytes[qhBase + l] >> 6) & 3) << 4
                        let q4 = (q4ql | q4qh) - 32

                        let sc0 = Float(Int8(bitPattern: bytes[scBase + scIdx]))
                        let sc2 = Float(Int8(bitPattern: bytes[scBase + scIdx + 2]))
                        let sc4 = Float(Int8(bitPattern: bytes[scBase + scIdx + 4]))
                        let sc6 = Float(Int8(bitPattern: bytes[scBase + scIdx + 6]))

                        result[outBase + l] = d * sc0 * Float(q1)
                        result[outBase + l + 32] = d * sc2 * Float(q2)
                        result[outBase + l + 64] = d * sc4 * Float(q3)
                        result[outBase + l + 96] = d * sc6 * Float(q4)
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q2_K Dequantization

    /// Q2_K: Super-block of 256 elements = 84 bytes
    /// Layout: scales(16) + qs(64) + d(2) + dmin(2)
    private func dequantizeQ2_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 84
                let scalesOffset = offset
                let qsOffset = offset + 16
                let d = readFloat16(bytes, offset + 80)
                let dmin = readFloat16(bytes, offset + 82)

                for subBlock in 0..<16 {
                    let scByte = bytes[scalesOffset + subBlock]
                    let sc = Float(scByte & 0x0F) * d
                    let mn = Float((scByte >> 4) & 0x0F) * dmin
                    let baseIdx = block * 256 + subBlock * 16

                    for j in 0..<16 {
                        let qByteIdx = qsOffset + (subBlock * 16 + j) / 4
                        let qShift = ((subBlock * 16 + j) % 4) * 2
                        let q = Int((bytes[qByteIdx] >> qShift) & 0x03)
                        result[baseIdx + j] = sc * Float(q) - mn
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q3_K Dequantization

    /// Q3_K: Super-block of 256 elements = 110 bytes
    /// Layout: hmask(32) + qs(64) + scales(12) + d(2)
    ///
    /// Scale encoding (12 bytes → 16 6-bit signed values):
    /// Treated as three uint32 words. Each output byte = 4-bit lower | 2-bit upper << 4.
    /// Then subtract 32 for signed range.
    private func dequantizeQ3_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 110
                let hmaskOffset = offset
                let qsOffset = offset + 32
                let scalesOffset = offset + 96
                let d = readFloat16(bytes, offset + 108)

                // Unpack 12 bytes → 16 6-bit scales (matches llama.cpp kmask approach)
                var scales = [Int](repeating: 0, count: 16)
                unpackQ3KScales(bytes: bytes, offset: scalesOffset, scales: &scales)

                for subBlock in 0..<16 {
                    let sc = d * Float(scales[subBlock] - 32)
                    let baseIdx = block * 256 + subBlock * 16

                    for j in 0..<16 {
                        let elemIdx = subBlock * 16 + j
                        let qByteIdx = qsOffset + elemIdx / 4
                        let qShift = (elemIdx % 4) * 2
                        let q2 = Int((bytes[qByteIdx] >> qShift) & 0x03)

                        let hmBit = Int((bytes[hmaskOffset + elemIdx / 8] >> (elemIdx % 8)) & 1)
                        let q = q2 | (hmBit << 2)

                        result[baseIdx + j] = sc * Float(q - 4)
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    /// Unpack 12-byte Q3_K scale section into 16 6-bit values.
    ///
    /// Layout mirrors llama.cpp's uint32 kmask approach:
    ///   aux[0] bytes: low nibble = scales[0..3], high nibble = scales[8..11]
    ///   aux[1] bytes: low nibble = scales[4..7], high nibble = scales[12..15]
    ///   aux[2] bytes: 2-bit pairs = upper bits for all 16 scales
    private func unpackQ3KScales(bytes: UnsafeBufferPointer<UInt8>, offset: Int, scales: inout [Int]) {
        // Read raw bytes as 4 groups of 4 bytes
        var raw = [UInt8](repeating: 0, count: 16)
        for i in 0..<12 {
            raw[i] = bytes[offset + i]
        }

        // Treat as uint32 words (little-endian)
        let aux0_bytes = (0..<4).map { raw[$0] }
        let aux1_bytes = (4..<8).map { raw[$0] }
        let tmp_bytes = (8..<12).map { raw[$0] }

        // scales[0..3]: low nibble of aux[0] | bits 0-1 of tmp
        for i in 0..<4 {
            let lo4 = Int(aux0_bytes[i] & 0x0F)
            let hi2 = Int((tmp_bytes[i] >> 0) & 0x03) << 4
            scales[i] = lo4 | hi2
        }
        // scales[4..7]: low nibble of aux[1] | bits 2-3 of tmp
        for i in 0..<4 {
            let lo4 = Int(aux1_bytes[i] & 0x0F)
            let hi2 = Int((tmp_bytes[i] >> 2) & 0x03) << 4
            scales[4 + i] = lo4 | hi2
        }
        // scales[8..11]: high nibble of aux[0] | bits 4-5 of tmp
        for i in 0..<4 {
            let lo4 = Int((aux0_bytes[i] >> 4) & 0x0F)
            let hi2 = Int((tmp_bytes[i] >> 4) & 0x03) << 4
            scales[8 + i] = lo4 | hi2
        }
        // scales[12..15]: high nibble of aux[1] | bits 6-7 of tmp
        for i in 0..<4 {
            let lo4 = Int((aux1_bytes[i] >> 4) & 0x0F)
            let hi2 = Int((tmp_bytes[i] >> 6) & 0x03) << 4
            scales[12 + i] = lo4 | hi2
        }
    }

    // MARK: - Q5_K Dequantization

    /// Q5_K: Super-block of 256 elements = 176 bytes
    /// Layout: d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
    ///
    /// Scale encoding: same 12-byte format as Q4_K (get_scale_min_k4).
    private func dequantizeQ5_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 176
                let dVal = readFloat16(bytes, offset)
                let dmin = readFloat16(bytes, offset + 2)
                let q = offset + 4  // scales[12] start
                let qhOffset = offset + 16
                let qsOffset = offset + 48

                // Decode scales using get_scale_min_k4 pattern (same as Q4_K)
                var scales = [Float](repeating: 0, count: 8)
                var mins = [Float](repeating: 0, count: 8)
                for j in 0..<8 {
                    if j < 4 {
                        scales[j] = Float(bytes[q + j] & 63) * dVal
                        mins[j] = Float(bytes[q + j + 4] & 63) * dmin
                    } else {
                        let sc = (bytes[q + j + 4] & 0x0F) | ((bytes[q + j - 4] >> 6) << 4)
                        let mn = (bytes[q + j + 4] >> 4) | ((bytes[q + j] >> 6) << 4)
                        scales[j] = Float(sc) * dVal
                        mins[j] = Float(mn) * dmin
                    }
                }

                for subBlock in 0..<8 {
                    let sc = scales[subBlock]
                    let mn = mins[subBlock]
                    let baseIdx = block * 256 + subBlock * 32

                    for j in 0..<16 {
                        let byte = bytes[qsOffset + subBlock * 16 + j]
                        let qhBit0 = Int((bytes[qhOffset + j] >> subBlock) & 1) << 4
                        let qhBit1 = Int((bytes[qhOffset + j + 16] >> subBlock) & 1) << 4

                        result[baseIdx + j] = sc * Float(Int(byte & 0x0F) | qhBit0) - mn
                        result[baseIdx + j + 16] = sc * Float(Int((byte >> 4) & 0x0F) | qhBit1) - mn
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Helpers

    /// Read a little-endian float16 from a byte buffer at the given offset.
    private func readFloat16(_ bytes: UnsafeBufferPointer<UInt8>, _ offset: Int) -> Float {
        let bits = UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
        return Float(Float16(bitPattern: bits))
    }
}
