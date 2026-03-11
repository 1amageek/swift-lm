import Foundation
import MLX
import MLXFast
import GGUFParser

/// GPU-accelerated GGUF → MLX affine quantization repacking.
///
/// Each kernel converts raw GGUF bytes directly to MLX's native packed format
/// on the GPU, avoiding CPU-bound bit manipulation for large tensors.
/// One GPU thread processes one output group (16 or 32 elements depending on type).
struct GGUFGPUPacker {

    /// Minimum super-block count to justify GPU dispatch overhead.
    static let gpuThreshold = 2048

    /// Try GPU-accelerated packing. Returns nil if type is unsupported or tensor too small.
    static func tryPack(
        qtype: GGUFQuantizationType, data: Data, shape: [Int]
    ) -> ConvertedTensor? {
        let totalElements = shape.reduce(1, *)

        switch qtype {
        case .q4_0:
            let blockCount = totalElements / 32
            guard blockCount >= gpuThreshold else { return nil }
            return packGPU(data: data, shape: shape, kernel: q4_0Kernel,
                           groupCount: blockCount, packedPerGroup: 4,
                           groupSize: 32, bits: 4)
        case .q4_1:
            let blockCount = totalElements / 32
            guard blockCount >= gpuThreshold else { return nil }
            return packGPU(data: data, shape: shape, kernel: q4_1Kernel,
                           groupCount: blockCount, packedPerGroup: 4,
                           groupSize: 32, bits: 4)
        case .q8_0:
            let blockCount = totalElements / 32
            guard blockCount >= gpuThreshold else { return nil }
            return packGPU(data: data, shape: shape, kernel: q8_0Kernel,
                           groupCount: blockCount, packedPerGroup: 8,
                           groupSize: 32, bits: 8)
        case .q8_1:
            let blockCount = totalElements / 32
            guard blockCount >= gpuThreshold else { return nil }
            return packGPU(data: data, shape: shape, kernel: q8_1Kernel,
                           groupCount: blockCount, packedPerGroup: 8,
                           groupSize: 32, bits: 8)
        case .q5_0:
            let blockCount = totalElements / 32
            guard blockCount >= gpuThreshold else { return nil }
            return packGPU(data: data, shape: shape, kernel: q5_0Kernel,
                           groupCount: blockCount, packedPerGroup: 5,
                           groupSize: 32, bits: 5)
        case .q5_1:
            let blockCount = totalElements / 32
            guard blockCount >= gpuThreshold else { return nil }
            return packGPU(data: data, shape: shape, kernel: q5_1Kernel,
                           groupCount: blockCount, packedPerGroup: 5,
                           groupSize: 32, bits: 5)
        case .q4_K:
            let sbCount = totalElements / 256
            guard sbCount >= gpuThreshold / 8 else { return nil }
            let groupCount = totalElements / 32
            return packGPU(data: data, shape: shape, kernel: q4_KKernel,
                           groupCount: groupCount, packedPerGroup: 4,
                           groupSize: 32, bits: 4)
        case .q5_K:
            let sbCount = totalElements / 256
            guard sbCount >= gpuThreshold / 8 else { return nil }
            let groupCount = totalElements / 32
            return packGPU(data: data, shape: shape, kernel: q5_KKernel,
                           groupCount: groupCount, packedPerGroup: 5,
                           groupSize: 32, bits: 5)
        case .q6_K:
            let sbCount = totalElements / 256
            guard sbCount >= gpuThreshold / 8 else { return nil }
            let groupCount = totalElements / 32
            return packGPU(data: data, shape: shape, kernel: q6_KKernel,
                           groupCount: groupCount, packedPerGroup: 6,
                           groupSize: 32, bits: 6)
        case .q8_K:
            let sbCount = totalElements / 256
            guard sbCount >= gpuThreshold / 8 else { return nil }
            let groupCount = totalElements / 32
            return packGPU(data: data, shape: shape, kernel: q8_KKernel,
                           groupCount: groupCount, packedPerGroup: 8,
                           groupSize: 32, bits: 8)
        default:
            return nil
        }
    }

    // MARK: - Generic GPU dispatch

    private static func packGPU(
        data: Data, shape: [Int],
        kernel: MLXFast.MLXFastKernel,
        groupCount: Int, packedPerGroup: Int,
        groupSize: Int, bits: Int
    ) -> ConvertedTensor {
        let rawArray = MLXArray(data, [data.count], type: UInt8.self)
        let packedCount = groupCount * packedPerGroup

        let results = kernel(
            [rawArray],
            grid: (groupCount, 1, 1),
            threadGroup: (min(256, groupCount), 1, 1),
            outputShapes: [[packedCount], [groupCount], [groupCount]],
            outputDTypes: [.uint32, .float32, .float32],
            initValue: 0
        )

        let packedColumns = shape.last! * bits / 32
        let weightShape = shape.dropLast().map { $0 } + [packedColumns]
        let scaleShape = shape.dropLast().map { $0 } + [shape.last! / groupSize]

        return .quantized(
            weight: results[0].reshaped(weightShape),
            scales: results[1].reshaped(scaleShape).asType(.float16),
            biases: results[2].reshaped(scaleShape).asType(.float16),
            groupSize: groupSize, bits: bits
        )
    }

    // MARK: - Metal kernel header: float16/float32 read helpers

    private static let metalHeader = """
        inline float read_f16(const device uint8_t* raw, uint offset) {
            uint16_t bits = uint16_t(raw[offset]) | (uint16_t(raw[offset + 1]) << 8);
            return float(as_type<half>(bits));
        }
        inline float read_f32(const device uint8_t* raw, uint offset) {
            uint32_t bits = uint32_t(raw[offset]) | (uint32_t(raw[offset+1]) << 8)
                          | (uint32_t(raw[offset+2]) << 16) | (uint32_t(raw[offset+3]) << 24);
            return as_type<float>(bits);
        }
        """

    // MARK: - Kernel instances (lazy, compiled once)

    /// Q4_0: 18 bytes/block, 32 elements, groupSize=32, 4-bit
    private static let q4_0Kernel = MLXFast.metalKernel(
        name: "gguf_repack_q4_0",
        inputNames: ["raw"],
        outputNames: ["packed", "scales", "biases"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint offset = gid * 18;
            float scale = read_f16(raw, offset);
            scales[gid] = scale;
            biases[gid] = -8.0f * scale;

            uint pBase = gid * 4;
            uint qs = offset + 2;
            for (uint k = 0; k < 2; k++) {
                uint32_t word = 0;
                for (uint j = 0; j < 8; j++) {
                    word |= uint32_t(raw[qs + k * 8 + j] & 0x0F) << (j * 4);
                }
                packed[pBase + k] = word;
            }
            for (uint k = 0; k < 2; k++) {
                uint32_t word = 0;
                for (uint j = 0; j < 8; j++) {
                    word |= uint32_t(raw[qs + k * 8 + j] >> 4) << (j * 4);
                }
                packed[pBase + 2 + k] = word;
            }
        """,
        header: metalHeader,
        ensureRowContiguous: false
    )

    /// Q4_1: 20 bytes/block, 32 elements, groupSize=32, 4-bit
    private static let q4_1Kernel = MLXFast.metalKernel(
        name: "gguf_repack_q4_1",
        inputNames: ["raw"],
        outputNames: ["packed", "scales", "biases"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint offset = gid * 20;
            scales[gid] = read_f16(raw, offset);
            biases[gid] = read_f16(raw, offset + 2);

            uint pBase = gid * 4;
            uint qs = offset + 4;
            for (uint k = 0; k < 2; k++) {
                uint32_t word = 0;
                for (uint j = 0; j < 8; j++) {
                    word |= uint32_t(raw[qs + k * 8 + j] & 0x0F) << (j * 4);
                }
                packed[pBase + k] = word;
            }
            for (uint k = 0; k < 2; k++) {
                uint32_t word = 0;
                for (uint j = 0; j < 8; j++) {
                    word |= uint32_t(raw[qs + k * 8 + j] >> 4) << (j * 4);
                }
                packed[pBase + 2 + k] = word;
            }
        """,
        header: metalHeader,
        ensureRowContiguous: false
    )

    /// Q8_0: 34 bytes/block, 32 elements, groupSize=32, 8-bit
    private static let q8_0Kernel = MLXFast.metalKernel(
        name: "gguf_repack_q8_0",
        inputNames: ["raw"],
        outputNames: ["packed", "scales", "biases"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint offset = gid * 34;
            float scale = read_f16(raw, offset);
            scales[gid] = scale;
            biases[gid] = -128.0f * scale;

            uint pBase = gid * 8;
            uint qs = offset + 2;
            for (uint k = 0; k < 8; k++) {
                uint32_t b0 = uint32_t(raw[qs + k * 4 + 0] ^ 0x80u);
                uint32_t b1 = uint32_t(raw[qs + k * 4 + 1] ^ 0x80u);
                uint32_t b2 = uint32_t(raw[qs + k * 4 + 2] ^ 0x80u);
                uint32_t b3 = uint32_t(raw[qs + k * 4 + 3] ^ 0x80u);
                packed[pBase + k] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            }
        """,
        header: metalHeader,
        ensureRowContiguous: false
    )

    /// Q8_1: 36 bytes/block, 32 elements, groupSize=32, 8-bit
    /// Layout: d(f16) + s(f16, sum for dot product, unused) + qs(int8[32])
    private static let q8_1Kernel = MLXFast.metalKernel(
        name: "gguf_repack_q8_1",
        inputNames: ["raw"],
        outputNames: ["packed", "scales", "biases"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint offset = gid * 36;
            float d = read_f16(raw, offset);
            scales[gid] = d;
            biases[gid] = -128.0f * d;

            uint pBase = gid * 8;
            uint qs = offset + 4;
            for (uint k = 0; k < 8; k++) {
                uint32_t b0 = uint32_t(raw[qs + k * 4 + 0] ^ 0x80u);
                uint32_t b1 = uint32_t(raw[qs + k * 4 + 1] ^ 0x80u);
                uint32_t b2 = uint32_t(raw[qs + k * 4 + 2] ^ 0x80u);
                uint32_t b3 = uint32_t(raw[qs + k * 4 + 3] ^ 0x80u);
                packed[pBase + k] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            }
        """,
        header: metalHeader,
        ensureRowContiguous: false
    )

    /// Q5_0: 22 bytes/block, 32 elements, groupSize=32, 5-bit
    private static let q5_0Kernel = MLXFast.metalKernel(
        name: "gguf_repack_q5_0",
        inputNames: ["raw"],
        outputNames: ["packed", "scales", "biases"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint offset = gid * 22;
            float scale = read_f16(raw, offset);
            scales[gid] = scale;
            biases[gid] = -16.0f * scale;

            uint qhOffset = offset + 2;
            uint32_t qhBits = uint32_t(raw[qhOffset]) | (uint32_t(raw[qhOffset+1]) << 8)
                            | (uint32_t(raw[qhOffset+2]) << 16) | (uint32_t(raw[qhOffset+3]) << 24);
            uint qsOffset = offset + 6;

            uint pBase = gid * 5;
            uint32_t w[5] = {0, 0, 0, 0, 0};
            for (uint i = 0; i < 16; i++) {
                uint8_t byte = raw[qsOffset + i];
                uint32_t low  = uint32_t(byte & 0x0F) | (((qhBits >> i) & 1) << 4);
                uint32_t high = uint32_t(byte >> 4)    | (((qhBits >> (i + 16)) & 1) << 4);

                uint bi0 = i * 5;
                uint wi0 = bi0 / 32;
                uint bo0 = bi0 % 32;
                w[wi0] |= low << bo0;
                if (bo0 + 5 > 32) w[wi0 + 1] |= low >> (32 - bo0);

                uint bi1 = (i + 16) * 5;
                uint wi1 = bi1 / 32;
                uint bo1 = bi1 % 32;
                w[wi1] |= high << bo1;
                if (bo1 + 5 > 32) w[wi1 + 1] |= high >> (32 - bo1);
            }
            for (uint i = 0; i < 5; i++) packed[pBase + i] = w[i];
        """,
        header: metalHeader,
        ensureRowContiguous: false
    )

    /// Q5_1: 24 bytes/block, 32 elements, groupSize=32, 5-bit
    private static let q5_1Kernel = MLXFast.metalKernel(
        name: "gguf_repack_q5_1",
        inputNames: ["raw"],
        outputNames: ["packed", "scales", "biases"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint offset = gid * 24;
            scales[gid] = read_f16(raw, offset);
            biases[gid] = read_f16(raw, offset + 2);

            uint qhOffset = offset + 4;
            uint32_t qhBits = uint32_t(raw[qhOffset]) | (uint32_t(raw[qhOffset+1]) << 8)
                            | (uint32_t(raw[qhOffset+2]) << 16) | (uint32_t(raw[qhOffset+3]) << 24);
            uint qsOffset = offset + 8;

            uint pBase = gid * 5;
            uint32_t w[5] = {0, 0, 0, 0, 0};
            for (uint i = 0; i < 16; i++) {
                uint8_t byte = raw[qsOffset + i];
                uint32_t low  = uint32_t(byte & 0x0F) | (((qhBits >> i) & 1) << 4);
                uint32_t high = uint32_t(byte >> 4)    | (((qhBits >> (i + 16)) & 1) << 4);

                uint bi0 = i * 5;
                uint wi0 = bi0 / 32;
                uint bo0 = bi0 % 32;
                w[wi0] |= low << bo0;
                if (bo0 + 5 > 32) w[wi0 + 1] |= low >> (32 - bo0);

                uint bi1 = (i + 16) * 5;
                uint wi1 = bi1 / 32;
                uint bo1 = bi1 % 32;
                w[wi1] |= high << bo1;
                if (bo1 + 5 > 32) w[wi1 + 1] |= high >> (32 - bo1);
            }
            for (uint i = 0; i < 5; i++) packed[pBase + i] = w[i];
        """,
        header: metalHeader,
        ensureRowContiguous: false
    )

    /// Q4_K: 144 bytes/super-block, 256 elements, 8 groups of 32, groupSize=32, 4-bit
    private static let q4_KKernel = MLXFast.metalKernel(
        name: "gguf_repack_q4_k",
        inputNames: ["raw"],
        outputNames: ["packed", "scales", "biases"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint block = gid / 8;
            uint localGroup = gid % 8;
            uint chunk = localGroup / 2;
            bool isHigh = (localGroup & 1) != 0;

            uint offset = block * 144;
            float d = read_f16(raw, offset);
            float dmin = read_f16(raw, offset + 2);
            uint q = offset + 4;

            float sc, mn;
            if (localGroup < 4) {
                sc = float(raw[q + localGroup] & 63) * d;
                mn = float(raw[q + localGroup + 4] & 63) * dmin;
            } else {
                uint j = localGroup;
                uint scVal = (raw[q + j + 4] & 0x0Fu) | (uint(raw[q + j - 4] >> 6) << 4);
                uint mnVal = (raw[q + j + 4] >> 4)     | (uint(raw[q + j]     >> 6) << 4);
                sc = float(scVal) * d;
                mn = float(mnVal) * dmin;
            }
            scales[gid] = sc;
            biases[gid] = -mn;

            uint qsBase = offset + 16 + chunk * 32;
            uint pBase = gid * 4;
            for (uint k = 0; k < 4; k++) {
                uint32_t word = 0;
                for (uint j = 0; j < 8; j++) {
                    uint8_t byte = raw[qsBase + k * 8 + j];
                    uint32_t nibble = isHigh ? uint32_t(byte >> 4) : uint32_t(byte & 0x0F);
                    word |= nibble << (j * 4);
                }
                packed[pBase + k] = word;
            }
        """,
        header: metalHeader,
        ensureRowContiguous: false
    )

    /// Q5_K: 176 bytes/super-block, 256 elements, 8 groups of 32, groupSize=32, 5-bit
    private static let q5_KKernel = MLXFast.metalKernel(
        name: "gguf_repack_q5_k",
        inputNames: ["raw"],
        outputNames: ["packed", "scales", "biases"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint block = gid / 8;
            uint localGroup = gid % 8;
            uint chunk = localGroup / 2;
            bool isHigh = (localGroup & 1) != 0;

            uint offset = block * 176;
            float d = read_f16(raw, offset);
            float dmin = read_f16(raw, offset + 2);
            uint q = offset + 4;

            float sc, mn;
            if (localGroup < 4) {
                sc = float(raw[q + localGroup] & 63) * d;
                mn = float(raw[q + localGroup + 4] & 63) * dmin;
            } else {
                uint j = localGroup;
                uint scVal = (raw[q + j + 4] & 0x0Fu) | (uint(raw[q + j - 4] >> 6) << 4);
                uint mnVal = (raw[q + j + 4] >> 4)     | (uint(raw[q + j]     >> 6) << 4);
                sc = float(scVal) * d;
                mn = float(mnVal) * dmin;
            }
            scales[gid] = sc;
            biases[gid] = -mn;

            uint qhOffset = offset + 16;
            uint qsBase = offset + 48 + chunk * 32;
            uint pBase = gid * 5;
            uint32_t w[5] = {0, 0, 0, 0, 0};

            for (uint l = 0; l < 32; l++) {
                uint8_t byte = raw[qsBase + l];
                uint8_t qh = raw[qhOffset + l];
                uint32_t val;
                if (isHigh) {
                    val = uint32_t((byte >> 4) & 0x0F) | (uint32_t((qh >> (chunk * 2 + 1)) & 1) << 4);
                } else {
                    val = uint32_t(byte & 0x0F) | (uint32_t((qh >> (chunk * 2)) & 1) << 4);
                }
                uint bi = l * 5;
                uint wi = bi / 32;
                uint bo = bi % 32;
                w[wi] |= val << bo;
                if (bo + 5 > 32) w[wi + 1] |= val >> (32 - bo);
            }
            for (uint i = 0; i < 5; i++) packed[pBase + i] = w[i];
        """,
        header: metalHeader,
        ensureRowContiguous: false
    )

    /// Q6_K: 210 bytes/super-block, 256 elements → 8 groups of 32, groupSize=32, 6-bit
    /// 1 thread decodes 2 adjacent 16-element sub-groups, finds min/max, re-quantizes to 32-element group
    private static let q6_KKernel = MLXFast.metalKernel(
        name: "gguf_repack_q6k",
        inputNames: ["raw"],
        outputNames: ["packed", "scales", "biases"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint block = gid / 8;
            uint pairIdx = gid % 8;

            uint offset = block * 210;
            float d = read_f16(raw, offset + 208);

            // Decode 32 float values from 2 adjacent 16-element sub-groups
            float vals[32];
            for (uint s = 0; s < 2; s++) {
                uint sg = pairIdx * 2 + s;
                uint half_idx = sg / 8;
                uint localGroup = sg % 8;

                uint qlBase = offset + half_idx * 64;
                uint qhBase = offset + 128 + half_idx * 32;
                uint scBase = offset + 192 + half_idx * 8;

                int8_t subScale = as_type<int8_t>(raw[scBase + localGroup]);
                float scale = d * float(subScale);
                float bias = -32.0f * scale;

                uint lStart = (localGroup & 1) << 4;
                uint qlAdd = ((localGroup >> 1) & 1) << 5;
                bool useLow = (localGroup & 4) == 0;
                uint qhShift = localGroup & 6;

                for (uint i = 0; i < 16; i++) {
                    uint l = lStart + i;
                    uint8_t qlByte = raw[qlBase + qlAdd + l];
                    uint32_t ql = useLow ? uint32_t(qlByte & 0x0F) : uint32_t(qlByte >> 4);
                    uint32_t qh = uint32_t((raw[qhBase + l] >> qhShift) & 3) << 4;
                    vals[s * 16 + i] = float(ql | qh) * scale + bias;
                }
            }

            // Find min/max of 32 values
            float vmin = vals[0], vmax = vals[0];
            for (uint i = 1; i < 32; i++) {
                vmin = min(vmin, vals[i]);
                vmax = max(vmax, vals[i]);
            }
            float range = vmax - vmin;
            float newScale = range > 0.0f ? range / 63.0f : 0.0f;
            scales[gid] = newScale;
            biases[gid] = vmin;

            // Re-quantize and pack 32 6-bit values into 6 UInt32
            uint pBase = gid * 6;
            uint32_t w[6] = {0, 0, 0, 0, 0, 0};
            float invScale = newScale > 0.0f ? 1.0f / newScale : 0.0f;
            for (uint i = 0; i < 32; i++) {
                int q = int(round((vals[i] - vmin) * invScale));
                q = clamp(q, 0, 63);
                uint bi = i * 6;
                uint wi = bi / 32;
                uint bo = bi % 32;
                w[wi] |= uint32_t(q) << bo;
                if (bo + 6 > 32) w[wi + 1] |= uint32_t(q) >> (32 - bo);
            }
            for (uint i = 0; i < 6; i++) packed[pBase + i] = w[i];
        """,
        header: metalHeader,
        ensureRowContiguous: false
    )

    /// Q8_K: 292 bytes/super-block, 256 elements, 8 groups of 32, groupSize=32, 8-bit
    /// Layout: d(f32, 4) + qs(int8, 256) + bsums(int16, 16) = 292 bytes
    private static let q8_KKernel = MLXFast.metalKernel(
        name: "gguf_repack_q8_k",
        inputNames: ["raw"],
        outputNames: ["packed", "scales", "biases"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint block = gid / 8;
            uint localGroup = gid % 8;

            uint offset = block * 292;
            float d = read_f32(raw, offset);
            scales[gid] = d;
            biases[gid] = -128.0f * d;

            uint qsBase = offset + 4 + localGroup * 32;
            uint pBase = gid * 8;
            for (uint k = 0; k < 8; k++) {
                uint32_t b0 = uint32_t(raw[qsBase + k * 4 + 0] ^ 0x80u);
                uint32_t b1 = uint32_t(raw[qsBase + k * 4 + 1] ^ 0x80u);
                uint32_t b2 = uint32_t(raw[qsBase + k * 4 + 2] ^ 0x80u);
                uint32_t b3 = uint32_t(raw[qsBase + k * 4 + 3] ^ 0x80u);
                packed[pBase + k] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            }
        """,
        header: metalHeader,
        ensureRowContiguous: false
    )
}
