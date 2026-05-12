extension MetalSourceGenerator {
/// Generate quantized GEMM (Q8 group, multi-row prefill sequence).
///
/// Signature matches `generateQuantizedGEMM_Q4` exactly so dispatch builder
/// can route any Q* scheme through the same buffer-binding convention.
///
/// - Buffer 0: input  (F16 decode / F32 prefill)
/// - Buffer 1: packed weights (uchar, 36/68 bytes per block for Q8G32/Q8G64)
/// - Buffer 2: output (F16 decode / F32 prefill)
/// - Buffer 3: inputDimension   (uint32)
/// - Buffer 4: outputDimension  (uint32)
/// - Buffer 5: sequenceLength   (uint32)
/// - Buffer 6: inputRowStride   (uint32)
/// - Buffer 7: outputRowStride  (uint32)
///
/// Block layout (per MLX Q8 affine):
/// ```
/// ┌──────────┬──────────┬──────────────────────────┐
/// │scale (2B)│ zero (2B)│ packed quants (groupSize B) │
/// └──────────┴──────────┴──────────────────────────┘
/// ```
/// Each quantized value is stored as uint8 (0..255). Dequant: `w = scale*q + zero`.
public static func generateQuantizedGEMM_Q8(
    name: String,
    bufferPrecision: BufferPrecision,
    groupSize: Int
) -> String {
    let bt = bufferPrecision.metalType
    let bytesPerBlock = 4 + groupSize  // scale(f16) + zero(f16) + uint8 × groupSize
    let tileElements = max(groupSize * 2, 256)
    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight     [[buffer(1)]],
        device \(bt)* output            [[buffer(2)]],
        constant uint& inputDimension  [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        constant uint& sequenceLength  [[buffer(5)]],
        constant uint& inputRowStride  [[buffer(6)]],
        constant uint& outputRowStride [[buffer(7)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint GROUP_SIZE = \(groupSize);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint rowsPerThreadgroup = 2;
        const uint THREADS_PER_THREADGROUP = SIMD_WIDTH * rowsPerThreadgroup;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        if (seqPos >= sequenceLength) return;
        const bool active = row < outputDimension;

        const uint blocksPerRow = inputDimension / GROUP_SIZE;
        const uint safeRow = active ? row : 0;
        device const uchar* rowBase = weight + safeRow * blocksPerRow * BYTES_PER_BLOCK;
        device const \(bt)* inputRow = input + seqPos * inputRowStride;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = inputRow[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / GROUP_SIZE;
            const uint blockCount = tileCount / GROUP_SIZE;
            if (active) {
                for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                    device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                    float blockScale = float(*(device const half*)(block));
                    float blockZero = float(*(device const half*)(block + 2));
                    device const uchar* quantized = block + 4;
                    const uint tileOffset = localBlock * GROUP_SIZE;
                    for (uint i = tiisg; i < GROUP_SIZE; i += SIMD_WIDTH) {
                        float w = blockScale * float(quantized[i]) + blockZero;
                        sum += w * float(inputTile[tileOffset + i]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (active && tiisg == 0) output[seqPos * outputRowStride + row] = \(bt)(sum);
    }
    """
}

/// Generate quantized GEMM (Q4 group, prefill sequence).
public static func generateQuantizedGEMM_Q4(
    name: String,
    bufferPrecision: BufferPrecision,
    groupSize: Int
) -> String {
    let bt = bufferPrecision.metalType
    let weightsPerBlock = groupSize
    let bytesPerBlock = 4 + groupSize / 2  // scale(f16) + zero(f16) + nibbles
    let tileElements = max(groupSize * 2, 256)
    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight     [[buffer(1)]],
        device \(bt)* output            [[buffer(2)]],
        constant uint& inputDimension  [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        constant uint& sequenceLength  [[buffer(5)]],
        constant uint& inputRowStride  [[buffer(6)]],
        constant uint& outputRowStride [[buffer(7)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint rowsPerThreadgroup = 2;
        const uint THREADS_PER_THREADGROUP = SIMD_WIDTH * rowsPerThreadgroup;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        if (seqPos >= sequenceLength) return;
        const bool active = row < outputDimension;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        const uint safeRow = active ? row : 0;
        device const uchar* rowBase = weight + safeRow * blocksPerRow * BYTES_PER_BLOCK;
        device const \(bt)* inputRow = input + seqPos * inputRowStride;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = inputRow[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            if (active) {
                for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                    device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                    float blockScale = float(*(device const half*)(block));
                    float blockZero = float(*(device const half*)(block + 2));
                    device const uchar* nibbles = block + 4;
                    const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                    for (uint i = tiisg; i < WEIGHTS_PER_BLOCK / 2; i += SIMD_WIDTH) {
                        uchar packed = nibbles[i];
                        const uint inputOffset = tileOffset + i * 2;
                        float w0 = float(packed & 0x0F) * blockScale + blockZero;
                        float w1 = float(packed >> 4) * blockScale + blockZero;
                        sum += w0 * float(inputTile[inputOffset]);
                        sum += w1 * float(inputTile[inputOffset + 1]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (active && tiisg == 0) output[seqPos * outputRowStride + row] = \(bt)(sum);
    }
    """
}

// MARK: - Batched Quantized GEMM (Prefill)

/// Generate batched Q4 GEMM kernel for 2 projections sharing the same input.
/// Combines Q4 block unpacking (from generateQuantizedGEMM_Q4) with
/// multi-projection routing (from generateBatchedGEMV3).
///
/// Grid: (ceil(totalOutputDim/2), seqLen, 1)
/// Threadgroup: (SIMD_WIDTH * 2, 1, 1)
public static func generateBatchedQuantizedGEMM_Q4_2(
    name: String,
    bufferPrecision: BufferPrecision,
    groupSize: Int
) -> String {
    let bt = bufferPrecision.metalType
    let weightsPerBlock = groupSize
    let bytesPerBlock = 4 + groupSize / 2
    let tileElements = max(groupSize * 2, 256)
    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight0    [[buffer(1)]],
        device const uchar* weight1    [[buffer(2)]],
        device \(bt)* output0           [[buffer(3)]],
        device \(bt)* output1           [[buffer(4)]],
        constant uint& inputDimension  [[buffer(5)]],
        constant uint& outputDim0      [[buffer(6)]],
        constant uint& outputDim1      [[buffer(7)]],
        constant uint& sequenceLength  [[buffer(8)]],
        constant uint& inputRowStride  [[buffer(9)]],
        constant uint& outputRowStride [[buffer(10)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint rowsPerThreadgroup = 2;
        const uint THREADS_PER_THREADGROUP = SIMD_WIDTH * rowsPerThreadgroup;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint globalRow = gid.x * rowsPerThreadgroup + sgitg;
        const uint totalRows = outputDim0 + outputDim1;
        const uint seqPos = gid.y;
        if (seqPos >= sequenceLength) return;
        const bool active = globalRow < totalRows;

        device const uchar* weight = weight0;
        device \(bt)* output = output0;
        uint localRow = 0;
        if (globalRow < outputDim0) {
            weight = weight0; output = output0; localRow = globalRow;
        } else if (globalRow < totalRows) {
            weight = weight1; output = output1; localRow = globalRow - outputDim0;
        }

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + localRow * blocksPerRow * BYTES_PER_BLOCK;
        device const \(bt)* inputRow = input + seqPos * inputRowStride;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = inputRow[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            if (active) {
                for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                    device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                    float blockScale = float(*(device const half*)(block));
                    float blockZero = float(*(device const half*)(block + 2));
                    device const uchar* nibbles = block + 4;
                    const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                    for (uint i = tiisg; i < WEIGHTS_PER_BLOCK / 2; i += SIMD_WIDTH) {
                        uchar packed = nibbles[i];
                        const uint inputOffset = tileOffset + i * 2;
                        float w0 = float(packed & 0x0F) * blockScale + blockZero;
                        float w1 = float(packed >> 4) * blockScale + blockZero;
                        sum += w0 * float(inputTile[inputOffset]);
                        sum += w1 * float(inputTile[inputOffset + 1]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (active && tiisg == 0) output[seqPos * outputRowStride + localRow] = \(bt)(sum);
    }
    """
}

/// Generate batched Q4 GEMM kernel for 3 projections sharing the same input.
public static func generateBatchedQuantizedGEMM_Q4_3(
    name: String,
    bufferPrecision: BufferPrecision,
    groupSize: Int
) -> String {
    let bt = bufferPrecision.metalType
    let weightsPerBlock = groupSize
    let bytesPerBlock = 4 + groupSize / 2
    let tileElements = max(groupSize * 2, 256)
    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight0    [[buffer(1)]],
        device const uchar* weight1    [[buffer(2)]],
        device const uchar* weight2    [[buffer(3)]],
        device \(bt)* output0           [[buffer(4)]],
        device \(bt)* output1           [[buffer(5)]],
        device \(bt)* output2           [[buffer(6)]],
        constant uint& inputDimension  [[buffer(7)]],
        constant uint& outputDim0      [[buffer(8)]],
        constant uint& outputDim1      [[buffer(9)]],
        constant uint& outputDim2      [[buffer(10)]],
        constant uint& sequenceLength  [[buffer(11)]],
        constant uint& inputRowStride  [[buffer(12)]],
        constant uint& outputRowStride [[buffer(13)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint rowsPerThreadgroup = 2;
        const uint THREADS_PER_THREADGROUP = SIMD_WIDTH * rowsPerThreadgroup;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint globalRow = gid.x * rowsPerThreadgroup + sgitg;
        const uint totalRows = outputDim0 + outputDim1 + outputDim2;
        const uint seqPos = gid.y;
        if (seqPos >= sequenceLength) return;
        const bool active = globalRow < totalRows;

        device const uchar* weight = weight0;
        device \(bt)* output = output0;
        uint localRow = 0;
        if (globalRow < outputDim0) {
            weight = weight0; output = output0; localRow = globalRow;
        } else if (globalRow < outputDim0 + outputDim1) {
            weight = weight1; output = output1; localRow = globalRow - outputDim0;
        } else if (globalRow < totalRows) {
            weight = weight2; output = output2; localRow = globalRow - outputDim0 - outputDim1;
        }

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + localRow * blocksPerRow * BYTES_PER_BLOCK;
        device const \(bt)* inputRow = input + seqPos * inputRowStride;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = inputRow[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            if (active) {
                for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                    device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                    float blockScale = float(*(device const half*)(block));
                    float blockZero = float(*(device const half*)(block + 2));
                    device const uchar* nibbles = block + 4;
                    const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                    for (uint i = tiisg; i < WEIGHTS_PER_BLOCK / 2; i += SIMD_WIDTH) {
                        uchar packed = nibbles[i];
                        const uint inputOffset = tileOffset + i * 2;
                        float w0 = float(packed & 0x0F) * blockScale + blockZero;
                        float w1 = float(packed >> 4) * blockScale + blockZero;
                        sum += w0 * float(inputTile[inputOffset]);
                        sum += w1 * float(inputTile[inputOffset + 1]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (active && tiisg == 0) output[seqPos * outputRowStride + localRow] = \(bt)(sum);
    }
    """
}

/// Stable kernel name for a format's dequant→BF16 prefill path.
///
/// Uses the generic `dequant_q{bits}_g{group}_bf16` form produced by
/// `generateUnifiedDequantToBFloat`.
public static func unifiedDequantKernelName(
    for format: any QuantizationFormat
) -> String {
    "dequant_q\(format.bits)_g\(format.groupSize)_bf16"
}

/// Generic dequant→BF16 kernel generator driven by `QuantizationFormat`.
///
/// Emits one threadgroup per output row, 256 threads each, unpacking every
/// weight in the row into BFloat16 laid out row-major. The BF16 output is the
/// input format expected by the Metal 4 MPP GEMM path used during prefill.
///
/// Uses per-weight parallelism via `format.perWeightReadExpression`: each
/// thread handles one weight at a time and the block header (scale/zero) is
/// loaded fresh per iteration. Cache coalescing across threads in the same
/// block keeps the repeated header loads cheap. Aligned (Q2/Q4/Q8) and
/// non-aligned (Q3/Q5/Q6) formats share the same scaffold.
public static func generateUnifiedDequantToBFloat(
    name: String,
    format: any QuantizationFormat
) -> String {
    precondition(
        format.isQuantized,
        "generateUnifiedDequantToBFloat requires isQuantized=true; got \(format.schemeIdentifier)"
    )

    guard let readExpression = format.perWeightReadExpression(
        blocksVar: "qs",
        weightIndexVar: "k"
    ) else {
        fatalError(
            "Format \(format.schemeIdentifier) did not provide perWeightReadExpression"
        )
    }

    let weightsPerBlock = format.weightsPerBlock
    let bytesPerBlock = format.bytesPerBlock
    return """
    #include <metal_stdlib>
    using namespace metal;

    kernel void \(name)(
        device const uchar* packed       [[buffer(0)]],
        device bfloat* output            [[buffer(1)]],
        constant uint& inputDimension    [[buffer(2)]],
        constant uint& outputDimension   [[buffer(3)]],
        uint tgpos [[threadgroup_position_in_grid]],
        uint tid   [[thread_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint THREADS_PER_TG = 256;
        const uint row = tgpos;
        if (row >= outputDimension) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = packed + row * blocksPerRow * BYTES_PER_BLOCK;
        device bfloat* outRow = output + row * inputDimension;

        for (uint weightIdx = tid; weightIdx < inputDimension; weightIdx += THREADS_PER_TG) {
            uint blockIdx = weightIdx / WEIGHTS_PER_BLOCK;
            uint k = weightIdx % WEIGHTS_PER_BLOCK;

            device const uchar* block = rowBase + blockIdx * BYTES_PER_BLOCK;
            float scale = float(*(device const half*)(block));
            float zero  = float(*(device const half*)(block + 2));
            device const uchar* qs = block + 4;

            outRow[weightIdx] = bfloat(\(readExpression));
        }
    }
    """
}

static let kvQuantizationSource = """
kernel void quantize_kv_q8(
    device const half* input [[buffer(0)]], device uchar* output [[buffer(1)]],
    constant uint& totalElements [[buffer(2)]], constant uint& groupSize [[buffer(3)]],
    constant uint& bytesPerBlock [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    uint blocksTotal = totalElements / groupSize;
    if (gid >= blocksTotal) return;
    device const half* groupInput = input + gid * groupSize;
    float minV = HUGE_VALF, maxV = -HUGE_VALF;
    for (uint i = 0; i < groupSize; i++) { float v = float(groupInput[i]); minV = min(minV, v); maxV = max(maxV, v); }
    float scale = (maxV - minV) / 255.0f; float zero = minV;
    if (scale < 1e-10f) scale = 1e-10f;
    device uchar* blockOut = output + gid * bytesPerBlock;
    *(device half*)(blockOut) = half(scale); *(device half*)(blockOut + 2) = half(zero);
    for (uint i = 0; i < groupSize; i++) { int q = int(round((float(groupInput[i]) - zero) / scale)); *(device uchar*)(blockOut + 4 + i) = uchar(clamp(q, 0, 255)); }
}
kernel void dequantize_kv_q8(
    device const uchar* input [[buffer(0)]], device half* output [[buffer(1)]],
    constant uint& totalElements [[buffer(2)]], constant uint& groupSize [[buffer(3)]],
    constant uint& bytesPerBlock [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= totalElements / groupSize) return;
    device const uchar* block = input + gid * bytesPerBlock;
    float scale = float(*(device const half*)(block)); float zero = float(*(device const half*)(block + 2));
    for (uint i = 0; i < groupSize; i++) output[gid * groupSize + i] = half(scale * float(*(device const uchar*)(block + 4 + i)) + zero);
}
"""

static let gemmMixedPrecisionSource = """
kernel void gemm_bf16_f32_to_half(
    device const float* input [[buffer(0)]], device const uint16_t* weight [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& inputDimension [[buffer(3)]],
    constant uint& outputDimension [[buffer(4)]],
    uint2 gid [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint row = gid.x * 2 + sgitg; if (row >= outputDimension) return;
    float sum = 0.0f;
    for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) sum += bf16_to_float(weight[row * inputDimension + j]) * input[j];
    sum = simd_sum(sum); if (tiisg == 0) output[row] = half(sum);
}
kernel void gemm_bf16_f32s_halfout(
    device const float* input [[buffer(0)]], device const uint16_t* weight [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& inputDimension [[buffer(3)]],
    constant uint& outputDimension [[buffer(4)]], constant uint& sequenceLength [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint row = gid.x * 2 + sgitg, seqPos = gid.y;
    if (row >= outputDimension || seqPos >= sequenceLength) return;
    float sum = 0.0f;
    for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) sum += bf16_to_float(weight[row * inputDimension + j]) * input[seqPos * inputDimension + j];
    sum = simd_sum(sum); if (tiisg == 0) output[seqPos * outputDimension + row] = half(sum);
}
"""

// MARK: - Unified quantized GEMV (Phase 1 skeleton)

/// Generate a GEMV kernel for any `QuantizationFormat` that is `isQuantized`.
///
/// This is the Phase 1 unified scaffold used by new formats (Q2/Q3/Q5/Q6).
/// Existing Q4/Q8 kernels in `generateQuantizedGEMV_Q4*` / `...Q8*` are
/// retained untouched — migration to this generator is deferred to Phase 5.
///
/// The block layout assumed here matches the MLX-compatible interleaved layout:
///
/// ```
/// ┌──────────┬──────────┬────────────────────┐
/// │scale (2B)│ zero (2B)│ packed quants      │
/// └──────────┴──────────┴────────────────────┘
/// ```
///
/// Dispatch:
/// Each simdgroup thread reads and dequantizes a single weight per iteration
/// via `format.perWeightReadExpression`. This is work-efficient: total work =
/// weightsPerBlock, parallelism spread across SIMD_WIDTH threads. Aligned
/// formats (Q2/Q4/Q8) and non-aligned formats (Q3/Q5/Q6) share the same
/// scaffold; the Metal compiler flattens ternary-chain expressions used by
/// non-aligned formats into predicated selection.
public static func generateUnifiedQuantizedGEMV(
    name: String,
    format: any QuantizationFormat,
    bufferPrecision: BufferPrecision,
    tileElements: Int = 256
) -> String {
    precondition(
        format.isQuantized,
        "generateUnifiedQuantizedGEMV requires isQuantized=true; got \(format.schemeIdentifier)"
    )

    let bt = bufferPrecision.metalType
    let weightsPerBlock = format.weightsPerBlock
    let bytesPerBlock = format.bytesPerBlock

    guard let readExpression = format.perWeightReadExpression(
        blocksVar: "qs",
        weightIndexVar: "k"
    ) else {
        fatalError(
            "Format \(format.schemeIdentifier) did not provide perWeightReadExpression"
        )
    }
    let scaffoldBody = """
                for (uint k = tiisg; k < WEIGHTS_PER_BLOCK; k += SIMD_WIDTH) {
                    float w = \(readExpression);
                    sum += w * float(inputTile[tileOffset + k]);
                }
    """

    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight     [[buffer(1)]],
        device \(bt)* output            [[buffer(2)]],
        constant uint& inputDimension  [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]],
        uint2 tptg                     [[threads_per_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint THREADS_PER_THREADGROUP = tptg.x;
        const uint rowsPerThreadgroup = THREADS_PER_THREADGROUP / SIMD_WIDTH;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const bool active = row < outputDimension;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        const uint safeRow = active ? row : 0;
        device const uchar* rowBase = weight + safeRow * blocksPerRow * BYTES_PER_BLOCK;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = input[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            if (active) {
                for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                    device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                    float scale = float(*(device const half*)(block));
                    float zero = float(*(device const half*)(block + 2));
                    device const uchar* qs = block + 4;
                    const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
    \(scaffoldBody)
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (active && tiisg == 0) output[row] = \(bt)(sum);
    }
    """
}

/// Generate a sequence GEMV kernel for packed quantized weights.
///
/// The reduction contract matches decode GEMV for each row-token pair: one SIMD
/// group owns one output row for one sequence position, uses the same packed
/// weight read expression, and rounds the stored activation to Float16 semantics.
public static func generateUnifiedQuantizedSequenceGEMV(
    name: String,
    format: any QuantizationFormat,
    bufferPrecision: BufferPrecision,
    tileElements: Int = 256
) -> String {
    precondition(
        format.isQuantized,
        "generateUnifiedQuantizedSequenceGEMV requires isQuantized=true; got \(format.schemeIdentifier)"
    )

    let bt = bufferPrecision.metalType
    let weightsPerBlock = format.weightsPerBlock
    let bytesPerBlock = format.bytesPerBlock
    let storeValue = MetalSourceGenerator.sequenceStorageValue("sum", weightFormat: .float16)

    guard let readExpression = format.perWeightReadExpression(
        blocksVar: "qs",
        weightIndexVar: "k"
    ) else {
        fatalError(
            "Format \(format.schemeIdentifier) did not provide perWeightReadExpression"
        )
    }
    let scaffoldBody = """
                for (uint k = tiisg; k < WEIGHTS_PER_BLOCK; k += SIMD_WIDTH) {
                    float w = \(readExpression);
                    sum += w * float(inputTile[tileOffset + k]);
                }
    """

    return """
    kernel void \(name)(
        device const \(bt)* input              [[buffer(0)]],
        device const uchar* weight            [[buffer(1)]],
        device \(bt)* output                  [[buffer(2)]],
        constant uint& inputDimension         [[buffer(3)]],
        constant uint& outputDimension        [[buffer(4)]],
        constant uint& sequenceLength         [[buffer(5)]],
        constant uint& inputRowStride         [[buffer(6)]],
        constant uint& outputRowStride        [[buffer(7)]],
        uint2 gid                             [[threadgroup_position_in_grid]],
        uint tid                              [[thread_index_in_threadgroup]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]],
        uint2 tptg                            [[threads_per_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint THREADS_PER_THREADGROUP = tptg.x;
        const uint rowsPerThreadgroup = THREADS_PER_THREADGROUP / SIMD_WIDTH;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        const bool active = row < outputDimension && seqPos < sequenceLength;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        const uint safeRow = min(row, max(outputDimension, 1u) - 1u);
        const uint safeSeqPos = min(seqPos, max(sequenceLength, 1u) - 1u);
        device const uchar* rowBase = weight + safeRow * blocksPerRow * BYTES_PER_BLOCK;
        device const \(bt)* inputRow = input + safeSeqPos * inputRowStride;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = inputRow[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            if (active) {
                for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                    device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                    float scale = float(*(device const half*)(block));
                    float zero = float(*(device const half*)(block + 2));
                    device const uchar* qs = block + 4;
                    const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
    \(scaffoldBody)
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (active && tiisg == 0) {
            output[seqPos * outputRowStride + row] = \(bt)(\(storeValue));
        }
    }
    """
}

/// Generate a sequence GEMV kernel for packed quantized weights that covers
/// multiple adjacent sequence positions per threadgroup.
///
/// Each SIMD group still owns one output row for one sequence position. The
/// sequence tile only reduces grid height and amortizes input staging across
/// adjacent tokens, while preserving the single row-token reduction contract.
public static func generateTiledQuantizedSequenceGEMV(
    name: String,
    format: any QuantizationFormat,
    bufferPrecision: BufferPrecision,
    sequenceTile: Int,
    tileElements: Int = 256
) -> String {
    precondition(sequenceTile >= 1, "sequence tile must be positive")
    precondition(
        format.isQuantized,
        "generateTiledQuantizedSequenceGEMV requires isQuantized=true; got \(format.schemeIdentifier)"
    )

    let bt = bufferPrecision.metalType
    let weightsPerBlock = format.weightsPerBlock
    let bytesPerBlock = format.bytesPerBlock
    let storeValue = MetalSourceGenerator.sequenceStorageValue("sum", weightFormat: .float16)

    guard let readExpression = format.perWeightReadExpression(
        blocksVar: "qs",
        weightIndexVar: "k"
    ) else {
        fatalError(
            "Format \(format.schemeIdentifier) did not provide perWeightReadExpression"
        )
    }

    return """
    kernel void \(name)(
        device const \(bt)* input              [[buffer(0)]],
        device const uchar* weight            [[buffer(1)]],
        device \(bt)* output                  [[buffer(2)]],
        constant uint& inputDimension         [[buffer(3)]],
        constant uint& outputDimension        [[buffer(4)]],
        constant uint& sequenceLength         [[buffer(5)]],
        constant uint& inputRowStride         [[buffer(6)]],
        constant uint& outputRowStride        [[buffer(7)]],
        uint2 gid                             [[threadgroup_position_in_grid]],
        uint tid                              [[thread_index_in_threadgroup]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]],
        uint2 threadsPerThreadgroup           [[threads_per_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint TILE_ELEMENTS = \(tileElements);
        const uint sequenceTile = \(sequenceTile);
        const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
        const uint rowsPerThreadgroup = max(1u, simdgroupsPerThreadgroup / sequenceTile);
        const uint localSeq = min(sequenceTile - 1u, sgitg / rowsPerThreadgroup);
        const uint localRow = sgitg - localSeq * rowsPerThreadgroup;
        const uint row = gid.x * rowsPerThreadgroup + localRow;
        const uint seqPos = gid.y * sequenceTile + localSeq;
        const bool validSeq = seqPos < sequenceLength;
        const bool validRow = row < outputDimension;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        threadgroup \(bt) inputTile[\(sequenceTile * tileElements)];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            if (localRow == 0u && validSeq) {
                device const \(bt)* inputRow = input + seqPos * inputRowStride;
                threadgroup \(bt)* tile = inputTile + localSeq * TILE_ELEMENTS;
                for (uint j = tiisg; j < TILE_ELEMENTS; j += SIMD_WIDTH) {
                    const uint inputIndex = base + j;
                    tile[j] = inputIndex < inputDimension ? inputRow[inputIndex] : \(bt)(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (validSeq && validRow) {
                device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
                threadgroup \(bt)* tile = inputTile + localSeq * TILE_ELEMENTS;
                const uint blockBase = base / WEIGHTS_PER_BLOCK;
                const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
                for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                    device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                    float scale = float(*(device const half*)(block));
                    float zero = float(*(device const half*)(block + 2));
                    device const uchar* qs = block + 4;
                    const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                    for (uint k = tiisg; k < WEIGHTS_PER_BLOCK; k += SIMD_WIDTH) {
                        float w = \(readExpression);
                        sum += w * float(tile[tileOffset + k]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (validSeq && validRow) {
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[seqPos * outputRowStride + row] = \(bt)(\(storeValue));
            }
        }
    }
    """
}

/// Generate a batched sequence GEMV kernel for multiple packed quantized
/// projections sharing the same sequence input.
///
/// The numeric contract mirrors `generateUnifiedQuantizedSequenceGEMV`: one SIMD
/// group owns one output row for one sequence position, preserving decode-style
/// reduction order and sequence-storage rounding while reducing dispatch count
/// for Q/K/V and gate/up style projection groups.
public static func generateBatchedQuantizedSequenceGEMV(
    name: String,
    count: Int,
    format: any QuantizationFormat,
    bufferPrecision: BufferPrecision,
    tileElements: Int = 256
) -> String {
    precondition((2...4).contains(count), "batched quantized sequence GEMV supports 2...4 projections")
    precondition(
        format.isQuantized,
        "generateBatchedQuantizedSequenceGEMV requires isQuantized=true; got \(format.schemeIdentifier)"
    )

    let bt = bufferPrecision.metalType
    let weightsPerBlock = format.weightsPerBlock
    let bytesPerBlock = format.bytesPerBlock
    let storeValue = MetalSourceGenerator.sequenceStorageValue("sum", weightFormat: .float16)

    guard let readExpression = format.perWeightReadExpression(
        blocksVar: "qs",
        weightIndexVar: "k"
    ) else {
        fatalError(
            "Format \(format.schemeIdentifier) did not provide perWeightReadExpression"
        )
    }

    let weightBindings = (0..<count).map { i in
        "device const uchar* weight\(i)        [[buffer(\(1 + i))]],"
    }.joined(separator: "\n        ")
    let outputBindings = (0..<count).map { i in
        "device \(bt)* output\(i)              [[buffer(\(1 + count + i))]],"
    }.joined(separator: "\n        ")
    let outputDimBindings = (0..<count).map { i in
        "constant uint& outputDim\(i)          [[buffer(\(2 + 2 * count + i))]],"
    }.joined(separator: "\n        ")
    let totalRows = (0..<count).map { "outputDim\($0)" }.joined(separator: " + ")
    let branchBlocks = (0..<count).map { i -> String in
        let condition: String
        if i == 0 {
            condition = "if (globalRow < outputDim0)"
        } else if i == count - 1 {
            condition = "else"
        } else {
            let cumulative = (0...i).map { "outputDim\($0)" }.joined(separator: " + ")
            condition = "else if (globalRow < \(cumulative))"
        }
        let prior = i == 0 ? "0u" : (0..<i).map { "outputDim\($0)" }.joined(separator: " + ")
        return """
                \(condition) {
                    weight = weight\(i);
                    output = output\(i);
                    localRow = globalRow - (\(prior));
                }
        """
    }.joined(separator: "\n        ")

    return """
    kernel void \(name)(
        device const \(bt)* input              [[buffer(0)]],
        \(weightBindings)
        \(outputBindings)
        constant uint& inputDimension          [[buffer(\(1 + 2 * count))]],
        \(outputDimBindings)
        constant uint& sequenceLength          [[buffer(\(2 + 3 * count))]],
        constant uint& inputRowStride          [[buffer(\(3 + 3 * count))]],
        constant uint& outputRowStride         [[buffer(\(4 + 3 * count))]],
        uint2 gid                              [[threadgroup_position_in_grid]],
        uint tid                               [[thread_index_in_threadgroup]],
        uint tiisg                             [[thread_index_in_simdgroup]],
        uint sgitg                             [[simdgroup_index_in_threadgroup]],
        uint2 tptg                             [[threads_per_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint TILE_ELEMENTS = \(tileElements);
        const uint THREADS_PER_THREADGROUP = tptg.x;
        const uint rowsPerThreadgroup = THREADS_PER_THREADGROUP / SIMD_WIDTH;
        const uint globalRow = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        const uint totalRows = \(totalRows);
        const bool active = globalRow < totalRows && seqPos < sequenceLength;

        device const uchar* weight = weight0;
        device \(bt)* output = output0;
        uint localRow = 0;
        if (active) {
        \(branchBlocks)
        }

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        const uint safeSeqPos = min(seqPos, max(sequenceLength, 1u) - 1u);
        device const \(bt)* inputRow = input + safeSeqPos * inputRowStride;
        device const uchar* rowBase = weight + localRow * blocksPerRow * BYTES_PER_BLOCK;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = inputRow[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            if (active) {
                for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                    device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                    float scale = float(*(device const half*)(block));
                    float zero = float(*(device const half*)(block + 2));
                    device const uchar* qs = block + 4;
                    const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                    for (uint k = tiisg; k < WEIGHTS_PER_BLOCK; k += SIMD_WIDTH) {
                        float w = \(readExpression);
                        sum += w * float(inputTile[tileOffset + k]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (active && tiisg == 0) {
            output[seqPos * outputRowStride + localRow] = \(bt)(\(storeValue));
        }
    }
    """
}

/// Generate a batched GEMV kernel for multiple quantized projections sharing
/// the same decode input.
///
/// The binding convention intentionally matches the dense batched GEMV kernels:
/// input at buffer 0, `count` packed weight buffers, `count` output buffers,
/// followed by `inputDimension` and one output dimension per projection.
/// Dequantization is delegated to `QuantizationFormat.perWeightReadExpression`
/// so Q2/Q3/Q4/Q5/Q6/Q8 formats use one source template instead of being
/// decomposed into per-projection dispatches.
public static func generateBatchedQuantizedGEMV(
    name: String,
    count: Int,
    format: any QuantizationFormat,
    bufferPrecision: BufferPrecision,
    tileElements: Int = 256
) -> String {
    precondition((2...4).contains(count), "batched quantized GEMV supports 2...4 projections")
    precondition(
        format.isQuantized,
        "generateBatchedQuantizedGEMV requires isQuantized=true; got \(format.schemeIdentifier)"
    )

    let bt = bufferPrecision.metalType
    let weightsPerBlock = format.weightsPerBlock
    let bytesPerBlock = format.bytesPerBlock

    guard let readExpression = format.perWeightReadExpression(
        blocksVar: "qs",
        weightIndexVar: "k"
    ) else {
        fatalError(
            "Format \(format.schemeIdentifier) did not provide perWeightReadExpression"
        )
    }

    let weightBindings = (0..<count).map { i in
        "device const uchar* weight\(i)    [[buffer(\(1 + i))]],"
    }.joined(separator: "\n        ")
    let outputBindings = (0..<count).map { i in
        "device \(bt)* output\(i)           [[buffer(\(1 + count + i))]],"
    }.joined(separator: "\n        ")
    let outputDimBindings = (0..<count).map { i in
        "constant uint& outputDim\(i)      [[buffer(\(2 + 2 * count + i))]],"
    }.joined(separator: "\n        ")
    let totalRows = (0..<count).map { "outputDim\($0)" }.joined(separator: " + ")
    let branchBlocks = (0..<count).map { i -> String in
        let condition: String
        if i == 0 {
            condition = "if (globalRow < outputDim0)"
        } else if i == count - 1 {
            condition = "else"
        } else {
            let cumulative = (0...i).map { "outputDim\($0)" }.joined(separator: " + ")
            condition = "else if (globalRow < \(cumulative))"
        }
        let prior = i == 0 ? "0u" : (0..<i).map { "outputDim\($0)" }.joined(separator: " + ")
        return """
                \(condition) {
                    weight = weight\(i);
                    output = output\(i);
                    localRow = globalRow - (\(prior));
                }
        """
    }.joined(separator: "\n        ")

    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        \(weightBindings)
        \(outputBindings)
        constant uint& inputDimension  [[buffer(\(1 + 2 * count))]],
        \(outputDimBindings)
        uint gid                       [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]],
        uint threadsPerThreadgroup     [[threads_per_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint TILE_ELEMENTS = \(tileElements);
        const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
        const uint globalRow = gid * rowsPerThreadgroup + sgitg;
        const uint totalRows = \(totalRows);
        const bool active = globalRow < totalRows;

        device const uchar* weight = weight0;
        device \(bt)* output = output0;
        uint localRow = 0;
        if (active) {
        \(branchBlocks)
        }

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + localRow * blocksPerRow * BYTES_PER_BLOCK;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += threadsPerThreadgroup) {
                inputTile[j] = input[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            if (active) {
                for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                    device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                    float scale = float(*(device const half*)(block));
                    float zero = float(*(device const half*)(block + 2));
                    device const uchar* qs = block + 4;
                    const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                    for (uint k = tiisg; k < WEIGHTS_PER_BLOCK; k += SIMD_WIDTH) {
                        float w = \(readExpression);
                        sum += w * float(inputTile[tileOffset + k]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (active && tiisg == 0) {
            output[localRow] = \(bt)(sum);
        }
    }
    """
}

}
