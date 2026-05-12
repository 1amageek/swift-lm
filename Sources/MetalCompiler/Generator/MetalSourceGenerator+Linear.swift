extension MetalSourceGenerator {
// MARK: - Linear Kernels

    /// Supported MPP GEMM M-tile sizes. Callers emit one kernel per size and
    /// select the appropriate pipeline at dispatch time based on runtime
    /// sequence length, trading off padding waste (small seq → small tile) and
    /// per-threadgroup work (long seq → large tile).
    public static let mppGEMMTileSizes: [Int] = [16, 32, 64]

    /// Default M-tile used when only a single variant is needed (long-seq baseline).
    public static let mppGEMMDefaultTileSize: Int = 64

    /// Suffix for a tile-specific kernel variant name.
    public static func mppGEMMVariantName(baseName: String, tileSize: Int) -> String {
        "\(baseName)_mtile\(tileSize)"
    }

    public static func generateMPPGEMM(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        mTile: Int = mppGEMMDefaultTileSize
    ) -> String {
        let bt = bufferPrecision.metalType
        let tensorWeightType: String
        if weightFormat.isQuantized {
            tensorWeightType = bt
        } else if weightFormat.isBFloat16 {
            tensorWeightType = "bfloat"
        } else if weightFormat.isFloat32 {
            tensorWeightType = "float"
        } else {
            tensorWeightType = "half"
        }

        return """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;

        kernel void \(name)(
            device \(bt)* input              [[buffer(0)]],
            device \(tensorWeightType)* weight [[buffer(1)]],
            device \(bt)* output             [[buffer(2)]],
            constant uint& inputDimension    [[buffer(3)]],
            constant uint& outputDimension   [[buffer(4)]],
            constant uint& sequenceLength    [[buffer(5)]],
            constant uint& inputRowStride    [[buffer(6)]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            using namespace mpp::tensor_ops;
            (void)inputRowStride;

            // Pad sequence extent to the M-tile boundary so that edge
            // threadgroups never slice beyond the tensor declaration.
            // The backing buffers are allocated for maximumSequenceLength,
            // which is always >= paddedSeqLen.
            constexpr uint M_TILE = \(mTile);
            const uint paddedSeqLen = ((sequenceLength + M_TILE - 1) / M_TILE) * M_TILE;

            auto A = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                input, dextents<int32_t, 2>(inputDimension, paddedSeqLen));
            auto B = tensor<device \(tensorWeightType), dextents<int32_t, 2>, tensor_inline>(
                weight, dextents<int32_t, 2>(inputDimension, outputDimension));
            auto C = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                output, dextents<int32_t, 2>(outputDimension, paddedSeqLen));

            constexpr auto desc = matmul2d_descriptor(
                M_TILE, 32, dynamic_length_v<int>,
                false, true, false,
                matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;

            auto mA = A.slice(0, tgid.y * M_TILE);
            auto mB = B.slice(0, tgid.x * 32);
            auto mC = C.slice(tgid.x * 32, tgid.y * M_TILE);
            op.run(mA, mB, mC);
        }
        """
    }

    /// Generate MSL source for a batched MPP GEMM kernel with BF16 or native dense weights.
    ///
    /// All `count` projections share the same input `A`. Each projection has its
    /// own weight matrix and output buffer. tgid.x linearly indexes N-tiles across
    /// all projections; each threadgroup maps to one projection based on
    /// cumulative N-tile counts. tgid.y indexes the M-tile (sequence position).
    ///
    /// Emitting one kernel for multiple projections removes barriers between
    /// them and reduces dispatch encoding cost on the CPU side, which is the
    /// dominant cost for short-sequence prefill on Apple Silicon.
    ///
    /// Assumptions:
    /// - Every `outputDim_i` is a multiple of 32 (N_TILE).
    /// - Count is 2 or 3 (used for gate/up and Q/K/V respectively).
    public static func generateBatchedMPPGEMM(
        name: String,
        count: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        mTile: Int = mppGEMMDefaultTileSize
    ) -> String {
        precondition(count >= 2, "batched MPP GEMM requires count >= 2")
        let bt = bufferPrecision.metalType
        let tensorWeightType: String
        if weightFormat.isQuantized {
            tensorWeightType = bt
        } else if weightFormat.isBFloat16 {
            tensorWeightType = "bfloat"
        } else if weightFormat.isFloat32 {
            tensorWeightType = "float"
        } else {
            tensorWeightType = "half"
        }

        // Buffer binding layout:
        //   0           : input
        //   1..count    : weight_i
        //   count+1..2*count : output_i
        //   2*count+1   : inputDimension
        //   2*count+2..3*count+1 : outputDim_i
        //   3*count+2   : sequenceLength
        //   3*count+3   : inputRowStride
        let weightBindings = (0..<count).map { i in
            "device \(tensorWeightType)* weight\(i)      [[buffer(\(1 + i))]],"
        }.joined(separator: "\n    ")
        let outputBindings = (0..<count).map { i in
            "device \(bt)* output\(i)         [[buffer(\(1 + count + i))]],"
        }.joined(separator: "\n    ")
        let outputDimBindings = (0..<count).map { i in
            "constant uint& outputDim\(i)     [[buffer(\(2 + 2 * count + i))]],"
        }.joined(separator: "\n    ")

        // Per-projection run blocks. Each block constructs local B/C tensor
        // slices from the projection-local N-tile index and runs matmul2d.
        var runBlocks: [String] = []
        for i in 0..<count {
            let priorTilesExpr: String
            if i == 0 {
                priorTilesExpr = "0u"
            } else {
                priorTilesExpr = (0..<i).map { "(outputDim\($0) / 32)" }.joined(separator: " + ")
            }
            // nTileLimit excludes the early-return case — we only enter this
            // branch when nTile is in this projection's range.
            let conditionExpr: String
            if i == 0 {
                conditionExpr = "if (nTile < outputDim0 / 32)"
            } else if i == count - 1 {
                conditionExpr = "else"
            } else {
                let cumulative = (0...i).map { "(outputDim\($0) / 32)" }.joined(separator: " + ")
                conditionExpr = "else if (nTile < \(cumulative))"
            }
            runBlocks.append("""
                \(conditionExpr) {
                    const uint localNTile = nTile - (\(priorTilesExpr));
                    auto B = tensor<device \(tensorWeightType), dextents<int32_t, 2>, tensor_inline>(
                        weight\(i), dextents<int32_t, 2>(inputDimension, outputDim\(i)));
                    auto C = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                        output\(i), dextents<int32_t, 2>(outputDim\(i), paddedSeqLen));
                    auto mB = B.slice(0, localNTile * 32);
                    auto mC = C.slice(localNTile * 32, tgid.y * M_TILE);
                    op.run(mA, mB, mC);
                }
                """)
        }
        let runBody = runBlocks.joined(separator: "\n            ")

        return """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;

        kernel void \(name)(
            device \(bt)* input              [[buffer(0)]],
            \(weightBindings)
            \(outputBindings)
            constant uint& inputDimension    [[buffer(\(1 + 2 * count))]],
            \(outputDimBindings)
            constant uint& sequenceLength    [[buffer(\(2 + 3 * count))]],
            constant uint& inputRowStride    [[buffer(\(3 + 3 * count))]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            using namespace mpp::tensor_ops;
            (void)inputRowStride;

            constexpr uint M_TILE = \(mTile);
            constexpr uint N_TILE = 32;
            const uint paddedSeqLen = ((sequenceLength + M_TILE - 1) / M_TILE) * M_TILE;

            auto A = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                input, dextents<int32_t, 2>(inputDimension, paddedSeqLen));

            constexpr auto desc = matmul2d_descriptor(
                M_TILE, N_TILE, dynamic_length_v<int>,
                false, true, false,
                matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;

            auto mA = A.slice(0, tgid.y * M_TILE);

            const uint nTile = tgid.x;
            \(runBody)
        }
        """
    }

    /// Generate MSL source for a GEMM kernel (prefill projection, naive fallback).
    public static func generateGEMM(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let storeValue: (String) -> String = { expr in
            bufferPrecision.isPrefillSequencePrecision
                ? MetalSourceGenerator.sequenceStorageValue(expr, weightFormat: weightFormat)
                : expr
        }

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            constant uint& sequenceLength          [[buffer(5)]],
            constant uint& inputRowStride         [[buffer(6)]],
            constant uint& outputRowStride         [[buffer(7)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]]
        ) {
            const uint rowsPerThreadgroup = 2;
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            const uint seqPos = gid.y;
            if (row >= outputDimension || seqPos >= sequenceLength) return;

            float sum = 0.0f;
            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            device const \(wt)* weightRow = weight + row * inputDimension;
            for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
                sum += \(readWeight("weightRow[j]")) * float(inputRow[j]);
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[seqPos * outputRowStride + row] = \(bt)(\(storeValue("sum")));
            }
        }
        """
    }

    /// Generate MSL source for a GEMV kernel (decode projection, single token).
    ///
    /// Optimization: input is staged into threadgroup memory in tiles and reused
    /// by all rows in the threadgroup. This cuts repeated input reads on the
    /// decode hot path where multiple output rows share the same activation.
    public static func generateGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        tileElements: Int = 128
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            uint gid                               [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup             [[threads_per_threadgroup]]
        ) {
            const uint tileElements = \(tileElements);
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            const bool active = row < outputDimension;

            threadgroup \(bt) inputTile[tileElements];
            float sum = 0.0f;
            const uint safeRow = active ? row : 0;
            device const \(wt)* weightRow = weight + safeRow * inputDimension;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup) {
                    const uint inputIndex = base + j;
                    inputTile[j] = inputIndex < inputDimension ? input[inputIndex] : \(bt)(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileCount = min(tileElements, inputDimension - base);
                if (active) {
                    for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                        sum += \(readWeight("weightRow[base + j]")) * float(inputTile[j]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            sum = simd_sum(sum);
            if (active && tiisg == 0) {
                output[row] = \(bt)(sum);
            }
        }
        """
    }

    /// Generate a sequence GEMV kernel for decode-equivalent prefill projections.
    ///
    /// This preserves the decode GEMV reduction order for every token while still
    /// encoding one sequence-wide dispatch. Hybrid prefill uses this for
    /// state/cache-sensitive projections where GEMM/MPP accumulation differences
    /// can otherwise perturb subsequent decode state.
    public static func generateSequenceGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        tileElements: Int = 256
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let storeValue: (String) -> String = { expr in
            bufferPrecision.isPrefillSequencePrecision
                ? MetalSourceGenerator.sequenceStorageValue(expr, weightFormat: weightFormat)
                : expr
        }

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            constant uint& sequenceLength          [[buffer(5)]],
            constant uint& inputRowStride          [[buffer(6)]],
            constant uint& outputRowStride         [[buffer(7)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint tileElements = \(tileElements);
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            const uint seqPos = gid.y;
            if (seqPos >= sequenceLength) return;
            const bool active = row < outputDimension;

            threadgroup \(bt) inputTile[tileElements];
            float sum = 0.0f;
            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            const uint safeRow = active ? row : 0;
            device const \(wt)* weightRow = weight + safeRow * inputDimension;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup.x) {
                    const uint inputIndex = base + j;
                    inputTile[j] = inputIndex < inputDimension ? inputRow[inputIndex] : \(bt)(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileCount = min(tileElements, inputDimension - base);
                if (active) {
                    for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                        sum += \(readWeight("weightRow[base + j]")) * float(inputTile[j]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            sum = simd_sum(sum);
            if (active && tiisg == 0) {
                output[seqPos * outputRowStride + row] = \(bt)(\(storeValue("sum")));
            }
        }
        """
    }

    /// Generate a sequence GEMV kernel that covers multiple tokens per threadgroup.
    ///
    /// Each SIMD group still owns one output row and one token, preserving the
    /// decode GEMV reduction order for every row-token pair. The sequence tile
    /// only amortizes input staging and dispatch overhead across adjacent tokens.
    public static func generateTiledSequenceGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        sequenceTile: Int,
        tileElements: Int = 256
    ) -> String {
        precondition(sequenceTile >= 1, "sequence tile must be positive")
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let storeValue: (String) -> String = { expr in
            bufferPrecision.isPrefillSequencePrecision
                ? MetalSourceGenerator.sequenceStorageValue(expr, weightFormat: weightFormat)
                : expr
        }

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            constant uint& sequenceLength          [[buffer(5)]],
            constant uint& inputRowStride          [[buffer(6)]],
            constant uint& outputRowStride         [[buffer(7)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint tileElements = \(tileElements);
            const uint sequenceTile = \(sequenceTile);
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint rowsPerThreadgroup = max(1u, simdgroupsPerThreadgroup / sequenceTile);
            const uint localSeq = min(sequenceTile - 1u, sgitg / rowsPerThreadgroup);
            const uint localRow = sgitg - localSeq * rowsPerThreadgroup;
            const uint row = gid.x * rowsPerThreadgroup + localRow;
            const uint seqPos = gid.y * sequenceTile + localSeq;
            const bool validSeq = seqPos < sequenceLength;
            const bool validRow = row < outputDimension;

            threadgroup \(bt) inputTile[\(sequenceTile * tileElements)];
            float sum = 0.0f;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                if (localRow == 0u && validSeq) {
                    device const \(bt)* inputRow = input + seqPos * inputRowStride;
                    threadgroup \(bt)* tile = inputTile + localSeq * tileElements;
                    for (uint j = tiisg; j < tileElements; j += SIMD_WIDTH) {
                        const uint inputIndex = base + j;
                        tile[j] = inputIndex < inputDimension ? inputRow[inputIndex] : \(bt)(0.0f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (validSeq && validRow) {
                    device const \(wt)* weightRow = weight + row * inputDimension;
                    threadgroup \(bt)* tile = inputTile + localSeq * tileElements;
                    const uint tileCount = min(tileElements, inputDimension - base);
                    for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                        sum += \(readWeight("weightRow[base + j]")) * float(tile[j]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (validSeq && validRow) {
                sum = simd_sum(sum);
                if (tiisg == 0) {
                    output[seqPos * outputRowStride + row] = \(bt)(\(storeValue("sum")));
                }
            }
        }
        """
    }

    /// Generate a fused SwiGLU + sequence GEMV kernel for the MLP down projection.
    ///
    /// Producer-consumer fusion: computes `silu(gate[j]) * up[j]` on-the-fly during
    /// the GEMV tile-load step, then accumulates `sum += weight[row, j] * tile[j]`
    /// in F32 with the same SIMD-group reduction order as `generateSequenceGEMV`.
    ///
    /// The intermediate SwiGLU output is never materialized to scratch — the kernel
    /// reads `gate` and `up` directly and writes only the projected hidden output.
    /// This eliminates the SwiGLU dispatch, its barrier, and the round-trip of
    /// `intermediateDim * sequenceLength * 4 bytes` through scratch slot 0.
    ///
    /// Numerical contract:
    ///   * Each `silu(g)*up` intermediate remains F32, matching the materialized
    ///     `swiglu_seq_f32 + gemv_seq_bf16_f32s` prefill path. The fused kernel
    ///     avoids writing the intermediate to scratch, but it does not change the
    ///     down-projection input precision.
    ///   * The reduction order, accumulator precision, and final-output cast are
    ///     identical to `generateSequenceGEMV` for BF16 weight: the running F32
    ///     `sum` is reduced via `simd_sum` and stored as `float(bfloat(sum))`.
    ///
    /// Currently only BF16 weight is supported because the rounding contract is
    /// weight-format-specific. Q3/Q4/Q8 fused variants would require separate
    /// rounding/dequant contracts and are out of scope for this generator.
    public static func generateFusedSwigluDownSequenceGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        tileElements: Int = 256,
        rowsPerSimdgroup: Int = 1
    ) -> String {
        precondition(
            weightFormat.isBFloat16,
            "generateFusedSwigluDownSequenceGEMV currently supports only BF16 weight format"
        )
        precondition(
            bufferPrecision.isPrefillSequencePrecision,
            "generateFusedSwigluDownSequenceGEMV requires the prefill sequence buffer precision"
        )
        precondition(
            rowsPerSimdgroup >= 1 && rowsPerSimdgroup <= 4,
            "generateFusedSwigluDownSequenceGEMV supports 1...4 rows per simdgroup"
        )
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let sumDeclarations = (0..<rowsPerSimdgroup)
            .map { "            float sum\($0) = 0.0f;" }
            .joined(separator: "\n")
        let accumulationLines = (0..<rowsPerSimdgroup)
            .map { rowOffset in
                """
                    if (row\(rowOffset) < outputDimension) {
                        sum\(rowOffset) += \(readWeight("weight[row\(rowOffset) * intermediateDim + base + j]")) * tileValue;
                    }
                """
            }
            .joined(separator: "\n")
        let reductionLines = (0..<rowsPerSimdgroup)
            .map { rowOffset in
                let stored = MetalSourceGenerator.sequenceStorageValue(
                    "sum\(rowOffset)",
                    weightFormat: weightFormat
                )
                return """
                    sum\(rowOffset) = simd_sum(sum\(rowOffset));
                    if (tiisg == 0 && row\(rowOffset) < outputDimension) {
                        output[seqPos * outputRowStride + row\(rowOffset)] = \(bt)(\(stored));
                    }
                """
            }
            .joined(separator: "\n")
        let rowDeclarations = (0..<rowsPerSimdgroup)
            .map { "            const uint row\($0) = rowBase + \($0)u;" }
            .joined(separator: "\n")

        return """
        kernel void \(name)(
            device const \(bt)* gate               [[buffer(0)]],
            device const \(bt)* up                 [[buffer(1)]],
            device const \(wt)* weight             [[buffer(2)]],
            device \(bt)* output                   [[buffer(3)]],
            constant uint& intermediateDim         [[buffer(4)]],
            constant uint& outputDimension         [[buffer(5)]],
            constant uint& sequenceLength          [[buffer(6)]],
            constant uint& inputRowStride          [[buffer(7)]],
            constant uint& outputRowStride         [[buffer(8)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint tileElements = \(tileElements);
            const uint rowsPerSimdgroup = \(rowsPerSimdgroup)u;
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint rowsPerThreadgroup = simdgroupsPerThreadgroup * rowsPerSimdgroup;
            const uint rowBase = gid.x * rowsPerThreadgroup + sgitg * rowsPerSimdgroup;
            const uint seqPos = gid.y;
            if (seqPos >= sequenceLength) return;
            \(rowDeclarations)

            threadgroup \(bt) inputTile[tileElements];
            \(sumDeclarations)
            device const \(bt)* gateRow = gate + seqPos * inputRowStride;
            device const \(bt)* upRow   = up   + seqPos * inputRowStride;
            for (uint base = 0; base < intermediateDim; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup.x) {
                    const uint inputIndex = base + j;
                    if (inputIndex < intermediateDim) {
                        float g = float(gateRow[inputIndex]);
                        float u = float(upRow[inputIndex]);
                        float silu_up = g * (1.0f / (1.0f + exp(-g))) * u;
                        inputTile[j] = \(bt)(silu_up);
                    } else {
                        inputTile[j] = \(bt)(0.0f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileCount = min(tileElements, intermediateDim - base);
                for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                    const float tileValue = float(inputTile[j]);
                    \(accumulationLines)
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            \(reductionLines)
        }
        """
    }

    /// Generate a GEMV kernel specialized for vocab/output-head style projections.
    ///
    /// The input dimension is expected to be 2048. The entire input vector is staged
    /// into threadgroup memory once, avoiding the repeated tile barriers used by the
    /// generic large GEMV path.
    public static func generateVocabGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        return generateSpecializedDenseGEMV(
            name: name,
            bufferPrecision: bufferPrecision,
            weightFormat: weightFormat,
            stagedInputElements: 2_048,
            fixedInputDimension: 2_048,
            inputStaging: .bufferPrecision,
            accumulationStyle: .pointerIncrement
        )
    }

    /// Generate a GEMV kernel specialized for decode projections with inputDimension=2048.
    ///
    /// This family stages the full hidden vector once into threadgroup memory and reuses it
    /// across all rows in the threadgroup. It is used both for the output head and for the
    /// common 2048→{2048,6144,8192} decode projections.
    public static func generateInput2048GEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        fixedOutputDimension: Int? = nil,
        fixedRowsPerThreadgroup: Int? = nil,
        fixedSimdgroups: Int? = nil,
        stagesInputAsFloat: Bool = true,
        weightLayoutPolicy: Input2048WeightLayoutPolicy = .rowMajor,
        unrollFactor: Int = 4
    ) -> String {
        _ = weightLayoutPolicy
        return generateSpecializedDenseGEMV(
            name: name,
            bufferPrecision: bufferPrecision,
            weightFormat: weightFormat,
            stagedInputElements: 2_048,
            fixedInputDimension: 2_048,
            fixedOutputDimension: fixedOutputDimension,
            fixedRowsPerThreadgroup: fixedRowsPerThreadgroup,
            fixedSimdgroups: fixedSimdgroups,
            inputStaging: stagesInputAsFloat ? .float : .bufferPrecision,
            accumulationStyle: .indexed,
            unrollFactor: unrollFactor
        )
    }


    public static func generateInput8192TiledGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        stagesInputAsFloat: Bool = true,
        fixedOutputDimension: Int? = nil,
        tileElements: Int = 1_024,
        unrollFactor: Int = 4
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagedInputType = stagesInputAsFloat ? "float" : bt
        let stagedInputRead = stagesInputAsFloat ? "" : "float"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let unrolledAccumulate = (0..<effectiveUnroll).map { lane -> String in
            if lane == 0 {
                return "sum += \(readWeight("tileWeight[0]")) * \(stagedInputRead)(tileInput[0]);"
            }
            let offset = "\(lane)"
            return "sum += \(readWeight("tileWeight[\(offset)]")) * \(stagedInputRead)(tileInput[\(offset)]);"
        }.joined(separator: "\n")

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            uint gid                               [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup             [[threads_per_threadgroup]]
        ) {
            const uint fixedInputDimension = 8192u;
            const uint stagedInputElements = \(tileElements);
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            const bool active = row < \(fixedOutputDimension.map { "\($0)u" } ?? "outputDimension");

            threadgroup \(stagedInputType) inputTile[stagedInputElements];
            float sum = 0.0f;
            const uint safeRow = active ? row : 0;
            device const \(wt)* weightRow = weight + safeRow * fixedInputDimension;
            for (uint base = 0; base < fixedInputDimension; base += stagedInputElements) {
                device const \(bt)* inputTileSource = input + base + tid;
                for (uint j = tid; j < stagedInputElements; j += threadsPerThreadgroup) {
                    inputTile[j] = \(stagesInputAsFloat ? "float(inputTileSource[0])" : "inputTileSource[0]");
                    inputTileSource += threadsPerThreadgroup;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (active) {
                    device const \(wt)* tileWeight = weightRow + base + tiisg * \(effectiveUnroll);
                    threadgroup const \(stagedInputType)* tileInput = inputTile + tiisg * \(effectiveUnroll);
                    for (uint j = tiisg * \(effectiveUnroll); j < stagedInputElements; j += SIMD_WIDTH * \(effectiveUnroll)) {
                        \(unrolledAccumulate)
                        tileWeight += SIMD_WIDTH * \(effectiveUnroll);
                        tileInput += SIMD_WIDTH * \(effectiveUnroll);
                    }
                }
                if (base + stagedInputElements < fixedInputDimension) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            sum = simd_sum(sum);
            if (active && tiisg == 0) {
                output[row] = \(bt)(sum);
            }
        }
        """
    }


    private static func generateSpecializedDenseGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        stagedInputElements: Int,
        fixedInputDimension: Int? = nil,
        fixedOutputDimension: Int? = nil,
        fixedRowsPerThreadgroup: Int? = nil,
        fixedSimdgroups: Int? = nil,
        inputStaging: SpecializedDenseInputStaging = .bufferPrecision,
        accumulationStyle: SpecializedDenseAccumulationStyle = .indexed,
        unrollFactor: Int = 4,
        forcePointerIncrementLoop: Bool = false
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagesInputAsFloat = inputStaging.stagesAsFloat
        let stagedInputType = stagesInputAsFloat ? "float" : bt
        let stagedInputRead = stagesInputAsFloat ? "" : "float"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let inputDimensionExpr = fixedInputDimension.map { "\($0)u" } ?? "inputDimension"
        let outputDimensionExpr = fixedOutputDimension.map { "\($0)u" } ?? "outputDimension"
        let effectiveThreadsPerThreadgroupExpr = fixedSimdgroups.map { "SIMD_WIDTH * \($0)u" } ?? "threadsPerThreadgroup"
        let rowsPerThreadgroupExpr = fixedRowsPerThreadgroup.map { "\($0)u" } ?? "max(1u, threadsPerThreadgroup / SIMD_WIDTH)"
        let canElideInputBounds = if let fixedInputDimension {
            fixedInputDimension % (32 * effectiveUnroll) == 0
        } else {
            false
        }
        let unrolledAccumulate = (0..<effectiveUnroll).map { lane -> String in
            let accumulator = "sum"
            if lane == 0 {
                return "\(accumulator) += \(readWeight("weightRow[j]")) * \(stagedInputRead)(inputTile[j]);"
            }
            let offset = "\(lane)"
            let nextName = "next\(lane)"
            if canElideInputBounds {
                return "\(accumulator) += \(readWeight("weightRow[j + \(offset)]")) * \(stagedInputRead)(inputTile[j + \(offset)]);"
            }
            return """
                const uint \(nextName) = j + \(offset);
                if (\(nextName) < \(inputDimensionExpr)) {
                    \(accumulator) += \(readWeight("weightRow[\(nextName)]")) * \(stagedInputRead)(inputTile[\(nextName)]);
                }
                """
        }.joined(separator: "\n")
        let pointerAccumulate = (0..<effectiveUnroll).map { lane -> String in
            "sum += \(readWeight("weightLane[\(lane)]")) * \(stagedInputRead)(inputLane[\(lane)]);"
        }.joined(separator: "\n")
        let inputTileLoad: String
        if let fixedInputDimension, fixedInputDimension == stagedInputElements {
            inputTileLoad = stagesInputAsFloat ? "inputTile[j] = float(input[j]);" : "inputTile[j] = input[j];"
        } else {
            inputTileLoad = stagesInputAsFloat
                ? "inputTile[j] = j < \(inputDimensionExpr) ? float(input[j]) : 0.0f;"
                : "inputTile[j] = j < \(inputDimensionExpr) ? input[j] : \(bt)(0.0f);"
        }
        let usePointerIncrementLoop: Bool
        switch accumulationStyle {
        case .indexed:
            usePointerIncrementLoop = canElideInputBounds && forcePointerIncrementLoop
        case .pointerIncrement:
            usePointerIncrementLoop = canElideInputBounds
        }
        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            uint gid                               [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup             [[threads_per_threadgroup]]
        ) {
            const uint stagedInputElements = \(stagedInputElements);
            const uint rowsPerThreadgroup = \(rowsPerThreadgroupExpr);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            const bool active = row < \(outputDimensionExpr);

            threadgroup \(stagedInputType) inputTile[stagedInputElements];
            for (uint j = tid; j < stagedInputElements; j += \(effectiveThreadsPerThreadgroupExpr)) {
                \(inputTileLoad)
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sum = 0.0f;
            if (active) {
                device const \(wt)* weightRow = weight + row * \(inputDimensionExpr);
                \(usePointerIncrementLoop ? """
                device const \(wt)* weightLane = weightRow + tiisg * \(effectiveUnroll);
                threadgroup const \(stagedInputType)* inputLane = inputTile + tiisg * \(effectiveUnroll);
                for (uint j = tiisg * \(effectiveUnroll); j < \(inputDimensionExpr); j += SIMD_WIDTH * \(effectiveUnroll)) {
                    \(pointerAccumulate)
                    weightLane += SIMD_WIDTH * \(effectiveUnroll);
                    inputLane += SIMD_WIDTH * \(effectiveUnroll);
                }
                """ : """
                for (uint j = tiisg * \(effectiveUnroll); j < \(inputDimensionExpr); j += SIMD_WIDTH * \(effectiveUnroll)) {
                    \(unrolledAccumulate)
                }
                """)
            }
            sum = simd_sum(sum);
            if (active && tiisg == 0) {
                output[row] = \(bt)(sum);
            }
        }
        """
    }
}
