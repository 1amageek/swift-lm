extension MetalSourceGenerator {
// MARK: - Linear Kernels

    public static func generateMPPGEMM(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        // MPP tensor type for weight: bfloat for BF16, half for FP16
        let tensorWeightType = weightFormat == .bfloat16 ? "bfloat" : bt

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
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            using namespace mpp::tensor_ops;

            auto A = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                input, dextents<int32_t, 2>(inputDimension, sequenceLength));
            auto B = tensor<device \(tensorWeightType), dextents<int32_t, 2>, tensor_inline>(
                weight, dextents<int32_t, 2>(inputDimension, outputDimension));
            auto C = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                output, dextents<int32_t, 2>(outputDimension, sequenceLength));

            constexpr auto desc = matmul2d_descriptor(
                64, 32, dynamic_length_v<int>,
                false, true, false,
                matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;

            auto mA = A.slice(0, tgid.y * 64);
            auto mB = B.slice(0, tgid.x * 32);
            auto mC = C.slice(tgid.x * 32, tgid.y * 64);
            op.run(mA, mB, mC);
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

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            constant uint& sequenceLength          [[buffer(5)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]]
        ) {
            const uint rowsPerThreadgroup = 2;
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            const uint seqPos = gid.y;
            if (row >= outputDimension || seqPos >= sequenceLength) return;

            float sum = 0.0f;
            device const \(bt)* inputRow = input + seqPos * inputDimension;
            device const \(wt)* weightRow = weight + row * inputDimension;
            for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
                sum += \(readWeight("weightRow[j]")) * float(inputRow[j]);
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[seqPos * outputDimension + row] = \(bt)(sum);
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
            if (row >= outputDimension) return;

            threadgroup \(bt) inputTile[tileElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = weight + row * inputDimension;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup) {
                    const uint inputIndex = base + j;
                    inputTile[j] = inputIndex < inputDimension ? input[inputIndex] : \(bt)(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileCount = min(tileElements, inputDimension - base);
                for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                    sum += \(readWeight("weightRow[base + j]")) * float(inputTile[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[row] = \(bt)(sum);
            }
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
            if (row >= \(fixedOutputDimension.map { "\($0)u" } ?? "outputDimension")) return;

            threadgroup \(stagedInputType) inputTile[stagedInputElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = weight + row * fixedInputDimension;
            for (uint base = 0; base < fixedInputDimension; base += stagedInputElements) {
                device const \(bt)* inputTileSource = input + base + tid;
                for (uint j = tid; j < stagedInputElements; j += threadsPerThreadgroup) {
                    inputTile[j] = \(stagesInputAsFloat ? "float(inputTileSource[0])" : "inputTileSource[0]");
                    inputTileSource += threadsPerThreadgroup;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                device const \(wt)* tileWeight = weightRow + base + tiisg * \(effectiveUnroll);
                threadgroup const \(stagedInputType)* tileInput = inputTile + tiisg * \(effectiveUnroll);
                for (uint j = tiisg * \(effectiveUnroll); j < stagedInputElements; j += SIMD_WIDTH * \(effectiveUnroll)) {
                    \(unrolledAccumulate)
                    tileWeight += SIMD_WIDTH * \(effectiveUnroll);
                    tileInput += SIMD_WIDTH * \(effectiveUnroll);
                }
                if (base + stagedInputElements < fixedInputDimension) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
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
            if (row >= \(outputDimensionExpr)) return;

            threadgroup \(stagedInputType) inputTile[stagedInputElements];
            for (uint j = tid; j < stagedInputElements; j += \(effectiveThreadsPerThreadgroupExpr)) {
                \(inputTileLoad)
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sum = 0.0f;
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
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[row] = \(bt)(sum);
            }
        }
        """
    }
    public static func generateFusedSwiGLUProjection(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device const \(bt)* input          [[buffer(0)]],
            device const \(wt)* gateWeight     [[buffer(1)]],
            device const \(wt)* upWeight       [[buffer(2)]],
            device \(bt)* output               [[buffer(3)]],
            constant uint& inputDimension      [[buffer(4)]],
            constant uint& outputDimension     [[buffer(5)]],
            uint gid                           [[threadgroup_position_in_grid]],
            uint tid                           [[thread_index_in_threadgroup]],
            uint tiisg                         [[thread_index_in_simdgroup]],
            uint sgitg                         [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup         [[threads_per_threadgroup]]
        ) {
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            float gateSum = 0.0f;
            float upSum = 0.0f;
            device const \(wt)* gateRow = gateWeight + row * inputDimension;
            device const \(wt)* upRow = upWeight + row * inputDimension;
            for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
                float x = float(input[j]);
                gateSum += \(readWeight("gateRow[j]")) * x;
                upSum += \(readWeight("upRow[j]")) * x;
            }
            gateSum = simd_sum(gateSum);
            upSum = simd_sum(upSum);
            if (tiisg == 0) {
                float sig = 1.0f / (1.0f + exp(-gateSum));
                output[row] = \(bt)(gateSum * sig * upSum);
            }
        }
        """
    }

    /// Generate a decode-only fused gate/up projection specialized for inputDimension=2048.
    ///
    /// This mirrors the exact-shape dense GEMV family: the full hidden vector is staged once
    /// into threadgroup memory, then both branch projections accumulate from that shared tile.
    public static func generateInput2048FusedSwiGLUProjection(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        stagesInputAsFloat: Bool = true,
        fixedRowsPerThreadgroup: Int? = nil,
        fixedSimdgroups: Int? = nil,
        unrollFactor: Int = 4
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagedInputType = stagesInputAsFloat ? "float" : bt
        let stagedInputRead = stagesInputAsFloat ? "" : "float"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let effectiveThreadsPerThreadgroupExpr = fixedSimdgroups.map { "SIMD_WIDTH * \($0)u" } ?? "threadsPerThreadgroup"
        let rowsPerThreadgroupExpr = fixedRowsPerThreadgroup.map { "\($0)u" } ?? "max(1u, threadsPerThreadgroup / SIMD_WIDTH)"
        let gateAccumulate = (0..<effectiveUnroll).map { lane -> String in
            if lane == 0 {
                return "gateSum += \(readWeight("gateRow[0]")) * \(stagedInputRead)(inputLane[0]);"
            }
            let offset = "\(lane)"
            return "gateSum += \(readWeight("gateRow[\(offset)]")) * \(stagedInputRead)(inputLane[\(offset)]);"
        }.joined(separator: "\n")
        let upAccumulate = (0..<effectiveUnroll).map { lane -> String in
            if lane == 0 {
                return "upSum += \(readWeight("upRow[0]")) * \(stagedInputRead)(inputLane[0]);"
            }
            let offset = "\(lane)"
            return "upSum += \(readWeight("upRow[\(offset)]")) * \(stagedInputRead)(inputLane[\(offset)]);"
        }.joined(separator: "\n")

        return """
        kernel void \(name)(
            device const \(bt)* input          [[buffer(0)]],
            device const \(wt)* gateWeight     [[buffer(1)]],
            device const \(wt)* upWeight       [[buffer(2)]],
            device \(bt)* output               [[buffer(3)]],
            constant uint& inputDimension      [[buffer(4)]],
            constant uint& outputDimension     [[buffer(5)]],
            uint gid                           [[threadgroup_position_in_grid]],
            uint tid                           [[thread_index_in_threadgroup]],
            uint tiisg                         [[thread_index_in_simdgroup]],
            uint sgitg                         [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup         [[threads_per_threadgroup]]
        ) {
            const uint fixedInputDimension = 2048u;
            const uint rowsPerThreadgroup = \(rowsPerThreadgroupExpr);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            threadgroup \(stagedInputType) inputTile[fixedInputDimension];
            for (uint j = tid; j < fixedInputDimension; j += \(effectiveThreadsPerThreadgroupExpr)) {
                inputTile[j] = \(stagesInputAsFloat ? "float(input[j])" : "input[j]");
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float gateSum = 0.0f;
            float upSum = 0.0f;
            device const \(wt)* gateRow = gateWeight + row * fixedInputDimension + tiisg * \(effectiveUnroll);
            device const \(wt)* upRow = upWeight + row * fixedInputDimension + tiisg * \(effectiveUnroll);
            threadgroup const \(stagedInputType)* inputLane = inputTile + tiisg * \(effectiveUnroll);
            for (uint j = tiisg * \(effectiveUnroll); j < fixedInputDimension; j += SIMD_WIDTH * \(effectiveUnroll)) {
                \(gateAccumulate)
                \(upAccumulate)
                gateRow += SIMD_WIDTH * \(effectiveUnroll);
                upRow += SIMD_WIDTH * \(effectiveUnroll);
                inputLane += SIMD_WIDTH * \(effectiveUnroll);
            }
            gateSum = simd_sum(gateSum);
            upSum = simd_sum(upSum);
            if (tiisg == 0) {
                float sig = 1.0f / (1.0f + fast::exp(-gateSum));
                output[row] = \(bt)(gateSum * sig * upSum);
            }
        }
        """
    }
}
