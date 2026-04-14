extension MetalSourceGenerator {
// MARK: - Linear Argument Table Kernels

    public static func generateInput2048GEMVArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        fixedOutputDimension: Int,
        includesDimensionBindings: Bool = true,
        fixedRowsPerThreadgroup: Int? = nil,
        fixedSimdgroups: Int? = nil,
        stagesInputAsFloat: Bool = true,
        weightLayoutPolicy: Input2048WeightLayoutPolicy = .rowMajor,
        bf16ArgumentReadPolicy: Input2048BF16ArgumentReadPolicy = .scalar,
        unrollFactor: Int = 4
    ) -> String {
        _ = weightLayoutPolicy
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagedInputType = stagesInputAsFloat ? "float" : bt
        let stagedInputRead = { (index: String) in "float(inputTile[\(index)])" }
        let stagedInputStore = stagesInputAsFloat ? "inputTile[j] = float(args.input[j]);" : "inputTile[j] = args.input[j];"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let effectiveThreadsPerThreadgroup = fixedSimdgroups.map { "SIMD_WIDTH * \($0)u" } ?? "threadsPerThreadgroup"
        let rowsPerThreadgroupExpr = fixedRowsPerThreadgroup.map { "\($0)u" } ?? "max(1u, threadsPerThreadgroup / SIMD_WIDTH)"
        let requiresThreadsPerThreadgroupBuiltin = fixedSimdgroups == nil || fixedRowsPerThreadgroup == nil
        let inputStructName = "\(name)_args"
        let usesPairwiseWeightRead =
            (bf16ArgumentReadPolicy == .pairwise ||
             bf16ArgumentReadPolicy == .pairwisePointerInput ||
             bf16ArgumentReadPolicy == .pairwisePointerFloatInput) &&
            weightFormat == .bfloat16 &&
            effectiveUnroll.isMultiple(of: 2)
        let usesPointerInputRead =
            bf16ArgumentReadPolicy == .pairwisePointerInput
        let usesPointerFloatInputRead = bf16ArgumentReadPolicy == .pairwisePointerFloatInput
        let usesPacked4PointerInputRead =
            (bf16ArgumentReadPolicy == .packed4PointerInput ||
             bf16ArgumentReadPolicy == .packed4FixedPointerInput ||
             bf16ArgumentReadPolicy == .packed4ThreadgroupFixedPointerInput) &&
            weightFormat == .bfloat16 &&
            effectiveUnroll == 4
        let usesPacked4FixedPointerInputRead =
            (bf16ArgumentReadPolicy == .packed4FixedPointerInput ||
             bf16ArgumentReadPolicy == .packed4ThreadgroupFixedPointerInput) &&
            weightFormat == .bfloat16 &&
            effectiveUnroll == 4
        let usesPacked4ThreadgroupFixedPointerInputRead =
            bf16ArgumentReadPolicy == .packed4ThreadgroupFixedPointerInput &&
            weightFormat == .bfloat16 &&
            effectiveUnroll == 4
        let usesBlockedRows8Tile128Layout =
            weightLayoutPolicy == .blockedRows8Tiles128 &&
            usesPairwiseWeightRead &&
            fixedRowsPerThreadgroup == 8 &&
            fixedOutputDimension.isMultiple(of: 8)
        let usesBlockedRows4Tile128Layout =
            weightLayoutPolicy == .blockedRows4Tiles128 &&
            usesPacked4PointerInputRead &&
            fixedRowsPerThreadgroup == 4 &&
            fixedOutputDimension.isMultiple(of: 4)
        let weightRowDeclaration = (usesPairwiseWeightRead || usesPacked4PointerInputRead)
            ? ""
            : "device const \(wt)* weightRow = args.weight + row * 2048u;"
        let pairCount = effectiveUnroll / 2
        let nextIndices = (1..<effectiveUnroll).map { lane in
            "const uint next\(lane) = j + \(lane);"
        }.joined(separator: "\n")
        let unrolledAccumulate = (0..<effectiveUnroll).map { lane -> String in
            if lane == 0 {
                return "sum += \(readWeight("weightRow[j]")) * \(stagedInputRead("j"));"
            }
            return "sum += \(readWeight("weightRow[next\(lane)]")) * \(stagedInputRead("next\(lane)"));"
        }.joined(separator: "\n")
        let pairwiseAccumulate = (0..<pairCount).map { pair -> String in
            let base = pair * 2
            if usesPointerInputRead {
                return """
                    float2 w\(pair) = bf16x2_to_float2(weightLane[\(pair)]);
                    sum += w\(pair).x * float(inputLane[\(base)]);
                    sum += w\(pair).y * float(inputLane[\(base + 1)]);
                    """
            }
            if usesPointerFloatInputRead {
                return """
                    float2 w\(pair) = bf16x2_to_float2(weightLane[\(pair)]);
                    sum += w\(pair).x * inputLane[\(base)];
                    sum += w\(pair).y * inputLane[\(base + 1)];
                    """
            }
            return """
                float2 w\(pair) = bf16x2_to_float2(weightLane[\(pair)]);
                sum += w\(pair).x * \(stagedInputRead("j + \(base)"));
                sum += w\(pair).y * \(stagedInputRead("j + \(base + 1)"));
                """
        }.joined(separator: "\n")
        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            \(includesDimensionBindings ? "constant uint& inputDimension             [[buffer(3)]],\n            constant uint& outputDimension            [[buffer(4)]]," : "")
            uint gid                                  [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]]\(requiresThreadsPerThreadgroupBuiltin ? ",\n            uint threadsPerThreadgroup                [[threads_per_threadgroup]]" : "")
        ) {
            const uint stagedInputElements = 2048u;
            const uint rowsPerThreadgroup = \(rowsPerThreadgroupExpr);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= \(fixedOutputDimension)u) return;

            threadgroup \(stagedInputType) inputTile[stagedInputElements];
            for (uint j = tid; j < stagedInputElements; j += \(effectiveThreadsPerThreadgroup)) {
                \(stagedInputStore)
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sum = 0.0f;
            \(weightRowDeclaration)
            \(usesPacked4PointerInputRead ? """
            \(usesBlockedRows4Tile128Layout
                ? "device const ushort4* weightLane = (device const ushort4*)args.weight + gid * 2048u + sgitg * 32u + tiisg;"
                : usesPacked4ThreadgroupFixedPointerInputRead
                    ? "device const ushort4* weightLane = (device const ushort4*)args.weight + gid * 2048u + sgitg * 512u + tiisg;"
                : "device const ushort4* weightLane = (device const ushort4*)(args.weight + row * 2048u) + tiisg;")
            threadgroup const \(bt)* inputLane = inputTile + tiisg * 4;
            \(usesPacked4FixedPointerInputRead
                ? "for (uint iteration = 0; iteration < stagedInputElements / (SIMD_WIDTH * 4u); ++iteration) {"
                : "for (uint j = tiisg * 4; j < 2048u; j += SIMD_WIDTH * 4) {")
                float4 w = bf16x4_to_float4(weightLane[0]);
                sum += w.x * float(inputLane[0]);
                sum += w.y * float(inputLane[1]);
                sum += w.z * float(inputLane[2]);
                sum += w.w * float(inputLane[3]);
                weightLane += \(usesBlockedRows4Tile128Layout ? "128u" : "SIMD_WIDTH");
                inputLane += SIMD_WIDTH * 4;
            }
            """ : usesPairwiseWeightRead ? """
            \(usesBlockedRows8Tile128Layout
                ? "device const ushort2* weightLane = (device const ushort2*)args.weight + gid * 8192u + sgitg * 64u + tiisg * \(pairCount);"
                : "device const ushort2* weightLane = (device const ushort2*)(args.weight + row * 2048u) + tiisg * \(pairCount);")
            \(usesPointerInputRead ? "threadgroup const \(bt)* inputLane = inputTile + tiisg * \(effectiveUnroll);" : "")
            \(usesPointerFloatInputRead ? "threadgroup const float* inputLane = inputTile + tiisg * \(effectiveUnroll);" : "")
            for (uint j = tiisg * \(effectiveUnroll); j < 2048u; j += SIMD_WIDTH * \(effectiveUnroll)) {
                \(pairwiseAccumulate)
                weightLane += \(usesBlockedRows8Tile128Layout ? "512u" : "SIMD_WIDTH * \(pairCount)");
                \(usesPointerInputRead || usesPointerFloatInputRead ? "inputLane += SIMD_WIDTH * \(effectiveUnroll);" : "")
            }
            """ : """
            for (uint j = tiisg * \(effectiveUnroll); j < 2048u; j += SIMD_WIDTH * \(effectiveUnroll)) {
                \(nextIndices)
                \(unrolledAccumulate)
            }
            """)
            sum = simd_sum(sum);
            if (tiisg == 0) {
                args.output[row] = \(bt)(sum);
            }
        }
        """
    }

    public static func generateVocabGEMVArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& inputDimension             [[buffer(3)]],
            constant uint& outputDimension            [[buffer(4)]],
            uint gid                                  [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup                [[threads_per_threadgroup]]
        ) {
            const uint stagedInputElements = 2048u;
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            threadgroup \(bt) inputTile[stagedInputElements];
            for (uint j = tid; j < stagedInputElements; j += threadsPerThreadgroup) {
                inputTile[j] = args.input[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sum = 0.0f;
            device const \(wt)* weightRow = args.weight + row * 2048u + tiisg;
            threadgroup const \(bt)* inputLane = inputTile + tiisg;
            for (uint j = tiisg; j < 2048u; j += SIMD_WIDTH) {
                sum += \(readWeight("weightRow[0]")) * float(inputLane[0]);
                weightRow += SIMD_WIDTH;
                inputLane += SIMD_WIDTH;
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                args.output[row] = \(bt)(sum);
            }
        }
        """
    }

    /// Generate a GEMV kernel specialized for decode projections with inputDimension=8192.
    ///
    /// This family keeps the tiled structure to preserve occupancy, but fixes the
    /// input dimension and tile size so the inner loop can avoid dynamic bounds work.

    public static func generateInput8192TiledGEMVArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        fixedOutputDimension: Int? = nil,
        tileElements: Int = 1_024,
        unrollFactor: Int = 4
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagedInputType = "float"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let inputStructName = "\(name)_args"
        let unrolledAccumulate = (0..<effectiveUnroll).map { lane -> String in
            "sum += \(readWeight("tileWeight[\(lane)]")) * (tileInput[\(lane)]);"
        }.joined(separator: "\n")

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& inputDimension             [[buffer(3)]],
            constant uint& outputDimension            [[buffer(4)]],
            uint gid                                  [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup                [[threads_per_threadgroup]]
        ) {
            const uint fixedInputDimension = 8192u;
            const uint stagedInputElements = \(tileElements);
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= \(fixedOutputDimension.map { "\($0)u" } ?? "outputDimension")) return;

            threadgroup \(stagedInputType) inputTile[stagedInputElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = args.weight + row * fixedInputDimension;
            for (uint base = 0; base < fixedInputDimension; base += stagedInputElements) {
                device const \(bt)* inputTileSource = args.input + base + tid;
                for (uint j = tid; j < stagedInputElements; j += threadsPerThreadgroup) {
                    inputTile[j] = float(inputTileSource[0]);
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
                args.output[row] = \(bt)(sum);
            }
        }
        """
    }

    public static func generateGEMVArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        tileElements: Int = 256
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let inputStructName = "\(name)_args"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& inputDimension             [[buffer(3)]],
            constant uint& outputDimension            [[buffer(4)]],
            uint gid                                  [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup                [[threads_per_threadgroup]]
        ) {
            const uint tileElements = \(tileElements);
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            threadgroup \(bt) inputTile[tileElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = args.weight + row * inputDimension;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup) {
                    const uint inputIndex = base + j;
                    inputTile[j] = inputIndex < inputDimension ? args.input[inputIndex] : \(bt)(0.0f);
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
                args.output[row] = \(bt)(sum);
            }
        }
        """
    }
}
