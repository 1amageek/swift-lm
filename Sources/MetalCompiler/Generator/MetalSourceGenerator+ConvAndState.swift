extension MetalSourceGenerator {
private static func convStateStorageType(weightFormat: WeightFormat) -> String {
    weightFormat.isBFloat16 ? "uint16_t" : "half"
}

private static func convStateRead(_ expression: String, weightFormat: WeightFormat) -> String {
    weightFormat.isBFloat16 ? "bf16_to_float(\(expression))" : "float(\(expression))"
}

private static func convStateWrite(_ expression: String, weightFormat: WeightFormat) -> String {
    weightFormat.isBFloat16 ? "float_to_bf16(\(expression))" : "half(\(expression))"
}

private static func decodeActivationStorageValue(_ expression: String, weightFormat: WeightFormat) -> String {
    if weightFormat.isBFloat16 {
        return "float(bfloat(\(expression)))"
    }
    if weightFormat.isFloat32 {
        return expression
    }
    return "float(half(\(expression)))"
}

// MARK: - Conv1d

/// Generate conv_state_update kernel (decode: single token with persistent state).
public static func generateConvStateUpdate(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat
) -> String {
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let ct = convStateStorageType(weightFormat: weightFormat)
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
    let readConvState = { (expr: String) in convStateRead(expr, weightFormat: weightFormat) }
    let writeConvState = { (expr: String) in convStateWrite(expr, weightFormat: weightFormat) }

    return """
    kernel void \(name)(
        device \(ct)* convState          [[buffer(0)]],
        device const \(bt)* inProjOutput [[buffer(1)]],
        device const \(wt)* weight       [[buffer(2)]],
        device \(bt)* output             [[buffer(3)]],
        constant uint& dimension         [[buffer(4)]],
        constant uint& kernelSize        [[buffer(5)]],
        uint gid                         [[thread_position_in_grid]]
    ) {
        if (gid >= dimension) return;

        float B = float(inProjOutput[gid]);
        float C = float(inProjOutput[dimension + gid]);
        float x = float(inProjOutput[2 * dimension + gid]);
        float Bx = B * x;

        for (uint k = 0; k < kernelSize - 1; k++) {
            convState[k * dimension + gid] = convState[(k + 1) * dimension + gid];
        }
        convState[(kernelSize - 1) * dimension + gid] = \(writeConvState("Bx"));

        float convOut = 0.0f;
        for (uint k = 0; k + 1 < kernelSize; k++) {
            convOut += \(readConvState("convState[k * dimension + gid]")) * \(readWeight("weight[gid * kernelSize + k]"));
        }
        convOut += Bx * \(readWeight("weight[gid * kernelSize + (kernelSize - 1)]"));

        output[gid] = \(bt)(C * convOut);
    }
    """
}

public static func generateConvStateUpdateArgumentTableVariant(
    name: String,
    argumentBufferIndex: Int,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat
) -> String {
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let ct = convStateStorageType(weightFormat: weightFormat)
    let inputStructName = "\(name)_args"
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
    let readConvState = { (expr: String) in convStateRead(expr, weightFormat: weightFormat) }
    let writeConvState = { (expr: String) in convStateWrite(expr, weightFormat: weightFormat) }

    return """
    struct \(inputStructName) {
        device \(ct)* convState [[id(0)]];
        device const \(bt)* inProjOutput [[id(1)]];
        device const \(wt)* weight [[id(2)]];
        device \(bt)* output [[id(3)]];
    };

    kernel void \(name)(
        constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
        constant uint& dimension                  [[buffer(4)]],
        constant uint& kernelSize                 [[buffer(5)]],
        uint gid                                  [[thread_position_in_grid]]
    ) {
        if (gid >= dimension) return;

        float B = float(args.inProjOutput[gid]);
        float C = float(args.inProjOutput[dimension + gid]);
        float x = float(args.inProjOutput[2 * dimension + gid]);
        float Bx = B * x;

        for (uint k = 0; k < kernelSize - 1; k++) {
            args.convState[k * dimension + gid] = args.convState[(k + 1) * dimension + gid];
        }
        args.convState[(kernelSize - 1) * dimension + gid] = \(writeConvState("Bx"));

        float convOut = 0.0f;
        for (uint k = 0; k + 1 < kernelSize; k++) {
            convOut += \(readConvState("args.convState[k * dimension + gid]")) * \(readWeight("args.weight[gid * kernelSize + k]"));
        }
        convOut += Bx * \(readWeight("args.weight[gid * kernelSize + (kernelSize - 1)]"));

        args.output[gid] = \(bt)(C * convOut);
    }
    """
}

/// Generate a decode-only fused ShortConv input projection and state update.
///
/// This specializes the LFM2.5 A1B shape: hidden size 2048, BF16 weights, and
/// BF16 decode buffers. It computes B/C/x for one channel in a single SIMDgroup,
/// rounds those projection outputs through BF16 to preserve the unfused contract,
/// updates the depthwise convolution state, and writes the gated convolution
/// output consumed by the following out projection.
public static func generateShortConvInProjUpdateBF16(name: String) -> String {
    """
    kernel void \(name)(
        device const bfloat* input              [[buffer(0)]],
        device const uint16_t* inProjWeight     [[buffer(1)]],
        device uint16_t* convState              [[buffer(2)]],
        device const uint16_t* convWeight       [[buffer(3)]],
        device bfloat* output                  [[buffer(4)]],
        constant uint& dimension                [[buffer(5)]],
        constant uint& kernelSize               [[buffer(6)]],
        uint gid                                [[threadgroup_position_in_grid]],
        uint tid                                [[thread_index_in_threadgroup]],
        uint tiisg                              [[thread_index_in_simdgroup]],
        uint sgitg                              [[simdgroup_index_in_threadgroup]],
        uint threadsPerThreadgroup              [[threads_per_threadgroup]]
    ) {
        const uint fixedDimension = 2048u;
        const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
        const uint channel = gid * rowsPerThreadgroup + sgitg;
        const bool active = channel < dimension && dimension == fixedDimension;

        threadgroup bfloat inputTile[fixedDimension];
        for (uint j = tid; j < fixedDimension; j += threadsPerThreadgroup) {
            inputTile[j] = input[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float bSum = 0.0f;
        float cSum = 0.0f;
        float xSum = 0.0f;
        if (active) {
            device const uint16_t* bRow = inProjWeight + channel * fixedDimension;
            device const uint16_t* cRow = inProjWeight + (fixedDimension + channel) * fixedDimension;
            device const uint16_t* xRow = inProjWeight + (2u * fixedDimension + channel) * fixedDimension;
            device const ushort4* bLane = (device const ushort4*)bRow + tiisg;
            device const ushort4* cLane = (device const ushort4*)cRow + tiisg;
            device const ushort4* xLane = (device const ushort4*)xRow + tiisg;
            threadgroup const bfloat4* inputLane = (threadgroup const bfloat4*)inputTile + tiisg;
            for (uint j = tiisg * 4u; j < fixedDimension; j += SIMD_WIDTH * 4u) {
                const float4 value = float4(inputLane[0]);
                const float4 bWeight = bf16x4_to_float4(bLane[0]);
                const float4 cWeight = bf16x4_to_float4(cLane[0]);
                const float4 xWeight = bf16x4_to_float4(xLane[0]);
                bSum += bWeight[0] * value[0];
                bSum += bWeight[1] * value[1];
                bSum += bWeight[2] * value[2];
                bSum += bWeight[3] * value[3];
                cSum += cWeight[0] * value[0];
                cSum += cWeight[1] * value[1];
                cSum += cWeight[2] * value[2];
                cSum += cWeight[3] * value[3];
                xSum += xWeight[0] * value[0];
                xSum += xWeight[1] * value[1];
                xSum += xWeight[2] * value[2];
                xSum += xWeight[3] * value[3];
                bLane += SIMD_WIDTH;
                cLane += SIMD_WIDTH;
                xLane += SIMD_WIDTH;
                inputLane += SIMD_WIDTH;
            }
        }

        bSum = simd_sum(bSum);
        cSum = simd_sum(cSum);
        xSum = simd_sum(xSum);

        if (active && tiisg == 0) {
            const float bValue = float(bfloat(bSum));
            const float cValue = float(bfloat(cSum));
            const float xValue = float(bfloat(xSum));
            const float bx = bValue * xValue;

            float convOut = 0.0f;
            for (uint k = 0; k + 1u < kernelSize; k++) {
                const uint dst = k * fixedDimension + channel;
                const uint src = (k + 1u) * fixedDimension + channel;
                const uint16_t shifted = convState[src];
                convState[dst] = shifted;
                convOut += bf16_to_float(shifted) * bf16_to_float(convWeight[channel * kernelSize + k]);
            }

            convState[(kernelSize - 1u) * fixedDimension + channel] = float_to_bf16(bx);
            convOut += bx * bf16_to_float(convWeight[channel * kernelSize + (kernelSize - 1u)]);
            output[channel] = bfloat(cValue * convOut);
        }
    }
    """
}

/// Generate conv1d_causal_seq kernel (prefill: temporal conv across positions).
public static func generateConv1dCausalSeq(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat
) -> String {
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
    let quantizedStateValue = if weightFormat.isBFloat16 {
        { (expr: String) in "bf16_to_float(float_to_bf16(\(expr)))" }
    } else {
        { (expr: String) in "float(half(\(expr)))" }
    }

    return """
    kernel void \(name)(
        device const \(bt)* input         [[buffer(0)]],
        device const \(wt)* weight        [[buffer(1)]],
        device \(bt)* output              [[buffer(2)]],
        constant uint& convDim            [[buffer(3)]],
        constant uint& inProjDim          [[buffer(4)]],
        constant uint& kernelSize         [[buffer(5)]],
        constant uint& sequenceLength     [[buffer(6)]],
        constant uint& inputRowStride     [[buffer(7)]],
        constant uint& outputRowStride    [[buffer(8)]],
        uint2 gid                         [[thread_position_in_grid]]
    ) {
        uint ch = gid.x;
        uint pos = gid.y;
        if (ch >= convDim || pos >= sequenceLength) return;

        // Sequence prefill packs the in_proj output (3 * convDim columns: B|C|x)
        // into scratch slots that span slotDimension (>= inProjDim). Using the
        // narrower `inProjDim` as a row stride misaligns positions >= 1 because
        // the GEMM producer writes at slot stride. The conv output is consumed
        // by out_proj at slot stride too, so writes use `outputRowStride`.
        // Standalone unit tests can pass `inputRowStride == inProjDim` and
        // `outputRowStride == convDim` for native-packed buffers.
        float convOut = 0.0f;
        for (uint k = 0; k < kernelSize; k++) {
            int srcPos = int(pos) - int(kernelSize - 1) + int(k);
            if (srcPos >= 0) {
                float B = float(input[uint(srcPos) * inputRowStride + ch]);
                float x = float(input[uint(srcPos) * inputRowStride + 2 * convDim + ch]);
                float Bx = B * x;
                if (uint(srcPos) != pos) {
                    Bx = \(quantizedStateValue("Bx"));
                }
                convOut += Bx * \(readWeight("weight[ch * kernelSize + k]"));
            }
        }

        float C = float(input[pos * inputRowStride + convDim + ch]);
        output[pos * outputRowStride + ch] = \(bt)(C * convOut);
    }
    """
}

/// Generate extract_conv_state kernel (saves last kernelSize positions' B*x to conv_state).
public static func generateExtractConvState(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat
) -> String {
    let bt = bufferPrecision.metalType
    let ct = convStateStorageType(weightFormat: weightFormat)
    let writeConvState = { (expr: String) in convStateWrite(expr, weightFormat: weightFormat) }

    return """
    kernel void \(name)(
        device const \(bt)* inProjOutput   [[buffer(0)]],
        device \(ct)* convState            [[buffer(1)]],
        constant uint& convDim             [[buffer(2)]],
        constant uint& inProjDim           [[buffer(3)]],
        constant uint& kernelSize          [[buffer(4)]],
        constant uint& sequenceLength      [[buffer(5)]],
        constant uint& inputRowStride      [[buffer(6)]],
        uint2 gid                          [[thread_position_in_grid]]
    ) {
        uint ch = gid.x;
        uint k = gid.y;
        if (ch >= convDim || k >= kernelSize) return;
        // Read the in_proj scratch slot at `inputRowStride` (slotDimension in
        // production prefill, or inProjDim in standalone unit tests) — the same
        // stride the conv1d_causal_seq kernel and the producing GEMM use.
        int srcPos = int(sequenceLength) - int(kernelSize) + int(k);
        if (srcPos >= 0 && uint(srcPos) < sequenceLength) {
            float B = float(inProjOutput[uint(srcPos) * inputRowStride + ch]);
            float x = float(inProjOutput[uint(srcPos) * inputRowStride + 2 * convDim + ch]);
            convState[k * convDim + ch] = \(writeConvState("B * x"));
        } else {
            convState[k * convDim + ch] = 0;
        }
    }
    """
}

// MARK: - State Space

/// Weight-independent SSM helper functions (emitted once per compilation unit).
public static func generateSSMWeightIndependentHelpers() -> String {
    return """
    inline float stable_softplus(float x) { return max(x, 0.0f) + log(1.0f + exp(-abs(x))); }
    inline float stable_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
    inline float compute_l2_inv_norm(
        threadgroup float* vec,
        uint dim,
        uint tid,
        uint tgSize,
        threadgroup float* scratch
    ) {
        float sumSq = 0.0f;
        for (uint d = tid; d < dim; d += tgSize) sumSq += vec[d] * vec[d];
        sumSq = simd_sum(sumSq);
        if (tid % SIMD_WIDTH == 0) scratch[tid / SIMD_WIDTH] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint i = 0; i < (tgSize + SIMD_WIDTH - 1) / SIMD_WIDTH; ++i) total += scratch[i];
            scratch[0] = rsqrt(total + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return scratch[0];
    }
    """
}

/// Weight-dependent conv_silu overload (emitted per weight format).
/// MSL function overloading resolves the correct variant based on convWeight pointer type.
public static func generateSSMConvSiluHelper(weightFormat: WeightFormat) -> String {
    let wt = weightFormat.bufferType
    let ct = convStateStorageType(weightFormat: weightFormat)
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
    let readConvState = { (expr: String) in convStateRead(expr, weightFormat: weightFormat) }

    return """
    inline float conv_silu(
        device const \(ct)* convState,
        device const \(wt)* convWeight,
        uint channel,
        uint convDimension,
        uint convKernelSize
    ) {
        float sum = 0.0f;
        for (uint k = 0; k < convKernelSize; ++k) {
            sum += \(readConvState("convState[k * convDimension + channel]"))
                * \(readWeight("convWeight[channel * convKernelSize + k]"));
        }
        return sum * stable_sigmoid(sum);
    }
    """
}

/// Combined SSM helper source for a single weight format (convenience for library use).
public static func generateSSMHelperSource(weightFormat: WeightFormat) -> String {
    return generateSSMWeightIndependentHelpers() + "\n" + generateSSMConvSiluHelper(weightFormat: weightFormat)
}

public static func generateSSMRecurrence(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat,
    convDimension: Int,
    maxThreadgroupSize: Int,
    headCount: Int,
    groupCount: Int,
    keyHeadDimension: Int,
    valueHeadDimension: Int,
    shareRMSScale: Bool = false
) -> String {
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let ct = convStateStorageType(weightFormat: weightFormat)
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
    let readConvState = { (expr: String) in convStateRead(expr, weightFormat: weightFormat) }
    let writeConvState = { (expr: String) in convStateWrite(expr, weightFormat: weightFormat) }
    let activationStorageValue: (String) -> String = { expr in
        switch bufferPrecision {
        case .bfloat16:
            return "float(bfloat(\(expr)))"
        case .float16:
            return "float(half(\(expr)))"
        case .float32, .float32Decode:
            return expr
        }
    }

    // Per-threadgroup (per key-group) local channel cache: Q (dk) + K (dk) + V (headsPerGroup*dv).
    // Sized at compile time using the fragment's known dimensions.
    let safeGroupCount = max(groupCount, 1)
    let headsPerGroup = max(1, headCount / safeGroupCount)
    let localDim = 2 * keyHeadDimension + headsPerGroup * valueHeadDimension
    let rmsScaleCacheDeclaration = shareRMSScale
        ? "\n        threadgroup float rmsScaleCache[\(headsPerGroup)];"
        : ""
    let rmsPhase = shareRMSScale
        ? """
            {
                const uint threadsPerHead = min(tgSize / max(headsPerGroup, 1u), dv);
                const uint activeThreads = headsPerGroup * threadsPerHead;
                if (tid < activeThreads) {
                    const uint localHead = tid / threadsPerHead;
                    const uint localTid = tid % threadsPerHead;
                    if (localTid == 0) {
                        float totalNormSq = 0.0f;
                        for (uint t = 0; t < threadsPerHead; ++t) {
                            totalNormSq += normPartials[localHead * threadsPerHead + t];
                        }
                        rmsScaleCache[localHead] = rsqrt(totalNormSq / float(dv) + 1e-6f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < activeThreads) {
                    const uint localHead = tid / threadsPerHead;
                    const uint headIndex = headStart + localHead;
                    const uint localTid = tid % threadsPerHead;
                    const uint dChunk = dv / threadsPerHead;
                    const uint dStart = localTid * dChunk;
                    const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;
                    float rmsScale = rmsScaleCache[localHead];
                    for (uint d = dStart; d < dEnd; ++d) {
                        float normed = dotCache[localHead * dv + d] * rmsScale * normWeight[d];
                        float z = float(projectedZ[headIndex * dv + d]);
                        output[headIndex * dv + d] = \(bt)(normed * z * stable_sigmoid(z));
                    }
                }
            }
        """
        : """
            {
                const uint threadsPerHead = min(tgSize / max(headsPerGroup, 1u), dv);
                const uint activeThreads = headsPerGroup * threadsPerHead;
                if (tid < activeThreads) {
                    const uint localHead = tid / threadsPerHead;
                    const uint headIndex = headStart + localHead;
                    const uint localTid = tid % threadsPerHead;
                    const uint dChunk = dv / threadsPerHead;
                    const uint dStart = localTid * dChunk;
                    const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;
                    float totalNormSq = 0.0f;
                    for (uint t = 0; t < threadsPerHead; ++t) {
                        totalNormSq += normPartials[localHead * threadsPerHead + t];
                    }
                    float rmsScale = rsqrt(totalNormSq / float(dv) + 1e-6f);
                    for (uint d = dStart; d < dEnd; ++d) {
                        float normed = dotCache[localHead * dv + d] * rmsScale * normWeight[d];
                        float z = float(projectedZ[headIndex * dv + d]);
                        output[headIndex * dv + d] = \(bt)(normed * z * stable_sigmoid(z));
                    }
                }
            }
        """

    return """
    kernel void \(name)(
        device const \(bt)* projectedQKV [[buffer(0)]],
        device const \(bt)* projectedZ [[buffer(1)]],
        device const \(bt)* projectedBeta [[buffer(2)]],
        device const \(bt)* projectedAlpha [[buffer(3)]],
        device const \(wt)* convWeight [[buffer(4)]],
        device const float* normWeight [[buffer(5)]],
        device const \(wt)* dtBias [[buffer(6)]],
        device const float* aLog [[buffer(7)]],
        device float* recurrentState [[buffer(8)]],
        device \(ct)* convState [[buffer(9)]],
        device \(bt)* output [[buffer(10)]],
        constant uint& numHeads [[buffer(11)]],
        constant uint& groupCount [[buffer(12)]],
        constant uint& keyDimension [[buffer(13)]],
        constant uint& valueDimension [[buffer(14)]],
        constant uint& convKernelSize [[buffer(15)]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgSize [[threads_per_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]]
    ) {
        const uint dk = keyDimension;
        const uint dv = valueDimension;
        const uint keyGroupDim = groupCount * dk;
        const uint convDim = 2 * keyGroupDim + numHeads * dv;
        const uint safeGroupCount = max(groupCount, 1u);
        const uint headsPerGroup = max(1u, numHeads / safeGroupCount);

        // Each threadgroup owns exactly one key-group. Work across key-groups
        // is dispatched in parallel (grid = (groupCount, 1, 1)). Every
        // threadgroup touches disjoint conv channels and a disjoint recurrent
        // state slice, so no cross-TG synchronization is required.
        const uint groupIndex = tgid;
        if (groupIndex >= safeGroupCount) {
            return;
        }
        const uint headStart = groupIndex * headsPerGroup;

        // Global conv-channel offsets owned by this threadgroup.
        const uint qBaseGlobal = groupIndex * dk;
        const uint kBaseGlobal = keyGroupDim + groupIndex * dk;
        const uint vBaseGlobal = 2u * keyGroupDim + headStart * dv;

        // Local channel layout inside convSiluCache:
        //   [0, dk)            : Q for this group
        //   [dk, 2*dk)         : K for this group
        //   [2*dk, localDim)   : V for this group (headsPerGroup * dv values)
        const uint localDim = 2u * dk + headsPerGroup * dv;

        threadgroup float convSiluCache[\(localDim)];
        threadgroup float dotCache[\(headsPerGroup * valueHeadDimension)];
        threadgroup float normPartials[\(maxThreadgroupSize)];\(rmsScaleCacheDeclaration)

        // Phase 1: fused conv-shift + SiLU for this threadgroup's owned channels.
        // Each threadgroup touches only its own Q/K/V channels in convState and
        // convWeight — disjoint partitions, no cross-TG write conflicts.
        for (uint localCh = tid; localCh < localDim; localCh += tgSize) {
            uint globalCh;
            if (localCh < dk) {
                globalCh = qBaseGlobal + localCh;
            } else if (localCh < 2u * dk) {
                globalCh = kBaseGlobal + (localCh - dk);
            } else {
                globalCh = vBaseGlobal + (localCh - 2u * dk);
            }

            float sum = 0.0f;
            for (uint k = 0; k + 1 < convKernelSize; ++k) {
                float val = \(readConvState("convState[(k + 1) * convDim + globalCh]"));
                convState[k * convDim + globalCh] = \(writeConvState("val"));
                sum += val * \(readWeight("convWeight[globalCh * convKernelSize + k]"));
            }
            float newVal = float(projectedQKV[globalCh]);
            convState[(convKernelSize - 1) * convDim + globalCh] = \(writeConvState("newVal"));
            sum += newVal * \(readWeight("convWeight[globalCh * convKernelSize + convKernelSize - 1]"));
            convSiluCache[localCh] = sum * stable_sigmoid(sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: per-head recurrence restricted to this threadgroup's owned heads.
        // recurrentState partitioning is head-disjoint → disjoint per threadgroup.
        {
            const uint threadsPerHead = min(tgSize / max(headsPerGroup, 1u), dv);
            const uint activeThreads = headsPerGroup * threadsPerHead;
            if (tid < activeThreads) {
                const uint localHead = tid / threadsPerHead;
                const uint headIndex = headStart + localHead;
                const uint localTid = tid % threadsPerHead;
                const uint dChunk = dv / threadsPerHead;
                const uint dStart = localTid * dChunk;
                const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;

                float decay = exp(-exp(aLog[headIndex]) * stable_softplus(float(projectedAlpha[headIndex]) + \(readWeight("dtBias[headIndex]"))));
                float beta = stable_sigmoid(float(projectedBeta[headIndex]));
                device float* state = recurrentState + headIndex * dk * dv;

                // Local cache indices for this threadgroup.
                const uint qBase = 0u;
                const uint kBase = dk;
                const uint vBase = 2u * dk + localHead * dv;

                // Precompute Q/K norms and kqSum = Σ Q[j]·K[j] once per head
                // (constant across d — eliminates redundant work inside the d loop).
                float qNormSq = 0.0f;
                float kNormSq = 0.0f;
                float kqSum = 0.0f;
                for (uint j = 0; j < dk; ++j) {
                    float q = convSiluCache[qBase + j];
                    float k = convSiluCache[kBase + j];
                    qNormSq += q * q;
                    kNormSq += k * k;
                    kqSum += q * k;
                }
                float qInv = rsqrt(qNormSq + 1e-6f) * rsqrt(float(dk));
                float kInv = rsqrt(kNormSq + 1e-6f);

                // Algebraic identity used below:
                //   state_after[j] = state_before[j]·decay + K[j]·kInvDelta
                //   dot = Σ state_after[j]·Q[j]
                //       = Σ (state_before[j]·decay)·Q[j] + kInvDelta · Σ K[j]·Q[j]
                //       = sqSum + kInvDelta · kqSum
                // This lets us compute `dot` from sqSum (accumulated during the
                // decay pass) + a precomputed kqSum, eliminating the second Σ
                // over updated state. Per j we now do 2 state reads + 1 state
                // write (vs 3 reads + 2 writes previously).
                float localNormSq = 0.0f;
                for (uint d = dStart; d < dEnd; ++d) {
                    // Pass 1: read state, accumulate kvmemRaw and sqSum.
                    // No state write here — deferred to Pass 2 for a single full write.
                    float kvmemRaw = 0.0f;
                    float sqSum = 0.0f;
                    for (uint j = 0; j < dk; ++j) {
                        float s = state[j * dv + d] * decay;
                        kvmemRaw += s * convSiluCache[kBase + j];
                        sqSum += s * convSiluCache[qBase + j];
                    }

                    float delta = beta * (convSiluCache[vBase + d] - kvmemRaw * kInv);
                    float kInvDelta = kInv * delta;
                    float dot = (sqSum + kInvDelta * kqSum) * qInv;

                    // Pass 2: final state = decay·old + K·kInvDelta (full write, no RMW).
                    for (uint j = 0; j < dk; ++j) {
                        state[j * dv + d] = state[j * dv + d] * decay + convSiluCache[kBase + j] * kInvDelta;
                    }

                    float storedDot = \(activationStorageValue("dot"));
                    dotCache[localHead * dv + d] = storedDot;
                    output[headIndex * dv + d] = \(bt)(storedDot);
                    localNormSq += dot * dot;
                }

                normPartials[tid] = localNormSq;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: RMS norm + gated activation for owned heads.
        \(rmsPhase)
    }
    """
}

public static func generateSSMRecurrenceSequence(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat,
    convDimension: Int,
    maxThreadgroupSize: Int,
    headCount: Int,
    groupCount: Int,
    keyHeadDimension: Int,
    valueHeadDimension: Int,
    shareRMSScale: Bool = false,
    prewriteDecayedState: Bool = false,
    emitsGroupOwnedPartialProjection: Bool = false,
    emitsPartitionOwnedPartialProjection: Bool = false,
    parallelQKReduction: Bool = false,
    cacheHeadParameters: Bool = false,
    parallelStateUpdate: Bool = false,
    coalescedStateThreadMapping: Bool = false
) -> String {
    precondition(
        !(emitsGroupOwnedPartialProjection && emitsPartitionOwnedPartialProjection),
        "SSM partial projection ownership modes are mutually exclusive"
    )
    precondition(
        !coalescedStateThreadMapping || parallelStateUpdate,
        "Coalesced state thread mapping requires parallel state update"
    )
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let ct = convStateStorageType(weightFormat: weightFormat)
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
    let readConvState = { (expr: String) in convStateRead(expr, weightFormat: weightFormat) }
    let writeConvState = { (expr: String) in convStateWrite(expr, weightFormat: weightFormat) }
    let activationStorageValue: (String) -> String = { expr in
        decodeActivationStorageValue(expr, weightFormat: weightFormat)
    }
    // Per-threadgroup (per key-group) local channel cache: Q (dk) + K (dk) + V (headsPerGroup*dv).
    // Sized at compile time using fragment's known dimensions.
    let safeGroupCount = max(groupCount, 1)
    let headsPerGroup = max(1, headCount / safeGroupCount)
    let localDim = 2 * keyHeadDimension + headsPerGroup * valueHeadDimension
    let stateReductionLanes = max(1, maxThreadgroupSize / max(1, headsPerGroup * valueHeadDimension))
    let rmsScaleCacheDeclaration = shareRMSScale
        ? "\n        threadgroup float rmsScaleCache[\(headsPerGroup)];"
        : ""
    let needsHeadParameterCache = cacheHeadParameters || parallelStateUpdate
    let headParameterCacheDeclaration = needsHeadParameterCache
        ? """

        threadgroup float decayCache[\(headsPerGroup)];
        threadgroup float betaCache[\(headsPerGroup)];
        """
        : ""
    let qkPartialCacheDeclaration = parallelQKReduction
        ? """

        threadgroup float qNormSqPartials[\(headsPerGroup * keyHeadDimension)];
        threadgroup float kNormSqPartials[\(headsPerGroup * keyHeadDimension)];
        threadgroup float kqSumPartials[\(headsPerGroup * keyHeadDimension)];
        """
        : ""
    let parallelStateCacheDeclaration = parallelStateUpdate
        ? """

        threadgroup float kvmemPartials[\(headsPerGroup * valueHeadDimension * stateReductionLanes)];
        threadgroup float sqSumPartials[\(headsPerGroup * valueHeadDimension * stateReductionLanes)];
        threadgroup float kInvDeltaCache[\(headsPerGroup * valueHeadDimension)];
        """
        : ""
    let headParameterCachePhase = cacheHeadParameters && !parallelStateUpdate
        ? """

                            const uint headIndex = headStart + localHead;
                            float alpha = \(activationStorageValue("float(projectedAlphaPos[headIndex])"));
                            float betaInput = \(activationStorageValue("float(projectedBetaPos[headIndex])"));
                            decayCache[localHead] = exp(-exp(aLog[headIndex]) * stable_softplus(alpha + \(readWeight("dtBias[headIndex]"))));
                            betaCache[localHead] = stable_sigmoid(betaInput);
        """
        : ""
    let rmsPhase = shareRMSScale
        ? """
                {
                    const uint threadsPerHead = min(tgSize / max(headsPerGroup, 1u), dv);
                    const uint activeThreads = headsPerGroup * threadsPerHead;
                    if (tid < activeThreads) {
                        const uint localHead = tid / threadsPerHead;
                        const uint localTid = tid % threadsPerHead;
                        if (localTid == 0) {
                            float totalNormSq = 0.0f;
                            for (uint t = 0; t < threadsPerHead; ++t) {
                                totalNormSq += normPartials[localHead * threadsPerHead + t];
                            }
                            rmsScaleCache[localHead] = rsqrt(totalNormSq / float(dv) + 1e-6f);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < activeThreads) {
                        const uint localHead = tid / threadsPerHead;
                        const uint headIndex = headStart + localHead;
                        const uint localTid = tid % threadsPerHead;
                        const uint dChunk = dv / threadsPerHead;
                        const uint dStart = localTid * dChunk;
                        const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;
                        float rmsScale = rmsScaleCache[localHead];
                        for (uint d = dStart; d < dEnd; ++d) {
                            float normed = dotCache[localHead * dv + d] * rmsScale * normWeight[d];
                            float z = \(activationStorageValue("float(projectedZPos[headIndex * dv + d])"));
                            float gated = normed * z * stable_sigmoid(z);
                            outputPos[headIndex * dv + d] = \(bt)(\(activationStorageValue("gated")));
                        }
                    }
                }
        """
        : """
                {
                    const uint threadsPerHead = min(tgSize / max(headsPerGroup, 1u), dv);
                    const uint activeThreads = headsPerGroup * threadsPerHead;
                    if (tid < activeThreads) {
                        const uint localHead = tid / threadsPerHead;
                        const uint headIndex = headStart + localHead;
                        const uint localTid = tid % threadsPerHead;
                        const uint dChunk = dv / threadsPerHead;
                        const uint dStart = localTid * dChunk;
                        const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;
                        float totalNormSq = 0.0f;
                        for (uint t = 0; t < threadsPerHead; ++t) {
                            totalNormSq += normPartials[localHead * threadsPerHead + t];
                        }
                        float rmsScale = rsqrt(totalNormSq / float(dv) + 1e-6f);
                        for (uint d = dStart; d < dEnd; ++d) {
                            float normed = dotCache[localHead * dv + d] * rmsScale * normWeight[d];
                            float z = \(activationStorageValue("float(projectedZPos[headIndex * dv + d])"));
                            float gated = normed * z * stable_sigmoid(z);
                            outputPos[headIndex * dv + d] = \(bt)(\(activationStorageValue("gated")));
                        }
                    }
                }
        """

    let recurrencePass1StateLine = prewriteDecayedState
        ? """
                            float s = state[j * dv + d] * decay;
                            state[j * dv + d] = s;
        """
        : """
                            float s = state[j * dv + d] * decay;
        """
    let recurrencePass2StateExpr = prewriteDecayedState
        ? "state[j * dv + d]"
        : "state[j * dv + d] * decay"
    let emitsPartialProjection = emitsGroupOwnedPartialProjection || emitsPartitionOwnedPartialProjection
    let partialPartitionCountSignature = emitsPartitionOwnedPartialProjection
        ? "constant uint& partialPartitionCount [[buffer(26)]],\n        "
        : ""
    let ownerPartitionCountExpression = emitsPartitionOwnedPartialProjection
        ? "max(1u, min(partialPartitionCount, safeGroupCount))"
        : "safeGroupCount"
    let partialProjectionSignature = emitsPartialProjection
        ? """
        device const \(wt)* partialWeight [[buffer(21)]],
        device \(bt)* partialOutput [[buffer(22)]],
        constant uint& partialOutputDimension [[buffer(23)]],
        constant uint& partialRowStride [[buffer(24)]],
        constant uint& partialProjectionEnabled [[buffer(25)]],
        \(partialPartitionCountSignature)
        """
        : ""
    let partialProjectionPhase = emitsPartialProjection
        ? """
            if (partialProjectionEnabled != 0) {
                threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
                const uint partitionInputDimension = groupsPerOwner * headsPerGroup * dv;
                const uint inputDimension = safeGroupCount * headsPerGroup * dv;
                const uint partitionInputBase = ownerIndex * partitionInputDimension;
                const uint tiisg = tid % SIMD_WIDTH;
                const uint sgitg = tid / SIMD_WIDTH;
                const uint rowsPerThreadgroup = max(1u, tgSize / SIMD_WIDTH);
                for (uint rowBase = 0; rowBase < partialOutputDimension; rowBase += rowsPerThreadgroup) {
                    const uint row = rowBase + sgitg;
                    float sum = 0.0f;
                    if (row < partialOutputDimension) {
                        device const \(wt)* weightRow = partialWeight + row * inputDimension + partitionInputBase;
                        for (uint j = tiisg; j < partitionInputDimension; j += SIMD_WIDTH) {
                            sum += \(readWeight("weightRow[j]")) * float(outputPos[partitionInputBase + j]);
                        }
                        sum = simd_sum(sum);
                        if (tiisg == 0) {
                            const uint outputIndex = ownerIndex * sequenceLength * partialRowStride
                                + pos * partialRowStride
                                + row;
                            partialOutput[outputIndex] = \(bt)(sum);
                        }
                    }
                }
            }
        """
        : ""
    let qkReductionPhase = parallelQKReduction
        ? """
                    if (tid < activeThreads) {
                        const uint localHead = tid / threadsPerHead;
                        const uint localTid = tid % threadsPerHead;
                        const uint qBase = 0u;
                        const uint kBase = dk;
                        for (uint j = localTid; j < dk; j += threadsPerHead) {
                            float q = convSiluCache[qBase + j];
                            float k = convSiluCache[kBase + j];
                            const uint partialIndex = localHead * dk + j;
                            qNormSqPartials[partialIndex] = q * q;
                            kNormSqPartials[partialIndex] = k * k;
                            kqSumPartials[partialIndex] = q * k;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < activeThreads) {
                        const uint localHead = tid / threadsPerHead;
                        const uint localTid = tid % threadsPerHead;
                        if (localTid == 0) {
                            float qNormSq = 0.0f;
                            float kNormSq = 0.0f;
                            float kqSum = 0.0f;
                            for (uint j = 0; j < dk; ++j) {
                                const uint partialIndex = localHead * dk + j;
                                qNormSq += qNormSqPartials[partialIndex];
                                kNormSq += kNormSqPartials[partialIndex];
                                kqSum += kqSumPartials[partialIndex];
                            }
                            qInvCache[localHead] = rsqrt(qNormSq + 1e-6f) * rsqrt(float(dk));
                            kInvCache[localHead] = rsqrt(kNormSq + 1e-6f);
                            kqSumCache[localHead] = kqSum;\(headParameterCachePhase)
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
        """
        : """
                    if (tid < activeThreads) {
                        const uint localHead = tid / threadsPerHead;
                        const uint localTid = tid % threadsPerHead;
                        if (localTid == 0) {
                            const uint qBase = 0u;
                            const uint kBase = dk;
                            float qNormSq = 0.0f;
                            float kNormSq = 0.0f;
                            float kqSum = 0.0f;
                            for (uint j = 0; j < dk; ++j) {
                                float q = convSiluCache[qBase + j];
                                float k = convSiluCache[kBase + j];
                                qNormSq += q * q;
                                kNormSq += k * k;
                                kqSum += q * k;
                            }
                            qInvCache[localHead] = rsqrt(qNormSq + 1e-6f) * rsqrt(float(dk));
                            kInvCache[localHead] = rsqrt(kNormSq + 1e-6f);
                            kqSumCache[localHead] = kqSum;\(headParameterCachePhase)
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
        """
    let headParameterLoad = cacheHeadParameters
        ? """
                        float decay = decayCache[localHead];
                        float beta = betaCache[localHead];
        """
        : """
                        float alpha = \(activationStorageValue("float(projectedAlphaPos[headIndex])"));
                        float betaInput = \(activationStorageValue("float(projectedBetaPos[headIndex])"));
                        float decay = exp(-exp(aLog[headIndex]) * stable_softplus(alpha + \(readWeight("dtBias[headIndex]"))));
                        float beta = stable_sigmoid(betaInput);
        """
    let parallelStateThreadMapping = coalescedStateThreadMapping
        ? """
                        const uint lane = headLocalTid / dv;
                        const uint d = headLocalTid - lane * dv;
        """
        : """
                        const uint d = headLocalTid / stateLaneCount;
                        const uint lane = headLocalTid - d * stateLaneCount;
        """

    let stateRecurrencePhase = parallelStateUpdate
        ? """
                    if (tid < headsPerGroup) {
                        const uint headIndex = headStart + tid;
                        float alpha = \(activationStorageValue("float(projectedAlphaPos[headIndex])"));
                        float betaInput = \(activationStorageValue("float(projectedBetaPos[headIndex])"));
                        decayCache[tid] = exp(-exp(aLog[headIndex]) * stable_softplus(alpha + \(readWeight("dtBias[headIndex]"))));
                        betaCache[tid] = stable_sigmoid(betaInput);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    const uint stateLaneCount = max(1u, tgSize / max(headsPerGroup * dv, 1u));
                    const uint stateThreadsPerHead = dv * stateLaneCount;
                    const uint stateActiveThreads = headsPerGroup * stateThreadsPerHead;
                    if (tid < stateActiveThreads) {
                        const uint localHead = tid / stateThreadsPerHead;
                        const uint headLocalTid = tid - localHead * stateThreadsPerHead;
                        \(parallelStateThreadMapping)
                        const uint headIndex = headStart + localHead;
                        device float* state = recurrentState + headIndex * dk * dv;
                        const uint qBase = 0u;
                        const uint kBase = dk;
                        const uint vBase = 2u * dk + localHead * dv;
                        float decay = decayCache[localHead];
                        float kvmemRaw = 0.0f;
                        float sqSum = 0.0f;
                        for (uint j = lane; j < dk; j += stateLaneCount) {
                            float s = state[j * dv + d] * decay;
                            kvmemRaw += s * convSiluCache[kBase + j];
                            sqSum += s * convSiluCache[qBase + j];
                        }
                        const uint partialIndex = (localHead * dv + d) * stateLaneCount + lane;
                        kvmemPartials[partialIndex] = kvmemRaw;
                        sqSumPartials[partialIndex] = sqSum;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    const uint stateReducerThreads = headsPerGroup * dv;
                    if (tid < stateReducerThreads) {
                        const uint localHead = tid / dv;
                        const uint d = tid - localHead * dv;
                        const uint headIndex = headStart + localHead;
                        const uint kBase = dk;
                        const uint vBase = 2u * dk + localHead * dv;
                        float kvmemRaw = 0.0f;
                        float sqSum = 0.0f;
                        for (uint lane = 0; lane < stateLaneCount; ++lane) {
                            const uint partialIndex = (localHead * dv + d) * stateLaneCount + lane;
                            kvmemRaw += kvmemPartials[partialIndex];
                            sqSum += sqSumPartials[partialIndex];
                        }
                        float beta = betaCache[localHead];
                        float qInv = qInvCache[localHead];
                        float kInv = kInvCache[localHead];
                        float kqSum = kqSumCache[localHead];
                        float delta = beta * (convSiluCache[vBase + d] - kvmemRaw * kInv);
                        float kInvDelta = kInv * delta;
                        float dot = (sqSum + kInvDelta * kqSum) * qInv;
                        kInvDeltaCache[localHead * dv + d] = kInvDelta;
                        float storedDot = \(activationStorageValue("dot"));
                        dotCache[localHead * dv + d] = storedDot;
                        outputPos[headIndex * dv + d] = \(bt)(storedDot);
                        normPartials[localHead * dv + d] = dot * dot;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (tid < stateActiveThreads) {
                        const uint localHead = tid / stateThreadsPerHead;
                        const uint headLocalTid = tid - localHead * stateThreadsPerHead;
                        \(parallelStateThreadMapping)
                        const uint headIndex = headStart + localHead;
                        device float* state = recurrentState + headIndex * dk * dv;
                        const uint kBase = dk;
                        float decay = decayCache[localHead];
                        float kInvDelta = kInvDeltaCache[localHead * dv + d];
                        for (uint j = lane; j < dk; j += stateLaneCount) {
                            state[j * dv + d] = state[j * dv + d] * decay + convSiluCache[kBase + j] * kInvDelta;
                        }
                    }
        """
        : """
                    if (tid < activeThreads) {
                        const uint localHead = tid / threadsPerHead;
                        const uint headIndex = headStart + localHead;
                        const uint localTid = tid % threadsPerHead;
                        const uint dChunk = dv / threadsPerHead;
                        const uint dStart = localTid * dChunk;
                        const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;

                        \(headParameterLoad)
                        device float* state = recurrentState + headIndex * dk * dv;

                        // Local cache indices for this recurrent group:
                        const uint qBase = 0u;
                        const uint kBase = dk;
                        const uint vBase = 2u * dk + localHead * dv;
                        float qInv = qInvCache[localHead];
                        float kInv = kInvCache[localHead];
                        float kqSum = kqSumCache[localHead];

                        // Algebraic identity used below:
                        //   state_after[j] = state_before[j]·decay + K[j]·kInvDelta
                        //   dot = Σ state_after[j]·Q[j]
                        //       = Σ (state_before[j]·decay)·Q[j] + kInvDelta · Σ K[j]·Q[j]
                        //       = sqSum + kInvDelta · kqSum
                        // This lets us compute `dot` from sqSum (one accumulator gathered
                        // during the decay pass) + a precomputed kqSum, eliminating the
                        // second Σ over updated state. Per j we now do 2 state reads + 1
                        // state write (vs 3 reads + 2 writes previously).
                        float localNormSq = 0.0f;
                        for (uint d = dStart; d < dEnd; ++d) {
                            // Pass 1: read state, accumulate kvmemRaw and sqSum.
                            // No state write here — deferred to Pass 2 for a single full write.
                            float kvmemRaw = 0.0f;
                            float sqSum = 0.0f;
                            for (uint j = 0; j < dk; ++j) {
                                \(recurrencePass1StateLine)
                                kvmemRaw += s * convSiluCache[kBase + j];
                                sqSum += s * convSiluCache[qBase + j];
                            }

                            float delta = beta * (convSiluCache[vBase + d] - kvmemRaw * kInv);
                            float kInvDelta = kInv * delta;
                            float dot = (sqSum + kInvDelta * kqSum) * qInv;

                            // Pass 2: write final state = decay·old + K·kInvDelta (full write, no RMW).
                            for (uint j = 0; j < dk; ++j) {
                                state[j * dv + d] = \(recurrencePass2StateExpr) + convSiluCache[kBase + j] * kInvDelta;
                            }

                            float storedDot = \(activationStorageValue("dot"));
                            dotCache[localHead * dv + d] = storedDot;
                            outputPos[headIndex * dv + d] = \(bt)(storedDot);
                            localNormSq += dot * dot;
                        }

                        normPartials[tid] = localNormSq;
                    }
        """

    return """
    kernel void \(name)(
        device const \(bt)* projectedQKV [[buffer(0)]],
        device const \(bt)* projectedZ [[buffer(1)]],
        device const \(bt)* projectedBeta [[buffer(2)]],
        device const \(bt)* projectedAlpha [[buffer(3)]],
        device const \(wt)* convWeight [[buffer(4)]],
        device const float* normWeight [[buffer(5)]],
        device const \(wt)* dtBias [[buffer(6)]],
        device const float* aLog [[buffer(7)]],
        device float* recurrentState [[buffer(8)]],
        device \(ct)* convState [[buffer(9)]],
        device \(bt)* output [[buffer(10)]],
        constant uint& numHeads [[buffer(11)]],
        constant uint& groupCount [[buffer(12)]],
        constant uint& keyDimension [[buffer(13)]],
        constant uint& valueDimension [[buffer(14)]],
        constant uint& convKernelSize [[buffer(15)]],
        constant uint& sequenceLength [[buffer(16)]],
        constant uint& activationRowStride [[buffer(17)]],
        device float* debugConvSilu [[buffer(18)]],
        constant uint& debugConvRowStride [[buffer(19)]],
        constant uint& debugConvEnabled [[buffer(20)]],
        \(partialProjectionSignature)
        uint tid [[thread_index_in_threadgroup]],
        uint tgSize [[threads_per_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]]
    ) {
        const uint dk = keyDimension;
        const uint dv = valueDimension;
        const uint keyGroupDim = groupCount * dk;
        const uint convDim = 2 * keyGroupDim + numHeads * dv;
        const uint safeGroupCount = max(groupCount, 1u);
        const uint headsPerGroup = max(1u, numHeads / safeGroupCount);

        // Each threadgroup owns one execution partition. In the default and
        // group-owned partial modes, an execution partition is one recurrent
        // group. In partition-owned partial mode, it may cover multiple
        // recurrent groups and updates them serially before emitting partial
        // rows for that partition.
        const uint ownerPartitionCount = \(ownerPartitionCountExpression);
        if (ownerPartitionCount == 0u || safeGroupCount % ownerPartitionCount != 0u) {
            return;
        }
        const uint ownerIndex = tgid;
        if (ownerIndex >= ownerPartitionCount) {
            return;
        }
        const uint groupsPerOwner = max(1u, safeGroupCount / ownerPartitionCount);
        const uint ownerGroupStart = ownerIndex * groupsPerOwner;

        // Local channel layout inside convSiluCache for one recurrent group:
        //   [0, dk)            : Q
        //   [dk, 2*dk)         : K
        //   [2*dk, localDim)   : V (headsPerGroup * dv values)
        const uint localDim = 2u * dk + headsPerGroup * dv;

        threadgroup float convSiluCache[\(localDim)];
        threadgroup float dotCache[\(headsPerGroup * valueHeadDimension)];
        threadgroup float normPartials[\(maxThreadgroupSize)];
        threadgroup float qInvCache[\(headsPerGroup)];
        threadgroup float kInvCache[\(headsPerGroup)];
        threadgroup float kqSumCache[\(headsPerGroup)];\(rmsScaleCacheDeclaration)\(headParameterCacheDeclaration)\(qkPartialCacheDeclaration)\(parallelStateCacheDeclaration)

        for (uint pos = 0; pos < sequenceLength; ++pos) {
            device const \(bt)* projectedQKVPos = projectedQKV + pos * activationRowStride;
            device const \(bt)* projectedZPos = projectedZ + pos * activationRowStride;
            device const \(bt)* projectedBetaPos = projectedBeta + pos * activationRowStride;
            device const \(bt)* projectedAlphaPos = projectedAlpha + pos * activationRowStride;
            device \(bt)* outputPos = output + pos * activationRowStride;

            for (uint ownerGroupOffset = 0; ownerGroupOffset < groupsPerOwner; ++ownerGroupOffset) {
                const uint groupIndex = ownerGroupStart + ownerGroupOffset;
                const uint headStart = groupIndex * headsPerGroup;

                // Global conv-channel offsets owned by this recurrent group.
                const uint qBaseGlobal = groupIndex * dk;
                const uint kBaseGlobal = keyGroupDim + groupIndex * dk;
                const uint vBaseGlobal = 2u * keyGroupDim + headStart * dv;

                // Phase 1: fused conv-shift + SiLU for this recurrent group's channels.
                // Each execution owner touches only the Q/K/V channels assigned to its groups.
                for (uint localCh = tid; localCh < localDim; localCh += tgSize) {
                    uint globalCh;
                    if (localCh < dk) {
                        globalCh = qBaseGlobal + localCh;
                    } else if (localCh < 2u * dk) {
                        globalCh = kBaseGlobal + (localCh - dk);
                    } else {
                        globalCh = vBaseGlobal + (localCh - 2u * dk);
                    }

                    float sum = 0.0f;
                    for (uint k = 0; k + 1 < convKernelSize; ++k) {
                        float val = \(readConvState("convState[(k + 1) * convDim + globalCh]"));
                        convState[k * convDim + globalCh] = \(writeConvState("val"));
                        sum += val * \(readWeight("convWeight[globalCh * convKernelSize + k]"));
                    }
                    float newVal = \(activationStorageValue("float(projectedQKVPos[globalCh])"));
                    convState[(convKernelSize - 1) * convDim + globalCh] = \(writeConvState("newVal"));
                    sum += newVal * \(readWeight("convWeight[globalCh * convKernelSize + convKernelSize - 1]"));
                    float convSilu = sum * stable_sigmoid(sum);
                    convSiluCache[localCh] = convSilu;
                    if (debugConvEnabled != 0) {
                        debugConvSilu[pos * debugConvRowStride + globalCh] = convSilu;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Phase 2: per-head recurrence restricted to owned heads.
                // recurrentState partition is disjoint per head, so disjoint per execution owner.
                {
                    const uint threadsPerHead = min(tgSize / max(headsPerGroup, 1u), dv);
                    const uint activeThreads = headsPerGroup * threadsPerHead;
                    \(qkReductionPhase)

                    \(stateRecurrencePhase)
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Phase 3: RMS norm + gated activation for owned heads.
                \(rmsPhase)
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            \(partialProjectionPhase)
            // Skip barrier after final position — no subsequent iteration reads this state,
            // and the command encoder's implicit barrier handles cross-dispatch visibility.
            if (pos + 1 < sequenceLength) {
                threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            }
        }
    }
    """
}
}
