extension MetalSourceGenerator {
// MARK: - Conv1d

/// Generate conv_state_update kernel (decode: single token with persistent state).
public static func generateConvStateUpdate(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat
) -> String {
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

    return """
    kernel void \(name)(
        device half* convState           [[buffer(0)]],
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
        convState[(kernelSize - 1) * dimension + gid] = half(Bx);

        float convOut = 0.0f;
        for (uint k = 0; k + 1 < kernelSize; k++) {
            convOut += float(convState[k * dimension + gid]) * \(readWeight("weight[gid * kernelSize + k]"));
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
    let inputStructName = "\(name)_args"
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

    return """
    struct \(inputStructName) {
        device half* convState [[id(0)]];
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
        args.convState[(kernelSize - 1) * dimension + gid] = half(Bx);

        float convOut = 0.0f;
        for (uint k = 0; k + 1 < kernelSize; k++) {
            convOut += float(args.convState[k * dimension + gid]) * \(readWeight("args.weight[gid * kernelSize + k]"));
        }
        convOut += Bx * \(readWeight("args.weight[gid * kernelSize + (kernelSize - 1)]"));

        args.output[gid] = \(bt)(C * convOut);
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

    return """
    kernel void \(name)(
        device const \(bt)* input      [[buffer(0)]],
        device const \(wt)* weight     [[buffer(1)]],
        device \(bt)* output           [[buffer(2)]],
        constant uint& convDim         [[buffer(3)]],
        constant uint& inProjDim       [[buffer(4)]],
        constant uint& kernelSize      [[buffer(5)]],
        constant uint& sequenceLength  [[buffer(6)]],
        uint2 gid                      [[thread_position_in_grid]]
    ) {
        uint ch = gid.x;
        uint pos = gid.y;
        if (ch >= convDim || pos >= sequenceLength) return;

        float convOut = 0.0f;
        for (uint k = 0; k < kernelSize; k++) {
            int srcPos = int(pos) - int(kernelSize - 1) + int(k);
            if (srcPos >= 0) {
                float B = float(input[uint(srcPos) * inProjDim + ch]);
                float x = float(input[uint(srcPos) * inProjDim + 2 * convDim + ch]);
                convOut += B * x * \(readWeight("weight[ch * kernelSize + k]"));
            }
        }

        float C = float(input[pos * inProjDim + convDim + ch]);
        output[pos * convDim + ch] = \(bt)(C * convOut);
    }
    """
}

/// Generate extract_conv_state kernel (saves last kernelSize positions' B*x to conv_state).
public static func generateExtractConvState(
    name: String,
    bufferPrecision: BufferPrecision
) -> String {
    let bt = bufferPrecision.metalType

    return """
    kernel void \(name)(
        device const \(bt)* inProjOutput   [[buffer(0)]],
        device half* convState             [[buffer(1)]],
        constant uint& convDim             [[buffer(2)]],
        constant uint& inProjDim           [[buffer(3)]],
        constant uint& kernelSize          [[buffer(4)]],
        constant uint& sequenceLength      [[buffer(5)]],
        uint2 gid                          [[thread_position_in_grid]]
    ) {
        uint ch = gid.x;
        uint k = gid.y;
        if (ch >= convDim || k >= kernelSize) return;
        int srcPos = int(sequenceLength) - int(kernelSize) + int(k);
        if (srcPos >= 0 && uint(srcPos) < sequenceLength) {
            float B = float(inProjOutput[uint(srcPos) * inProjDim + ch]);
            float x = float(inProjOutput[uint(srcPos) * inProjDim + 2 * convDim + ch]);
            convState[k * convDim + ch] = half(B * x);
        } else {
            convState[k * convDim + ch] = 0;
        }
    }
    """
}

// MARK: - State Space

static let ssmRecurrenceSource = """
    inline float stable_softplus(float x) { return max(x, 0.0f) + log(1.0f + exp(-abs(x))); }
    inline float compute_l2_inv_norm(threadgroup float* vec, uint dim, uint tid, uint tgSize, threadgroup float* scratch) {
        float sumSq = 0.0f;
        for (uint d = tid; d < dim; d += tgSize) sumSq += vec[d] * vec[d];
        sumSq = simd_sum(sumSq);
        if (tid % SIMD_WIDTH == 0) scratch[tid / SIMD_WIDTH] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float t = 0; for (uint i = 0; i < (tgSize + SIMD_WIDTH - 1) / SIMD_WIDTH; i++) t += scratch[i]; scratch[0] = rsqrt(t + 1e-6f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return scratch[0];
    }
    kernel void ssm_recurrence(
        device const half* projectedQKV [[buffer(0)]], device const half* projectedZ [[buffer(1)]],
        device const half* projectedBeta [[buffer(2)]], device const half* projectedAlpha [[buffer(3)]],
        device const half* convWeight [[buffer(4)]], device const half* normWeight [[buffer(5)]],
        device const half* dtBias [[buffer(6)]], device const half* aLog [[buffer(7)]],
        device float* recurrentState [[buffer(8)]], device half* convState [[buffer(9)]],
        device half* output [[buffer(10)]], constant uint& numHeads [[buffer(11)]],
        constant uint& groupCount [[buffer(12)]], constant uint& keyDimension [[buffer(13)]],
        constant uint& valueDimension [[buffer(14)]], constant uint& convKernelSize [[buffer(15)]],
        uint headIndex [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threadgroupSize [[threads_per_threadgroup]]
    ) {
        const uint dk = keyDimension, dv = valueDimension;
        const uint keyGroupDim = groupCount * dk, convDim = 2 * keyGroupDim + numHeads * dv;
        const uint keyGroupIndex = headIndex / (numHeads / groupCount);
        device float* sharedConvOut = (device float*)output;
        for (uint ch = tid; ch < convDim; ch += threadgroupSize) {
            for (uint k = 0; k < convKernelSize - 1; k++) convState[ch * convKernelSize + k] = convState[ch * convKernelSize + k + 1];
            convState[ch * convKernelSize + convKernelSize - 1] = projectedQKV[ch];
            float sum = 0.0f;
            for (uint k = 0; k < convKernelSize; k++) sum += float(convState[ch * convKernelSize + k]) * float(convWeight[ch * convKernelSize + k]);
            sharedConvOut[ch] = sum / (1.0f + exp(-sum));
        }
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup float sharedQ[256], sharedK[256], sharedV[256];
        for (uint d = tid; d < dk; d += threadgroupSize) { sharedK[d] = sharedConvOut[keyGroupIndex * dk + d]; sharedQ[d] = sharedConvOut[keyGroupDim + keyGroupIndex * dk + d]; }
        for (uint d = tid; d < dv; d += threadgroupSize) sharedV[d] = sharedConvOut[2 * keyGroupDim + headIndex * dv + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float normScratch[32];
        float qInv = compute_l2_inv_norm(sharedQ, dk, tid, threadgroupSize, normScratch);
        for (uint d = tid; d < dk; d += threadgroupSize) sharedQ[d] *= qInv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kInv = compute_l2_inv_norm(sharedK, dk, tid, threadgroupSize, normScratch);
        for (uint d = tid; d < dk; d += threadgroupSize) sharedK[d] *= kInv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qScale = rsqrt(float(dk));
        for (uint d = tid; d < dk; d += threadgroupSize) sharedQ[d] *= qScale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float decay = exp(-exp(float(aLog[headIndex])) * stable_softplus(float(projectedAlpha[headIndex]) + float(dtBias[headIndex])));
        float beta = 1.0f / (1.0f + exp(-float(projectedBeta[headIndex])));
        device float* S = recurrentState + headIndex * dk * dv;
        for (uint idx = tid; idx < dk * dv; idx += threadgroupSize) S[idx] *= decay;
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup float sharedKVMem[256];
        for (uint d = tid; d < dv; d += threadgroupSize) { float dot = 0; for (uint j = 0; j < dk; j++) dot += S[j * dv + d] * sharedK[j]; sharedKVMem[d] = dot; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float sharedDelta[256];
        for (uint d = tid; d < dv; d += threadgroupSize) sharedDelta[d] = beta * (sharedV[d] - sharedKVMem[d]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < dk * dv; idx += threadgroupSize) S[idx] += sharedK[idx / dv] * sharedDelta[idx % dv];
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup float sharedOutput[256];
        for (uint d = tid; d < dv; d += threadgroupSize) { float dot = 0; for (uint j = 0; j < dk; j++) dot += S[j * dv + d] * sharedQ[j]; sharedOutput[d] = dot; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumSq = 0; for (uint d = tid; d < dv; d += threadgroupSize) sumSq += sharedOutput[d] * sharedOutput[d];
        sumSq = simd_sum(sumSq); threadgroup float normReduce[32];
        if (tid % SIMD_WIDTH == 0) normReduce[tid / SIMD_WIDTH] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float t = 0; for (uint i = 0; i < (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH; i++) t += normReduce[i]; normReduce[0] = rsqrt(t / float(dv) + 1e-6f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rmsScale = normReduce[0];
        for (uint d = tid; d < dv; d += threadgroupSize) {
            float normed = sharedOutput[d] * rmsScale * float(normWeight[d]);
            float z = float(projectedZ[headIndex * dv + d]);
            output[headIndex * dv + d] = half(normed * z / (1.0f + exp(-z)));
        }
    }
    """
}
