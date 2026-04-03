extension MetalSourceGenerator {
    public static func generateArgmaxArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const \(bt)* logits [[id(0)]];
            device int* result [[id(1)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& vocabularySize             [[buffer(2)]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            threadgroup float sharedValues[32];
            threadgroup int sharedIndices[32];

            float localMax = -HUGE_VALF;
            int localIndex = 0;
            for (uint i = tid; i < vocabularySize; i += threadgroupSize) {
                float value = float(args.logits[i]);
                if (value > localMax) { localMax = value; localIndex = int(i); }
            }

            for (uint offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
                float otherValue = simd_shuffle_down(localMax, offset);
                int otherIndex = simd_shuffle_down(localIndex, offset);
                if (otherValue > localMax) { localMax = otherValue; localIndex = otherIndex; }
            }

            uint simdIndex = tid / SIMD_WIDTH;
            if (tid % SIMD_WIDTH == 0) {
                sharedValues[simdIndex] = localMax;
                sharedIndices[simdIndex] = localIndex;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float bestValue = -HUGE_VALF;
                int bestIndex = 0;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint i = 0; i < sgCount; i++) {
                    if (sharedValues[i] > bestValue) {
                        bestValue = sharedValues[i];
                        bestIndex = sharedIndices[i];
                    }
                }
                args.result[0] = bestIndex;
            }
        }
        """
    }

    /// Generate MSL source for argmax.
    public static func generateArgmax(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType

        return """
        kernel void \(name)(
            device const \(bt)* logits       [[buffer(0)]],
            device int* result               [[buffer(1)]],
            constant uint& vocabularySize    [[buffer(2)]],
            uint tid                         [[thread_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            threadgroup float sharedValues[32];
            threadgroup int sharedIndices[32];

            float localMax = -HUGE_VALF;
            int localIndex = 0;
            for (uint i = tid; i < vocabularySize; i += threadgroupSize) {
                float value = float(logits[i]);
                if (value > localMax) { localMax = value; localIndex = int(i); }
            }

            for (uint offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
                float otherValue = simd_shuffle_down(localMax, offset);
                int otherIndex = simd_shuffle_down(localIndex, offset);
                if (otherValue > localMax) { localMax = otherValue; localIndex = otherIndex; }
            }

            uint simdIndex = tid / SIMD_WIDTH;
            if (tid % SIMD_WIDTH == 0) { sharedValues[simdIndex] = localMax; sharedIndices[simdIndex] = localIndex; }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float bestValue = -HUGE_VALF;
                int bestIndex = 0;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint i = 0; i < sgCount; i++) {
                    if (sharedValues[i] > bestValue) { bestValue = sharedValues[i]; bestIndex = sharedIndices[i]; }
                }
                result[0] = bestIndex;
            }
        }
        """
    }
}
