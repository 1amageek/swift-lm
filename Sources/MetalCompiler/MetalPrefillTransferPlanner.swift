import Metal

struct MetalPrefillTransferPlanner: Sendable {
    func makeTransferPlan(
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        sequenceLength: Int
    ) -> PostPrefillTransferPlan {
        let decodeElementSize = decodePlan.buffers.bufferPrecision.byteSize
        let decodeHiddenSize = decodePlan.buffers.hidden.length / decodeElementSize
        let prefillHiddenStride = decodeHiddenSize * MemoryLayout<Float32>.size
        let hiddenSourceOffset = (sequenceLength - 1) * prefillHiddenStride

        // GPU-side F32→decode precision conversion element count.
        // When decode precision is already F32, a blit copy suffices.
        let hiddenConversionElementCount: Int
        let hiddenBlitCopySize: Int
        if decodePlan.buffers.bufferPrecision == .float32 {
            hiddenConversionElementCount = 0
            hiddenBlitCopySize = decodeHiddenSize * decodeElementSize
        } else {
            hiddenConversionElementCount = decodeHiddenSize
            hiddenBlitCopySize = 0
        }

        let kvCopySize: Int
        if let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache,
           prefillKV.keys !== decodeKV.keys {
            kvCopySize = min(decodeKV.keys.length, prefillKV.keys.length)
        } else {
            kvCopySize = 0
        }

        let valueCopySize: Int
        if let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache,
           prefillKV.values !== decodeKV.values {
            valueCopySize = min(decodeKV.values.length, prefillKV.values.length)
        } else {
            valueCopySize = 0
        }

        let convCopySize: Int
        if let prefillConvState = prefillPlan.buffers.convState,
           let decodeConvState = decodePlan.buffers.convState,
           prefillConvState !== decodeConvState {
            convCopySize = min(decodeConvState.length, prefillConvState.length)
        } else {
            convCopySize = 0
        }

        let recurrentCopySize: Int
        if let prefillRecurrentState = prefillPlan.buffers.recurrentState,
           let decodeRecurrentState = decodePlan.buffers.recurrentState,
           prefillRecurrentState !== decodeRecurrentState {
            recurrentCopySize = min(decodeRecurrentState.length, prefillRecurrentState.length)
        } else {
            recurrentCopySize = 0
        }

        return PostPrefillTransferPlan(
            hiddenSourceOffset: hiddenSourceOffset,
            hiddenConversionElementCount: hiddenConversionElementCount,
            hiddenBlitCopySize: hiddenBlitCopySize,
            kvCopySize: kvCopySize,
            valueCopySize: valueCopySize,
            convCopySize: convCopySize,
            recurrentCopySize: recurrentCopySize
        )
    }
}

struct PostPrefillTransferPlan: Sendable {
    let hiddenSourceOffset: Int
    /// Element count for GPU-side F32→F16/BF16 hidden conversion.
    /// Zero when decode precision is F32 (blit copy used instead).
    let hiddenConversionElementCount: Int
    /// Byte count for F32→F32 hidden blit copy (only when decode is F32).
    let hiddenBlitCopySize: Int
    let kvCopySize: Int
    let valueCopySize: Int
    let convCopySize: Int
    let recurrentCopySize: Int
}
