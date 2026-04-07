import LMIR

extension AttentionAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public func fragment(context: KernelContext) -> some MetalKernelFragment {
        LinearFragment(field: "q_proj", inputDimension: hiddenSize, outputDimension: headCount * headDimension)
        LinearFragment(field: "k_proj", inputDimension: hiddenSize, outputDimension: kvHeadCount * headDimension)
        LinearFragment(field: "v_proj", inputDimension: hiddenSize, outputDimension: kvHeadCount * headDimension)
        if let qkNorm = qkNorm, qkNorm != .none {
            QKNormFragment(headCount: headCount, headDimension: headDimension, epsilon: 1e-6, weightRole: "q_layernorm")
            QKNormFragment(headCount: kvHeadCount, headDimension: headDimension, epsilon: 1e-6, weightRole: "k_layernorm")
        }
        // RoPE is inlined into FlashAttentionFragment for decode (saves 1 dispatch + 1 barrier per layer).
        // For prefill, FlashAttentionFragment emits a separate rope_seq step internally.
        FlashAttentionFragment(
            headCount: headCount, kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            ropeDimension: rope?.dimension ?? 0,
            ropeBase: rope?.base ?? 0,
            mropeAxes: rope?.mropeAxes)
        LinearFragment(field: "o_proj", inputDimension: headCount * headDimension, outputDimension: hiddenSize)
    }
    public var isFusable: Bool { false }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment(context: context)) }
}
