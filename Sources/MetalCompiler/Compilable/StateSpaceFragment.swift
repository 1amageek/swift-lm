import LMIR

extension StateSpaceAttributes: MetalCompilable {
    @MetalKernelFragmentBuilder
    package func fragment(context: KernelContext) -> some MetalKernelFragment {
        let projectedStateDimension = 2 * groupCount * keyHeadDim + numHeads * valueHeadDim
        let outputDimension = numHeads * valueHeadDim

        // Batched input projections (component-internal optimization)
        BatchedProjection(projections: [
            .init(field: "in_proj_qkv", inputDimension: hiddenSize, outputDimension: projectedStateDimension),
            .init(field: "in_proj_z", inputDimension: hiddenSize, outputDimension: outputDimension),
            .init(field: "in_proj_b", inputDimension: hiddenSize, outputDimension: numHeads),
            .init(field: "in_proj_a", inputDimension: hiddenSize, outputDimension: numHeads),
        ])
        SSMRecurrenceFragment(
            headCount: numHeads,
            groupCount: groupCount,
            keyHeadDimension: keyHeadDim,
            valueHeadDimension: valueHeadDim,
            convKernelSize: convKernelSize
        )
        LinearFragment(
            field: "out_proj",
            inputDimension: outputDimension,
            outputDimension: hiddenSize
        )
    }
}
