import LMIR

extension LinearAttributes: MetalCompilable {
    package func fragment(context: KernelContext) -> LinearFragment {
        LinearFragment(field: "weight", inputDimension: inputSize, outputDimension: outputSize)
    }
}
