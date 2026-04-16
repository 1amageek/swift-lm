import LMIR

extension StandardizeAttributes: MetalCompilable {

    /// Fragment expansion for Standardize.
    package func fragment(context: KernelContext) -> StandardizeFragment {
        StandardizeFragment(dimension: dimension)
    }
}
