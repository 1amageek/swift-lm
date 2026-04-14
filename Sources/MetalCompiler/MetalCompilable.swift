import LMIR

/// Backend-specific compilation protocol for Metal.
///
/// Each `OperationAttributes` type declares its Metal compilation capability
/// by conforming to this protocol in the MetalCompiler module. The compiler
/// queries this capability via `as? any MetalCompilable` — a capability check,
/// not type detection.
///
/// Other backends (MLX, TPU) would define analogous protocols:
/// ```swift
/// protocol MLXCompilable { ... }
/// protocol TPUCompilable { ... }
/// ```
package protocol MetalCompilable {
    associatedtype Fragment: MetalKernelFragment
    func fragment(context: KernelContext) -> Fragment
}
