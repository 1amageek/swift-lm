import LMCompiler
import SwiftLM

/// Maps HF safetensors tensor names to canonical ParameterSlots.
///
/// Each model family has its own HF tensor naming convention.
/// The mapper produces SlotManifestEntry arrays that the binder uses
/// to match raw tensors to IR parameter slots.
public protocol WeightNameMapper: Sendable {
    /// Produce a manifest mapping HF tensor names to canonical IR slots.
    func manifest(for graph: ModelGraph) -> [SlotManifestEntry]
}
