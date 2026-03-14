import LMCompiler
import SwiftLM

/// Weight binding error for the inference compiler path.
enum WeightBindingError: Error, CustomStringConvertible {

    /// A tensor's MLX path could not be mapped to any ParameterSlot.
    case unmappedTensor(mlxPath: String)

    /// A required parameter slot has no matching tensor.
    case missingRequiredSlot(slot: ParameterSlot, mlxPath: String)

    var description: String {
        switch self {
        case .unmappedTensor(let path):
            return "No ParameterSlot found for MLX weight path: \(path)"
        case .missingRequiredSlot(let slot, let mlxPath):
            return "Missing tensor for required slot \(slot.role) at \(mlxPath)"
        }
    }
}

/// Binds MLX-path-keyed weight tensors to semantic ParameterSlots.
///
/// Accepts `RawWeights` keyed by MLX weight paths (e.g., "model.layers.0.self_attn.q_proj.weight")
/// and maps them to `ParameterSlot`s using a pre-built slot manifest.
///
/// Usage:
/// ```swift
/// let manifest = mapper.manifest(for: graph)
/// let bound = try MLXWeightPathBinder(manifest: manifest).bind(rawWeights, to: graph)
/// let store = try InferenceWeightStore(boundWeights: bound)
/// ```
package struct MLXWeightPathBinder: WeightBinder {

    let manifest: [SlotManifestEntry]

    package init(manifest: [SlotManifestEntry]) {
        self.manifest = manifest
    }

    package func bind(_ raw: RawWeights, to graph: ModelGraph) throws -> BoundWeights {
        // Build MLX path → slot lookup, allowing multiple slots per path (weight sharing)
        var slotsByPath: [String: [ParameterSlot]] = [:]
        for entry in manifest {
            slotsByPath[entry.mlxWeightPath, default: []].append(entry.slot)
        }

        // Map each raw tensor to its slot(s)
        var result: [ParameterSlot: TensorData] = [:]
        for (mlxPath, tensor) in raw.tensors {
            guard let slots = slotsByPath[mlxPath] else {
                // Skip unmapped tensors (e.g., LoRA weights, quantization metadata)
                // The caller can validate completeness separately
                continue
            }
            for slot in slots {
                result[slot] = tensor
            }
        }

        return BoundWeights(tensors: result)
    }
}
