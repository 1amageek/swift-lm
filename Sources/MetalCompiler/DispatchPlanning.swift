extension DispatchEntry {
    var decodeWeightBindingBase: (roles: [String], inputDimension: Int, outputDimension: Int)? {
        switch kind {
        case .projection(let projection, _):
            return ([projection.field], projection.inputDimension, projection.outputDimension)
        case .fusedSwiGLUProjection(let fused):
            return ([fused.gateField, fused.upField], fused.inputDimension, fused.outputDimension)
        case .batchedProjection(let batched):
            return (
                batched.projections.map(\.field),
                batched.inputDimension,
                batched.totalOutputDimension
            )
        default:
            return nil
        }
    }
}

extension DispatchKind {
    var dispatchDimension: MetalDispatchDimension {
        switch self {
        case .projection(let projection, _):
            return .gemv(outputDimension: projection.outputDimension, inputDimension: projection.inputDimension)
        case .fusedCopyNorm(let fused):
            return .reduction(dimension: fused.dimension)
        case .fusedResidualAddCopyNorm(let fused):
            return .reduction(dimension: fused.dimension)
        case .fragment(let fragment):
            return fragment.dispatchDimension
        case .structuralCopy(let dimension):
            return .elementwise(count: dimension)
        case .structuralAdd(let dimension):
            return .elementwise(count: dimension)
        case .fusedResidualAddNorm(let fused):
            return .reduction(dimension: fused.dimension)
        case .fusedSwiGLUProjection(let fused):
            return .gemv(outputDimension: fused.outputDimension, inputDimension: fused.inputDimension)
        case .batchedProjection(let batched):
            return .gemv(outputDimension: batched.totalOutputDimension, inputDimension: batched.inputDimension)
        case .batchedFragment(let batch):
            return batch.dispatchDimension
        }
    }
}
