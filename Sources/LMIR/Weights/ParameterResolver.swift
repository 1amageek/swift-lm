/// Resolves parameter bindings for a semantic model graph.
///
/// Parameter resolution is part of the backend-independent model contract:
/// every backend consumes the same graph and weight names. Model declarations
/// provide the naming convention while this type owns the graph walk.
public struct ParameterResolver: Sendable {

    public init() {}

    /// Resolve all parameter bindings in a model graph.
    public func resolve(
        graph: ModelGraph,
        convention: any WeightNamingConvention
    ) -> ModelGraph {
        let resolvedRegion = resolveRegion(
            graph.rootRegion,
            convention: convention,
            scope: .root,
            residualIndex: 0
        )
        return ModelGraph(rootRegion: resolvedRegion)
    }

    private func resolveRegion(
        _ region: Region,
        convention: any WeightNamingConvention,
        scope: WeightNamingScope,
        residualIndex: Int
    ) -> Region {
        var operations: [Operation] = []
        var currentResidualIndex = residualIndex
        var residualCount = 0
        var normCounter = 0

        for operation in region.operations {
            var effectiveScope = scope
            if case .root = scope, case .residual = operation.kind {
                effectiveScope = .layer(index: residualCount / 2)
                residualCount += 1
            }

            let resolved = resolveOperation(
                operation,
                convention: convention,
                scope: effectiveScope,
                residualIndex: &currentResidualIndex,
                normIndex: normCounter
            )
            operations.append(resolved)

            if case .primitive(let attributes) = operation.kind,
               attributes is RMSNormAttributes || attributes is LayerNormAttributes {
                normCounter += 1
            }
        }

        return Region(
            parameters: region.parameters,
            operations: operations,
            results: region.results
        )
    }

    private func resolveOperation(
        _ operation: Operation,
        convention: any WeightNamingConvention,
        scope: WeightNamingScope,
        residualIndex: inout Int,
        normIndex: Int
    ) -> Operation {
        switch operation.kind {
        case .primitive(let attributes):
            return Operation(
                key: operation.key,
                kind: operation.kind,
                operands: operation.operands,
                results: operation.results,
                parameterBindings: convention.bindings(
                    for: attributes,
                    scope: scope,
                    residualIndex: residualIndex,
                    normIndex: normIndex
                )
            )

        case .residual(let strategy, let body):
            let savedIndex = residualIndex
            let resolvedBody = resolveRegion(
                body,
                convention: convention,
                scope: scope,
                residualIndex: savedIndex % 2
            )
            residualIndex = savedIndex + 1
            return Operation(
                key: operation.key,
                kind: .residual(strategy: strategy, body: resolvedBody),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: operation.parameterBindings
            )

        case .repeating(let count, let body):
            let templateBody = resolveRegion(
                body,
                convention: convention,
                scope: .layer(index: 0),
                residualIndex: 0
            )
            return Operation(
                key: operation.key,
                kind: .repeating(count: count, body: templateBody),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: operation.parameterBindings
            )

        case .conditional(let condition, let thenBody, let elseBody):
            return Operation(
                key: operation.key,
                kind: .conditional(
                    condition: condition,
                    then: resolveRegion(
                        thenBody,
                        convention: convention,
                        scope: scope,
                        residualIndex: 0
                    ),
                    else: resolveRegion(
                        elseBody,
                        convention: convention,
                        scope: scope,
                        residualIndex: 0
                    )
                ),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: operation.parameterBindings
            )

        case .parallel(let merge, let branches):
            return Operation(
                key: operation.key,
                kind: .parallel(
                    merge: merge,
                    branches: branches.map {
                        resolveRegion(
                            $0,
                            convention: convention,
                            scope: scope,
                            residualIndex: 0
                        )
                    }
                ),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: operation.parameterBindings
            )
        }
    }
}
