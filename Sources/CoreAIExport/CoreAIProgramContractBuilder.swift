import LMIR

/// Derives the executable Core AI function contract from canonical Swift LMIR.
struct CoreAIProgramContractBuilder {
    struct Result {
        let contract: CoreAIProgramContract
        let stateBindings: [OperationKey: [CoreAIExportDocument.StateBinding]]
    }

    func make(
        graph: ModelGraph,
        configuration: CoreAIExportConfiguration
    ) throws -> Result {
        switch configuration.execution {
        case .stateless:
            return Result(
                contract: makeStatelessContract(configuration: configuration),
                stateBindings: [:]
            )
        case .stateful:
            return try makeStatefulContract(graph: graph, configuration: configuration)
        }
    }

    private func makeStatelessContract(
        configuration: CoreAIExportConfiguration
    ) -> CoreAIProgramContract {
        let sequence = CoreAIProgramContract.Dimension.dynamic(
            "sequence_length",
            minimum: 1,
            maximum: configuration.maxContextLength
        )
        return CoreAIProgramContract(
            execution: .stateless,
            functions: [
                .init(
                    name: "main",
                    inputs: [
                        .init(name: "input_ids", dataType: .int32, dimensions: [.fixed(1), sequence]),
                        .init(name: "position_ids", dataType: .int32, dimensions: [.fixed(1), sequence]),
                    ],
                    outputs: [
                        .init(
                            name: "logits",
                            dataType: .float16,
                            dimensions: [.fixed(1), sequence, .fixed(configuration.vocabSize)]
                        )
                    ],
                    states: []
                )
            ]
        )
    }

    private func makeStatefulContract(
        graph: ModelGraph,
        configuration: CoreAIExportConfiguration
    ) throws -> Result {
        guard configuration.target == .macOSDynamic else {
            throw CoreAIExportError.invalidConfiguration(
                "stateful execution currently requires the macos_dynamic target"
            )
        }

        var collector = PrimitiveCollector()
        collector.collect(graph.rootRegion)
        var states: [CoreAIProgramContract.Tensor] = []
        var stateBindings: [OperationKey: [CoreAIExportDocument.StateBinding]] = [:]

        let attentionGroups = Dictionary(grouping: collector.attention) {
            AttentionShape(kvHeads: $0.attributes.kvHeadCount, headDimension: $0.attributes.headDimension)
        }
        for (groupIndex, group) in attentionGroups.sorted(by: { $0.key < $1.key }).enumerated() {
            let suffix = attentionGroups.count == 1 ? "" : String(groupIndex)
            let keyName = "keyCache\(suffix)"
            let valueName = "valueCache\(suffix)"
            let cacheLength = CoreAIProgramContract.Dimension.dynamic(
                "cache_length",
                minimum: 1,
                maximum: configuration.maxContextLength
            )
            let dimensions: [CoreAIProgramContract.Dimension] = [
                .fixed(group.value.count),
                .fixed(1),
                .fixed(group.key.kvHeads),
                cacheLength,
                .fixed(group.key.headDimension),
            ]
            states.append(.init(name: keyName, dataType: .float16, dimensions: dimensions))
            states.append(.init(name: valueName, dataType: .float16, dimensions: dimensions))
            for (axisIndex, operation) in group.value.enumerated() {
                stateBindings[operation.key, default: []].append(
                    .init(role: "key_cache", state: keyName, axisIndex: axisIndex)
                )
                stateBindings[operation.key, default: []].append(
                    .init(role: "value_cache", state: valueName, axisIndex: axisIndex)
                )
            }
        }

        let convolutionGroups = Dictionary(grouping: collector.shortConvolution) {
            ConvolutionShape(
                hiddenSize: $0.attributes.hiddenSize,
                kernelSize: $0.attributes.kernelSize
            )
        }
        for (groupIndex, group) in convolutionGroups.sorted(by: { $0.key < $1.key }).enumerated() {
            let suffix = convolutionGroups.count == 1 ? "" : String(groupIndex)
            let name = "convCache\(suffix)"
            states.append(
                .init(
                    name: name,
                    dataType: .float16,
                    dimensions: [
                        .fixed(group.value.count),
                        .fixed(1),
                        .fixed(group.key.hiddenSize),
                        .fixed(group.key.kernelSize),
                    ]
                )
            )
            for (axisIndex, operation) in group.value.enumerated() {
                stateBindings[operation.key, default: []].append(
                    .init(role: "conv_cache", state: name, axisIndex: axisIndex)
                )
            }
        }

        guard !states.isEmpty else {
            throw CoreAIExportError.invalidConfiguration(
                "stateful execution requires at least one mutable LMIR operation"
            )
        }

        let prefix = CoreAIProgramContract.Dimension.dynamic(
            "prefix_length",
            minimum: 1,
            maximum: configuration.maxContextLength
        )
        let contract = CoreAIProgramContract(
            execution: .stateful,
            functions: [
                .init(
                    name: "main",
                    inputs: [
                        .init(
                            name: "input_ids",
                            dataType: .int32,
                            dimensions: [.fixed(1), .fixed(1)]
                        ),
                        .init(
                            name: "position_ids",
                            dataType: .int32,
                            dimensions: [.fixed(1), prefix]
                        ),
                    ],
                    outputs: [
                        .init(
                            name: "logits",
                            dataType: .float16,
                            dimensions: [.fixed(1), .fixed(1), .fixed(configuration.vocabSize)]
                        )
                    ],
                    states: states
                )
            ]
        )
        return Result(contract: contract, stateBindings: stateBindings)
    }
}

private struct PrimitiveCollector {
    struct AttentionOperation {
        let key: OperationKey
        let attributes: AttentionAttributes
    }

    struct ShortConvolutionOperation {
        let key: OperationKey
        let attributes: ShortConvAttributes
    }

    var attention: [AttentionOperation] = []
    var shortConvolution: [ShortConvolutionOperation] = []

    mutating func collect(_ region: Region) {
        for operation in region.operations {
            switch operation.kind {
            case .primitive(let attributes):
                if let attributes = attributes as? AttentionAttributes {
                    attention.append(.init(key: operation.key, attributes: attributes))
                } else if let attributes = attributes as? ShortConvAttributes {
                    shortConvolution.append(.init(key: operation.key, attributes: attributes))
                }
            case .residual(_, let body), .repeating(_, let body):
                collect(body)
            case .parallel(_, let branches):
                for branch in branches {
                    collect(branch)
                }
            case .conditional(_, let thenRegion, let elseRegion):
                collect(thenRegion)
                collect(elseRegion)
            }
        }
    }
}

private struct AttentionShape: Hashable, Comparable {
    let kvHeads: Int
    let headDimension: Int

    static func < (lhs: AttentionShape, rhs: AttentionShape) -> Bool {
        (lhs.kvHeads, lhs.headDimension) < (rhs.kvHeads, rhs.headDimension)
    }
}

private struct ConvolutionShape: Hashable, Comparable {
    let hiddenSize: Int
    let kernelSize: Int

    static func < (lhs: ConvolutionShape, rhs: ConvolutionShape) -> Bool {
        (lhs.hiddenSize, lhs.kernelSize) < (rhs.hiddenSize, rhs.kernelSize)
    }
}
