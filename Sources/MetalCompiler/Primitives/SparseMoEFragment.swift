import Metal
import LMIR

public struct SparseMoEFragment: PrimitiveMetalKernelFragment {
    private struct ProjectionKernelSelection {
        let gateUpName: String
        let downName: String
        let usesGateUpRow2: Bool
        let usesGateUpSplit2: Bool
        let usesDownSplit2: Bool
    }

    public let expertCount: Int
    public let expertsPerToken: Int
    public let gateKind: MoEGateKind
    public let inputDimension: Int
    public let outputDimension: Int
    public let intermediateDimension: Int
    public let normalizeRoutingWeights: Bool
    public let routedScalingFactor: Float
    public let useExpertBias: Bool

    public init(
        expertCount: Int,
        expertsPerToken: Int,
        gateKind: MoEGateKind,
        inputDimension: Int,
        outputDimension: Int,
        intermediateDimension: Int,
        normalizeRoutingWeights: Bool,
        routedScalingFactor: Float,
        useExpertBias: Bool
    ) {
        precondition(expertCount > 0, "expertCount must be positive")
        precondition(expertsPerToken > 0, "expertsPerToken must be positive")
        precondition(expertsPerToken <= expertCount, "expertsPerToken must not exceed expertCount")
        precondition(expertCount <= 128, "SparseMoEFragment supports up to 128 experts")
        precondition(expertsPerToken <= 8, "SparseMoEFragment supports top-k up to 8")
        switch gateKind {
        case .topK, .sigmoidTopK:
            break
        case .custom:
            preconditionFailure("SparseMoEFragment does not support custom MoE routing gates")
        }
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.gateKind = gateKind
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        self.intermediateDimension = intermediateDimension
        self.normalizeRoutingWeights = normalizeRoutingWeights
        self.routedScalingFactor = routedScalingFactor
        self.useExpertBias = useExpertBias
    }

    public var dispatchDimension: MetalDispatchDimension {
        .sparseMoE(outputDimension: outputDimension, inputDimension: inputDimension, maxSimdgroups: 32)
    }

    public var scratchElementsPerToken: Int {
        2 * expertsPerToken + 2 * 128 + expertsPerToken * intermediateDimension
    }

    var usesSplitRoute: Bool {
        !Self.isEnabled(ProcessInfo.processInfo.environment["SWIFTLM_DIAGNOSTIC_SPARSE_MOE_MONOLITHIC"])
    }

    public var weightSlots: [MetalWeightSlot] {
        var slots = [
            MetalWeightSlot(field: "router", role: .weight),
            MetalWeightSlot(field: "expert_gate_up_proj", role: .weight),
            MetalWeightSlot(field: "expert_down_proj", role: .weight),
        ]
        if useExpertBias {
            slots.append(MetalWeightSlot(field: "expert_bias", role: .weight))
        }
        return slots
    }

    public func kernelName(context: KernelContext) -> String {
        let weightSuffix: String
        if context.weightFormat.isBFloat16 {
            weightSuffix = "_bf16"
        } else if context.weightFormat.isQuantized {
            weightSuffix = "_q\(context.weightFormat.bits)_g\(context.weightFormat.groupSize)"
        } else {
            weightSuffix = ""
        }
        return context.bufferPrecision.isPrefillSequencePrecision
            ? "sparse_moe_seq\(weightSuffix)_f32"
            : "sparse_moe\(weightSuffix)\(context.bufferPrecision.decodeKernelNameSuffix)"
    }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        MetalSourceGenerator.generateSparseMoE(
            name: name,
            bufferPrecision: bufferPrecision,
            weightFormat: weightFormat,
            gateKind: gateKind
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (routerBuffer, routerOffset) = context.resolveWeight("router")
        let (gateUpBuffer, gateUpOffset) = context.resolveWeight("expert_gate_up_proj")
        let (downBuffer, downOffset) = context.resolveWeight("expert_down_proj")
        let (biasBuffer, biasOffset) = useExpertBias
            ? context.resolveWeight("expert_bias")
            : (routerBuffer, routerOffset)

        var buffers: [(index: Int, buffer: MTLBuffer, offset: Int)] = [
                (0, context.currentInputBuffer, context.currentInputOffset),
                (1, routerBuffer, routerOffset),
                (2, gateUpBuffer, gateUpOffset),
                (3, downBuffer, downOffset),
                (4, biasBuffer, biasOffset),
                (5, context.bufferSet.hidden, 0),
        ]
        if let moeScratch = context.bufferSet.moeScratch {
            buffers.append((6, moeScratch, 0))
        }

        return FragmentBindings(
            buffers: buffers,
            bytes: Self.constantBindings(
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                intermediateDimension: intermediateDimension,
                expertCount: expertCount,
                expertsPerToken: expertsPerToken,
                normalizeRoutingWeights: normalizeRoutingWeights,
                routedScalingFactor: routedScalingFactor,
                useExpertBias: useExpertBias,
                sequenceLength: 1,
                inputRowStride: inputDimension,
                outputRowStride: outputDimension,
                startingIndex: 6
            ),
            outputIsHidden: true,
            resetsProjectionIndex: true,
            writeBufferIndices: Set<Int>([5])
        )
    }

    func splitDecodeSteps(
        bindings: (buffers: [(index: Int, buffer: MTLBuffer, offset: Int)], bytes: [(index: Int, value: [UInt8])]),
        pipelineCache: [String: MTLComputePipelineState],
        kernelContext: KernelContext
    ) throws -> [MetalDispatchStep] {
        guard let inputBinding = Self.binding(at: 0, in: bindings.buffers),
              let routerBinding = Self.binding(at: 1, in: bindings.buffers),
              let gateUpBinding = Self.binding(at: 2, in: bindings.buffers),
              let downBinding = Self.binding(at: 3, in: bindings.buffers),
              let biasBinding = Self.binding(at: 4, in: bindings.buffers),
              let outputBinding = Self.binding(at: 5, in: bindings.buffers),
              let moeScratchBinding = Self.binding(at: 6, in: bindings.buffers) else {
            throw MetalCompilerError.deviceSetupFailed("Sparse MoE split decode received incomplete bindings")
        }
        let moeScratch = moeScratchBinding.buffer
        let baseName = kernelName(context: kernelContext)
        guard let routerScoresPipeline = pipelineCache["\(baseName)_router_scores"] else {
            throw MetalCompilerError.kernelNotFound("\(baseName)_router_scores")
        }
        guard let routerSelectPipeline = pipelineCache["\(baseName)_router_select"] else {
            throw MetalCompilerError.kernelNotFound("\(baseName)_router_select")
        }
        guard let routerParallelPipeline = pipelineCache["\(baseName)_router_parallel"] else {
            throw MetalCompilerError.kernelNotFound("\(baseName)_router_parallel")
        }

        let routerBuffer = routerBinding.buffer
        let routerOffset = routerBinding.offset
        let gateUpBuffer = gateUpBinding.buffer
        let gateUpOffset = gateUpBinding.offset
        let downBuffer = downBinding.buffer
        let downOffset = downBinding.offset
        let biasBuffer = biasBinding.buffer
        let biasOffset = biasBinding.offset
        let scratchRowStride = scratchElementsPerToken

        let routerScoresThreads = max(routerScoresPipeline.threadExecutionWidth, 1)
        let routerSelectThreads = max(routerSelectPipeline.threadExecutionWidth, 1)
        let routerParallelSimdWidth = max(routerParallelPipeline.threadExecutionWidth, 1)
        let routerParallelMaxSimdgroups = max(1, routerParallelPipeline.maxTotalThreadsPerThreadgroup / routerParallelSimdWidth)
        let disablesParallelRouter = Self.isEnabled(ProcessInfo.processInfo.environment["SWIFTLM_SPARSE_MOE_DISABLE_ROUTER_PARALLEL"])
        let usesParallelRouter = !disablesParallelRouter
            && expertCount <= routerParallelMaxSimdgroups
        let routerParallelThreads = expertCount * routerParallelSimdWidth
        let flatGateUpCount = expertsPerToken * intermediateDimension
        let projectionSelection = projectionKernelSelection(
            baseName: baseName,
            weightFormat: kernelContext.weightFormat,
            usesExperimentalKernels: true
        )
        guard let selectedGateUpPipeline = pipelineCache[projectionSelection.gateUpName] else {
            throw MetalCompilerError.kernelNotFound(projectionSelection.gateUpName)
        }
        let gateUpSimdWidth = max(selectedGateUpPipeline.threadExecutionWidth, 1)
        let requestedGateUpSimdgroups = Self.resolvedSimdgroups(
            environmentKey: "SWIFTLM_SPARSE_MOE_GATE_UP_SIMDGROUPS",
            defaultValue: 32,
            simdWidth: gateUpSimdWidth,
            pipeline: selectedGateUpPipeline
        )
        let gateUpSimdgroups = projectionSelection.usesGateUpSplit2
            ? max(2, requestedGateUpSimdgroups - requestedGateUpSimdgroups % 2)
            : requestedGateUpSimdgroups
        let gateUpThreads = gateUpSimdgroups * gateUpSimdWidth
        let gateUpRowsPerThreadgroup = projectionSelection.usesGateUpSplit2
            ? max(1, gateUpSimdgroups / 2)
            : gateUpSimdgroups
        let gateUpEffectiveRowsPerThreadgroup = projectionSelection.usesGateUpRow2
            ? gateUpRowsPerThreadgroup * 2
            : gateUpRowsPerThreadgroup
        let gateUpGridX = (flatGateUpCount + gateUpEffectiveRowsPerThreadgroup - 1) / gateUpEffectiveRowsPerThreadgroup
        let gateUpThreadgroupMemoryLength = projectionSelection.usesGateUpSplit2
            ? gateUpRowsPerThreadgroup * 2 * 2 * MemoryLayout<Float>.stride
            : 0
        guard let selectedDownPipeline = pipelineCache[projectionSelection.downName] else {
            throw MetalCompilerError.kernelNotFound(projectionSelection.downName)
        }
        let downSimdWidth = max(selectedDownPipeline.threadExecutionWidth, 1)
        let requestedDownSimdgroups = Self.resolvedSimdgroups(
            environmentKey: "SWIFTLM_SPARSE_MOE_DOWN_SIMDGROUPS",
            defaultValue: 32,
            simdWidth: downSimdWidth,
            pipeline: selectedDownPipeline
        )
        let downSimdgroups = projectionSelection.usesDownSplit2
            ? max(2, requestedDownSimdgroups - requestedDownSimdgroups % 2)
            : requestedDownSimdgroups
        let downThreads = downSimdgroups * downSimdWidth
        let downRowsPerThreadgroup = projectionSelection.usesDownSplit2
            ? max(1, downSimdgroups / 2)
            : downSimdgroups
        let downGridX = (outputDimension + downRowsPerThreadgroup - 1) / downRowsPerThreadgroup
        let downThreadgroupMemoryLength = projectionSelection.usesDownSplit2
            ? downRowsPerThreadgroup * 2 * MemoryLayout<Float>.stride
            : 0

        var steps: [MetalDispatchStep] = []
        if usesParallelRouter {
            steps.append(MetalDispatchStep(
                pipeline: routerParallelPipeline,
                gridSize: MTLSize(width: 1, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: routerParallelThreads, height: 1, depth: 1),
                bufferBindings: [
                    (0, inputBinding.buffer, inputBinding.offset),
                    (1, routerBuffer, routerOffset),
                    (2, biasBuffer, biasOffset),
                    (3, moeScratch, 0),
                ],
                bytesBindings: Self.routerConstantBindings(
                    inputDimension: inputDimension,
                    expertCount: expertCount,
                    expertsPerToken: expertsPerToken,
                    normalizeRoutingWeights: normalizeRoutingWeights,
                    routedScalingFactor: routedScalingFactor,
                    useExpertBias: useExpertBias,
                    sequenceLength: 1,
                    inputRowStride: inputDimension,
                    scratchRowStride: scratchRowStride,
                    startingIndex: 4
                ),
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                bufferAccesses: MetalBufferAccesses(
                    readBuffers: [
                        (buffer: inputBinding.buffer, offset: inputBinding.offset),
                        (buffer: routerBuffer, offset: routerOffset),
                        (buffer: biasBuffer, offset: biasOffset),
                    ],
                    writeBuffers: [(buffer: moeScratch, offset: 0)]
                ),
                metadata: .init(
                    kernelName: "\(baseName)_router_parallel",
                    bufferAccessPattern: .init(reads: [0, 1, 2], writes: [3])
                )
            ))
        } else {
            steps.append(MetalDispatchStep(
                pipeline: routerScoresPipeline,
                gridSize: MTLSize(width: expertCount, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: routerScoresThreads, height: 1, depth: 1),
                bufferBindings: [
                    (0, inputBinding.buffer, inputBinding.offset),
                    (1, routerBuffer, routerOffset),
                    (2, biasBuffer, biasOffset),
                    (3, moeScratch, 0),
                ],
                bytesBindings: Self.routerConstantBindings(
                    inputDimension: inputDimension,
                    expertCount: expertCount,
                    expertsPerToken: expertsPerToken,
                    normalizeRoutingWeights: normalizeRoutingWeights,
                    routedScalingFactor: routedScalingFactor,
                    useExpertBias: useExpertBias,
                    sequenceLength: 1,
                    inputRowStride: inputDimension,
                    scratchRowStride: scratchRowStride,
                    startingIndex: 4
                ),
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                bufferAccesses: MetalBufferAccesses(
                    readBuffers: [
                        (buffer: inputBinding.buffer, offset: inputBinding.offset),
                        (buffer: routerBuffer, offset: routerOffset),
                        (buffer: biasBuffer, offset: biasOffset),
                    ],
                    writeBuffers: [(buffer: moeScratch, offset: 0)]
                ),
                metadata: .init(
                    kernelName: "\(baseName)_router_scores",
                    bufferAccessPattern: .init(reads: [0, 1, 2], writes: [3])
                )
            ))
            steps.append(MetalDispatchStep(
                pipeline: routerSelectPipeline,
                gridSize: MTLSize(width: 1, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: routerSelectThreads, height: 1, depth: 1),
                bufferBindings: [
                    (0, moeScratch, 0),
                ],
                bytesBindings: Self.routerSelectConstantBindings(
                    expertCount: expertCount,
                    expertsPerToken: expertsPerToken,
                    normalizeRoutingWeights: normalizeRoutingWeights,
                    routedScalingFactor: routedScalingFactor,
                    sequenceLength: 1,
                    scratchRowStride: scratchRowStride,
                    startingIndex: 1
                ),
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                bufferAccesses: MetalBufferAccesses(
                    readBuffers: [(buffer: moeScratch, offset: 0)],
                    writeBuffers: [(buffer: moeScratch, offset: 0)]
                ),
                metadata: .init(
                    kernelName: "\(baseName)_router_select",
                    bufferAccessPattern: .init(reads: [0], writes: [0])
                )
            ))
        }
        steps.append(MetalDispatchStep(
                pipeline: selectedGateUpPipeline,
                gridSize: MTLSize(width: gateUpGridX, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: gateUpThreads, height: 1, depth: 1),
                bufferBindings: [
                    (0, inputBinding.buffer, inputBinding.offset),
                    (1, gateUpBuffer, gateUpOffset),
                    (2, moeScratch, 0),
                ],
                bytesBindings: Self.gateUpConstantBindings(
                    inputDimension: inputDimension,
                    intermediateDimension: intermediateDimension,
                    expertsPerToken: expertsPerToken,
                    sequenceLength: 1,
                    inputRowStride: inputDimension,
                    scratchRowStride: scratchRowStride,
                    startingIndex: 3
                ),
                threadgroupMemoryLength: gateUpThreadgroupMemoryLength,
                sync: .bufferBarrier,
                bufferAccesses: MetalBufferAccesses(
                    readBuffers: [
                        (buffer: inputBinding.buffer, offset: inputBinding.offset),
                        (buffer: gateUpBuffer, offset: gateUpOffset),
                        (buffer: moeScratch, offset: 0),
                    ],
                    writeBuffers: [(buffer: moeScratch, offset: 0)]
                ),
                metadata: .init(
                    kernelName: projectionSelection.gateUpName,
                    bufferAccessPattern: .init(reads: [0, 1, 2], writes: [2])
                )
            ))
        steps.append(MetalDispatchStep(
                pipeline: selectedDownPipeline,
                gridSize: MTLSize(width: downGridX, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: downThreads, height: 1, depth: 1),
                bufferBindings: [
                    (0, moeScratch, 0),
                    (1, downBuffer, downOffset),
                    (2, outputBinding.buffer, outputBinding.offset),
                ],
                bytesBindings: Self.downConstantBindings(
                    outputDimension: outputDimension,
                    intermediateDimension: intermediateDimension,
                    expertsPerToken: expertsPerToken,
                    sequenceLength: 1,
                    outputRowStride: outputDimension,
                    scratchRowStride: scratchRowStride,
                    startingIndex: 3
                ),
                threadgroupMemoryLength: downThreadgroupMemoryLength,
                sync: .bufferBarrier,
                bufferAccesses: MetalBufferAccesses(
                    readBuffers: [
                        (buffer: moeScratch, offset: 0),
                        (buffer: downBuffer, offset: downOffset),
                    ],
                    writeBuffers: [(buffer: outputBinding.buffer, offset: outputBinding.offset)]
                ),
                metadata: .init(
                    kernelName: projectionSelection.downName,
                    bufferAccessPattern: .init(reads: [0, 1], writes: [2])
                )
            ))
        return steps
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        guard usesSplitRoute else {
            return try monolithicPrefillSteps(context: context)
        }
        guard let moeScratch = context.buffers.moeScratch else {
            throw MetalCompilerError.deviceSetupFailed("Sparse MoE split prefill requires moeScratch")
        }
        let kernelName = kernelName(context: context.kernelContext)
        let projectionSelection = projectionKernelSelection(
            baseName: kernelName,
            weightFormat: context.kernelContext.weightFormat,
            usesExperimentalKernels: false
        )
        let routerScoresPipeline = try context.getPipeline("\(kernelName)_router_scores")
        let routerSelectPipeline = try context.getPipeline("\(kernelName)_router_select")
        let gateUpPipeline = try context.getPipeline(projectionSelection.gateUpName)
        let downPipeline = try context.getPipeline(projectionSelection.downName)
        let routerScoresThreads = max(routerScoresPipeline.threadExecutionWidth, 1)
        let routerSelectThreads = max(routerSelectPipeline.threadExecutionWidth, 1)
        let flatGateUpCount = expertsPerToken * intermediateDimension
        let gateUpSimdWidth = max(gateUpPipeline.threadExecutionWidth, 1)
        let requestedGateUpSimdgroups = Self.resolvedSimdgroups(
            environmentKey: "SWIFTLM_SPARSE_MOE_GATE_UP_SIMDGROUPS",
            defaultValue: 32,
            simdWidth: gateUpSimdWidth,
            pipeline: gateUpPipeline
        )
        let gateUpSimdgroups = projectionSelection.usesGateUpSplit2
            ? max(2, requestedGateUpSimdgroups - requestedGateUpSimdgroups % 2)
            : requestedGateUpSimdgroups
        let gateUpThreads = gateUpSimdgroups * gateUpSimdWidth
        let gateUpRowsPerThreadgroup = projectionSelection.usesGateUpSplit2
            ? max(1, gateUpSimdgroups / 2)
            : gateUpSimdgroups
        let gateUpEffectiveRowsPerThreadgroup = projectionSelection.usesGateUpRow2
            ? gateUpRowsPerThreadgroup * 2
            : gateUpRowsPerThreadgroup
        let gateUpGridX = (flatGateUpCount + gateUpEffectiveRowsPerThreadgroup - 1) / gateUpEffectiveRowsPerThreadgroup
        let gateUpThreadgroupMemoryLength = projectionSelection.usesGateUpSplit2
            ? gateUpRowsPerThreadgroup * 2 * 2 * MemoryLayout<Float>.stride
            : 0
        let simdWidth = max(downPipeline.threadExecutionWidth, 1)
        let requestedSimdgroups = Self.resolvedSimdgroups(
            environmentKey: "SWIFTLM_SPARSE_MOE_DOWN_SIMDGROUPS",
            defaultValue: 32,
            simdWidth: simdWidth,
            pipeline: downPipeline
        )
        let simdgroups = projectionSelection.usesDownSplit2
            ? max(2, requestedSimdgroups - requestedSimdgroups % 2)
            : requestedSimdgroups
        let threads = simdgroups * simdWidth
        let downRowsPerThreadgroup = projectionSelection.usesDownSplit2
            ? max(1, simdgroups / 2)
            : simdgroups
        let gridX = (outputDimension + downRowsPerThreadgroup - 1) / downRowsPerThreadgroup
        let downThreadgroupMemoryLength = projectionSelection.usesDownSplit2
            ? downRowsPerThreadgroup * 2 * MemoryLayout<Float>.stride
            : 0
        let (routerBuffer, routerOffset) = context.resolveWeight("router")
        let (gateUpBuffer, gateUpOffset) = context.resolveWeight("expert_gate_up_proj")
        let (downBuffer, downOffset) = context.resolveWeight("expert_down_proj")
        let (biasBuffer, biasOffset) = useExpertBias
            ? context.resolveWeight("expert_bias")
            : (routerBuffer, routerOffset)

        let scratchRowStride = scratchElementsPerToken
        let inputRowStride = Self.inputRowStride(context: context)

        return FragmentPrefillSteps(
            steps: [
                MetalPrefillStep(
                    pipeline: routerScoresPipeline,
                    gridSize: MTLSize(width: expertCount, height: context.maximumSequenceLength, depth: 1),
                    threadgroupSize: MTLSize(width: routerScoresThreads, height: 1, depth: 1),
                    bufferBindings: [
                        (0, context.currentInputBuffer, context.currentInputOffset),
                        (1, routerBuffer, routerOffset),
                        (2, biasBuffer, biasOffset),
                        (3, moeScratch, 0),
                    ],
                    bytesBindings: Self.routerConstantBindings(
                        inputDimension: inputDimension,
                        expertCount: expertCount,
                        expertsPerToken: expertsPerToken,
                        normalizeRoutingWeights: normalizeRoutingWeights,
                        routedScalingFactor: routedScalingFactor,
                        useExpertBias: useExpertBias,
                        sequenceLength: context.maximumSequenceLength,
                        inputRowStride: inputRowStride,
                        scratchRowStride: scratchRowStride,
                        startingIndex: 4
                    ),
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 10),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: "\(kernelName)_router_scores",
                        bufferAccessPattern: .init(reads: [0, 1, 2], writes: [3])
                    )
                ),
                MetalPrefillStep(
                    pipeline: routerSelectPipeline,
                    gridSize: MTLSize(width: context.maximumSequenceLength, height: 1, depth: 1),
                    threadgroupSize: MTLSize(width: routerSelectThreads, height: 1, depth: 1),
                    bufferBindings: [
                        (0, moeScratch, 0),
                    ],
                    bytesBindings: Self.routerSelectConstantBindings(
                        expertCount: expertCount,
                        expertsPerToken: expertsPerToken,
                        normalizeRoutingWeights: normalizeRoutingWeights,
                        routedScalingFactor: routedScalingFactor,
                        sequenceLength: context.maximumSequenceLength,
                        scratchRowStride: scratchRowStride,
                        startingIndex: 1
                    ),
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bind(index: 5),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: "\(kernelName)_router_select",
                        bufferAccessPattern: .init(reads: [0], writes: [0])
                    )
                ),
                MetalPrefillStep(
                    pipeline: gateUpPipeline,
                    gridSize: MTLSize(
                        width: gateUpGridX,
                        height: context.maximumSequenceLength,
                        depth: 1
                    ),
                    threadgroupSize: MTLSize(width: gateUpThreads, height: 1, depth: 1),
                    bufferBindings: [
                        (0, context.currentInputBuffer, context.currentInputOffset),
                        (1, gateUpBuffer, gateUpOffset),
                        (2, moeScratch, 0),
                    ],
                    bytesBindings: Self.gateUpConstantBindings(
                        inputDimension: inputDimension,
                        intermediateDimension: intermediateDimension,
                        expertsPerToken: expertsPerToken,
                        sequenceLength: context.maximumSequenceLength,
                        inputRowStride: inputRowStride,
                        scratchRowStride: scratchRowStride,
                        startingIndex: 3
                    ),
                    threadgroupMemoryLength: gateUpThreadgroupMemoryLength,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 6),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: projectionSelection.gateUpName,
                        bufferAccessPattern: .init(reads: [0, 1, 2], writes: [2])
                    )
                ),
                MetalPrefillStep(
                    pipeline: downPipeline,
                    gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                    threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                    bufferBindings: [
                        (0, moeScratch, 0),
                        (1, downBuffer, downOffset),
                        (2, context.buffers.hidden, 0),
                    ],
                    bytesBindings: Self.downConstantBindings(
                        outputDimension: outputDimension,
                        intermediateDimension: intermediateDimension,
                        expertsPerToken: expertsPerToken,
                        sequenceLength: context.maximumSequenceLength,
                        outputRowStride: outputDimension,
                        scratchRowStride: scratchRowStride,
                        startingIndex: 3
                    ),
                    threadgroupMemoryLength: downThreadgroupMemoryLength,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 6),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: projectionSelection.downName,
                        bufferAccessPattern: .init(reads: [0, 1], writes: [2])
                    )
                )
            ],
            outputIsHidden: true,
            resetsProjectionIndex: true
        )
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        switch role {
        case "router":
            return expertCount * inputDimension * bytesPerScalar
        case "expert_gate_up_proj":
            return expertCount * 2 * intermediateDimension * inputDimension * bytesPerScalar
        case "expert_down_proj":
            return expertCount * outputDimension * intermediateDimension * bytesPerScalar
        case "expert_bias":
            return expertCount * MemoryLayout<Float>.stride
        default:
            return 0
        }
    }

    private static func resolvedSimdgroups(
        environmentKey: String,
        defaultValue: Int,
        simdWidth: Int,
        pipeline: MTLComputePipelineState
    ) -> Int {
        let maxSupported = max(1, pipeline.maxTotalThreadsPerThreadgroup / max(simdWidth, 1))
        let requested = ProcessInfo.processInfo.environment[environmentKey].flatMap(Int.init) ?? defaultValue
        return max(1, min(requested, maxSupported))
    }

    private func monolithicPrefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let simdgroups = max(1, min(32, pipeline.maxTotalThreadsPerThreadgroup / simdWidth))
        let threads = simdgroups * simdWidth
        let gridX = (outputDimension + simdgroups - 1) / simdgroups
        let (routerBuffer, routerOffset) = context.resolveWeight("router")
        let (gateUpBuffer, gateUpOffset) = context.resolveWeight("expert_gate_up_proj")
        let (downBuffer, downOffset) = context.resolveWeight("expert_down_proj")
        let (biasBuffer, biasOffset) = useExpertBias
            ? context.resolveWeight("expert_bias")
            : (routerBuffer, routerOffset)
        let inputRowStride = Self.inputRowStride(context: context)

        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.currentInputBuffer, context.currentInputOffset),
                    (1, routerBuffer, routerOffset),
                    (2, gateUpBuffer, gateUpOffset),
                    (3, downBuffer, downOffset),
                    (4, biasBuffer, biasOffset),
                    (5, context.buffers.hidden, 0),
                ],
                bytesBindings: Self.constantBindings(
                    inputDimension: inputDimension,
                    outputDimension: outputDimension,
                    intermediateDimension: intermediateDimension,
                    expertCount: expertCount,
                    expertsPerToken: expertsPerToken,
                    normalizeRoutingWeights: normalizeRoutingWeights,
                    routedScalingFactor: routedScalingFactor,
                    useExpertBias: useExpertBias,
                    sequenceLength: context.maximumSequenceLength,
                    inputRowStride: inputRowStride,
                    outputRowStride: outputDimension,
                    startingIndex: 6
                ),
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 14),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: kernelName,
                    bufferAccessPattern: .init(reads: [0, 1, 2, 3, 4], writes: [5])
                )
            )],
            outputIsHidden: true,
            resetsProjectionIndex: true
        )
    }

    private static func isEnabled(_ value: String?) -> Bool {
        guard let value else { return false }
        return value == "1" || value.lowercased() == "true"
    }

    private func projectionKernelSelection(
        baseName: String,
        weightFormat: WeightFormat,
        usesExperimentalKernels: Bool
    ) -> ProjectionKernelSelection {
        let usesPacked4 = weightFormat.isBFloat16
            && inputDimension.isMultiple(of: 4)
            && intermediateDimension.isMultiple(of: 4)
            && !Self.isEnabled(ProcessInfo.processInfo.environment["SWIFTLM_SPARSE_MOE_DISABLE_PACKED4"])
        let usesPacked8 = usesPacked4
            && inputDimension.isMultiple(of: 8)
            && intermediateDimension.isMultiple(of: 8)
            && usesExperimentalKernels
            && Self.isEnabled(ProcessInfo.processInfo.environment["SWIFTLM_SPARSE_MOE_ENABLE_PACKED8"])
        let usesGateUpSplit2 = usesExperimentalKernels
            && Self.isEnabled(ProcessInfo.processInfo.environment["SWIFTLM_SPARSE_MOE_GATE_UP_SPLIT2"])
        let usesGateUpRow2 = usesPacked4
            && usesExperimentalKernels
            && !usesGateUpSplit2
            && Self.isEnabled(ProcessInfo.processInfo.environment["SWIFTLM_SPARSE_MOE_GATE_UP_ROW2"])
        let usesDownSplit2 = usesExperimentalKernels
            && Self.isEnabled(ProcessInfo.processInfo.environment["SWIFTLM_SPARSE_MOE_DOWN_SPLIT2"])
        let gateUpName = usesGateUpSplit2
            ? "\(baseName)_gate_up_split2"
            : (
                usesGateUpRow2
                    ? "\(baseName)_gate_up_row2_packed4"
                    : (
                        usesPacked8
                            ? "\(baseName)_gate_up_packed8"
                            : (usesPacked4 ? "\(baseName)_gate_up_packed4" : "\(baseName)_gate_up")
                    )
            )
        let downName = usesDownSplit2
            ? "\(baseName)_down_split2"
            : (
                usesPacked8
                    ? "\(baseName)_down_packed8"
                    : (usesPacked4 ? "\(baseName)_down_packed4" : "\(baseName)_down")
            )

        return ProjectionKernelSelection(
            gateUpName: gateUpName,
            downName: downName,
            usesGateUpRow2: usesGateUpRow2,
            usesGateUpSplit2: usesGateUpSplit2,
            usesDownSplit2: usesDownSplit2
        )
    }

    private static func inputRowStride(context: PrefillBindingContext) -> Int {
        if context.currentInputBuffer === context.buffers.hidden {
            return (context.buffers.hidden.length / max(context.maximumSequenceLength, 1))
                / max(context.buffers.bufferPrecision.byteSize, 1)
        }
        return context.slotDimension
    }

    private static func constantBindings(
        inputDimension: Int,
        outputDimension: Int,
        intermediateDimension: Int,
        expertCount: Int,
        expertsPerToken: Int,
        normalizeRoutingWeights: Bool,
        routedScalingFactor: Float,
        useExpertBias: Bool,
        sequenceLength: Int,
        inputRowStride: Int,
        outputRowStride: Int,
        startingIndex: Int
    ) -> [(index: Int, value: [UInt8])] {
        [
            uint32Binding(startingIndex + 0, UInt32(inputDimension)),
            uint32Binding(startingIndex + 1, UInt32(outputDimension)),
            uint32Binding(startingIndex + 2, UInt32(intermediateDimension)),
            uint32Binding(startingIndex + 3, UInt32(expertCount)),
            uint32Binding(startingIndex + 4, UInt32(expertsPerToken)),
            uint32Binding(startingIndex + 5, normalizeRoutingWeights ? 1 : 0),
            floatBinding(startingIndex + 6, routedScalingFactor),
            uint32Binding(startingIndex + 7, useExpertBias ? 1 : 0),
            uint32Binding(startingIndex + 8, UInt32(sequenceLength)),
            uint32Binding(startingIndex + 9, UInt32(inputRowStride)),
            uint32Binding(startingIndex + 10, UInt32(outputRowStride)),
        ]
    }

    private static func binding(
        at index: Int,
        in bindings: [(index: Int, buffer: MTLBuffer, offset: Int)]
    ) -> (buffer: MTLBuffer, offset: Int)? {
        bindings.first(where: { $0.index == index }).map { (buffer: $0.buffer, offset: $0.offset) }
    }

    private static func routerConstantBindings(
        inputDimension: Int,
        expertCount: Int,
        expertsPerToken: Int,
        normalizeRoutingWeights: Bool,
        routedScalingFactor: Float,
        useExpertBias: Bool,
        sequenceLength: Int,
        inputRowStride: Int,
        scratchRowStride: Int,
        startingIndex: Int
    ) -> [(index: Int, value: [UInt8])] {
        [
            uint32Binding(startingIndex + 0, UInt32(inputDimension)),
            uint32Binding(startingIndex + 1, UInt32(expertCount)),
            uint32Binding(startingIndex + 2, UInt32(expertsPerToken)),
            uint32Binding(startingIndex + 3, normalizeRoutingWeights ? 1 : 0),
            floatBinding(startingIndex + 4, routedScalingFactor),
            uint32Binding(startingIndex + 5, useExpertBias ? 1 : 0),
            uint32Binding(startingIndex + 6, UInt32(sequenceLength)),
            uint32Binding(startingIndex + 7, UInt32(inputRowStride)),
            uint32Binding(startingIndex + 8, UInt32(scratchRowStride)),
        ]
    }

    private static func routerSelectConstantBindings(
        expertCount: Int,
        expertsPerToken: Int,
        normalizeRoutingWeights: Bool,
        routedScalingFactor: Float,
        sequenceLength: Int,
        scratchRowStride: Int,
        startingIndex: Int
    ) -> [(index: Int, value: [UInt8])] {
        [
            uint32Binding(startingIndex + 0, UInt32(expertCount)),
            uint32Binding(startingIndex + 1, UInt32(expertsPerToken)),
            uint32Binding(startingIndex + 2, normalizeRoutingWeights ? 1 : 0),
            floatBinding(startingIndex + 3, routedScalingFactor),
            uint32Binding(startingIndex + 4, UInt32(sequenceLength)),
            uint32Binding(startingIndex + 5, UInt32(scratchRowStride)),
        ]
    }

    private static func gateUpConstantBindings(
        inputDimension: Int,
        intermediateDimension: Int,
        expertsPerToken: Int,
        sequenceLength: Int,
        inputRowStride: Int,
        scratchRowStride: Int,
        startingIndex: Int
    ) -> [(index: Int, value: [UInt8])] {
        [
            uint32Binding(startingIndex + 0, UInt32(inputDimension)),
            uint32Binding(startingIndex + 1, UInt32(intermediateDimension)),
            uint32Binding(startingIndex + 2, UInt32(expertsPerToken)),
            uint32Binding(startingIndex + 3, UInt32(sequenceLength)),
            uint32Binding(startingIndex + 4, UInt32(inputRowStride)),
            uint32Binding(startingIndex + 5, UInt32(scratchRowStride)),
        ]
    }

    private static func downConstantBindings(
        outputDimension: Int,
        intermediateDimension: Int,
        expertsPerToken: Int,
        sequenceLength: Int,
        outputRowStride: Int,
        scratchRowStride: Int,
        startingIndex: Int
    ) -> [(index: Int, value: [UInt8])] {
        [
            uint32Binding(startingIndex + 0, UInt32(outputDimension)),
            uint32Binding(startingIndex + 1, UInt32(intermediateDimension)),
            uint32Binding(startingIndex + 2, UInt32(expertsPerToken)),
            uint32Binding(startingIndex + 3, UInt32(sequenceLength)),
            uint32Binding(startingIndex + 4, UInt32(outputRowStride)),
            uint32Binding(startingIndex + 5, UInt32(scratchRowStride)),
        ]
    }
}
