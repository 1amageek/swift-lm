import Metal
import LMIR

public struct SparseMoEFragment: PrimitiveMetalKernelFragment {
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
        expertsPerToken * (intermediateDimension + 2)
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
        let weightSuffix = context.weightFormat.isBFloat16 ? "_bf16" : ""
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
        guard let routerPipeline = pipelineCache["\(baseName)_router"] else {
            throw MetalCompilerError.kernelNotFound("\(baseName)_router")
        }
        guard let gateUpPipeline = pipelineCache["\(baseName)_gate_up"] else {
            throw MetalCompilerError.kernelNotFound("\(baseName)_gate_up")
        }
        guard let downPipeline = pipelineCache["\(baseName)_down"] else {
            throw MetalCompilerError.kernelNotFound("\(baseName)_down")
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

        let routerThreads = max(routerPipeline.threadExecutionWidth, 1)
        let gateUpThreads = min(256, max(gateUpPipeline.threadExecutionWidth, 1) * 8)
        let flatGateUpCount = expertsPerToken * intermediateDimension
        let downSimdWidth = max(downPipeline.threadExecutionWidth, 1)
        let downSimdgroups = max(1, min(32, downPipeline.maxTotalThreadsPerThreadgroup / downSimdWidth))
        let downThreads = downSimdgroups * downSimdWidth
        let downGridX = (outputDimension + downSimdgroups - 1) / downSimdgroups

        return [
            MetalDispatchStep(
                pipeline: routerPipeline,
                gridSize: MTLSize(width: 1, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: routerThreads, height: 1, depth: 1),
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
                    kernelName: "\(baseName)_router",
                    bufferAccessPattern: .init(reads: [0, 1, 2], writes: [3])
                )
            ),
            MetalDispatchStep(
                pipeline: gateUpPipeline,
                gridSize: MTLSize(width: (flatGateUpCount + gateUpThreads - 1) / gateUpThreads, height: 1, depth: 1),
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
                threadgroupMemoryLength: 0,
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
                    kernelName: "\(baseName)_gate_up",
                    bufferAccessPattern: .init(reads: [0, 1, 2], writes: [2])
                )
            ),
            MetalDispatchStep(
                pipeline: downPipeline,
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
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                bufferAccesses: MetalBufferAccesses(
                    readBuffers: [
                        (buffer: moeScratch, offset: 0),
                        (buffer: downBuffer, offset: downOffset),
                    ],
                    writeBuffers: [(buffer: outputBinding.buffer, offset: outputBinding.offset)]
                ),
                metadata: .init(
                    kernelName: "\(baseName)_down",
                    bufferAccessPattern: .init(reads: [0, 1], writes: [2])
                )
            ),
        ]
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        guard let moeScratch = context.buffers.moeScratch else {
            throw MetalCompilerError.deviceSetupFailed("Sparse MoE split prefill requires moeScratch")
        }
        let kernelName = kernelName(context: context.kernelContext)
        let routerPipeline = try context.getPipeline("\(kernelName)_router")
        let gateUpPipeline = try context.getPipeline("\(kernelName)_gate_up")
        let downPipeline = try context.getPipeline("\(kernelName)_down")
        let routerThreads = max(routerPipeline.threadExecutionWidth, 1)
        let gateUpThreads = min(256, max(gateUpPipeline.threadExecutionWidth, 1) * 8)
        let flatGateUpCount = expertsPerToken * intermediateDimension
        let simdWidth = max(downPipeline.threadExecutionWidth, 1)
        let simdgroups = max(1, min(32, downPipeline.maxTotalThreadsPerThreadgroup / simdWidth))
        let threads = simdgroups * simdWidth
        let gridX = (outputDimension + simdgroups - 1) / simdgroups
        let (routerBuffer, routerOffset) = context.resolveWeight("router")
        let (gateUpBuffer, gateUpOffset) = context.resolveWeight("expert_gate_up_proj")
        let (downBuffer, downOffset) = context.resolveWeight("expert_down_proj")
        let (biasBuffer, biasOffset) = useExpertBias
            ? context.resolveWeight("expert_bias")
            : (routerBuffer, routerOffset)

        let scratchRowStride = scratchElementsPerToken

        return FragmentPrefillSteps(
            steps: [
                MetalPrefillStep(
                    pipeline: routerPipeline,
                    gridSize: MTLSize(width: context.maximumSequenceLength, height: 1, depth: 1),
                    threadgroupSize: MTLSize(width: routerThreads, height: 1, depth: 1),
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
                        inputRowStride: context.slotDimension,
                        scratchRowStride: scratchRowStride,
                        startingIndex: 4
                    ),
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bind(index: 10),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: "\(kernelName)_router",
                        bufferAccessPattern: .init(reads: [0, 1, 2], writes: [3])
                    )
                ),
                MetalPrefillStep(
                    pipeline: gateUpPipeline,
                    gridSize: MTLSize(
                        width: (flatGateUpCount + gateUpThreads - 1) / gateUpThreads,
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
                        inputRowStride: context.slotDimension,
                        scratchRowStride: scratchRowStride,
                        startingIndex: 3
                    ),
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 6),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: "\(kernelName)_gate_up",
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
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 6),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: "\(kernelName)_down",
                    bufferAccessPattern: .init(reads: [0, 1], writes: [2])
                )
            )],
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
