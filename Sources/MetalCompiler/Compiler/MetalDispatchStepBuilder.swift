import Metal

struct MetalDispatchStepBuilder {
    func buildDecodePlan(
        fusedEntries: [DispatchEntry],
        unfusedCount: Int,
        bufferSet: MetalBufferSet,
        slotDimension: Int,
        stafWeightStore: STAFWeightStore?,
        hiddenSize: Int,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver,
        planBuildContext: PlanBuildContext,
        argumentEncoders: [String: MTLArgumentEncoder],
        resolveDispatch: (DispatchEntry) throws -> (
            name: String,
            pipeline: MTLComputePipelineState,
            config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
        )
    ) throws -> MetalDispatchPlan {
        let constantAllocator = MetalConstantBindingAllocator(device: planBuildContext.device)
        let argumentAllocator = MetalArgumentBindingAllocator()
        let preparedArgumentAllocator = MetalPreparedArgumentBufferAllocator(device: planBuildContext.device)

        var steps: [MetalDispatchStep] = []
        var quantizationEntries: [MetalQuantizationPlanEntry] = []
        var routingPlanner = DecodeRoutingPlanner(
            bufferSet: bufferSet,
            stafWeightStore: stafWeightStore,
            hiddenSize: hiddenSize,
            slotDimension: slotDimension,
            fallbackWeightFormat: planBuildContext.kernelContext.weightFormat,
            minimumFallbackLength: max(
                hiddenSize * hiddenSize,
                hiddenSize * slotDimension
            ) * planBuildContext.kernelContext.weightFormat.storageByteSize,
            accessPolicyResolver: accessPolicyResolver
        )

        var entryCursor = 0
        while entryCursor < fusedEntries.count {
            let entry = fusedEntries[entryCursor]
            routingPlanner.lastFragmentWriteBufferIndices = nil
            let bindings = routingPlanner.bindings(for: entry)
            if let fusedResidualRouterSteps = try Self.makeFusedResidualRMSRouterStepsIfEnabled(
                entry: entry,
                nextEntry: entryCursor + 1 < fusedEntries.count ? fusedEntries[entryCursor + 1] : nil,
                synthesizedBindings: bindings,
                routingPlanner: &routingPlanner,
                pipelineCache: planBuildContext.pipelineCache,
                kernelContext: planBuildContext.kernelContext,
                stafWeightStore: stafWeightStore,
                accessPolicyResolver: accessPolicyResolver
            ) {
                steps.append(contentsOf: fusedResidualRouterSteps)
                entryCursor += 2
                continue
            }
            if let sparseMoE = entry.fragment as? SparseMoEFragment,
               sparseMoE.usesSplitRoute {
                let weightFormat = KernelWeightFormatResolver(stafWeightStore: stafWeightStore)
                    .resolve(forFragment: sparseMoE, entry: entry)
                let sparseMoEKernelContext = KernelContext(
                    bufferPrecision: planBuildContext.kernelContext.bufferPrecision,
                    weightFormat: weightFormat
                )
                let splitSteps = try sparseMoE.splitDecodeSteps(
                    bindings: bindings,
                    pipelineCache: planBuildContext.pipelineCache,
                    kernelContext: sparseMoEKernelContext
                )
                steps.append(contentsOf: splitSteps.map { step in
                    MetalDispatchStep(
                        descriptor: step.descriptor,
                        bindings: step.bindings,
                        bufferAccesses: step.bufferAccesses,
                        metadata: MetalDispatchStepMetadata(
                            kernelName: step.metadata.kernelName,
                            entryIndex: entry.index,
                            layerIndex: entry.layerIndex,
                            weightTensorName: step.metadata.weightTensorName,
                            bufferAccessPattern: step.metadata.bufferAccessPattern
                        )
                    )
                })
                entryCursor += 1
                continue
            }
            let resolved = try resolveDispatch(entry)
            if let outputHeadSteps = try Self.makePartialArgmaxOutputHeadStepsIfEnabled(
                entry: entry,
                nextEntry: entryCursor + 1 < fusedEntries.count ? fusedEntries[entryCursor + 1] : nil,
                resolved: resolved,
                bindings: bindings,
                bufferSet: bufferSet,
                hiddenSize: hiddenSize,
                pipelineCache: planBuildContext.pipelineCache
            ) {
                Self.recordQuantizationEntries(
                    for: entry,
                    selectedKernelName: resolved.name,
                    stafWeightStore: stafWeightStore,
                    accessPolicyResolver: accessPolicyResolver,
                    fallbackSchemeIdentifier: planBuildContext.compileContext.weightFormat.schemeIdentifier,
                    into: &quantizationEntries
                )
                steps.append(contentsOf: outputHeadSteps)
                entryCursor += 2
                continue
            }
            let writeIndices = routingPlanner.lastFragmentWriteBufferIndices
            let bufferAccessPattern = Self.decodeBufferAccessPattern(
                for: entry,
                buffers: bindings.buffers,
                writeBufferIndices: writeIndices
            )
            let weightTensorName = Self.primaryWeightTensorName(for: entry)
            Self.recordQuantizationEntries(
                for: entry,
                selectedKernelName: resolved.name,
                stafWeightStore: stafWeightStore,
                accessPolicyResolver: accessPolicyResolver,
                fallbackSchemeIdentifier: planBuildContext.compileContext.weightFormat.schemeIdentifier,
                into: &quantizationEntries
            )
            steps.append(MetalDispatchStep(
                pipeline: resolved.pipeline,
                gridSize: resolved.config.grid,
                threadgroupSize: resolved.config.threadgroup,
                bufferBindings: bindings.buffers,
                bytesBindings: bindings.bytes,
                threadgroupMemoryLength: resolved.config.sharedMemoryBytes,
                sync: .bufferBarrier,
                bufferAccesses: Self.decodeBufferAccesses(
                    for: entry,
                    buffers: bindings.buffers,
                    writeBufferIndices: writeIndices),
                metadata: MetalDispatchStepMetadata(
                    kernelName: resolved.name,
                    entryIndex: entry.index,
                    layerIndex: entry.layerIndex,
                    weightTensorName: weightTensorName,
                    bufferAccessPattern: bufferAccessPattern
                )
            ))
            entryCursor += 1
        }

        let residentSteps = try makeResidentConstantSteps(steps, allocator: constantAllocator)
        let argumentTableSteps = makeArgumentTableSteps(residentSteps, allocator: argumentAllocator)
        let preparedArgumentSteps = try makePreparedArgumentTableSteps(
            argumentTableSteps,
            allocator: preparedArgumentAllocator
        )
        let encodedArgumentSteps = try makeEncodedArgumentTableSteps(
            preparedArgumentSteps,
            pipelineCache: planBuildContext.pipelineCache,
            argumentEncoders: argumentEncoders
        )
        let optimizedBarrierSteps = Self.optimizeDecodeBarrierPolicies(encodedArgumentSteps)
        let supplementalResidencyBuffers = Self.supplementalResidencyBuffers(in: optimizedBarrierSteps)

        return MetalDispatchPlan(
            steps: optimizedBarrierSteps,
            buffers: bufferSet,
            unfusedEntryCount: unfusedCount,
            fusedEntryCount: fusedEntries.count,
            quantizationPlan: MetalQuantizationPlan(
                capabilities: planBuildContext.quantizationCapabilities,
                entries: quantizationEntries
            ),
            supplementalResidencyBuffers: supplementalResidencyBuffers
        )
    }

    private static func makePartialArgmaxOutputHeadStepsIfEnabled(
        entry: DispatchEntry,
        nextEntry: DispatchEntry?,
        resolved: (
            name: String,
            pipeline: MTLComputePipelineState,
            config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
        ),
        bindings: (
            buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
            bytes: [(index: Int, value: [UInt8])]
        ),
        bufferSet: MetalBufferSet,
        hiddenSize: Int,
        pipelineCache: [String: MTLComputePipelineState]
    ) throws -> [MetalDispatchStep]? {
        guard ProcessInfo.processInfo.environment["SWIFTLM_OUTPUT_HEAD_PARTIAL_ARGMAX"] == "1",
              let linear = entry.fragment as? LinearFragment,
              linear.isOutput,
              linear.inputDimension == hiddenSize,
              linear.outputDimension > hiddenSize,
              nextEntry?.fragment is ArgmaxFragment else {
            return nil
        }

        let partialKernelName = "\(resolved.name)_argmax_partial"
        guard let partialPipeline = pipelineCache[partialKernelName],
              let reducePipeline = pipelineCache["argmax_partial_reduce"] else {
            throw MetalCompilerError.kernelNotFound(partialKernelName)
        }

        let simdWidth = max(partialPipeline.threadExecutionWidth, 1)
        let rowsPerThreadgroup = max(1, resolved.config.threadgroup.width / simdWidth)
        let partialCount = max(1, (linear.outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup)
        let partialValueBytes = partialCount * MemoryLayout<Float>.stride
        let partialIndexOffset = ((partialValueBytes + 255) / 256) * 256
        let partialIndexBytes = partialCount * MemoryLayout<Int32>.stride
        guard bufferSet.scratch.length >= partialIndexOffset + partialIndexBytes else {
            return nil
        }

        let inputBinding = try requiredBinding(index: 0, bindings: bindings.buffers, kernelName: partialKernelName)
        let weightBinding = try requiredBinding(index: 1, bindings: bindings.buffers, kernelName: partialKernelName)
        let outputBinding = try requiredBinding(index: 2, bindings: bindings.buffers, kernelName: partialKernelName)
        let weightTensorName = primaryWeightTensorName(for: entry)

        let reduceSimdWidth = max(reducePipeline.threadExecutionWidth, 1)
        let clampedReduceThreads = min(max(partialCount, 1), reducePipeline.maxTotalThreadsPerThreadgroup)
        let reduceThreads = max(
            reduceSimdWidth,
            ((clampedReduceThreads + reduceSimdWidth - 1) / reduceSimdWidth) * reduceSimdWidth
        )

        let partialStep = MetalDispatchStep(
            pipeline: partialPipeline,
            gridSize: resolved.config.grid,
            threadgroupSize: resolved.config.threadgroup,
            bufferBindings: [
                (0, inputBinding.buffer, inputBinding.offset),
                (1, weightBinding.buffer, weightBinding.offset),
                (2, outputBinding.buffer, outputBinding.offset),
                (3, bufferSet.scratch, 0),
                (4, bufferSet.scratch, partialIndexOffset),
            ],
            bytesBindings: [
                uint32Binding(5, UInt32(linear.inputDimension)),
                uint32Binding(6, UInt32(linear.outputDimension)),
            ],
            threadgroupMemoryLength: resolved.config.sharedMemoryBytes,
            sync: .bufferBarrier,
            bufferAccesses: MetalBufferAccesses(
                readBuffers: [
                    (buffer: inputBinding.buffer, offset: inputBinding.offset),
                    (buffer: weightBinding.buffer, offset: weightBinding.offset),
                ],
                writeBuffers: [
                    (buffer: outputBinding.buffer, offset: outputBinding.offset),
                    (buffer: bufferSet.scratch, offset: 0),
                    (buffer: bufferSet.scratch, offset: partialIndexOffset),
                ]
            ),
            metadata: MetalDispatchStepMetadata(
                kernelName: partialKernelName,
                entryIndex: entry.index,
                layerIndex: entry.layerIndex,
                weightTensorName: weightTensorName,
                bufferAccessPattern: .init(reads: [0, 1], writes: [2, 3, 4])
            )
        )

        let reduceStep = MetalDispatchStep(
            pipeline: reducePipeline,
            gridSize: MTLSize(width: 1, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: reduceThreads, height: 1, depth: 1),
            bufferBindings: [
                (0, bufferSet.scratch, 0),
                (1, bufferSet.scratch, partialIndexOffset),
                (2, bufferSet.tokenOut, 0),
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(partialCount)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            bufferAccesses: MetalBufferAccesses(
                readBuffers: [
                    (buffer: bufferSet.scratch, offset: 0),
                    (buffer: bufferSet.scratch, offset: partialIndexOffset),
                ],
                writeBuffers: [(buffer: bufferSet.tokenOut, offset: 0)]
            ),
            metadata: MetalDispatchStepMetadata(
                kernelName: "argmax_partial_reduce",
                entryIndex: nextEntry?.index,
                layerIndex: nextEntry?.layerIndex,
                bufferAccessPattern: .init(reads: [0, 1], writes: [2])
            )
        )

        return [partialStep, reduceStep]
    }

    private static func makeFusedResidualRMSRouterStepsIfEnabled(
        entry: DispatchEntry,
        nextEntry: DispatchEntry?,
        synthesizedBindings: (
            buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
            bytes: [(index: Int, value: [UInt8])]
        ),
        routingPlanner: inout DecodeRoutingPlanner,
        pipelineCache: [String: MTLComputePipelineState],
        kernelContext: KernelContext,
        stafWeightStore: STAFWeightStore?,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    ) throws -> [MetalDispatchStep]? {
        guard ProcessInfo.processInfo.environment["SWIFTLM_LFM25_FUSED_RMS_ROUTER"] == "1" else {
            return nil
        }
        let shouldTrace = ProcessInfo.processInfo.environment["SWIFTLM_LFM25_FUSED_RMS_ROUTER_TRACE"] == "1"
        func trace(_ message: String) {
            if shouldTrace {
                print("[FusedRMSRouter] \(message)")
            }
        }
        guard let synthesized = entry.fragment as? SynthesizedFragment else {
            return nil
        }
        guard let sparseMoEEntry = nextEntry,
              let sparseMoE = sparseMoEEntry.fragment as? SparseMoEFragment else {
            return nil
        }
        guard kernelContext.bufferPrecision == .bfloat16 else {
            trace("skip precision=\(kernelContext.bufferPrecision)")
            return nil
        }
        guard sparseMoE.usesSplitRoute else {
            trace("skip unsplit sparse moe")
            return nil
        }
        guard sparseMoE.inputDimension == 2048,
              sparseMoE.outputDimension == 2048,
              sparseMoE.expertCount == 32,
              sparseMoE.expertsPerToken == 4,
              sparseMoE.intermediateDimension == 1792 else {
            trace("skip shape input=\(sparseMoE.inputDimension) output=\(sparseMoE.outputDimension) experts=\(sparseMoE.expertCount) topK=\(sparseMoE.expertsPerToken) intermediate=\(sparseMoE.intermediateDimension)")
            return nil
        }
        let synthesizedKernelName = synthesized.kernelName(
            context: KernelContext(bufferPrecision: .float16, weightFormat: WeightFormats.bfloat16)
        )
        guard synthesizedKernelName == "synthesized_3way_residualadd_copy_reduction_4p_row2048_f16_wbf16" else {
            trace("skip synthesized=\(synthesizedKernelName)")
            return nil
        }
        guard let reduction = synthesized.fragments.compactMap({ $0 as? Reduction }).last,
              reduction.dimension == 2048,
              reduction.withScale else {
            trace("skip reduction")
            return nil
        }

        let sparseMoEWeightFormat = KernelWeightFormatResolver(stafWeightStore: stafWeightStore)
            .resolve(forFragment: sparseMoE, entry: sparseMoEEntry)
        guard sparseMoEWeightFormat.isBFloat16 else {
            trace("skip sparse weight=\(sparseMoEWeightFormat.schemeIdentifier)")
            return nil
        }
        trace("admit layer=\(entry.layerIndex.map(String.init) ?? "-")")

        let fusedKernelName = "residual_rms_router_parallel_bf16_sigmoid"
        guard let fusedPipeline = pipelineCache[fusedKernelName] else {
            throw MetalCompilerError.kernelNotFound(fusedKernelName)
        }
        let simdWidth = max(fusedPipeline.threadExecutionWidth, 1)
        let threads = sparseMoE.expertCount * simdWidth
        guard threads <= fusedPipeline.maxTotalThreadsPerThreadgroup else {
            return nil
        }

        let inputBinding = try requiredBinding(
            index: 0,
            bindings: synthesizedBindings.buffers,
            kernelName: fusedKernelName
        )
        let residualBinding = try requiredBinding(
            index: 1,
            bindings: synthesizedBindings.buffers,
            kernelName: fusedKernelName
        )
        let normWeightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            executionPhase: .decode,
            accessPolicyResolver: accessPolicyResolver
        )
        let normWeightBinding = normWeightResolver.resolve(role: reduction.weightRole)
        let nextBindings = routingPlanner.bindings(for: sparseMoEEntry)
        let sparseMoEKernelContext = KernelContext(
            bufferPrecision: kernelContext.bufferPrecision,
            weightFormat: sparseMoEWeightFormat
        )
        let splitSteps = try sparseMoE.splitDecodeSteps(
            bindings: nextBindings,
            pipelineCache: pipelineCache,
            kernelContext: sparseMoEKernelContext
        )
        guard splitSteps.count == 3,
              splitSteps[0].metadata.kernelName?.hasSuffix("_router_parallel") == true else {
            return nil
        }
        let routerBinding = try requiredBinding(index: 1, bindings: nextBindings.buffers, kernelName: fusedKernelName)
        let biasBinding = try requiredBinding(index: 4, bindings: nextBindings.buffers, kernelName: fusedKernelName)
        let hiddenBinding = try requiredBinding(index: 5, bindings: nextBindings.buffers, kernelName: fusedKernelName)
        let moeScratchBinding = try requiredBinding(index: 6, bindings: nextBindings.buffers, kernelName: fusedKernelName)

        let fusedStep = MetalDispatchStep(
            pipeline: fusedPipeline,
            gridSize: MTLSize(width: 1, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, inputBinding.buffer, inputBinding.offset),
                (1, residualBinding.buffer, residualBinding.offset),
                (2, normWeightBinding.0, normWeightBinding.1),
                (3, routerBinding.buffer, routerBinding.offset),
                (4, biasBinding.buffer, biasBinding.offset),
                (5, hiddenBinding.buffer, hiddenBinding.offset),
                (6, moeScratchBinding.buffer, moeScratchBinding.offset),
            ],
            bytesBindings: [
                uint32Binding(7, UInt32(reduction.dimension)),
                floatBinding(8, reduction.epsilon),
                floatBinding(9, reduction.weightBias),
                uint32Binding(10, UInt32(sparseMoE.inputDimension)),
                uint32Binding(11, UInt32(sparseMoE.expertCount)),
                uint32Binding(12, UInt32(sparseMoE.expertsPerToken)),
                uint32Binding(13, sparseMoE.normalizeRoutingWeights ? 1 : 0),
                floatBinding(14, sparseMoE.routedScalingFactor),
                uint32Binding(15, sparseMoE.useExpertBias ? 1 : 0),
                uint32Binding(16, UInt32(sparseMoE.scratchElementsPerToken)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            bufferAccesses: MetalBufferAccesses(
                readBuffers: [
                    (buffer: inputBinding.buffer, offset: inputBinding.offset),
                    (buffer: residualBinding.buffer, offset: residualBinding.offset),
                    (buffer: normWeightBinding.0, offset: normWeightBinding.1),
                    (buffer: routerBinding.buffer, offset: routerBinding.offset),
                    (buffer: biasBinding.buffer, offset: biasBinding.offset),
                ],
                writeBuffers: [
                    (buffer: residualBinding.buffer, offset: residualBinding.offset),
                    (buffer: hiddenBinding.buffer, offset: hiddenBinding.offset),
                    (buffer: moeScratchBinding.buffer, offset: moeScratchBinding.offset),
                ]
            ),
            metadata: MetalDispatchStepMetadata(
                kernelName: fusedKernelName,
                entryIndex: entry.index,
                layerIndex: entry.layerIndex,
                weightTensorName: primaryWeightTensorName(for: entry),
                bufferAccessPattern: .init(reads: [0, 1, 2, 3, 4], writes: [1, 5, 6])
            )
        )

        return [fusedStep] + splitSteps.dropFirst().map { step in
            MetalDispatchStep(
                descriptor: step.descriptor,
                bindings: step.bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: MetalDispatchStepMetadata(
                    kernelName: step.metadata.kernelName,
                    entryIndex: sparseMoEEntry.index,
                    layerIndex: sparseMoEEntry.layerIndex,
                    weightTensorName: step.metadata.weightTensorName,
                    bufferAccessPattern: step.metadata.bufferAccessPattern
                )
            )
        }
    }

    private static func requiredBinding(
        index: Int,
        bindings: [(index: Int, buffer: MTLBuffer, offset: Int)],
        kernelName: String
    ) throws -> (buffer: MTLBuffer, offset: Int) {
        guard let binding = bindings.first(where: { $0.index == index }) else {
            throw MetalCompilerError.deviceSetupFailed("\(kernelName) missing buffer binding \(index)")
        }
        return (binding.buffer, binding.offset)
    }

    private func makeResidentConstantSteps(
        _ steps: [MetalDispatchStep],
        allocator: MetalConstantBindingAllocator
    ) throws -> [MetalDispatchStep] {
        let bindingTables = steps.map(\.bindings)
        let residentBindings = try allocator.makeBindingTables(from: bindingTables)
        return zip(steps, residentBindings).map { step, bindings in
            MetalDispatchStep(
                descriptor: step.descriptor,
                bindings: bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: step.metadata
            )
        }
    }

    private func makeArgumentTableSteps(
        _ steps: [MetalDispatchStep],
        allocator: MetalArgumentBindingAllocator
    ) -> [MetalDispatchStep] {
        let bindingTables = steps.map(\.bindings)
        let plannedBindings = allocator.makeBindingTables(from: bindingTables)
        return zip(steps, plannedBindings).map { step, bindings in
            MetalDispatchStep(
                descriptor: step.descriptor,
                bindings: bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: step.metadata
            )
        }
    }

    private func makePreparedArgumentTableSteps(
        _ steps: [MetalDispatchStep],
        allocator: MetalPreparedArgumentBufferAllocator
    ) throws -> [MetalDispatchStep] {
        let bindingTables = steps.map(\.bindings)
        let preparedBindings = try allocator.makeBindingTables(from: bindingTables)
        return zip(steps, preparedBindings).map { step, bindings in
            MetalDispatchStep(
                descriptor: step.descriptor,
                bindings: bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: step.metadata
            )
        }
    }

    private func makeEncodedArgumentTableSteps(
        _ steps: [MetalDispatchStep],
        pipelineCache: [String: MTLComputePipelineState],
        argumentEncoders: [String: MTLArgumentEncoder]
    ) throws -> [MetalDispatchStep] {
        try steps.map { step in
            guard
                let kernelLabel = step.pipeline.label,
                let variantKernelName = Self.encodedArgumentTableKernelName(
                    for: kernelLabel,
                    bindings: step.bindings
                ),
                let variantPipeline = pipelineCache[variantKernelName],
                let argumentEncoder = argumentEncoders[variantKernelName],
                case .argumentTable(let table) = step.bindings.bufferBindings,
                case .prepared(_, let index, let offset) = table.encodingState
            else {
                return step
            }

            guard let encodedArgumentBuffer = variantPipeline.device.makeBuffer(
                length: argumentEncoder.encodedLength,
                options: .storageModeShared
            ) else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Cannot allocate encoded argument buffer for \(variantKernelName)"
                )
            }
            encodedArgumentBuffer.label =
                "swift-lm.argtable.encoded.\(variantKernelName).layout\(table.layout.id)"
            argumentEncoder.setArgumentBuffer(encodedArgumentBuffer, offset: 0)
            for binding in table.bindings {
                argumentEncoder.setBuffer(binding.buffer, offset: binding.offset, index: binding.index)
            }
            let encodedBindings = MetalBindingTable(
                bufferBindings: .argumentTable(MetalArgumentTableBindings(
                    layout: table.layout,
                    bindings: table.bindings,
                    encodingState: .encoded(
                        buffer: encodedArgumentBuffer,
                        index: index,
                        offset: offset
                    )
                )),
                constantBindings: Self.constantBindingsForEncodedVariant(
                    step.bindings.constantBindings,
                    variantKernelName: variantKernelName
                )
            )
            let encodedDescriptor = MetalDispatchDescriptor(
                pipeline: variantPipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: step.barrierPolicy
            )
            return MetalDispatchStep(
                descriptor: encodedDescriptor,
                bindings: encodedBindings,
                bufferAccesses: step.bufferAccesses,
                metadata: MetalDispatchStepMetadata(
                    kernelName: variantKernelName,
                    entryIndex: step.metadata.entryIndex,
                    layerIndex: step.metadata.layerIndex,
                    weightTensorName: step.metadata.weightTensorName,
                    bufferAccessPattern: step.metadata.bufferAccessPattern
                )
            )
        }
    }

    private static func primaryWeightTensorName(for entry: DispatchEntry) -> String? {
        if let linear = entry.fragment as? LinearFragment {
            return entry.parameterBindings.first(where: { $0.role == linear.field })?.tensorName
        }
        return nil
    }

    private static func constantBindingsForEncodedVariant(
        _ bindings: MetalConstantBindingSet,
        variantKernelName: String
    ) -> MetalConstantBindingSet {
        if (
            variantKernelName.hasPrefix("gemv_2048_sq") ||
            variantKernelName.hasPrefix("gemv_2048_6144")
        ) && variantKernelName.hasSuffix("_argbuf") {
            return .inline([])
        }
        return bindings
    }

    private static func supplementalResidencyBuffers(
        in steps: [MetalDispatchStep]
    ) -> [MTLBuffer] {
        var seen = Set<ObjectIdentifier>()
        var buffers: [MTLBuffer] = []
        for step in steps {
            for buffer in step.bindings.ownedResidencyBuffers {
                let identifier = ObjectIdentifier(buffer as AnyObject)
                guard seen.insert(identifier).inserted else { continue }
                buffers.append(buffer)
            }
        }
        return buffers
    }

    private static func decodeBufferAccesses(
        for entry: DispatchEntry,
        buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
        writeBufferIndices: Set<Int>? = nil
    ) -> MetalBufferAccesses {
        let mapped = buffers.map { MetalBufferBinding(index: $0.index, buffer: $0.buffer, offset: $0.offset) }

        func bindingTuple(_ index: Int) -> (buffer: MTLBuffer, offset: Int)? {
            mapped.first(where: { $0.index == index }).map { ($0.buffer, $0.offset) }
        }

        func bindingTuples(in indices: some Sequence<Int>) -> [(buffer: MTLBuffer, offset: Int)] {
            indices.compactMap(bindingTuple(_:))
        }

        if entry.fragment is LinearFragment {
            return MetalBufferAccesses(
                readBuffers: bindingTuples(in: [0, 1]),
                writeBuffers: bindingTuples(in: [2])
            )
        }
        if let writeBufferIndices {
            let allRegions = Set(mapped.map { BufferRegion(buffer: $0.buffer, offset: $0.offset) })
            let writeRegions = Set(
                buffers.filter { writeBufferIndices.contains($0.index) }
                    .map { BufferRegion(buffer: $0.buffer, offset: $0.offset) }
            )
            return MetalBufferAccesses(reads: allRegions, writes: writeRegions)
        }
        return MetalBufferAccesses.conservative(mapped)
    }

    private static func decodeBufferAccessPattern(
        for entry: DispatchEntry,
        buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
        writeBufferIndices: Set<Int>? = nil
    ) -> MetalDispatchStepMetadata.BufferAccessPattern? {
        if entry.fragment is LinearFragment {
            return .init(reads: [0, 1], writes: [2])
        }
        let bindingIndices = Set(buffers.map(\.index))
        guard let writeBufferIndices else {
            return nil
        }
        return .init(reads: bindingIndices, writes: writeBufferIndices)
    }

    private static func optimizeDecodeBarrierPolicies(
        _ steps: [MetalDispatchStep]
    ) -> [MetalDispatchStep] {
        var pendingReads = Set<BufferRegion>()
        var pendingWrites = Set<BufferRegion>()
        return steps.map { step in
            let requiresBarrier = step.bufferAccesses.requiresBarrier(
                after: pendingReads,
                pendingWrites: pendingWrites
            )
            let barrierPolicy: MetalBarrierPolicy
            if requiresBarrier {
                let visibility: MTL4VisibilityOptions =
                    MetalBufferAccesses.pendingWritesInvolveSharedBuffer(pendingWrites)
                    ? .device : []
                barrierPolicy = .barrier(visibility: visibility)
            } else {
                barrierPolicy = .none
            }
            let descriptor = MetalDispatchDescriptor(
                pipeline: step.pipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: barrierPolicy
            )

            if requiresBarrier {
                pendingReads = step.bufferAccesses.reads
                pendingWrites = step.bufferAccesses.writes
            } else {
                pendingReads.formUnion(step.bufferAccesses.reads)
                pendingWrites.formUnion(step.bufferAccesses.writes)
            }

            return MetalDispatchStep(
                descriptor: descriptor,
                bindings: step.bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: step.metadata
            )
        }
    }

    private static func recordQuantizationEntries(
        for entry: DispatchEntry,
        selectedKernelName: String,
        stafWeightStore: STAFWeightStore?,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver,
        fallbackSchemeIdentifier: QuantizationSchemeIdentifier,
        into entries: inout [MetalQuantizationPlanEntry]
    ) {
        let fragment = entry.fragment
        if let linear = fragment as? LinearFragment {
            let descriptor = resolveWeightDescriptor(
                role: linear.field,
                entry: entry,
                executionPhase: .decode,
                stafWeightStore: stafWeightStore,
                accessPolicyResolver: accessPolicyResolver,
                fallbackSchemeIdentifier: fallbackSchemeIdentifier
            )
            entries.append(
                MetalQuantizationPlanEntry(
                    entryIndex: entry.index,
                    layerIndex: entry.layerIndex,
                    tensorName: descriptor.tensorName,
                    path: .decodeProjection,
                    schemeIdentifier: descriptor.schemeIdentifier,
                    layout: descriptor.layout,
                    kernelFamily: .classify(kernelName: selectedKernelName, usesMPP: false),
                    usedFallback: descriptor.usedFallback,
                    fallbackReason: descriptor.fallbackReason,
                    prefillGEMM: nil
                )
            )
        } else if let batched = fragment as? BatchedProjection {
            for projection in batched.projections {
                let descriptor = resolveWeightDescriptor(
                    role: projection.field,
                    entry: entry,
                    executionPhase: .decode,
                    stafWeightStore: stafWeightStore,
                    accessPolicyResolver: accessPolicyResolver,
                    fallbackSchemeIdentifier: fallbackSchemeIdentifier
                )
                entries.append(
                    MetalQuantizationPlanEntry(
                        entryIndex: entry.index,
                        layerIndex: entry.layerIndex,
                        tensorName: descriptor.tensorName,
                        path: .decodeProjection,
                        schemeIdentifier: descriptor.schemeIdentifier,
                        layout: descriptor.layout,
                        kernelFamily: .classify(kernelName: selectedKernelName, usesMPP: false),
                        usedFallback: descriptor.usedFallback,
                        fallbackReason: descriptor.fallbackReason,
                        prefillGEMM: nil
                    )
                )
            }
        } else if fragment is GatherFragment {
            let descriptor = resolveWeightDescriptor(
                role: "embedding_table",
                entry: entry,
                executionPhase: .decode,
                stafWeightStore: stafWeightStore,
                accessPolicyResolver: accessPolicyResolver,
                fallbackSchemeIdentifier: fallbackSchemeIdentifier
            )
            entries.append(
                MetalQuantizationPlanEntry(
                    entryIndex: entry.index,
                    layerIndex: entry.layerIndex,
                    tensorName: descriptor.tensorName,
                    path: .embeddingLookup,
                    schemeIdentifier: descriptor.schemeIdentifier,
                    layout: descriptor.layout,
                    kernelFamily: .classify(kernelName: selectedKernelName, usesMPP: false),
                    usedFallback: descriptor.usedFallback,
                    fallbackReason: descriptor.fallbackReason,
                    prefillGEMM: nil
                )
            )
        }
    }

    private static func resolveWeightDescriptor(
        role: String,
        entry: DispatchEntry,
        executionPhase: STAFWeightExecutionPhase,
        stafWeightStore: STAFWeightStore?,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver,
        fallbackSchemeIdentifier: QuantizationSchemeIdentifier
    ) -> DecodeWeightDescriptor {
        guard let binding = entry.parameterBindings.first(where: { $0.role == role }) else {
            return DecodeWeightDescriptor(
                tensorName: nil,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: .rowMajor,
                usedFallback: true,
                fallbackReason: .missingTensorBinding
            )
        }
        guard let stafWeightStore else {
            return DecodeWeightDescriptor(
                tensorName: binding.tensorName,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: .rowMajor,
                usedFallback: true,
                fallbackReason: .missingWeightStore
            )
        }

        let request = accessPolicyResolver.accessRequest(
            for: entry,
            role: role,
            binding: binding,
            executionPhase: executionPhase,
            stafWeightStore: stafWeightStore
        )
        let layout = stafWeightStore.resolvedBufferAccess(for: request)?.layout ?? request.preferredLayout
        guard let tensorEntry = stafWeightStore.entries[binding.tensorName] else {
            return DecodeWeightDescriptor(
                tensorName: binding.tensorName,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: layout,
                usedFallback: true,
                fallbackReason: .missingTensorMetadata
            )
        }
        return DecodeWeightDescriptor(
            tensorName: binding.tensorName,
            schemeIdentifier: tensorEntry.schemeIdentifier,
            layout: layout,
            usedFallback: false,
            fallbackReason: nil
        )
    }

    private static func encodedArgumentTableKernelName(
        for kernelName: String,
        bindings: MetalBindingTable
    ) -> String? {
        guard case .argumentTable(let table) = bindings.bufferBindings else {
            return nil
        }
        switch table.layout.indices {
        case [0, 1]:
            switch kernelName {
            case "argmax":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case "residual_add_inplace":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case "rms_norm", "rms_norm_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case "qk_rms_norm", "qk_rms_norm_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            default:
                return nil
            }
        case [0, 1, 2]:
            switch kernelName {
            case "embedding_lookup", "embedding_lookup_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            default:
                if kernelName.hasPrefix("gemv_2048_sq")
                    || kernelName.hasPrefix("gemv_2048_6144")
                    || kernelName == "gemv_8192_tiled"
                    || kernelName == "gemv_8192_tiled_bf16"
                    || kernelName == "gemv"
                    || kernelName == "gemv_bf16"
                    || kernelName == "gemv_vocab"
                    || kernelName == "gemv_vocab_bf16"
                {
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                }
                switch kernelName {
                case "residual_add":
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                case "rope":
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                default:
                    return nil
                }
            }
        case [0, 1, 2, 3]:
            switch kernelName {
            case let name where name.hasPrefix("fused_copy_rms_norm"):
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case let name where name.hasPrefix("fused_residual_add_copy_rms_norm"):
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case let name where name.hasPrefix("fused_residual_add_rms_norm"):
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case let name where name.hasSuffix("glu_projection_2048") || name.hasSuffix("glu_projection_2048_bf16"):
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case "conv_state_update", "conv_state_update_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case "batched_qk_rms_norm_2", "batched_qk_rms_norm_bf16_2":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            default:
                return nil
            }
        case [0, 1, 2, 3, 4]:
            switch kernelName {
            case "batched_gemv2", "batched_gemv2_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            default:
                return nil
            }
        default:
            if table.layout.indices == [0, 1, 2, 3, 4, 5, 6, 17, 18, 19] {
                switch kernelName {
                case "flash_attn_decode":
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                case "batched_gemv3", "batched_gemv3_bf16":
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                default:
                    return nil
                }
            }
            if table.layout.indices == [0, 1, 2, 3, 4, 5, 6, 7, 8] {
                switch kernelName {
                case "batched_gemv4", "batched_gemv4_bf16":
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                default:
                    return nil
                }
            }
            return nil
        }
    }
}

private struct DecodeWeightDescriptor {
    let tensorName: String?
    let schemeIdentifier: QuantizationSchemeIdentifier
    let layout: STAFWeightLayout
    let usedFallback: Bool
    let fallbackReason: MetalQuantizationFallbackReason?
}

struct DecodeRoutingPlanner {
    let bufferSet: MetalBufferSet
    let stafWeightStore: STAFWeightStore?
    let hiddenSize: Int
    let slotDimension: Int
    let fallbackWeightFormat: WeightFormat
    let minimumFallbackLength: Int
    let accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    private let elementSize: Int
    private var kvCacheIndex: Int = 0
    private var routingState = BufferRoutingState()
    private var activeCompositeID: Int?
    private var compositeInputSource: (buffer: MTLBuffer, offset: Int)?

    /// Write buffer indices from the most recent fragment binding.
    /// Set by `bindings(for:)` when entry is .fragment.
    /// nil for non-fragment entries or when fragment does not declare write indices.
    var lastFragmentWriteBufferIndices: Set<Int>?

    init(
        bufferSet: MetalBufferSet,
        stafWeightStore: STAFWeightStore?,
        hiddenSize: Int,
        slotDimension: Int,
        fallbackWeightFormat: WeightFormat,
        minimumFallbackLength: Int,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    ) {
        self.bufferSet = bufferSet
        self.stafWeightStore = stafWeightStore
        self.hiddenSize = hiddenSize
        self.slotDimension = slotDimension
        self.fallbackWeightFormat = fallbackWeightFormat
        self.minimumFallbackLength = minimumFallbackLength
        self.accessPolicyResolver = accessPolicyResolver
        self.elementSize = bufferSet.bufferPrecision.byteSize
    }

    mutating func bindings(
        for entry: DispatchEntry
    ) -> (
        buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
        bytes: [(index: Int, value: [UInt8])]
    ) {
        updateCompositeInputSource(for: entry)

        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            executionPhase: .decode,
            accessPolicyResolver: accessPolicyResolver
        )

        if let linear = entry.fragment as? LinearFragment {
            let projection = linear
            let isOutput = linear.isOutput
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)

            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if !isOutput, let compositeInputSource {
                inputBuffer = compositeInputSource.buffer
                inputOffset = compositeInputSource.offset
            } else if routingState.lastOutputIsHidden {
                inputBuffer = bufferSet.hidden
                inputOffset = 0
            } else {
                inputBuffer = bufferSet.scratch
                inputOffset = routingState.currentInputOffset
            }

            let outputBuffer: MTLBuffer
            let outputOffset: Int

            if isOutput && projection.outputDimension > hiddenSize {
                outputBuffer = bufferSet.logits
                outputOffset = 0
                routingState.lastOutputIsHidden = false
            } else if isOutput {
                outputBuffer = bufferSet.hidden
                outputOffset = 0
                routingState.lastOutputIsHidden = true
            } else {
                let scratchSlot = routingState.projectionIndex + 1
                outputBuffer = bufferSet.scratch
                outputOffset = scratchSlot * slotDimension * elementSize
                routingState.lastOutputIsHidden = false
                routingState.currentInputOffset = outputOffset
            }

            routingState.projectionIndex += 1

            return (
                buffers: [
                    (0, inputBuffer, inputOffset),
                    (1, weightBuffer, weightOffset),
                    (2, outputBuffer, outputOffset),
                ],
                bytes: [
                    uint32Binding(3, UInt32(projection.inputDimension)),
                    uint32Binding(4, UInt32(projection.outputDimension)),
                ]
            )
        } else {
            let fragment = entry.fragment
            let resolvedKVCacheIndex = fragment.kvCacheIndexOverride ?? kvCacheIndex
            let currentInputBuffer: MTLBuffer
            let currentInputOffset: Int
            if routingState.lastOutputIsHidden {
                currentInputBuffer = bufferSet.hidden
                currentInputOffset = 0
            } else {
                currentInputBuffer = bufferSet.scratch
                currentInputOffset = routingState.currentInputOffset
            }
            let bindingContext = BufferBindingContext(
                bufferSet: bufferSet,
                slotDimension: slotDimension,
                elementSize: elementSize,
                currentInputBuffer: currentInputBuffer,
                currentInputOffset: currentInputOffset,
                layerIndex: entry.layerIndex,
                kvCacheIndex: resolvedKVCacheIndex,
                convLayerIndex: routingState.convLayerIndex,
                recurrentLayerIndex: routingState.recurrentLayerIndex,
                projectionIndex: routingState.projectionIndex,
                resolveWeight: weightResolver.resolve
            )
            let bindings = fragment.decodeBindings(context: bindingContext)
            if bindings.resetsProjectionIndex {
                routingState.projectionIndex = 0
                if !bindings.outputIsHidden {
                    routingState.currentInputOffset = 0
                }
            }
            if bindings.consumesKVCacheLayer { kvCacheIndex += 1 }
            if bindings.consumesConvLayer { routingState.convLayerIndex += 1 }
            if bindings.consumesRecurrentLayer { routingState.recurrentLayerIndex += 1 }
            routingState.lastOutputIsHidden = bindings.outputIsHidden
            // Advance projection index for projection-type fragments
            if bindings.projectionSlotsConsumed > 0 {
                routingState.projectionIndex += bindings.projectionSlotsConsumed
                routingState.currentInputOffset = routingState.projectionIndex * slotDimension * elementSize
            }
            if bindings.resetsProjectionIndex {
                refreshCompositeInputSource()
            }
            lastFragmentWriteBufferIndices = bindings.writeBufferIndices
            return (buffers: bindings.buffers, bytes: bindings.bytes)
        }
    }

    private mutating func updateCompositeInputSource(for entry: DispatchEntry) {
        guard activeCompositeID != entry.compositeID else { return }
        activeCompositeID = entry.compositeID
        refreshCompositeInputSource()
    }

    private mutating func refreshCompositeInputSource() {
        if routingState.lastOutputIsHidden {
            compositeInputSource = (bufferSet.hidden, 0)
        } else {
            compositeInputSource = (bufferSet.scratch, routingState.currentInputOffset)
        }
    }
}
