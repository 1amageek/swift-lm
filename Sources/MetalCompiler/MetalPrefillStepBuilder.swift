import Metal

struct MetalPrefillStepBuilder {
    func buildPrefillPlan(
        fusedEntries: [DispatchEntry],
        buffers: PrefillBufferSet,
        slotDimension: Int,
        maximumSequenceLength: Int,
        hiddenSize: Int,
        scratchElementSize: Int,
        usesMPP: Bool,
        planBuildContext: PlanBuildContext,
        resolveDispatch: @escaping (DispatchEntry) throws -> (
            name: String,
            pipeline: MTLComputePipelineState,
            config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
        )
    ) throws -> MetalPrefillPlan {
        let constantAllocator = MetalConstantBindingAllocator(device: planBuildContext.device)
        var steps: [MetalPrefillStep] = []
        var planner = PrefillStepPlanner(
            buffers: buffers,
            stafWeightStore: planBuildContext.stafWeightStore,
            hiddenSize: hiddenSize,
            slotDimension: slotDimension,
            maximumSequenceLength: maximumSequenceLength,
            scratchElementSize: scratchElementSize,
            usesMPP: usesMPP,
            planBuildContext: planBuildContext,
            resolveDispatch: resolveDispatch
        )

        for entry in fusedEntries {
            let prefillSteps = try planner.buildSteps(for: entry)
            steps.append(contentsOf: prefillSteps)
        }

        let residentSteps = try makeResidentConstantSteps(steps, allocator: constantAllocator)
        let optimizedSteps = Self.optimizePrefillBarrierPolicies(residentSteps)
        let supplementalResidencyBuffers = Self.supplementalResidencyBuffers(in: optimizedSteps)
        let finalHiddenSource = planner.finalHiddenSource()
        return MetalPrefillPlan(
            steps: optimizedSteps,
            buffers: buffers,
            slotDimension: slotDimension,
            maximumSequenceLength: maximumSequenceLength,
            stepCount: optimizedSteps.count,
            usesMPP: usesMPP,
            quantizationPlan: planner.makeQuantizationPlan(),
            finalHiddenBuffer: finalHiddenSource.buffer,
            finalHiddenBaseOffset: finalHiddenSource.offset,
            finalHiddenRowStride: finalHiddenSource.rowStride,
            supplementalResidencyBuffers: supplementalResidencyBuffers
        )
    }

    /// Offset-aware buffer region for precise hazard detection.
    /// Distinguishes scratch[0] from scratch[1] on the same MTLBuffer.
    /// Eliminate unnecessary memory barriers between prefill steps using
    /// offset-aware buffer region tracking.
    ///
    /// Each step's `metadata.bufferAccessPattern` declares which binding indices are
    /// reads vs writes. Steps without a declared pattern are treated conservatively
    /// (all bindings as both read and written).
    static func optimizePrefillBarrierPolicies(
        _ steps: [MetalPrefillStep]
    ) -> [MetalPrefillStep] {
        var pendingReads = Set<BufferRegion>()
        var pendingWrites = Set<BufferRegion>()
        return steps.map { step in
            let accesses = resolveBufferRegions(for: step)
            if step.mode == .lastToken {
                pendingReads = accesses.reads
                pendingWrites = accesses.writes
                return step
            }
            let requiresBarrier = accesses.requiresBarrier(
                after: pendingReads,
                pendingWrites: pendingWrites
            )
            let newBarrierPolicy: MetalBarrierPolicy
            if requiresBarrier {
                let resources = accesses.conflictingResources(
                    from: pendingReads,
                    pendingWrites: pendingWrites
                )
                newBarrierPolicy = resources.isEmpty ? .bufferBarrier : .resourceBarrier(resources: resources)
            } else {
                newBarrierPolicy = .none
            }

            if requiresBarrier {
                pendingReads = accesses.reads
                pendingWrites = accesses.writes
            } else {
                pendingReads.formUnion(accesses.reads)
                pendingWrites.formUnion(accesses.writes)
            }

            guard newBarrierPolicy != step.barrierPolicy else { return step }

            let descriptor = MetalDispatchDescriptor(
                pipeline: step.pipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: newBarrierPolicy
            )
            return MetalPrefillStep(
                descriptor: descriptor,
                bindings: step.bindings,
                mode: step.mode,
                sequenceLengthPolicy: step.sequenceLengthPolicy,
                positionBufferIndex: step.positionBufferIndex,
                perPositionStrides: step.perPositionStrides,
                metadata: step.metadata
            )
        }
    }

    private static func supplementalResidencyBuffers(
        in steps: [MetalPrefillStep]
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

    /// Convert a step's declared buffer access pattern into concrete buffer regions.
    /// Falls back to treating all bindings as read+written when no pattern is declared.
    private static func resolveBufferRegions(
        for step: MetalPrefillStep
    ) -> MetalBufferAccesses {
        let buffers = step.bindings.buffers

        func regions(for indices: Set<Int>) -> Set<BufferRegion> {
            Set(buffers.filter { indices.contains($0.index) }
                .map { BufferRegion(buffer: $0.buffer, offset: $0.offset) })
        }

        if let pattern = step.metadata.bufferAccessPattern {
            return MetalBufferAccesses(
                reads: regions(for: pattern.readIndices),
                writes: regions(for: pattern.writeIndices))
        }

        // Conservative fallback: treat all bindings as both read and written.
        return MetalBufferAccesses.conservative(buffers)
    }

    private func makeResidentConstantSteps(
        _ steps: [MetalPrefillStep],
        allocator: MetalConstantBindingAllocator
    ) throws -> [MetalPrefillStep] {
        let bindingTables = steps.map(\.bindings)
        let residentBindings = try allocator.makeBindingTables(from: bindingTables)
        return zip(steps, residentBindings).map { step, bindings in
            MetalPrefillStep(
                descriptor: step.descriptor,
                bindings: bindings,
                mode: step.mode,
                sequenceLengthPolicy: step.sequenceLengthPolicy,
                positionBufferIndex: step.positionBufferIndex,
                perPositionStrides: step.perPositionStrides,
                metadata: step.metadata
            )
        }
    }
}

private struct PrefillStepPlanner {
    let buffers: PrefillBufferSet
    let stafWeightStore: STAFWeightStore?
    let hiddenSize: Int
    let slotDimension: Int
    let maximumSequenceLength: Int
    let scratchElementSize: Int
    let usesMPP: Bool
    let planBuildContext: PlanBuildContext
    let fallbackWeightFormat: WeightFormat
    let minimumFallbackLength: Int
    let resolveDispatch: (DispatchEntry) throws -> (
        name: String,
        pipeline: MTLComputePipelineState,
        config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
    )
    var kvCacheIndex: Int = 0
    var routingState = BufferRoutingState()
    var outputHeadInputSource: (buffer: MTLBuffer, offset: Int, rowStride: Int)?
    var activeCompositeID: Int?
    var compositeInputSource: (buffer: MTLBuffer, offset: Int)?
    var quantizationEntries: [MetalQuantizationPlanEntry] = []

    init(
        buffers: PrefillBufferSet,
        stafWeightStore: STAFWeightStore?,
        hiddenSize: Int,
        slotDimension: Int,
        maximumSequenceLength: Int,
        scratchElementSize: Int,
        usesMPP: Bool,
        planBuildContext: PlanBuildContext,
        resolveDispatch: @escaping (DispatchEntry) throws -> (
            name: String,
            pipeline: MTLComputePipelineState,
            config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
        )
    ) {
        self.buffers = buffers
        self.stafWeightStore = stafWeightStore
        self.hiddenSize = hiddenSize
        self.slotDimension = slotDimension
        self.maximumSequenceLength = maximumSequenceLength
        self.scratchElementSize = scratchElementSize
        self.usesMPP = usesMPP
        self.planBuildContext = planBuildContext
        self.fallbackWeightFormat = planBuildContext.kernelContext.weightFormat
        self.minimumFallbackLength = max(
            hiddenSize * hiddenSize,
            hiddenSize * slotDimension
        ) * planBuildContext.kernelContext.weightFormat.storageByteSize
        self.resolveDispatch = resolveDispatch
    }

    private func annotate(
        _ steps: [MetalPrefillStep],
        entryIndex: Int,
        layerIndex: Int?
    ) -> [MetalPrefillStep] {
        steps.map { step in
            MetalPrefillStep(
                descriptor: step.descriptor,
                bindings: step.bindings,
                mode: step.mode,
                sequenceLengthPolicy: step.sequenceLengthPolicy,
                positionBufferIndex: step.positionBufferIndex,
                perPositionStrides: step.perPositionStrides,
                metadata: MetalDispatchStepMetadata(
                    kernelName: step.metadata.kernelName,
                    entryIndex: entryIndex,
                    layerIndex: layerIndex,
                    weightTensorName: step.metadata.weightTensorName,
                    bufferAccessPattern: step.metadata.bufferAccessPattern
                )
            )
        }
    }

    private func fragmentKernelContext(
        for fragment: any PrimitiveMetalKernelFragment,
        entry: DispatchEntry
    ) -> KernelContext {
        let weightFormatResolver = KernelWeightFormatResolver(stafWeightStore: stafWeightStore)
        return KernelContext(
            bufferPrecision: planBuildContext.kernelContext.bufferPrecision,
            weightFormat: weightFormatResolver.resolve(forFragment: fragment, entry: entry)
        )
    }

    mutating func buildSteps(for entry: DispatchEntry) throws -> [MetalPrefillStep] {
        updateCompositeInputSource(for: entry)

        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            fallbackBuffer: buffers.hidden,
            fallbackWeightFormat: fallbackWeightFormat,
            minimumFallbackLength: minimumFallbackLength,
            logsMisses: false,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )

        switch entry.kind {
        case .fragment(let frag):
            let pipelineCache = planBuildContext.pipelineCache
            let kernelContext = fragmentKernelContext(for: frag, entry: entry)
            let resolvedKVCacheIndex = frag.kvCacheIndexOverride ?? kvCacheIndex
            let currentInputBuffer: MTLBuffer
            let currentInputOffset: Int
            if routingState.lastOutputIsHidden {
                currentInputBuffer = buffers.hidden
                currentInputOffset = 0
            } else {
                currentInputBuffer = buffers.scratch
                currentInputOffset = routingState.currentInputOffset
            }
            let prefillContext = PrefillBindingContext(
                buffers: buffers,
                slotDimension: slotDimension,
                scratchElementSize: scratchElementSize,
                maximumSequenceLength: maximumSequenceLength,
                currentInputBuffer: currentInputBuffer,
                currentInputOffset: currentInputOffset,
                layerIndex: entry.layerIndex,
                kvCacheIndex: resolvedKVCacheIndex,
                convLayerIndex: routingState.convLayerIndex,
                recurrentLayerIndex: routingState.recurrentLayerIndex,
                kernelContext: kernelContext,
                resolveWeight: weightResolver.resolve,
                getPipeline: { name in
                    guard let pipeline = pipelineCache[name] else {
                        let relatedKernelNames = pipelineCache.keys
                            .filter {
                                $0.contains("embedding_lookup")
                                    || $0.contains("rms_norm_seq")
                                    || $0.contains("qk_rms_norm_seq")
                            }
                            .sorted()
                        if !relatedKernelNames.isEmpty {
                            print("[Compiler] missing prefill kernel '\(name)'; related compiled kernels: \(relatedKernelNames)")
                        }
                        throw MetalCompilerError.kernelNotFound(name)
                    }
                    return pipeline
                }
            )
            if let reduction = frag as? Reduction,
               shouldCaptureResidualInput(for: reduction.weightRole),
               currentInputBuffer === buffers.hidden,
               currentInputOffset == 0
            {
                var steps: [MetalPrefillStep] = []
                steps.append(try makeHiddenToResidualCopyStep(
                    dimension: reduction.dimension,
                    entry: entry
                ))
                steps.append(contentsOf: try buildNormToHiddenStep(
                    inputBuffer: buffers.residual,
                    inputOffset: 0,
                    dimension: reduction.dimension,
                    epsilon: reduction.epsilon,
                    weightRole: reduction.weightRole,
                    weightBias: reduction.weightBias,
                    entry: entry
                ))
                routingState.projectionIndex = 0
                routingState.lastOutputIsHidden = true
                routingState.currentInputOffset = 0
                refreshCompositeInputSource()
                return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)
            }
            let result = try frag.prefillSteps(context: prefillContext)
            if frag is GatherFragment, let selectedKernelName = result.steps.first?.pipeline.label {
                let descriptor = resolveProjectionWeightDescriptor(role: "embedding_table", entry: entry)
                quantizationEntries.append(
                    MetalQuantizationPlanEntry(
                        entryIndex: entry.index,
                        layerIndex: entry.layerIndex,
                        tensorName: descriptor.tensorName,
                        path: .embeddingLookup,
                        schemeIdentifier: descriptor.schemeIdentifier,
                        layout: descriptor.layout,
                        kernelFamily: .classify(
                            kernelName: selectedKernelName,
                            usesMPP: false
                        ),
                        usedFallback: descriptor.usedFallback,
                        fallbackReason: descriptor.fallbackReason
                    )
                )
            }
            if result.resetsProjectionIndex {
                routingState.projectionIndex = 0
                if !result.outputIsHidden {
                    routingState.currentInputOffset = 0
                }
            }
            if result.consumesKVCacheLayer { kvCacheIndex += 1 }
            if result.consumesConvLayer { routingState.convLayerIndex += 1 }
            if result.consumesRecurrentLayer { routingState.recurrentLayerIndex += 1 }
            routingState.lastOutputIsHidden = result.outputIsHidden
            if result.resetsProjectionIndex {
                refreshCompositeInputSource()
            }
            return annotate(result.steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .batchedProjection(let batched):
            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if routingState.lastOutputIsHidden {
                inputBuffer = buffers.hidden
                inputOffset = 0
            } else {
                inputBuffer = buffers.scratch
                inputOffset = routingState.currentInputOffset
            }
            let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength
            var steps: [MetalPrefillStep] = []
            steps.reserveCapacity(batched.projections.count)
            var lastOutputOffset = routingState.currentInputOffset
            for projection in batched.projections {
                let inputRowStride = inputBuffer === buffers.hidden
                    ? (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize
                    : projection.inputDimension
                let resolved = try resolveDispatch(
                    DispatchEntry(
                        index: entry.index,
                        kind: .projection(
                            MetalProjection(
                                field: projection.field,
                                inputDimension: projection.inputDimension,
                                outputDimension: projection.outputDimension
                            ),
                            isOutput: false
                        ),
                        parameterBindings: entry.parameterBindings,
                        layerIndex: entry.layerIndex
                    )
                )
                let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
                let weightTensorName = entry.parameterBindings.first(where: { $0.role == projection.field })?.tensorName
                let quantizationDescriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)
                let outputOffset = (routingState.projectionIndex + 1) * scratchSlotSize
                lastOutputOffset = outputOffset
                routingState.projectionIndex += 1

                // Q4 with dequant scratch → dequant to BF16, then AMX matmul2d
                let canDequantForAMX = quantizationDescriptor.schemeIdentifier.isWeightQuantized
                    && buffers.dequantScratch != nil
                    && dequantKernelName(for: quantizationDescriptor.schemeIdentifier) != nil
                let usesMPPForStep = usesMPP
                    && inputRowStride == projection.inputDimension
                    && (!quantizationDescriptor.schemeIdentifier.isWeightQuantized || canDequantForAMX)

                // Emit dequant step: Q4 weight → BF16 dequant scratch
                if canDequantForAMX && usesMPPForStep,
                   let dequantName = dequantKernelName(for: quantizationDescriptor.schemeIdentifier),
                   let dequantPipeline = planBuildContext.pipelineCache[dequantName],
                   let dequantScratch = buffers.dequantScratch {
                    steps.append(
                        MetalPrefillStep(
                            pipeline: dequantPipeline,
                            gridSize: MTLSize(width: projection.outputDimension, height: 1, depth: 1),
                            threadgroupSize: MTLSize(width: 256, height: 1, depth: 1),
                            bufferBindings: [
                                (0, weightBuffer, weightOffset),
                                (1, dequantScratch, 0),
                            ],
                            bytesBindings: [
                                uint32Binding(2, UInt32(projection.inputDimension)),
                                uint32Binding(3, UInt32(projection.outputDimension)),
                            ],
                            threadgroupMemoryLength: 0,
                            sync: .bufferBarrier,
                            mode: .batch,
                            sequenceLengthPolicy: .none,
                            positionBufferIndex: nil,
                            perPositionStrides: [:],
                            metadata: .init(
                                kernelName: dequantName,
                                entryIndex: entry.index,
                                weightTensorName: weightTensorName,
                                bufferAccessPattern: .init(reads: [0], writes: [1])
                            )
                        )
                    )
                }

                // Resolve GEMM pipeline
                let selectedPipeline: MTLComputePipelineState
                let selectedKernelName: String
                if canDequantForAMX && usesMPPForStep,
                   let mppPipeline = planBuildContext.pipelineCache["gemm_bf16_f32s"] {
                    // Dequant path: Q4 unpacked to BF16, use BF16 MPP GEMM
                    selectedPipeline = mppPipeline
                    selectedKernelName = "gemm_bf16_f32s"
                } else if !usesMPPForStep,
                   let naivePipeline = planBuildContext.pipelineCache["naive::\(resolved.name)"] {
                    selectedPipeline = naivePipeline
                    selectedKernelName = "naive::\(resolved.name)"
                } else {
                    selectedPipeline = resolved.pipeline
                    selectedKernelName = resolved.name
                }

                // GEMM weight source: dequant scratch (BF16) or original weight buffer
                let gemmWeightBuffer: MTLBuffer
                let gemmWeightOffset: Int
                if canDequantForAMX && usesMPPForStep, let dequantScratch = buffers.dequantScratch {
                    gemmWeightBuffer = dequantScratch
                    gemmWeightOffset = 0
                } else {
                    gemmWeightBuffer = weightBuffer
                    gemmWeightOffset = weightOffset
                }

                let gridSize: MTLSize
                let threadgroupSize: MTLSize
                if usesMPPForStep {
                    let simdWidth = selectedPipeline.threadExecutionWidth
                    gridSize = MTLSize(
                        width: (projection.outputDimension + 31) / 32,
                        height: (maximumSequenceLength + 63) / 64,
                        depth: 1
                    )
                    threadgroupSize = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
                } else {
                    let simdWidth = max(selectedPipeline.threadExecutionWidth, 1)
                    let rowsPerThreadgroup = 2
                    let threads = min(
                        simdWidth * rowsPerThreadgroup,
                        selectedPipeline.maxTotalThreadsPerThreadgroup
                    )
                    gridSize = MTLSize(
                        width: (projection.outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                        height: maximumSequenceLength,
                        depth: 1
                    )
                    threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)
                }

                let gemmPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
                recordProjectionQuantization(
                    entry: entry,
                    descriptor: quantizationDescriptor,
                    mode: .batch,
                    inputRowStride: inputRowStride,
                    inputDimension: projection.inputDimension,
                    selectedKernelName: selectedKernelName,
                    usesMPPForStep: usesMPPForStep
                )
                steps.append(
                    MetalPrefillStep(
                        pipeline: selectedPipeline,
                        gridSize: gridSize,
                        threadgroupSize: threadgroupSize,
                        bufferBindings: [
                            (0, inputBuffer, inputOffset),
                            (1, gemmWeightBuffer, gemmWeightOffset),
                            (2, buffers.scratch, outputOffset),
                        ],
                        bytesBindings: [
                            uint32Binding(3, UInt32(projection.inputDimension)),
                            uint32Binding(4, UInt32(projection.outputDimension)),
                            uint32Binding(5, UInt32(maximumSequenceLength)),
                            uint32Binding(6, UInt32(inputRowStride)),
                        ],
                        threadgroupMemoryLength: usesMPPForStep ? 0 : resolved.config.sharedMemoryBytes,
                        sync: .bufferBarrier,
                        mode: .batch,
                        sequenceLengthPolicy: usesMPPForStep
                            ? .bindAndAdjustGridHeightTiled(index: 5, tileHeight: 64)
                            : .bindAndAdjustGridHeight(index: 5),
                        positionBufferIndex: nil,
                        perPositionStrides: [:],
                        metadata: .init(
                            kernelName: selectedKernelName,
                            entryIndex: entry.index,
                            weightTensorName: weightTensorName,
                            bufferAccessPattern: gemmPattern
                        )
                    )
                )
            }

            routingState.lastOutputIsHidden = false
            routingState.currentInputOffset = lastOutputOffset
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .batchedFragment(let batch):
            var steps: [MetalPrefillStep] = []
            for (i, frag) in batch.fragments.enumerated() {
                let singleEntry = DispatchEntry(
                    index: entry.index + i,
                    kind: .fragment(frag),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex
                )
                let fragSteps = try buildSteps(for: singleEntry)
                steps.append(contentsOf: fragSteps)
            }
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .fusedResidualAddNorm(let fusedOp):
            let addEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralAdd(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
            var steps: [MetalPrefillStep] = []
            steps.append(contentsOf: try buildSteps(for: addEntry))
            steps.append(try makeHiddenToResidualCopyStep(
                dimension: fusedOp.dimension,
                entry: entry
            ))
            steps.append(contentsOf: try buildNormToHiddenStep(
                inputBuffer: buffers.residual,
                inputOffset: 0,
                dimension: fusedOp.dimension,
                epsilon: fusedOp.epsilon,
                weightRole: "scale",
                weightBias: 0,
                entry: entry
            ))
            // Standalone sequence RMSNorm kernels use logical `dimension`
            // row strides, which matches `hidden` but not `scratch`'s
            // slotDimension stride. Keep normalized activations in hidden
            // for subsequent projections / output-head routing.
            routingState.lastOutputIsHidden = true
            routingState.currentInputOffset = 0
            routingState.projectionIndex = 0
            refreshCompositeInputSource()
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .fusedSwiGLUProjection(let fusedOp):
            let batchedEntry = DispatchEntry(
                index: entry.index,
                kind: .batchedProjection(BatchedProjection(projections: [
                    .init(field: fusedOp.gateField, inputDimension: fusedOp.inputDimension, outputDimension: fusedOp.outputDimension),
                    .init(field: fusedOp.upField, inputDimension: fusedOp.inputDimension, outputDimension: fusedOp.outputDimension),
                ])),
                parameterBindings: entry.parameterBindings,
                layerIndex: entry.layerIndex
            )
            let elementwiseKind: ElementwiseFragment.ElementwiseKind = switch fusedOp.activation {
            case .silu: .swiglu
            case .geluTanh: .geluGated
            }
            let swigluEntry = DispatchEntry(
                index: entry.index + 1,
                kind: .fragment(ElementwiseFragment(count: fusedOp.outputDimension, kind: elementwiseKind)),
                parameterBindings: entry.parameterBindings,
                layerIndex: entry.layerIndex
            )

            var steps: [MetalPrefillStep] = []
            for decomposed in [batchedEntry, swigluEntry] {
                let built = try buildSteps(for: decomposed)
                steps.append(contentsOf: built)
            }
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .fusedCopyNorm(let fusedOp):
            var steps: [MetalPrefillStep] = []
            let copyEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralCopy(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
            steps.append(contentsOf: try buildSteps(for: copyEntry))
            steps.append(contentsOf: try buildNormToHiddenStep(
                inputBuffer: buffers.residual,
                inputOffset: 0,
                dimension: fusedOp.dimension,
                epsilon: fusedOp.epsilon,
                weightRole: "scale",
                weightBias: 0,
                entry: entry
            ))
            routingState.lastOutputIsHidden = true
            routingState.currentInputOffset = 0
            routingState.projectionIndex = 0
            refreshCompositeInputSource()
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .fusedResidualAddCopyNorm(let fusedOp):
            var steps: [MetalPrefillStep] = []
            let addEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralAdd(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
            let copyEntry = DispatchEntry(
                index: entry.index + 1,
                kind: .structuralCopy(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
            steps.append(contentsOf: try buildSteps(for: addEntry))
            steps.append(contentsOf: try buildSteps(for: copyEntry))
            steps.append(contentsOf: try buildNormToHiddenStep(
                inputBuffer: buffers.residual,
                inputOffset: 0,
                dimension: fusedOp.dimension,
                epsilon: fusedOp.epsilon,
                weightRole: "scale",
                weightBias: 0,
                entry: entry
            ))
            routingState.lastOutputIsHidden = true
            routingState.currentInputOffset = 0
            routingState.projectionIndex = 0
            refreshCompositeInputSource()
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .projection(let projection, let isOutput):
            let resolved = try resolveDispatch(entry)
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
            let weightTensorName = entry.parameterBindings.first(where: { $0.role == projection.field })?.tensorName
            let quantizationDescriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)

            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if !isOutput, let compositeInputSource {
                inputBuffer = compositeInputSource.buffer
                inputOffset = compositeInputSource.offset
            } else if routingState.lastOutputIsHidden {
                inputBuffer = buffers.hidden
                inputOffset = 0
            } else {
                inputBuffer = buffers.scratch
                inputOffset = routingState.currentInputOffset
            }

            let outputBuffer: MTLBuffer
            let outputOffset: Int
            let mode: PrefillStepMode
            let seqLenValue: UInt32
            let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength
            let inputRowStride = inputBuffer === buffers.hidden
                ? (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize
                : projection.inputDimension

            if isOutput && projection.outputDimension > hiddenSize {
                let inputRowStride = inputBuffer === buffers.hidden
                    ? buffers.hidden.length / max(maximumSequenceLength, 1)
                    : projection.inputDimension * scratchElementSize
                outputHeadInputSource = (
                    buffer: inputBuffer,
                    offset: inputOffset,
                    rowStride: inputRowStride
                )
                outputBuffer = buffers.logits
                outputOffset = 0
                mode = .lastToken
                seqLenValue = 1
                routingState.lastOutputIsHidden = false
                routingState.currentInputOffset = 0
            } else if isOutput {
                outputBuffer = buffers.hidden
                outputOffset = 0
                mode = .batch
                seqLenValue = UInt32(maximumSequenceLength)
                routingState.lastOutputIsHidden = true
                routingState.currentInputOffset = 0
            } else {
                let scratchSlot = routingState.projectionIndex + 1
                outputBuffer = buffers.scratch
                outputOffset = scratchSlot * scratchSlotSize
                mode = .batch
                seqLenValue = UInt32(maximumSequenceLength)
                routingState.lastOutputIsHidden = false
                routingState.currentInputOffset = outputOffset
            }
            routingState.projectionIndex += 1

            var perPositionStrides: [Int: Int] = [:]
            if mode == .lastToken {
                let inputRowStride = inputBuffer === buffers.hidden
                    ? buffers.hidden.length / max(maximumSequenceLength, 1)
                    : projection.inputDimension * scratchElementSize
                perPositionStrides[0] = inputRowStride
            }
            // Q4 with dequant scratch → dequant to BF16, then AMX matmul2d
            let canDequantForAMX = quantizationDescriptor.schemeIdentifier.isWeightQuantized
                && buffers.dequantScratch != nil
                && dequantKernelName(for: quantizationDescriptor.schemeIdentifier) != nil
            let usesMPPForStep = usesMPP
                && mode == .batch
                && inputRowStride == projection.inputDimension
                && (!quantizationDescriptor.schemeIdentifier.isWeightQuantized || canDequantForAMX)

            // Emit dequant step: Q4 weight → BF16 dequant scratch
            var dequantSteps: [MetalPrefillStep] = []
            if canDequantForAMX && usesMPPForStep,
               let dequantName = dequantKernelName(for: quantizationDescriptor.schemeIdentifier),
               let dequantPipeline = planBuildContext.pipelineCache[dequantName],
               let dequantScratch = buffers.dequantScratch {
                dequantSteps.append(
                    MetalPrefillStep(
                        pipeline: dequantPipeline,
                        gridSize: MTLSize(width: projection.outputDimension, height: 1, depth: 1),
                        threadgroupSize: MTLSize(width: 256, height: 1, depth: 1),
                        bufferBindings: [
                            (0, weightBuffer, weightOffset),
                            (1, dequantScratch, 0),
                        ],
                        bytesBindings: [
                            uint32Binding(2, UInt32(projection.inputDimension)),
                            uint32Binding(3, UInt32(projection.outputDimension)),
                        ],
                        threadgroupMemoryLength: 0,
                        sync: .bufferBarrier,
                        mode: .batch,
                        sequenceLengthPolicy: .none,
                        positionBufferIndex: nil,
                        perPositionStrides: [:],
                        metadata: .init(
                            kernelName: dequantName,
                            entryIndex: entry.index,
                            weightTensorName: weightTensorName,
                            bufferAccessPattern: .init(reads: [0], writes: [1])
                        )
                    )
                )
            }

            // Resolve GEMM pipeline
            let selectedPipeline: MTLComputePipelineState
            let selectedKernelName: String
            if canDequantForAMX && usesMPPForStep,
               let mppPipeline = planBuildContext.pipelineCache["gemm_bf16_f32s"] {
                // Dequant path: Q4 unpacked to BF16, use BF16 MPP GEMM
                selectedPipeline = mppPipeline
                selectedKernelName = "gemm_bf16_f32s"
            } else if !usesMPPForStep,
               let naivePipeline = planBuildContext.pipelineCache["naive::\(resolved.name)"] {
                selectedPipeline = naivePipeline
                selectedKernelName = "naive::\(resolved.name)"
            } else {
                selectedPipeline = resolved.pipeline
                selectedKernelName = resolved.name
            }

            // GEMM weight source: dequant scratch (BF16) or original weight buffer
            let gemmWeightBuffer: MTLBuffer
            let gemmWeightOffset: Int
            if canDequantForAMX && usesMPPForStep, let dequantScratch = buffers.dequantScratch {
                gemmWeightBuffer = dequantScratch
                gemmWeightOffset = 0
            } else {
                gemmWeightBuffer = weightBuffer
                gemmWeightOffset = weightOffset
            }

            let gridSize: MTLSize
            let threadgroupSize: MTLSize
            if usesMPPForStep {
                let simdWidth = selectedPipeline.threadExecutionWidth
                gridSize = MTLSize(
                    width: (projection.outputDimension + 31) / 32,
                    height: (maximumSequenceLength + 63) / 64,
                    depth: 1
                )
                threadgroupSize = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
            } else if mode == .batch {
                let simdWidth = max(selectedPipeline.threadExecutionWidth, 1)
                let rowsPerThreadgroup = 2
                let threads = min(
                    simdWidth * rowsPerThreadgroup,
                    selectedPipeline.maxTotalThreadsPerThreadgroup
                )
                gridSize = MTLSize(
                    width: (projection.outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                    height: maximumSequenceLength,
                    depth: 1
                )
                threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)
            } else if mode == .lastToken {
                gridSize = MTLSize(width: resolved.config.grid.width, height: 1, depth: 1)
                threadgroupSize = resolved.config.threadgroup
            } else {
                gridSize = MTLSize(
                    width: resolved.config.grid.width,
                    height: maximumSequenceLength,
                    depth: 1
                )
                threadgroupSize = resolved.config.threadgroup
            }

            // GEMM: reads input[0] + weight[1], writes output[2]
            let gemmPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
            recordProjectionQuantization(
                entry: entry,
                descriptor: quantizationDescriptor,
                mode: mode,
                inputRowStride: inputRowStride,
                inputDimension: projection.inputDimension,
                selectedKernelName: selectedKernelName,
                usesMPPForStep: usesMPPForStep
            )
            return dequantSteps + [MetalPrefillStep(
                pipeline: selectedPipeline,
                gridSize: gridSize,
                threadgroupSize: threadgroupSize,
                bufferBindings: [
                    (0, inputBuffer, inputOffset),
                    (1, gemmWeightBuffer, gemmWeightOffset),
                    (2, outputBuffer, outputOffset),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(projection.inputDimension)),
                    uint32Binding(4, UInt32(projection.outputDimension)),
                    uint32Binding(5, seqLenValue),
                    uint32Binding(6, UInt32(inputRowStride)),
                ],
                threadgroupMemoryLength: usesMPPForStep ? 0 : resolved.config.sharedMemoryBytes,
                sync: .bufferBarrier,
                mode: mode,
                sequenceLengthPolicy: mode == .batch
                    ? (usesMPPForStep
                        ? .bindAndAdjustGridHeightTiled(index: 5, tileHeight: 64)
                        : .bindAndAdjustGridHeight(index: 5))
                    : .none,
                positionBufferIndex: nil,
                perPositionStrides: perPositionStrides,
                metadata: .init(
                    kernelName: selectedKernelName,
                    entryIndex: entry.index,
                    weightTensorName: weightTensorName,
                    bufferAccessPattern: gemmPattern
                )
            )]

        case .structuralCopy(let dimension):
            let resolved = try resolveDispatch(entry)
            routingState.projectionIndex = 0
            routingState.currentInputOffset = 0

            // copy: reads source[0], writes destination[1]
            let copyPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0], writes: [1])
            return [MetalPrefillStep(
                pipeline: resolved.pipeline,
                gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
                threadgroupSize: resolved.config.threadgroup,
                bufferBindings: [
                    (0, buffers.hidden, 0),
                    (1, buffers.residual, 0),
                ],
                bytesBindings: [
                    uint32Binding(2, UInt32(dimension)),
                    uint32Binding(3, UInt32(maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 3),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(entryIndex: entry.index, bufferAccessPattern: copyPattern)
            )]

        case .structuralAdd(let dimension):
            let resolved = try resolveDispatch(entry)
            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if routingState.lastOutputIsHidden {
                inputBuffer = buffers.hidden
                inputOffset = 0
            } else {
                inputBuffer = buffers.scratch
                inputOffset = routingState.currentInputOffset
            }
            routingState.lastOutputIsHidden = true
            routingState.currentInputOffset = 0

            if inputBuffer === buffers.hidden, inputOffset == 0 {
                guard let inplacePipeline = planBuildContext.pipelineCache["residual_add_inplace_seq_f32"] else {
                    throw MetalCompilerError.kernelNotFound("residual_add_inplace_seq_f32")
                }
                let addPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [0])
                return [MetalPrefillStep(
                    pipeline: inplacePipeline,
                    gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
                    threadgroupSize: resolved.config.threadgroup,
                    bufferBindings: [
                        (0, buffers.hidden, 0),
                        (1, buffers.residual, 0),
                    ],
                    bytesBindings: [
                        uint32Binding(2, UInt32(dimension)),
                        uint32Binding(3, UInt32(maximumSequenceLength)),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 3),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: "residual_add_inplace_seq_f32",
                        entryIndex: entry.index,
                        bufferAccessPattern: addPattern
                    )
                )]
            }

            // add: reads operands[0,1], writes result[2]
            let addPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
            return [MetalPrefillStep(
                pipeline: resolved.pipeline,
                gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
                threadgroupSize: resolved.config.threadgroup,
                bufferBindings: [
                    (0, inputBuffer, inputOffset),
                    (1, buffers.residual, 0),
                    (2, buffers.hidden, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(dimension)),
                    uint32Binding(4, UInt32(maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(entryIndex: entry.index, bufferAccessPattern: addPattern)
            )]
        }
    }

    private mutating func updateCompositeInputSource(for entry: DispatchEntry) {
        guard activeCompositeID != entry.compositeID else { return }
        activeCompositeID = entry.compositeID
        refreshCompositeInputSource()
    }

    private mutating func refreshCompositeInputSource() {
        if routingState.lastOutputIsHidden {
            compositeInputSource = (buffers.hidden, 0)
        } else {
            compositeInputSource = (buffers.scratch, routingState.currentInputOffset)
        }
    }

    private func buildNormToHiddenStep(
        inputBuffer: MTLBuffer,
        inputOffset: Int,
        dimension: Int,
        epsilon: Float,
        weightRole: String,
        weightBias: Float,
        entry: DispatchEntry
    ) throws -> [MetalPrefillStep] {
        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            fallbackBuffer: buffers.hidden,
            fallbackWeightFormat: fallbackWeightFormat,
            minimumFallbackLength: minimumFallbackLength,
            logsMisses: false,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )

        let normKernelName = Reduction(
            dimension: dimension,
            epsilon: epsilon,
            weightRole: weightRole,
            weightBias: weightBias
        )
            .kernelName(context: planBuildContext.kernelContext)
        guard let pipeline = planBuildContext.pipelineCache[normKernelName] else {
            throw MetalCompilerError.kernelNotFound(normKernelName)
        }
        let simdWidth = pipeline.threadExecutionWidth
        let clamped = min(max(dimension, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

        let (weightBuffer, weightOffset) = weightResolver.resolve(role: weightRole)

        // norm: reads input[0] + weight[1], writes output[2]
        let normPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
        return [MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, inputBuffer, inputOffset),
                (1, weightBuffer, weightOffset),
                (2, buffers.hidden, 0),
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(dimension)),
                floatBinding(4, epsilon),
                floatBinding(5, weightBias),
                uint32Binding(6, UInt32(maximumSequenceLength)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bind(index: 6),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                entryIndex: entry.index,
                weightTensorName: entry.parameterBindings.first(where: { $0.role == weightRole })?.tensorName,
                bufferAccessPattern: normPattern
            )
        )]
    }

    private func makeHiddenToResidualCopyStep(
        dimension: Int,
        entry: DispatchEntry
    ) throws -> MetalPrefillStep {
        let resolved = try resolveDispatch(
            DispatchEntry(
                index: entry.index,
                kind: .structuralCopy(dimension: dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
        )
        let copyPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0], writes: [1])
        return MetalPrefillStep(
            pipeline: resolved.pipeline,
            gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
            threadgroupSize: resolved.config.threadgroup,
            bufferBindings: [
                (0, buffers.hidden, 0),
                (1, buffers.residual, 0),
            ],
            bytesBindings: [
                uint32Binding(2, UInt32(dimension)),
                uint32Binding(3, UInt32(maximumSequenceLength)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 3),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(entryIndex: entry.index, bufferAccessPattern: copyPattern)
        )
    }

    private func shouldCaptureResidualInput(for weightRole: String) -> Bool {
        switch weightRole {
        case "input_layernorm", "pre_feedforward_layernorm", "operator_norm":
            return true
        default:
            return false
        }
    }

    func finalHiddenSource() -> (buffer: MTLBuffer, offset: Int, rowStride: Int) {
        if let outputHeadInputSource {
            return outputHeadInputSource
        }
        if routingState.lastOutputIsHidden {
            let rowStride = buffers.hidden.length / max(maximumSequenceLength, 1)
            return (buffers.hidden, 0, rowStride)
        }
        // Scratch is laid out using the slot dimension for every token row.
        // The hidden vector may occupy only a prefix of that row, but per-token
        // addressing must still advance by the full slot stride.
        let rowStride = slotDimension * scratchElementSize
        return (buffers.scratch, routingState.currentInputOffset, rowStride)
    }

    mutating func makeQuantizationPlan() -> MetalQuantizationPlan {
        MetalQuantizationPlan(
            capabilities: planBuildContext.quantizationCapabilities,
            entries: quantizationEntries
        )
    }

    private mutating func recordProjectionQuantization(
        entry: DispatchEntry,
        descriptor: ProjectionWeightDescriptor,
        mode: PrefillStepMode,
        inputRowStride: Int,
        inputDimension: Int,
        selectedKernelName: String,
        usesMPPForStep: Bool
    ) {
        let fallbackReason = resolveProjectionFallbackReason(
            descriptor: descriptor,
            mode: mode,
            inputRowStride: inputRowStride,
            inputDimension: inputDimension,
            usesMPPForStep: usesMPPForStep
        )
        quantizationEntries.append(
            MetalQuantizationPlanEntry(
                entryIndex: entry.index,
                layerIndex: entry.layerIndex,
                tensorName: descriptor.tensorName,
                path: .prefillProjection,
                schemeIdentifier: descriptor.schemeIdentifier,
                layout: descriptor.layout,
                kernelFamily: .classify(
                    kernelName: selectedKernelName,
                    usesMPP: usesMPPForStep
                ),
                usedFallback: descriptor.usedFallback || fallbackReason != nil,
                fallbackReason: descriptor.fallbackReason ?? fallbackReason
            )
        )
    }

    private func resolveProjectionWeightDescriptor(
        role: String,
        entry: DispatchEntry
    ) -> ProjectionWeightDescriptor {
        guard let binding = entry.parameterBindings.first(where: { $0.role == role }) else {
            return ProjectionWeightDescriptor(
                tensorName: nil,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: .rowMajor,
                usedFallback: true,
                fallbackReason: .missingTensorBinding
            )
        }
        guard let stafWeightStore else {
            return ProjectionWeightDescriptor(
                tensorName: binding.tensorName,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: .rowMajor,
                usedFallback: true,
                fallbackReason: .missingWeightStore
            )
        }

        let request = planBuildContext.compileContext.accessPolicyResolver.accessRequest(
            for: entry,
            role: role,
            binding: binding,
            executionPhase: .prefill,
            stafWeightStore: stafWeightStore
        )
        let layout = stafWeightStore.resolvedBufferAccess(for: request)?.layout ?? request.preferredLayout
        guard let tensorEntry = stafWeightStore.entries[binding.tensorName] else {
            return ProjectionWeightDescriptor(
                tensorName: binding.tensorName,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: layout,
                usedFallback: true,
                fallbackReason: .missingTensorMetadata
            )
        }
        return ProjectionWeightDescriptor(
            tensorName: binding.tensorName,
            schemeIdentifier: tensorEntry.schemeIdentifier,
            layout: layout,
            usedFallback: false,
            fallbackReason: nil
        )
    }

    private func resolveProjectionFallbackReason(
        descriptor: ProjectionWeightDescriptor,
        mode: PrefillStepMode,
        inputRowStride: Int,
        inputDimension: Int,
        usesMPPForStep: Bool
    ) -> MetalQuantizationFallbackReason? {
        if let fallbackReason = descriptor.fallbackReason {
            return fallbackReason
        }
        if mode == .lastToken {
            return .lastTokenProjectionUsesDecodeKernel
        }
        if inputRowStride != inputDimension {
            return .inputStrideMismatch
        }
        guard !descriptor.schemeIdentifier.isWeightQuantized else {
            return nil
        }
        guard !usesMPPForStep else {
            return nil
        }
        switch planBuildContext.quantizationCapabilities.prefillProjectionAcceleration {
        case .disabledByEnvironment:
            return .disabledByEnvironment
        case .unavailable:
            return .unavailableAcceleration
        case .enabled:
            return nil
        }
    }

    private var fallbackSchemeIdentifier: QuantizationSchemeIdentifier {
        switch fallbackWeightFormat {
        case .float16:
            return .fp16RowMajor
        case .bfloat16:
            return .bf16RowMajor
        case .float32:
            return .fp32RowMajor
        case .quantized4Bit(let groupSize):
            switch groupSize {
            case 64:
                return .q4Group64ScaleF16
            case 128:
                return .q4Group128ScaleF16
            default:
                return .passthrough
            }
        case .quantized8Bit(let groupSize):
            switch groupSize {
            case 32:
                return .q8Group32ScaleF16
            case 64:
                return .q8Group64ScaleF16
            case 128:
                return .q8Group128ScaleF16
            default:
                return .passthrough
            }
        }
    }
}

private struct ProjectionWeightDescriptor {
    let tensorName: String?
    let schemeIdentifier: QuantizationSchemeIdentifier
    let layout: STAFWeightLayout
    let usedFallback: Bool
    let fallbackReason: MetalQuantizationFallbackReason?
}

// MARK: - Q4 Dequant → AMX Helpers

/// Dequant kernel name for the given quantization scheme.
/// Returns nil for non-Q4 schemes.
private func dequantKernelName(for scheme: QuantizationSchemeIdentifier) -> String? {
    switch scheme {
    case .q4Group64ScaleF16: return "dequant_q4_g64_bf16"
    case .q4Group128ScaleF16: return "dequant_q4_g128_bf16"
    default: return nil
    }
}
