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

        var index = 0
        while index < fusedEntries.count {
            let entry = fusedEntries[index]
            // Lookahead admission: when the next entry is a compatible
            // `mlp.down_proj` and the current entry is the producing SwiGLU,
            // we may emit a single fused kernel instead of the two-step path.
            // The admission check is enabled by default for stateful hybrid
            // sequence prefill and can be overridden with
            // `SWIFTLM_PREFILL_BF16_FUSED_MLP_DOWN`. It only succeeds when
            // every contract gate is satisfied. On success we consume two
            // entries and continue; on failure we fall through to the per-entry
            // path.
            if index + 1 < fusedEntries.count {
                let consumer = fusedEntries[index + 1]
                if planner.usesRuntimeGatedFusedMlpDown {
                    var fusedPlanner = planner
                    if let fusedSteps = try fusedPlanner.tryBuildFusedSwigluDownSteps(
                        producer: entry,
                        consumer: consumer
                    ) {
                        var unfusedPlanner = planner
                        let producerSteps = try unfusedPlanner.buildSteps(for: entry)
                        let consumerSteps = try unfusedPlanner.buildSteps(for: consumer)
                        // The fused and unfused paths must leave identical
                        // routing state for downstream entries. Keep the
                        // fused planner as the canonical state because the
                        // fused path owns the long-sequence execution branch.
                        let threshold = planner.fusedMlpDownMinimumSequenceLength
                        steps.append(contentsOf: (producerSteps + consumerSteps).map {
                            $0.withExecutionCondition(.sequenceLengthAtMost(threshold - 1))
                        })
                        steps.append(contentsOf: fusedSteps.map {
                            $0.withExecutionCondition(.sequenceLengthAtLeast(threshold))
                        })
                        planner = fusedPlanner
                        index += 2
                        continue
                    }
                } else if let fusedSteps = try planner.tryBuildFusedSwigluDownSteps(
                    producer: entry,
                    consumer: consumer
                ) {
                    steps.append(contentsOf: fusedSteps)
                    index += 2
                    continue
                }
            }
            let prefillSteps = try planner.buildSteps(for: entry)
            steps.append(contentsOf: prefillSteps)
            index += 1
        }

        let correctedSteps = try Self.insertDecodeEquivalentSequenceStorageRoundingIfNeeded(
            steps,
            buffers: buffers,
            slotDimension: slotDimension,
            hiddenSize: hiddenSize,
            maximumSequenceLength: maximumSequenceLength,
            planBuildContext: planBuildContext
        )
        let residentSteps = try makeResidentConstantSteps(correctedSteps, allocator: constantAllocator)
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

    private static func insertDecodeEquivalentSequenceStorageRoundingIfNeeded(
        _ steps: [MetalPrefillStep],
        buffers: PrefillBufferSet,
        slotDimension: Int,
        hiddenSize: Int,
        maximumSequenceLength: Int,
        planBuildContext: PlanBuildContext
    ) throws -> [MetalPrefillStep] {
        guard buffers.bufferPrecision.isPrefillSequencePrecision else {
            return steps
        }
        let kernelName: String
        switch planBuildContext.compileContext.decodeBufferPrecision {
        case .float16:
            kernelName = "round_f16_seq_f32"
        case .bfloat16:
            kernelName = "round_bf16_seq_f32"
        case .float32, .float32Decode:
            return steps
        }
        guard let pipeline = planBuildContext.pipelineCache[kernelName] else {
            throw MetalCompilerError.kernelNotFound(kernelName)
        }

        var corrected: [MetalPrefillStep] = []
        corrected.reserveCapacity(steps.count * 2)
        for step in steps {
            corrected.append(step)
            corrected.append(contentsOf: try makeSequenceStorageRoundingSteps(
                after: step,
                kernelName: kernelName,
                pipeline: pipeline,
                buffers: buffers,
                slotDimension: slotDimension,
                hiddenSize: hiddenSize,
                maximumSequenceLength: maximumSequenceLength
            ))
        }
        return corrected
    }

    private static func makeSequenceStorageRoundingSteps(
        after step: MetalPrefillStep,
        kernelName: String,
        pipeline: MTLComputePipelineState,
        buffers: PrefillBufferSet,
        slotDimension: Int,
        hiddenSize: Int,
        maximumSequenceLength: Int
    ) throws -> [MetalPrefillStep] {
        if shouldPreserveFloat32SequenceStorage(after: step) {
            return []
        }
        guard let bufferAccessPattern = step.metadata.bufferAccessPattern else {
            if step.bufferBindings.contains(where: {
                isSequenceActivationBuffer($0.buffer, buffers: buffers)
            }) {
                let producer = step.metadata.kernelName ?? step.pipeline.label ?? "<unknown>"
                throw MetalCompilerError.deviceSetupFailed(
                    "Sequence storage rounding requires buffer access metadata for \(producer)"
                )
            }
            return []
        }
        let writeIndices = bufferAccessPattern.writeIndices
        guard !writeIndices.isEmpty else {
            return []
        }

        var roundedRegions = Set<BufferRegion>()
        var roundSteps: [MetalPrefillStep] = []
        for binding in step.bufferBindings where writeIndices.contains(binding.index) {
            guard let elementCount = float16RoundElementCount(
                buffer: binding.buffer,
                offset: binding.offset,
                buffers: buffers,
                slotDimension: slotDimension,
                hiddenSize: hiddenSize,
                maximumSequenceLength: maximumSequenceLength
            ), elementCount > 0 else {
                continue
            }
            let region = BufferRegion(buffer: binding.buffer, offset: binding.offset)
            guard roundedRegions.insert(region).inserted else { continue }

            let threads = min(
                max(pipeline.threadExecutionWidth, 1) * 4,
                pipeline.maxTotalThreadsPerThreadgroup
            )
            let groups = (elementCount + threads - 1) / threads
            roundSteps.append(MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: max(groups, 1), height: 1, depth: 1),
                threadgroupSize: MTLSize(width: max(threads, 1), height: 1, depth: 1),
                bufferBindings: [(0, binding.buffer, binding.offset)],
                bytesBindings: [uint32Binding(1, UInt32(elementCount))],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .none,
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: kernelName,
                    entryIndex: step.metadata.entryIndex,
                    layerIndex: step.metadata.layerIndex,
                    bufferAccessPattern: .init(reads: [0], writes: [0])
                ),
                executionCondition: step.executionCondition
            ))
        }
        return roundSteps
    }

    private static func shouldPreserveFloat32SequenceStorage(after step: MetalPrefillStep) -> Bool {
        guard let kernelName = step.metadata.kernelName else {
            return false
        }
        if kernelName.hasPrefix("gemv_seq_") {
            return true
        }
        if kernelName.hasPrefix("batched_gemv")
            && kernelName.contains("_seq_") {
            return true
        }
        if kernelName.hasPrefix("synthesized_") {
            return true
        }
        return false
    }

    private static func float16RoundElementCount(
        buffer: MTLBuffer,
        offset: Int,
        buffers: PrefillBufferSet,
        slotDimension: Int,
        hiddenSize: Int,
        maximumSequenceLength: Int
    ) -> Int? {
        let availableElements = max(0, (buffer.length - offset) / MemoryLayout<Float>.stride)
        if buffer === buffers.scratch {
            return min(slotDimension * maximumSequenceLength, availableElements)
        }
        if buffer === buffers.hidden || buffer === buffers.residual {
            return min(hiddenSize * maximumSequenceLength, availableElements)
        }
        return nil
    }

    private static func isSequenceActivationBuffer(
        _ buffer: MTLBuffer,
        buffers: PrefillBufferSet
    ) -> Bool {
        buffer === buffers.scratch || buffer === buffers.hidden || buffer === buffers.residual
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
                let visibility: MTL4VisibilityOptions =
                    MetalBufferAccesses.pendingWritesInvolveSharedBuffer(pendingWrites)
                    ? .device : []
                newBarrierPolicy = .barrier(visibility: visibility)
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
                metadata: step.metadata,
                executionCondition: step.executionCondition,
                tileVariants: step.tileVariants
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
                metadata: step.metadata,
                executionCondition: step.executionCondition,
                tileVariants: step.tileVariants
            )
        }
    }
}

/// Selection of a sequence-equivalent GEMV kernel for one prefill projection.
///
/// `kernelName` is the catalog kernel that will be dispatched. `sequenceTile`
/// is the number of token rows that one threadgroup processes along the
/// sequence axis. `1` means the base, untiled kernel (`gemv_seq_*`). Values
/// greater than 1 select a tiled variant (`gemv_seq_*_tile<N>`) and require
/// the planner to scale grid height, threadgroup shape, diagnostics, and
/// `bindAndAdjustGridHeightTiled` consistently. The planner must derive all
/// tile-aware behaviour from this value rather than parsing the kernel name.
private struct SequenceGEMVKernelSelection: Equatable {
    let kernelName: String
    let sequenceTile: Int
    let rowsPerSimdgroup: Int

    var isTiled: Bool { sequenceTile > 1 }
}

private struct PendingRecurrentBlockOutputProjection {
    let compositeID: Int?
    let layerIndex: Int?
    let recurrentGroupCount: Int
    let recurrentOutputDimension: Int
}

private struct PrefillStepPlanner {
    private static let decodeEquivalentSequenceRowsPerThreadgroup = 2

    /// Process-wide BF16 single sequence GEMV tile2 feature flag.
    ///
    /// When `SWIFTLM_PREFILL_BF16_SINGLE_TILE2=1`, BF16 single (non-batched)
    /// sequence GEMV projections route to `gemv_seq_bf16_f32s_tile2`. Other
    /// schemes (FP16, Q3-*) and the batched path are unaffected. Read once
    /// per process to avoid repeated `getenv` calls.
    private static let bf16SingleTile2Enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_BF16_SINGLE_TILE2"] else {
            return false
        }
        return raw == "1" || raw.lowercased() == "true"
    }()

    /// Process-wide BF16 single sequence GEMV row2 feature flag.
    ///
    /// When `SWIFTLM_PREFILL_BF16_SINGLE_RPS2=1`, BF16 single sequence GEMV
    /// projections route to `gemv_seq_bf16_f32s_rps2`. This keeps the sequence
    /// tile at 1 and computes two independent output rows per SIMD group,
    /// sharing the staged input tile without changing each row's reduction
    /// order. It is mutually conservative with the tile2 experiment: tile2
    /// takes precedence when both flags are enabled.
    private static let bf16SingleRowsPerSimdgroup2Enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_BF16_SINGLE_RPS2"] else {
            return false
        }
        return raw == "1" || raw.lowercased() == "true"
    }()

    /// Process-wide BF16 fused SwiGLU+down_proj override.
    ///
    /// By default the planner enables the fused route only for stateful hybrid
    /// sequence prefill (`convState` or `recurrentState` present), where Qwen
    /// reference gates and profile windows show a stable long-prompt benefit.
    /// Set `SWIFTLM_PREFILL_BF16_FUSED_MLP_DOWN=0` to disable it, or `=1` to
    /// force-enable it for experiments on non-hybrid plans. Admission still
    /// requires matching compositeID/layerIndex, BF16 sequence prefill
    /// precision, BF16 dense weight, output writing back to hidden, and `.batch`
    /// mode. Other projections, schemes, and decode are unaffected.
    private static let bf16FusedMlpDownOverride: Bool? = {
        guard let raw = ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_BF16_FUSED_MLP_DOWN"] else {
            return nil
        }
        return raw == "1" || raw.lowercased() == "true"
    }()

    /// Row-group width for the opt-in fused SwiGLU+down prefill experiment.
    ///
    /// The fused kernel computes the SwiGLU tile inside each output-row
    /// threadgroup. More rows per threadgroup amortize that tile computation
    /// across more down-projection rows while preserving the per-row SIMD
    /// reduction contract. The value is intentionally capped to keep occupancy
    /// predictable during experiments.
    private static let fusedMlpDownRowsPerThreadgroup: Int = {
        guard let raw = ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_BF16_FUSED_MLP_DOWN_ROWS"],
              let value = Int(raw) else {
            return 8
        }
        return min(max(value, 1), 16)
    }()

    /// Number of output rows computed by each SIMD group in the opt-in fused
    /// SwiGLU+down prefill experiment.
    ///
    /// `1` preserves the original one-SIMD-one-row reduction shape. `2`
    /// shares one staged SwiGLU tile across two independent row accumulators
    /// inside the same SIMD group. This reduces activation recompute while
    /// keeping each output row's reduction order unchanged.
    private static let fusedMlpDownRowsPerSimdgroup: Int = {
        guard let raw = ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_BF16_FUSED_MLP_DOWN_ROWS_PER_SIMDGROUP"],
              let value = Int(raw) else {
            return 1
        }
        return min(max(value, 1), 2)
    }()

    /// Minimum sequence length for the fused SwiGLU+down prefill route.
    ///
    /// Values greater than 1 make the planner emit both the unfused and fused
    /// paths with explicit runtime execution conditions. This is deterministic
    /// runtime admission, not a fallback: short sequences run the existing
    /// unfused contract, and longer sequences run the admitted fused contract.
    /// The default stays at 64 because the Qwen profile shows the fused route
    /// is a long-prompt win while seqLen 16 should keep the original path.
    private static let fusedMlpDownMinimumSequenceLength: Int = {
        guard let raw = ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_BF16_FUSED_MLP_DOWN_MIN_SEQUENCE_LENGTH"],
              let value = Int(raw) else {
            return 64
        }
        return max(value, 1)
    }()

    /// Process-wide BF16 recurrent block partial-projection feature flag.
    ///
    /// When `SWIFTLM_PREFILL_BF16_RECURRENT_BLOCK_PARTIAL=1`, a matching
    /// linear-attention `(SSM recurrence, out_proj)` pair routes through:
    ///
    ///   1. `recurrent_block_partial_projection_seq_bf16_f32`
    ///   2. `recurrent_block_partial_reduce_seq_f32`
    ///
    /// The path is opt-in while real-bundle profile evidence is collected.
    /// Admission is strict; a targeted recurrent out-projection that violates
    /// the contract fails explicitly instead of falling back silently.
    private static let bf16RecurrentBlockPartialProjectionEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_BF16_RECURRENT_BLOCK_PARTIAL"] else {
            return false
        }
        return raw == "1" || raw.lowercased() == "true"
    }()

    private static func largestDivisor(of value: Int, notExceeding maximum: Int) -> Int {
        guard value > 0, maximum > 0 else { return 0 }
        for candidate in stride(from: min(value, maximum), through: 1, by: -1) {
            if value % candidate == 0 {
                return candidate
            }
        }
        return 0
    }

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
    var pendingRecurrentBlockOutputProjection: PendingRecurrentBlockOutputProjection?

    var fusedMlpDownMinimumSequenceLength: Int {
        Self.fusedMlpDownMinimumSequenceLength
    }

    var bf16FusedMlpDownEnabled: Bool {
        Self.bf16FusedMlpDownOverride ?? needsDecodeEquivalentSequenceProjectionMath
    }

    var usesRuntimeGatedFusedMlpDown: Bool {
        bf16FusedMlpDownEnabled && Self.fusedMlpDownMinimumSequenceLength > 1
    }

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
                ),
                executionCondition: step.executionCondition,
                tileVariants: step.tileVariants
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

    private var needsDecodeEquivalentSequenceProjectionMath: Bool {
        buffers.bufferPrecision.isPrefillSequencePrecision
            && (buffers.convState != nil || buffers.recurrentState != nil)
    }

    private func decodeEquivalentSequenceGEMVKernelName(
        for descriptor: ProjectionWeightDescriptor
    ) -> SequenceGEMVKernelSelection? {
        guard needsDecodeEquivalentSequenceProjectionMath else { return nil }
        switch descriptor.schemeIdentifier {
        case .bf16RowMajor:
            // Feature-flagged tile2 routing: only BF16 single sequence GEMV.
            // Quantized and FP16 paths intentionally stay on base kernels until
            // their own profile evidence is collected.
            if Self.bf16SingleTile2Enabled {
                return SequenceGEMVKernelSelection(
                    kernelName: "gemv_seq_bf16_f32s_tile2",
                    sequenceTile: 2,
                    rowsPerSimdgroup: 1
                )
            }
            if Self.bf16SingleRowsPerSimdgroup2Enabled {
                return SequenceGEMVKernelSelection(
                    kernelName: "gemv_seq_bf16_f32s_rps2",
                    sequenceTile: 1,
                    rowsPerSimdgroup: 2
                )
            }
            return SequenceGEMVKernelSelection(
                kernelName: "gemv_seq_bf16_f32s",
                sequenceTile: 1,
                rowsPerSimdgroup: 1
            )
        case .fp16RowMajor:
            return SequenceGEMVKernelSelection(
                kernelName: "gemv_seq_f32s",
                sequenceTile: 1,
                rowsPerSimdgroup: 1
            )
        case .q3Group16ScaleF16:
            return SequenceGEMVKernelSelection(
                kernelName: "gemv_seq_q3_g16_f32s",
                sequenceTile: 1,
                rowsPerSimdgroup: 1
            )
        case .q3Group32ScaleF16:
            return SequenceGEMVKernelSelection(
                kernelName: "gemv_seq_q3_g32_f32s",
                sequenceTile: 1,
                rowsPerSimdgroup: 1
            )
        case .q3Group64ScaleF16:
            return SequenceGEMVKernelSelection(
                kernelName: "gemv_seq_q3_g64_f32s",
                sequenceTile: 1,
                rowsPerSimdgroup: 1
            )
        default:
            return nil
        }
    }

    private func decodeEquivalentBatchedSequenceGEMVKernelName(
        for descriptor: ProjectionWeightDescriptor,
        count: Int
    ) -> SequenceGEMVKernelSelection? {
        guard needsDecodeEquivalentSequenceProjectionMath else { return nil }
        guard count >= 2 && count <= 4 else { return nil }
        let baseName: String
        switch descriptor.schemeIdentifier {
        case .bf16RowMajor:
            baseName = "batched_gemv\(count)_seq_bf16_f32s"
        case .fp16RowMajor:
            baseName = "batched_gemv\(count)_seq_f32s"
        case .fp32RowMajor:
            baseName = "batched_gemv\(count)_seq_fp32_f32s"
        case .q3Group16ScaleF16:
            baseName = "batched_gemv\(count)_seq_q3_g16_f32s"
        case .q3Group32ScaleF16:
            baseName = "batched_gemv\(count)_seq_q3_g32_f32s"
        case .q3Group64ScaleF16:
            baseName = "batched_gemv\(count)_seq_q3_g64_f32s"
        default:
            return nil
        }
        // Batched path stays on the base (untiled) kernel. The tile2
        // feature flag is single-projection only by design.
        return SequenceGEMVKernelSelection(kernelName: baseName, sequenceTile: 1, rowsPerSimdgroup: 1)
    }

    /// Resolve threadgroup shape and grid-height tile for a tiled sequence GEMV.
    ///
    /// `requestedSequenceTile` must come from `SequenceGEMVKernelSelection.sequenceTile`
    /// — the planner never invents a tile size or parses it from the kernel
    /// name. If the requested tile exceeds the pipeline's max simdgroups per
    /// threadgroup, the kernel cannot be safely scheduled (each row needs one
    /// simdgroup) and we throw rather than silently fall back.
    private func decodeEquivalentSequenceThreadShape(
        pipeline: MTLComputePipelineState,
        requestedSequenceTile: Int,
        rowsPerSimdgroup: Int = 1
    ) throws -> (sequenceTile: Int, rowsPerThreadgroup: Int, threadgroupSize: MTLSize) {
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let maxSimdgroups = max(1, pipeline.maxTotalThreadsPerThreadgroup / simdWidth)
        guard requestedSequenceTile >= 1 else {
            throw MetalCompilerError.deviceSetupFailed(
                "Sequence tile \(requestedSequenceTile) is invalid for kernel \(pipeline.label ?? "(unlabeled)")"
            )
        }
        guard requestedSequenceTile <= maxSimdgroups else {
            throw MetalCompilerError.deviceSetupFailed(
                "Sequence tile \(requestedSequenceTile) exceeds max simdgroups \(maxSimdgroups) for kernel \(pipeline.label ?? "(unlabeled)")"
            )
        }
        let sequenceTile = requestedSequenceTile
        let simdgroupsPerThreadgroup = min(
            Self.decodeEquivalentSequenceRowsPerThreadgroup,
            max(1, maxSimdgroups / sequenceTile)
        )
        let rowsPerThreadgroup = simdgroupsPerThreadgroup * max(rowsPerSimdgroup, 1)
        let threads = simdWidth * sequenceTile * simdgroupsPerThreadgroup
        return (
            sequenceTile,
            rowsPerThreadgroup,
            MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    /// Attempt to fuse a `(SwiGLU, mlp.down_proj)` entry pair into a single
    /// `mlp_fused_swiglu_down_seq_bf16_f32s` dispatch.
    ///
    /// Returns the fused steps when every admission gate is satisfied, or `nil`
    /// when any gate fails — in which case the caller must fall back to the
    /// per-entry build path. The gates protect every assumption baked into the
    /// fused kernel:
    ///
    ///   1. **Fragment pair** — producer is an `ElementwiseFragment(.swiglu)`
    ///      and consumer is a `LinearFragment` for `mlp.down_proj` with
    ///      `isOutput == true`. GeluGated and other elementwise variants do
    ///      not share the SwiGLU rounding contract.
    ///   2. **CompositeID** — both entries must belong to the same MLP
    ///      composite. Different composites mean different scratch slot owners
    ///      and the fused kernel would read stale data.
    ///   3. **LayerIndex** — both entries must reference the same layer.
    ///   4. **Shape** — `linear.inputDimension == swiglu.count` and the down
    ///      projection must write at most `hiddenSize` rows. Output-head
    ///      projections (vocab GEMV) take a different routing path.
    ///   5. **Routing state** — the SwiGLU output must be the consumer's input
    ///      (`lastOutputIsHidden == false`). At admission time the routing
    ///      state still points at the post-up-projection scratch slot because
    ///      the producing SwiGLU step is intentionally skipped; the fused
    ///      kernel reads gate/up slots directly.
    ///   6. **Buffer precision** — must be the prefill sequence precision
    ///      (F32 hidden/scratch, BF16 storage rounding).
    ///   7. **Weight format** — `mlp.down_proj` must use a non-quantized
    ///      `.bf16RowMajor` scheme. Quantized variants need their own dequant
    ///      contracts.
    ///   8. **Mode** — must run in `.batch` mode. Last-token projections take
    ///      the lastToken path.
    ///
    /// All gates must succeed; partial admission is not allowed.
    mutating func tryBuildFusedSwigluDownSteps(
        producer: DispatchEntry,
        consumer: DispatchEntry
    ) throws -> [MetalPrefillStep]? {
        guard bf16FusedMlpDownEnabled else { return nil }

        // Gate 1: Fragment pair (SwiGLU producer + LinearFragment consumer).
        guard let swiglu = producer.fragment as? ElementwiseFragment,
              swiglu.kind == .swiglu else {
            return nil
        }
        guard let linear = consumer.fragment as? LinearFragment else {
            return nil
        }
        // Gate 1 (cont.): consumer must be the down projection writing back to
        // the hidden buffer. We restrict to the canonical `down_proj` role to
        // avoid accidentally fusing past parallel projections that happen to
        // sit next to a SwiGLU in the dispatch list.
        guard linear.field == "down_proj", linear.isOutput else {
            return nil
        }

        // Gate 2: CompositeID — both entries must share the same composite so
        // that scratch slot ownership is unambiguous.
        guard let composite = producer.compositeID,
              consumer.compositeID == composite else {
            return nil
        }

        // Gate 3: LayerIndex — both must reference the same layer.
        guard producer.layerIndex == consumer.layerIndex else {
            return nil
        }

        // Gate 4: Shape — SwiGLU output count must match the down projection's
        // input dimension; the projection output must fit into hidden.
        let intermediateDim = swiglu.count
        guard linear.inputDimension == intermediateDim,
              linear.outputDimension <= hiddenSize else {
            return nil
        }

        // Gate 5: Routing state — at this point the routing reflects the
        // post-up_proj state (gate_proj wrote slot 1, up_proj wrote slot 2,
        // `lastOutputIsHidden=false`). The fused kernel reads slots 1 and 2
        // directly, so we only require that the previous output stayed in
        // scratch (`lastOutputIsHidden == false`); `currentInputOffset` would
        // otherwise point at up_proj's slot 2, which the fused kernel does not
        // consume. The unfused path's reset to 0 happens inside SwiGLU's
        // prefillSteps, which the fused path skips entirely.
        guard !routingState.lastOutputIsHidden else {
            return nil
        }

        // Gate 6: Buffer precision — must be prefill sequence precision (F32
        // working buffers with BF16 storage rounding).
        guard buffers.bufferPrecision.isPrefillSequencePrecision else {
            return nil
        }

        // Gate 7: Weight format — fused kernel only supports BF16 row-major.
        let descriptor = resolveProjectionWeightDescriptor(role: linear.field, entry: consumer)
        guard descriptor.schemeIdentifier == .bf16RowMajor,
              !descriptor.schemeIdentifier.isWeightQuantized else {
            return nil
        }

        // Gate 8: Mode — fused kernel runs as `.batch` over the full sequence.
        // (`lastToken` projections take a different code path; we only fuse
        // when the consumer would also pick `.batch` in the unfused path,
        // which is true whenever `outputDimension <= hiddenSize` for an
        // `isOutput` projection.)
        let mode: PrefillStepMode = .batch

        // Resolve the fused kernel pipeline.
        let rowsPerSimdgroup = Self.fusedMlpDownRowsPerSimdgroup
        let fusedKernelName = rowsPerSimdgroup == 1
            ? "mlp_fused_swiglu_down_seq_bf16_f32s"
            : "mlp_fused_swiglu_down_seq_bf16_f32s_rps\(rowsPerSimdgroup)"
        guard let pipeline = planBuildContext.pipelineCache[fusedKernelName] else {
            throw MetalCompilerError.kernelNotFound(fusedKernelName)
        }

        // Resolve down_proj weight buffer.
        let weightResolver = WeightResolver(
            entry: consumer,
            stafWeightStore: stafWeightStore,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )
        let (weightBuffer, weightOffset) = weightResolver.resolve(role: linear.field)
        let weightTensorName = consumer.parameterBindings.first(where: { $0.role == linear.field })?.tensorName

        // Buffer layout matches `generateFusedSwigluDownSequenceGEMV`:
        //   buffer(0) = gate   = scratch slot 1
        //   buffer(1) = up     = scratch slot 2
        //   buffer(2) = weight = down_proj BF16
        //   buffer(3) = output = hidden offset 0
        let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength
        let inputRowStride = slotDimension
        let outputRowStride = (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize

        // Grid: (output rows / rowsPerThreadgroup) × sequenceLength.
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let requestedRowsPerThreadgroup = Self.fusedMlpDownRowsPerThreadgroup
        let requestedSimdgroupsPerThreadgroup = (requestedRowsPerThreadgroup + rowsPerSimdgroup - 1) / rowsPerSimdgroup
        let simdgroupsPerThreadgroup = max(
            1,
            min(requestedSimdgroupsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup / max(simdWidth, 1))
        )
        let rowsPerThreadgroup = simdgroupsPerThreadgroup * rowsPerSimdgroup
        let threads = simdWidth * simdgroupsPerThreadgroup
        let gridSize = MTLSize(
            width: (linear.outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: maximumSequenceLength,
            depth: 1
        )
        let threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)

        // Buffer access pattern: reads gate(0), up(1), weight(2); writes output(3).
        let accessPattern = MetalDispatchStepMetadata.BufferAccessPattern(
            reads: [0, 1, 2],
            writes: [3]
        )

        let fusedStep = MetalPrefillStep(
            pipeline: pipeline,
            gridSize: gridSize,
            threadgroupSize: threadgroupSize,
            bufferBindings: [
                (0, buffers.scratch, 1 * scratchSlotSize),
                (1, buffers.scratch, 2 * scratchSlotSize),
                (2, weightBuffer, weightOffset),
                (3, buffers.hidden, 0),
            ],
            bytesBindings: [
                uint32Binding(4, UInt32(intermediateDim)),
                uint32Binding(5, UInt32(linear.outputDimension)),
                uint32Binding(6, UInt32(maximumSequenceLength)),
                uint32Binding(7, UInt32(inputRowStride)),
                uint32Binding(8, UInt32(outputRowStride)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: mode,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 6),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: fusedKernelName,
                entryIndex: consumer.index,
                weightTensorName: weightTensorName,
                bufferAccessPattern: accessPattern
            )
        )

        // Record quantization plan entry for `down_proj` to keep the plan
        // consistent with the unfused path. SwiGLU is not a projection so
        // no entry is recorded for the producer.
        recordProjectionQuantization(
            entry: consumer,
            descriptor: descriptor,
            mode: mode,
            inputRowStride: inputRowStride,
            inputDimension: linear.inputDimension,
            outputDimension: linear.outputDimension,
            outputRowStride: outputRowStride,
            selectedKernelName: fusedKernelName,
            usesMPPForStep: false,
            usesSequenceGEMVForStep: true,
            sequenceTileHeight: nil,
            tileVariantHeights: []
        )

        // Update routing state so subsequent entries see the same state the
        // unfused (SwiGLU → LinearFragment isOutput) path would leave behind:
        //   * `lastOutputIsHidden = true` (down_proj wrote to hidden)
        //   * `currentInputOffset = 0`
        //   * `projectionIndex = 1` to match the unfused increment after
        //     SwiGLU's reset to 0 followed by LinearFragment's `+= 1`. The
        //     next reduction-bearing layer norm resets it again, so the
        //     specific value matters only for symmetry with the unfused path.
        routingState.lastOutputIsHidden = true
        routingState.currentInputOffset = 0
        routingState.projectionIndex = 1

        // Refresh composite-input tracking. Both entries share `composite`, so
        // `activeCompositeID` advances exactly once for the fused pair.
        activeCompositeID = composite
        refreshCompositeInputSource()

        let annotated = annotate([fusedStep], entryIndex: consumer.index, layerIndex: consumer.layerIndex)
        return annotated
    }

    mutating func buildSteps(for entry: DispatchEntry) throws -> [MetalPrefillStep] {
        updateCompositeInputSource(for: entry)

        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )

        if let linear = entry.fragment as? LinearFragment {
            if let recurrentSteps = try buildRecurrentBlockPartialOutputProjectionSteps(
                linear,
                entry: entry,
                weightResolver: weightResolver
            ) {
                return annotate(recurrentSteps, entryIndex: entry.index, layerIndex: entry.layerIndex)
            }
            pendingRecurrentBlockOutputProjection = nil

            let projection = linear
            let isOutput = linear.isOutput
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
                : slotDimension

            if isOutput && projection.outputDimension > hiddenSize {
                let inputRowStride = inputBuffer === buffers.hidden
                    ? buffers.hidden.length / max(maximumSequenceLength, 1)
                    : slotDimension * scratchElementSize
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
                    : slotDimension * scratchElementSize
                perPositionStrides[0] = inputRowStride
            }
            let outputRowStride: Int
            if outputBuffer === buffers.hidden {
                outputRowStride = (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize
            } else if outputBuffer === buffers.logits {
                outputRowStride = projection.outputDimension
            } else {
                outputRowStride = slotDimension
            }
            // Prefer direct quantized GEMM (dequant in registers) when available.
            // Falls back to dequant→AMX when no direct kernel exists.
            let directGEMM = resolveDirectQuantizedGEMM(for: quantizationDescriptor.schemeIdentifier)
            let useDirectQuantizedGEMM = directGEMM.flatMap {
                planBuildContext.pipelineCache[$0.kernelName]
            } != nil
            let sequenceGEMVSelection = mode == .batch
                ? decodeEquivalentSequenceGEMVKernelName(for: quantizationDescriptor)
                : nil
            let usesSequenceGEMVForStep = sequenceGEMVSelection != nil

            let canDequantForAMX = quantizationDescriptor.schemeIdentifier.isWeightQuantized
                && buffers.dequantScratch != nil
                && dequantKernelName(for: quantizationDescriptor.schemeIdentifier) != nil
            let usesMPPForStep = usesMPP
                && !usesSequenceGEMVForStep
                && mode == .batch
                && inputRowStride == projection.inputDimension
                && outputRowStride == projection.outputDimension
                && (!quantizationDescriptor.schemeIdentifier.isWeightQuantized || canDequantForAMX)
            let usesDequantScratchForStep = !useDirectQuantizedGEMM
                && canDequantForAMX
                && !usesSequenceGEMVForStep

            // Emit dequant step whenever a quantized projection has no direct prefill kernel.
            var dequantSteps: [MetalPrefillStep] = []
            if usesDequantScratchForStep,
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
            if let sequenceGEMVSelection {
                guard let sequencePipeline = planBuildContext.pipelineCache[sequenceGEMVSelection.kernelName] else {
                    throw MetalCompilerError.kernelNotFound(sequenceGEMVSelection.kernelName)
                }
                selectedPipeline = sequencePipeline
                selectedKernelName = sequenceGEMVSelection.kernelName
            } else if useDirectQuantizedGEMM,
               let resolvedGEMM = directGEMM,
               let directPipeline = planBuildContext.pipelineCache[resolvedGEMM.kernelName] {
                selectedPipeline = directPipeline
                selectedKernelName = resolvedGEMM.kernelName
            } else if canDequantForAMX && usesMPPForStep,
               let mppPipeline = planBuildContext.pipelineCache["gemm_bf16_f32s"] {
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

            // GEMM weight source: original packed weights (direct) or dequant scratch (BF16)
            let gemmWeightBuffer: MTLBuffer
            let gemmWeightOffset: Int
            if useDirectQuantizedGEMM {
                gemmWeightBuffer = weightBuffer
                gemmWeightOffset = weightOffset
            } else if usesDequantScratchForStep, let dequantScratch = buffers.dequantScratch {
                gemmWeightBuffer = dequantScratch
                gemmWeightOffset = 0
            } else {
                gemmWeightBuffer = weightBuffer
                gemmWeightOffset = weightOffset
            }

            let gridSize: MTLSize
            let threadgroupSize: MTLSize
            let sequenceGEMVTile: Int?
            if let sequenceGEMVSelection {
                let shape = try decodeEquivalentSequenceThreadShape(
                    pipeline: selectedPipeline,
                    requestedSequenceTile: sequenceGEMVSelection.sequenceTile,
                    rowsPerSimdgroup: sequenceGEMVSelection.rowsPerSimdgroup
                )
                sequenceGEMVTile = sequenceGEMVSelection.isTiled ? shape.sequenceTile : nil
                gridSize = MTLSize(
                    width: (projection.outputDimension + shape.rowsPerThreadgroup - 1) / shape.rowsPerThreadgroup,
                    height: sequenceGEMVSelection.isTiled
                        ? (maximumSequenceLength + shape.sequenceTile - 1) / shape.sequenceTile
                        : maximumSequenceLength,
                    depth: 1
                )
                threadgroupSize = shape.threadgroupSize
            } else if usesMPPForStep && !useDirectQuantizedGEMM {
                sequenceGEMVTile = nil
                let simdWidth = selectedPipeline.threadExecutionWidth
                gridSize = MTLSize(
                    width: (projection.outputDimension + 31) / 32,
                    height: (maximumSequenceLength + 63) / 64,
                    depth: 1
                )
                threadgroupSize = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
            } else if mode == .batch {
                sequenceGEMVTile = nil
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
                sequenceGEMVTile = nil
                gridSize = MTLSize(width: resolved.config.grid.width, height: 1, depth: 1)
                threadgroupSize = resolved.config.threadgroup
            } else {
                sequenceGEMVTile = nil
                gridSize = MTLSize(
                    width: resolved.config.grid.width,
                    height: maximumSequenceLength,
                    depth: 1
                )
                threadgroupSize = resolved.config.threadgroup
            }

            // GEMM: reads input[0] + weight[1], writes output[2]
            let gemmPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
            let mppTileVariants: [PrefillTileVariant]
            if mode == .batch && usesMPPForStep && !useDirectQuantizedGEMM {
                mppTileVariants = makeMPPTileVariants(
                    baseKernelName: selectedKernelName,
                    gridWidth: gridSize.width,
                    maxSequenceLength: maximumSequenceLength,
                    threadgroupSize: threadgroupSize,
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier)
            } else {
                mppTileVariants = []
            }
            recordProjectionQuantization(
                entry: entry,
                descriptor: quantizationDescriptor,
                mode: mode,
                inputRowStride: inputRowStride,
                inputDimension: projection.inputDimension,
                outputDimension: projection.outputDimension,
                outputRowStride: outputRowStride,
                selectedKernelName: selectedKernelName,
                usesMPPForStep: usesMPPForStep,
                usesSequenceGEMVForStep: usesSequenceGEMVForStep,
                sequenceTileHeight: sequenceGEMVTile
                    ?? (mode == .batch && usesMPPForStep && !useDirectQuantizedGEMM ? 64 : nil),
                tileVariantHeights: mppTileVariants.map(\.tileHeight)
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
                bytesBindings: {
                    var bindings: [(index: Int, value: [UInt8])] = [
                        uint32Binding(3, UInt32(projection.inputDimension)),
                        uint32Binding(4, UInt32(projection.outputDimension)),
                        uint32Binding(5, seqLenValue),
                        uint32Binding(6, UInt32(inputRowStride)),
                    ]
                    if !usesMPPForStep {
                        bindings.append(uint32Binding(7, UInt32(outputRowStride)))
                    }
                    return bindings
                }(),
                threadgroupMemoryLength: useDirectQuantizedGEMM
                    ? (directGEMM?.threadgroupMemoryLength ?? 0)
                    : ((usesMPPForStep || usesSequenceGEMVForStep) ? 0 : resolved.config.sharedMemoryBytes),
                sync: .bufferBarrier,
                mode: mode,
                sequenceLengthPolicy: mode == .batch
                    ? (usesMPPForStep && !useDirectQuantizedGEMM
                        ? .bindAndAdjustGridHeightTiled(index: 5, tileHeight: 64)
                        : (sequenceGEMVTile.map { .bindAndAdjustGridHeightTiled(index: 5, tileHeight: $0) }
                           ?? .bindAndAdjustGridHeight(index: 5)))
                    : .none,
                positionBufferIndex: nil,
                perPositionStrides: perPositionStrides,
                metadata: .init(
                    kernelName: selectedKernelName,
                    entryIndex: entry.index,
                    weightTensorName: weightTensorName,
                    bufferAccessPattern: gemmPattern
                ),
                tileVariants: mppTileVariants
            )]
        } else {
            let frag = entry.fragment
            // Projection-type fragments decompose in prefill
            if let batched = frag as? BatchedProjection {
                return try buildBatchedProjectionPrefillSteps(
                    batched, entry: entry, weightResolver: weightResolver
                )
            }
            if let batch = frag as? BatchedFragment {
                return try buildBatchedFragmentPrefillSteps(
                    batch, entry: entry
                )
            }
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
                            InternalLog.error("[Compiler] missing prefill kernel '\(name)'; related compiled kernels: \(relatedKernelNames)")
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
                        fallbackReason: descriptor.fallbackReason,
                        prefillGEMM: nil
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
            if let recurrence = frag as? SSMRecurrenceFragment,
               result.outputIsHidden == false,
               result.resetsProjectionIndex {
                pendingRecurrentBlockOutputProjection = PendingRecurrentBlockOutputProjection(
                    compositeID: entry.compositeID,
                    layerIndex: entry.layerIndex,
                    recurrentGroupCount: max(recurrence.groupCount, 1),
                    recurrentOutputDimension: recurrence.headCount * recurrence.valueHeadDimension
                )
            } else {
                pendingRecurrentBlockOutputProjection = nil
            }
            return annotate(result.steps, entryIndex: entry.index, layerIndex: entry.layerIndex)
        }
    }

    private mutating func buildRecurrentBlockPartialOutputProjectionSteps(
        _ linear: LinearFragment,
        entry: DispatchEntry,
        weightResolver: WeightResolver
    ) throws -> [MetalPrefillStep]? {
        guard let pending = pendingRecurrentBlockOutputProjection else {
            return nil
        }
        guard linear.field == "out_proj", linear.isOutput else {
            pendingRecurrentBlockOutputProjection = nil
            return nil
        }

        guard Self.bf16RecurrentBlockPartialProjectionEnabled else {
            return nil
        }
        guard pending.recurrentGroupCount > 1 else {
            throw MetalCompilerError.deviceSetupFailed(
                "Recurrent block partial projection requires more than one recurrent partition"
            )
        }
        guard entry.compositeID == pending.compositeID,
              entry.layerIndex == pending.layerIndex else {
            throw MetalCompilerError.deviceSetupFailed(
                "Recurrent block partial projection requires adjacent recurrence/output projection in the same layer"
            )
        }
        guard buffers.bufferPrecision.isPrefillSequencePrecision else {
            throw MetalCompilerError.deviceSetupFailed(
                "Recurrent block partial projection requires prefill sequence precision"
            )
        }
        guard !routingState.lastOutputIsHidden,
              routingState.currentInputOffset == 0 else {
            throw MetalCompilerError.deviceSetupFailed(
                "Recurrent block partial projection requires recurrence output in scratch slot 0"
            )
        }
        guard linear.inputDimension == pending.recurrentOutputDimension else {
            throw MetalCompilerError.deviceSetupFailed(
                "Recurrent block partial projection input dimension \(linear.inputDimension) does not match recurrence output \(pending.recurrentOutputDimension)"
            )
        }
        guard linear.outputDimension <= hiddenSize else {
            throw MetalCompilerError.deviceSetupFailed(
                "Recurrent block partial projection does not support output-head projections"
            )
        }

        let descriptor = resolveProjectionWeightDescriptor(role: linear.field, entry: entry)
        guard descriptor.schemeIdentifier == .bf16RowMajor,
              !descriptor.schemeIdentifier.isWeightQuantized else {
            throw MetalCompilerError.deviceSetupFailed(
                "Recurrent block partial projection requires BF16 row-major weights"
            )
        }

        let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength
        let availableScratchSlots = buffers.scratch.length / max(scratchSlotSize, 1)
        let availablePartialSlots = max(availableScratchSlots - 1, 0)
        let partialPartitionCount = Self.largestDivisor(
            of: pending.recurrentGroupCount,
            notExceeding: availablePartialSlots
        )
        guard partialPartitionCount > 1 else {
            throw MetalCompilerError.deviceSetupFailed(
                "Recurrent block partial projection requires a scratch-compatible recurrent partition count"
            )
        }
        guard pending.recurrentOutputDimension % partialPartitionCount == 0 else {
            throw MetalCompilerError.deviceSetupFailed(
                "Recurrent output dimension \(pending.recurrentOutputDimension) is not divisible by partial partition count \(partialPartitionCount)"
            )
        }
        let partitionInputDimension = pending.recurrentOutputDimension / partialPartitionCount

        let projectionKernelName = "recurrent_block_partial_projection_seq_bf16_f32"
        let reduceKernelName = "recurrent_block_partial_reduce_seq_f32"
        guard let projectionPipeline = planBuildContext.pipelineCache[projectionKernelName] else {
            throw MetalCompilerError.kernelNotFound(projectionKernelName)
        }
        guard let reducePipeline = planBuildContext.pipelineCache[reduceKernelName] else {
            throw MetalCompilerError.kernelNotFound(reduceKernelName)
        }

        let (weightBuffer, weightOffset) = weightResolver.resolve(role: linear.field)
        let weightTensorName = entry.parameterBindings.first(where: { $0.role == linear.field })?.tensorName
        let inputRowStride = slotDimension
        let partialRowStride = slotDimension
        let outputRowStride = (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize

        let projectionSimdWidth = max(projectionPipeline.threadExecutionWidth, 1)
        let projectionThreads = min(
            projectionSimdWidth * Self.decodeEquivalentSequenceRowsPerThreadgroup,
            projectionPipeline.maxTotalThreadsPerThreadgroup
        )
        let projectionRowsPerThreadgroup = max(1, projectionThreads / projectionSimdWidth)
        let projectionStep = MetalPrefillStep(
            pipeline: projectionPipeline,
            gridSize: MTLSize(
                width: (linear.outputDimension + projectionRowsPerThreadgroup - 1) / projectionRowsPerThreadgroup,
                height: maximumSequenceLength,
                depth: partialPartitionCount
            ),
            threadgroupSize: MTLSize(width: projectionThreads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.scratch, 0),
                (1, weightBuffer, weightOffset),
                (2, buffers.scratch, scratchSlotSize),
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(partitionInputDimension)),
                uint32Binding(4, UInt32(linear.outputDimension)),
                uint32Binding(5, UInt32(partialPartitionCount)),
                uint32Binding(6, UInt32(maximumSequenceLength)),
                uint32Binding(7, UInt32(inputRowStride)),
                uint32Binding(8, UInt32(partialRowStride)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 6),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: projectionKernelName,
                entryIndex: entry.index,
                weightTensorName: weightTensorName,
                bufferAccessPattern: .init(reads: [0, 1], writes: [2])
            )
        )

        let reduceThreads = min(256, reducePipeline.maxTotalThreadsPerThreadgroup)
        let reduceStep = MetalPrefillStep(
            pipeline: reducePipeline,
            gridSize: MTLSize(
                width: (linear.outputDimension + reduceThreads - 1) / reduceThreads,
                height: maximumSequenceLength,
                depth: 1
            ),
            threadgroupSize: MTLSize(width: reduceThreads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.scratch, scratchSlotSize),
                (1, buffers.hidden, 0),
            ],
            bytesBindings: [
                uint32Binding(2, UInt32(partialPartitionCount)),
                uint32Binding(3, UInt32(linear.outputDimension)),
                uint32Binding(4, UInt32(maximumSequenceLength)),
                uint32Binding(5, UInt32(partialRowStride)),
                uint32Binding(6, UInt32(outputRowStride)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: reduceKernelName,
                entryIndex: entry.index,
                weightTensorName: weightTensorName,
                bufferAccessPattern: .init(reads: [0], writes: [1])
            )
        )

        recordProjectionQuantization(
            entry: entry,
            descriptor: descriptor,
            mode: .batch,
            inputRowStride: inputRowStride,
            inputDimension: linear.inputDimension,
            outputDimension: linear.outputDimension,
            outputRowStride: outputRowStride,
            selectedKernelName: projectionKernelName,
            usesMPPForStep: false,
            usesSequenceGEMVForStep: true,
            sequenceTileHeight: nil,
            tileVariantHeights: []
        )

        routingState.lastOutputIsHidden = true
        routingState.currentInputOffset = 0
        routingState.projectionIndex = 1
        pendingRecurrentBlockOutputProjection = nil
        refreshCompositeInputSource()

        return [projectionStep, reduceStep]
    }

    // MARK: - Projection-type fragment prefill decomposition

    private mutating func buildBatchedProjectionPrefillSteps(
        _ batched: BatchedProjection,
        entry: DispatchEntry,
        weightResolver: WeightResolver
    ) throws -> [MetalPrefillStep] {
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
        let inputRowStride = inputBuffer === buffers.hidden
            ? (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize
            : slotDimension
        let firstOutputSlot = firstNonAliasingScratchOutputSlot(
            inputBuffer: inputBuffer,
            inputOffset: inputOffset,
            scratchSlotSize: scratchSlotSize
        )

        // Try direct quantized GEMM: single dispatch for all projections
        let firstDescriptor = resolveProjectionWeightDescriptor(
            role: batched.projections[0].field, entry: entry
        )

        if let sequenceStep = try buildDecodeEquivalentBatchedSequenceGEMVStep(
            batched: batched,
            entry: entry,
            weightResolver: weightResolver,
            firstDescriptor: firstDescriptor,
            inputBuffer: inputBuffer,
            inputOffset: inputOffset,
            inputRowStride: inputRowStride,
            scratchSlotSize: scratchSlotSize
        ) {
            return annotate([sequenceStep], entryIndex: entry.index, layerIndex: entry.layerIndex)
        }

        // BF16 / FP16 / FP32 dense weights → batched MPP GEMM (matmul2d-based).
        // This path runs a single MPP kernel that processes all N projections
        // sharing the same input A, removing the barriers and dispatch-encode
        // cost that the per-projection fallback would incur.
        if let mppStep = try buildBatchedMPPGEMMStep(
            batched: batched,
            entry: entry,
            weightResolver: weightResolver,
            firstDescriptor: firstDescriptor,
            inputBuffer: inputBuffer,
            inputOffset: inputOffset,
            inputRowStride: inputRowStride,
            scratchSlotSize: scratchSlotSize
        ) {
            return annotate([mppStep], entryIndex: entry.index, layerIndex: entry.layerIndex)
        }

        if let batchedGEMM = resolveBatchedQuantizedGEMM(
               for: firstDescriptor.schemeIdentifier, count: batched.projections.count
           ),
           let batchedPipeline = planBuildContext.pipelineCache[batchedGEMM.kernelName] {

            let count = batched.projections.count

            // Buffer layout: input(0), weight0..N-1(1..N), output0..N-1(N+1..2N)
            var bufferBindings: [(Int, MTLBuffer, Int)] = [(0, inputBuffer, inputOffset)]
            var totalOutputDim = 0
            var lastOutputOffset = routingState.currentInputOffset

            for (i, projection) in batched.projections.enumerated() {
                let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
                bufferBindings.append((1 + i, weightBuffer, weightOffset))

                let outputOffset = (firstOutputSlot + i) * scratchSlotSize
                lastOutputOffset = outputOffset
                bufferBindings.append((1 + count + i, buffers.scratch, outputOffset))

                totalOutputDim += projection.outputDimension
            }
            routingState.projectionIndex = firstOutputSlot + count - 1

            // Bytes layout: inputDim(2N+1), outDim0..N-1(2N+2..3N+1), seqLen(3N+2), rowStride(3N+3)
            let dimBase = 1 + 2 * count
            var bytesBindings: [(index: Int, value: [UInt8])] = [
                uint32Binding(dimBase, UInt32(batched.projections[0].inputDimension)),
            ]
            for (i, projection) in batched.projections.enumerated() {
                bytesBindings.append(uint32Binding(dimBase + 1 + i, UInt32(projection.outputDimension)))
            }
            let seqLenIndex = dimBase + 1 + count
            let outputRowStride = slotDimension
            bytesBindings.append(uint32Binding(seqLenIndex, UInt32(maximumSequenceLength)))
            bytesBindings.append(uint32Binding(seqLenIndex + 1, UInt32(inputRowStride)))
            bytesBindings.append(uint32Binding(seqLenIndex + 2, UInt32(outputRowStride)))

            // Grid covers all output rows across all projections
            let simdWidth = max(batchedPipeline.threadExecutionWidth, 1)
            let rowsPerThreadgroup = 2
            let threads = min(
                simdWidth * rowsPerThreadgroup,
                batchedPipeline.maxTotalThreadsPerThreadgroup
            )
            let gridSize = MTLSize(
                width: (totalOutputDim + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                height: maximumSequenceLength,
                depth: 1
            )
            let threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)

            // Threadgroup memory: input tile for quantized block unpacking
            let threadgroupMemoryLength = batchedGEMM.threadgroupMemoryLength

            // Buffer access pattern: reads input + all weights, writes all outputs
            let readIndices = Set(0...count)
            let writeIndices = Set((count + 1)...(2 * count))

            let step = MetalPrefillStep(
                pipeline: batchedPipeline,
                gridSize: gridSize,
                threadgroupSize: threadgroupSize,
                bufferBindings: bufferBindings,
                bytesBindings: bytesBindings,
                threadgroupMemoryLength: threadgroupMemoryLength,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: seqLenIndex),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: batchedGEMM.kernelName,
                    entryIndex: entry.index,
                    weightTensorName: Self.batchedWeightTensorName(for: batched, entry: entry),
                    bufferAccessPattern: .init(reads: readIndices, writes: writeIndices)
                )
            )

            for projection in batched.projections {
                let descriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)
                recordProjectionQuantization(
                    entry: entry,
                    descriptor: descriptor,
                    mode: .batch,
                    inputRowStride: inputRowStride,
                    inputDimension: projection.inputDimension,
                    outputDimension: projection.outputDimension,
                    outputRowStride: outputRowStride,
                    selectedKernelName: batchedGEMM.kernelName,
                    usesMPPForStep: false,
                    usesSequenceGEMVForStep: false,
                    projectionCount: count
                )
            }

            routingState.lastOutputIsHidden = false
            routingState.currentInputOffset = lastOutputOffset
            return annotate([step], entryIndex: entry.index, layerIndex: entry.layerIndex)
        }

        // Fallback: expand to individual projection steps
        var steps: [MetalPrefillStep] = []
        steps.reserveCapacity(batched.projections.count)
        var lastOutputOffset = routingState.currentInputOffset
        for (projectionIndex, projection) in batched.projections.enumerated() {
            let projInputRowStride = inputBuffer === buffers.hidden
                ? (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize
                : slotDimension
            let resolved = try resolveDispatch(
                DispatchEntry(
                    index: entry.index,
                    fragment: LinearFragment(
                        field: projection.field,
                        inputDimension: projection.inputDimension,
                        outputDimension: projection.outputDimension
                    ),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex
                )
            )
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
            let weightTensorName = entry.parameterBindings.first(where: { $0.role == projection.field })?.tensorName
            let quantizationDescriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)
            let outputOffset = (firstOutputSlot + projectionIndex) * scratchSlotSize
            let outputRowStride = slotDimension
            lastOutputOffset = outputOffset
            routingState.projectionIndex = firstOutputSlot + projectionIndex

            // Prefer direct quantized GEMM (dequant in registers) when available.
            // Falls back to dequant→AMX when no direct kernel exists.
            let directGEMM = resolveDirectQuantizedGEMM(for: quantizationDescriptor.schemeIdentifier)
            let useDirectQuantizedGEMM = directGEMM.flatMap {
                planBuildContext.pipelineCache[$0.kernelName]
            } != nil
            let sequenceGEMVSelection = decodeEquivalentSequenceGEMVKernelName(
                for: quantizationDescriptor
            )
            let usesSequenceGEMVForStep = sequenceGEMVSelection != nil

            let canDequantForAMX = quantizationDescriptor.schemeIdentifier.isWeightQuantized
                && buffers.dequantScratch != nil
                && dequantKernelName(for: quantizationDescriptor.schemeIdentifier) != nil
            let usesMPPForStep = usesMPP
                && !usesSequenceGEMVForStep
                && projInputRowStride == projection.inputDimension
                && outputRowStride == projection.outputDimension
                && (!quantizationDescriptor.schemeIdentifier.isWeightQuantized || canDequantForAMX)
            let usesDequantScratchForStep = !useDirectQuantizedGEMM
                && canDequantForAMX
                && !usesSequenceGEMVForStep

            if usesDequantScratchForStep,
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

            let selectedPipeline: MTLComputePipelineState
            let selectedKernelName: String
            if let sequenceGEMVSelection {
                guard let sequencePipeline = planBuildContext.pipelineCache[sequenceGEMVSelection.kernelName] else {
                    throw MetalCompilerError.kernelNotFound(sequenceGEMVSelection.kernelName)
                }
                selectedPipeline = sequencePipeline
                selectedKernelName = sequenceGEMVSelection.kernelName
            } else if useDirectQuantizedGEMM,
               let resolved = directGEMM,
               let directPipeline = planBuildContext.pipelineCache[resolved.kernelName] {
                selectedPipeline = directPipeline
                selectedKernelName = resolved.kernelName
            } else if canDequantForAMX && usesMPPForStep,
               let mppPipeline = planBuildContext.pipelineCache["gemm_bf16_f32s"] {
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

            let gemmWeightBuffer: MTLBuffer
            let gemmWeightOffset: Int
            if useDirectQuantizedGEMM {
                gemmWeightBuffer = weightBuffer
                gemmWeightOffset = weightOffset
            } else if usesDequantScratchForStep, let dequantScratch = buffers.dequantScratch {
                gemmWeightBuffer = dequantScratch
                gemmWeightOffset = 0
            } else {
                gemmWeightBuffer = weightBuffer
                gemmWeightOffset = weightOffset
            }

            let gridSize: MTLSize
            let threadgroupSize: MTLSize
            if usesSequenceGEMVForStep {
                gridSize = MTLSize(
                    width: resolved.config.grid.width,
                    height: maximumSequenceLength,
                    depth: 1
                )
                threadgroupSize = resolved.config.threadgroup
            } else if usesMPPForStep && !useDirectQuantizedGEMM {
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
            let batchedMPPTileVariants: [PrefillTileVariant]
            if usesMPPForStep && !useDirectQuantizedGEMM {
                batchedMPPTileVariants = makeMPPTileVariants(
                    baseKernelName: selectedKernelName,
                    gridWidth: gridSize.width,
                    maxSequenceLength: maximumSequenceLength,
                    threadgroupSize: threadgroupSize,
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier)
            } else {
                batchedMPPTileVariants = []
            }
            recordProjectionQuantization(
                entry: entry,
                descriptor: quantizationDescriptor,
                mode: .batch,
                inputRowStride: projInputRowStride,
                inputDimension: projection.inputDimension,
                outputDimension: projection.outputDimension,
                outputRowStride: outputRowStride,
                selectedKernelName: selectedKernelName,
                usesMPPForStep: usesMPPForStep,
                usesSequenceGEMVForStep: usesSequenceGEMVForStep,
                sequenceTileHeight: usesMPPForStep && !useDirectQuantizedGEMM ? 64 : nil,
                tileVariantHeights: batchedMPPTileVariants.map(\.tileHeight)
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
                    bytesBindings: {
                        var bindings = [
                            uint32Binding(3, UInt32(projection.inputDimension)),
                            uint32Binding(4, UInt32(projection.outputDimension)),
                            uint32Binding(5, UInt32(maximumSequenceLength)),
                            uint32Binding(6, UInt32(projInputRowStride)),
                        ]
                        if !usesMPPForStep {
                            bindings.append(uint32Binding(7, UInt32(outputRowStride)))
                        }
                        return bindings
                    }(),
                    threadgroupMemoryLength: useDirectQuantizedGEMM
                        ? (directGEMM?.threadgroupMemoryLength ?? 0)
                        : ((usesMPPForStep || usesSequenceGEMVForStep) ? 0 : resolved.config.sharedMemoryBytes),
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: usesMPPForStep && !useDirectQuantizedGEMM
                        ? .bindAndAdjustGridHeightTiled(index: 5, tileHeight: 64)
                        : .bindAndAdjustGridHeight(index: 5),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: selectedKernelName,
                        entryIndex: entry.index,
                        weightTensorName: weightTensorName,
                        bufferAccessPattern: gemmPattern
                    ),
                    tileVariants: batchedMPPTileVariants
                )
            )
        }

        routingState.lastOutputIsHidden = false
        routingState.currentInputOffset = lastOutputOffset
        return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)
    }

    private func firstNonAliasingScratchOutputSlot(
        inputBuffer: MTLBuffer,
        inputOffset: Int,
        scratchSlotSize: Int
    ) -> Int {
        let nextOutputSlot = routingState.projectionIndex + 1
        guard inputBuffer === buffers.scratch, scratchSlotSize > 0 else {
            return nextOutputSlot
        }
        return max(nextOutputSlot, inputOffset / scratchSlotSize + 1)
    }

    private mutating func buildDecodeEquivalentBatchedSequenceGEMVStep(
        batched: BatchedProjection,
        entry: DispatchEntry,
        weightResolver: WeightResolver,
        firstDescriptor: ProjectionWeightDescriptor,
        inputBuffer: MTLBuffer,
        inputOffset: Int,
        inputRowStride: Int,
        scratchSlotSize: Int
    ) throws -> MetalPrefillStep? {
        let count = batched.projections.count
        guard let selection = decodeEquivalentBatchedSequenceGEMVKernelName(
            for: firstDescriptor,
            count: count
        ) else {
            return nil
        }
        let kernelName = selection.kernelName
        guard let pipeline = planBuildContext.pipelineCache[kernelName] else {
            throw MetalCompilerError.kernelNotFound(kernelName)
        }

        var bufferBindings: [(Int, MTLBuffer, Int)] = [(0, inputBuffer, inputOffset)]
        var totalOutputDim = 0
        var lastOutputOffset = routingState.currentInputOffset
        let firstOutputSlot = firstNonAliasingScratchOutputSlot(
            inputBuffer: inputBuffer,
            inputOffset: inputOffset,
            scratchSlotSize: scratchSlotSize
        )

        for (i, projection) in batched.projections.enumerated() {
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
            bufferBindings.append((1 + i, weightBuffer, weightOffset))
        }

        for (i, projection) in batched.projections.enumerated() {
            let outputOffset = (firstOutputSlot + i) * scratchSlotSize
            lastOutputOffset = outputOffset
            bufferBindings.append((1 + count + i, buffers.scratch, outputOffset))
            totalOutputDim += projection.outputDimension
        }
        routingState.projectionIndex = firstOutputSlot + count - 1

        let dimBase = 1 + 2 * count
        var bytesBindings: [(index: Int, value: [UInt8])] = [
            uint32Binding(dimBase, UInt32(batched.inputDimension)),
        ]
        for (i, projection) in batched.projections.enumerated() {
            bytesBindings.append(uint32Binding(dimBase + 1 + i, UInt32(projection.outputDimension)))
        }
        let seqLenIndex = dimBase + 1 + count
        bytesBindings.append(uint32Binding(seqLenIndex, UInt32(maximumSequenceLength)))
        bytesBindings.append(uint32Binding(seqLenIndex + 1, UInt32(inputRowStride)))
        bytesBindings.append(uint32Binding(seqLenIndex + 2, UInt32(slotDimension)))

        let usesTiledKernel = selection.isTiled
        let shape = usesTiledKernel
            ? try decodeEquivalentSequenceThreadShape(
                pipeline: pipeline,
                requestedSequenceTile: selection.sequenceTile
            )
            : nil
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let threads = shape?.threadgroupSize.width
            ?? min(
                simdWidth * Self.decodeEquivalentSequenceRowsPerThreadgroup,
                pipeline.maxTotalThreadsPerThreadgroup
            )
        let rowsPerThreadgroup = shape?.rowsPerThreadgroup ?? max(1, threads / simdWidth)
        let gridHeight = shape.map { (maximumSequenceLength + $0.sequenceTile - 1) / $0.sequenceTile }
            ?? maximumSequenceLength
        let gridSize = MTLSize(
            width: (totalOutputDim + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: gridHeight,
            depth: 1
        )
        let threadgroupSize = shape?.threadgroupSize ?? MTLSize(width: threads, height: 1, depth: 1)
        let readIndices = Set(0...count)
        let writeIndices = Set((count + 1)...(2 * count))

        for projection in batched.projections {
            let descriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)
            recordProjectionQuantization(
                entry: entry,
                descriptor: descriptor,
                mode: .batch,
                inputRowStride: inputRowStride,
                inputDimension: projection.inputDimension,
                outputDimension: projection.outputDimension,
                outputRowStride: slotDimension,
                selectedKernelName: kernelName,
                usesMPPForStep: false,
                usesSequenceGEMVForStep: true,
                sequenceTileHeight: shape?.sequenceTile,
                projectionCount: count
            )
        }

        routingState.lastOutputIsHidden = false
        routingState.currentInputOffset = lastOutputOffset
        return MetalPrefillStep(
            pipeline: pipeline,
            gridSize: gridSize,
            threadgroupSize: threadgroupSize,
            bufferBindings: bufferBindings,
            bytesBindings: bytesBindings,
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: shape.map {
                .bindAndAdjustGridHeightTiled(index: seqLenIndex, tileHeight: $0.sequenceTile)
            } ?? .bindAndAdjustGridHeight(index: seqLenIndex),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: Self.batchedWeightTensorName(for: batched, entry: entry),
                bufferAccessPattern: .init(reads: readIndices, writes: writeIndices)
            )
        )
    }

    private mutating func buildBatchedFragmentPrefillSteps(
        _ batch: BatchedFragment,
        entry: DispatchEntry
    ) throws -> [MetalPrefillStep] {
        // Fast path: batched QK norm → single dispatch
        if batch.fragments.count == 2,
           let qNorm = batch.fragments[0] as? QKNormFragment,
           let kNorm = batch.fragments[1] as? QKNormFragment,
           let batchedStep = try makeBatchedQKNormStep(
               qNorm: qNorm, kNorm: kNorm, entry: entry) {
            return annotate([batchedStep], entryIndex: entry.index, layerIndex: entry.layerIndex)
        }
        // Fallback: decompose to individual fragments
        var steps: [MetalPrefillStep] = []
        for (i, frag) in batch.fragments.enumerated() {
            let singleEntry = DispatchEntry(
                index: entry.index + i,
                fragment: frag,
                parameterBindings: entry.parameterBindings,
                layerIndex: entry.layerIndex
            )
            let fragSteps = try buildSteps(for: singleEntry)
            steps.append(contentsOf: fragSteps)
        }
        return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)
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
                fragment: CopyFragment(dimension: dimension),
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

    /// Build a single batched QK norm step for prefill.
    ///
    /// Merges Q and K per-head RMS norm into a single dispatch.
    /// Grid: (qHeadCount + kHeadCount, sequenceLength, 1)
    ///
    /// Buffer layout: [0]=qData, [1]=kData, [2]=qWeight, [3]=kWeight
    /// Bytes: [4]=qHeadCount, [5]=kHeadCount, [6]=headDimension, [7]=epsilon,
    ///        [8]=weightBias, [9]=sequenceLength, [10]=qTotalDim, [11]=kTotalDim
    private func makeBatchedQKNormStep(
        qNorm: QKNormFragment,
        kNorm: QKNormFragment,
        entry: DispatchEntry
    ) throws -> MetalPrefillStep? {
        let kernelName = fallbackWeightFormat.isBFloat16
            ? "batched_qk_rms_norm_seq_bf16_f32"
            : "batched_qk_rms_norm_seq_f32"
        guard let pipeline = planBuildContext.pipelineCache[kernelName] else { return nil }

        let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength

        // Resolve Q and K weights
        let qWeightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )
        let (qWeightBuffer, qWeightOffset) = qWeightResolver.resolve(role: qNorm.weightRole)
        let (kWeightBuffer, kWeightOffset) = qWeightResolver.resolve(role: kNorm.weightRole)

        let totalHeadCount = qNorm.headCount + kNorm.headCount
        let threads = min(256, pipeline.maxTotalThreadsPerThreadgroup)

        let qTotalDimension = qNorm.headCount * qNorm.headDimension
        let kTotalDimension = kNorm.headCount * kNorm.headDimension

        // Reads qData[0] + kData[1] + qWeight[2] + kWeight[3], writes qData[0] + kData[1]
        let pattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1, 2, 3], writes: [0, 1])
        return MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: totalHeadCount, height: maximumSequenceLength, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.scratch, qNorm.scratchSlotIndex * scratchSlotSize),
                (1, buffers.scratch, kNorm.scratchSlotIndex * scratchSlotSize),
                (2, qWeightBuffer, qWeightOffset),
                (3, kWeightBuffer, kWeightOffset),
            ],
            bytesBindings: [
                uint32Binding(4, UInt32(qNorm.headCount)),
                uint32Binding(5, UInt32(kNorm.headCount)),
                uint32Binding(6, UInt32(qNorm.headDimension)),
                floatBinding(7, qNorm.epsilon),
                floatBinding(8, qNorm.weightBias),
                uint32Binding(9, UInt32(maximumSequenceLength)),
                uint32Binding(10, UInt32(qTotalDimension)),
                uint32Binding(11, UInt32(kTotalDimension)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 9),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: entry.parameterBindings.first(where: { $0.role == qNorm.weightRole })?.tensorName,
                bufferAccessPattern: pattern
            )
        )
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
        outputDimension: Int,
        outputRowStride: Int,
        selectedKernelName: String,
        usesMPPForStep: Bool,
        usesSequenceGEMVForStep: Bool,
        sequenceTileHeight: Int? = nil,
        tileVariantHeights: [Int] = [],
        projectionCount: Int = 1
    ) {
        let fallbackReason = resolveProjectionFallbackReason(
            descriptor: descriptor,
            mode: mode,
            inputRowStride: inputRowStride,
            inputDimension: inputDimension,
            outputRowStride: outputRowStride,
            outputDimension: outputDimension,
            usesMPPForStep: usesMPPForStep,
            usesSequenceGEMVForStep: usesSequenceGEMVForStep
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
                fallbackReason: descriptor.fallbackReason ?? fallbackReason,
                prefillGEMM: MetalPrefillGEMMDiagnostics(
                    selectedKernelName: selectedKernelName,
                    inputDimension: inputDimension,
                    outputDimension: outputDimension,
                    inputRowStride: inputRowStride,
                    outputRowStride: outputRowStride,
                    maximumSequenceLength: maximumSequenceLength,
                    sequenceTileHeight: sequenceTileHeight,
                    tileVariantHeights: tileVariantHeights,
                    projectionCount: projectionCount
                )
            )
        )
    }

    /// Build tile-size variants for an MPP GEMM prefill step.
    ///
    /// Returns one `PrefillTileVariant` per emitted tile size (16/32/64) that
    /// is present in the pipeline cache. Returns an empty array when the
    /// variant kernels are not available (e.g. MPP library compile failed, or
    /// the kernel name is a direct-quantized GEMM that does not emit variants),
    /// in which case the caller uses the base descriptor unconditionally.
    ///
    /// Each variant's gridSize uses `(maxSeqLen + tileSize - 1) / tileSize`
    /// for the tile dimension; the runtime further narrows the grid height
    /// via `resolvedGridSize` using the actual sequence length.
    private func makeMPPTileVariants(
        baseKernelName: String,
        gridWidth: Int,
        maxSequenceLength: Int,
        threadgroupSize: MTLSize,
        threadgroupMemoryLength: Int,
        sync: SynchronizationKind
    ) -> [PrefillTileVariant] {
        var variants: [PrefillTileVariant] = []
        variants.reserveCapacity(MetalSourceGenerator.mppGEMMTileSizes.count)
        for tileSize in MetalSourceGenerator.mppGEMMTileSizes {
            let variantName = MetalSourceGenerator.mppGEMMVariantName(
                baseName: baseKernelName, tileSize: tileSize)
            guard let variantPipeline = planBuildContext.pipelineCache[variantName] else {
                continue
            }
            let paddedHeight = ((maxSequenceLength + tileSize - 1) / tileSize)
            let variantDescriptor = MetalDispatchDescriptor(
                pipeline: variantPipeline,
                gridSize: MTLSize(width: gridWidth, height: paddedHeight, depth: 1),
                threadgroupSize: threadgroupSize,
                threadgroupMemoryLength: threadgroupMemoryLength,
                barrierPolicy: MetalBarrierPolicy(sync))
            variants.append(PrefillTileVariant(
                tileHeight: tileSize,
                descriptor: variantDescriptor))
        }
        return variants
    }

    /// Build a single batched MPP GEMM step for BF16/FP16/FP32 dense weights.
    ///
    /// Returns nil when the pipeline is not available or the weight format is
    /// not dense (quantized weights go through `buildBatchedProjectionPrefillSteps`'s
    /// Q4 path or the per-projection fallback).
    private mutating func buildBatchedMPPGEMMStep(
        batched: BatchedProjection,
        entry: DispatchEntry,
        weightResolver: WeightResolver,
        firstDescriptor: ProjectionWeightDescriptor,
        inputBuffer: MTLBuffer,
        inputOffset: Int,
        inputRowStride: Int,
        scratchSlotSize: Int
    ) throws -> MetalPrefillStep? {
        let count = batched.projections.count
        guard count >= 2 else { return nil }

        // Only handle dense weight schemes here.
        let scheme = firstDescriptor.schemeIdentifier
        let kernelName: String
        switch scheme {
        case .bf16RowMajor:
            kernelName = "batched_gemm_bf16_f32s_\(count)"
        case .fp16RowMajor:
            kernelName = "batched_gemm_f16_f32s_\(count)"
        case .fp32RowMajor:
            kernelName = "batched_gemm_f32_f32s_\(count)"
        default:
            return nil
        }

        // Every projection's output dimension must be a multiple of N_TILE=32.
        // If any projection violates this, fall through to the per-projection
        // path so the edge handling stays correct.
        let nTile = 32
        for projection in batched.projections {
            if projection.outputDimension % nTile != 0 { return nil }
        }

        // All projections must share the same input dimension (shared A).
        let sharedInputDim = batched.projections[0].inputDimension
        for projection in batched.projections where projection.inputDimension != sharedInputDim {
            return nil
        }

        // Input row stride must match input dimension for MPP tensor_inline.
        guard inputRowStride == sharedInputDim else { return nil }

        let outputRowStride = scratchSlotSize / max(maximumSequenceLength, 1) / scratchElementSize
        for projection in batched.projections where projection.outputDimension != outputRowStride {
            return nil
        }

        guard let pipeline = planBuildContext.pipelineCache[kernelName] else {
            return nil
        }

        // Build buffer bindings: input(0), weight0..N-1(1..N), output0..N-1(N+1..2N)
        var bufferBindings: [(Int, MTLBuffer, Int)] = [(0, inputBuffer, inputOffset)]
        var lastOutputOffset = routingState.currentInputOffset
        var totalNTiles = 0
        let firstOutputSlot = firstNonAliasingScratchOutputSlot(
            inputBuffer: inputBuffer,
            inputOffset: inputOffset,
            scratchSlotSize: scratchSlotSize
        )

        for (i, projection) in batched.projections.enumerated() {
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
            bufferBindings.append((1 + i, weightBuffer, weightOffset))

            let outputOffset = (firstOutputSlot + i) * scratchSlotSize
            lastOutputOffset = outputOffset
            bufferBindings.append((1 + count + i, buffers.scratch, outputOffset))

            totalNTiles += projection.outputDimension / nTile
        }
        routingState.projectionIndex = firstOutputSlot + count - 1

        // Bytes layout: inputDim(2N+1), outDim0..N-1(2N+2..3N+1), seqLen(3N+2), rowStride(3N+3)
        let dimBase = 1 + 2 * count
        var bytesBindings: [(index: Int, value: [UInt8])] = [
            uint32Binding(dimBase, UInt32(sharedInputDim)),
        ]
        for (i, projection) in batched.projections.enumerated() {
            bytesBindings.append(uint32Binding(dimBase + 1 + i, UInt32(projection.outputDimension)))
        }
        let seqLenIndex = dimBase + 1 + count
        bytesBindings.append(uint32Binding(seqLenIndex, UInt32(maximumSequenceLength)))
        bytesBindings.append(uint32Binding(seqLenIndex + 1, UInt32(inputRowStride)))

        // Grid:
        //   width  = total N-tiles across all projections (linear mapping)
        //   height = paddedSeqLen / M_TILE (set to maximumSequenceLength here;
        //            runtime will tile down via bindAndAdjustGridHeightTiled)
        //   depth  = 1
        let mTile = 64
        let paddedMaxSeqLen = ((maximumSequenceLength + mTile - 1) / mTile) * mTile
        let gridSize = MTLSize(
            width: totalNTiles,
            height: paddedMaxSeqLen / mTile,
            depth: 1
        )

        // Threadgroup: SIMD_WIDTH * 4 (execution_simdgroups<4>)
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let threads = min(simdWidth * 4, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)

        // Buffer access pattern: reads input + all weights, writes all outputs.
        let readIndices = Set(0...count)
        let writeIndices = Set((count + 1)...(2 * count))

        let batchedTileVariants = makeMPPTileVariants(
            baseKernelName: kernelName,
            gridWidth: totalNTiles,
            maxSequenceLength: maximumSequenceLength,
            threadgroupSize: threadgroupSize,
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier)

        let step = MetalPrefillStep(
            pipeline: pipeline,
            gridSize: gridSize,
            threadgroupSize: threadgroupSize,
            bufferBindings: bufferBindings,
            bytesBindings: bytesBindings,
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeightTiled(index: seqLenIndex, tileHeight: mTile),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: Self.batchedWeightTensorName(for: batched, entry: entry),
                bufferAccessPattern: .init(reads: readIndices, writes: writeIndices)
            ),
            tileVariants: batchedTileVariants
        )

        routingState.lastOutputIsHidden = false
        routingState.currentInputOffset = lastOutputOffset

        // Record quantization classification for each projection so the
        // observability plan reflects the MPP batched kernel choice.
        for projection in batched.projections {
            let descriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)
            recordProjectionQuantization(
                entry: entry,
                descriptor: descriptor,
                mode: .batch,
                inputRowStride: inputRowStride,
                inputDimension: projection.inputDimension,
                outputDimension: projection.outputDimension,
                outputRowStride: outputRowStride,
                selectedKernelName: kernelName,
                usesMPPForStep: true,
                usesSequenceGEMVForStep: false,
                sequenceTileHeight: mTile,
                tileVariantHeights: batchedTileVariants.map(\.tileHeight),
                projectionCount: count
            )
        }

        return step
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
        outputRowStride: Int,
        outputDimension: Int,
        usesMPPForStep: Bool,
        usesSequenceGEMVForStep: Bool
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
        if !usesSequenceGEMVForStep && !usesMPPForStep && outputRowStride != outputDimension {
            return .outputStrideMismatch
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

    private static func batchedWeightTensorName(
        for batched: BatchedProjection,
        entry: DispatchEntry
    ) -> String? {
        let tensorNames = batched.projections.compactMap { projection in
            entry.parameterBindings.first(where: { $0.role == projection.field })?.tensorName
        }
        guard !tensorNames.isEmpty else { return nil }
        return tensorNames.joined(separator: ";")
    }

    private var fallbackSchemeIdentifier: QuantizationSchemeIdentifier {
        fallbackWeightFormat.schemeIdentifier
    }
}

private struct ProjectionWeightDescriptor {
    let tensorName: String?
    let schemeIdentifier: QuantizationSchemeIdentifier
    let layout: STAFWeightLayout
    let usedFallback: Bool
    let fallbackReason: MetalQuantizationFallbackReason?
}

// MARK: - Quantized Dispatch Resolution
//
// Each capability (dequant kernel, direct GEMM, batched GEMM) is declared on
// `QuantizationFormat` itself. These private helpers just forward a scheme
// identifier to the corresponding format — no switch-over-format lives here.
//
// IMPORTANT: direct GEMM kernels MUST be multi-row GEMMs with `sequenceLength`
// and `inputRowStride` parameters. Decode-only GEMV kernels
// (`gemv_q4_g64` / `gemv_q4_g128` / `gemv_q8_g*`) must never be returned here
// — they ignore `gid.y` and silently corrupt non-first positions in prefill.

private func dequantKernelName(for scheme: QuantizationSchemeIdentifier) -> String? {
    QuantizationFormatRegistry.format(for: scheme)?.dequantToBFloatKernelName
}

private func resolveDirectQuantizedGEMM(
    for scheme: QuantizationSchemeIdentifier
) -> DirectQuantizedGEMM? {
    QuantizationFormatRegistry.format(for: scheme)?.directGEMMKernel()
}

private func resolveBatchedQuantizedGEMM(
    for scheme: QuantizationSchemeIdentifier,
    count: Int
) -> DirectQuantizedGEMM? {
    QuantizationFormatRegistry.format(for: scheme)?.batchedGEMMKernel(count: count)
}
