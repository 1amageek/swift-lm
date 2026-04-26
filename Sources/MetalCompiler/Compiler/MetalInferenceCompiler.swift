import Metal
import LMIR

// MARK: - Compiler

/// Compiles a `ModelGraph` into a `MetalCompiledModel`.
///
/// ## Phases
///
/// 1. **IR walk**: traverse the graph, read each MetalComponent's dispatchDeclarations
/// 2. **Fusion pass**: detect adjacent fusable operations via pattern matching
/// 3. **Compile**: build one MTLLibrary from compiler-owned kernel sources
/// 4. **Buffer routing**: assign concrete MTLBuffers and offsets to each dispatch
/// 5. **Dispatch plan**: compute grid/threadgroup, build MetalDispatchStep array
public struct MetalInferenceCompiler: Sendable {
    static let argumentTableBindingIndex = 30

    private let weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride?
    private let decodeBufferPrecisionOverride: BufferPrecision?
    private let bufferAllocator = MetalBufferAllocator()
    private let dispatchStepBuilder = MetalDispatchStepBuilder()
    private let prefillStepBuilder = MetalPrefillStepBuilder()

    private var entryCollector: MetalEntryCollector {
        MetalEntryCollector()
    }

    public init() {
        self.weightAccessPolicyOverride = nil
        self.decodeBufferPrecisionOverride = nil
    }

    public init(decodeBufferPrecisionOverride: BufferPrecision?) {
        self.weightAccessPolicyOverride = nil
        self.decodeBufferPrecisionOverride = decodeBufferPrecisionOverride
    }

    init(
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil,
        decodeBufferPrecisionOverride: BufferPrecision? = nil
    ) {
        self.weightAccessPolicyOverride = weightAccessPolicyOverride
        self.decodeBufferPrecisionOverride = decodeBufferPrecisionOverride
    }

    private func makeCompileContext(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int,
        vocabSize: Int,
        inferencePolicy: InferencePolicy = .default,
        stafWeightStore: STAFWeightStore?,
        device: MTLDevice
    ) -> CompileContext {
        let kernelNameResolver = MetalKernelNameResolver(
            stafWeightStore: stafWeightStore,
            weightAccessPolicyOverride: weightAccessPolicyOverride
        )
        let weightFormat = kernelNameResolver.resolveModelWeightFormat()
        let decodeBufferPrecision = decodeBufferPrecisionOverride
            ?? kernelNameResolver.preferredDecodeBufferPrecision(for: weightFormat)
        return CompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            inferencePolicy: inferencePolicy,
            stafWeightStore: stafWeightStore,
            device: device,
            weightFormat: weightFormat,
            decodeBufferPrecision: decodeBufferPrecision,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver(
                override: weightAccessPolicyOverride
            )
        )
    }

    private func makePlanBuildContext(
        compileContext: CompileContext,
        kernelContext: KernelContext,
        pipelineCache: [String: MTLComputePipelineState],
        quantizationCapabilities: MetalQuantizationCapabilities = .none
    ) -> PlanBuildContext {
        PlanBuildContext(
            compileContext: compileContext,
            kernelContext: kernelContext,
            pipelineCache: pipelineCache,
            quantizationCapabilities: quantizationCapabilities,
            dispatchHeuristics: DispatchHeuristics())
    }

    private func prepareSpecializedWeightStore(
        _ store: STAFWeightStore?,
        for entries: [DispatchEntry],
        device: MTLDevice,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    ) throws -> STAFWeightStore? {
        let builder = STAFSpecializedWeightStoreBuilder(
            device: device,
            accessPolicyResolver: accessPolicyResolver
        )
        return try builder.prepare(store: store, entries: entries)
    }

    private func resolvedPipeline(
        for entry: DispatchEntry,
        using context: PlanBuildContext
    ) throws -> (name: String, pipeline: MTLComputePipelineState) {
        let kernelNameResolver = MetalKernelNameResolver(
            stafWeightStore: context.stafWeightStore,
            weightAccessPolicyOverride: weightAccessPolicyOverride
        )
        let resolvedKernelName = kernelNameResolver.kernelName(
            for: entry,
            kernelContext: context.kernelContext
        )
        guard let pipeline = context.pipelineCache[resolvedKernelName] else {
            let relatedKernelNames = context.pipelineCache.keys
                .filter { $0.contains("embedding_lookup") || $0.contains("gather") }
                .sorted()
            if !relatedKernelNames.isEmpty {
                InternalLog.error("[Prewarm/Compiler] missing kernel '\(resolvedKernelName)'; related compiled kernels: \(relatedKernelNames)")
            } else {
                InternalLog.error("[Prewarm/Compiler] missing kernel '\(resolvedKernelName)'; no embedding-related kernels compiled")
            }
            throw MetalCompilerError.kernelNotFound(resolvedKernelName)
        }
        return (resolvedKernelName, pipeline)
    }

    private func resolvedDispatch(
        for entry: DispatchEntry,
        using context: PlanBuildContext
    ) throws -> (
        name: String,
        pipeline: MTLComputePipelineState,
        config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
    ) {
        let resolved = try resolvedPipeline(for: entry, using: context)
        let dimension = entry.fragment.dispatchDimension
        var config = context.dispatchHeuristics.config(
            for: dimension,
            pipeline: resolved.pipeline,
            roundUp: roundUp(_:to:))
        if entry.fragment is FlashAttentionFragment {
            let simdWidth = max(resolved.pipeline.threadExecutionWidth, 1)
            config.threadgroup = MTLSize(width: min(simdWidth, resolved.pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        }
        return (
            resolved.name,
            resolved.pipeline,
            config
        )
    }

    /// Dump optimized dispatch entries for diagnostic purposes.
    /// Returns a human-readable list of all dispatch entries after optimization.
    public func dumpDispatchEntries(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> String {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: 0,
            vocabSize: 0,
            stafWeightStore: stafWeightStore,
            device: MTLCreateSystemDefaultDevice()!)
        let optimization = entryCollector.collect(using: context, kernelContext: context.decodeKernelContext)
        let formatter = DispatchEntryDiagnosticsFormatter(kernelContext: context.decodeKernelContext)
        return formatter.format(entries: optimization.fusedEntries, unfusedCount: optimization.unfusedCount)
    }

    /// Analyze optimization without Metal compilation.
    /// Returns a report comparing unfused vs optimized dispatch counts.
    public func analyzeOptimization(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> OptimizationReport {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: 0,
            vocabSize: 0,
            stafWeightStore: stafWeightStore,
            device: MTLCreateSystemDefaultDevice()!)
        let optimization = entryCollector.collect(using: context, kernelContext: context.decodeKernelContext)
        let reportBuilder = OptimizationReportBuilder()
        return reportBuilder.makeReport(
            unfusedEntries: optimization.walkContext.entries,
            optimizedEntries: optimization.fusedEntries)
    }

    public func analyzeDecodeProjectionCosts(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        stafWeightStore: STAFWeightStore? = nil,
        device: MTLDevice
    ) throws -> DecodeProjectionCostReport {
        let initialContext = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            stafWeightStore: stafWeightStore,
            device: device
        )
        let initialOptimization = entryCollector.collect(
            using: initialContext,
            kernelContext: initialContext.decodeKernelContext
        )
        let specializedWeightStore = try prepareSpecializedWeightStore(
            initialContext.stafWeightStore,
            for: initialOptimization.fusedEntries,
            device: initialContext.device,
            accessPolicyResolver: initialContext.accessPolicyResolver
        )
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            stafWeightStore: specializedWeightStore,
            device: device
        )
        let optimization = entryCollector.collect(
            using: context,
            kernelContext: context.decodeKernelContext
        )
        let kernelNameResolver = MetalKernelNameResolver(
            stafWeightStore: specializedWeightStore,
            weightAccessPolicyOverride: weightAccessPolicyOverride
        )
        let reportBuilder = DecodeProjectionCostReportBuilder(
            decodeBufferPrecision: context.decodeBufferPrecision,
            kernelContext: context.decodeKernelContext,
            stafWeightStore: specializedWeightStore,
            accessPolicyResolver: context.accessPolicyResolver,
            kernelNameResolver: kernelNameResolver
        )
        return reportBuilder.makeReport(entries: optimization.fusedEntries)
    }

    /// Dump the compiled decode plan with concrete kernels, grid sizes, and bindings.
    ///
    /// This is a post-compilation diagnostic. Unlike `dumpDispatchEntries`, it shows
    /// the actual pipeline selected for each step after optimization, lowering,
    /// buffer routing, and kernel name resolution.
    public func dumpCompiledDecodePlan(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        stafWeightStore: STAFWeightStore? = nil,
        device: MTLDevice
    ) throws -> String {
        let plan = try compile(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            stafWeightStore: stafWeightStore,
            device: device)
        let formatter = CompiledPlanDiagnosticsFormatter()
        return formatter.formatDecodePlan(plan.decodePlan)
    }

    /// Dump the compiled prefill plan with concrete kernels, launch modes, and bindings.
    public func dumpCompiledPrefillPlan(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        inferencePolicy: InferencePolicy = .default,
        stafWeightStore: STAFWeightStore? = nil,
        device: MTLDevice
    ) throws -> String {
        let plan = try compilePrefill(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            inferencePolicy: inferencePolicy,
            stafWeightStore: stafWeightStore,
            device: device)
        let formatter = CompiledPlanDiagnosticsFormatter()
        return formatter.formatPrefillPlan(plan, maximumSequenceLength: inferencePolicy.maximumSequenceLength)
    }

    struct DecodeWeightBindingSummary: Sendable {
        let stepIndex: Int
        let kernelName: String
        let layerIndex: Int?
        let roles: [String]
        let tensorNames: [String]
        let inputDimension: Int
        let outputDimension: Int
        let preferredLayouts: [STAFWeightLayout]
        let resolvedLayouts: [STAFWeightLayout]
        let resolvedBufferLabels: [String]
    }

    func summarizeCompiledDecodeWeightBindings(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        stafWeightStore: STAFWeightStore? = nil,
        device: MTLDevice
    ) throws -> [DecodeWeightBindingSummary] {
        let initialContext = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            stafWeightStore: stafWeightStore,
            device: device)
        let initialOptimization = entryCollector.collect(
            using: initialContext,
            kernelContext: initialContext.decodeKernelContext)
        let specializedWeightStore = try prepareSpecializedWeightStore(
            initialContext.stafWeightStore,
            for: initialOptimization.fusedEntries,
            device: initialContext.device,
            accessPolicyResolver: initialContext.accessPolicyResolver
        )
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            stafWeightStore: specializedWeightStore,
            device: device)
        let optimization = entryCollector.collect(
            using: context,
            kernelContext: context.decodeKernelContext)
        let plan = try compile(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            stafWeightStore: specializedWeightStore,
            device: device)

        let paired = zip(plan.steps.indices, zip(plan.steps, optimization.fusedEntries))
        let accessPolicyResolver = context.accessPolicyResolver
        return paired.compactMap { index, pair -> DecodeWeightBindingSummary? in
            let step = pair.0
            let entry = pair.1
            guard let base = entry.decodeWeightBindingBase else {
                return nil
            }
            let requests = base.roles.compactMap { role -> STAFWeightAccessRequest? in
                guard
                    let store = specializedWeightStore,
                    let binding = entry.parameterBindings.first(where: { $0.role == role })
                else {
                    return nil
                }
                return accessPolicyResolver.accessRequest(
                    for: entry,
                    role: role,
                    binding: binding,
                    executionPhase: .decode,
                    stafWeightStore: store
                )
            }
            let preferredLayouts = requests.map(\.preferredLayout)
            let resolvedAccesses = requests.compactMap { request in
                specializedWeightStore?.resolvedBufferAccess(for: request)
            }
            return DecodeWeightBindingSummary(
                stepIndex: index,
                kernelName: step.pipeline.label ?? "(unlabeled)",
                layerIndex: entry.layerIndex,
                roles: base.roles,
                tensorNames: base.roles.compactMap { role in
                    entry.parameterBindings.first(where: { $0.role == role })?.tensorName
                },
                inputDimension: base.inputDimension,
                outputDimension: base.outputDimension,
                preferredLayouts: preferredLayouts,
                resolvedLayouts: resolvedAccesses.map(\.layout),
                resolvedBufferLabels: resolvedAccesses.map { $0.buffer.label ?? "(unlabeled)" }
            )
        }
    }

    /// Dump the generated decode kernel source for the optimized entry set.
    ///
    /// Unlike `dumpCompiledDecodePlan`, this returns the actual MSL that is fed
    /// into `makeLibrary` for the current model after optimization and kernel
    /// selection.
    public func dumpGeneratedDecodeKernelLibrary(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> String {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: 0,
            vocabSize: 0,
            stafWeightStore: stafWeightStore,
            device: MTLCreateSystemDefaultDevice()!)
        let optimization = entryCollector.collect(using: context, kernelContext: context.decodeKernelContext)
        let kernelNameResolver = MetalKernelNameResolver(
            stafWeightStore: context.stafWeightStore,
            weightAccessPolicyOverride: weightAccessPolicyOverride
        )
        let sourceCatalog = MetalKernelSourceCatalog(
            stafWeightStore: context.stafWeightStore,
            modelWeightFormat: context.weightFormat,
            bufferPrecision: context.decodeBufferPrecision,
            accessPolicyResolver: context.accessPolicyResolver,
            kernelNameResolver: kernelNameResolver
        )
        let generated = sourceCatalog.generateSources(entries: optimization.fusedEntries)
        return sourceCatalog.format(generated)
    }

    /// Dump the generated prefill kernel source for the optimized entry set.
    public func dumpGeneratedPrefillKernelLibrary(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> String {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: 0,
            vocabSize: 0,
            stafWeightStore: stafWeightStore,
            device: MTLCreateSystemDefaultDevice()!)
        let optimization = entryCollector.collect(using: context, kernelContext: context.prefillKernelContext)
        let kernelNameResolver = MetalKernelNameResolver(
            stafWeightStore: context.stafWeightStore,
            weightAccessPolicyOverride: weightAccessPolicyOverride
        )
        let sourceCatalog = MetalKernelSourceCatalog(
            stafWeightStore: context.stafWeightStore,
            modelWeightFormat: context.weightFormat,
            bufferPrecision: .float32,
            accessPolicyResolver: context.accessPolicyResolver,
            kernelNameResolver: kernelNameResolver
        )
        let generated = sourceCatalog.generateSources(entries: optimization.fusedEntries)
        return sourceCatalog.format(generated)
    }

    /// Dump kernel details for the prefill path.
    ///
    /// For each unique kernel, reports MSL source, constituent components,
    /// and which layers use it. Detects name collisions where entries share
    /// a name but produce different MSL.
    public func dumpKernels(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> KernelReport {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: 0,
            vocabSize: 0,
            stafWeightStore: stafWeightStore,
            device: MTLCreateSystemDefaultDevice()!)
        let kernelContext = context.prefillKernelContext
        let optimization = entryCollector.collect(
            using: context,
            kernelContext: kernelContext
        )

        var kernelsByName: [String: KernelReport.Kernel] = [:]
        var collisions: [KernelReport.Collision] = []

        for entry in optimization.fusedEntries {
            guard let synth = entry.fragment as? SynthesizedFragment else { continue }

            let name = synth.kernelName(context: kernelContext)
            let source = synth.kernelSource(
                name: name,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            )
            let components = synth.fragments.map { String(describing: type(of: $0)) }

            if var existing = kernelsByName[name] {
                existing.layers.append(entry.layerIndex)
                kernelsByName[name] = existing
                if existing.source != source {
                    collisions.append(KernelReport.Collision(
                        kernelName: name,
                        firstLayer: existing.layers.first ?? nil,
                        secondLayer: entry.layerIndex,
                        firstComponents: existing.components,
                        secondComponents: components
                    ))
                }
            } else {
                kernelsByName[name] = KernelReport.Kernel(
                    name: name,
                    source: source,
                    components: components,
                    layers: [entry.layerIndex]
                )
            }
        }

        return KernelReport(
            kernels: kernelsByName.values.sorted { $0.name < $1.name },
            collisions: collisions
        )
    }

    public func compile(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        inferencePolicy: InferencePolicy = .default,
        stafWeightStore: STAFWeightStore? = nil,
        device: MTLDevice
    ) throws -> MetalCompiledModel {
        let initialContext = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            inferencePolicy: inferencePolicy,
            stafWeightStore: stafWeightStore,
            device: device)
        let walk1Start = CFAbsoluteTimeGetCurrent()
        let initialOptimization = entryCollector.collect(using: initialContext, kernelContext: initialContext.decodeKernelContext)
        let walk1Time = CFAbsoluteTimeGetCurrent() - walk1Start
        InternalLog.info("[Prewarm/Compiler] decode walk #1 (initial): \(String(format: "%.3f", walk1Time))s")

        let specializeStart = CFAbsoluteTimeGetCurrent()
        let specializedWeightStore = try prepareSpecializedWeightStore(
            initialContext.stafWeightStore,
            for: initialOptimization.fusedEntries,
            device: initialContext.device,
            accessPolicyResolver: initialContext.accessPolicyResolver
        )
        let specializeTime = CFAbsoluteTimeGetCurrent() - specializeStart
        InternalLog.info("[Prewarm/Compiler] specialize weight store: \(String(format: "%.3f", specializeTime))s")

        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            inferencePolicy: inferencePolicy,
            stafWeightStore: specializedWeightStore,
            device: device)
        let walk2Start = CFAbsoluteTimeGetCurrent()
        let optimization = entryCollector.collect(using: context, kernelContext: context.decodeKernelContext)
        let walk2Time = CFAbsoluteTimeGetCurrent() - walk2Start
        InternalLog.info("[Prewarm/Compiler] decode walk #2 (final): \(String(format: "%.3f", walk2Time))s")

        let walkContext = optimization.walkContext
        let unfusedCount = optimization.unfusedCount
        let fusedEntries = optimization.fusedEntries

        // Phase 3: Compile only the kernels needed by this model's dispatch entries
        // Decode uses F16 buffers (single token, no accumulation)
        let pipelineStart = CFAbsoluteTimeGetCurrent()
        let (pipelineCache, argumentEncoderCache, quantizationCapabilities) = try compilePipelineCache(
            entries: fusedEntries, stafWeightStore: context.stafWeightStore,
            bufferPrecision: context.decodeBufferPrecision, device: context.device,
            label: "decode")
        let pipelineTime = CFAbsoluteTimeGetCurrent() - pipelineStart
        InternalLog.info("[Prewarm/Compiler] decode compilePipelineCache: \(String(format: "%.3f", pipelineTime))s")

        let planBuildContext = makePlanBuildContext(
            compileContext: context,
            kernelContext: context.decodeKernelContext,
            pipelineCache: pipelineCache,
            quantizationCapabilities: quantizationCapabilities)
        let allocStart = CFAbsoluteTimeGetCurrent()
        let allocation = try bufferAllocator.makeDecodeBufferAllocation(
            compileContext: context,
            walkContext: walkContext,
            fusedEntries: fusedEntries)
        let allocTime = CFAbsoluteTimeGetCurrent() - allocStart
        InternalLog.info("[Prewarm/Compiler] decode buffer allocation: \(String(format: "%.3f", allocTime))s")

        let bufferSet = allocation.bufferSet
        let decodeSlotDimension = allocation.slotDimension
        InternalLog.info("[Prewarm/Compiler] \(fusedEntries.count) dispatch entries")
        let planStart = CFAbsoluteTimeGetCurrent()
        let decodePlan = try dispatchStepBuilder.buildDecodePlan(
            fusedEntries: fusedEntries,
            unfusedCount: unfusedCount,
            bufferSet: bufferSet,
            slotDimension: decodeSlotDimension,
            stafWeightStore: context.stafWeightStore,
            hiddenSize: context.hiddenSize,
            accessPolicyResolver: context.accessPolicyResolver,
            planBuildContext: planBuildContext,
            argumentEncoders: argumentEncoderCache,
            resolveDispatch: { try resolvedDispatch(for: $0, using: planBuildContext) }
        )
        let planTime = CFAbsoluteTimeGetCurrent() - planStart
        InternalLog.info("[Prewarm/Compiler] decode plan build: \(String(format: "%.3f", planTime))s")
        return MetalCompiledModel(
            decodePlan: decodePlan,
            prefillPlan: nil,
            auxiliaryPipelines: pipelineCache
        )
    }

    // MARK: - Prefill Compilation

    /// Compile a sequence-aware prefill plan.
    ///
    /// The prefill plan is a **sequence graph**: step count is O(layers × ops_per_layer),
    /// NOT O(tokens × layers × ops_per_layer). Each kernel operates on [seqLen × dim]
    /// buffers. The GPU kernel itself iterates over the sequence dimension.
    ///
    /// - Projections: GEMM instead of GEMV ([seqLen × in] × [out × in]^T → [seqLen × out])
    /// - Embedding/Norm/Activation/Structural: batched variants with seqLen grid dimension
    /// - Attention: perPosition mode — runtime loops over positions for KV cache fill
    public func compilePrefill(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        inferencePolicy: InferencePolicy = .default,
        stafWeightStore: STAFWeightStore? = nil,
        sharedKVCache: MetalKVCache? = nil,
        sharedConvState: MTLBuffer? = nil,
        sharedConvStateDimension: Int = 0,
        sharedConvStateKernelSize: Int = 0,
        sharedRecurrentState: MTLBuffer? = nil,
        sharedRecurrentStateBytesPerLayer: Int = 0,
        device: MTLDevice
    ) throws -> MetalPrefillPlan {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            inferencePolicy: inferencePolicy,
            stafWeightStore: stafWeightStore,
            device: device)
        let walkStart = CFAbsoluteTimeGetCurrent()
        let optimization = entryCollector.collect(using: context, kernelContext: context.prefillKernelContext)
        let walkTime = CFAbsoluteTimeGetCurrent() - walkStart
        InternalLog.info("[Prewarm/Compiler] prefill walk: \(String(format: "%.3f", walkTime))s")
        let walkContext = optimization.walkContext
        let fusedEntries = optimization.fusedEntries

        // Compile only the kernels needed by this model's prefill dispatch entries
        // For prefill (F32), attempts Metal 4 MPP GEMM with fallback to naive GEMM.
        let pipelineStart = CFAbsoluteTimeGetCurrent()
        let (pipelineCache, _, quantizationCapabilities) = try compilePipelineCache(
            entries: fusedEntries, stafWeightStore: context.stafWeightStore,
            bufferPrecision: .float32, device: context.device,
            label: "prefill")
        let pipelineTime = CFAbsoluteTimeGetCurrent() - pipelineStart
        InternalLog.info("[Prewarm/Compiler] prefill compilePipelineCache: \(String(format: "%.3f", pipelineTime))s")
        let prefillUsesMPP = quantizationCapabilities.prefillProjectionAcceleration.isEnabled
        let planBuildContext = makePlanBuildContext(
            compileContext: context,
            kernelContext: context.prefillKernelContext,
            pipelineCache: pipelineCache,
            quantizationCapabilities: quantizationCapabilities)
        let allocStart = CFAbsoluteTimeGetCurrent()
        let allocation = try bufferAllocator.makePrefillBufferAllocation(
            compileContext: context,
            walkContext: walkContext,
            fusedEntries: fusedEntries,
            sharedKVCache: sharedKVCache,
            sharedConvState: sharedConvState,
            sharedConvStateDimension: sharedConvStateDimension,
            sharedConvStateKernelSize: sharedConvStateKernelSize,
            sharedRecurrentState: sharedRecurrentState,
            sharedRecurrentStateBytesPerLayer: sharedRecurrentStateBytesPerLayer)
        let allocTime = CFAbsoluteTimeGetCurrent() - allocStart
        InternalLog.info("[Prewarm/Compiler] prefill buffer allocation: \(String(format: "%.3f", allocTime))s")
        let prefillBuffers = allocation.bufferSet
        let slotDimension = allocation.slotDimension
        let maxSeq = allocation.maximumSequenceLength
        InternalLog.info("[Prewarm/Compiler] prefill \(fusedEntries.count) dispatch entries")
        let planStart = CFAbsoluteTimeGetCurrent()
        let plan = try prefillStepBuilder.buildPrefillPlan(
            fusedEntries: fusedEntries,
            buffers: prefillBuffers,
            slotDimension: slotDimension,
            maximumSequenceLength: maxSeq,
            hiddenSize: context.hiddenSize,
            scratchElementSize: MemoryLayout<Float32>.size,
            usesMPP: prefillUsesMPP,
            planBuildContext: planBuildContext,
            resolveDispatch: { try resolvedDispatch(for: $0, using: planBuildContext) }
        )
        let planTime = CFAbsoluteTimeGetCurrent() - planStart
        InternalLog.info("[Prewarm/Compiler] prefill plan build: \(String(format: "%.3f", planTime))s")
        return plan
    }

    // MARK: - Metal Library Cache


    /// Compile Metal libraries and build a pipeline cache for the given dispatch entries.
    ///
    /// Kernel source is generated on-demand from fragment parameters + STAF weight format.
    /// No hardcoded catalog — only the kernels actually used are compiled.
    /// For prefill (F32), attempts Metal 4 MPP GEMM with fallback to naive GEMM.
    private func compilePipelineCache(
        entries: [DispatchEntry],
        stafWeightStore: STAFWeightStore?,
        bufferPrecision: MetalSourceGenerator.BufferPrecision,
        device: MTLDevice,
        label: String
    ) throws -> (
        pipelines: [String: MTLComputePipelineState],
        argumentEncoders: [String: MTLArgumentEncoder],
        quantizationCapabilities: MetalQuantizationCapabilities
    ) {
        let kernelNameResolver = MetalKernelNameResolver(
            stafWeightStore: stafWeightStore,
            weightAccessPolicyOverride: weightAccessPolicyOverride
        )
        let sourceCatalog = MetalKernelSourceCatalog(
            stafWeightStore: stafWeightStore,
            modelWeightFormat: kernelNameResolver.resolveModelWeightFormat(),
            bufferPrecision: bufferPrecision,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver(
                override: weightAccessPolicyOverride
            ),
            kernelNameResolver: kernelNameResolver
        )
        let generateStart = CFAbsoluteTimeGetCurrent()
        let generated = sourceCatalog.generateSources(entries: entries)
        let generateTime = CFAbsoluteTimeGetCurrent() - generateStart
        InternalLog.info("[Prewarm/Compiler] \(label) generateSources: \(String(format: "%.3f", generateTime))s")
        let pipelineCompiler = MetalPipelineCompiler(device: device, label: label)
        return try pipelineCompiler.compile(generated)
    }


    // MARK: - Helpers

    private func roundUp(_ value: Int, to multiple: Int) -> Int {
        guard multiple > 0 else { return max(value, 1) }
        return ((value + multiple - 1) / multiple) * multiple
    }

}
