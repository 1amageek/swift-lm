import Metal

private enum SharedPipelineCache {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var pipelines: [String: MTLComputePipelineState] = [:]

    static func pipeline(named name: String) -> MTLComputePipelineState? {
        lock.lock()
        defer { lock.unlock() }
        return pipelines[name]
    }

    static func store(_ pipeline: MTLComputePipelineState, named name: String) {
        lock.lock()
        pipelines[name] = pipeline
        lock.unlock()
    }
}

struct MetalPipelineCompiler {
    let device: MTLDevice
    let label: String

    init(device: MTLDevice, label: String = "compile") {
        self.device = device
        self.label = label
    }

    func compile(_ generated: GeneratedKernelSources) throws -> (
        pipelines: [String: MTLComputePipelineState],
        argumentEncoders: [String: MTLArgumentEncoder],
        quantizationCapabilities: MetalQuantizationCapabilities
    ) {
        let mppDisabled = ProcessInfo.processInfo.environment["SWIFTLM_DISABLE_MPP"] == "1"
        let baseLibStart = CFAbsoluteTimeGetCurrent()
        let baseLibrary = try makeLibrary(source: generated.baseSource, options: baseCompileOptions())
        let baseLibTime = CFAbsoluteTimeGetCurrent() - baseLibStart
        InternalLog.info("[Prewarm/Pipeline] \(label) base makeLibrary: \(String(format: "%.3f", baseLibTime))s (\(baseLibrary.functionNames.count) functions)")
        emitKernelDiagnosticsIfRequested(
            generated: generated,
            library: baseLibrary
        )
        let basePipelineStart = CFAbsoluteTimeGetCurrent()
        let baseResult = try makeBasePipelineCache(from: baseLibrary)
        let basePipelineTime = CFAbsoluteTimeGetCurrent() - basePipelineStart
        InternalLog.info("[Prewarm/Pipeline] \(label) base PSO loop: \(String(format: "%.3f", basePipelineTime))s (\(baseResult.cacheHits) hits, \(baseResult.cacheMisses) misses)")
        var pipelineCache = baseResult.cache
        for kernelName in generated.mppKernelNames {
            if let basePipeline = pipelineCache[kernelName] {
                pipelineCache["naive::\(kernelName)"] = basePipeline
            }
        }
        var argumentEncoderCache = makeArgumentEncoderCache(from: baseLibrary)

        guard !mppDisabled, !generated.mppSources.isEmpty else {
            let accelerationAvailability: MetalPrefillProjectionAccelerationAvailability = if mppDisabled {
                .disabledByEnvironment
            } else {
                .unavailable
            }
            return (
                pipelineCache,
                argumentEncoderCache,
                MetalQuantizationCapabilities(
                    prefillProjectionAcceleration: accelerationAvailability
                )
            )
        }

        do {
            let mppLibStart = CFAbsoluteTimeGetCurrent()
            let mppLibrary = try makeLibrary(
                source: generated.mppSources.joined(separator: "\n\n"),
                options: mppCompileOptions())
            let mppLibTime = CFAbsoluteTimeGetCurrent() - mppLibStart
            InternalLog.info("[Prewarm/Pipeline] \(label) mpp makeLibrary: \(String(format: "%.3f", mppLibTime))s (\(mppLibrary.functionNames.count) functions)")
            let mppPipelineStart = CFAbsoluteTimeGetCurrent()
            let mppStats = try mergeMPPipelines(from: mppLibrary, into: &pipelineCache)
            let mppPipelineTime = CFAbsoluteTimeGetCurrent() - mppPipelineStart
            InternalLog.info("[Prewarm/Pipeline] \(label) mpp PSO loop: \(String(format: "%.3f", mppPipelineTime))s (\(mppStats.cacheHits) hits, \(mppStats.cacheMisses) misses)")
            argumentEncoderCache.merge(
                makeArgumentEncoderCache(from: mppLibrary),
                uniquingKeysWith: { existing, _ in existing })
            return (
                pipelineCache,
                argumentEncoderCache,
                MetalQuantizationCapabilities(prefillProjectionAcceleration: .enabled)
            )
        } catch {
            return (
                pipelineCache,
                argumentEncoderCache,
                MetalQuantizationCapabilities(prefillProjectionAcceleration: .unavailable)
            )
        }
    }

    private func makeLibrary(source: String, options: MTLCompileOptions) throws -> MTLLibrary {
        do {
            return try device.makeLibrary(source: source, options: options)
        } catch {
            // Diagnostic: dump source on compilation failure. The dump is a
            // best-effort debugging aid; if writing it also fails we surface
            // the I/O error to stderr but still rethrow the original
            // compilation error (which is what the caller actually asked
            // about).
            let url = URL(fileURLWithPath: "/tmp/swiftlm_failed_msl.metal")
            do {
                try source.write(to: url, atomically: true, encoding: .utf8)
            } catch let dumpError {
                InternalLog.error(
                    "[Prewarm/Pipeline] failed to dump MSL source to \(url.path): \(dumpError)"
                )
            }
            throw error
        }
    }

    private func emitKernelDiagnosticsIfRequested(
        generated: GeneratedKernelSources,
        library: MTLLibrary
    ) {
        guard ProcessInfo.processInfo.environment["SWIFTLM_DEBUG_KERNELS"] == "1" else {
            return
        }
        let interestingNames = [
            "embedding_lookup",
            "embedding_lookup_bf16",
            "embedding_lookup_argbuf",
            "embedding_lookup_bf16_argbuf",
        ]
        let available = Set(library.functionNames)
        let emitted = interestingNames.filter { generated.baseSource.contains("kernel void \($0)(") }
        let compiled = interestingNames.filter { available.contains($0) }
        InternalLog.info("[Prewarm/Pipeline] emitted embedding kernels: \(emitted)")
        InternalLog.info("[Prewarm/Pipeline] compiled embedding kernels: \(compiled)")
    }

    private func baseCompileOptions() -> MTLCompileOptions {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.languageVersion = .version4_0
        return options
    }

    private func mppCompileOptions() -> MTLCompileOptions {
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        return options
    }

    struct PipelineCacheResult {
        var cache: [String: MTLComputePipelineState]
        var cacheHits: Int
        var cacheMisses: Int
    }

    struct PipelineCacheStats {
        var cacheHits: Int
        var cacheMisses: Int
    }

    private func makeBasePipelineCache(
        from library: MTLLibrary
    ) throws -> PipelineCacheResult {
        var pipelineCache: [String: MTLComputePipelineState] = [:]
        var hits = 0
        var misses = 0
        for name in library.functionNames {
            if let cached = SharedPipelineCache.pipeline(named: name) {
                pipelineCache[name] = cached
                hits += 1
                continue
            }
            guard let function = library.makeFunction(name: name) else {
                continue
            }
            let pipeline = try makePipeline(function: function, label: name)
            pipelineCache[name] = pipeline
            SharedPipelineCache.store(pipeline, named: name)
            misses += 1
        }
        return PipelineCacheResult(cache: pipelineCache, cacheHits: hits, cacheMisses: misses)
    }

    private func mergeMPPipelines(
        from library: MTLLibrary,
        into pipelineCache: inout [String: MTLComputePipelineState]
    ) throws -> PipelineCacheStats {
        var hits = 0
        var misses = 0
        for name in library.functionNames {
            let cacheKey = "mpp::\(name)"
            if let cached = SharedPipelineCache.pipeline(named: cacheKey) {
                pipelineCache[name] = cached
                hits += 1
                continue
            }
            guard let function = library.makeFunction(name: name) else {
                continue
            }
            let pipeline = try makePipeline(function: function, label: name)
            pipelineCache[name] = pipeline
            SharedPipelineCache.store(pipeline, named: cacheKey)
            misses += 1
        }
        return PipelineCacheStats(cacheHits: hits, cacheMisses: misses)
    }

    private func makeArgumentEncoderCache(
        from library: MTLLibrary
    ) -> [String: MTLArgumentEncoder] {
        var cache: [String: MTLArgumentEncoder] = [:]
        for name in library.functionNames where name.hasSuffix("_argbuf") {
            guard let function = library.makeFunction(name: name) else {
                continue
            }
            cache[name] = function.makeArgumentEncoder(
                bufferIndex: MetalInferenceCompiler.argumentTableBindingIndex)
        }
        return cache
    }

    private func makePipeline(
        function: MTLFunction,
        label: String
    ) throws -> MTLComputePipelineState {
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.label = label
        return try device.makeComputePipelineState(
            descriptor: descriptor,
            options: [],
            reflection: nil)
    }
}
