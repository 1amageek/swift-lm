import Metal

struct MetalBufferAllocator {
    func makeDecodeBufferAllocation(
        compileContext context: CompileContext,
        walkContext: WalkContext,
        fusedEntries: [DispatchEntry]
    ) throws -> DecodeBufferAllocation {
        let elementSize = context.decodeBufferPrecision.byteSize
        let resolvedIntermediateSize = context.resolvedIntermediateSize
        let resolvedVocabSize = context.resolvedVocabSize
        let slotDimension = max(
            context.hiddenSize,
            resolvedIntermediateSize,
            maximumScratchProjectionDimension(in: fusedEntries)
        )
        let scratchElementCount = max(slotDimension * 4, resolvedIntermediateSize * 4)

        let gpuOnlyOptions: MTLResourceOptions = [.storageModePrivate, .hazardTrackingModeUntracked]
        let cpuAccessOptions: MTLResourceOptions = [.storageModeShared]

        let hiddenBuffer = context.device.makeBuffer(length: context.hiddenSize * elementSize, options: gpuOnlyOptions)!
        let residualBuffer = context.device.makeBuffer(length: context.hiddenSize * elementSize, options: gpuOnlyOptions)!
        let scratchBuffer = context.device.makeBuffer(length: scratchElementCount * elementSize, options: gpuOnlyOptions)!
        let logitsBuffer = context.device.makeBuffer(length: resolvedVocabSize * elementSize, options: gpuOnlyOptions)!
        let positionBuffer = context.device.makeBuffer(length: 4, options: cpuAccessOptions)!
        let tokenInputBuffer = context.device.makeBuffer(length: 4, options: cpuAccessOptions)!
        let tokenOutputBuffer = context.device.makeBuffer(length: 4, options: cpuAccessOptions)!

        let kvCache: MetalKVCache?
        if let firstSlot = walkContext.cacheSlots.first {
            let kvCacheScheme = preferredKVCacheScheme(for: context.weightFormat)
            kvCache = try MetalKVCache(
                device: context.device,
                specification: KVCacheSpecification(
                    keyQuantizationScheme: kvCacheScheme,
                    valueQuantizationScheme: kvCacheScheme,
                    layerCount: walkContext.cacheSlots.count,
                    kvHeadCount: firstSlot.kvHeadCount,
                    headDimension: firstSlot.headDimension
                ),
                resourceOptions: gpuOnlyOptions
            )
        } else {
            kvCache = nil
        }

        let convState = convStateRequirements(in: fusedEntries)
        let convStateBuffer: MTLBuffer?
        if convState.layerCount > 0 {
            let byteCount = convState.layerCount * convState.kernelSize * convState.dimension * elementSize
            convStateBuffer = context.device.makeBuffer(length: byteCount, options: gpuOnlyOptions)
        } else {
            convStateBuffer = nil
        }

        let weightBuffers = context.stafWeightStore.map { [$0.buffer] } ?? []
        let bufferSet = MetalBufferSet(
            bufferPrecision: context.decodeBufferPrecision,
            hidden: hiddenBuffer,
            residual: residualBuffer,
            scratch: scratchBuffer,
            weights: weightBuffers,
            kvCache: kvCache,
            convState: convStateBuffer,
            convStateDimension: convState.dimension,
            convStateKernelSize: convState.kernelSize,
            logits: logitsBuffer,
            position: positionBuffer,
            tokenIn: tokenInputBuffer,
            tokenOut: tokenOutputBuffer
        )
        return DecodeBufferAllocation(bufferSet: bufferSet, slotDimension: slotDimension)
    }

    func makePrefillBufferAllocation(
        compileContext context: CompileContext,
        walkContext: WalkContext,
        fusedEntries: [DispatchEntry],
        sharedKVCache: MetalKVCache?,
        sharedConvState: MTLBuffer?,
        sharedConvStateDimension: Int,
        sharedConvStateKernelSize: Int
    ) throws -> PrefillBufferAllocation {
        let elementSize = MemoryLayout<Float16>.size
        let f32ElementSize = MemoryLayout<Float32>.size
        let resolvedIntermediateSize = context.resolvedIntermediateSize
        let resolvedVocabSize = context.resolvedVocabSize
        let maximumSequenceLength = context.maximumSequenceLength
        let slotDimension = max(
            context.hiddenSize,
            resolvedIntermediateSize,
            maximumScratchProjectionDimension(in: fusedEntries)
        )
        let scratchElementCount = max(slotDimension * 4, resolvedIntermediateSize * 4)
        let gpuOptions: MTLResourceOptions = [.storageModeShared]

        let convStateRequirements = convStateRequirements(in: fusedEntries)
        let prefillConvStateBuffer: MTLBuffer?
        let resolvedConvDimension: Int
        let resolvedConvKernelSize: Int
        if let sharedConvState {
            prefillConvStateBuffer = sharedConvState
            resolvedConvDimension = sharedConvStateDimension
            resolvedConvKernelSize = sharedConvStateKernelSize
        } else if convStateRequirements.layerCount > 0 {
            let byteCount = convStateRequirements.layerCount
                * convStateRequirements.kernelSize
                * convStateRequirements.dimension
                * elementSize
            prefillConvStateBuffer = context.device.makeBuffer(length: byteCount, options: gpuOptions)
            if let prefillConvStateBuffer {
                memset(prefillConvStateBuffer.contents(), 0, prefillConvStateBuffer.length)
            }
            resolvedConvDimension = convStateRequirements.dimension
            resolvedConvKernelSize = convStateRequirements.kernelSize
        } else {
            prefillConvStateBuffer = nil
            resolvedConvDimension = 0
            resolvedConvKernelSize = 0
        }

        let prefillKVCache: MetalKVCache?
        if let sharedKVCache {
            prefillKVCache = sharedKVCache
        } else if let firstSlot = walkContext.cacheSlots.first {
            let kvCacheScheme = preferredKVCacheScheme(for: context.weightFormat)
            prefillKVCache = try MetalKVCache(
                device: context.device,
                specification: KVCacheSpecification(
                    keyQuantizationScheme: kvCacheScheme,
                    valueQuantizationScheme: kvCacheScheme,
                    layerCount: walkContext.cacheSlots.count,
                    kvHeadCount: firstSlot.kvHeadCount,
                    headDimension: firstSlot.headDimension
                ),
                resourceOptions: gpuOptions
            )
        } else {
            prefillKVCache = nil
        }

        let bufferSet = PrefillBufferSet(
            bufferPrecision: .float32,
            hidden: context.device.makeBuffer(length: maximumSequenceLength * context.hiddenSize * f32ElementSize, options: gpuOptions)!,
            residual: context.device.makeBuffer(length: maximumSequenceLength * context.hiddenSize * f32ElementSize, options: gpuOptions)!,
            scratch: context.device.makeBuffer(length: maximumSequenceLength * scratchElementCount * f32ElementSize, options: gpuOptions)!,
            weights: context.stafWeightStore.map { [$0.buffer] } ?? [],
            kvCache: prefillKVCache,
            convState: prefillConvStateBuffer,
            convStateDimension: resolvedConvDimension,
            convStateKernelSize: resolvedConvKernelSize,
            logits: context.device.makeBuffer(length: resolvedVocabSize * f32ElementSize, options: gpuOptions)!,
            tokenIDs: context.device.makeBuffer(length: maximumSequenceLength * 4, options: [.storageModeShared])!,
            positions: context.device.makeBuffer(length: maximumSequenceLength * 4, options: [.storageModeShared])!,
            tokenOut: context.device.makeBuffer(length: 4, options: [.storageModeShared])!
        )

        return PrefillBufferAllocation(
            bufferSet: bufferSet,
            slotDimension: slotDimension,
            resolvedIntermediateSize: resolvedIntermediateSize,
            resolvedVocabSize: resolvedVocabSize,
            maximumSequenceLength: maximumSequenceLength
        )
    }

    private func maximumScratchProjectionDimension(in entries: [DispatchEntry]) -> Int {
        var maximumOutputDimension = 0
        for entry in entries {
            if case .projection(let projection, let isOutput) = entry.kind, !isOutput {
                maximumOutputDimension = max(maximumOutputDimension, projection.outputDimension)
            }
        }
        return maximumOutputDimension
    }

    private func convStateRequirements(in entries: [DispatchEntry]) -> ConvStateRequirements {
        var layerCount = 0
        var dimension = 0
        var kernelSize = 0
        for entry in entries {
            if case .fragment(let fragment) = entry.kind,
               let convSlot = fragment.cacheSlots.first(where: { $0.kind == .conv }),
               case .elementwise(let fragmentDimension) = fragment.dispatchDimension {
                layerCount += 1
                dimension = max(dimension, fragmentDimension)
                kernelSize = max(kernelSize, convSlot.temporalSize)
            }
        }
        return ConvStateRequirements(
            layerCount: layerCount,
            dimension: dimension,
            kernelSize: kernelSize
        )
    }

    private func preferredKVCacheScheme(for weightFormat: WeightFormat) -> QuantizationSchemeIdentifier {
        weightFormat == .bfloat16 ? .bf16RowMajor : .fp16RowMajor
    }
}
