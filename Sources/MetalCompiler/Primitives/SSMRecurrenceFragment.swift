import Foundation
import Metal

/// DeltaNet/Mamba state-space model recurrence step.
public struct SSMRecurrenceFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let groupCount: Int
    public let keyHeadDimension: Int
    public let valueHeadDimension: Int
    public let convKernelSize: Int

    public init(
        headCount: Int,
        groupCount: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int,
        convKernelSize: Int
    ) {
        self.headCount = headCount
        self.groupCount = groupCount
        self.keyHeadDimension = keyHeadDimension
        self.valueHeadDimension = valueHeadDimension
        self.convKernelSize = convKernelSize
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        Self.kernelName(
            bufferPrecision: context.bufferPrecision,
            weightFormat: context.weightFormat
        )
    }
    public var dispatchDimension: MetalDispatchDimension {
        // Partition work by key-group: each threadgroup owns disjoint Q/K/V
        // conv channels and the recurrent state slice of its heads — no
        // cross-threadgroup synchronization required.
        let safeGroupCount = max(groupCount, 1)
        let headsPerGroup = max(1, headCount / safeGroupCount)
        let localDim = 2 * keyHeadDimension + headsPerGroup * valueHeadDimension
        let phase2Threads = headsPerGroup * min(valueHeadDimension, 256)
        let desiredThreads = max(localDim, phase2Threads)
        let clamped = min(Self.maxThreadgroupSize, max(desiredThreads, 1))
        return .partitionedReduction(
            partitionCount: safeGroupCount,
            threadsPerPartition: clamped
        )
    }
    public var weightSlots: [MetalWeightSlot] {
        [
            MetalWeightSlot(field: "conv_weight", role: .weight),
            MetalWeightSlot(field: "scale", role: .scale),
        ]
    }
    public var cacheSlots: [MetalCacheSlot] {
        [
            MetalCacheSlot(name: "linear_conv_cache", kind: .conv, temporalSize: convKernelSize),
            MetalCacheSlot(name: "linear_recurrent_state", kind: .recurrent),
        ]
    }

    public var convDimension: Int {
        2 * groupCount * keyHeadDimension + headCount * valueHeadDimension
    }

    /// Upper bound for threadgroup size used in SSM kernels.
    /// Matches the normPartials array size in the generated kernel source.
    /// At dispatch time, actual threads = min(this, pipeline.maxTotalThreadsPerThreadgroup).
    public static let maxThreadgroupSize = 1024

    static func kernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bf16 = weightFormat == .bfloat16
        if bufferPrecision == .float32 {
            return bf16 ? "ssm_recurrence_bf16_f32" : "ssm_recurrence_f32"
        }
        return bf16 ? "ssm_recurrence_bf16" : "ssm_recurrence"
    }

    static func sequenceKernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bf16 = weightFormat == .bfloat16
        if bufferPrecision == .float32 {
            return bf16 ? "ssm_recurrence_seq_bf16_f32" : "ssm_recurrence_seq_f32"
        }
        return bf16 ? "ssm_recurrence_seq_bf16" : "ssm_recurrence_seq"
    }

    static func sharedRMSSequenceKernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        sequenceKernelName(bufferPrecision: bufferPrecision, weightFormat: weightFormat) + "_shared_rms"
    }

    static func prewriteDecaySequenceKernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        sequenceKernelName(bufferPrecision: bufferPrecision, weightFormat: weightFormat) + "_prewrite_decay"
    }

    static func qkParallelSequenceKernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        sequenceKernelName(bufferPrecision: bufferPrecision, weightFormat: weightFormat) + "_qkpar"
    }

    static func cachedParametersSequenceKernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        sequenceKernelName(bufferPrecision: bufferPrecision, weightFormat: weightFormat) + "_cached_params"
    }

    static func parallelStateSequenceKernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        sequenceKernelName(bufferPrecision: bufferPrecision, weightFormat: weightFormat) + "_parallel_state"
    }

    static func groupOwnedPartialProjectionSequenceKernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        sequenceKernelName(bufferPrecision: bufferPrecision, weightFormat: weightFormat) + "_group_owned_partial"
    }

    static func partitionOwnedPartialProjectionSequenceKernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        sequenceKernelName(bufferPrecision: bufferPrecision, weightFormat: weightFormat) + "_partition_owned_partial"
    }

    static var isSharedRMSPrefillEnabled: Bool {
        ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_SSM_SHARED_RMS"] == "1"
    }

    static var isPrewriteDecayPrefillEnabled: Bool {
        ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_SSM_PREWRITE_DECAY"] == "1"
    }

    static var isQKParallelPrefillEnabled: Bool {
        ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_SSM_QKPAR"] == "1"
    }

    static var isCachedParametersPrefillEnabled: Bool {
        ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_SSM_CACHED_PARAMS"] == "1"
    }

    static var parallelStatePrefillEnvironmentOverride: Bool? {
        guard let raw = ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_SSM_PARALLEL_STATE"] else {
            return nil
        }
        if raw == "1" { return true }
        if raw == "0" { return false }
        return nil
    }

    static func parallelStatePrefillOverride() throws -> Bool? {
        guard let raw = ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_SSM_PARALLEL_STATE"] else {
            return nil
        }
        if raw == "1" { return true }
        if raw == "0" { return false }
        throw MetalCompilerError.deviceSetupFailed(
            "SWIFTLM_PREFILL_SSM_PARALLEL_STATE must be 0 or 1, got '\(raw)'"
        )
    }

    static func isParallelStateDefaultEligible(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        headCount: Int,
        groupCount: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int
    ) -> Bool {
        bufferPrecision == .float32
            && weightFormat == .bfloat16
            && headCount == groupCount
            && keyHeadDimension == 128
            && valueHeadDimension == 128
    }

    static var isConvDebugPrefillEnabled: Bool {
        getenv("SWIFTLM_PREFILL_DEBUG_SSM_CONV") != nil
    }

    static func prefillThreadgroupWidthOverride(
        defaultThreads: Int,
        minimumActiveThreads: Int,
        simdWidth: Int
    ) throws -> Int {
        guard let raw = ProcessInfo.processInfo.environment["SWIFTLM_PREFILL_SSM_THREADGROUP_WIDTH"] else {
            return defaultThreads
        }
        guard let requested = Int(raw) else {
            throw MetalCompilerError.deviceSetupFailed(
                "SWIFTLM_PREFILL_SSM_THREADGROUP_WIDTH must be an integer, got '\(raw)'"
            )
        }
        guard requested >= minimumActiveThreads else {
            throw MetalCompilerError.deviceSetupFailed(
                "SWIFTLM_PREFILL_SSM_THREADGROUP_WIDTH \(requested) is below required active threads \(minimumActiveThreads)"
            )
        }
        guard requested <= defaultThreads else {
            throw MetalCompilerError.deviceSetupFailed(
                "SWIFTLM_PREFILL_SSM_THREADGROUP_WIDTH \(requested) exceeds default threadgroup width \(defaultThreads)"
            )
        }
        guard simdWidth <= 1 || requested % simdWidth == 0 else {
            throw MetalCompilerError.deviceSetupFailed(
                "SWIFTLM_PREFILL_SSM_THREADGROUP_WIDTH \(requested) must be a multiple of SIMD width \(simdWidth)"
            )
        }
        return requested
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        switch role {
        case "conv_weight":
            return convDimension * convKernelSize * bytesPerScalar
        default:
            return convDimension * bytesPerScalar
        }
    }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateSSMRecurrence(
            name: name,
            bufferPrecision: bufferPrecision,
            weightFormat: weightFormat,
            convDimension: convDimension,
            maxThreadgroupSize: Self.maxThreadgroupSize,
            headCount: headCount,
            groupCount: groupCount,
            keyHeadDimension: keyHeadDimension,
            valueHeadDimension: valueHeadDimension
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let scratchSlotBytes = context.slotDimension * context.elementSize

        let (convWeightBuffer, convWeightOffset) = context.resolveWeight("conv_weight")
        let (normWeightBuffer, normWeightOffset) = context.resolveWeight("scale")
        let (dtBiasBuffer, dtBiasOffset) = context.resolveWeight("dt_bias")
        let (aLogBuffer, aLogOffset) = context.resolveWeight("A_log")

        guard let recurrentState = context.bufferSet.recurrentState else {
            fatalError("[Compiler] SSMRecurrenceFragment requires recurrent state buffer")
        }
        guard let convState = context.bufferSet.convState else {
            fatalError("[Compiler] SSMRecurrenceFragment requires conv state buffer")
        }

        let recurrentLayerOffset = context.recurrentLayerIndex * context.bufferSet.recurrentStateBytesPerLayer
        let convStateElementSize = MemoryLayout<Float16>.size
        let convLayerOffset = context.convLayerIndex
            * context.bufferSet.convStateKernelSize
            * context.bufferSet.convStateDimension
            * convStateElementSize

        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 1 * scratchSlotBytes),
                (1, context.bufferSet.scratch, 2 * scratchSlotBytes),
                (2, context.bufferSet.scratch, 3 * scratchSlotBytes),
                (3, context.bufferSet.scratch, 4 * scratchSlotBytes),
                (4, convWeightBuffer, convWeightOffset),
                (5, normWeightBuffer, normWeightOffset),
                (6, dtBiasBuffer, dtBiasOffset),
                (7, aLogBuffer, aLogOffset),
                (8, recurrentState, recurrentLayerOffset),
                (9, convState, convLayerOffset),
                (10, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(11, UInt32(headCount)),
                uint32Binding(12, UInt32(groupCount)),
                uint32Binding(13, UInt32(keyHeadDimension)),
                uint32Binding(14, UInt32(valueHeadDimension)),
                uint32Binding(15, UInt32(convKernelSize)),
            ],
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesConvLayer: true,
            consumesRecurrentLayer: true,
            writeBufferIndices: Set<Int>([3, 8, 9, 10])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength

        let (convWeightBuffer, convWeightOffset) = context.resolveWeight("conv_weight")
        let (normWeightBuffer, normWeightOffset) = context.resolveWeight("scale")
        let (dtBiasBuffer, dtBiasOffset) = context.resolveWeight("dt_bias")
        let (aLogBuffer, aLogOffset) = context.resolveWeight("A_log")

        guard let recurrentState = context.buffers.recurrentState else {
            fatalError("[Compiler] SSMRecurrenceFragment requires recurrent state buffer")
        }
        guard let convState = context.buffers.convState else {
            fatalError("[Compiler] SSMRecurrenceFragment requires conv state buffer")
        }

        let recurrentLayerOffset = context.recurrentLayerIndex * context.buffers.recurrentStateBytesPerLayer
        let convLayerOffset = context.convLayerIndex
            * context.buffers.convStateKernelSize
            * context.buffers.convStateDimension
            * MemoryLayout<Float16>.size
        let sharedRMSPrefillEnabled = Self.isSharedRMSPrefillEnabled
        let prewriteDecayPrefillEnabled = Self.isPrewriteDecayPrefillEnabled
        let qkParallelPrefillEnabled = Self.isQKParallelPrefillEnabled
        let cachedParametersPrefillEnabled = Self.isCachedParametersPrefillEnabled
        let parallelStateOverride = try Self.parallelStatePrefillOverride()
        let explicitParallelStatePrefillEnabled = parallelStateOverride == true
        let explicitVariantCount = [
            sharedRMSPrefillEnabled,
            prewriteDecayPrefillEnabled,
            qkParallelPrefillEnabled,
            cachedParametersPrefillEnabled,
            explicitParallelStatePrefillEnabled,
        ]
            .filter { $0 }
            .count
        if explicitVariantCount > 1 {
            throw MetalCompilerError.deviceSetupFailed(
                "SWIFTLM_PREFILL_SSM_SHARED_RMS, SWIFTLM_PREFILL_SSM_PREWRITE_DECAY, SWIFTLM_PREFILL_SSM_QKPAR, SWIFTLM_PREFILL_SSM_CACHED_PARAMS, and SWIFTLM_PREFILL_SSM_PARALLEL_STATE are mutually exclusive"
            )
        }
        let defaultParallelStatePrefillEnabled = explicitVariantCount == 0
            && Self.isParallelStateDefaultEligible(
                bufferPrecision: context.kernelContext.bufferPrecision,
                weightFormat: context.kernelContext.weightFormat,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyHeadDimension,
                valueHeadDimension: valueHeadDimension
            )
        let parallelStatePrefillEnabled = parallelStateOverride ?? defaultParallelStatePrefillEnabled
        if qkParallelPrefillEnabled || cachedParametersPrefillEnabled || parallelStatePrefillEnabled {
            guard context.kernelContext.bufferPrecision == .float32,
                  context.kernelContext.weightFormat == .bfloat16 else {
                throw MetalCompilerError.deviceSetupFailed(
                    "SWIFTLM_PREFILL_SSM_QKPAR, SWIFTLM_PREFILL_SSM_CACHED_PARAMS, and SWIFTLM_PREFILL_SSM_PARALLEL_STATE currently support only BF16 weights with F32 sequence buffers"
                )
            }
        }
        let kernelName = if prewriteDecayPrefillEnabled {
            Self.prewriteDecaySequenceKernelName(
                bufferPrecision: context.kernelContext.bufferPrecision,
                weightFormat: context.kernelContext.weightFormat
            )
        } else if qkParallelPrefillEnabled {
            Self.qkParallelSequenceKernelName(
                bufferPrecision: context.kernelContext.bufferPrecision,
                weightFormat: context.kernelContext.weightFormat
            )
        } else if cachedParametersPrefillEnabled {
            Self.cachedParametersSequenceKernelName(
                bufferPrecision: context.kernelContext.bufferPrecision,
                weightFormat: context.kernelContext.weightFormat
            )
        } else if parallelStatePrefillEnabled {
            Self.parallelStateSequenceKernelName(
                bufferPrecision: context.kernelContext.bufferPrecision,
                weightFormat: context.kernelContext.weightFormat
            )
        } else if sharedRMSPrefillEnabled {
            Self.sharedRMSSequenceKernelName(
                bufferPrecision: context.kernelContext.bufferPrecision,
                weightFormat: context.kernelContext.weightFormat
            )
        } else {
            Self.sequenceKernelName(
                bufferPrecision: context.kernelContext.bufferPrecision,
                weightFormat: context.kernelContext.weightFormat
            )
        }
        let pipeline = try context.getPipeline(kernelName)
        let convDebugEnabled = Self.isConvDebugPrefillEnabled
        let convDebugBuffer: MTLBuffer
        if convDebugEnabled {
            guard let buffer = context.buffers.ssmConvDebug else {
                throw MetalCompilerError.deviceSetupFailed(
                    "SWIFTLM_PREFILL_DEBUG_SSM_CONV requires an SSM conv debug buffer"
                )
            }
            convDebugBuffer = buffer
        } else {
            convDebugBuffer = context.buffers.scratch
        }
        // Size threadgroup to cover Phase 1 (localDim channels) and Phase 2 (headsPerGroup × dv threads).
        // Each threadgroup owns one key-group and runs independently on its own GPU core.
        let safeGroupCount = max(groupCount, 1)
        let headsPerGroup = max(1, headCount / safeGroupCount)
        let localDim = 2 * keyHeadDimension + headsPerGroup * valueHeadDimension
        let phase2Threads = headsPerGroup * min(valueHeadDimension, 256)
        let desiredThreads = max(localDim, phase2Threads)
        let defaultThreads = min(
            min(Self.maxThreadgroupSize, desiredThreads),
            pipeline.maxTotalThreadsPerThreadgroup
        )
        let minimumActiveThreads = headsPerGroup * min(valueHeadDimension, 256)
        let threads = try Self.prefillThreadgroupWidthOverride(
            defaultThreads: defaultThreads,
            minimumActiveThreads: minimumActiveThreads,
            simdWidth: max(pipeline.threadExecutionWidth, 1)
        )
        let step = MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: safeGroupCount, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, context.buffers.scratch, 1 * scratchSlotSize),
                (1, context.buffers.scratch, 2 * scratchSlotSize),
                (2, context.buffers.scratch, 3 * scratchSlotSize),
                (3, context.buffers.scratch, 4 * scratchSlotSize),
                (4, convWeightBuffer, convWeightOffset),
                (5, normWeightBuffer, normWeightOffset),
                (6, dtBiasBuffer, dtBiasOffset),
                (7, aLogBuffer, aLogOffset),
                (8, recurrentState, recurrentLayerOffset),
                (9, convState, convLayerOffset),
                (10, context.buffers.scratch, 0),
                (18, convDebugBuffer, 0),
            ],
            bytesBindings: [
                uint32Binding(11, UInt32(headCount)),
                uint32Binding(12, UInt32(groupCount)),
                uint32Binding(13, UInt32(keyHeadDimension)),
                uint32Binding(14, UInt32(valueHeadDimension)),
                uint32Binding(15, UInt32(convKernelSize)),
                uint32Binding(16, 1),
                uint32Binding(17, UInt32(context.slotDimension)),
                uint32Binding(19, UInt32(context.slotDimension)),
                uint32Binding(20, UInt32(convDebugEnabled ? 1 : 0)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bind(index: 16),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                bufferAccessPattern: .init(
                    reads: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    writes: convDebugEnabled ? [8, 9, 10, 18] : [8, 9, 10]
                )
            )
        )

        return FragmentPrefillSteps(
            steps: [step],
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesConvLayer: true,
            consumesRecurrentLayer: true
        )
    }
}

extension SSMRecurrenceFragment: ConvStateRequiring {
    public var convStateDimension: Int { convDimension }
}

extension SSMRecurrenceFragment: RecurrentStateRequiring {
    public var recurrentStateBytesPerLayer: Int {
        headCount * keyHeadDimension * valueHeadDimension * MemoryLayout<Float>.size
    }
}
