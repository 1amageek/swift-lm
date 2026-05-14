import Metal
import Testing
@testable import MetalCompiler

@Suite("SSM Recurrence Sequence Equivalence", .serialized)
struct SSMRecurrenceSequenceEquivalenceTests {
    @Test("BF16 SSM sequence recurrence matches repeated decode recurrence")
    func bf16SSMSequenceRecurrenceMatchesRepeatedDecodeRecurrence() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let headCount = 16
        let groupCount = 16
        let keyDimension = 128
        let valueDimension = 128
        let convKernelSize = 4
        let sequenceLength = 19
        let keyGroupDimension = groupCount * keyDimension
        let convDimension = 2 * keyGroupDimension + headCount * valueDimension
        let outputDimension = headCount * valueDimension
        let decodeKernelName = "ssm_recurrence_bf16_decode_equivalence"
        let sequenceKernelName = "ssm_recurrence_bf16_sequence_equivalence"
        let sharedRMSSequenceKernelName = "ssm_recurrence_bf16_sequence_shared_rms_equivalence"
        let prewriteDecaySequenceKernelName = "ssm_recurrence_bf16_sequence_prewrite_decay_equivalence"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateSSMWeightIndependentHelpers(),
            MetalSourceGenerator.generateSSMConvSiluHelper(weightFormat: .bfloat16),
            MetalSourceGenerator.generateSSMRecurrence(
                name: decodeKernelName,
                bufferPrecision: .bfloat16,
                weightFormat: .bfloat16,
                convDimension: convDimension,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyDimension,
                valueHeadDimension: valueDimension
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: sequenceKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: convDimension,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyDimension,
                valueHeadDimension: valueDimension
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: sharedRMSSequenceKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: convDimension,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyDimension,
                valueHeadDimension: valueDimension,
                shareRMSScale: true
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: prewriteDecaySequenceKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: convDimension,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyDimension,
                valueHeadDimension: valueDimension,
                prewriteDecayedState: true
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let decodePipeline = try harness.pipeline(named: decodeKernelName)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)
        let sharedRMSSequencePipeline = try harness.pipeline(named: sharedRMSSequenceKernelName)
        let prewriteDecaySequencePipeline = try harness.pipeline(named: prewriteDecaySequenceKernelName)

        let projectedQKV = roundedBFloat16Values(
            count: sequenceLength * convDimension,
            multiplier: 13,
            modulus: 23,
            scale: 0.125
        )
        let projectedZ = roundedBFloat16Values(
            count: sequenceLength * outputDimension,
            multiplier: 17,
            modulus: 19,
            scale: 0.125
        )
        let projectedBeta = roundedBFloat16Values(
            count: sequenceLength * headCount,
            multiplier: 7,
            modulus: 11,
            scale: 0.125
        )
        let projectedAlpha = roundedBFloat16Values(
            count: sequenceLength * headCount,
            multiplier: 5,
            modulus: 13,
            scale: 0.125
        )
        let convWeight = (0..<(convDimension * convKernelSize)).map { index in
            BFloat16(Float((index * 11) % 17 - 8) * 0.03125)
        }
        let normWeight = (0..<valueDimension).map { index in
            0.75 + Float(index) * 0.0625
        }
        let dtBias = (0..<headCount).map { index in
            BFloat16(Float(index - 1) * 0.03125)
        }
        let aLog = (0..<headCount).map { index in
            Float(index) * 0.0625 - 0.125
        }

        let decode = try runDecodeSSMTrace(
            harness: harness,
            pipeline: decodePipeline,
            projectedQKV: projectedQKV,
            projectedZ: projectedZ,
            projectedBeta: projectedBeta,
            projectedAlpha: projectedAlpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            sequenceLength: sequenceLength,
            convDimension: convDimension,
            outputDimension: outputDimension
        )
        let sequence = try runSequenceSSMTrace(
            harness: harness,
            pipeline: sequencePipeline,
            projectedQKV: projectedQKV,
            projectedZ: projectedZ,
            projectedBeta: projectedBeta,
            projectedAlpha: projectedAlpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            sequenceLength: sequenceLength,
            convDimension: convDimension,
            outputDimension: outputDimension
        )
        let sharedRMSSequence = try runSequenceSSMTrace(
            harness: harness,
            pipeline: sharedRMSSequencePipeline,
            projectedQKV: projectedQKV,
            projectedZ: projectedZ,
            projectedBeta: projectedBeta,
            projectedAlpha: projectedAlpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            sequenceLength: sequenceLength,
            convDimension: convDimension,
            outputDimension: outputDimension
        )
        let prewriteDecaySequence = try runSequenceSSMTrace(
            harness: harness,
            pipeline: prewriteDecaySequencePipeline,
            projectedQKV: projectedQKV,
            projectedZ: projectedZ,
            projectedBeta: projectedBeta,
            projectedAlpha: projectedAlpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            sequenceLength: sequenceLength,
            convDimension: convDimension,
            outputDimension: outputDimension
        )

        let outputMismatch = harness.firstMismatch(
            expected: decode.output,
            actual: sequence.output,
            tolerance: 0.000_01
        )
        #expect(
            outputMismatch == nil,
            "SSM output drifted: \(String(describing: outputMismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.output, actual: sequence.output))"
        )
        let recurrentMismatch = harness.firstMismatch(
            expected: decode.recurrentState,
            actual: sequence.recurrentState,
            tolerance: 0.000_01
        )
        #expect(
            recurrentMismatch == nil,
            "SSM recurrent state drifted: \(String(describing: recurrentMismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.recurrentState, actual: sequence.recurrentState))"
        )
        #expect(
            decode.convStateBits == sequence.convStateBits,
            "SSM conv state drifted: decode=\(decode.convStateBits), sequence=\(sequence.convStateBits)"
        )
        let sharedRMSOutputMismatch = harness.firstMismatch(
            expected: decode.output,
            actual: sharedRMSSequence.output,
            tolerance: 0.000_01
        )
        #expect(
            sharedRMSOutputMismatch == nil,
            "Shared-RMS SSM output drifted: \(String(describing: sharedRMSOutputMismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.output, actual: sharedRMSSequence.output))"
        )
        let sharedRMSRecurrentMismatch = harness.firstMismatch(
            expected: decode.recurrentState,
            actual: sharedRMSSequence.recurrentState,
            tolerance: 0.000_01
        )
        #expect(
            sharedRMSRecurrentMismatch == nil,
            "Shared-RMS SSM recurrent state drifted: \(String(describing: sharedRMSRecurrentMismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.recurrentState, actual: sharedRMSSequence.recurrentState))"
        )
        #expect(
            decode.convStateBits == sharedRMSSequence.convStateBits,
            "Shared-RMS SSM conv state drifted: decode=\(decode.convStateBits), sequence=\(sharedRMSSequence.convStateBits)"
        )
        let prewriteDecayOutputMismatch = harness.firstMismatch(
            expected: decode.output,
            actual: prewriteDecaySequence.output,
            tolerance: 0.000_01
        )
        #expect(
            prewriteDecayOutputMismatch == nil,
            "Prewrite-decay SSM output drifted: \(String(describing: prewriteDecayOutputMismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.output, actual: prewriteDecaySequence.output))"
        )
        let prewriteDecayRecurrentMismatch = harness.firstMismatch(
            expected: decode.recurrentState,
            actual: prewriteDecaySequence.recurrentState,
            tolerance: 0.000_01
        )
        #expect(
            prewriteDecayRecurrentMismatch == nil,
            "Prewrite-decay SSM recurrent state drifted: \(String(describing: prewriteDecayRecurrentMismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.recurrentState, actual: prewriteDecaySequence.recurrentState))"
        )
        #expect(
            decode.convStateBits == prewriteDecaySequence.convStateBits,
            "Prewrite-decay SSM conv state drifted: decode=\(decode.convStateBits), sequence=\(prewriteDecaySequence.convStateBits)"
        )

        for threadgroupWidth in [128, 256] {
            let narrowSequence = try runSequenceSSMTrace(
                harness: harness,
                pipeline: sequencePipeline,
                projectedQKV: projectedQKV,
                projectedZ: projectedZ,
                projectedBeta: projectedBeta,
                projectedAlpha: projectedAlpha,
                convWeight: convWeight,
                normWeight: normWeight,
                dtBias: dtBias,
                aLog: aLog,
                headCount: headCount,
                groupCount: groupCount,
                keyDimension: keyDimension,
                valueDimension: valueDimension,
                convKernelSize: convKernelSize,
                sequenceLength: sequenceLength,
                convDimension: convDimension,
                outputDimension: outputDimension,
                threadgroupWidthOverride: threadgroupWidth
            )
            let narrowOutputMismatch = harness.firstMismatch(
                expected: decode.output,
                actual: narrowSequence.output,
                tolerance: 0.000_01
            )
            #expect(
                narrowOutputMismatch == nil,
                "SSM output drifted at tg=\(threadgroupWidth): \(String(describing: narrowOutputMismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.output, actual: narrowSequence.output))"
            )
            let narrowRecurrentMismatch = harness.firstMismatch(
                expected: decode.recurrentState,
                actual: narrowSequence.recurrentState,
                tolerance: 0.000_01
            )
            #expect(
                narrowRecurrentMismatch == nil,
                "SSM recurrent state drifted at tg=\(threadgroupWidth): \(String(describing: narrowRecurrentMismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.recurrentState, actual: narrowSequence.recurrentState))"
            )
            #expect(
                decode.convStateBits == narrowSequence.convStateBits,
                "SSM conv state drifted at tg=\(threadgroupWidth): decode=\(String(describing: decode.convStateBits)), sequence=\(String(describing: narrowSequence.convStateBits))"
            )
        }
    }

    @Test("BF16 SSM sequence recurrence can emit group-owned partial projection")
    func bf16SSMSequenceRecurrenceEmitsGroupOwnedPartialProjection() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let headCount = 4
        let groupCount = 4
        let keyDimension = 8
        let valueDimension = 4
        let convKernelSize = 3
        let sequenceLength = 5
        let keyGroupDimension = groupCount * keyDimension
        let convDimension = 2 * keyGroupDimension + headCount * valueDimension
        let recurrentOutputDimension = headCount * valueDimension
        let outputDimension = 7
        let sequenceKernelName = "ssm_recurrence_bf16_sequence_partial_source"
        let partialKernelName = "ssm_recurrence_bf16_sequence_partial_emission"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateSSMWeightIndependentHelpers(),
            MetalSourceGenerator.generateSSMConvSiluHelper(weightFormat: .bfloat16),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: sequenceKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: convDimension,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyDimension,
                valueHeadDimension: valueDimension
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: partialKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: convDimension,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyDimension,
                valueHeadDimension: valueDimension,
                emitsGroupOwnedPartialProjection: true
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)
        let partialPipeline = try harness.pipeline(named: partialKernelName)

        let projectedQKV = roundedBFloat16Values(
            count: sequenceLength * convDimension,
            multiplier: 13,
            modulus: 23,
            scale: 0.125
        )
        let projectedZ = roundedBFloat16Values(
            count: sequenceLength * recurrentOutputDimension,
            multiplier: 17,
            modulus: 19,
            scale: 0.125
        )
        let projectedBeta = roundedBFloat16Values(
            count: sequenceLength * headCount,
            multiplier: 7,
            modulus: 11,
            scale: 0.125
        )
        let projectedAlpha = roundedBFloat16Values(
            count: sequenceLength * headCount,
            multiplier: 5,
            modulus: 13,
            scale: 0.125
        )
        let convWeight = (0..<(convDimension * convKernelSize)).map { index in
            BFloat16(Float((index * 11) % 17 - 8) * 0.03125)
        }
        let normWeight = (0..<valueDimension).map { index in
            0.75 + Float(index) * 0.0625
        }
        let dtBias = (0..<headCount).map { index in
            BFloat16(Float(index - 1) * 0.03125)
        }
        let aLog = (0..<headCount).map { index in
            Float(index) * 0.0625 - 0.125
        }
        let partialWeight = (0..<(outputDimension * recurrentOutputDimension)).map { index in
            BFloat16(Float((index * 11) % 31 - 15) * 0.015625)
        }

        let sequence = try runSequenceSSMTrace(
            harness: harness,
            pipeline: sequencePipeline,
            projectedQKV: projectedQKV,
            projectedZ: projectedZ,
            projectedBeta: projectedBeta,
            projectedAlpha: projectedAlpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            sequenceLength: sequenceLength,
            convDimension: convDimension,
            outputDimension: recurrentOutputDimension
        )
        let partial = try runSequenceSSMTraceWithPartialProjection(
            harness: harness,
            pipeline: partialPipeline,
            projectedQKV: projectedQKV,
            projectedZ: projectedZ,
            projectedBeta: projectedBeta,
            projectedAlpha: projectedAlpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog,
            partialWeight: partialWeight,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            sequenceLength: sequenceLength,
            convDimension: convDimension,
            recurrentOutputDimension: recurrentOutputDimension,
            partialOutputDimension: outputDimension
        )

        let outputMismatch = harness.firstMismatch(
            expected: sequence.output,
            actual: partial.output,
            tolerance: 0.000_01
        )
        #expect(
            outputMismatch == nil,
            "Partial-emitting SSM output drifted: \(String(describing: outputMismatch)), maxError=\(harness.maxAbsoluteError(expected: sequence.output, actual: partial.output))"
        )
        let recurrentMismatch = harness.firstMismatch(
            expected: sequence.recurrentState,
            actual: partial.recurrentState,
            tolerance: 0.000_01
        )
        #expect(
            recurrentMismatch == nil,
            "Partial-emitting SSM recurrent state drifted: \(String(describing: recurrentMismatch)), maxError=\(harness.maxAbsoluteError(expected: sequence.recurrentState, actual: partial.recurrentState))"
        )
        #expect(
            sequence.convStateBits == partial.convStateBits,
            "Partial-emitting SSM conv state drifted"
        )

        let expectedPartials = expectedGroupOwnedPartials(
            input: sequence.output,
            weights: partialWeight,
            groupCount: groupCount,
            partitionInputDimension: recurrentOutputDimension / groupCount,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )
        let partialMismatch = harness.firstMismatch(
            expected: expectedPartials,
            actual: partial.partials,
            tolerance: 0.000_01
        )
        #expect(
            partialMismatch == nil,
            "Partial-emitting SSM partials drifted: \(String(describing: partialMismatch)), maxError=\(harness.maxAbsoluteError(expected: expectedPartials, actual: partial.partials))"
        )
    }

    @Test("BF16 SSM sequence recurrence can emit partition-owned partial projection")
    func bf16SSMSequenceRecurrenceEmitsPartitionOwnedPartialProjection() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let headCount = 8
        let groupCount = 8
        let partitionCount = 4
        let keyDimension = 8
        let valueDimension = 4
        let convKernelSize = 3
        let sequenceLength = 5
        let keyGroupDimension = groupCount * keyDimension
        let convDimension = 2 * keyGroupDimension + headCount * valueDimension
        let recurrentOutputDimension = headCount * valueDimension
        let outputDimension = 7
        let sequenceKernelName = "ssm_recurrence_bf16_sequence_partition_partial_source"
        let partialKernelName = "ssm_recurrence_bf16_sequence_partition_partial_emission"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateSSMWeightIndependentHelpers(),
            MetalSourceGenerator.generateSSMConvSiluHelper(weightFormat: .bfloat16),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: sequenceKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: convDimension,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyDimension,
                valueHeadDimension: valueDimension
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: partialKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: convDimension,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyDimension,
                valueHeadDimension: valueDimension,
                emitsPartitionOwnedPartialProjection: true
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)
        let partialPipeline = try harness.pipeline(named: partialKernelName)

        let projectedQKV = roundedBFloat16Values(
            count: sequenceLength * convDimension,
            multiplier: 13,
            modulus: 23,
            scale: 0.125
        )
        let projectedZ = roundedBFloat16Values(
            count: sequenceLength * recurrentOutputDimension,
            multiplier: 17,
            modulus: 19,
            scale: 0.125
        )
        let projectedBeta = roundedBFloat16Values(
            count: sequenceLength * headCount,
            multiplier: 7,
            modulus: 11,
            scale: 0.125
        )
        let projectedAlpha = roundedBFloat16Values(
            count: sequenceLength * headCount,
            multiplier: 5,
            modulus: 13,
            scale: 0.125
        )
        let convWeight = (0..<(convDimension * convKernelSize)).map { index in
            BFloat16(Float((index * 11) % 17 - 8) * 0.03125)
        }
        let normWeight = (0..<valueDimension).map { index in
            0.75 + Float(index) * 0.0625
        }
        let dtBias = (0..<headCount).map { index in
            BFloat16(Float(index - 1) * 0.03125)
        }
        let aLog = (0..<headCount).map { index in
            Float(index) * 0.0625 - 0.125
        }
        let partialWeight = (0..<(outputDimension * recurrentOutputDimension)).map { index in
            BFloat16(Float((index * 11) % 31 - 15) * 0.015625)
        }

        let sequence = try runSequenceSSMTrace(
            harness: harness,
            pipeline: sequencePipeline,
            projectedQKV: projectedQKV,
            projectedZ: projectedZ,
            projectedBeta: projectedBeta,
            projectedAlpha: projectedAlpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            sequenceLength: sequenceLength,
            convDimension: convDimension,
            outputDimension: recurrentOutputDimension
        )
        let partial = try runSequenceSSMTraceWithPartialProjection(
            harness: harness,
            pipeline: partialPipeline,
            projectedQKV: projectedQKV,
            projectedZ: projectedZ,
            projectedBeta: projectedBeta,
            projectedAlpha: projectedAlpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog,
            partialWeight: partialWeight,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            sequenceLength: sequenceLength,
            convDimension: convDimension,
            recurrentOutputDimension: recurrentOutputDimension,
            partialOutputDimension: outputDimension,
            partialPartitionCount: partitionCount
        )

        let outputMismatch = harness.firstMismatch(
            expected: sequence.output,
            actual: partial.output,
            tolerance: 0.000_01
        )
        #expect(
            outputMismatch == nil,
            "Partition partial SSM output drifted: \(String(describing: outputMismatch)), maxError=\(harness.maxAbsoluteError(expected: sequence.output, actual: partial.output))"
        )
        let recurrentMismatch = harness.firstMismatch(
            expected: sequence.recurrentState,
            actual: partial.recurrentState,
            tolerance: 0.000_01
        )
        #expect(
            recurrentMismatch == nil,
            "Partition partial SSM recurrent state drifted: \(String(describing: recurrentMismatch)), maxError=\(harness.maxAbsoluteError(expected: sequence.recurrentState, actual: partial.recurrentState))"
        )
        #expect(
            sequence.convStateBits == partial.convStateBits,
            "Partition partial SSM conv state drifted"
        )

        let expectedPartials = expectedPartitionOwnedPartials(
            input: sequence.output,
            weights: partialWeight,
            partitionCount: partitionCount,
            partitionInputDimension: recurrentOutputDimension / partitionCount,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )
        let partialMismatch = harness.firstMismatch(
            expected: expectedPartials,
            actual: partial.partials,
            tolerance: 0.000_01
        )
        #expect(
            partialMismatch == nil,
            "Partition partial SSM partials drifted: \(String(describing: partialMismatch)), maxError=\(harness.maxAbsoluteError(expected: expectedPartials, actual: partial.partials))"
        )
    }

    private func roundedBFloat16Values(
        count: Int,
        multiplier: Int,
        modulus: Int,
        scale: Float
    ) -> [Float] {
        (0..<count).map { index in
            Float(BFloat16(Float((index * multiplier) % modulus - modulus / 2) * scale))
        }
    }

    private func runDecodeSSMTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        projectedQKV: [Float],
        projectedZ: [Float],
        projectedBeta: [Float],
        projectedAlpha: [Float],
        convWeight: [BFloat16],
        normWeight: [Float],
        dtBias: [BFloat16],
        aLog: [Float],
        headCount: Int,
        groupCount: Int,
        keyDimension: Int,
        valueDimension: Int,
        convKernelSize: Int,
        sequenceLength: Int,
        convDimension: Int,
        outputDimension: Int
    ) throws -> (output: [Float], recurrentState: [Float], convStateBits: [UInt16]) {
        let convWeightBuffer = try harness.makeSharedBuffer(values: convWeight)
        let normWeightBuffer = try harness.makeSharedBuffer(values: normWeight)
        let dtBiasBuffer = try harness.makeSharedBuffer(values: dtBias)
        let aLogBuffer = try harness.makeSharedBuffer(values: aLog)
        let recurrentState = try harness.makeZeroedSharedBuffer(
            byteLength: headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride
        )
        let convState = try harness.makeZeroedSharedBuffer(
            byteLength: convKernelSize * convDimension * MemoryLayout<BFloat16>.stride
        )
        var trace = [Float](repeating: .zero, count: sequenceLength * outputDimension)
        let threads = ssmThreadCount(
            pipeline: pipeline,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension
        )
        let grid = MTLSize(width: max(groupCount, 1), height: 1, depth: 1)
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        for position in 0..<sequenceLength {
            let qkvOffset = position * convDimension
            let zOffset = position * outputDimension
            let headOffset = position * headCount
            let qkvInput = projectedQKV[qkvOffset..<(qkvOffset + convDimension)].map { BFloat16($0) }
            let zInput = projectedZ[zOffset..<(zOffset + outputDimension)].map { BFloat16($0) }
            let betaInput = projectedBeta[headOffset..<(headOffset + headCount)].map { BFloat16($0) }
            let alphaInput = projectedAlpha[headOffset..<(headOffset + headCount)].map { BFloat16($0) }
            let qkvBuffer = try harness.makeSharedBuffer(values: Array(qkvInput))
            let zBuffer = try harness.makeSharedBuffer(values: Array(zInput))
            let betaBuffer = try harness.makeSharedBuffer(values: Array(betaInput))
            let alphaBuffer = try harness.makeSharedBuffer(values: Array(alphaInput))
            let outputBuffer = try harness.makeZeroedSharedBuffer(
                byteLength: outputDimension * MemoryLayout<BFloat16>.stride
            )

            let (commandBuffer, encoder) = try harness.makeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(qkvBuffer, offset: 0, index: 0)
            encoder.setBuffer(zBuffer, offset: 0, index: 1)
            encoder.setBuffer(betaBuffer, offset: 0, index: 2)
            encoder.setBuffer(alphaBuffer, offset: 0, index: 3)
            encoder.setBuffer(convWeightBuffer, offset: 0, index: 4)
            encoder.setBuffer(normWeightBuffer, offset: 0, index: 5)
            encoder.setBuffer(dtBiasBuffer, offset: 0, index: 6)
            encoder.setBuffer(aLogBuffer, offset: 0, index: 7)
            encoder.setBuffer(recurrentState, offset: 0, index: 8)
            encoder.setBuffer(convState, offset: 0, index: 9)
            encoder.setBuffer(outputBuffer, offset: 0, index: 10)
            setSSMConstants(
                encoder: encoder,
                headCount: headCount,
                groupCount: groupCount,
                keyDimension: keyDimension,
                valueDimension: valueDimension,
                convKernelSize: convKernelSize
            )
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
            encoder.endEncoding()
            try harness.complete(commandBuffer)

            let output = harness.readBFloat16AsFloat(outputBuffer, count: outputDimension)
            trace.replaceSubrange(zOffset..<(zOffset + outputDimension), with: output)
        }

        return (
            output: trace,
            recurrentState: harness.readFloat32(
                recurrentState,
                count: headCount * keyDimension * valueDimension
            ),
            convStateBits: harness.readBFloat16Bits(
                convState,
                count: convKernelSize * convDimension
            )
        )
    }

    private func runSequenceSSMTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        projectedQKV: [Float],
        projectedZ: [Float],
        projectedBeta: [Float],
        projectedAlpha: [Float],
        convWeight: [BFloat16],
        normWeight: [Float],
        dtBias: [BFloat16],
        aLog: [Float],
        headCount: Int,
        groupCount: Int,
        keyDimension: Int,
        valueDimension: Int,
        convKernelSize: Int,
        sequenceLength: Int,
        convDimension: Int,
        outputDimension: Int,
        threadgroupWidthOverride: Int? = nil
    ) throws -> (output: [Float], recurrentState: [Float], convStateBits: [UInt16]) {
        let activationRowStride = max(convDimension, outputDimension, headCount)
        let qkvBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedQKV,
            rowCount: sequenceLength,
            logicalWidth: convDimension,
            rowStride: activationRowStride
        ))
        let zBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedZ,
            rowCount: sequenceLength,
            logicalWidth: outputDimension,
            rowStride: activationRowStride
        ))
        let betaBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedBeta,
            rowCount: sequenceLength,
            logicalWidth: headCount,
            rowStride: activationRowStride
        ))
        let alphaBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedAlpha,
            rowCount: sequenceLength,
            logicalWidth: headCount,
            rowStride: activationRowStride
        ))
        let convWeightBuffer = try harness.makeSharedBuffer(values: convWeight)
        let normWeightBuffer = try harness.makeSharedBuffer(values: normWeight)
        let dtBiasBuffer = try harness.makeSharedBuffer(values: dtBias)
        let aLogBuffer = try harness.makeSharedBuffer(values: aLog)
        let recurrentState = try harness.makeZeroedSharedBuffer(
            byteLength: headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride
        )
        let convState = try harness.makeZeroedSharedBuffer(
            byteLength: convKernelSize * convDimension * MemoryLayout<BFloat16>.stride
        )
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: sequenceLength * activationRowStride * MemoryLayout<Float>.stride
        )
        let defaultThreads = ssmThreadCount(
            pipeline: pipeline,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension
        )
        let threads = min(threadgroupWidthOverride ?? defaultThreads, defaultThreads)
        let grid = MTLSize(width: max(groupCount, 1), height: 1, depth: 1)
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qkvBuffer, offset: 0, index: 0)
        encoder.setBuffer(zBuffer, offset: 0, index: 1)
        encoder.setBuffer(betaBuffer, offset: 0, index: 2)
        encoder.setBuffer(alphaBuffer, offset: 0, index: 3)
        encoder.setBuffer(convWeightBuffer, offset: 0, index: 4)
        encoder.setBuffer(normWeightBuffer, offset: 0, index: 5)
        encoder.setBuffer(dtBiasBuffer, offset: 0, index: 6)
        encoder.setBuffer(aLogBuffer, offset: 0, index: 7)
        encoder.setBuffer(recurrentState, offset: 0, index: 8)
        encoder.setBuffer(convState, offset: 0, index: 9)
        encoder.setBuffer(outputBuffer, offset: 0, index: 10)
        encoder.setBuffer(outputBuffer, offset: 0, index: 18)
        setSSMConstants(
            encoder: encoder,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize
        )
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(activationRowStride)
        var debugEnabled: UInt32 = 0
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 17)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 19)
        encoder.setBytes(&debugEnabled, length: MemoryLayout<UInt32>.stride, index: 20)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        let paddedOutput = harness.readFloat32(
            outputBuffer,
            count: sequenceLength * activationRowStride
        )
        var output: [Float] = []
        output.reserveCapacity(sequenceLength * outputDimension)
        for position in 0..<sequenceLength {
            let start = position * activationRowStride
            output.append(contentsOf: paddedOutput[start..<(start + outputDimension)])
        }

        return (
            output: output,
            recurrentState: harness.readFloat32(
                recurrentState,
                count: headCount * keyDimension * valueDimension
            ),
            convStateBits: harness.readBFloat16Bits(
                convState,
                count: convKernelSize * convDimension
            )
        )
    }

    private func runSequenceSSMTraceWithPartialProjection(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        projectedQKV: [Float],
        projectedZ: [Float],
        projectedBeta: [Float],
        projectedAlpha: [Float],
        convWeight: [BFloat16],
        normWeight: [Float],
        dtBias: [BFloat16],
        aLog: [Float],
        partialWeight: [BFloat16],
        headCount: Int,
        groupCount: Int,
        keyDimension: Int,
        valueDimension: Int,
        convKernelSize: Int,
        sequenceLength: Int,
        convDimension: Int,
        recurrentOutputDimension: Int,
        partialOutputDimension: Int,
        partialPartitionCount: Int? = nil
    ) throws -> (output: [Float], recurrentState: [Float], convStateBits: [UInt16], partials: [Float]) {
        let activationRowStride = max(convDimension, recurrentOutputDimension, headCount)
        let partialRowStride = partialOutputDimension + 3
        let effectivePartialPartitionCount = partialPartitionCount ?? groupCount
        let qkvBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedQKV,
            rowCount: sequenceLength,
            logicalWidth: convDimension,
            rowStride: activationRowStride
        ))
        let zBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedZ,
            rowCount: sequenceLength,
            logicalWidth: recurrentOutputDimension,
            rowStride: activationRowStride
        ))
        let betaBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedBeta,
            rowCount: sequenceLength,
            logicalWidth: headCount,
            rowStride: activationRowStride
        ))
        let alphaBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedAlpha,
            rowCount: sequenceLength,
            logicalWidth: headCount,
            rowStride: activationRowStride
        ))
        let convWeightBuffer = try harness.makeSharedBuffer(values: convWeight)
        let normWeightBuffer = try harness.makeSharedBuffer(values: normWeight)
        let dtBiasBuffer = try harness.makeSharedBuffer(values: dtBias)
        let aLogBuffer = try harness.makeSharedBuffer(values: aLog)
        let partialWeightBuffer = try harness.makeSharedBuffer(values: partialWeight)
        let recurrentState = try harness.makeZeroedSharedBuffer(
            byteLength: headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride
        )
        let convState = try harness.makeZeroedSharedBuffer(
            byteLength: convKernelSize * convDimension * MemoryLayout<BFloat16>.stride
        )
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: sequenceLength * activationRowStride * MemoryLayout<Float>.stride
        )
        let partialBuffer = try harness.makeSharedBuffer(values: [Float](
            repeating: -777.0,
            count: effectivePartialPartitionCount * sequenceLength * partialRowStride
        ))
        let threads = ssmThreadCount(
            pipeline: pipeline,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension
        )
        let grid = MTLSize(width: max(effectivePartialPartitionCount, 1), height: 1, depth: 1)
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qkvBuffer, offset: 0, index: 0)
        encoder.setBuffer(zBuffer, offset: 0, index: 1)
        encoder.setBuffer(betaBuffer, offset: 0, index: 2)
        encoder.setBuffer(alphaBuffer, offset: 0, index: 3)
        encoder.setBuffer(convWeightBuffer, offset: 0, index: 4)
        encoder.setBuffer(normWeightBuffer, offset: 0, index: 5)
        encoder.setBuffer(dtBiasBuffer, offset: 0, index: 6)
        encoder.setBuffer(aLogBuffer, offset: 0, index: 7)
        encoder.setBuffer(recurrentState, offset: 0, index: 8)
        encoder.setBuffer(convState, offset: 0, index: 9)
        encoder.setBuffer(outputBuffer, offset: 0, index: 10)
        encoder.setBuffer(outputBuffer, offset: 0, index: 18)
        encoder.setBuffer(partialWeightBuffer, offset: 0, index: 21)
        encoder.setBuffer(partialBuffer, offset: 0, index: 22)
        setSSMConstants(
            encoder: encoder,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize
        )
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(activationRowStride)
        var debugEnabled: UInt32 = 0
        var partialRows = UInt32(partialOutputDimension)
        var partialStride = UInt32(partialRowStride)
        var partialEnabled: UInt32 = 1
        var partitionCount = UInt32(effectivePartialPartitionCount)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 17)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 19)
        encoder.setBytes(&debugEnabled, length: MemoryLayout<UInt32>.stride, index: 20)
        encoder.setBytes(&partialRows, length: MemoryLayout<UInt32>.stride, index: 23)
        encoder.setBytes(&partialStride, length: MemoryLayout<UInt32>.stride, index: 24)
        encoder.setBytes(&partialEnabled, length: MemoryLayout<UInt32>.stride, index: 25)
        if partialPartitionCount != nil {
            encoder.setBytes(&partitionCount, length: MemoryLayout<UInt32>.stride, index: 26)
        }
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        let paddedOutput = harness.readFloat32(
            outputBuffer,
            count: sequenceLength * activationRowStride
        )
        var output: [Float] = []
        output.reserveCapacity(sequenceLength * recurrentOutputDimension)
        for position in 0..<sequenceLength {
            let start = position * activationRowStride
            output.append(contentsOf: paddedOutput[start..<(start + recurrentOutputDimension)])
        }
        let paddedPartial = harness.readFloat32(
            partialBuffer,
            count: effectivePartialPartitionCount * sequenceLength * partialRowStride
        )
        var partials: [Float] = []
        partials.reserveCapacity(effectivePartialPartitionCount * sequenceLength * partialOutputDimension)
        for group in 0..<effectivePartialPartitionCount {
            for position in 0..<sequenceLength {
                let start = group * sequenceLength * partialRowStride + position * partialRowStride
                partials.append(contentsOf: paddedPartial[start..<(start + partialOutputDimension)])
                for row in partialOutputDimension..<partialRowStride {
                    #expect(paddedPartial[start + row] == -777.0)
                }
            }
        }

        return (
            output: output,
            recurrentState: harness.readFloat32(
                recurrentState,
                count: headCount * keyDimension * valueDimension
            ),
            convStateBits: harness.readBFloat16Bits(
                convState,
                count: convKernelSize * convDimension
            ),
            partials: partials
        )
    }

    private func expectedGroupOwnedPartials(
        input: [Float],
        weights: [BFloat16],
        groupCount: Int,
        partitionInputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int
    ) -> [Float] {
        let inputDimension = groupCount * partitionInputDimension
        var partials: [Float] = []
        partials.reserveCapacity(groupCount * sequenceLength * outputDimension)
        for group in 0..<groupCount {
            let groupInputBase = group * partitionInputDimension
            for position in 0..<sequenceLength {
                for row in 0..<outputDimension {
                    var sum: Float = 0
                    for column in 0..<partitionInputDimension {
                        let inputValue = input[position * inputDimension + groupInputBase + column]
                        let weightValue = Float(weights[row * inputDimension + groupInputBase + column])
                        sum += inputValue * weightValue
                    }
                    partials.append(sum)
                }
            }
        }
        return partials
    }

    private func expectedPartitionOwnedPartials(
        input: [Float],
        weights: [BFloat16],
        partitionCount: Int,
        partitionInputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int
    ) -> [Float] {
        let inputDimension = partitionCount * partitionInputDimension
        var partials: [Float] = []
        partials.reserveCapacity(partitionCount * sequenceLength * outputDimension)
        for partition in 0..<partitionCount {
            let partitionInputBase = partition * partitionInputDimension
            for position in 0..<sequenceLength {
                for row in 0..<outputDimension {
                    var sum: Float = 0
                    for column in 0..<partitionInputDimension {
                        let inputValue = input[position * inputDimension + partitionInputBase + column]
                        let weightValue = Float(weights[row * inputDimension + partitionInputBase + column])
                        sum += inputValue * weightValue
                    }
                    partials.append(sum)
                }
            }
        }
        return partials
    }

    private func paddedRows(
        _ values: [Float],
        rowCount: Int,
        logicalWidth: Int,
        rowStride: Int
    ) -> [Float] {
        var padded = [Float](repeating: .zero, count: rowCount * rowStride)
        for row in 0..<rowCount {
            let sourceStart = row * logicalWidth
            let destinationStart = row * rowStride
            padded.replaceSubrange(
                destinationStart..<(destinationStart + logicalWidth),
                with: values[sourceStart..<(sourceStart + logicalWidth)]
            )
        }
        return padded
    }

    private func ssmThreadCount(
        pipeline: MTLComputePipelineState,
        headCount: Int,
        groupCount: Int,
        keyDimension: Int,
        valueDimension: Int
    ) -> Int {
        let safeGroupCount = max(groupCount, 1)
        let headsPerGroup = max(1, headCount / safeGroupCount)
        let localDimension = 2 * keyDimension + headsPerGroup * valueDimension
        let phase2Threads = headsPerGroup * min(valueDimension, 256)
        let desiredThreads = max(localDimension, phase2Threads)
        return min(
            min(SSMRecurrenceFragment.maxThreadgroupSize, desiredThreads),
            pipeline.maxTotalThreadsPerThreadgroup
        )
    }

    private func setSSMConstants(
        encoder: MTLComputeCommandEncoder,
        headCount: Int,
        groupCount: Int,
        keyDimension: Int,
        valueDimension: Int,
        convKernelSize: Int
    ) {
        var heads = UInt32(headCount)
        var groups = UInt32(groupCount)
        var keyDim = UInt32(keyDimension)
        var valueDim = UInt32(valueDimension)
        var kernel = UInt32(convKernelSize)
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&groups, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 15)
    }
}
