import Metal

struct MetalPromptStateStore: Sendable {

    func makePromptState(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        position: Int,
        firstToken: Int32
    ) throws -> MetalPromptState {
        let device = submission.device
        let snapshotKVKeys = try plan.buffers.kvCache.map {
            try makePrivateBuffer(length: $0.keys.length, device: device)
        }
        let snapshotKVValues = try plan.buffers.kvCache.map {
            try makePrivateBuffer(length: $0.values.length, device: device)
        }
        let snapshotConvState = try plan.buffers.convState.map {
            try makePrivateBuffer(length: $0.length, device: device)
        }

        _ = try submission.withTransaction(label: "prompt.snapshot") { transaction in
            try transaction.withBlitEncoder { blit in
                if let liveKV = plan.buffers.kvCache,
                   let snapshotKVKeys,
                   let snapshotKVValues {
                    blit.copy(from: liveKV.keys, sourceOffset: 0, to: snapshotKVKeys, destinationOffset: 0, size: liveKV.keys.length)
                    blit.copy(from: liveKV.values, sourceOffset: 0, to: snapshotKVValues, destinationOffset: 0, size: liveKV.values.length)
                }
                if let liveConvState = plan.buffers.convState,
                   let snapshotConvState {
                    blit.copy(from: liveConvState, sourceOffset: 0, to: snapshotConvState, destinationOffset: 0, size: liveConvState.length)
                }
            }
        }

        return MetalPromptState(
            position: position,
            firstToken: firstToken,
            kvKeys: snapshotKVKeys,
            kvValues: snapshotKVValues,
            convState: snapshotConvState
        )
    }

    func restore(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        promptState: MetalPromptState
    ) throws {
        _ = try submission.withTransaction(label: "prompt.restore") { transaction in
            try transaction.withBlitEncoder { blit in
                blit.fill(buffer: plan.buffers.hidden, range: 0..<plan.buffers.hidden.length, value: 0)
                blit.fill(buffer: plan.buffers.residual, range: 0..<plan.buffers.residual.length, value: 0)
                blit.fill(buffer: plan.buffers.scratch, range: 0..<plan.buffers.scratch.length, value: 0)
                blit.fill(buffer: plan.buffers.logits, range: 0..<plan.buffers.logits.length, value: 0)

                if let liveKV = plan.buffers.kvCache,
                   let snapshotKVKeys = promptState.kvKeys,
                   let snapshotKVValues = promptState.kvValues {
                    blit.copy(from: snapshotKVKeys, sourceOffset: 0, to: liveKV.keys, destinationOffset: 0, size: liveKV.keys.length)
                    blit.copy(from: snapshotKVValues, sourceOffset: 0, to: liveKV.values, destinationOffset: 0, size: liveKV.values.length)
                }
                if let liveConvState = plan.buffers.convState,
                   let snapshotConvState = promptState.convState {
                    blit.copy(from: snapshotConvState, sourceOffset: 0, to: liveConvState, destinationOffset: 0, size: liveConvState.length)
                }
            }
        }
    }

    private func makePrivateBuffer(length: Int, device: MTLDevice) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: length, options: .storageModePrivate) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate prompt state buffer")
        }
        return buffer
    }
}
