import Metal

struct MetalDecodeExecutor: Sendable {

    func decode(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        position: inout Int,
        pendingCommandBuffer: inout MTLCommandBuffer?,
        hasPendingResult: inout Bool,
        tokenID: Int32
    ) -> Int32 {
        let buffers = plan.buffers
        var result: Int32 = -1

        if hasPendingResult {
            result = consumePendingDecodeResult(
                plan: plan,
                submission: submission,
                pendingCommandBuffer: pendingCommandBuffer
            )
        }

        buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID

        do {
            let commandBuffer = try submission.withCompute(label: "decode", waitUntilCompleted: false) { encoder in
                encodeSteps(plan: plan, on: encoder)
            }
            pendingCommandBuffer = commandBuffer
            hasPendingResult = true
            position += 1
        } catch {
            print("[MetalInference] Failed to submit decode: \(error)")
        }

        return result
    }

    func decodeSync(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        position: inout Int,
        tokenID: Int32
    ) -> Int32 {
        let buffers = plan.buffers
        buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID

        do {
            _ = try submission.withCompute(label: "decode.sync") { encoder in
                encodeSteps(plan: plan, on: encoder)
            }
            position += 1
            return buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        } catch {
            print("[MetalInference] GPU error: \(error)")
            return -1
        }
    }

    func flush(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        pendingCommandBuffer: inout MTLCommandBuffer?,
        hasPendingResult: inout Bool
    ) -> Int32 {
        guard hasPendingResult else {
            return -1
        }
        let result = consumePendingDecodeResult(
            plan: plan,
            submission: submission,
            pendingCommandBuffer: pendingCommandBuffer
        )
        hasPendingResult = false
        pendingCommandBuffer = nil
        return result
    }

    private func encodeSteps(plan: MetalDispatchPlan, on encoder: MTLComputeCommandEncoder) {
        for step in plan.steps {
            step.bindings.bind(to: encoder)
            step.descriptor.encode(on: encoder)
        }
    }

    private func consumePendingDecodeResult(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        pendingCommandBuffer: MTLCommandBuffer?
    ) -> Int32 {
        guard let pendingCommandBuffer else {
            return -1
        }
        do {
            try submission.waitUntilCompleted(pendingCommandBuffer)
            return plan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        } catch {
            print("[MetalInference] Pending decode failed: \(error)")
            return -1
        }
    }
}
