import Metal

/// Command submission infrastructure using Metal 4 command queue APIs.
///
/// Wraps `MTL4CommandQueue`, a reusable `MTL4ArgumentTable`, and a pool of
/// `MTL4CommandAllocator` instances for efficient per-token decode submission.
/// Command buffers remain fresh per submission until replay correctness is
/// verified across prompt-state and real-model decode runs.
struct MetalSubmissionContext: @unchecked Sendable {
    let queue: MTL4CommandQueue
    let argumentTable: MTL4ArgumentTable
    private let allocators: [MTL4CommandAllocator]
    private let completionSemaphore: DispatchSemaphore
    private let usesFreshSubmissionResources: Bool
    private let usesFreshArgumentTable: Bool
    private var frameIndex: Int = 0

    static let maxInFlight = 2
    static let maxBufferBindCount = 31

    private init(
        queue: MTL4CommandQueue,
        argumentTable: MTL4ArgumentTable,
        allocators: [MTL4CommandAllocator],
        completionSemaphore: DispatchSemaphore,
        usesFreshSubmissionResources: Bool,
        usesFreshArgumentTable: Bool
    ) {
        self.queue = queue
        self.argumentTable = argumentTable
        self.allocators = allocators
        self.completionSemaphore = completionSemaphore
        self.usesFreshSubmissionResources = usesFreshSubmissionResources
        self.usesFreshArgumentTable = usesFreshArgumentTable
    }

    init(device: MTLDevice) throws {
        guard let queue = device.makeMTL4CommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create MTL4CommandQueue")
        }
        self.queue = queue

        var allocators: [MTL4CommandAllocator] = []
        for i in 0..<Self.maxInFlight {
            guard let allocator = device.makeCommandAllocator() else {
                throw MetalCompilerError.deviceSetupFailed("Cannot create MTL4CommandAllocator[\(i)]")
            }
            allocators.append(allocator)
        }
        self.allocators = allocators

        let atDesc = MTL4ArgumentTableDescriptor()
        atDesc.maxBufferBindCount = Self.maxBufferBindCount
        do {
            self.argumentTable = try device.makeArgumentTable(descriptor: atDesc)
        } catch {
            throw MetalCompilerError.deviceSetupFailed(
                "Cannot create MTL4ArgumentTable: \(error)"
            )
        }
        self.completionSemaphore = DispatchSemaphore(value: 0)

        let environment = ProcessInfo.processInfo.environment
        self.usesFreshSubmissionResources = environment["SWIFTLM_METAL_FRESH_SUBMISSION"] == "1"
        self.usesFreshArgumentTable = environment["SWIFTLM_METAL_FRESH_ARGUMENT_TABLE"] == "1"
    }

    /// Submit a compute pass using Metal 4 APIs.
    ///
    /// The allocator is rotated each call. The caller must ensure the previous
    /// submission on the same allocator slot has completed before calling again.
    mutating func withCompute(
        ephemeralResidency: MetalResidencyLease = .empty,
        waitUntilCompleted: Bool = true,
        _ encode: (MTL4ComputeCommandEncoder, MTL4ArgumentTable) throws -> Void
    ) throws {
        let useFreshArgumentTable = shouldUseFreshArgumentTable(waitUntilCompleted: waitUntilCompleted)
        let allocator = allocators[frameIndex % Self.maxInFlight]
        frameIndex += 1

        // Reusing MTL4CommandBuffer across submissions has produced prompt-state and
        // decode corruption in real-model runs. Always use a fresh command buffer.
        guard let commandBuffer = device.makeCommandBuffer() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create fresh MTL4CommandBuffer")
        }

        let argumentTable: MTL4ArgumentTable
        if useFreshArgumentTable {
            let descriptor = MTL4ArgumentTableDescriptor()
            descriptor.maxBufferBindCount = Self.maxBufferBindCount
            do {
                argumentTable = try device.makeArgumentTable(descriptor: descriptor)
            } catch {
                throw MetalCompilerError.deviceSetupFailed("Cannot create fresh MTL4ArgumentTable")
            }
        } else {
            argumentTable = self.argumentTable
        }

        allocator.reset()
        commandBuffer.beginCommandBuffer(allocator: allocator)
        ephemeralResidency.use(on: commandBuffer)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            commandBuffer.endCommandBuffer()
            throw MetalCompilerError.deviceSetupFailed("Cannot create MTL4ComputeCommandEncoder")
        }
        do {
            try encode(encoder, argumentTable)
        } catch {
            encoder.endEncoding()
            commandBuffer.endCommandBuffer()
            throw error
        }
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()

        if waitUntilCompleted {
            nonisolated(unsafe) var gpuError: NSError?
            let semaphore = completionSemaphore

            let options = MTL4CommitOptions()
            options.addFeedbackHandler { feedback in
                gpuError = feedback.error as NSError?
                semaphore.signal()
            }
            queue.commit([commandBuffer], options: options)
            semaphore.wait()

            if let error = gpuError {
                throw MetalCompilerError.deviceSetupFailed("GPU error: \(error.localizedDescription)")
            }
        } else {
            queue.commit([commandBuffer])
        }
    }

    /// Submit with GPU timing feedback.
    mutating func withComputeTimed(
        ephemeralResidency: MetalResidencyLease = .empty,
        _ encode: (MTL4ComputeCommandEncoder, MTL4ArgumentTable) throws -> Void
    ) throws -> (gpuStartTime: CFTimeInterval, gpuEndTime: CFTimeInterval) {
        let useFreshArgumentTable = shouldUseFreshArgumentTable(waitUntilCompleted: true)
        let allocator = allocators[frameIndex % Self.maxInFlight]
        frameIndex += 1

        guard let commandBuffer = device.makeCommandBuffer() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create fresh MTL4CommandBuffer")
        }

        let argumentTable: MTL4ArgumentTable
        if useFreshArgumentTable {
            let descriptor = MTL4ArgumentTableDescriptor()
            descriptor.maxBufferBindCount = Self.maxBufferBindCount
            do {
                argumentTable = try device.makeArgumentTable(descriptor: descriptor)
            } catch {
                throw MetalCompilerError.deviceSetupFailed("Cannot create fresh MTL4ArgumentTable")
            }
        } else {
            argumentTable = self.argumentTable
        }

        allocator.reset()
        commandBuffer.beginCommandBuffer(allocator: allocator)
        ephemeralResidency.use(on: commandBuffer)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            commandBuffer.endCommandBuffer()
            throw MetalCompilerError.deviceSetupFailed("Cannot create MTL4ComputeCommandEncoder")
        }
        do {
            try encode(encoder, argumentTable)
        } catch {
            encoder.endEncoding()
            commandBuffer.endCommandBuffer()
            throw error
        }
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()

        nonisolated(unsafe) var gpuError: NSError?
        nonisolated(unsafe) var gpuStartTime: CFTimeInterval = 0
        nonisolated(unsafe) var gpuEndTime: CFTimeInterval = 0
        let semaphore = completionSemaphore

        let options = MTL4CommitOptions()
        options.addFeedbackHandler { feedback in
            gpuError = feedback.error as NSError?
            gpuStartTime = feedback.gpuStartTime
            gpuEndTime = feedback.gpuEndTime
            semaphore.signal()
        }
        queue.commit([commandBuffer], options: options)
        semaphore.wait()

        if let error = gpuError {
            throw MetalCompilerError.deviceSetupFailed("GPU error: \(error.localizedDescription)")
        }
        return (gpuStartTime, gpuEndTime)
    }

    /// Fill buffer contents using unified compute+blit encoder.
    ///
    /// A device-visibility barrier is inserted after the fills to ensure
    /// `hazardTrackingModeUntracked` buffers are flushed from any GPU cache
    /// before subsequent kernels read them in a later command buffer.
    mutating func fillBuffers(
        _ fills: [(buffer: MTLBuffer, value: UInt8)],
        ephemeralResidency: MetalResidencyLease = .empty
    ) throws {
        try withCompute(ephemeralResidency: ephemeralResidency) { encoder, _ in
            for fill in fills {
                encoder.fill(buffer: fill.buffer, range: 0..<fill.buffer.length, value: fill.value)
            }
            encoder.barrier(
                afterEncoderStages: [.dispatch, .blit],
                beforeEncoderStages: .dispatch,
                visibilityOptions: .device
            )
        }
    }

    /// Copy buffers using unified compute+blit encoder.
    ///
    /// A device-visibility barrier is inserted before the copies to ensure
    /// `hazardTrackingModeUntracked` source buffers are flushed from any GPU
    /// cache left over from previous command buffers.
    mutating func copyBuffers(
        _ copies: [(from: MTLBuffer, sourceOffset: Int, to: MTLBuffer, destinationOffset: Int, size: Int)],
        ephemeralResidency: MetalResidencyLease = .empty
    ) throws {
        try withCompute(ephemeralResidency: ephemeralResidency) { encoder, _ in
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .blit,
                visibilityOptions: .device
            )
            for copy in copies {
                encoder.copy(
                    sourceBuffer: copy.from, sourceOffset: copy.sourceOffset,
                    destinationBuffer: copy.to, destinationOffset: copy.destinationOffset,
                    size: copy.size
                )
            }
            encoder.barrier(
                afterEncoderStages: .blit,
                beforeEncoderStages: .dispatch,
                visibilityOptions: .device
            )
        }
    }

    var device: MTLDevice { queue.device }

    private func shouldUseFreshArgumentTable(waitUntilCompleted: Bool) -> Bool {
        if usesFreshSubmissionResources {
            return true
        }
        if usesFreshArgumentTable {
            return true
        }
        // The shared argument table is safe only when this call waits for GPU
        // completion before returning. Asynchronous callers get an isolated table
        // so later submissions cannot mutate descriptors still in flight.
        return !waitUntilCompleted
    }

    mutating func resetReuseState() {
        frameIndex = 0
    }

    func makeReplayContext() throws -> MetalSubmissionContext {
        var allocators: [MTL4CommandAllocator] = []
        for i in 0..<Self.maxInFlight {
            guard let allocator = device.makeCommandAllocator() else {
                throw MetalCompilerError.deviceSetupFailed("Cannot create replay MTL4CommandAllocator[\(i)]")
            }
            allocators.append(allocator)
        }

        let atDesc = MTL4ArgumentTableDescriptor()
        atDesc.maxBufferBindCount = Self.maxBufferBindCount
        do {
            let argumentTable = try device.makeArgumentTable(descriptor: atDesc)
            return MetalSubmissionContext(
                queue: queue,
                argumentTable: argumentTable,
                allocators: allocators,
                completionSemaphore: DispatchSemaphore(value: 0),
                usesFreshSubmissionResources: usesFreshSubmissionResources,
                usesFreshArgumentTable: usesFreshArgumentTable
            )
        } catch {
            throw MetalCompilerError.deviceSetupFailed("Cannot create replay MTL4ArgumentTable")
        }
    }
}
