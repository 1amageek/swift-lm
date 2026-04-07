import Metal

/// Command submission infrastructure using Metal 4 reusable command buffer pattern.
///
/// Wraps `MTL4CommandQueue`, reusable `MTL4CommandBuffer`, and a pool of
/// `MTL4CommandAllocator` for efficient per-token decode submission.
struct MetalSubmissionContext: @unchecked Sendable {
    let queue: MTL4CommandQueue
    let commandBuffer: MTL4CommandBuffer
    let argumentTable: MTL4ArgumentTable
    private let allocators: [MTL4CommandAllocator]
    private var frameIndex: Int = 0

    static let maxInFlight = 2
    static let maxBufferBindCount = 31

    init(device: MTLDevice) throws {
        guard let queue = device.makeMTL4CommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create MTL4CommandQueue")
        }
        self.queue = queue

        guard let commandBuffer = device.makeCommandBuffer() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create MTL4CommandBuffer")
        }
        self.commandBuffer = commandBuffer

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
        guard let argumentTable = try? device.makeArgumentTable(descriptor: atDesc) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create MTL4ArgumentTable")
        }
        self.argumentTable = argumentTable
    }

    /// Submit a compute pass using Metal 4 APIs.
    ///
    /// The allocator is rotated each call. The caller must ensure the previous
    /// submission on the same allocator slot has completed before calling again.
    mutating func withCompute(
        waitUntilCompleted: Bool = true,
        _ encode: (MTL4ComputeCommandEncoder, MTL4ArgumentTable) throws -> Void
    ) throws {
        let allocator = allocators[frameIndex % Self.maxInFlight]
        frameIndex += 1

        allocator.reset()
        commandBuffer.beginCommandBuffer(allocator: allocator)

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
            let semaphore = DispatchSemaphore(value: 0)
            nonisolated(unsafe) var gpuError: NSError?

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
        _ encode: (MTL4ComputeCommandEncoder, MTL4ArgumentTable) throws -> Void
    ) throws -> (gpuStartTime: CFTimeInterval, gpuEndTime: CFTimeInterval) {
        let allocator = allocators[frameIndex % Self.maxInFlight]
        frameIndex += 1

        allocator.reset()
        commandBuffer.beginCommandBuffer(allocator: allocator)

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

        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var gpuError: NSError?
        nonisolated(unsafe) var gpuStartTime: CFTimeInterval = 0
        nonisolated(unsafe) var gpuEndTime: CFTimeInterval = 0

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
    mutating func fillBuffers(_ fills: [(buffer: MTLBuffer, value: UInt8)]) throws {
        try withCompute { encoder, _ in
            for fill in fills {
                encoder.fill(buffer: fill.buffer, range: 0..<fill.buffer.length, value: fill.value)
            }
        }
    }
}
