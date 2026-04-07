import Metal

/// Identifies a specific region within an MTLBuffer for barrier optimization.
///
/// Scratch buffer holds multiple independent slots at different offsets.
/// Using (buffer identity, offset) prevents false barrier dependencies
/// between independent scratch slots.
public struct BufferRegion: Hashable, @unchecked Sendable {
    public let buffer: ObjectIdentifier
    public let offset: Int
    /// Raw MTLBuffer reference for `memoryBarrier(resources:)`.
    public let rawBuffer: MTLBuffer

    public init(buffer: MTLBuffer, offset: Int) {
        self.buffer = ObjectIdentifier(buffer)
        self.offset = offset
        self.rawBuffer = buffer
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(buffer)
        hasher.combine(offset)
    }

    public static func == (lhs: BufferRegion, rhs: BufferRegion) -> Bool {
        lhs.buffer == rhs.buffer && lhs.offset == rhs.offset
    }
}

public struct MetalBufferAccesses: @unchecked Sendable {
    public let reads: Set<BufferRegion>
    public let writes: Set<BufferRegion>

    public init(reads: Set<BufferRegion>, writes: Set<BufferRegion>) {
        self.reads = reads
        self.writes = writes
    }

    public init(readBuffers: [(buffer: MTLBuffer, offset: Int)],
                writeBuffers: [(buffer: MTLBuffer, offset: Int)]) {
        self.reads = Set(readBuffers.map { BufferRegion(buffer: $0.buffer, offset: $0.offset) })
        self.writes = Set(writeBuffers.map { BufferRegion(buffer: $0.buffer, offset: $0.offset) })
    }

    public static func conservative(_ bindings: [MetalBufferBinding]) -> Self {
        let regions = Set(bindings.map { BufferRegion(buffer: $0.buffer, offset: $0.offset) })
        return Self(reads: regions, writes: regions)
    }

    public func requiresBarrier(after pendingWrites: Set<BufferRegion>) -> Bool {
        !pendingWrites.isDisjoint(with: reads.union(writes))
    }

    /// Returns the unique MTLBuffer objects that conflict with pending writes.
    /// Used to generate `memoryBarrier(resources:)` instead of `memoryBarrier(scope: .buffers)`.
    public func conflictingResources(from pendingWrites: Set<BufferRegion>) -> [MTLResource] {
        let conflicting = pendingWrites.intersection(reads.union(writes))
        var seen = Set<ObjectIdentifier>()
        var resources: [MTLResource] = []
        for region in conflicting {
            if seen.insert(region.buffer).inserted {
                resources.append(region.rawBuffer)
            }
        }
        return resources
    }
}
