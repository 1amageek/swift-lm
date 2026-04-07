import Metal
import LMIR

public enum MetalArgumentBindingPolicy: Sendable, Equatable {
    case inlineBindings
    case argumentTable
}

public enum MetalConstantBindingPolicy: Sendable, Equatable {
    case inlineBytes
    case residentConstantBuffer
}

public enum MetalBarrierPolicy: @unchecked Sendable, Equatable {
    case none
    case bufferBarrier
    case resourceBarrier(resources: [MTLResource])

    public init(_ synchronizationKind: SynchronizationKind) {
        switch synchronizationKind {
        case .none:
            self = .none
        case .bufferBarrier:
            self = .bufferBarrier
        }
    }

    public var synchronizationKind: SynchronizationKind {
        switch self {
        case .none:
            return .none
        case .bufferBarrier, .resourceBarrier:
            return .bufferBarrier
        }
    }

    public var isBarrier: Bool {
        switch self {
        case .none: return false
        case .bufferBarrier, .resourceBarrier: return true
        }
    }

    public static func == (lhs: MetalBarrierPolicy, rhs: MetalBarrierPolicy) -> Bool {
        switch (lhs, rhs) {
        case (.none, .none): return true
        case (.bufferBarrier, .bufferBarrier): return true
        case (.resourceBarrier(let a), .resourceBarrier(let b)):
            return a.count == b.count
        default: return false
        }
    }
}

public struct MetalBufferBinding: @unchecked Sendable {
    public let index: Int
    public let buffer: MTLBuffer
    public let offset: Int
}

public enum MetalArgumentTableEncodingState: @unchecked Sendable {
    case planned
    case prepared(buffer: MTLBuffer, index: Int, offset: Int)
    case encoded(buffer: MTLBuffer, index: Int, offset: Int)

    public var isEncoded: Bool {
        switch self {
        case .planned:
            return false
        case .prepared:
            return false
        case .encoded:
            return true
        }
    }
}

public struct MetalArgumentTableBindings: @unchecked Sendable {
    public let layout: MetalArgumentTableLayout
    public let bindings: [MetalBufferBinding]
    public let encodingState: MetalArgumentTableEncodingState

    public init(
        layout: MetalArgumentTableLayout,
        bindings: [MetalBufferBinding],
        encodingState: MetalArgumentTableEncodingState = .planned
    ) {
        self.layout = layout
        self.bindings = bindings
        self.encodingState = encodingState
    }

    public var hasEncodedArgumentBuffer: Bool {
        encodingState.isEncoded
    }
}

public enum MetalBufferBindingSet: @unchecked Sendable {
    case inline([MetalBufferBinding])
    case argumentTable(MetalArgumentTableBindings)

    public var policy: MetalArgumentBindingPolicy {
        switch self {
        case .inline:
            return .inlineBindings
        case .argumentTable:
            return .argumentTable
        }
    }

    public var bindings: [MetalBufferBinding] {
        switch self {
        case .inline(let bindings):
            return bindings
        case .argumentTable(let table):
            return table.bindings
        }
    }
}

public struct MetalBytesBinding: Sendable {
    public let index: Int
    public let value: [UInt8]
}

public struct MetalConstantBufferBinding: @unchecked Sendable {
    public let index: Int
    public let buffer: MTLBuffer
    public let offset: Int
    public let length: Int
}

public struct MetalResidentConstantBindings: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let bindings: [MetalConstantBufferBinding]
}

public enum MetalConstantBinding: @unchecked Sendable {
    case inline(MetalBytesBinding)
    case buffer(MetalConstantBufferBinding)

    public var index: Int {
        switch self {
        case .inline(let binding):
            return binding.index
        case .buffer(let binding):
            return binding.index
        }
    }
}

public enum MetalConstantBindingSet: @unchecked Sendable {
    case inline([MetalBytesBinding])
    case resident(MetalResidentConstantBindings)
    case mixed([MetalConstantBinding])

    public var policy: MetalConstantBindingPolicy {
        switch self {
        case .inline:
            return .inlineBytes
        case .resident:
            return .residentConstantBuffer
        case .mixed(let bindings):
            if bindings.allSatisfy({
                if case .buffer = $0 { return true }
                return false
            }) {
                return .residentConstantBuffer
            }
            return .inlineBytes
        }
    }

    public var bindings: [MetalConstantBinding] {
        switch self {
        case .inline(let bindings):
            return bindings.map(MetalConstantBinding.inline)
        case .resident(let resident):
            return resident.bindings.map(MetalConstantBinding.buffer)
        case .mixed(let bindings):
            return bindings
        }
    }

    public var inlineBindings: [MetalBytesBinding] {
        bindings.compactMap { binding in
            guard case .inline(let bytes) = binding else { return nil }
            return bytes
        }
    }
}

public struct MetalBindingTable: @unchecked Sendable {
    private static let bufferEncoder = MetalBufferBindingEncoder()
    private static let constantEncoder = MetalConstantBindingEncoder()

    public let bufferBindings: MetalBufferBindingSet
    public let constantBindings: MetalConstantBindingSet
    public var argumentPolicy: MetalArgumentBindingPolicy {
        bufferBindings.policy
    }
    public var buffers: [MetalBufferBinding] {
        bufferBindings.bindings
    }
    public var constantPolicy: MetalConstantBindingPolicy {
        constantBindings.policy
    }
    public var constants: [MetalConstantBinding] {
        constantBindings.bindings
    }

    public init(
        buffers: [MetalBufferBinding] = [],
        constants: [MetalConstantBinding] = [],
        argumentPolicy: MetalArgumentBindingPolicy = .inlineBindings,
        constantPolicy: MetalConstantBindingPolicy = .inlineBytes
    ) {
        switch argumentPolicy {
        case .inlineBindings:
            self.bufferBindings = .inline(buffers)
        case .argumentTable:
            self.bufferBindings = .argumentTable(MetalArgumentTableBindings(
                layout: MetalArgumentTableLayout(id: 0, indices: buffers.map(\.index)),
                bindings: buffers))
        }
        switch constantPolicy {
        case .inlineBytes:
            self.constantBindings = .mixed(constants)
        case .residentConstantBuffer:
            self.constantBindings = .mixed(constants)
        }
    }

    public init(
        buffers: [MetalBufferBinding] = [],
        bufferBindings: MetalBufferBindingSet,
        constantBindings: MetalConstantBindingSet,
        argumentPolicy: MetalArgumentBindingPolicy = .inlineBindings,
        constantPolicy: MetalConstantBindingPolicy? = nil
    ) {
        _ = buffers
        self.bufferBindings = bufferBindings
        self.constantBindings = constantBindings
        _ = argumentPolicy
        _ = constantPolicy
    }

    public init(
        bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)],
        bytesBindings: [(index: Int, value: [UInt8])],
        argumentPolicy: MetalArgumentBindingPolicy = .inlineBindings,
        constantPolicy: MetalConstantBindingPolicy = .inlineBytes
    ) {
        let mappedBuffers = bufferBindings.map { MetalBufferBinding(index: $0.index, buffer: $0.buffer, offset: $0.offset) }
        switch argumentPolicy {
        case .inlineBindings:
            self.bufferBindings = .inline(mappedBuffers)
        case .argumentTable:
            self.bufferBindings = .argumentTable(MetalArgumentTableBindings(
                layout: MetalArgumentTableLayout(id: 0, indices: mappedBuffers.map(\.index)),
                bindings: mappedBuffers))
        }
        let inlineBindings = bytesBindings.map { MetalBytesBinding(index: $0.index, value: $0.value) }
        switch constantPolicy {
        case .inlineBytes:
            self.constantBindings = .inline(inlineBindings)
        case .residentConstantBuffer:
            self.constantBindings = .mixed(inlineBindings.map(MetalConstantBinding.inline))
        }
    }

    public func bind(to encoder: MTLComputeCommandEncoder) {
        bind(to: encoder, adjustedBufferOffsets: [:])
    }

    public func bind(
        to encoder: MTLComputeCommandEncoder,
        adjustedBufferOffsets: [Int: Int]
    ) {
        Self.bufferEncoder.bind(
            bufferBindings,
            to: encoder,
            adjustedBufferOffsets: adjustedBufferOffsets)
        Self.constantEncoder.bind(
            constantBindings,
            to: encoder)
    }
}

public struct MetalDispatchDescriptor: @unchecked Sendable {
    public let pipeline: MTLComputePipelineState
    public let gridSize: MTLSize
    public let threadgroupSize: MTLSize
    public let threadgroupMemoryLength: Int
    public let barrierPolicy: MetalBarrierPolicy

    public var sync: SynchronizationKind {
        barrierPolicy.synchronizationKind
    }

    public func encode(on encoder: MTLComputeCommandEncoder, gridSize overrideGridSize: MTLSize? = nil) {
        switch barrierPolicy {
        case .none:
            break
        case .bufferBarrier:
            encoder.memoryBarrier(scope: .buffers)
        case .resourceBarrier(let resources):
            encoder.memoryBarrier(resources: resources)
        }
        encoder.setComputePipelineState(pipeline)
        if threadgroupMemoryLength > 0 {
            encoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)
        }
        encoder.dispatchThreadgroups(overrideGridSize ?? gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}
