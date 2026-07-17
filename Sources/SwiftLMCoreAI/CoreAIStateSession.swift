import CoreAI
import Foundation
import Metal

/// Serial stateful execution for Core AI tensor functions.
///
/// The session owns one mutable value for every tensor state declared by the
/// function. It is intentionally an actor because state updates and GPU
/// submission order are part of the execution contract. Dynamic state shapes
/// must be supplied at initialization, and dynamic output shapes must be
/// supplied for each run.
@available(macOS 27.0, iOS 27.0, *)
public actor CoreAIStateSession: CoreAIExecutableSession {
    private final class StateValue {
        let name: String
        let buffer: any MTLBuffer
        let descriptor: NDArrayDescriptor

        init(name: String, buffer: any MTLBuffer, descriptor: NDArrayDescriptor) {
            self.name = name
            self.buffer = buffer
            self.descriptor = descriptor
        }
    }

    private let function: InferenceFunction
    private let stream: ComputeStream
    private var states: [StateValue]
    private var isExecuting = false
    private var executionWaiters: [CheckedContinuation<Void, Never>] = []

    public init(
        model: AIModel,
        functionName: String,
        stateShapes: [String: [Int]] = [:]
    ) throws {
        guard let function = try model.loadFunction(named: functionName) else {
            throw CoreAIModelAssetError.functionNotFound(functionName)
        }
        let stateNames = Set(function.descriptor.stateNames)
        for name in stateShapes.keys where !stateNames.contains(name) {
            throw CoreAIModelAssetError.stateNotFound(function: functionName, state: name)
        }
        self.function = function
        self.stream = ComputeStream()
        self.stateShapes = stateShapes
        self.states = try Self.makeStates(for: function, stateShapes: stateShapes)
    }

    public var functionDescriptor: InferenceFunctionDescriptor {
        function.descriptor
    }

    public func reset() async throws {
        await acquireExecution()
        do {
            states = try Self.makeStates(for: function, stateShapes: stateShapes)
            releaseExecution()
        } catch {
            releaseExecution()
            throw error
        }
    }

    public func run(
        inputs: [String: NDArray],
        outputShapes: [String: [Int]] = [:]
    ) async throws -> [String: NDArray] {
        await acquireExecution()
        do {
            let output = try await runExclusively(inputs: inputs, outputShapes: outputShapes)
            releaseExecution()
            return output
        } catch {
            releaseExecution()
            throw error
        }
    }

    private func runExclusively(
        inputs: [String: NDArray],
        outputShapes: [String: [Int]]
    ) async throws -> [String: NDArray] {
        try validate(inputs: inputs)
        guard function.descriptor.outputNames.count == 1 else {
            throw CoreAIModelAssetError.unsupportedOutputCount(function.descriptor.outputNames.count)
        }
        let outputNames = Set(function.descriptor.outputNames)
        for name in outputShapes.keys where !outputNames.contains(name) {
            throw CoreAIModelAssetError.outputUnavailable(name)
        }

        let outputName = function.descriptor.outputNames[0]
        let expectedDescriptor = try outputDescriptor(
            named: outputName,
            shape: outputShapes[outputName]
        )
        let asyncInputs = Dictionary(uniqueKeysWithValues: inputs.map { name, value in
            (name, InferenceFunction.AsyncValue(value))
        })
        let stateViews = InferenceFunction.AsyncMutableViews()
        let encodedOutputs = try encode(
            stateIndex: 0,
            inputs: asyncInputs,
            stateViews: consume stateViews,
            stream: stream
        )
        guard let asyncOutput = encodedOutputs[outputName],
              let resolvedOutput = try await asyncOutput.ndArray else {
            throw CoreAIModelAssetError.outputUnavailable(outputName)
        }
        guard resolvedOutput.shape == expectedDescriptor.shape else {
            throw CoreAIModelAssetError.invalidOutputShape(
                function: function.descriptor.name,
                output: outputName,
                expected: expectedDescriptor.shape,
                provided: resolvedOutput.shape
            )
        }
        return [outputName: resolvedOutput]
    }

    private func validate(inputs: [String: NDArray]) throws {
        let functionName = function.descriptor.name
        let expectedNames = Set(function.descriptor.inputNames)
        for name in inputs.keys where !expectedNames.contains(name) {
            throw CoreAIModelAssetError.unexpectedInput(function: functionName, input: name)
        }
        for name in function.descriptor.inputNames {
            guard let input = inputs[name] else {
                throw CoreAIModelAssetError.inputNotFound(function: functionName, input: name)
            }
            guard let descriptor = function.descriptor.inputDescriptor(of: name),
                  case .ndArray(let arrayDescriptor) = descriptor else {
                throw CoreAIModelAssetError.unsupportedInput(function: functionName, input: name)
            }
            guard input.scalarType == arrayDescriptor.scalarType else {
                throw CoreAIModelAssetError.invalidInputDataType(
                    function: functionName,
                    input: name,
                    expected: String(describing: arrayDescriptor.scalarType),
                    provided: String(describing: input.scalarType)
                )
            }
            guard input.shape.count == arrayDescriptor.rank,
                  zip(arrayDescriptor.shape, input.shape).allSatisfy({ expected, actual in
                      expected == -1 || expected == actual
                  }) else {
                throw CoreAIModelAssetError.invalidInputShape(
                    function: functionName,
                    input: name,
                    expected: arrayDescriptor.shape,
                    provided: input.shape
                )
            }
        }
    }

    private func encode(
        stateIndex: Int,
        inputs: [String: InferenceFunction.AsyncValue],
        stateViews: consuming InferenceFunction.AsyncMutableViews,
        stream: borrowing ComputeStream
    ) throws -> [String: InferenceFunction.AsyncValue] {
        guard stateIndex < states.count else {
            let outputViews = InferenceFunction.AsyncMutableViews()
            return try function.encode(
                inputs: inputs,
                states: stateViews,
                outputViews: consume outputViews,
                to: stream
            )
        }

        let state = states[stateIndex]
        var stateValue = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: state.buffer,
            scalarType: state.descriptor.scalarType,
            shape: state.descriptor.shape,
            strides: state.descriptor.preferredStrides,
            interleaveLayout: state.descriptor.interleaveLayout
        )
        var nextViews = stateViews
        nextViews.insert(&stateValue, for: state.name)
        return try encode(
            stateIndex: stateIndex + 1,
            inputs: inputs,
            stateViews: consume nextViews,
            stream: stream
        )
    }

    private func outputDescriptor(named name: String, shape: [Int]?) throws -> NDArrayDescriptor {
        guard let descriptor = function.descriptor.outputDescriptor(of: name) else {
            throw CoreAIModelAssetError.outputUnavailable(name)
        }
        guard case .ndArray(let arrayDescriptor) = descriptor else {
            throw CoreAIModelAssetError.unsupportedSpecializationOption(
                "non-NDArray output '\(name)' is not supported by CoreAIStateSession"
            )
        }
        let resolvedDescriptor = try Self.resolveOutput(
            arrayDescriptor,
            function: function.descriptor.name,
            output: name,
            requestedShape: shape
        )
        return resolvedDescriptor
    }

    private func acquireExecution() async {
        guard isExecuting else {
            isExecuting = true
            return
        }
        await withCheckedContinuation { continuation in
            executionWaiters.append(continuation)
        }
    }

    private func releaseExecution() {
        guard !executionWaiters.isEmpty else {
            isExecuting = false
            return
        }
        executionWaiters.removeFirst().resume()
    }

    private let stateShapes: [String: [Int]]

    private static func makeStates(
        for function: InferenceFunction,
        stateShapes: [String: [Int]]
    ) throws -> [StateValue] {
        try function.descriptor.stateNames.map { name in
            guard let descriptor = function.descriptor.stateDescriptor(of: name) else {
                throw CoreAIModelAssetError.stateNotFound(function: function.descriptor.name, state: name)
            }
            guard case .ndArray(let arrayDescriptor) = descriptor else {
                throw CoreAIModelAssetError.unsupportedSpecializationOption(
                    "image state '\(name)' is not supported by CoreAIStateSession"
                )
            }
            let resolvedDescriptor = try Self.resolve(
                arrayDescriptor,
                function: function.descriptor.name,
                state: name,
                requestedShape: stateShapes[name]
            )
            guard let device = MTLCreateSystemDefaultDevice(),
                  let buffer = device.makeBuffer(
                    length: resolvedDescriptor.minimumByteCount,
                    options: .storageModeShared
                  ) else {
                throw CoreAIModelAssetError.stateAllocationFailed(name)
            }
            memset(buffer.contents(), 0, buffer.length)
            return StateValue(
                name: name,
                buffer: buffer,
                descriptor: resolvedDescriptor
            )
        }
    }

    private static func resolve(
        _ descriptor: NDArrayDescriptor,
        function: String,
        state: String,
        requestedShape: [Int]?
    ) throws -> NDArrayDescriptor {
        guard descriptor.hasDynamicShape else {
            if let requestedShape, requestedShape != descriptor.shape {
                throw CoreAIModelAssetError.invalidStateShape(
                    function: function,
                    state: state,
                    expected: descriptor.shape,
                    provided: requestedShape
                )
            }
            return descriptor
        }

        guard let requestedShape else {
            throw CoreAIModelAssetError.missingDynamicStateShape(state)
        }
        guard requestedShape.count == descriptor.rank,
              requestedShape.allSatisfy({ $0 > 0 }),
              zip(descriptor.shape, requestedShape).allSatisfy({ expected, actual in
                  expected == -1 || expected == actual
              }) else {
            throw CoreAIModelAssetError.invalidStateShape(
                function: function,
                state: state,
                expected: descriptor.shape,
                provided: requestedShape
            )
        }
        return descriptor.resolvingDynamicDimensions(requestedShape)
    }

    private static func resolveOutput(
        _ descriptor: NDArrayDescriptor,
        function: String,
        output: String,
        requestedShape: [Int]?
    ) throws -> NDArrayDescriptor {
        guard descriptor.hasDynamicShape else {
            if let requestedShape, requestedShape != descriptor.shape {
                throw CoreAIModelAssetError.invalidOutputShape(
                    function: function,
                    output: output,
                    expected: descriptor.shape,
                    provided: requestedShape
                )
            }
            return descriptor
        }

        guard let requestedShape else {
            throw CoreAIModelAssetError.missingDynamicOutputShape(output)
        }
        guard requestedShape.count == descriptor.rank,
              requestedShape.allSatisfy({ $0 > 0 }),
              zip(descriptor.shape, requestedShape).allSatisfy({ expected, actual in
                  expected == -1 || expected == actual
              }) else {
            throw CoreAIModelAssetError.invalidOutputShape(
                function: function,
                output: output,
                expected: descriptor.shape,
                provided: requestedShape
            )
        }
        return descriptor.resolvingDynamicDimensions(requestedShape)
    }
}
