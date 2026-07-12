import CoreAI
import Foundation

/// Serial stateful execution for Core AI tensor functions.
///
/// The session owns one mutable value for every tensor state declared by the
/// function. It is intentionally an actor because state updates and GPU
/// submission order are part of the execution contract. Dynamic state shapes
/// must be supplied at initialization, and dynamic output shapes must be
/// supplied for each run.
@available(macOS 27.0, iOS 27.0, *)
public actor CoreAIStateSession {
    private final class StateValue {
        let name: String
        var value: NDArray

        init(name: String, value: NDArray) {
            self.name = name
            self.value = value
        }
    }

    private let function: InferenceFunction
    private var states: [StateValue]

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
        self.stateShapes = stateShapes
        self.states = try Self.makeStates(for: function, stateShapes: stateShapes)
    }

    public var functionDescriptor: InferenceFunctionDescriptor {
        function.descriptor
    }

    public func reset() throws {
        states = try Self.makeStates(for: function, stateShapes: stateShapes)
    }

    public func run(
        inputs: [String: NDArray],
        outputShapes: [String: [Int]] = [:]
    ) async throws -> [String: NDArray] {
        guard function.descriptor.outputNames.count <= 1 else {
            throw CoreAIModelAssetError.unsupportedOutputCount(function.descriptor.outputNames.count)
        }
        let outputNames = Set(function.descriptor.outputNames)
        for name in outputShapes.keys where !outputNames.contains(name) {
            throw CoreAIModelAssetError.outputUnavailable(name)
        }

        guard let outputName = function.descriptor.outputNames.first else {
            return try await runWithoutOutput(inputs: inputs)
        }

        switch states.count {
        case 0:
            var output = try makeOutput(named: outputName, shape: outputShapes[outputName])
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(&output, for: outputName)
            _ = try await function.run(inputs: inputs, outputViews: outputViews)
            return [outputName: output]
        case 1:
            var state0 = states[0].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            var output = try makeOutput(named: outputName, shape: outputShapes[outputName])
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(&output, for: outputName)
            _ = try await function.run(
                inputs: inputs,
                states: views,
                outputViews: outputViews
            )
            states[0].value = state0
            return [outputName: output]
        case 2:
            var state0 = states[0].value
            var state1 = states[1].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            views.insert(&state1, for: states[1].name)
            var output = try makeOutput(named: outputName, shape: outputShapes[outputName])
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(&output, for: outputName)
            _ = try await function.run(
                inputs: inputs,
                states: views,
                outputViews: outputViews
            )
            states[0].value = state0
            states[1].value = state1
            return [outputName: output]
        case 3:
            var state0 = states[0].value
            var state1 = states[1].value
            var state2 = states[2].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            views.insert(&state1, for: states[1].name)
            views.insert(&state2, for: states[2].name)
            var output = try makeOutput(named: outputName, shape: outputShapes[outputName])
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(&output, for: outputName)
            _ = try await function.run(
                inputs: inputs,
                states: views,
                outputViews: outputViews
            )
            states[0].value = state0
            states[1].value = state1
            states[2].value = state2
            return [outputName: output]
        case 4:
            var state0 = states[0].value
            var state1 = states[1].value
            var state2 = states[2].value
            var state3 = states[3].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            views.insert(&state1, for: states[1].name)
            views.insert(&state2, for: states[2].name)
            views.insert(&state3, for: states[3].name)
            var output = try makeOutput(named: outputName, shape: outputShapes[outputName])
            var outputViews = InferenceFunction.MutableViews()
            outputViews.insert(&output, for: outputName)
            _ = try await function.run(
                inputs: inputs,
                states: views,
                outputViews: outputViews
            )
            states[0].value = state0
            states[1].value = state1
            states[2].value = state2
            states[3].value = state3
            return [outputName: output]
        default:
            throw CoreAIModelAssetError.unsupportedStateCount(states.count)
        }
    }

    private func runWithoutOutput(inputs: [String: NDArray]) async throws -> [String: NDArray] {
        switch states.count {
        case 0:
            _ = try await function.run(inputs: inputs)
            return [:]
        case 1:
            var state0 = states[0].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            _ = try await function.run(inputs: inputs, states: views)
            states[0].value = state0
            return [:]
        case 2:
            var state0 = states[0].value
            var state1 = states[1].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            views.insert(&state1, for: states[1].name)
            _ = try await function.run(inputs: inputs, states: views)
            states[0].value = state0
            states[1].value = state1
            return [:]
        case 3:
            var state0 = states[0].value
            var state1 = states[1].value
            var state2 = states[2].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            views.insert(&state1, for: states[1].name)
            views.insert(&state2, for: states[2].name)
            _ = try await function.run(inputs: inputs, states: views)
            states[0].value = state0
            states[1].value = state1
            states[2].value = state2
            return [:]
        case 4:
            var state0 = states[0].value
            var state1 = states[1].value
            var state2 = states[2].value
            var state3 = states[3].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            views.insert(&state1, for: states[1].name)
            views.insert(&state2, for: states[2].name)
            views.insert(&state3, for: states[3].name)
            _ = try await function.run(inputs: inputs, states: views)
            states[0].value = state0
            states[1].value = state1
            states[2].value = state2
            states[3].value = state3
            return [:]
        default:
            throw CoreAIModelAssetError.unsupportedStateCount(states.count)
        }
    }

    private func makeOutput(named name: String, shape: [Int]?) throws -> NDArray {
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
        return NDArray(descriptor: resolvedDescriptor)
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
            return StateValue(name: name, value: NDArray(descriptor: resolvedDescriptor))
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
