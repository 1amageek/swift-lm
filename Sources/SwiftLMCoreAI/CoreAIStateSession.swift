import CoreAI
import Foundation

/// Serial stateful execution for Core AI tensor functions.
///
/// The session owns one mutable value for every tensor state declared by the
/// function. It is intentionally an actor because state updates and GPU
/// submission order are part of the execution contract.
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

    public init(model: AIModel, functionName: String) throws {
        guard let function = try model.loadFunction(named: functionName) else {
            throw CoreAIModelAssetError.functionNotFound(functionName)
        }
        self.function = function
        self.states = try Self.makeStates(for: function)
    }

    public var functionDescriptor: InferenceFunctionDescriptor {
        function.descriptor
    }

    public func reset() throws {
        states = try Self.makeStates(for: function)
    }

    public func run(inputs: [String: NDArray]) async throws -> [String: NDArray] {
        switch states.count {
        case 0:
            var outputs = try await function.run(inputs: inputs)
            return try collectOutputs(&outputs)
        case 1:
            var state0 = states[0].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            var outputs = try await function.run(inputs: inputs, states: views)
            states[0].value = state0
            return try collectOutputs(&outputs)
        case 2:
            var state0 = states[0].value
            var state1 = states[1].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            views.insert(&state1, for: states[1].name)
            var outputs = try await function.run(inputs: inputs, states: views)
            states[0].value = state0
            states[1].value = state1
            return try collectOutputs(&outputs)
        case 3:
            var state0 = states[0].value
            var state1 = states[1].value
            var state2 = states[2].value
            var views = InferenceFunction.MutableViews()
            views.insert(&state0, for: states[0].name)
            views.insert(&state1, for: states[1].name)
            views.insert(&state2, for: states[2].name)
            var outputs = try await function.run(inputs: inputs, states: views)
            states[0].value = state0
            states[1].value = state1
            states[2].value = state2
            return try collectOutputs(&outputs)
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
            var outputs = try await function.run(inputs: inputs, states: views)
            states[0].value = state0
            states[1].value = state1
            states[2].value = state2
            states[3].value = state3
            return try collectOutputs(&outputs)
        default:
            throw CoreAIModelAssetError.unsupportedStateCount(states.count)
        }
    }

    private func collectOutputs(
        _ outputs: inout InferenceFunction.Outputs
    ) throws -> [String: NDArray] {
        var result: [String: NDArray] = [:]
        for name in outputs.names {
            guard let value = outputs.remove(name), let array = value.ndArray else {
                throw CoreAIModelAssetError.outputUnavailable(name)
            }
            result[name] = array
        }
        return result
    }

    private static func makeStates(for function: InferenceFunction) throws -> [StateValue] {
        try function.descriptor.stateNames.map { name in
            guard let descriptor = function.descriptor.stateDescriptor(of: name) else {
                throw CoreAIModelAssetError.stateNotFound(function: function.descriptor.name, state: name)
            }
            guard case .ndArray(let arrayDescriptor) = descriptor else {
                throw CoreAIModelAssetError.unsupportedSpecializationOption(
                    "image state '\(name)' is not supported by CoreAIStateSession"
                )
            }
            return StateValue(name: name, value: NDArray(descriptor: arrayDescriptor))
        }
    }
}
