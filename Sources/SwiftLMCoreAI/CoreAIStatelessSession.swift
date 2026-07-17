import CoreAI

/// Serial execution of a stateless Core AI tensor function.
@available(macOS 27.0, iOS 27.0, *)
public struct CoreAIStatelessSession: CoreAIExecutableSession {
    private let executor: CoreAIStateSession

    public init(model: AIModel, functionName: String) throws {
        guard let function = try model.loadFunction(named: functionName) else {
            throw CoreAIModelAssetError.functionNotFound(functionName)
        }
        guard function.descriptor.stateNames.isEmpty else {
            throw CoreAIModelAssetError.statefulFunctionRequiresStateSession(functionName)
        }
        executor = try CoreAIStateSession(
            model: model,
            functionName: functionName
        )
    }

    public var functionDescriptor: InferenceFunctionDescriptor {
        get async {
            await executor.functionDescriptor
        }
    }

    public func run(
        inputs: [String: NDArray],
        outputShapes: [String: [Int]] = [:]
    ) async throws -> [String: NDArray] {
        try await executor.run(inputs: inputs, outputShapes: outputShapes)
    }
}
