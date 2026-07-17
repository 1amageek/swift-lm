import CoreAI

/// A specialized Core AI function that accepts named tensors and returns named tensors.
@available(macOS 27.0, iOS 27.0, *)
public protocol CoreAIExecutableSession: Sendable {
    func run(
        inputs: [String: NDArray],
        outputShapes: [String: [Int]]
    ) async throws -> [String: NDArray]
}
