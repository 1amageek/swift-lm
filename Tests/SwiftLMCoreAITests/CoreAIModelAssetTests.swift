import CoreAI
import Foundation
import Testing
@testable import SwiftLMCoreAI

@Suite("Core AI model asset")
struct CoreAIModelAssetTests {
    @Test("Runs a stateless Swift LMIR bundle when provided")
    func statelessExecution() async throws {
        guard let path = ProcessInfo.processInfo.environment[
            "SWIFTLM_COREAI_TEST_STATELESS_BUNDLE"
        ] else {
            return
        }

        let bundle = try CoreAIModelBundle(contentsOf: URL(fileURLWithPath: path))
        #expect(bundle.document.program.execution == .stateless)
        let session = try await bundle.makeStatelessSession()
        let outputs = try await session.run(
            inputs: [
                "input_ids": NDArray(scalars: [Int32(1), Int32(2)], shape: [1, 2]),
                "position_ids": NDArray(scalars: [Int32(0), Int32(1)], shape: [1, 2]),
            ],
            outputShapes: ["logits": [1, 2, bundle.document.metadata.vocabSize]]
        )

        let expectedShape = [1, 2, bundle.document.metadata.vocabSize]
        #expect(outputs["logits"]?.shape == expectedShape)
        if let expectedValue = ProcessInfo.processInfo.environment[
            "SWIFTLM_COREAI_TEST_STATELESS_EXPECTED_LAST_TOKEN"
        ] {
            guard let expectedToken = Int(expectedValue) else {
                Issue.record("Expected last-token ID must be an integer")
                return
            }
            let logits = try #require(outputs["logits"])
            #expect(lastTokenArgmax(logits, vocabSize: bundle.document.metadata.vocabSize) == expectedToken)
        }
    }

    @Test("Invalid asset paths fail with a typed error")
    func invalidAsset() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-lm-\(UUID().uuidString).aimodel")

        do {
            _ = try CoreAIModelAsset(contentsOf: url)
            Issue.record("Expected CoreAIModelAsset to reject a missing asset")
        } catch let error as CoreAIModelAssetError {
            #expect(error == .invalidAsset(url))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("Runs the Swift LMIR state contract when a test bundle is provided")
    func dynamicStateExecution() async throws {
        guard let path = ProcessInfo.processInfo.environment["SWIFTLM_COREAI_TEST_BUNDLE"] else {
            return
        }

        let bundle = try CoreAIModelBundle(contentsOf: URL(fileURLWithPath: path))
        #expect(bundle.document.program.execution == .stateful)
        #expect(!bundle.document.program.functions[0].states.isEmpty)
        let session = try await bundle.makeStateSession(
            maxContextLength: min(32, bundle.maxContextLength)
        )
        do {
            _ = try await session.run(
                inputs: [
                    "input_ids": NDArray(scalars: [Int32(1)], shape: [1, 1])
                ]
            )
            Issue.record("Expected the session to reject a missing position_ids input")
        } catch let error as CoreAIModelAssetError {
            #expect(error == .inputNotFound(function: "main", input: "position_ids"))
        }
        let first = try await session.run(
            inputs: [
                "input_ids": NDArray(scalars: [Int32(1)], shape: [1, 1]),
                "position_ids": NDArray(scalars: [Int32(0)], shape: [1, 1]),
            ]
        )
        let second = try await session.run(
            inputs: [
                "input_ids": NDArray(scalars: [Int32(2)], shape: [1, 1]),
                "position_ids": NDArray(scalars: [Int32(0), Int32(1)], shape: [1, 2]),
            ]
        )
        try await session.reset()
        let isolatedSecond = try await session.run(
            inputs: [
                "input_ids": NDArray(scalars: [Int32(2)], shape: [1, 1]),
                "position_ids": NDArray(scalars: [Int32(0), Int32(1)], shape: [1, 2]),
            ]
        )

        let expectedShape = [1, 1, bundle.document.metadata.vocabSize]
        #expect(first["logits"]?.shape == expectedShape)
        #expect(second["logits"]?.shape == expectedShape)
        let continuedOutput = try #require(second["logits"])
        if let statelessPath = ProcessInfo.processInfo.environment[
            "SWIFTLM_COREAI_TEST_STATELESS_BUNDLE"
        ] {
            let statelessBundle = try CoreAIModelBundle(
                contentsOf: URL(fileURLWithPath: statelessPath)
            )
            let statelessSession = try await statelessBundle.makeStatelessSession()
            let fullSequence = try await statelessSession.run(
                inputs: [
                    "input_ids": NDArray(scalars: [Int32(1), Int32(2)], shape: [1, 2]),
                    "position_ids": NDArray(scalars: [Int32(0), Int32(1)], shape: [1, 2]),
                ],
                outputShapes: [
                    "logits": [1, 2, statelessBundle.document.metadata.vocabSize]
                ]
            )
            let fullSequenceOutput = try #require(fullSequence["logits"])
            #expect(
                lastTokenArgmax(continuedOutput, vocabSize: bundle.document.metadata.vocabSize)
                    == lastTokenArgmax(
                        fullSequenceOutput,
                        vocabSize: statelessBundle.document.metadata.vocabSize
                    )
            )
        }
        let isolatedOutput = try #require(isolatedSecond["logits"])
        let elementCount = expectedShape.reduce(1, *)
        let differs = continuedOutput.view(as: Float16.self).withUnsafePointer { continued, _, _ in
            isolatedOutput.view(as: Float16.self).withUnsafePointer { isolated, _, _ in
                (0..<elementCount).contains { continued[$0] != isolated[$0] }
            }
        }
        #expect(differs)
    }
}

private func lastTokenArgmax(_ logits: NDArray, vocabSize: Int) -> Int {
    let tokenCount = logits.shape.dropFirst().dropLast().reduce(1, *)
    let offset = (tokenCount - 1) * vocabSize
    return logits.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
        var maximumIndex = 0
        var maximumValue = pointer[offset]
        for index in 1..<vocabSize where pointer[offset + index] > maximumValue {
            maximumIndex = index
            maximumValue = pointer[offset + index]
        }
        return maximumIndex
    }
}
