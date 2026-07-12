import CoreAI
import Foundation
import Testing
@testable import SwiftLMCoreAI

@Suite("Core AI model asset")
struct CoreAIModelAssetTests {
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

    @Test("Runs a dynamic-state asset when a test asset is provided")
    func dynamicStateExecution() async throws {
        guard let path = ProcessInfo.processInfo.environment["SWIFTLM_COREAI_TEST_ASSET"] else {
            return
        }

        let asset = try CoreAIModelAsset(contentsOf: URL(fileURLWithPath: path))
        let model = try await asset.specialize()
        let session = try CoreAIStateSession(
            model: model,
            functionName: "main",
            stateShapes: [
                "keyCache": [2, 1, 1, 40960, 32],
                "valueCache": [2, 1, 1, 40960, 32],
            ]
        )
        let outputs = try await session.run(
            inputs: [
                "input_ids": NDArray(scalars: [Int32(151644)], shape: [1, 1]),
                "position_ids": NDArray(scalars: [Int32(0)], shape: [1, 1]),
            ],
            outputShapes: ["logits": [1, 1, 151936]]
        )

        #expect(outputs["logits"]?.shape == [1, 1, 151936])
    }
}
