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
}
