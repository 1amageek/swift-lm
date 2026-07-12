import Foundation
import SwiftLMFoundationModels
import Testing

@Suite("Core AI language-model bundle")
struct SwiftLMFoundationModelBundleTests {
    @Test("Loads an exported language bundle when a test asset is provided")
    func loadsExportedBundle() async throws {
        guard let path = ProcessInfo.processInfo.environment["SWIFTLM_COREAI_TEST_BUNDLE"] else {
            return
        }

        let bundle = try SwiftLMFoundationModelBundle(
            contentsOf: URL(fileURLWithPath: path, isDirectory: true)
        )
        #expect(bundle.vocabSize > 0)
        #expect(bundle.maxContextLength > 0)
        _ = try await bundle.makeLanguageModel()
    }
}
