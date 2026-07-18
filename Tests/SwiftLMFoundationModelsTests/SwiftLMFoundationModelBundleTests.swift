import Foundation
import Testing

@testable import SwiftLMFoundationModels

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

    @Test("Runs an official Core AI VLM bundle when test assets are provided")
    func runsVisionLanguageBundle() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard
            let bundlePath = environment["SWIFTLM_COREAI_TEST_VLM_BUNDLE"],
            let imagePath = environment["SWIFTLM_COREAI_TEST_VLM_IMAGE"]
        else {
            return
        }

        let bundle = try SwiftLMFoundationModelBundle(
            contentsOf: URL(fileURLWithPath: bundlePath, isDirectory: true)
        )
        let model = try await bundle.makeVisionLanguageModel()
        let prompt: SwiftLMVisionLanguagePrompt
        if let rawTokenIDs = environment["SWIFTLM_COREAI_TEST_VLM_TOKEN_IDS"] {
            let tokenIDs = try rawTokenIDs.split(separator: ",").map { value in
                try #require(Int32(value))
            }
            prompt = .tokens(tokenIDs)
        } else {
            prompt = .text("Describe the image.")
        }
        let output = try await model.generate(
            from: SwiftLMVisionLanguageInput(
                imageURL: URL(fileURLWithPath: imagePath),
                prompt: prompt
            ),
            options: SwiftLMVisionLanguageGenerationOptions(maxTokens: 1)
        )

        #expect(output.tokenIDs.count == 1)
    }

    @Test("Reads the official multi-asset VLM contract")
    func readsVisionLanguageBundleMetadata() throws {
        let directory = try makeBundleDirectory(metadata: visionLanguageMetadata)
        let bundle = try SwiftLMFoundationModelBundle(contentsOf: directory)

        #expect(bundle.name == "test-vlm")
        #expect(bundle.isVisionLanguageModel)
        #expect(bundle.visionConfiguration?.imageSize == 448)
        #expect(bundle.visionConfiguration?.patchSize == 16)
        #expect(bundle.visionConfiguration?.imageTokenCount == 196)
        #expect(bundle.visionConfiguration?.imageTokenID == 151655)

        try FileManager.default.removeItem(at: directory)
    }

    @Test("Rejects loading a VLM through the text-only API")
    func rejectsVisionBundleFromLanguageOnlyAPI() async throws {
        let directory = try makeBundleDirectory(metadata: visionLanguageMetadata)
        let bundle = try SwiftLMFoundationModelBundle(contentsOf: directory)

        await #expect(throws: SwiftLMVisionLanguageModelError.visionLanguageModelRequiresVisionAPI)
        {
            _ = try await bundle.makeLanguageModel()
        }

        try FileManager.default.removeItem(at: directory)
    }

    @Test("Rejects malformed VLM normalization metadata before loading assets")
    func rejectsMalformedVisionConfiguration() async throws {
        let metadata = visionLanguageMetadata.replacingOccurrences(
            of: "\"image_std\": [0.26862954, 0.26130258, 0.27577711]",
            with: "\"image_std\": [1.0]"
        )
        let directory = try makeBundleDirectory(metadata: metadata)
        let bundle = try SwiftLMFoundationModelBundle(contentsOf: directory)

        await #expect(
            throws: SwiftLMVisionLanguageModelError.invalidVisionConfiguration(
                field: "image_std",
                reason: "must contain three RGB values"
            )
        ) {
            _ = try await bundle.makeVisionLanguageModel()
        }

        try FileManager.default.removeItem(at: directory)
    }

    @Test("Expands one template image token to the declared visual token count")
    func expandsTemplateImageToken() throws {
        let expander = SwiftLMVisionPromptTokenExpander(
            imageTokenID: 99,
            imageTokenCount: 3
        )

        let tokenIDs = try expander.expandTemplatedTokenIDs([1, 99, 2])

        #expect(tokenIDs == [1, 99, 99, 99, 2])
    }

    @Test("Requires exactly one image token before template expansion")
    func rejectsAmbiguousTemplateImageTokens() throws {
        let expander = SwiftLMVisionPromptTokenExpander(
            imageTokenID: 99,
            imageTokenCount: 3
        )

        #expect(
            throws: SwiftLMVisionLanguageModelError.invalidImagePlaceholderCount(
                expected: 1,
                actual: 2
            )
        ) {
            _ = try expander.expandTemplatedTokenIDs([99, 1, 99])
        }
    }

    @Test("Validates pre-tokenized image placeholder count")
    func validatesPretokenizedImageTokens() throws {
        let expander = SwiftLMVisionPromptTokenExpander(
            imageTokenID: 99,
            imageTokenCount: 2
        )

        try expander.validatePretokenizedTokenIDs([1, 99, 99, 2])
        #expect(
            throws: SwiftLMVisionLanguageModelError.invalidImagePlaceholderCount(
                expected: 2,
                actual: 1
            )
        ) {
            try expander.validatePretokenizedTokenIDs([1, 99, 2])
        }
    }

    private var visionLanguageMetadata: String {
        """
        {
          "metadata_version": "0.2",
          "kind": "vlm",
          "name": "test-vlm",
          "assets": {
            "main": "decoder.aimodel",
            "embedding": "embed.aimodel",
            "vision": "vision.aimodel"
          },
          "language": {
            "tokenizer": "test/tokenizer",
            "vocab_size": 152064,
            "max_context_length": 4096,
            "embedded_tokenizer": false
          },
          "vision": {
            "image_size": 448,
            "patch_size": 16,
            "image_token_count": 196,
            "image_token_id": 151655,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "rescale_factor": 0.00392156862745098
          }
        }
        """
    }

    private func makeBundleDirectory(metadata: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "swift-lm-foundation-model-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try Data(metadata.utf8).write(to: directory.appending(path: "metadata.json"))
        return directory
    }
}
