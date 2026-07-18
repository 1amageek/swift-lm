import Foundation

@available(macOS 27.0, iOS 27.0, *)
struct SwiftLMVisionPromptTokenExpander: Sendable {
    let imageTokenID: Int32
    let imageTokenCount: Int

    func expandTemplatedTokenIDs(_ tokenIDs: [Int32]) throws -> [Int32] {
        let placeholderCount = countPlaceholders(in: tokenIDs)
        guard placeholderCount == 1 else {
            throw SwiftLMVisionLanguageModelError.invalidImagePlaceholderCount(
                expected: 1,
                actual: placeholderCount
            )
        }

        var expandedTokenIDs: [Int32] = []
        expandedTokenIDs.reserveCapacity(tokenIDs.count + imageTokenCount - 1)
        for tokenID in tokenIDs {
            if tokenID == imageTokenID {
                expandedTokenIDs.append(
                    contentsOf: repeatElement(imageTokenID, count: imageTokenCount)
                )
            } else {
                expandedTokenIDs.append(tokenID)
            }
        }
        return expandedTokenIDs
    }

    func validatePretokenizedTokenIDs(_ tokenIDs: [Int32]) throws {
        let actualCount = countPlaceholders(in: tokenIDs)
        guard actualCount == imageTokenCount else {
            throw SwiftLMVisionLanguageModelError.invalidImagePlaceholderCount(
                expected: imageTokenCount,
                actual: actualCount
            )
        }
    }

    private func countPlaceholders(in tokenIDs: [Int32]) -> Int {
        tokenIDs.count(where: { $0 == imageTokenID })
    }
}
