import CoreAILanguageModels
import Foundation

/// Vision metadata declared by an Apple Core AI VLM bundle.
@available(macOS 27.0, iOS 27.0, *)
public struct SwiftLMVisionConfiguration: Sendable, Equatable {
    public let imageSize: Int
    public let patchSize: Int
    public let imageTokenCount: Int
    public let imageTokenID: Int32
    public let imageMean: [Double]
    public let imageStandardDeviation: [Double]
    public let rescaleFactor: Double

    init(_ configuration: VisionConfig) {
        imageSize = configuration.imageSize
        patchSize = configuration.patchSize
        imageTokenCount = configuration.imageTokenCount
        imageTokenID = configuration.imageTokenId
        imageMean = configuration.imageMean
        imageStandardDeviation = configuration.imageStd
        rescaleFactor = configuration.rescaleFactor
    }
}
