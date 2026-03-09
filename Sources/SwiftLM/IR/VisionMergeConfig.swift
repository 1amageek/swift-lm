/// Configuration for merging vision embeddings into text embeddings.
///
/// Used as the associated value of `ParallelMergeStrategy.visionMerge`.
/// The merge operation replaces placeholder token positions in the text
/// sequence with encoded vision features.
public struct VisionMergeConfig: Codable, Equatable, Sendable {

    /// Token ID used as placeholder for image content in the text sequence.
    /// Vision embeddings replace positions where this token appears.
    public let imageTokenId: Int

    /// Optional token ID for video content placeholders.
    public let videoTokenId: Int?

    public init(
        imageTokenId: Int,
        videoTokenId: Int? = nil
    ) {
        self.imageTokenId = imageTokenId
        self.videoTokenId = videoTokenId
    }
}
