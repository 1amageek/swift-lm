import CoreImage
import Foundation
import MLX

/// Image preprocessor for Qwen 3.5 VL.
///
/// Performs smart resizing (dimensions divisible by imageFactor, aspect ratio preserved),
/// normalization, and pixel array construction for the vision encoder.
/// Unlike Qwen2.5-VL, uses Conv2d (no temporal dimension).
struct Qwen35VLImageProcessor: Sendable {

    let imageFactor: Int
    let patchSize: Int
    let spatialMergeSize: Int
    let minPixels: Int
    let maxPixels: Int
    let imageMean: [Float]
    let imageStd: [Float]

    init(config: Qwen35VLConfiguration.VisionConfiguration) {
        self.imageFactor = config.imageFactor
        self.patchSize = config.patchSize
        self.spatialMergeSize = config.spatialMergeSize
        self.minPixels = config.minPixels
        self.maxPixels = config.maxPixels
        self.imageMean = config.imageMean
        self.imageStd = config.imageStd
    }

    /// Preprocess a single image for the vision encoder.
    ///
    /// - Parameter image: Input image in any format.
    /// - Returns: Pixel tensor `[1, H, W, 3]` and grid dimensions for the vision encoder.
    func preprocess(image: CIImage) throws -> (MLXArray, LMInput.THW) {
        let (resizedWidth, resizedHeight) = smartResize(
            width: Int(image.extent.width),
            height: Int(image.extent.height)
        )

        let scaleX = CGFloat(resizedWidth) / image.extent.width
        let scaleY = CGFloat(resizedHeight) / image.extent.height
        let resized = image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        let context = CIContext()
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let width = resizedWidth
        let height = resizedHeight
        let bytesPerRow = width * 4

        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        context.render(
            resized,
            toBitmap: &pixelData,
            rowBytes: bytesPerRow,
            bounds: CGRect(x: 0, y: 0, width: width, height: height),
            format: .RGBA8,
            colorSpace: colorSpace
        )

        // Convert to float and normalize: [H, W, 3]
        var floatPixels = [Float](repeating: 0, count: height * width * 3)
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = y * bytesPerRow + x * 4
                let dstIdx = y * width * 3 + x * 3
                floatPixels[dstIdx + 0] = (Float(pixelData[srcIdx + 0]) / 255.0 - imageMean[0]) / imageStd[0]
                floatPixels[dstIdx + 1] = (Float(pixelData[srcIdx + 1]) / 255.0 - imageMean[1]) / imageStd[1]
                floatPixels[dstIdx + 2] = (Float(pixelData[srcIdx + 2]) / 255.0 - imageMean[2]) / imageStd[2]
            }
        }

        // [1, H, W, 3] (NHWC for Conv2d)
        let pixels = MLXArray(floatPixels, [1, height, width, 3])

        // Grid dimensions (in patch units, before spatial merge)
        let gridH = height / patchSize
        let gridW = width / patchSize
        let gridT = 1

        let thw = LMInput.THW(t: gridT, h: gridH, w: gridW)
        return (pixels, thw)
    }

    // MARK: - Smart Resize

    /// Compute target dimensions preserving aspect ratio with both sides divisible by imageFactor.
    func smartResize(width: Int, height: Int) -> (width: Int, height: Int) {
        let factor = imageFactor
        var h = height
        var w = width

        h = max(factor, ((h + factor / 2) / factor) * factor)
        w = max(factor, ((w + factor / 2) / factor) * factor)

        let totalPixels = h * w

        if totalPixels > maxPixels {
            let scale = sqrt(Double(maxPixels) / Double(totalPixels))
            h = Int(Double(h) * scale)
            w = Int(Double(w) * scale)
            h = max(factor, (h / factor) * factor)
            w = max(factor, (w / factor) * factor)
        }

        if h * w < minPixels {
            let scale = sqrt(Double(minPixels) / Double(h * w))
            h = Int(Double(h) * scale)
            w = Int(Double(w) * scale)
            h = max(factor, ((h + factor - 1) / factor) * factor)
            w = max(factor, ((w + factor - 1) / factor) * factor)
        }

        return (w, h)
    }
}
