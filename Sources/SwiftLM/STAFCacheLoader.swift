import Foundation
import Metal
import MetalCompiler

struct STAFCacheLoader {
    func load(
        resources: ModelBundleResources,
        device: MTLDevice
    ) throws -> STAFWeightStore {
        let metadataStart = CFAbsoluteTimeGetCurrent()
        let metadata = try STAFModelBundleMetadataBuilder().build(
            directory: resources.directory,
            modelType: resources.modelType,
            config: resources.config,
            configData: resources.configData,
            safetensorsURLs: resources.safetensorsURLs
        )
        let metadataTime = CFAbsoluteTimeGetCurrent() - metadataStart
        InternalLog.info("[Prewarm/STAF] metadata: \(String(format: "%.3f", metadataTime))s")

        let stafURL = resources.directory.appendingPathComponent("model.staf")
        let converter = STAFConverter()
        let needsConversion: Bool

        if FileManager.default.fileExists(atPath: stafURL.path) {
            let validateStart = CFAbsoluteTimeGetCurrent()
            let isValid = try converter.isValid(
                stafURL: stafURL,
                safetensorsURLs: resources.safetensorsURLs,
                expectedMetadata: metadata
            )
            let validateTime = CFAbsoluteTimeGetCurrent() - validateStart
            needsConversion = !isValid
            if isValid {
                InternalLog.info("[Prewarm/STAF] cache hit  (validated in \(String(format: "%.3f", validateTime))s)")
            } else {
                InternalLog.info("[Prewarm/STAF] cache miss (rejected in \(String(format: "%.3f", validateTime))s)")
            }
        } else {
            needsConversion = true
            InternalLog.info("[Prewarm/STAF] cache miss (no model.staf)")
        }

        if needsConversion {
            let convertStart = CFAbsoluteTimeGetCurrent()
            let quantization = try HFConfigDecoder().quantizationHint(from: resources.configData)
            try converter.convert(
                safetensorsURLs: resources.safetensorsURLs,
                outputURL: stafURL,
                quantization: quantization,
                metadata: metadata
            )
            let convertTime = CFAbsoluteTimeGetCurrent() - convertStart
            InternalLog.info("[Prewarm/STAF] convert: \(String(format: "%.3f", convertTime))s")
        }

        let mmapStart = CFAbsoluteTimeGetCurrent()
        let weightStore = try STAFLoader().load(at: stafURL, device: device)
        let mmapTime = CFAbsoluteTimeGetCurrent() - mmapStart
        InternalLog.info("[Prewarm/STAF] mmap: \(String(format: "%.3f", mmapTime))s")
        return weightStore
    }
}
