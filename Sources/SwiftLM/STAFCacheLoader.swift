import Foundation
import Metal
import MetalCompiler

struct STAFCacheLoader {
    func load(
        resources: ModelBundleResources,
        device: MTLDevice
    ) throws -> STAFWeightStore {
        let metadata = try STAFModelBundleMetadataBuilder().build(
            directory: resources.directory,
            modelType: resources.modelType,
            config: resources.config,
            configData: resources.configData,
            safetensorsURLs: resources.safetensorsURLs
        )

        let stafURL = resources.directory.appendingPathComponent("model.staf")
        let converter = STAFConverter()
        let needsConversion: Bool

        if FileManager.default.fileExists(atPath: stafURL.path) {
            needsConversion = !(try converter.isValid(
                stafURL: stafURL,
                safetensorsURLs: resources.safetensorsURLs,
                expectedMetadata: metadata
            ))
        } else {
            needsConversion = true
        }

        if needsConversion {
            let quantization = try HFConfigDecoder().quantizationHint(from: resources.configData)
            try converter.convert(
                safetensorsURLs: resources.safetensorsURLs,
                outputURL: stafURL,
                quantization: quantization,
                metadata: metadata
            )
        }

        return try STAFLoader().load(at: stafURL, device: device)
    }
}
