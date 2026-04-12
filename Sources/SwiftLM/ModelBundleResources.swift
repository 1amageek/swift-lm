import Foundation
import Jinja
import LMArchitecture

struct ModelBundleResources {
    let directory: URL
    let configData: Data
    let config: ModelConfig
    let modelType: String
    let safetensorsURLs: [URL]
    let modulesData: Data?
    let sentenceTransformersConfigData: Data?
    let chatTemplate: Template?
    let chatTemplateSource: String?
    let preprocessorConfigData: Data?
    let inputCapabilities: ModelInputCapabilities
    let visionConfiguration: ModelVisionConfiguration?
}
