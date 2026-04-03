import Foundation

struct STAFConversionPlan: Sendable {
    let sortedURLs: [URL]
    let entries: [STAFConversionEntry]
}
