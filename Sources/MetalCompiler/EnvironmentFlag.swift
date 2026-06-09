import Foundation

enum EnvironmentFlag {
    static func isEnabled(_ value: String?) -> Bool {
        guard let value else { return false }
        return value == "1" || value.lowercased() == "true"
    }
}
