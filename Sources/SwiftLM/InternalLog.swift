import Foundation

/// Internal logging helpers for SwiftLM.
///
/// - `info`: telemetry output (load times, throughput). Silent by default;
///   enable with `SWIFTLM_VERBOSE=1`.
/// - `error`: failure diagnostics. Always written to stderr so callers that
///   ignore `throws` still see the reason (no silent fallback).
enum InternalLog {
    private static let verboseEnabled: Bool = {
        ProcessInfo.processInfo.environment["SWIFTLM_VERBOSE"] == "1"
    }()

    static func info(_ message: @autoclosure () -> String) {
        guard verboseEnabled else { return }
        print(message())
    }

    static func error(_ message: @autoclosure () -> String) {
        let line = message() + "\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}
