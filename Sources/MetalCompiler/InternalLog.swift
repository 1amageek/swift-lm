import Foundation

/// Internal logging helpers for MetalCompiler.
///
/// - `info`: compiler statistics and diagnostics intended for developers
///   debugging the compiler. Silent by default; enable with
///   `SWIFTLM_DEBUG_COMPILER=1`.
/// - `error`: execution and compilation failures. Always written to stderr
///   so callers that ignore `throws` still see the reason (no silent
///   fallback).
enum InternalLog {
    private static let debugCompilerEnabled: Bool = {
        ProcessInfo.processInfo.environment["SWIFTLM_DEBUG_COMPILER"] == "1"
    }()

    static func info(_ message: @autoclosure () -> String) {
        guard debugCompilerEnabled else { return }
        print(message())
    }

    static func error(_ message: @autoclosure () -> String) {
        let line = message() + "\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}
