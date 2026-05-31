import Foundation

enum BenchmarkError: Error, CustomStringConvertible {
    case help
    case missingValue(String)
    case invalidValue(String, String)
    case unknownArgument(String)
    case modelNotFound(String)
    case noRuns
    case traceMismatch(expected: [Int], actual: [Int])
    case textMismatch(expected: String, actual: String)
    case m5GateFailed(Double)

    var description: String {
        switch self {
        case .help:
            return "usage: lfm25-a1b-benchmark [--model PATH] [--tokens 1...64] [--iterations N] [--require-m5]"
        case .missingValue(let argument):
            return "missing value for \(argument)"
        case .invalidValue(let argument, let value):
            return "invalid value for \(argument): \(value)"
        case .unknownArgument(let argument):
            return "unknown argument: \(argument)"
        case .modelNotFound(let path):
            return "LFM2.5-8B-A1B snapshot not found under \(path)"
        case .noRuns:
            return "no benchmark runs were executed"
        case .traceMismatch(let expected, let actual):
            return "token trace mismatch expected=\(expected) actual=\(actual)"
        case .textMismatch(let expected, let actual):
            return "text mismatch expected=\(expected) actual=\(actual)"
        case .m5GateFailed(let tokensPerSecond):
            return String(format: "M5 gate failed: %.1f wall tok/s < 90.0", tokensPerSecond)
        }
    }
}
