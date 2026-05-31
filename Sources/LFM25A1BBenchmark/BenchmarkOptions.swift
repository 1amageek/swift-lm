import Foundation

struct BenchmarkOptions {
    let modelDirectory: URL?
    let maxTokens: Int
    let warmupIterations: Int
    let iterations: Int
    let prompt: String
    let requiresM5Gate: Bool

    init(arguments: [String]) throws {
        var modelDirectory: URL?
        var maxTokens = 64
        var warmupIterations = 1
        var iterations = 1
        var prompt = "What is the capital of Japan? Answer with exactly one word."
        var requiresM5Gate = false

        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--model":
                index += 1
                guard index < arguments.count else { throw BenchmarkError.missingValue(argument) }
                modelDirectory = URL(fileURLWithPath: arguments[index])
            case "--tokens":
                index += 1
                guard index < arguments.count else { throw BenchmarkError.missingValue(argument) }
                guard let value = Int(arguments[index]), value > 0, value <= 64 else {
                    throw BenchmarkError.invalidValue(argument, arguments[index])
                }
                maxTokens = value
            case "--warmup":
                index += 1
                guard index < arguments.count else { throw BenchmarkError.missingValue(argument) }
                guard let value = Int(arguments[index]), value >= 0 else {
                    throw BenchmarkError.invalidValue(argument, arguments[index])
                }
                warmupIterations = value
            case "--iterations":
                index += 1
                guard index < arguments.count else { throw BenchmarkError.missingValue(argument) }
                guard let value = Int(arguments[index]), value > 0 else {
                    throw BenchmarkError.invalidValue(argument, arguments[index])
                }
                iterations = value
            case "--prompt":
                index += 1
                guard index < arguments.count else { throw BenchmarkError.missingValue(argument) }
                prompt = arguments[index]
            case "--require-m5":
                requiresM5Gate = true
            case "--help":
                throw BenchmarkError.help
            default:
                throw BenchmarkError.unknownArgument(argument)
            }
            index += 1
        }

        self.modelDirectory = modelDirectory
        self.maxTokens = maxTokens
        self.warmupIterations = warmupIterations
        self.iterations = iterations
        self.prompt = prompt
        self.requiresM5Gate = requiresM5Gate
    }
}
