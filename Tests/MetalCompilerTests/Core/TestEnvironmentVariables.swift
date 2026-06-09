import Darwin
import Foundation

enum TestEnvironmentVariables {
    private static let lock = NSRecursiveLock()

    static func withValue<T>(
        _ key: String,
        value: String?,
        _ body: () throws -> T
    ) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }

        let previous = ProcessInfo.processInfo.environment[key]
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let previous {
                setenv(key, previous, 1)
            } else {
                unsetenv(key)
            }
        }
        return try body()
    }
}
