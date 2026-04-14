import LMIR

// MARK: - Collected Primitive

/// A primitive fragment collected during fragment tree walk, before emission.
///
/// The compiler collects all primitives from a composite fragment's tree,
/// then emits them as dispatch entries.
public struct CollectedPrimitive: Sendable {
    /// The primitive fragment (carries dispatchDimension, isFusable, isInPlace, etc.).
    public let fragment: any PrimitiveMetalKernelFragment
    /// Parameter bindings for weight resolution (layer-resolved).
    public let parameterBindings: [ParameterBinding]
    /// Layer index for this operation (nil if not in a repeating block).
    public let layerIndex: Int?

    public init(fragment: any PrimitiveMetalKernelFragment,
                parameterBindings: [ParameterBinding],
                layerIndex: Int?) {
        self.fragment = fragment
        self.parameterBindings = parameterBindings
        self.layerIndex = layerIndex
    }
}

// MARK: - Optimization Report

/// Diagnostic report from the cross-component fusion pass.
///
/// Compares dispatch counts before and after fusion, with pattern breakdown.
public struct OptimizationReport: Sendable {
    public let unfusedCount: Int
    public let optimizedCount: Int
    public let patterns: [PatternMatch]
    public var totalSaved: Int { unfusedCount - optimizedCount }

    public struct PatternMatch: Sendable {
        public let name: String
        public let count: Int
        public let savedDispatches: Int

        public init(name: String, count: Int, savedDispatches: Int) {
            self.name = name
            self.count = count
            self.savedDispatches = savedDispatches
        }
    }

    public init(unfusedCount: Int, optimizedCount: Int, patterns: [PatternMatch]) {
        self.unfusedCount = unfusedCount
        self.optimizedCount = optimizedCount
        self.patterns = patterns
    }

    /// Print a formatted report to stdout.
    public func printReport() {
        print("[Fusion] \(unfusedCount) → \(optimizedCount) dispatches (saved \(totalSaved))")
        for p in patterns {
            print("  \(p.name): \(p.count)× saves \(p.savedDispatches)")
        }
    }
}
