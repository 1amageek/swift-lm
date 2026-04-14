/// Synthesizes a single fused MSL kernel from multiple adjacent fragments.
///
/// The synthesizer operates entirely on `FusionContract` and `kernelBody()` —
/// it never inspects concrete fragment types. This keeps the compiler generic
/// and ensures any fragment that provides a contract + body participates in
/// automatic fusion.
///
/// Pipeline:
/// 1. Compatibility check: all parallelism patterns must be compatible
/// 2. Intermediate storage: determine register vs threadgroup for each junction
/// 3. Variable renaming: producer output port → consumer input port
/// 4. Body concatenation: sequential composition of renamed bodies
/// 5. Contract merging: external-only ports, combined scalar constants, resolved parallelism
/// 6. KernelScaffold wrapping: merged contract + concatenated body → complete MSL
public struct FusionSynthesizer {

    /// A fragment entry for fusion synthesis.
    public struct Entry: Sendable {
        /// Fragment's fusion contract.
        public let contract: FusionContract
        /// MSL computation body from `kernelBody()`.
        public let body: String
        /// Resolved weight formats per weight port name.
        public let weightFormats: [String: WeightFormat]

        public init(contract: FusionContract, body: String, weightFormats: [String: WeightFormat] = [:]) {
            self.contract = contract
            self.body = body
            self.weightFormats = weightFormats
        }
    }

    /// Result of fusion synthesis.
    public struct SynthesisResult: Sendable {
        /// Merged fusion contract (external ports only).
        public let contract: FusionContract
        /// Concatenated and renamed MSL body.
        public let body: String
        /// Combined weight formats from all entries.
        public let weightFormats: [String: WeightFormat]
    }

    /// Synthesis error.
    public enum SynthesisError: Error, Sendable {
        /// Fewer than 2 entries provided.
        case insufficientEntries
        /// Adjacent fragments have incompatible parallelism.
        case incompatibleParallelism(KernelParallelism, KernelParallelism)
        /// No connectable port pair between adjacent fragments.
        case noConnectablePort(producerIndex: Int, consumerIndex: Int)
    }

    /// Synthesize a fused kernel body from a sequence of fragment entries.
    ///
    /// Entries must be in execution order: entry[0] produces data consumed by entry[1], etc.
    ///
    /// - Returns: A `SynthesisResult` containing the merged contract, concatenated body,
    ///   and combined weight formats. Pass these to `KernelScaffold.generate()` to produce
    ///   the complete MSL kernel.
    public static func synthesize(_ entries: [Entry]) throws -> SynthesisResult {
        guard entries.count >= 2 else {
            throw SynthesisError.insufficientEntries
        }

        // Phase 1: Verify parallelism compatibility across all entries
        var resolvedParallelism = entries[0].contract.parallelism
        for i in 1..<entries.count {
            let next = entries[i].contract.parallelism
            guard resolvedParallelism.isCompatible(with: next) else {
                throw SynthesisError.incompatibleParallelism(resolvedParallelism, next)
            }
            resolvedParallelism = resolvedParallelism.resolved(with: next)
        }

        // Phase 2: Determine intermediate storage and build rename map
        var bodyParts: [String] = []

        /// Tracks how each renamed variable should be substituted.
        /// `.scalar` strips array subscripts (register intermediate).
        /// `.array` preserves subscripts (threadgroup memory intermediate).
        enum RenameKind {
            case scalar   // register: `name[idx]` → `newName`
            case array    // threadgroup: `name` → `newName` (subscript preserved)
        }
        var renameMap: [(oldName: String, newName: String, kind: RenameKind)] = []
        var intermediateNames: Set<String> = []

        for i in 0..<entries.count {
            let entry = entries[i]
            var body = entry.body

            // Phase 2a: Parallelism adaptation
            // When the fused kernel runs under perRow scaffold but a fragment's
            // body was written for perElement dispatch, wrap it in a cooperative loop.
            let entryParallelism = entry.contract.parallelism
            if case .perRow = resolvedParallelism, case .perElement = entryParallelism {
                body = wrapPerElementBodyForPerRow(body)
            }

            // Apply accumulated renames to this body
            for rename in renameMap {
                switch rename.kind {
                case .scalar:
                    body = replaceArrayAccessWithScalar(in: body, arrayName: rename.oldName, scalarName: rename.newName)
                case .array:
                    body = replaceVariableName(in: body, oldName: rename.oldName, newName: rename.newName)
                }
            }

            if i < entries.count - 1 {
                let producer = entries[i].contract
                let consumer = entries[i + 1].contract
                let storage = producer.intermediateStorage(to: consumer)

                // Find the connection: producer's primary output → consumer's primary input
                guard let producerOutput = producer.primaryOutput,
                      let consumerInput = consumer.primaryInput else {
                    throw SynthesisError.noConnectablePort(producerIndex: i, consumerIndex: i + 1)
                }

                let intermediateName: String
                switch storage {
                case .register:
                    // Register intermediate: strip subscripts from array access
                    intermediateName = "_fused_\(i)"
                    let resolvedOutputName = renameMap.first(where: { $0.oldName == producerOutput.name })?.newName ?? producerOutput.name
                    body = replaceArrayAccessWithScalar(
                        in: body,
                        arrayName: resolvedOutputName,
                        scalarName: intermediateName
                    )
                    renameMap.append((consumerInput.name, intermediateName, .scalar))
                    intermediateNames.insert(intermediateName)

                case .threadgroupMemory(let dimension):
                    // Threadgroup memory: simple variable rename (subscript preserved)
                    intermediateName = "_tg_fused_\(i)"
                    let resolvedOutputName = renameMap.first(where: { $0.oldName == producerOutput.name })?.newName ?? producerOutput.name
                    body = replaceVariableName(
                        in: body,
                        oldName: resolvedOutputName,
                        newName: intermediateName
                    )
                    renameMap.append((consumerInput.name, intermediateName, .array))
                    intermediateNames.insert(intermediateName)
                    _ = dimension
                }
            }

            bodyParts.append(body)
        }

        // Phase 3: Insert intermediate declarations and barriers
        var fusedBody = ""
        var partIndex = 0
        for i in 0..<entries.count {
            if i > 0 {
                let storage = entries[i - 1].contract.intermediateStorage(to: entries[i].contract)
                switch storage {
                case .register:
                    // Register declaration before the producer body that writes it
                    // (already handled inline — the write becomes an assignment)
                    break
                case .threadgroupMemory(let dimension):
                    // Threadgroup barrier between producer write and consumer read
                    fusedBody += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
                    _ = dimension  // dimension used in declaration at top
                }
            }

            if !fusedBody.isEmpty && !fusedBody.hasSuffix("\n\n") {
                fusedBody += "\n"
            }
            fusedBody += bodyParts[partIndex]
            partIndex += 1
        }

        // Phase 4: Prepend intermediate variable declarations
        var declarations = ""
        for i in 0..<(entries.count - 1) {
            let storage = entries[i].contract.intermediateStorage(to: entries[i + 1].contract)
            switch storage {
            case .register:
                declarations += "    float _fused_\(i);\n"
            case .threadgroupMemory(let dimension):
                declarations += "    threadgroup float _tg_fused_\(i)[\(dimension)];\n"
            }
        }
        if !declarations.isEmpty {
            fusedBody = declarations + "\n" + fusedBody
        }

        // Phase 5: Build merged contract
        let mergedContract = mergeContracts(entries: entries, resolvedParallelism: resolvedParallelism, intermediateNames: intermediateNames)

        // Phase 6: Combine weight formats
        var combinedWeightFormats: [String: WeightFormat] = [:]
        for entry in entries {
            for (key, value) in entry.weightFormats {
                combinedWeightFormats[key] = value
            }
        }

        return SynthesisResult(
            contract: mergedContract,
            body: fusedBody,
            weightFormats: combinedWeightFormats
        )
    }

    // MARK: - Contract Merging

    /// Merge contracts from entries into a single fused contract.
    ///
    /// Called by the optimizer at graph-level to build the merged contract
    /// without generating bodies. The optimizer stores this in SynthesizedFragment;
    /// actual body synthesis happens lazily in kernelSource().
    public static func mergeContracts(
        entries: [Entry],
        resolvedParallelism: KernelParallelism,
        storage: IntermediateStorage
    ) -> FusionContract {
        var intermediateTGMemory = 0
        if case .threadgroupMemory(let dim) = storage {
            intermediateTGMemory = dim * MemoryLayout<Float>.size
        }
        return mergeContractsInternal(
            entries: entries,
            resolvedParallelism: resolvedParallelism,
            intermediateTGMemory: intermediateTGMemory
        )
    }

    /// Internal merge used by synthesize() — tracks intermediate names from body processing.
    private static func mergeContracts(
        entries: [Entry],
        resolvedParallelism: KernelParallelism,
        intermediateNames: Set<String>
    ) -> FusionContract {
        var intermediateTGMemory = 0
        for i in 0..<(entries.count - 1) {
            let storage = entries[i].contract.intermediateStorage(to: entries[i + 1].contract)
            if case .threadgroupMemory(let dim) = storage {
                intermediateTGMemory += dim * MemoryLayout<Float>.size
            }
        }
        return mergeContractsInternal(
            entries: entries,
            resolvedParallelism: resolvedParallelism,
            intermediateTGMemory: intermediateTGMemory
        )
    }

    private static func mergeContractsInternal(
        entries: [Entry],
        resolvedParallelism: KernelParallelism,
        intermediateTGMemory: Int
    ) -> FusionContract {
        // Collect all external ports (not consumed as intermediates)
        var externalPorts: [FusionPort] = []

        for i in 0..<entries.count {
            let contract = entries[i].contract

            for port in contract.ports {
                // Skip the primary output port of non-final entries (connected to next entry).
                // Non-primary outputs (e.g., CopyFragment's residual side output) remain external.
                if port.direction == .output && i < entries.count - 1 {
                    if contract.primaryOutput?.name == port.name {
                        continue  // internal junction
                    }
                }

                // Skip intermediate input ports (connected from previous entry)
                if port.direction == .input && i > 0 {
                    if case .buffer = port.role {
                        // Check if this is the primary buffer input connected from previous
                        if contract.primaryInput?.name == port.name {
                            continue  // internal junction
                        }
                    }
                }

                externalPorts.append(port)
            }
        }

        // Merge scalar constants from all entries
        var allScalarConstants: [ScalarConstant] = []
        for entry in entries {
            allScalarConstants.append(contentsOf: entry.contract.scalarConstants)
        }

        // Combined threadgroup memory
        let baseTGMemory = entries.map(\.contract.threadgroupMemoryBytes).max() ?? 0
        var intermediateTGMemory = 0
        for i in 0..<(entries.count - 1) {
            let storage = entries[i].contract.intermediateStorage(to: entries[i + 1].contract)
            if case .threadgroupMemory(let dim) = storage {
                intermediateTGMemory += dim * MemoryLayout<Float>.size
            }
        }

        let requiresSIMD = entries.contains { $0.contract.requiresSIMDReduction }

        return FusionContract(
            ports: externalPorts,
            scalarConstants: allScalarConstants,
            parallelism: resolvedParallelism,
            threadgroupMemoryBytes: baseTGMemory + intermediateTGMemory,
            requiresSIMDReduction: requiresSIMD
        )
    }

    // MARK: - Variable Renaming

    /// Replace a variable name in MSL body code.
    ///
    /// Uses word boundary matching to avoid replacing substrings
    /// (e.g., replacing "data" should not affect "data_base").
    private static func replaceVariableName(in body: String, oldName: String, newName: String) -> String {
        guard oldName != newName else { return body }
        // Match word boundaries: the variable name must be preceded and followed
        // by non-identifier characters (or start/end of string)
        var result = ""
        let chars = Array(body)
        let oldChars = Array(oldName)
        var i = 0

        while i < chars.count {
            if i + oldChars.count <= chars.count {
                let slice = Array(chars[i..<(i + oldChars.count)])
                if slice == oldChars {
                    let before = i > 0 ? chars[i - 1] : " "
                    let after = (i + oldChars.count) < chars.count ? chars[i + oldChars.count] : " "
                    if !isIdentifierChar(before) && !isIdentifierChar(after) {
                        result += newName
                        i += oldChars.count
                        continue
                    }
                }
            }
            result.append(chars[i])
            i += 1
        }

        return result
    }

    private static func isIdentifierChar(_ c: Character) -> Bool {
        c.isLetter || c.isNumber || c == "_"
    }

    /// Apply accumulated renames to a port name.
    private static func applyRenames(_ name: String, _ renames: [String: String]) -> String {
        renames[name] ?? name
    }

    // MARK: - Parallelism Adaptation

    /// Wrap a perElement body to run under perRow scaffold.
    ///
    /// perElement body uses `i` (element index) and `idx` (flat buffer index).
    /// perRow scaffold provides `tid`, `threadgroupSize`, `dimension`.
    /// Under perRow, each row is processed by one threadgroup, so `idx == i`
    /// (the scaffold already computes row-local pointers).
    ///
    /// Wraps the body in a cooperative loop:
    /// ```metal
    /// for (uint i = tid; i < dimension; i += threadgroupSize) {
    ///     uint idx = i;
    ///     // original perElement body
    /// }
    /// ```
    static func wrapPerElementBodyForPerRow(_ body: String) -> String {
        let indented = body
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { "    \($0)" }
            .joined(separator: "\n")
        return """
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            uint idx = i;
        \(indented)
        }
        """
    }

    // MARK: - Intermediate Storage Transforms

    /// For register intermediate: replace `name[expr]` with scalar variable name.
    ///
    /// perElement bodies use array-indexed access (`output[idx]`, `data[idx]`),
    /// but a register intermediate is a single `float` per thread.
    /// This strips the subscript: `output[idx] = expr` → `_fused_0 = expr`,
    /// `data[idx]` → `_fused_0`.
    static func replaceArrayAccessWithScalar(
        in body: String,
        arrayName: String,
        scalarName: String
    ) -> String {
        var result = ""
        let chars = Array(body)
        let nameChars = Array(arrayName)
        var i = 0

        while i < chars.count {
            if i + nameChars.count <= chars.count {
                let slice = Array(chars[i..<(i + nameChars.count)])
                if slice == nameChars {
                    let before = i > 0 ? chars[i - 1] : " "
                    let afterIdx = i + nameChars.count
                    if !isIdentifierChar(before) {
                        if afterIdx < chars.count && chars[afterIdx] == "[" {
                            // Found `name[` — replace name and strip subscript
                            result += scalarName
                            var depth = 0
                            var j = afterIdx
                            while j < chars.count {
                                if chars[j] == "[" { depth += 1 }
                                if chars[j] == "]" {
                                    depth -= 1
                                    if depth == 0 { break }
                                }
                                j += 1
                            }
                            i = j + 1  // skip past ']'
                            continue
                        }
                        let after = afterIdx < chars.count ? chars[afterIdx] : " "
                        if !isIdentifierChar(after) {
                            // Bare name without subscript — replace as-is
                            result += scalarName
                            i += nameChars.count
                            continue
                        }
                    }
                }
            }
            result.append(chars[i])
            i += 1
        }
        return result
    }

    /// For register intermediate: strip subscripts from producer output and consumer input.
    private static func replaceRegisterOutput(
        in body: String,
        outputName: String,
        intermediateName: String
    ) -> String {
        replaceArrayAccessWithScalar(in: body, arrayName: outputName, scalarName: intermediateName)
    }

    /// For threadgroup intermediate: simple variable rename (subscript preserved).
    private static func replaceThreadgroupOutput(
        in body: String,
        outputName: String,
        threadgroupName: String,
        dimension: Int
    ) -> String {
        replaceVariableName(in: body, oldName: outputName, newName: threadgroupName)
    }
}
