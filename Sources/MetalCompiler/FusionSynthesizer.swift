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
    /// The synthesis groups entries into **LoopGroups** separated by threadgroup memory
    /// boundaries. Within each group, bodies share a single cooperative loop and use
    /// register intermediates. Between groups, threadgroup barriers synchronize writes.
    ///
    /// This ensures register intermediates are only read within the same loop iteration
    /// that wrote them — preventing the stale-value bug where separate loops would only
    /// retain the last iteration's value.
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

        // Phase 2: Split entries into LoopGroups.
        //
        // A LoopGroup is a maximal contiguous run of entries connected by register
        // intermediates. Threadgroup memory intermediates create group boundaries.
        // Within a group, bodies are concatenated into a single loop so register
        // variables are read in the same iteration that wrote them.
        let groups = buildLoopGroups(entries: entries, resolvedParallelism: resolvedParallelism)

        // Phase 3: Process each LoopGroup — rename variables and concatenate bodies.
        //
        // Rename scoping: register renames are group-local (same loop context).
        // Threadgroup renames propagate across groups (different loop contexts).
        var interGroupRenames: [(oldName: String, newName: String, kind: RenameKind)] = []
        var processedGroupBodies: [String] = []
        var threadgroupDeclarations: [String] = []

        for (groupIdx, group) in groups.enumerated() {
            var intraGroupRenames: [(oldName: String, newName: String, kind: RenameKind)] = []
            var groupBodyParts: [String] = []
            var registerDeclarations: [String] = []

            for (localIdx, entryIdx) in group.entryIndices.enumerated() {
                var body = entries[entryIdx].body

                // Apply inter-group renames (threadgroup intermediates from previous groups)
                body = applyRenames(body, interGroupRenames)

                // Apply intra-group renames (register intermediates from earlier in this group)
                body = applyRenames(body, intraGroupRenames)

                // Register intermediate to next entry within this group
                if localIdx < group.entryIndices.count - 1 {
                    let nextEntryIdx = group.entryIndices[localIdx + 1]
                    guard let producerOutput = entries[entryIdx].contract.primaryOutput,
                          let consumerInput = entries[nextEntryIdx].contract.primaryInput else {
                        throw SynthesisError.noConnectablePort(
                            producerIndex: entryIdx, consumerIndex: nextEntryIdx)
                    }
                    let intermediateName = "_fused_\(entryIdx)"
                    body = replaceArrayAccessWithScalar(
                        in: body,
                        arrayName: producerOutput.name,
                        scalarName: intermediateName
                    )
                    intraGroupRenames.append((consumerInput.name, intermediateName, .scalar))
                    registerDeclarations.append("    float \(intermediateName);")
                }

                groupBodyParts.append(body)
            }

            // Concatenate bodies within the group (same loop context)
            var groupBody = groupBodyParts.joined(separator: "\n")

            // Threadgroup intermediate to the next group
            if groupIdx < groups.count - 1 {
                let lastEntryIdx = group.entryIndices.last!
                let nextGroupFirstIdx = groups[groupIdx + 1].entryIndices.first!
                guard let producerOutput = entries[lastEntryIdx].contract.primaryOutput,
                      let consumerInput = entries[nextGroupFirstIdx].contract.primaryInput else {
                    throw SynthesisError.noConnectablePort(
                        producerIndex: lastEntryIdx, consumerIndex: nextGroupFirstIdx)
                }
                let tgName = "_tg_fused_\(groupIdx)"
                groupBody = replaceVariableName(
                    in: groupBody,
                    oldName: producerOutput.name,
                    newName: tgName
                )
                interGroupRenames.append((consumerInput.name, tgName, .array))

                let storage = entries[lastEntryIdx].contract.intermediateStorage(
                    to: entries[nextGroupFirstIdx].contract)
                if case .threadgroupMemory(let dimension) = storage {
                    threadgroupDeclarations.append(
                        "    threadgroup float \(tgName)[\(dimension)];")
                }
            }

            // Prepend register declarations inside the group body
            if !registerDeclarations.isEmpty {
                groupBody = registerDeclarations.joined(separator: "\n") + "\n" + groupBody
            }

            // Wrap the entire group in a cooperative loop if perElement bodies
            // are running under a perRow kernel scaffold
            if group.needsPerRowWrapping {
                groupBody = wrapPerElementBodyForPerRow(groupBody)
            }

            processedGroupBodies.append(groupBody)
        }

        // Phase 4: Assemble final body — declarations, groups, barriers
        var fusedBody = ""
        if !threadgroupDeclarations.isEmpty {
            fusedBody += threadgroupDeclarations.joined(separator: "\n") + "\n\n"
        }
        for (groupIdx, groupBody) in processedGroupBodies.enumerated() {
            if groupIdx > 0 {
                fusedBody += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
            }
            fusedBody += groupBody
            if groupIdx < processedGroupBodies.count - 1 {
                fusedBody += "\n\n"
            }
        }

        // Phase 5: Build merged contract
        let mergedContract = mergeContracts(
            entries: entries,
            resolvedParallelism: resolvedParallelism,
            intermediateNames: Set<String>()
        )

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

    // MARK: - LoopGroup

    /// A contiguous group of entries that share a single cooperative loop.
    ///
    /// Entries within a group are connected by register intermediates. Group
    /// boundaries occur at threadgroup memory intermediates where a barrier
    /// is required between the producer's write and the consumer's read.
    private struct LoopGroup {
        /// Indices into the entries array, in execution order.
        var entryIndices: [Int]
        /// Whether any entry in this group requires perElement→perRow wrapping.
        var needsPerRowWrapping: Bool
    }

    /// Split entries into LoopGroups based on intermediate storage type.
    private static func buildLoopGroups(
        entries: [Entry],
        resolvedParallelism: KernelParallelism
    ) -> [LoopGroup] {
        func needsWrapping(_ entry: Entry) -> Bool {
            if case .perRow = resolvedParallelism, case .perElement = entry.contract.parallelism {
                return true
            }
            return false
        }

        var groups: [LoopGroup] = []
        var current = LoopGroup(
            entryIndices: [0],
            needsPerRowWrapping: needsWrapping(entries[0])
        )

        for i in 1..<entries.count {
            let storage = entries[i - 1].contract.intermediateStorage(to: entries[i].contract)
            switch storage {
            case .register:
                current.entryIndices.append(i)
                if needsWrapping(entries[i]) {
                    current.needsPerRowWrapping = true
                }
            case .threadgroupMemory:
                groups.append(current)
                current = LoopGroup(
                    entryIndices: [i],
                    needsPerRowWrapping: needsWrapping(entries[i])
                )
            }
        }
        groups.append(current)
        return groups
    }

    /// Tracks how each renamed variable should be substituted.
    private enum RenameKind {
        /// Register: `name[idx]` → `newName` (strip array subscript).
        case scalar
        /// Threadgroup: `name` → `newName` (preserve subscript).
        case array
    }

    /// Apply a list of renames to a body string.
    private static func applyRenames(
        _ body: String,
        _ renames: [(oldName: String, newName: String, kind: RenameKind)]
    ) -> String {
        var result = body
        for rename in renames {
            switch rename.kind {
            case .scalar:
                result = replaceArrayAccessWithScalar(
                    in: result, arrayName: rename.oldName, scalarName: rename.newName)
            case .array:
                result = replaceVariableName(
                    in: result, oldName: rename.oldName, newName: rename.newName)
            }
        }
        return result
    }

    // MARK: - Contract Merging

    /// Merge contracts from N entries into a single fused contract.
    ///
    /// Called by the fusion pass to build the merged contract without generating
    /// bodies. The result is stored in SynthesizedFragment; actual body synthesis
    /// happens lazily in kernelSource().
    ///
    /// Handles arbitrary entry counts — computes intermediate storage for each
    /// adjacent pair automatically.
    public static func mergeContracts(
        entries: [Entry],
        resolvedParallelism: KernelParallelism
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

                // Deduplicate same-named residual ports that reference the same physical buffer.
                // When ResidualAdd reads "residual" (input/const) and CopyFragment writes
                // "residual" (output/non-const), they share the same physical residual buffer.
                // Merge into a single output (non-const) port so the MSL parameter is writable.
                if port.bufferIntent == .residual,
                   let existingIndex = externalPorts.firstIndex(where: {
                       $0.name == port.name && $0.bufferIntent == .residual
                   }) {
                    let existing = externalPorts[existingIndex]
                    if existing.direction != port.direction {
                        // Same residual buffer, different directions → promote to output (non-const).
                        let merged = FusionPort(
                            name: port.name,
                            direction: .output,
                            role: existing.role,
                            accessPattern: existing.accessPattern,
                            bufferIntent: .residual
                        )
                        externalPorts[existingIndex] = merged
                        continue
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
