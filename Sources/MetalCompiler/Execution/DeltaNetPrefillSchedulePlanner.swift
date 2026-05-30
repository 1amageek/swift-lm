import Foundation

public enum DeltaNetPrefillScheduleID: String, CaseIterable, Sendable {
    case sequentialState = "sequential_state"
    case parallelState = "parallel_state"
    case blockStreaming = "block_streaming"
    case blockScan = "block_scan"
    case blockFusedProjection = "block_fused_projection"
}

public enum DeltaNetPrefillGate: String, Sendable {
    case baseline
    case pass
    case pending
    case missing
    case fail
}

public struct DeltaNetPrefillScheduleShape: Sendable, Equatable {
    public let headCount: Int
    public let groupCount: Int
    public let keyDimension: Int
    public let valueDimension: Int
    public let sequenceLength: Int
    public let threadgroupWidth: Int
    public let threadgroupMemoryLimitBytes: Int
    public let blockSize: Int
    public let streamingHeadsPerThreadgroup: Int

    public init(
        headCount: Int,
        groupCount: Int,
        keyDimension: Int,
        valueDimension: Int,
        sequenceLength: Int,
        threadgroupWidth: Int,
        threadgroupMemoryLimitBytes: Int,
        blockSize: Int,
        streamingHeadsPerThreadgroup: Int
    ) {
        self.headCount = headCount
        self.groupCount = groupCount
        self.keyDimension = keyDimension
        self.valueDimension = valueDimension
        self.sequenceLength = sequenceLength
        self.threadgroupWidth = threadgroupWidth
        self.threadgroupMemoryLimitBytes = threadgroupMemoryLimitBytes
        self.blockSize = blockSize
        self.streamingHeadsPerThreadgroup = streamingHeadsPerThreadgroup
    }

    public static func qwen35Default(sequenceLength: Int) -> Self {
        Self(
            headCount: 16,
            groupCount: 16,
            keyDimension: 128,
            valueDimension: 128,
            sequenceLength: sequenceLength,
            threadgroupWidth: 384,
            threadgroupMemoryLimitBytes: 32_768,
            blockSize: 4,
            streamingHeadsPerThreadgroup: 4
        )
    }
}

public struct DeltaNetPrefillScheduleEvidence: Sendable, Equatable {
    public let correctnessGate: DeltaNetPrefillGate
    public let referenceGate: DeltaNetPrefillGate
    public let profileGate: DeltaNetPrefillGate
    public let experimentRequested: Bool

    public init(
        correctnessGate: DeltaNetPrefillGate,
        referenceGate: DeltaNetPrefillGate,
        profileGate: DeltaNetPrefillGate,
        experimentRequested: Bool = false
    ) {
        self.correctnessGate = correctnessGate
        self.referenceGate = referenceGate
        self.profileGate = profileGate
        self.experimentRequested = experimentRequested
    }

    public static let missing = Self(
        correctnessGate: .missing,
        referenceGate: .missing,
        profileGate: .missing
    )
}

public struct DeltaNetPrefillScheduleDecision: Sendable, Equatable {
    public let scheduleID: DeltaNetPrefillScheduleID
    public let scheduleClass: String
    public let supportsNonzeroInitialState: Bool
    public let staticThreadgroupBytes: Int
    public let threadgroupMemoryLimitBytes: Int
    public let estimatedStateGlobalBytesPerToken: Int
    public let baselineStateGlobalBytesPerToken: Int
    public let estimatedStateTrafficReductionPercent: Double
    public let staticFeasibility: String
    public let staticRejectReason: String
    public let correctnessGate: DeltaNetPrefillGate
    public let referenceGate: DeltaNetPrefillGate
    public let profileGate: DeltaNetPrefillGate
    public let routePromotion: String

    public var isProductionCandidate: Bool {
        routePromotion == "candidate-production-route"
    }
}

public enum DeltaNetPrefillSchedulePlanner {
    public static let requiredTrafficReductionPercent = 50.0

    public static func decisions(
        shape: DeltaNetPrefillScheduleShape,
        evidenceBySchedule: [DeltaNetPrefillScheduleID: DeltaNetPrefillScheduleEvidence] = [:]
    ) -> [DeltaNetPrefillScheduleDecision] {
        DeltaNetPrefillScheduleID.allCases.map { scheduleID in
            decision(
                scheduleID: scheduleID,
                shape: shape,
                evidence: evidenceBySchedule[scheduleID] ?? defaultEvidence(scheduleID)
            )
        }
    }

    public static func productionDecision(
        shape: DeltaNetPrefillScheduleShape,
        evidenceBySchedule: [DeltaNetPrefillScheduleID: DeltaNetPrefillScheduleEvidence] = [:]
    ) -> DeltaNetPrefillScheduleDecision {
        let decisions = decisions(shape: shape, evidenceBySchedule: evidenceBySchedule)
        if let candidate = decisions.first(where: \.isProductionCandidate) {
            return candidate
        }
        return decisions.first { $0.scheduleID == .parallelState }
            ?? decisions.first { $0.scheduleID == .sequentialState }
            ?? decision(scheduleID: .sequentialState, shape: shape, evidence: defaultEvidence(.sequentialState))
    }

    private static func decision(
        scheduleID: DeltaNetPrefillScheduleID,
        shape: DeltaNetPrefillScheduleShape,
        evidence: DeltaNetPrefillScheduleEvidence
    ) -> DeltaNetPrefillScheduleDecision {
        let supportsNonzeroState = supportsNonzeroInitialState(scheduleID)
        let staticBytes = staticThreadgroupBytes(scheduleID: scheduleID, shape: shape)
        let baselineBytes = baselineStateGlobalBytesPerToken(shape)
        let estimatedBytes = estimatedStateGlobalBytesPerToken(scheduleID: scheduleID, shape: shape)
        let reduction = stateTrafficReductionPercent(baselineBytes: baselineBytes, estimatedBytes: estimatedBytes)
        let staticFeasibility = staticFeasibility(
            scheduleID: scheduleID,
            supportsNonzeroInitialState: supportsNonzeroState,
            staticThreadgroupBytes: staticBytes,
            threadgroupMemoryLimitBytes: shape.threadgroupMemoryLimitBytes,
            estimatedStateTrafficReductionPercent: reduction
        )
        let staticRejectReason = staticFeasibility == "pass" || staticFeasibility == "baseline"
            ? ""
            : staticFeasibility
        let routePromotion = routePromotion(
            scheduleID: scheduleID,
            staticFeasibility: staticFeasibility,
            evidence: evidence
        )
        return DeltaNetPrefillScheduleDecision(
            scheduleID: scheduleID,
            scheduleClass: scheduleClass(scheduleID),
            supportsNonzeroInitialState: supportsNonzeroState,
            staticThreadgroupBytes: staticBytes,
            threadgroupMemoryLimitBytes: shape.threadgroupMemoryLimitBytes,
            estimatedStateGlobalBytesPerToken: estimatedBytes,
            baselineStateGlobalBytesPerToken: baselineBytes,
            estimatedStateTrafficReductionPercent: reduction,
            staticFeasibility: staticFeasibility,
            staticRejectReason: staticRejectReason,
            correctnessGate: evidence.correctnessGate,
            referenceGate: evidence.referenceGate,
            profileGate: evidence.profileGate,
            routePromotion: routePromotion
        )
    }

    private static func defaultEvidence(_ scheduleID: DeltaNetPrefillScheduleID) -> DeltaNetPrefillScheduleEvidence {
        switch scheduleID {
        case .sequentialState, .parallelState:
            return DeltaNetPrefillScheduleEvidence(
                correctnessGate: .baseline,
                referenceGate: .baseline,
                profileGate: .baseline
            )
        case .blockStreaming, .blockScan, .blockFusedProjection:
            return .missing
        }
    }

    private static func routePromotion(
        scheduleID: DeltaNetPrefillScheduleID,
        staticFeasibility: String,
        evidence: DeltaNetPrefillScheduleEvidence
    ) -> String {
        if scheduleID == .sequentialState {
            return "baseline-reference-route"
        }
        if scheduleID == .parallelState {
            return "current-production-route"
        }
        guard staticFeasibility == "pass" else {
            return "reject-static-feasibility"
        }
        if evidence.experimentRequested {
            return "experiment-only"
        }
        guard evidence.correctnessGate == .pass else {
            return "reject-missing-correctness-gate"
        }
        guard evidence.referenceGate == .pass else {
            return "reject-missing-reference-gate"
        }
        guard evidence.profileGate == .pass else {
            return "reject-missing-profile-gate"
        }
        return "candidate-production-route"
    }

    private static func staticFeasibility(
        scheduleID: DeltaNetPrefillScheduleID,
        supportsNonzeroInitialState: Bool,
        staticThreadgroupBytes: Int,
        threadgroupMemoryLimitBytes: Int,
        estimatedStateTrafficReductionPercent: Double
    ) -> String {
        if scheduleID == .sequentialState || scheduleID == .parallelState {
            return "baseline"
        }
        guard supportsNonzeroInitialState else {
            return "reject-prefill-state-contract"
        }
        guard staticThreadgroupBytes <= threadgroupMemoryLimitBytes else {
            return "reject-threadgroup-memory-limit"
        }
        guard estimatedStateTrafficReductionPercent >= requiredTrafficReductionPercent else {
            return "reject-state-traffic-target"
        }
        return "pass"
    }

    private static func scheduleClass(_ scheduleID: DeltaNetPrefillScheduleID) -> String {
        switch scheduleID {
        case .sequentialState:
            return "reference"
        case .parallelState:
            return "current-production"
        case .blockStreaming:
            return "state-traffic-reduction"
        case .blockScan:
            return "sequence-parallel-scan"
        case .blockFusedProjection:
            return "block-fusion"
        }
    }

    private static func supportsNonzeroInitialState(_ scheduleID: DeltaNetPrefillScheduleID) -> Bool {
        switch scheduleID {
        case .sequentialState, .parallelState, .blockStreaming, .blockFusedProjection:
            return true
        case .blockScan:
            return false
        }
    }

    private static func baselineStateGlobalBytesPerToken(_ shape: DeltaNetPrefillScheduleShape) -> Int {
        stateBytesPerToken(shape) * 3
    }

    private static func estimatedStateGlobalBytesPerToken(
        scheduleID: DeltaNetPrefillScheduleID,
        shape: DeltaNetPrefillScheduleShape
    ) -> Int {
        switch scheduleID {
        case .sequentialState, .parallelState:
            return baselineStateGlobalBytesPerToken(shape)
        case .blockStreaming:
            return stateBytesPerToken(shape) * 2 / max(shape.blockSize, 1)
        case .blockScan:
            return 0
        case .blockFusedProjection:
            return baselineStateGlobalBytesPerToken(shape)
        }
    }

    private static func staticThreadgroupBytes(
        scheduleID: DeltaNetPrefillScheduleID,
        shape: DeltaNetPrefillScheduleShape
    ) -> Int {
        let floatBytes = MemoryLayout<Float>.stride
        let convSiluCacheBytes = shape.threadgroupWidth * floatBytes
        let scalarCacheBytes = 3 * floatBytes
        switch scheduleID {
        case .sequentialState, .parallelState, .blockScan, .blockFusedProjection:
            return convSiluCacheBytes + scalarCacheBytes
        case .blockStreaming:
            let heads = max(shape.streamingHeadsPerThreadgroup, 1)
            let stateReductionLanes = max(1, shape.threadgroupWidth / max(1, heads * shape.valueDimension))
            let partialOutputBytes = heads
                * shape.valueDimension
                * max(shape.blockSize, 1)
                * stateReductionLanes
                * floatBytes
            let deltaBytes = heads
                * shape.valueDimension
                * max(shape.blockSize, 1)
                * floatBytes
            return convSiluCacheBytes + scalarCacheBytes + partialOutputBytes + deltaBytes
        }
    }

    private static func stateBytesPerToken(_ shape: DeltaNetPrefillScheduleShape) -> Int {
        shape.headCount * shape.keyDimension * shape.valueDimension * MemoryLayout<Float>.stride
    }

    private static func stateTrafficReductionPercent(baselineBytes: Int, estimatedBytes: Int) -> Double {
        guard baselineBytes > 0 else { return 0.0 }
        return Double(baselineBytes - estimatedBytes) / Double(baselineBytes) * 100.0
    }
}
