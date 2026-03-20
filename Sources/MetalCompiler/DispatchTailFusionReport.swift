import Foundation

public struct DispatchTailFusionReport: Sendable {
    public struct Opportunity: Sendable {
        public let category: String
        public let label: String
        public let projectionKernelFamily: String?
        public let projectionInputDimension: Int?
        public let projectionOutputDimension: Int?
        public let isFeasibleInCurrentExecutionModel: Bool
        public let infeasibilityReason: String?
        public let count: Int
        public let potentialDispatchSavings: Int

        public init(
            category: String,
            label: String,
            projectionKernelFamily: String? = nil,
            projectionInputDimension: Int? = nil,
            projectionOutputDimension: Int? = nil,
            isFeasibleInCurrentExecutionModel: Bool,
            infeasibilityReason: String?,
            count: Int,
            potentialDispatchSavings: Int
        ) {
            self.category = category
            self.label = label
            self.projectionKernelFamily = projectionKernelFamily
            self.projectionInputDimension = projectionInputDimension
            self.projectionOutputDimension = projectionOutputDimension
            self.isFeasibleInCurrentExecutionModel = isFeasibleInCurrentExecutionModel
            self.infeasibilityReason = infeasibilityReason
            self.count = count
            self.potentialDispatchSavings = potentialDispatchSavings
        }
    }

    public let totalDispatches: Int
    public let outputProjectionDispatches: Int
    public let opportunities: [Opportunity]

    public init(
        totalDispatches: Int,
        outputProjectionDispatches: Int,
        opportunities: [Opportunity]
    ) {
        self.totalDispatches = totalDispatches
        self.outputProjectionDispatches = outputProjectionDispatches
        self.opportunities = opportunities
    }

    public var totalPotentialDispatchSavings: Int {
        opportunities.reduce(0) { $0 + $1.potentialDispatchSavings }
    }

    public var feasiblePotentialDispatchSavings: Int {
        opportunities
            .filter(\.isFeasibleInCurrentExecutionModel)
            .reduce(0) { $0 + $1.potentialDispatchSavings }
    }

    public func formatted() -> String {
        var lines: [String] = []
        lines.append("Tail fusion opportunities:")
        lines.append("  total dispatches: \(totalDispatches)")
        lines.append("  output projection dispatches: \(outputProjectionDispatches)")
        lines.append("  theoretical additional savings: \(totalPotentialDispatchSavings)")
        lines.append("  feasible additional savings (current execution model): \(feasiblePotentialDispatchSavings)")
        if opportunities.isEmpty {
            lines.append("  opportunities: none")
            return lines.joined(separator: "\n")
        }

        lines.append("  opportunities:")
        for opportunity in opportunities {
            let feasibility = opportunity.isFeasibleInCurrentExecutionModel ? "feasible" : "infeasible"
            let reason = opportunity.infeasibilityReason.map { " reason=\($0)" } ?? ""
            let projection = if let family = opportunity.projectionKernelFamily,
                                let input = opportunity.projectionInputDimension,
                                let output = opportunity.projectionOutputDimension {
                " projection=\(family)(\(input)->\(output))"
            } else {
                ""
            }
            lines.append(
                "    category=\(opportunity.category) label=\(opportunity.label)\(projection) count=\(opportunity.count) save=\(opportunity.potentialDispatchSavings) \(feasibility)\(reason)"
            )
        }
        return lines.joined(separator: "\n")
    }
}
