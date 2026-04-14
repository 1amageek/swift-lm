extension DispatchEntry {
    var decodeWeightBindingBase: (roles: [String], inputDimension: Int, outputDimension: Int)? {
        guard let projection = fragment as? ProjectionDescribing else { return nil }
        let fields = projection.projectionFields
        guard let first = fields.first else { return nil }
        return (
            fields.map(\.field),
            first.inputDimension,
            fields.reduce(0) { $0 + $1.outputDimension }
        )
    }
}
