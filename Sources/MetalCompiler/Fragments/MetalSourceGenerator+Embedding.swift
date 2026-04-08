extension MetalSourceGenerator {
    /// Generate MSL source for embedding lookup.
    public static func generateEmbeddingLookup(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        isSequence: Bool = true,
        embeddingScale: Float? = nil
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let scaleExpr = embeddingScale != nil ? " * scale" : ""

        if isSequence {
            let scaleParam = embeddingScale != nil ? "\n            constant float& scale            [[buffer(5)]]," : ""
            return """
            kernel void \(name)(
                device const int* tokenIDs       [[buffer(0)]],
                device const \(wt)* table        [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& embeddingDim      [[buffer(3)]],
                constant uint& sequenceLength    [[buffer(4)]],\(scaleParam)
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint dim = gid.x;
                uint seqPos = gid.y;
                if (dim >= embeddingDim || seqPos >= sequenceLength) return;
                int tokenID = tokenIDs[seqPos];
                output[seqPos * embeddingDim + dim] = \(bt)(float(\(readWeight("table[tokenID * embeddingDim + dim]")))\(scaleExpr));
            }
            """
        } else {
            let scaleParam = embeddingScale != nil ? "\n            constant float& scale            [[buffer(4)]]," : ""
            return """
            kernel void \(name)(
                device const int* tokenID        [[buffer(0)]],
                device const \(wt)* table        [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& embeddingDim      [[buffer(3)]],\(scaleParam)
                uint gid                         [[thread_position_in_grid]]
            ) {
                if (gid >= embeddingDim) return;
                output[gid] = \(bt)(float(\(readWeight("table[tokenID[0] * embeddingDim + gid]")))\(scaleExpr));
            }
            """
        }
    }

    public static func generateEmbeddingLookupArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const int* tokenID [[id(0)]];
            device const \(wt)* table [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& embeddingDim               [[buffer(3)]],
            uint gid                                  [[thread_position_in_grid]]
        ) {
            if (gid >= embeddingDim) return;
            args.output[gid] = \(bt)(\(readWeight("args.table[args.tokenID[0] * embeddingDim + gid]")));
        }
        """
    }
}
