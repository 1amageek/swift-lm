import Metal
import Testing

@Suite("SSM Chunked Affine Scan Reference")
struct SSMChunkedAffineScanReferenceTests {
    @Test("chunked affine scan matches sequential recurrence with nonzero state")
    func chunkedAffineScanMatchesSequentialRecurrenceWithNonzeroState() {
        let sequence = SSMChunkedAffineFixture.makeSequence(tokenCount: 9, keyDimension: 5, valueDimension: 7)
        let initialState = SSMChunkedAffineFixture.makeState(keyDimension: 5, valueDimension: 7)

        let sequential = SSMChunkedAffineReference.runSequential(sequence: sequence, initialState: initialState)
        let chunked = SSMChunkedAffineReference.runChunked(
            sequence: sequence,
            initialState: initialState,
            tileSize: 4
        )

        #expect(maxDifference(sequential.outputs, chunked.outputs) <= 0.000_001)
        #expect(maxDifference(sequential.finalState, chunked.finalState) <= 0.000_001)
    }

    @Test("chunked affine scan supports restored state boundary")
    func chunkedAffineScanSupportsRestoredStateBoundary() {
        let sequence = SSMChunkedAffineFixture.makeSequence(tokenCount: 10, keyDimension: 4, valueDimension: 6)
        let initialState = SSMChunkedAffineFixture.makeState(keyDimension: 4, valueDimension: 6)
        let prefix = Array(sequence.prefix(3))
        let suffix = Array(sequence.dropFirst(3))

        let fullSequential = SSMChunkedAffineReference.runSequential(sequence: sequence, initialState: initialState)
        let prefixSequential = SSMChunkedAffineReference.runSequential(sequence: prefix, initialState: initialState)
        let restoredChunked = SSMChunkedAffineReference.runChunked(
            sequence: suffix,
            initialState: prefixSequential.finalState,
            tileSize: 4
        )

        let stitchedOutputs = prefixSequential.outputs + restoredChunked.outputs
        #expect(maxDifference(fullSequential.outputs, stitchedOutputs) <= 0.000_001)
        #expect(maxDifference(fullSequential.finalState, restoredChunked.finalState) <= 0.000_001)
    }

    @Test("chunked affine scan tile size does not change results")
    func chunkedAffineScanTileSizeDoesNotChangeResults() {
        let sequence = SSMChunkedAffineFixture.makeSequence(tokenCount: 11, keyDimension: 6, valueDimension: 5)
        let initialState = SSMChunkedAffineFixture.makeState(keyDimension: 6, valueDimension: 5)
        let sequential = SSMChunkedAffineReference.runSequential(sequence: sequence, initialState: initialState)

        for tileSize in [1, 2, 4, 8] {
            let chunked = SSMChunkedAffineReference.runChunked(
                sequence: sequence,
                initialState: initialState,
                tileSize: tileSize
            )
            #expect(maxDifference(sequential.outputs, chunked.outputs) <= 0.000_001)
            #expect(maxDifference(sequential.finalState, chunked.finalState) <= 0.000_001)
        }
    }

    @Test("factorized affine tile composition matches sequential recurrence")
    func factorizedAffineTileCompositionMatchesSequentialRecurrence() {
        let sequence = SSMChunkedAffineFixture.makeSequence(tokenCount: 8, keyDimension: 5, valueDimension: 4)
        let initialState = SSMChunkedAffineFixture.makeState(keyDimension: 5, valueDimension: 4)
        let sequential = SSMChunkedAffineReference.runSequential(sequence: sequence, initialState: initialState)
        let factorized = SSMChunkedAffineReference.runFactorizedAffineScan(
            sequence: sequence,
            initialState: initialState,
            tileSize: 4
        )

        #expect(maxDifference(sequential.outputs, factorized.outputs) <= 0.000_001)
        #expect(maxDifference(sequential.finalState, factorized.finalState) <= 0.000_001)
    }

    @Test("synthetic Metal chunked affine scan matches CPU reference")
    func syntheticMetalChunkedAffineScanMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let sequence = SSMChunkedAffineFixture.makeSequence(tokenCount: 9, keyDimension: 5, valueDimension: 7)
        let initialState = SSMChunkedAffineFixture.makeState(keyDimension: 5, valueDimension: 7)
        let cpu = SSMChunkedAffineReference.runChunked(
            sequence: sequence,
            initialState: initialState,
            tileSize: 4
        )

        let metal = try SSMChunkedAffineMetalHarness(device: device).run(
            sequence: sequence,
            initialState: initialState,
            tileSize: 4
        )

        #expect(maxDifference(cpu.outputs, metal.outputs) <= 0.000_001)
        #expect(maxDifference(cpu.finalState, metal.finalState) <= 0.000_001)
    }
}

private struct SSMChunkedAffineToken {
    let query: [Float]
    let key: [Float]
    let value: [Float]
    let decay: Float
    let beta: Float
}

private struct SSMChunkedAffineResult {
    let outputs: [[Float]]
    let finalState: [[Float]]
}

private enum SSMChunkedAffineReference {
    static func runSequential(
        sequence: [SSMChunkedAffineToken],
        initialState: [[Float]]
    ) -> SSMChunkedAffineResult {
        var state = initialState
        var outputs: [[Float]] = []
        for token in sequence {
            outputs.append(apply(token: token, state: &state))
        }
        return SSMChunkedAffineResult(outputs: outputs, finalState: state)
    }

    static func runChunked(
        sequence: [SSMChunkedAffineToken],
        initialState: [[Float]],
        tileSize: Int
    ) -> SSMChunkedAffineResult {
        var state = initialState
        var outputs = Array(
            repeating: Array(repeating: Float(0), count: initialState.first?.count ?? 0),
            count: sequence.count
        )
        var tileStart = 0
        while tileStart < sequence.count {
            let tileEnd = min(sequence.count, tileStart + max(1, tileSize))
            applyTile(
                sequence: sequence,
                tileStart: tileStart,
                tileEnd: tileEnd,
                state: &state,
                outputs: &outputs
            )
            tileStart = tileEnd
        }
        return SSMChunkedAffineResult(outputs: outputs, finalState: state)
    }

    static func runFactorizedAffineScan(
        sequence: [SSMChunkedAffineToken],
        initialState: [[Float]],
        tileSize: Int
    ) -> SSMChunkedAffineResult {
        var state = initialState
        var outputs = Array(
            repeating: Array(repeating: Float(0), count: initialState.first?.count ?? 0),
            count: sequence.count
        )
        var tileStart = 0
        while tileStart < sequence.count {
            let tileEnd = min(sequence.count, tileStart + max(1, tileSize))
            applyFactorizedTile(
                sequence: sequence,
                tileStart: tileStart,
                tileEnd: tileEnd,
                state: &state,
                outputs: &outputs
            )
            tileStart = tileEnd
        }
        return SSMChunkedAffineResult(outputs: outputs, finalState: state)
    }

    private static func applyTile(
        sequence: [SSMChunkedAffineToken],
        tileStart: Int,
        tileEnd: Int,
        state: inout [[Float]],
        outputs: inout [[Float]]
    ) {
        guard let valueDimension = state.first?.count else { return }
        let keyDimension = state.count
        for valueIndex in 0..<valueDimension {
            var stateColumn = (0..<keyDimension).map { state[$0][valueIndex] }
            for tokenIndex in tileStart..<tileEnd {
                let token = sequence[tokenIndex]
                var kvmemRaw = Float(0)
                var decayedColumn = Array(repeating: Float(0), count: keyDimension)
                for keyIndex in 0..<keyDimension {
                    let decayed = stateColumn[keyIndex] * token.decay
                    decayedColumn[keyIndex] = decayed
                    kvmemRaw += decayed * token.key[keyIndex]
                }
                let delta = token.beta * (token.value[valueIndex] - kvmemRaw)
                var output = Float(0)
                for keyIndex in 0..<keyDimension {
                    let nextState = decayedColumn[keyIndex] + token.key[keyIndex] * delta
                    stateColumn[keyIndex] = nextState
                    output += nextState * token.query[keyIndex]
                }
                outputs[tokenIndex][valueIndex] = output
            }
            for keyIndex in 0..<keyDimension {
                state[keyIndex][valueIndex] = stateColumn[keyIndex]
            }
        }
    }

    private static func applyFactorizedTile(
        sequence: [SSMChunkedAffineToken],
        tileStart: Int,
        tileEnd: Int,
        state: inout [[Float]],
        outputs: inout [[Float]]
    ) {
        guard let valueDimension = state.first?.count else { return }
        let keyDimension = state.count
        for valueIndex in 0..<valueDimension {
            let initialColumn = (0..<keyDimension).map { state[$0][valueIndex] }
            var transform = identityMatrix(dimension: keyDimension)
            var bias = Array(repeating: Float(0), count: keyDimension)
            for tokenIndex in tileStart..<tileEnd {
                let token = sequence[tokenIndex]
                let tokenTransform = affineTransform(token: token)
                let tokenBias = affineBias(token: token, valueIndex: valueIndex)
                transform = multiply(tokenTransform, transform)
                bias = add(multiply(tokenTransform, bias), tokenBias)
                let stateAtToken = add(multiply(transform, initialColumn), bias)
                outputs[tokenIndex][valueIndex] = dot(token.query, stateAtToken)
            }
            let finalColumn = add(multiply(transform, initialColumn), bias)
            for keyIndex in 0..<keyDimension {
                state[keyIndex][valueIndex] = finalColumn[keyIndex]
            }
        }
    }

    private static func apply(token: SSMChunkedAffineToken, state: inout [[Float]]) -> [Float] {
        let keyDimension = state.count
        let valueDimension = state.first?.count ?? 0
        var output = Array(repeating: Float(0), count: valueDimension)
        for valueIndex in 0..<valueDimension {
            var kvmemRaw = Float(0)
            for keyIndex in 0..<keyDimension {
                kvmemRaw += state[keyIndex][valueIndex] * token.decay * token.key[keyIndex]
            }
            let delta = token.beta * (token.value[valueIndex] - kvmemRaw)
            for keyIndex in 0..<keyDimension {
                let nextState = state[keyIndex][valueIndex] * token.decay + token.key[keyIndex] * delta
                state[keyIndex][valueIndex] = nextState
                output[valueIndex] += nextState * token.query[keyIndex]
            }
        }
        return output
    }

    private static func affineTransform(token: SSMChunkedAffineToken) -> [[Float]] {
        let dimension = token.key.count
        return (0..<dimension).map { row in
            (0..<dimension).map { column in
                let identity = row == column ? Float(1) : Float(0)
                return token.decay * (identity - token.beta * token.key[row] * token.key[column])
            }
        }
    }

    private static func affineBias(token: SSMChunkedAffineToken, valueIndex: Int) -> [Float] {
        token.key.map { token.beta * token.value[valueIndex] * $0 }
    }

    private static func identityMatrix(dimension: Int) -> [[Float]] {
        (0..<dimension).map { row in
            (0..<dimension).map { column in row == column ? Float(1) : Float(0) }
        }
    }

    private static func multiply(_ matrix: [[Float]], _ vector: [Float]) -> [Float] {
        matrix.map { row in dot(row, vector) }
    }

    private static func multiply(_ lhs: [[Float]], _ rhs: [[Float]]) -> [[Float]] {
        guard let rhsColumnCount = rhs.first?.count else { return [] }
        return lhs.map { lhsRow in
            (0..<rhsColumnCount).map { column in
                var sum = Float(0)
                for index in lhsRow.indices {
                    sum += lhsRow[index] * rhs[index][column]
                }
                return sum
            }
        }
    }

    private static func add(_ lhs: [Float], _ rhs: [Float]) -> [Float] {
        zip(lhs, rhs).map(+)
    }

    private static func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float(0)) { $0 + $1.0 * $1.1 }
    }
}

private enum SSMChunkedAffineFixture {
    static func makeSequence(tokenCount: Int, keyDimension: Int, valueDimension: Int) -> [SSMChunkedAffineToken] {
        (0..<tokenCount).map { tokenIndex in
            SSMChunkedAffineToken(
                query: makeVector(count: keyDimension, seed: tokenIndex, scale: 0.011),
                key: makeVector(count: keyDimension, seed: tokenIndex + 17, scale: 0.009),
                value: makeVector(count: valueDimension, seed: tokenIndex + 31, scale: 0.013),
                decay: 0.82 + Float(tokenIndex % 5) * 0.025,
                beta: 0.18 + Float(tokenIndex % 3) * 0.04
            )
        }
    }

    static func makeState(keyDimension: Int, valueDimension: Int) -> [[Float]] {
        (0..<keyDimension).map { keyIndex in
            makeVector(count: valueDimension, seed: keyIndex + 101, scale: 0.007)
        }
    }

    private static func makeVector(count: Int, seed: Int, scale: Float) -> [Float] {
        (0..<count).map { index in
            let value = ((seed + 1) * (index + 3) * 37) % 23
            return (Float(value) - 11.0) * scale
        }
    }
}

private struct SSMChunkedAffineMetalHarness {
    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let commandQueue: MTLCommandQueue

    init(device: MTLDevice) throws {
        self.device = device
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let function = library.makeFunction(name: "chunked_affine_scan_reference") else {
            throw SSMChunkedAffineMetalError.missingFunction
        }
        self.pipeline = try device.makeComputePipelineState(function: function)
        guard let commandQueue = device.makeCommandQueue() else {
            throw SSMChunkedAffineMetalError.missingCommandQueue
        }
        self.commandQueue = commandQueue
    }

    func run(
        sequence: [SSMChunkedAffineToken],
        initialState: [[Float]],
        tileSize: Int
    ) throws -> SSMChunkedAffineResult {
        let tokenCount = sequence.count
        let keyDimension = initialState.count
        let valueDimension = initialState.first?.count ?? 0
        let query = sequence.flatMap(\.query)
        let key = sequence.flatMap(\.key)
        let value = sequence.flatMap(\.value)
        let decay = sequence.map(\.decay)
        let beta = sequence.map(\.beta)
        let state = flatten(initialState)
        let outputElementCount = tokenCount * valueDimension
        let stateElementCount = keyDimension * valueDimension

        let queryBuffer = try makeBuffer(query)
        let keyBuffer = try makeBuffer(key)
        let valueBuffer = try makeBuffer(value)
        let decayBuffer = try makeBuffer(decay)
        let betaBuffer = try makeBuffer(beta)
        let stateBuffer = try makeBuffer(state)
        let outputBuffer = try makeEmptyFloatBuffer(count: outputElementCount)
        let finalStateBuffer = try makeEmptyFloatBuffer(count: stateElementCount)
        let tokenCountBuffer = try makeUInt32Buffer(tokenCount)
        let keyDimensionBuffer = try makeUInt32Buffer(keyDimension)
        let valueDimensionBuffer = try makeUInt32Buffer(valueDimension)
        let tileSizeBuffer = try makeUInt32Buffer(tileSize)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SSMChunkedAffineMetalError.missingCommandEncoder
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(keyBuffer, offset: 0, index: 1)
        encoder.setBuffer(valueBuffer, offset: 0, index: 2)
        encoder.setBuffer(decayBuffer, offset: 0, index: 3)
        encoder.setBuffer(betaBuffer, offset: 0, index: 4)
        encoder.setBuffer(stateBuffer, offset: 0, index: 5)
        encoder.setBuffer(outputBuffer, offset: 0, index: 6)
        encoder.setBuffer(finalStateBuffer, offset: 0, index: 7)
        encoder.setBuffer(tokenCountBuffer, offset: 0, index: 8)
        encoder.setBuffer(keyDimensionBuffer, offset: 0, index: 9)
        encoder.setBuffer(valueDimensionBuffer, offset: 0, index: 10)
        encoder.setBuffer(tileSizeBuffer, offset: 0, index: 11)
        encoder.dispatchThreads(
            MTLSize(width: valueDimension, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(max(valueDimension, 1), pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let outputs = readFloatBuffer(outputBuffer, count: outputElementCount).chunked(rowWidth: valueDimension)
        let finalState = readFloatBuffer(finalStateBuffer, count: stateElementCount).chunked(rowWidth: valueDimension)
        return SSMChunkedAffineResult(outputs: outputs, finalState: finalState)
    }

    private func makeBuffer(_ values: [Float]) throws -> MTLBuffer {
        let byteCount = max(values.count * MemoryLayout<Float>.stride, MemoryLayout<Float>.stride)
        guard let buffer = device.makeBuffer(bytes: values, length: byteCount, options: .storageModeShared) else {
            throw SSMChunkedAffineMetalError.bufferAllocationFailed
        }
        return buffer
    }

    private func makeEmptyFloatBuffer(count: Int) throws -> MTLBuffer {
        let byteCount = max(count * MemoryLayout<Float>.stride, MemoryLayout<Float>.stride)
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw SSMChunkedAffineMetalError.bufferAllocationFailed
        }
        return buffer
    }

    private func makeUInt32Buffer(_ value: Int) throws -> MTLBuffer {
        var rawValue = UInt32(value)
        guard let buffer = device.makeBuffer(
            bytes: &rawValue,
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw SSMChunkedAffineMetalError.bufferAllocationFailed
        }
        return buffer
    }

    private func readFloatBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { pointer[$0] }
    }

    private func flatten(_ matrix: [[Float]]) -> [Float] {
        matrix.flatMap { $0 }
    }

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void chunked_affine_scan_reference(
        device const float* query [[buffer(0)]],
        device const float* key [[buffer(1)]],
        device const float* value [[buffer(2)]],
        device const float* decay [[buffer(3)]],
        device const float* beta [[buffer(4)]],
        device const float* initialState [[buffer(5)]],
        device float* output [[buffer(6)]],
        device float* finalState [[buffer(7)]],
        constant uint& tokenCount [[buffer(8)]],
        constant uint& keyDimension [[buffer(9)]],
        constant uint& valueDimension [[buffer(10)]],
        constant uint& tileSize [[buffer(11)]],
        uint valueIndex [[thread_position_in_grid]]
    ) {
        if (valueIndex >= valueDimension || keyDimension > 16) {
            return;
        }
        float stateColumn[16];
        for (uint keyIndex = 0; keyIndex < keyDimension; ++keyIndex) {
            stateColumn[keyIndex] = initialState[keyIndex * valueDimension + valueIndex];
        }

        uint tileStart = 0;
        const uint safeTileSize = max(tileSize, 1u);
        while (tileStart < tokenCount) {
            const uint tileEnd = min(tokenCount, tileStart + safeTileSize);
            for (uint tokenIndex = tileStart; tokenIndex < tileEnd; ++tokenIndex) {
                const float tokenDecay = decay[tokenIndex];
                float decayedColumn[16];
                float kvmemRaw = 0.0f;
                for (uint keyIndex = 0; keyIndex < keyDimension; ++keyIndex) {
                    const float decayed = stateColumn[keyIndex] * tokenDecay;
                    decayedColumn[keyIndex] = decayed;
                    kvmemRaw += decayed * key[tokenIndex * keyDimension + keyIndex];
                }
                const float delta = beta[tokenIndex]
                    * (value[tokenIndex * valueDimension + valueIndex] - kvmemRaw);
                float dot = 0.0f;
                for (uint keyIndex = 0; keyIndex < keyDimension; ++keyIndex) {
                    const float nextState = decayedColumn[keyIndex]
                        + key[tokenIndex * keyDimension + keyIndex] * delta;
                    stateColumn[keyIndex] = nextState;
                    dot += nextState * query[tokenIndex * keyDimension + keyIndex];
                }
                output[tokenIndex * valueDimension + valueIndex] = dot;
            }
            tileStart = tileEnd;
        }

        for (uint keyIndex = 0; keyIndex < keyDimension; ++keyIndex) {
            finalState[keyIndex * valueDimension + valueIndex] = stateColumn[keyIndex];
        }
    }
    """
}

private enum SSMChunkedAffineMetalError: Error {
    case bufferAllocationFailed
    case missingCommandEncoder
    case missingCommandQueue
    case missingFunction
}

private func maxDifference(_ lhs: [[Float]], _ rhs: [[Float]]) -> Float {
    guard lhs.count == rhs.count else { return .infinity }
    var maxError = Float(0)
    for rowIndex in lhs.indices {
        guard lhs[rowIndex].count == rhs[rowIndex].count else { return .infinity }
        for columnIndex in lhs[rowIndex].indices {
            maxError = max(maxError, abs(lhs[rowIndex][columnIndex] - rhs[rowIndex][columnIndex]))
        }
    }
    return maxError
}

private extension Array {
    func chunked(rowWidth: Int) -> [[Element]] {
        guard rowWidth > 0 else { return [] }
        var rows: [[Element]] = []
        var index = 0
        while index < count {
            rows.append(Array(self[index..<Swift.min(index + rowWidth, count)]))
            index += rowWidth
        }
        return rows
    }
}
