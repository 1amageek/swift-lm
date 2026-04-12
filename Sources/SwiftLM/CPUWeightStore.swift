import Foundation
import MetalCompiler

final class CPUWeightStore {
    struct DenseTensor {
        let values: [Float]
        let shape: [Int]
    }

    private let stafWeights: STAFWeightStore?
    private let denseTensors: [String: DenseTensor]
    private var floatTensorCache: [String: [Float]] = [:]

    init(weights: STAFWeightStore) {
        self.stafWeights = weights
        self.denseTensors = [:]
    }

    init(denseTensors: [String: DenseTensor]) {
        self.stafWeights = nil
        self.denseTensors = denseTensors
    }

    func floatTensor(named name: String) throws -> [Float] {
        if let dense = denseTensors[name] {
            return dense.values
        }
        if let cached = floatTensorCache[name] {
            return cached
        }
        guard let stafWeights,
              let entry = stafWeights.entries[name] else {
            throw ModelBundleLoaderError.invalidConfig("Missing tensor: \(name)")
        }

        let decoded = try decode(entry: entry, from: stafWeights)
        floatTensorCache[name] = decoded
        return decoded
    }

    func optionalFloatTensor(named name: String?) throws -> [Float]? {
        guard let name else { return nil }
        if denseTensors[name] != nil {
            return try floatTensor(named: name)
        }
        guard let stafWeights,
              stafWeights.entries[name] != nil else {
            return nil
        }
        return try floatTensor(named: name)
    }

    func shape(named name: String) throws -> [Int] {
        if let dense = denseTensors[name] {
            return dense.shape
        }
        guard let stafWeights,
              let entry = stafWeights.entries[name] else {
            throw ModelBundleLoaderError.invalidConfig("Missing tensor shape: \(name)")
        }
        return try logicalShape(for: entry)
    }

    private func logicalShape(for entry: STAFTensorEntry) throws -> [Int] {
        guard let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier) else {
            throw ModelBundleLoaderError.invalidConfig(
                "Unsupported tensor quantization scheme: \(entry.name)"
            )
        }
        switch entry.schemeIdentifier {
        case .fp16RowMajor, .bf16RowMajor, .fp32RowMajor, .passthrough:
            return entry.shape
        default:
            guard entry.shape.count >= 2 else {
                throw ModelBundleLoaderError.invalidConfig(
                    "Quantized tensor requires rank >= 2: \(entry.name)"
                )
            }
            let elementsPerPackedWord = 32 / format.bits
            var shape = entry.shape
            shape[1] *= elementsPerPackedWord
            return shape
        }
    }

    private func decode(
        entry: STAFTensorEntry,
        from weights: STAFWeightStore
    ) throws -> [Float] {
        let basePointer = weights.buffer.contents().advanced(by: entry.bufferOffset)
        switch entry.schemeIdentifier {
        case .fp16RowMajor, .passthrough:
            let count = entry.shape.reduce(1, *)
            let values = UnsafeRawPointer(basePointer).bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Float(Float16(bitPattern: values[$0])) }
        case .bf16RowMajor:
            let count = entry.shape.reduce(1, *)
            let values = UnsafeRawPointer(basePointer).bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Float(bitPattern: UInt32(values[$0]) << 16) }
        case .fp32RowMajor:
            let count = entry.shape.reduce(1, *)
            let values = UnsafeRawPointer(basePointer).bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: values, count: count))
        default:
            return try decodeQuantized(entry: entry, basePointer: basePointer)
        }
    }

    private func decodeQuantized(
        entry: STAFTensorEntry,
        basePointer: UnsafeMutableRawPointer
    ) throws -> [Float] {
        guard let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier) else {
            throw ModelBundleLoaderError.invalidConfig(
                "Unsupported tensor quantization scheme: \(entry.name)"
            )
        }
        guard entry.shape.count >= 2 else {
            throw ModelBundleLoaderError.invalidConfig(
                "Quantized tensor requires rank >= 2: \(entry.name)"
            )
        }

        let rows = entry.shape[0]
        let packedColumns = entry.shape[1]
        let logicalColumns = packedColumns * (32 / format.bits)
        guard logicalColumns % format.groupSize == 0 else {
            throw ModelBundleLoaderError.invalidConfig(
                "Quantized tensor logical width is not divisible by group size: \(entry.name)"
            )
        }

        let blocksPerRow = logicalColumns / format.groupSize
        let expectedPayloadSize = rows * blocksPerRow * format.bytesPerBlock
        guard expectedPayloadSize == entry.payloadSize else {
            throw ModelBundleLoaderError.invalidConfig(
                "Quantized tensor payload size mismatch: \(entry.name)"
            )
        }

        var decoded = [Float](repeating: 0, count: rows * logicalColumns)
        for row in 0..<rows {
            for block in 0..<blocksPerRow {
                let blockPointer = basePointer.advanced(
                    by: (row * blocksPerRow + block) * format.bytesPerBlock
                )
                let scaleBits = UnsafeRawPointer(blockPointer).loadUnaligned(as: UInt16.self)
                let zeroBits = UnsafeRawPointer(blockPointer.advanced(by: 2)).loadUnaligned(as: UInt16.self)
                let scale = Float(Float16(bitPattern: scaleBits))
                let zero = Float(Float16(bitPattern: zeroBits))
                let quantizedBytes = UnsafeRawPointer(blockPointer.advanced(by: 4))
                    .assumingMemoryBound(to: UInt8.self)

                for element in 0..<format.groupSize {
                    let quantizedValue: UInt8
                    switch format.bits {
                    case 4:
                        let byte = quantizedBytes[element / 2]
                        quantizedValue = element.isMultiple(of: 2) ? (byte & 0x0F) : (byte >> 4)
                    case 8:
                        quantizedValue = quantizedBytes[element]
                    default:
                        throw ModelBundleLoaderError.invalidConfig(
                            "Unsupported CPU dequantization bit width: \(format.bits)"
                        )
                    }

                    let column = block * format.groupSize + element
                    decoded[row * logicalColumns + column] = scale * Float(quantizedValue) + zero
                }
            }
        }
        return decoded
    }
}
