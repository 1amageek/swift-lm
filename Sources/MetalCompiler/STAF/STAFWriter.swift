import Foundation

struct STAFWriter: Sendable {

    private let payloadConverter: STAFPayloadConverter

    init(payloadConverter: STAFPayloadConverter = STAFPayloadConverter()) {
        self.payloadConverter = payloadConverter
    }

    func write(plan: STAFConversionPlan, outputURL: URL, metadata: STAFFileMetadata) throws {
        let entries = plan.entries
        let sectionCount = entries.count
        let sectionTableOffset = STAF.headerSize
        let sectionTableSize = sectionCount * STAF.sectionEntrySize

        let metadataEntries = metadata.values.sorted { lhs, rhs in
            lhs.key < rhs.key
        }
        let metadataTableOffset = sectionTableOffset + sectionTableSize
        let metadataTableSize = metadataEntries.count * STAF.metadataEntrySize

        var stringTableData = Data()
        var stringOffsets: [String: Int] = [:]
        var nameOffsets: [Int] = []
        var metadataTableEntries: [STAFMetadataTableEntry] = []

        func appendString(_ string: String) -> Int {
            if let existingOffset = stringOffsets[string] {
                return existingOffset
            }
            let offset = stringTableData.count
            stringTableData.append(contentsOf: string.utf8)
            stringTableData.append(0)
            stringOffsets[string] = offset
            return offset
        }

        for entry in entries {
            nameOffsets.append(appendString(entry.name))
        }

        for (key, value) in metadataEntries {
            let keyOffset = appendString(key)
            metadataTableEntries.append(
                buildMetadataTableEntry(
                    key: key,
                    keyOffset: keyOffset,
                    value: value,
                    appendString: appendString
                )
            )
        }

        let stringTableOffset = metadataTableOffset + metadataTableSize
        let metadataEnd = stringTableOffset + stringTableData.count
        let payloadStart = alignUp(metadataEnd, to: STAF.payloadAlignment)

        var payloadOffsets: [UInt64] = []
        var payloadSizes: [UInt64] = []
        var currentOffset = payloadStart

        for entry in entries {
            let tensorSize = computePayloadSize(entry: entry)
            let alignedOffset = alignUp(currentOffset, to: STAF.tensorAlignment)
            payloadOffsets.append(UInt64(alignedOffset))
            payloadSizes.append(UInt64(tensorSize))
            currentOffset = alignedOffset + tensorSize
        }

        var fileData = Data()
        fileData.append(buildHeaderData(
            metadataEntryCount: metadataTableEntries.count,
            metadataTableOffset: metadataTableOffset,
            sectionCount: sectionCount,
            sectionTableOffset: sectionTableOffset,
            stringTableOffset: stringTableOffset,
            stringTableCount: stringTableData.count
        ))

        for (index, entry) in entries.enumerated() {
            fileData.append(
                buildSectionEntryData(
                    entry: entry,
                    nameOffset: nameOffsets[index],
                    payloadOffset: payloadOffsets[index],
                    payloadSize: payloadSizes[index]
                )
            )
        }

        for metadataEntry in metadataTableEntries {
            var entryData = Data(count: STAF.metadataEntrySize)
            entryData.withUnsafeMutableBytes { buf in
                let base = buf.baseAddress!
                base.storeBytes(of: metadataEntry.keyOffset, toByteOffset: 0, as: UInt32.self)
                base.storeBytes(of: metadataEntry.keyLength, toByteOffset: 4, as: UInt32.self)
                base.storeBytes(of: metadataEntry.valueType.rawValue, toByteOffset: 8, as: UInt8.self)
                base.storeBytes(of: metadataEntry.payload0, toByteOffset: 12, as: UInt64.self)
                base.storeBytes(of: metadataEntry.payload1, toByteOffset: 20, as: UInt64.self)
            }
            fileData.append(entryData)
        }

        fileData.append(stringTableData)

        let paddingNeeded = payloadStart - fileData.count
        if paddingNeeded > 0 {
            fileData.append(Data(count: paddingNeeded))
        }

        for (index, entry) in entries.enumerated() {
            let targetOffset = Int(payloadOffsets[index])
            if fileData.count < targetOffset {
                fileData.append(Data(count: targetOffset - fileData.count))
            }
            fileData.append(try payloadConverter.convertPayload(for: entry))
        }

        try fileData.write(to: outputURL)
    }

    private func buildHeaderData(
        metadataEntryCount: Int,
        metadataTableOffset: Int,
        sectionCount: Int,
        sectionTableOffset: Int,
        stringTableOffset: Int,
        stringTableCount: Int
    ) -> Data {
        let packedHeaderSize = 64
        var headerData = Data(count: packedHeaderSize)
        headerData.withUnsafeMutableBytes { buf in
            let base = buf.baseAddress!
            base.storeBytes(of: STAF.magic, toByteOffset: 0, as: UInt32.self)
            base.storeBytes(of: STAF.currentFormatVersion, toByteOffset: 4, as: UInt32.self)
            base.storeBytes(of: UInt32(metadataEntryCount), toByteOffset: 8, as: UInt32.self)
            base.storeBytes(of: UInt32(metadataTableOffset), toByteOffset: 12, as: UInt32.self)
            base.storeBytes(of: UInt32(sectionCount), toByteOffset: 40, as: UInt32.self)
            base.storeBytes(of: UInt32(sectionTableOffset), toByteOffset: 44, as: UInt32.self)
            base.storeBytes(of: UInt32(stringTableOffset), toByteOffset: 48, as: UInt32.self)
            base.storeBytes(of: UInt32(stringTableCount), toByteOffset: 52, as: UInt32.self)
        }
        return headerData
    }

    private func buildSectionEntryData(
        entry: STAFConversionEntry,
        nameOffset: Int,
        payloadOffset: UInt64,
        payloadSize: UInt64
    ) -> Data {
        let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier)
        let shapeArray = entry.info.shape + Array(
            repeating: 0,
            count: STAF.maximumDimensions - min(entry.info.shape.count, STAF.maximumDimensions)
        )

        let packedEntrySize = 128
        var entryData = Data(count: packedEntrySize)
        entryData.withUnsafeMutableBytes { buf in
            let base = buf.baseAddress!
            base.storeBytes(of: UInt32(nameOffset), toByteOffset: 0, as: UInt32.self)
            base.storeBytes(of: UInt32(entry.name.utf8.count), toByteOffset: 4, as: UInt32.self)
            base.storeBytes(of: entry.schemeIdentifier.rawValue, toByteOffset: 8, as: UInt8.self)
            base.storeBytes(of: entry.semanticRole.rawValue, toByteOffset: 9, as: UInt8.self)
            base.storeBytes(of: entry.originalDType.rawValue, toByteOffset: 10, as: UInt8.self)
            base.storeBytes(of: UInt8(min(entry.info.shape.count, 8)), toByteOffset: 11, as: UInt8.self)
            for dimension in 0..<8 {
                base.storeBytes(of: UInt32(shapeArray[dimension]), toByteOffset: 12 + dimension * 4, as: UInt32.self)
            }
            base.storeBytes(of: payloadOffset, toByteOffset: 44, as: UInt64.self)
            base.storeBytes(of: payloadSize, toByteOffset: 52, as: UInt64.self)
            base.storeBytes(of: UInt32(STAF.tensorAlignment), toByteOffset: 60, as: UInt32.self)
            base.storeBytes(of: UInt32(format?.weightsPerBlock ?? 1), toByteOffset: 64, as: UInt32.self)
            base.storeBytes(of: UInt32(format?.groupSize ?? 1), toByteOffset: 68, as: UInt32.self)
            base.storeBytes(of: UInt32(0), toByteOffset: 72, as: UInt32.self)
            base.storeBytes(of: UInt32(entry.shardIndex), toByteOffset: 76, as: UInt32.self)
        }
        return entryData
    }

    private func buildMetadataTableEntry(
        key: String,
        keyOffset: Int,
        value: STAFMetadataValue,
        appendString: (String) -> Int
    ) -> STAFMetadataTableEntry {
        switch value {
        case .bool(let boolValue):
            return STAFMetadataTableEntry(
                keyOffset: UInt32(keyOffset),
                keyLength: UInt32(key.utf8.count),
                valueType: .bool,
                payload0: boolValue ? 1 : 0,
                payload1: 0
            )
        case .uint32(let uint32Value):
            return STAFMetadataTableEntry(
                keyOffset: UInt32(keyOffset),
                keyLength: UInt32(key.utf8.count),
                valueType: .uint32,
                payload0: UInt64(uint32Value),
                payload1: 0
            )
        case .uint64(let uint64Value):
            return STAFMetadataTableEntry(
                keyOffset: UInt32(keyOffset),
                keyLength: UInt32(key.utf8.count),
                valueType: .uint64,
                payload0: uint64Value,
                payload1: 0
            )
        case .float32(let float32Value):
            return STAFMetadataTableEntry(
                keyOffset: UInt32(keyOffset),
                keyLength: UInt32(key.utf8.count),
                valueType: .float32,
                payload0: UInt64(float32Value.bitPattern),
                payload1: 0
            )
        case .float64(let float64Value):
            return STAFMetadataTableEntry(
                keyOffset: UInt32(keyOffset),
                keyLength: UInt32(key.utf8.count),
                valueType: .float64,
                payload0: float64Value.bitPattern,
                payload1: 0
            )
        case .string(let stringValue):
            let valueOffset = appendString(stringValue)
            return STAFMetadataTableEntry(
                keyOffset: UInt32(keyOffset),
                keyLength: UInt32(key.utf8.count),
                valueType: .string,
                payload0: UInt64(valueOffset),
                payload1: UInt64(stringValue.utf8.count)
            )
        }
    }

    private func computePayloadSize(entry: STAFConversionEntry) -> Int {
        switch entry.schemeIdentifier {
        case .fp16RowMajor, .passthrough, .bf16RowMajor:
            return entry.info.shape.reduce(1, *) * 2
        case .fp32RowMajor:
            return entry.info.shape.reduce(1, *) * 4
        default:
            guard let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier) else {
                return entry.info.shape.reduce(1, *) * 2
            }
            let outputDimension = entry.info.shape[0]
            let packedDimension = entry.info.shape.count >= 2 ? entry.info.shape[1] : 1
            // Input dimension formula must match `STAFPayloadConverter.repackMLXQuantized`.
            // Non-aligned bit widths (3/5/6) require `(packedDim × 32) / bits` rather
            // than `packedDim × (32 / bits)`; the latter truncates and produces
            // under-sized payloads (or zero for bits that don't divide 32).
            let inputDimension = packedDimension * 32 / format.bits
            let blocksPerRow = inputDimension / format.groupSize
            let totalBlocks = outputDimension * blocksPerRow
            return totalBlocks * format.bytesPerBlock
        }
    }

    private func alignUp(_ value: Int, to alignment: Int) -> Int {
        let remainder = value % alignment
        return remainder == 0 ? value : value + (alignment - remainder)
    }
}
