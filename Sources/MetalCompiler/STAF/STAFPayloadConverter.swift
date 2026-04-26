import Foundation

struct STAFPayloadConverter: Sendable {

    func convertPayload(for entry: STAFConversionEntry) throws -> Data {
        var tensorData = try loadRawTensorData(entry: entry)

        // MLX VLM bundles bake the `+1.0` offset into Gemma-style layernorm
        // weights for `input_layernorm` and `post_attention_layernorm`. Because
        // swift-lm's RMSNorm kernel applies the offset itself via
        // `weightBias = 1` (see Reduction.swift), passing the MLX storage as-is
        // doubles the offset — every layer's norm scale becomes `1 + (1 + δ)`
        // instead of `1 + δ`, which collapses the residual stream and produces
        // garbage tokens. Subtract 1.0 here so downstream consumers see the
        // canonical raw delta convention shared by HuggingFace bundles.
        //
        // The SSM `linear_attn.norm.weight` is exempt: both HF and MLX store it
        // with the standard (`weightBias = 0`) convention.
        if needsMLXLayerNormBiasCorrection(entry: entry) {
            tensorData = try subtractOneFromLayerNormWeights(
                tensorData, dtype: entry.info.dtype, name: entry.name)
        }

        switch entry.schemeIdentifier {
        case .fp16RowMajor, .passthrough:
            return try convertDensePayload(entry: entry, tensorData: tensorData)
        case .bf16RowMajor:
            return tensorData
        case .fp32RowMajor:
            return try convertToFloat32(entry: entry, tensorData: tensorData)
        case .q2Group16ScaleF16, .q2Group32ScaleF16,
             .q3Group16ScaleF16, .q3Group32ScaleF16, .q3Group64ScaleF16,
             .q4Group64ScaleF16, .q4Group128ScaleF16, .q4Group128ScaleF16Zero,
             .q5Group32ScaleF16, .q5Group64ScaleF16,
             .q6Group16ScaleF16, .q6Group32ScaleF16,
             .q8Group32ScaleF16, .q8Group64ScaleF16, .q8Group128ScaleF16:
            return try repackMLXQuantized(entry: entry, weightData: tensorData)
        default:
            return tensorData
        }
    }

    /// Detect MLX VLM layernorm tensors whose stored values bake in the
    /// `+1.0` Gemma-style offset that swift-lm's RMSNorm kernel adds itself.
    ///
    /// Returns `true` only for `input_layernorm.weight` and
    /// `post_attention_layernorm.weight` whose source name carries the
    /// `language_model.model.` prefix (the MLX VLM marker). The SSM
    /// `linear_attn.norm.weight` is excluded — both HF and MLX store it with
    /// the standard (no-offset) convention.
    private func needsMLXLayerNormBiasCorrection(entry: STAFConversionEntry) -> Bool {
        guard entry.sourceName.hasPrefix("language_model.model.layers.") else {
            return false
        }
        return entry.sourceName.hasSuffix(".input_layernorm.weight")
            || entry.sourceName.hasSuffix(".post_attention_layernorm.weight")
    }

    private func subtractOneFromLayerNormWeights(
        _ data: Data, dtype: SafetensorsDType, name: String
    ) throws -> Data {
        switch dtype {
        case .bfloat16:
            let elementCount = data.count / 2
            var output = Data(count: data.count)
            data.withUnsafeBytes { source in
                output.withUnsafeMutableBytes { destination in
                    let bf16In = source.bindMemory(to: UInt16.self)
                    let bf16Out = destination.bindMemory(to: UInt16.self)
                    for index in 0..<elementCount {
                        let widened = UInt32(bf16In[index]) << 16
                        let asFloat = Float(bitPattern: widened) - 1.0
                        bf16Out[index] = Self.bfloat16Bits(from: asFloat)
                    }
                }
            }
            return output
        case .float16:
            let elementCount = data.count / MemoryLayout<Float16>.size
            var output = Data(count: data.count)
            data.withUnsafeBytes { source in
                output.withUnsafeMutableBytes { destination in
                    let fp16In = source.bindMemory(to: Float16.self)
                    let fp16Out = destination.bindMemory(to: Float16.self)
                    for index in 0..<elementCount {
                        fp16Out[index] = fp16In[index] - 1.0
                    }
                }
            }
            return output
        case .float32:
            let elementCount = data.count / MemoryLayout<Float>.size
            var output = Data(count: data.count)
            data.withUnsafeBytes { source in
                output.withUnsafeMutableBytes { destination in
                    let f32In = source.bindMemory(to: Float.self)
                    let f32Out = destination.bindMemory(to: Float.self)
                    for index in 0..<elementCount {
                        f32Out[index] = f32In[index] - 1.0
                    }
                }
            }
            return output
        default:
            fatalError(
                "Cannot apply MLX layernorm bias correction for tensor \(name): " +
                "unsupported source dtype \(dtype.rawValue). Expected BF16, F16, or F32.")
        }
    }

    /// Round a Float32 to BF16 with round-to-nearest-even.
    ///
    /// BF16 = upper 16 bits of Float32. Naive truncation introduces a
    /// downward bias; tiebreak using the LSB of the surviving mantissa keeps
    /// the round unbiased so the per-tensor mean is preserved within ULP.
    private static func bfloat16Bits(from value: Float) -> UInt16 {
        let bits = value.bitPattern
        let lsb = (bits >> 16) & 1
        let rounded = bits &+ UInt32(0x7FFF) &+ lsb
        return UInt16(rounded >> 16)
    }

    private func convertDensePayload(entry: STAFConversionEntry, tensorData: Data) throws -> Data {
        if entry.info.dtype != .float32 {
            return tensorData
        }

        let count = tensorData.count / MemoryLayout<Float>.size
        var output = Data(count: count * MemoryLayout<Float16>.size)
        tensorData.withUnsafeBytes { source in
            output.withUnsafeMutableBytes { destination in
                let floats = source.bindMemory(to: Float.self)
                let halfs = destination.bindMemory(to: Float16.self)
                for index in 0..<count {
                    halfs[index] = Float16(floats[index])
                }
            }
        }
        return output
    }

    /// Promote a tensor to Float32 storage regardless of source dtype.
    ///
    /// Some Metal kernels (notably the SSM/DeltaNet per-head RMS norm) bind
    /// their scale buffer with a hardcoded `device const float*` signature.
    /// When a bundle stores the same tensor as bfloat16 (MLX) or float16, the
    /// raw bytes must be widened so the kernel sees correct values. Float32
    /// sources pass through unchanged.
    private func convertToFloat32(entry: STAFConversionEntry, tensorData: Data) throws -> Data {
        switch entry.info.dtype {
        case .float32:
            return tensorData
        case .bfloat16:
            let elementCount = tensorData.count / 2
            var output = Data(count: elementCount * MemoryLayout<Float>.size)
            tensorData.withUnsafeBytes { source in
                output.withUnsafeMutableBytes { destination in
                    let bf16Words = source.bindMemory(to: UInt16.self)
                    let floats = destination.bindMemory(to: Float.self)
                    for index in 0..<elementCount {
                        let widened = UInt32(bf16Words[index]) << 16
                        floats[index] = Float(bitPattern: widened)
                    }
                }
            }
            return output
        case .float16:
            let elementCount = tensorData.count / MemoryLayout<Float16>.size
            var output = Data(count: elementCount * MemoryLayout<Float>.size)
            tensorData.withUnsafeBytes { source in
                output.withUnsafeMutableBytes { destination in
                    let halfs = source.bindMemory(to: Float16.self)
                    let floats = destination.bindMemory(to: Float.self)
                    for index in 0..<elementCount {
                        floats[index] = Float(halfs[index])
                    }
                }
            }
            return output
        default:
            // Float32 promotion is constrained to F16/BF16/F32 sources. Any
            // other dtype on a tensor declared as `.fp32RowMajor` indicates a
            // malformed conversion plan — fail loudly rather than silently
            // copying mismatched bytes into a buffer the kernel binds as f32.
            fatalError(
                "Unsupported source dtype for f32 promotion: \(entry.info.dtype.rawValue) " +
                "(tensor: \(entry.name)). Expected F16, BF16, or F32.")
        }
    }

    private func loadRawTensorData(entry: STAFConversionEntry) throws -> Data {
        let fileHandle = try FileHandle(forReadingFrom: entry.shardURL)
        defer { fileHandle.closeFile() }

        guard let sizeData = try fileHandle.read(upToCount: 8), sizeData.count == 8 else {
            throw STAFConversionError.readFailed(entry.name)
        }
        let headerSize = sizeData.withUnsafeBytes { Int(UInt64(littleEndian: $0.load(as: UInt64.self))) }
        let dataSectionOffset = 8 + headerSize

        try fileHandle.seek(toOffset: UInt64(dataSectionOffset + entry.info.dataOffset))
        guard let tensorData = try fileHandle.read(upToCount: entry.info.byteCount),
              tensorData.count == entry.info.byteCount else {
            throw STAFConversionError.readFailed(entry.name)
        }
        return tensorData
    }

    private func repackMLXQuantized(entry: STAFConversionEntry, weightData: Data) throws -> Data {
        // Companion tensors (`.scales`, `.biases`) are resolved against the
        // safetensors shard by their ON-DISK name, which may differ from the
        // canonicalized `entry.name` (e.g. MLX VLM → HF rewrite).
        let modulePath = String(entry.sourceName.dropLast(".weight".count))
        let (rawScalesData, scalesDType) = try loadTensorFromSafetensorsWithDType(
            name: modulePath + ".scales", shardURL: entry.shardURL)
        let (rawBiasesData, biasesDType) = try loadTensorFromSafetensorsWithDType(
            name: modulePath + ".biases", shardURL: entry.shardURL)

        // MLX stores scales/biases in the original model's dtype (BF16 for
        // bfloat16 models, F16 otherwise). The STAF Q4 block layout expects F16
        // scale/bias. Convert BF16 → F16 up-front so the packing loop below can
        // treat them uniformly as Float16.
        let scalesData = try normalizeScaleToFloat16(rawScalesData, dtype: scalesDType)
        let biasesData = try normalizeScaleToFloat16(rawBiasesData, dtype: biasesDType)

        guard let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier) else {
            throw STAFConversionError.unsupportedFormat(entry.schemeIdentifier.rawValue)
        }

        let outputDimension = entry.info.shape[0]
        let packedDimension = entry.info.shape.count >= 2 ? entry.info.shape[1] : 1
        let bytesPerRow = packedDimension * MemoryLayout<UInt32>.size
        let inputDimension = bytesPerRow * 8 / format.bits
        let blocksPerRow = inputDimension / format.groupSize
        let totalBlocks = outputDimension * blocksPerRow
        let bytesPerGroup = format.groupSize * format.bits / 8

        guard bytesPerGroup * 8 == format.groupSize * format.bits else {
            throw STAFConversionError.inconsistentQuantizationShape(
                name: entry.name,
                reason: "group_size=\(format.groupSize) × bits=\(format.bits) is not a whole number of bytes"
            )
        }

        var output = Data(count: totalBlocks * format.bytesPerBlock)
        weightData.withUnsafeBytes { weightBuffer in
            scalesData.withUnsafeBytes { scalesBuffer in
                biasesData.withUnsafeBytes { biasesBuffer in
                    output.withUnsafeMutableBytes { outputBuffer in
                        let weightBytes = weightBuffer.bindMemory(to: UInt8.self)
                        let scales = scalesBuffer.bindMemory(to: Float16.self)
                        let biases = biasesBuffer.bindMemory(to: Float16.self)
                        let outputBase = outputBuffer.baseAddress!
                        let weightBase = weightBytes.baseAddress!

                        for row in 0..<outputDimension {
                            let rowByteOffset = row * bytesPerRow
                            for block in 0..<blocksPerRow {
                                let blockOffset = (row * blocksPerRow + block) * format.bytesPerBlock
                                let destination = outputBase + blockOffset

                                var scale = scales[row * blocksPerRow + block]
                                memcpy(destination, &scale, 2)

                                var zero = biases[row * blocksPerRow + block]
                                memcpy(destination + 2, &zero, 2)

                                // MLX stores quantized weights as a contiguous
                                // LSB-first bit-stream packed into uint32 words.
                                // STAF uses the same bit-stream layout for the qs
                                // region, so a byte-level copy preserves the
                                // packing for every supported bit width.
                                let srcOffset = rowByteOffset + block * bytesPerGroup
                                memcpy(destination + 4, weightBase + srcOffset, bytesPerGroup)
                            }
                        }
                    }
                }
            }
        }
        return output
    }

    private func loadTensorFromSafetensors(name: String, shardURL: URL) throws -> Data {
        let (data, _) = try loadTensorFromSafetensorsWithDType(name: name, shardURL: shardURL)
        return data
    }

    private func loadTensorFromSafetensorsWithDType(
        name: String, shardURL: URL
    ) throws -> (Data, SafetensorsDType) {
        let loader = SafetensorsLoader()
        let tensors = try loader.parseHeader(at: shardURL)

        guard let tensor = tensors.first(where: { $0.name == name }) else {
            throw STAFConversionError.tensorNotFound(name)
        }

        let fileHandle = try FileHandle(forReadingFrom: shardURL)
        defer { fileHandle.closeFile() }

        guard let sizeData = try fileHandle.read(upToCount: 8), sizeData.count == 8 else {
            throw STAFConversionError.readFailed(name)
        }
        let headerSize = sizeData.withUnsafeBytes { Int(UInt64(littleEndian: $0.load(as: UInt64.self))) }
        let dataSectionOffset = 8 + headerSize

        try fileHandle.seek(toOffset: UInt64(dataSectionOffset + tensor.dataOffset))
        guard let data = try fileHandle.read(upToCount: tensor.byteCount),
              data.count == tensor.byteCount else {
            throw STAFConversionError.readFailed(name)
        }
        return (data, tensor.dtype)
    }

    /// Convert a raw scale/bias tensor to Float16 regardless of its source dtype.
    ///
    /// MLX affine quantization stores `scales` and `biases` in the same dtype
    /// as the original model. For BF16 models (e.g. Gemma3/4), these arrive as
    /// BF16 bytes even though packed-weight kernels consume them as Float16.
    /// Reading BF16 bytes as Float16 produces catastrophically wrong scales and
    /// biases, which previously caused quantized bundles to dequantize to
    /// near-identity layers (hidden collapses to the input embedding, argmax
    /// echoes the last prompt token). Normalize here so the kernel input is
    /// always F16.
    private func normalizeScaleToFloat16(_ data: Data, dtype: SafetensorsDType) throws -> Data {
        switch dtype {
        case .float16:
            return data
        case .bfloat16:
            let elementCount = data.count / 2
            var output = Data(count: elementCount * MemoryLayout<Float16>.size)
            data.withUnsafeBytes { source in
                output.withUnsafeMutableBytes { destination in
                    let bf16Words = source.bindMemory(to: UInt16.self)
                    let halfs = destination.bindMemory(to: Float16.self)
                    for index in 0..<elementCount {
                        // BF16 is the upper 16 bits of a Float32; reconstruct
                        // the Float32 and cast down to Float16.
                        let widened = UInt32(bf16Words[index]) << 16
                        let asFloat = Float(bitPattern: widened)
                        halfs[index] = Float16(asFloat)
                    }
                }
            }
            return output
        case .float32:
            let elementCount = data.count / MemoryLayout<Float>.size
            var output = Data(count: elementCount * MemoryLayout<Float16>.size)
            data.withUnsafeBytes { source in
                output.withUnsafeMutableBytes { destination in
                    let floats = source.bindMemory(to: Float.self)
                    let halfs = destination.bindMemory(to: Float16.self)
                    for index in 0..<elementCount {
                        halfs[index] = Float16(floats[index])
                    }
                }
            }
            return output
        default:
            // Scale/bias dtype is constrained by MLX to F16/BF16/F32. Any
            // other dtype indicates a malformed bundle — fail loudly rather
            // than silently producing bogus Q4 blocks.
            fatalError(
                "Unsupported scale/bias dtype for quantized block packing: \(dtype.rawValue). " +
                "Expected F16, BF16, or F32.")
        }
    }
}
