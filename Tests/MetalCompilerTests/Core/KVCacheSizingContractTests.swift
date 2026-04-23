import Testing
@testable import MetalCompiler

/// Pins `KVCacheSpecification`'s memory sizing contract.
///
/// The per-head byte slot must stay in lockstep with each format's block
/// geometry (`weightsPerBlock` × `bytesPerBlock`). A regression here hands
/// out undersized buffers and causes out-of-bounds KV writes, so the sizing
/// rule is pinned for every scheme reachable from `InferencePolicy`.
@Suite("KV Cache Sizing Contract")
struct KVCacheSizingContractTests {

    private func makeSpec(headDimension: Int) -> KVCacheSpecification {
        KVCacheSpecification(
            keyQuantizationScheme: .fp16RowMajor,
            valueQuantizationScheme: .fp16RowMajor,
            layoutMode: .sequenceMajor,
            layerCount: 1,
            kvHeadCount: 1,
            headDimension: headDimension,
            maximumSequenceLength: 1
        )
    }

    private func alignUp(_ value: Int, to alignment: Int) -> Int {
        let remainder = value % alignment
        return remainder == 0 ? value : value + (alignment - remainder)
    }

    // MARK: - Dense formats

    @Test("Dense FP16 head slot = headDim × 2, aligned to 256B")
    func denseFP16() {
        let spec = makeSpec(headDimension: 128)
        #expect(spec.tokenSlotAlignment(scheme: .fp16RowMajor) == 256)
        #expect(spec.bytesPerHeadSlot(scheme: .fp16RowMajor) == 256)
    }

    @Test("Dense BF16 head slot = headDim × 2, aligned to 256B")
    func denseBF16() {
        let spec = makeSpec(headDimension: 192)
        #expect(spec.tokenSlotAlignment(scheme: .bf16RowMajor) == 256)
        #expect(spec.bytesPerHeadSlot(scheme: .bf16RowMajor) == 512)
    }

    @Test("Dense FP32 head slot = headDim × 4, aligned to 256B")
    func denseFP32() {
        let spec = makeSpec(headDimension: 128)
        #expect(spec.tokenSlotAlignment(scheme: .fp32RowMajor) == 256)
        #expect(spec.bytesPerHeadSlot(scheme: .fp32RowMajor) == 512)
    }

    // MARK: - Block-quantized formats (weight-format paths)

    @Test("Q8G32 head slot: ceil(headDim/32) × 36B, aligned to 64B")
    func q8G32() {
        let spec = makeSpec(headDimension: 128)
        // 4 blocks × 36 = 144 → alignUp to 64B = 192
        #expect(spec.bytesPerHeadSlot(scheme: .q8Group32ScaleF16) == 192)
    }

    @Test("Q8G64 head slot: ceil(headDim/64) × 68B, aligned to 64B")
    func q8G64() {
        let spec = makeSpec(headDimension: 128)
        // 2 blocks × 68 = 136 → alignUp to 64B = 192
        #expect(spec.bytesPerHeadSlot(scheme: .q8Group64ScaleF16) == 192)
    }

    @Test("Q4G64 head slot: ceil(headDim/64) × 36B, aligned to 64B")
    func q4G64() {
        let spec = makeSpec(headDimension: 128)
        // 2 blocks × 36 = 72 → alignUp to 64B = 128
        #expect(spec.bytesPerHeadSlot(scheme: .q4Group64ScaleF16) == 128)
    }

    @Test("Q4G128Zero head slot: ceil(headDim/128) × 68B, aligned to 64B")
    func q4G128Zero() {
        let spec = makeSpec(headDimension: 128)
        // 1 block × 68 = 68 → alignUp to 64B = 128
        #expect(spec.bytesPerHeadSlot(scheme: .q4Group128ScaleF16Zero) == 128)
    }

    // MARK: - Formerly silent-fallback schemes (now protocol-driven)

    @Test("Q6G32 head slot: ceil(headDim/32) × 28B, aligned to 64B")
    func q6G32() {
        let spec = makeSpec(headDimension: 128)
        // 4 blocks × 28 = 112 → alignUp to 64B = 128
        #expect(spec.bytesPerHeadSlot(scheme: .q6Group32ScaleF16) == 128)
    }

    @Test("Q2G32 head slot: ceil(headDim/32) × 12B, aligned to 64B")
    func q2G32() {
        let spec = makeSpec(headDimension: 128)
        // 4 blocks × 12 = 48 → alignUp to 64B = 64
        #expect(spec.bytesPerHeadSlot(scheme: .q2Group32ScaleF16) == 64)
    }

    // MARK: - Rotor schemes delegate to base

    @Test("RotorQ8G32 head slot matches base Q8G32 layout")
    func rotorQ8G32MatchesBase() {
        let spec = makeSpec(headDimension: 128)
        #expect(
            spec.bytesPerHeadSlot(scheme: .rotorQ8Group32ScaleF16)
                == spec.bytesPerHeadSlot(scheme: .q8Group32ScaleF16)
        )
        #expect(
            spec.tokenSlotAlignment(scheme: .rotorQ8Group32ScaleF16)
                == spec.tokenSlotAlignment(scheme: .q8Group32ScaleF16)
        )
    }

    @Test("RotorQ4G64 head slot matches base Q4G64 layout")
    func rotorQ4G64MatchesBase() {
        let spec = makeSpec(headDimension: 256)
        #expect(
            spec.bytesPerHeadSlot(scheme: .rotorQ4Group64ScaleF16)
                == spec.bytesPerHeadSlot(scheme: .q4Group64ScaleF16)
        )
        #expect(
            spec.tokenSlotAlignment(scheme: .rotorQ4Group64ScaleF16)
                == spec.tokenSlotAlignment(scheme: .q4Group64ScaleF16)
        )
    }

    // MARK: - Protocol-formula invariant (covers every resolvable scheme)

    @Test("bytesPerHeadSlot matches ceil(headDim/weightsPerBlock) × bytesPerBlock for every reachable scheme")
    func formulaInvariantAcrossAllReachableSchemes() throws {
        let headDimension = 192  // non-power-of-two that exercises ceiling division
        let spec = makeSpec(headDimension: headDimension)

        // Every scheme that is reachable from InferencePolicy (weight formats +
        // rotor KV-cache schemes) must match the protocol-driven formula.
        let reachable = SupportedQuantizationsContractTests.registeredSchemes
            .union(SupportedQuantizationsContractTests.kvCacheOnlySchemes)

        for scheme in reachable {
            let format = try #require(
                QuantizationFormatRegistry.format(for: scheme.baseScheme),
                "baseScheme of \(scheme) must resolve to a QuantizationFormat"
            )
            let blockCount =
                (headDimension + format.weightsPerBlock - 1) / format.weightsPerBlock
            let expectedRaw = blockCount * format.bytesPerBlock
            let expectedAlignment = format.isQuantized ? 64 : 256
            let expected = alignUp(expectedRaw, to: expectedAlignment)

            #expect(
                spec.tokenSlotAlignment(scheme: scheme) == expectedAlignment,
                "tokenSlotAlignment mismatch for \(scheme)"
            )
            #expect(
                spec.bytesPerHeadSlot(scheme: scheme) == expected,
                "bytesPerHeadSlot mismatch for \(scheme): got \(spec.bytesPerHeadSlot(scheme: scheme)), expected \(expected)"
            )
        }
    }

    // MARK: - Derived sizing stays consistent

    @Test("Layer / token / total sizes are integer multiples of bytesPerHeadSlot")
    func derivedSizesAreConsistent() {
        let spec = KVCacheSpecification(
            keyQuantizationScheme: .q8Group32ScaleF16,
            valueQuantizationScheme: .q4Group64ScaleF16,
            layoutMode: .sequenceMajor,
            layerCount: 3,
            kvHeadCount: 4,
            headDimension: 128,
            maximumSequenceLength: 16
        )

        let kHead = spec.bytesPerHeadSlot(scheme: .q8Group32ScaleF16)
        let vHead = spec.bytesPerHeadSlot(scheme: .q4Group64ScaleF16)

        #expect(spec.bytesPerTokenSlot(scheme: .q8Group32ScaleF16) == 4 * kHead)
        #expect(spec.bytesPerTokenSlot(scheme: .q4Group64ScaleF16) == 4 * vHead)
        #expect(spec.bytesPerLayer(scheme: .q8Group32ScaleF16) == 16 * 4 * kHead)
        #expect(spec.bytesPerLayer(scheme: .q4Group64ScaleF16) == 16 * 4 * vHead)
        #expect(spec.totalBufferSize(scheme: .q8Group32ScaleF16) == 3 * 16 * 4 * kHead)
        #expect(spec.totalBufferSize(scheme: .q4Group64ScaleF16) == 3 * 16 * 4 * vHead)
    }
}
