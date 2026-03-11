import Testing
import Foundation
import MLX
import GGUFParser
@testable import MLXLM

/// Verify that GPU-accelerated pack kernels produce identical output to CPU pack functions.
///
/// Strategy: create synthetic data large enough to trigger GPU dispatch (>= gpuThreshold),
/// run GGUFGPUPacker.tryPack() for GPU, and compare against CPU result from
/// GGUFTensorBridge.convertDirect() on a small tensor of the same pattern.
@Suite("GPU Pack Correctness", .tags(.unit))
struct GPUPackCorrectnessTests {

    private let bridge = GGUFTensorBridge()

    // Large enough for GPU dispatch: 256 * 2048 = 524,288 elements
    // super-block count for 256-element types = 2048
    // block count for 32-element types = 16384
    private static let gpuRows = 2048
    private static let gpuCols = 256

    // Small tensor that stays on CPU path (below threshold)
    // 256 * 4 = 1024 elements → 4 super-blocks or 32 blocks
    private static let cpuRows = 4
    private static let cpuCols = 256

    // MARK: - Helper: generate synthetic data

    private func makeData(
        blockCount: Int, bytesPerBlock: Int, scaleOffset: Int, scaleIsF32: Bool = false
    ) -> Data {
        let dataSize = blockCount * bytesPerBlock
        var data = Data(count: dataSize)
        // Fill with deterministic non-zero pattern
        for i in 0..<dataSize {
            data[i] = UInt8(truncatingIfNeeded: i &* 7 &+ 13)
        }
        // Write valid scale values
        for b in 0..<blockCount {
            let offset = b * bytesPerBlock + scaleOffset
            if scaleIsF32 {
                let d: Float = 0.01
                withUnsafeBytes(of: d.bitPattern.littleEndian) {
                    data.replaceSubrange(offset..<offset + 4, with: $0)
                }
            } else {
                let d = Float16(0.01)
                withUnsafeBytes(of: d.bitPattern.littleEndian) {
                    data.replaceSubrange(offset..<offset + 2, with: $0)
                }
            }
        }
        return data
    }

    // Q4_K/Q5_K also need dmin written at offset+2
    private func makeKTypeData(
        superBlockCount: Int, bytesPerSB: Int
    ) -> Data {
        let dataSize = superBlockCount * bytesPerSB
        var data = Data(count: dataSize)
        for i in 0..<dataSize {
            data[i] = UInt8(truncatingIfNeeded: i &* 7 &+ 13)
        }
        for sb in 0..<superBlockCount {
            let offset = sb * bytesPerSB
            let d = Float16(0.01)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            let dmin = Float16(0.005)
            withUnsafeBytes(of: dmin.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 2..<offset + 4, with: $0)
            }
        }
        return data
    }

    // Q6_K: d at offset+208
    private func makeQ6KData(superBlockCount: Int) -> Data {
        let bytesPerSB = 210
        let dataSize = superBlockCount * bytesPerSB
        var data = Data(count: dataSize)
        for i in 0..<dataSize {
            data[i] = UInt8(truncatingIfNeeded: i &* 7 &+ 13)
        }
        for sb in 0..<superBlockCount {
            let offset = sb * bytesPerSB
            let d = Float16(0.01)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 208..<offset + 210, with: $0)
            }
        }
        return data
    }

    // MARK: - Comparison helper

    private func compareGPUvsCPU(
        qtype: GGUFQuantizationType,
        gpuData: Data, cpuData: Data,
        gpuShape: [Int], cpuShape: [Int],
        label: String
    ) throws {
        // GPU path
        guard let gpuResult = GGUFGPUPacker.tryPack(qtype: qtype, data: gpuData, shape: gpuShape) else {
            Issue.record("GPU packer returned nil for \(label)")
            return
        }

        // CPU path (small tensor, won't trigger GPU)
        let tensor = GGUFTensorInfo(
            name: "test", dimensions: cpuShape.reversed(),
            quantizationType: qtype, offset: 0
        )
        let cpuResult = try bridge.convertDirect(tensor: tensor, data: cpuData)

        // Extract arrays from both results
        guard case .quantized(let gpuW, let gpuS, let gpuB, let gpuGS, let gpuBits) = gpuResult,
              case .quantized(let cpuW, let cpuS, let cpuB, let cpuGS, let cpuBits) = cpuResult else {
            Issue.record("\(label): expected .quantized from both paths")
            return
        }

        #expect(gpuGS == cpuGS, "\(label): groupSize mismatch")
        #expect(gpuBits == cpuBits, "\(label): bits mismatch")

        // Compare the first cpuRows rows of GPU output against CPU output
        // GPU tensor is [gpuRows, ...], CPU is [cpuRows, ...]
        // We slice the GPU result to get [cpuRows, ...]
        let gpuWFlat = gpuW.reshaped(-1)
        let cpuWFlat = cpuW.reshaped(-1)
        let gpuSFlat = gpuS.reshaped(-1)
        let cpuSFlat = cpuS.reshaped(-1)
        let gpuBFlat = gpuB.reshaped(-1)
        let cpuBFlat = cpuB.reshaped(-1)

        eval(gpuWFlat, cpuWFlat, gpuSFlat, cpuSFlat, gpuBFlat, cpuBFlat)

        let cpuWCount = cpuWFlat.size
        let cpuSCount = cpuSFlat.size

        // Compare first cpuWCount elements of packed weights
        let gpuWSlice = gpuWFlat[0..<cpuWCount]
        let cpuWSlice = cpuWFlat[0..<cpuWCount]
        eval(gpuWSlice, cpuWSlice)
        let wMatch = (gpuWSlice .== cpuWSlice).all()
        eval(wMatch)
        #expect(wMatch.item(Bool.self), "\(label): packed weights mismatch")

        // Compare first cpuSCount scales
        let gpuSSlice = gpuSFlat[0..<cpuSCount]
        let cpuSSlice = cpuSFlat[0..<cpuSCount]
        eval(gpuSSlice, cpuSSlice)
        let sMatch = MLX.allClose(gpuSSlice, cpuSSlice, atol: 1e-4)
        eval(sMatch)
        #expect(sMatch.item(Bool.self), "\(label): scales mismatch")

        // Compare first cpuSCount biases
        let gpuBSlice = gpuBFlat[0..<cpuSCount]
        let cpuBSlice = cpuBFlat[0..<cpuSCount]
        eval(gpuBSlice, cpuBSlice)
        let bMatch = MLX.allClose(gpuBSlice, cpuBSlice, atol: 1e-4)
        eval(bMatch)
        #expect(bMatch.item(Bool.self), "\(label): biases mismatch")
    }

    // MARK: - Tests for each quantization type

    @Test("Q4_0 GPU matches CPU")
    func testQ4_0() throws {
        let gpuElements = Self.gpuRows * Self.gpuCols
        let cpuElements = Self.cpuRows * Self.gpuCols
        let gpuBlocks = gpuElements / 32
        let cpuBlocks = cpuElements / 32
        let gpuData = makeData(blockCount: gpuBlocks, bytesPerBlock: 18, scaleOffset: 0)
        let cpuData = Data(gpuData.prefix(cpuBlocks * 18))
        try compareGPUvsCPU(
            qtype: .q4_0, gpuData: gpuData, cpuData: cpuData,
            gpuShape: [Self.gpuRows, Self.gpuCols],
            cpuShape: [Self.cpuRows, Self.gpuCols],
            label: "Q4_0"
        )
    }

    @Test("Q4_1 GPU matches CPU")
    func testQ4_1() throws {
        let gpuElements = Self.gpuRows * Self.gpuCols
        let cpuElements = Self.cpuRows * Self.gpuCols
        let gpuBlocks = gpuElements / 32
        let cpuBlocks = cpuElements / 32
        // Q4_1: d at offset+0, m at offset+2
        var gpuData = makeData(blockCount: gpuBlocks, bytesPerBlock: 20, scaleOffset: 0)
        // Write m (bias) at offset+2
        for b in 0..<gpuBlocks {
            let offset = b * 20 + 2
            let m = Float16(0.005)
            withUnsafeBytes(of: m.bitPattern.littleEndian) {
                gpuData.replaceSubrange(offset..<offset + 2, with: $0)
            }
        }
        let cpuData = Data(gpuData.prefix(cpuBlocks * 20))
        try compareGPUvsCPU(
            qtype: .q4_1, gpuData: gpuData, cpuData: cpuData,
            gpuShape: [Self.gpuRows, Self.gpuCols],
            cpuShape: [Self.cpuRows, Self.gpuCols],
            label: "Q4_1"
        )
    }

    @Test("Q8_0 GPU matches CPU")
    func testQ8_0() throws {
        let gpuBlocks = (Self.gpuRows * Self.gpuCols) / 32
        let cpuBlocks = (Self.cpuRows * Self.gpuCols) / 32
        let gpuData = makeData(blockCount: gpuBlocks, bytesPerBlock: 34, scaleOffset: 0)
        let cpuData = Data(gpuData.prefix(cpuBlocks * 34))
        try compareGPUvsCPU(
            qtype: .q8_0, gpuData: gpuData, cpuData: cpuData,
            gpuShape: [Self.gpuRows, Self.gpuCols],
            cpuShape: [Self.cpuRows, Self.gpuCols],
            label: "Q8_0"
        )
    }

    @Test("Q8_1 GPU matches CPU")
    func testQ8_1() throws {
        let gpuBlocks = (Self.gpuRows * Self.gpuCols) / 32
        let cpuBlocks = (Self.cpuRows * Self.gpuCols) / 32
        // Q8_1: d at offset+0, m at offset+2
        var gpuData = makeData(blockCount: gpuBlocks, bytesPerBlock: 36, scaleOffset: 0)
        for b in 0..<gpuBlocks {
            let offset = b * 36 + 2
            let m = Float16(0.005)
            withUnsafeBytes(of: m.bitPattern.littleEndian) {
                gpuData.replaceSubrange(offset..<offset + 2, with: $0)
            }
        }
        let cpuData = Data(gpuData.prefix(cpuBlocks * 36))
        try compareGPUvsCPU(
            qtype: .q8_1, gpuData: gpuData, cpuData: cpuData,
            gpuShape: [Self.gpuRows, Self.gpuCols],
            cpuShape: [Self.cpuRows, Self.gpuCols],
            label: "Q8_1"
        )
    }

    @Test("Q5_0 GPU matches CPU")
    func testQ5_0() throws {
        let gpuBlocks = (Self.gpuRows * Self.gpuCols) / 32
        let cpuBlocks = (Self.cpuRows * Self.gpuCols) / 32
        let gpuData = makeData(blockCount: gpuBlocks, bytesPerBlock: 22, scaleOffset: 0)
        let cpuData = Data(gpuData.prefix(cpuBlocks * 22))
        try compareGPUvsCPU(
            qtype: .q5_0, gpuData: gpuData, cpuData: cpuData,
            gpuShape: [Self.gpuRows, Self.gpuCols],
            cpuShape: [Self.cpuRows, Self.gpuCols],
            label: "Q5_0"
        )
    }

    @Test("Q5_1 GPU matches CPU")
    func testQ5_1() throws {
        let gpuBlocks = (Self.gpuRows * Self.gpuCols) / 32
        let cpuBlocks = (Self.cpuRows * Self.gpuCols) / 32
        // Q5_1: d at offset+0, m at offset+2
        var gpuData = makeData(blockCount: gpuBlocks, bytesPerBlock: 24, scaleOffset: 0)
        for b in 0..<gpuBlocks {
            let offset = b * 24 + 2
            let m = Float16(0.005)
            withUnsafeBytes(of: m.bitPattern.littleEndian) {
                gpuData.replaceSubrange(offset..<offset + 2, with: $0)
            }
        }
        let cpuData = Data(gpuData.prefix(cpuBlocks * 24))
        try compareGPUvsCPU(
            qtype: .q5_1, gpuData: gpuData, cpuData: cpuData,
            gpuShape: [Self.gpuRows, Self.gpuCols],
            cpuShape: [Self.cpuRows, Self.gpuCols],
            label: "Q5_1"
        )
    }

    @Test("Q4_K GPU matches CPU")
    func testQ4_K() throws {
        let gpuSBs = (Self.gpuRows * Self.gpuCols) / 256
        let cpuSBs = (Self.cpuRows * Self.gpuCols) / 256
        let gpuData = makeKTypeData(superBlockCount: gpuSBs, bytesPerSB: 144)
        let cpuData = Data(gpuData.prefix(cpuSBs * 144))
        try compareGPUvsCPU(
            qtype: .q4_K, gpuData: gpuData, cpuData: cpuData,
            gpuShape: [Self.gpuRows, Self.gpuCols],
            cpuShape: [Self.cpuRows, Self.gpuCols],
            label: "Q4_K"
        )
    }

    @Test("Q5_K GPU matches CPU")
    func testQ5_K() throws {
        let gpuSBs = (Self.gpuRows * Self.gpuCols) / 256
        let cpuSBs = (Self.cpuRows * Self.gpuCols) / 256
        let gpuData = makeKTypeData(superBlockCount: gpuSBs, bytesPerSB: 176)
        let cpuData = Data(gpuData.prefix(cpuSBs * 176))
        try compareGPUvsCPU(
            qtype: .q5_K, gpuData: gpuData, cpuData: cpuData,
            gpuShape: [Self.gpuRows, Self.gpuCols],
            cpuShape: [Self.cpuRows, Self.gpuCols],
            label: "Q5_K"
        )
    }

    @Test("Q6_K GPU matches CPU")
    func testQ6_K() throws {
        let gpuSBs = (Self.gpuRows * Self.gpuCols) / 256
        let cpuSBs = (Self.cpuRows * Self.gpuCols) / 256
        let gpuData = makeQ6KData(superBlockCount: gpuSBs)
        let cpuData = Data(gpuData.prefix(cpuSBs * 210))
        try compareGPUvsCPU(
            qtype: .q6_K, gpuData: gpuData, cpuData: cpuData,
            gpuShape: [Self.gpuRows, Self.gpuCols],
            cpuShape: [Self.cpuRows, Self.gpuCols],
            label: "Q6_K"
        )
    }

    @Test("Q8_K GPU matches CPU")
    func testQ8_K() throws {
        let gpuSBs = (Self.gpuRows * Self.gpuCols) / 256
        let cpuSBs = (Self.cpuRows * Self.gpuCols) / 256
        // Q8_K: d is float32 at offset+0
        let gpuData = makeData(blockCount: gpuSBs, bytesPerBlock: 292, scaleOffset: 0, scaleIsF32: true)
        let cpuData = Data(gpuData.prefix(cpuSBs * 292))
        try compareGPUvsCPU(
            qtype: .q8_K, gpuData: gpuData, cpuData: cpuData,
            gpuShape: [Self.gpuRows, Self.gpuCols],
            cpuShape: [Self.cpuRows, Self.gpuCols],
            label: "Q8_K"
        )
    }
}
