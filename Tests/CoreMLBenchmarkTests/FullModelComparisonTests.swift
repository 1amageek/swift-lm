import Testing
import TestHeartbeat
@preconcurrency import MLX
import MLXFast
import MLXNN
import CoreML
import Foundation
@testable import SwiftLM
@testable import MLXCompiler

// MARK: - Full Model Comparison: Same graph, all layers in ONE execution
//
// MLX:    24L in 1 eval() (lazy graph optimization across all layers)
// CoreML: 24L in 1 prediction() (MPSGraph optimization across all layers)
//
// This is the CORRECT comparison — both backends execute ALL layers at once.

private let fullModelDir = "/tmp/full-coreml"

private func bench(warmup: Int = 10, iters: Int = 30, _ body: () throws -> Void) rethrows -> Double {
    let clock = ContinuousClock()
    for _ in 0..<warmup { try body() }
    var ds: [Double] = []
    for _ in 0..<iters {
        let s = clock.now; try body(); let e = clock.now - s
        ds.append(Double(e.components.seconds) * 1000 + Double(e.components.attoseconds) * 1e-15)
    }
    ds.sort()
    return ds[ds.count / 2]
}

// MARK: - MLX Model

private func slot(_ c: [StructuralPathComponent], role: ParameterRole) -> ParameterSlot {
    ParameterSlot(path: StructuralPath(components: c), role: role)
}

private struct BenchTransformer: ModelComponent {
    let L: Int; let V: Int; let D: Int; let H: Int; let KVH: Int; let hd: Int; let I: Int
    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: V, embeddingSize: D)
        Repeat(count: L) {
            Residual { RMSNorm(dimension: D); Attention(hiddenSize: D, headCount: H, kvHeadCount: KVH, headDimension: hd) }
            Residual { RMSNorm(dimension: D); MLP(inputSize: D, intermediateSize: I) }
        }
        RMSNorm(dimension: D)
        OutputHead(inputSize: D, vocabSize: V, tiedToEmbedding: true)
    }
}

private func buildMLX(D: Int, H: Int, KVH: Int, hd: Int, I: Int, L: Int, V: Int = 32000) throws -> MLXLoweredInferenceModel {
    MLXRandom.seed(42)
    let graph = try BenchTransformer(L: L, V: V, D: D, H: H, KVH: KVH, hd: hd, I: I).makeModelGraph()
    var dict: [ParameterSlot: MLXArray] = [:]
    dict[slot([.operation(0)], role: .embeddingTable)] = (MLXRandom.normal([V, D]) * 0.02).asType(.float16)
    for i in 0..<L {
        let lp: [StructuralPathComponent] = [.operation(1), .regionBody, .index(i)]
        dict[slot(lp + [.operation(0), .regionBody, .operation(0)], role: .scale)] = MLXArray.ones([D]).asType(.float16)
        let ap = lp + [.operation(0), .regionBody, .operation(1)]
        dict[slot(ap + [.field("q_proj")], role: .weight)] = (MLXRandom.normal([H*hd, D]) * 0.02).asType(.float16)
        dict[slot(ap + [.field("k_proj")], role: .weight)] = (MLXRandom.normal([KVH*hd, D]) * 0.02).asType(.float16)
        dict[slot(ap + [.field("v_proj")], role: .weight)] = (MLXRandom.normal([KVH*hd, D]) * 0.02).asType(.float16)
        dict[slot(ap + [.field("o_proj")], role: .weight)] = (MLXRandom.normal([D, H*hd]) * 0.02).asType(.float16)
        dict[slot(lp + [.operation(1), .regionBody, .operation(0)], role: .scale)] = MLXArray.ones([D]).asType(.float16)
        let mp = lp + [.operation(1), .regionBody, .operation(1)]
        dict[slot(mp + [.field("gate_proj")], role: .weight)] = (MLXRandom.normal([I, D]) * 0.02).asType(.float16)
        dict[slot(mp + [.field("up_proj")], role: .weight)] = (MLXRandom.normal([I, D]) * 0.02).asType(.float16)
        dict[slot(mp + [.field("down_proj")], role: .weight)] = (MLXRandom.normal([D, I]) * 0.02).asType(.float16)
    }
    dict[slot([.operation(2)], role: .scale)] = MLXArray.ones([D]).asType(.float16)
    eval(Array(dict.values))
    var tensors: [ParameterSlot: TensorData] = [:]
    for (s, a) in dict { tensors[s] = TensorData(shape: a.shape.map{$0}, dtype: .float16, storage: a) }
    return try MLXInferenceCompiler().compile(graph: graph, weights: BoundWeights(tensors: tensors))
}

// MARK: - Suite

@Suite("Full Model Comparison", .serialized, .heartbeat)
struct FullModelComparisonTests {

    @Test("Full model decode: 4L and 24L — CoreML vs MLX")
    func fullModelDecode() throws {
        let configs: [(D: Int, H: Int, KVH: Int, hd: Int, I: Int, label: String)] = [
            (896, 14, 2, 64, 4864, "0.6B"),
        ]
        let layerCounts = [4, 24]

        print("\n" + String(repeating: "=", count: 100))
        print("FULL MODEL DECODE (T=1) — ALL layers in ONE execution")
        print(String(repeating: "=", count: 100))
        print("Model   L     MLX-1eval   CM.all      CM.GPU      CM.ANE      per-layer-MLX  per-layer-CM")
        print(String(repeating: "-", count: 100))

        for c in configs {
            for L in layerCounts {
                let modelName = "full_D\(c.D)_L\(L)_seq256"
                let modelPath = "\(fullModelDir)/\(modelName).mlpackage"

                guard FileManager.default.fileExists(atPath: modelPath) else {
                    print("\(c.label)   \(L)     SKIPPED (run compile_full_model.py)")
                    continue
                }

                // MLX: build and decode
                let mlxModel = try buildMLX(D: c.D, H: c.H, KVH: c.KVH, hd: c.hd, I: c.I, L: L)
                let prompt = MLXArray([Int32(1)]).expandedDimensions(axis: 0)
                eval(prompt)
                let (_, warmState) = mlxModel.prefill(tokenIDs: prompt, state: mlxModel.makeState())
                let tok = MLXArray([Int32(2)]).expandedDimensions(axis: 0)
                eval(tok)

                let tMLX = bench {
                    let (logits, _) = mlxModel.decode(tokenIDs: tok, state: warmState)
                    eval(logits)
                }

                // CoreML: load and decode
                func benchCM(units: MLComputeUnits) throws -> Double {
                    let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: modelPath))
                    let cfg = MLModelConfiguration()
                    cfg.computeUnits = units
                    let model = try MLModel(contentsOf: compiled, configuration: cfg)

                    let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
                    tokenInput[0] = 1
                    let offsetInput = try MLMultiArray(shape: [1], dataType: .int32)
                    let state = model.makeState()

                    // Warmup prefill
                    offsetInput[0] = 0
                    let prov0 = try MLDictionaryFeatureProvider(
                        dictionary: ["token_ids": MLFeatureValue(multiArray: tokenInput),
                                     "offset": MLFeatureValue(multiArray: offsetInput)])
                    let _ = try model.prediction(from: prov0, using: state)

                    return try bench {
                        offsetInput[0] = 1
                        let provider = try MLDictionaryFeatureProvider(
                            dictionary: ["token_ids": MLFeatureValue(multiArray: tokenInput),
                                         "offset": MLFeatureValue(multiArray: offsetInput)])
                        let _ = try model.prediction(from: provider, using: state)
                    }
                }

                let tAll = try benchCM(units: .all)
                let tGPU = try benchCM(units: .cpuAndGPU)
                let tANE = try benchCM(units: .cpuAndNeuralEngine)

                let cmBest = min(tAll, tGPU, tANE)
                let perLayerMLX = tMLX / Double(L)
                let perLayerCM = cmBest / Double(L)

                let lbl = c.label.padding(toLength: 5, withPad: " ", startingAt: 0)
                print(String(format: "%@   %-3d   %10.3f  %10.3f  %10.3f  %10.3f    %8.3f       %8.3f",
                              lbl, L, tMLX, tAll, tGPU, tANE, perLayerMLX, perLayerCM))
            }
        }

        print(String(repeating: "=", count: 100))
        print("MLX-1eval = all layers in single eval() | CM = all layers in single prediction()")
        print("per-layer = total / L (effective per-layer cost with multi-layer optimization)")
    }
}
