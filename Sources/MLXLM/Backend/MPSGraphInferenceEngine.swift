import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX

/// Transformer inference engine built on MPSGraph.
///
/// Compiles all layers into a single fused execution plan at init.
/// Each call processes token IDs through the full model and returns logits.
public final class MPSGraphInferenceEngine: @unchecked Sendable {

    public struct Config: Sendable {
        public let hiddenSize: Int
        public let headCount: Int
        public let kvHeadCount: Int
        public let headDim: Int
        public let intermediateSize: Int
        public let layerCount: Int
        public let vocabSize: Int

        public init(hiddenSize: Int, headCount: Int, kvHeadCount: Int, headDim: Int,
                    intermediateSize: Int, layerCount: Int, vocabSize: Int) {
            self.hiddenSize = hiddenSize
            self.headCount = headCount
            self.kvHeadCount = kvHeadCount
            self.headDim = headDim
            self.intermediateSize = intermediateSize
            self.layerCount = layerCount
            self.vocabSize = vocabSize
        }
    }

    public enum Error: Swift.Error, LocalizedError {
        case noMetalDevice
        case noCommandQueue
        case missingWeight(String)

        public var errorDescription: String? {
            switch self {
            case .noMetalDevice: return "No Metal device available"
            case .noCommandQueue: return "Failed to create Metal command queue"
            case .missingWeight(let name): return "Missing weight: \(name)"
            }
        }
    }

    public let config: Config
    private let device: MTLDevice
    private let graph: MPSGraph
    private let commandQueue: MTLCommandQueue
    private let inputPlaceholder: MPSGraphTensor
    private let outputTensor: MPSGraphTensor

    /// Build and compile a transformer graph with the given weights.
    public init(config: Config, weights: [String: Data]) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Error.noMetalDevice }
        guard let queue = device.makeCommandQueue() else { throw Error.noCommandQueue }

        self.config = config
        self.device = device
        self.commandQueue = queue

        let graph = MPSGraph()
        self.graph = graph

        let (input, output) = try Self.buildGraph(graph: graph, config: config, weights: weights)
        self.inputPlaceholder = input
        self.outputTensor = output
    }

    /// Forward pass: token IDs → logits.
    public func callAsFunction(_ tokenIDs: [Int32]) -> MLXArray {
        let data = tokenIDs.withUnsafeBufferPointer { ptr in
            MPSGraphTensorData(
                device: MPSGraphDevice(mtlDevice: device),
                data: Data(buffer: ptr),
                shape: [1, tokenIDs.count as NSNumber],
                dataType: .int32)
        }

        let results = graph.run(
            with: commandQueue,
            feeds: [inputPlaceholder: data],
            targetTensors: [outputTensor],
            targetOperations: nil)

        let result = results[outputTensor]!
        let shape = result.shape.map { $0.intValue }
        let count = shape.reduce(1, *)
        var buffer = [Float16](repeating: 0, count: count)
        result.mpsndarray().readBytes(&buffer, strideBytes: nil)
        return MLXArray(buffer, shape)
    }

    // MARK: - Graph Construction

    private static func buildGraph(
        graph: MPSGraph, config: Config, weights: [String: Data]
    ) throws -> (input: MPSGraphTensor, output: MPSGraphTensor) {
        let D = config.hiddenSize
        let H = config.headCount
        let KVH = config.kvHeadCount
        let hd = config.headDim
        let I = config.intermediateSize
        let V = config.vocabSize

        func w(_ name: String, _ shape: [Int]) throws -> MPSGraphTensor {
            guard let data = weights[name] else { throw Error.missingWeight(name) }
            return graph.variable(with: data, shape: shape.map { $0 as NSNumber },
                                   dataType: .float16, name: name)
        }

        let embedding = try w("model.embed_tokens.weight", [V, D])
        let input = graph.placeholder(shape: [1, 1], dataType: .int32, name: "token_ids")

        var h = graph.gatherAlongAxis(0, updates: embedding, indices: input, name: "embed")

        for i in 0..<config.layerCount {
            let p = "model.layers.\(i)"
            h = try buildLayer(
                graph: graph, input: h, index: i,
                normW1: w("\(p).input_layernorm.weight", [D]),
                qW: w("\(p).self_attn.q_proj.weight", [H * hd, D]),
                kW: w("\(p).self_attn.k_proj.weight", [KVH * hd, D]),
                vW: w("\(p).self_attn.v_proj.weight", [KVH * hd, D]),
                oW: w("\(p).self_attn.o_proj.weight", [D, H * hd]),
                normW2: w("\(p).post_attention_layernorm.weight", [D]),
                gateW: w("\(p).mlp.gate_proj.weight", [I, D]),
                upW: w("\(p).mlp.up_proj.weight", [I, D]),
                downW: w("\(p).mlp.down_proj.weight", [D, I]),
                H: H, KVH: KVH, hd: hd)
        }

        let finalNorm = try w("model.norm.weight", [D])
        h = rmsNorm(graph, h, finalNorm, "final")
        let logits = linear(graph, h, embedding, "lm_head")

        return (input, logits)
    }

    private static func buildLayer(
        graph: MPSGraph, input: MPSGraphTensor, index i: Int,
        normW1: MPSGraphTensor, qW: MPSGraphTensor, kW: MPSGraphTensor,
        vW: MPSGraphTensor, oW: MPSGraphTensor, normW2: MPSGraphTensor,
        gateW: MPSGraphTensor, upW: MPSGraphTensor, downW: MPSGraphTensor,
        H: Int, KVH: Int, hd: Int
    ) -> MPSGraphTensor {
        var h = input

        // Attention
        let n1 = rmsNorm(graph, h, normW1, "l\(i).an")
        let q = toHeads(graph, linear(graph, n1, qW, "l\(i).q"), H, hd, "l\(i).q")
        let k = toHeads(graph, linear(graph, n1, kW, "l\(i).k"), KVH, hd, "l\(i).k")
        let v = toHeads(graph, linear(graph, n1, vW, "l\(i).v"), KVH, hd, "l\(i).v")
        let attn = sdpa(graph, q, k, v, hd, "l\(i)")
        let flat = graph.reshape(
            graph.transposeTensor(attn, dimension: 1, withDimension: 2, name: nil),
            shape: [1, 1, (H * hd) as NSNumber], name: "l\(i).flat")
        h = graph.addition(h, linear(graph, flat, oW, "l\(i).o"), name: "l\(i).ar")

        // MLP
        let n2 = rmsNorm(graph, h, normW2, "l\(i).mn")
        let gate = linear(graph, n2, gateW, "l\(i).gate")
        let up = linear(graph, n2, upW, "l\(i).up")
        let silu = graph.multiplication(
            gate, graph.sigmoid(with: gate, name: "l\(i).sig"), name: "l\(i).silu")
        let act = graph.multiplication(silu, up, name: "l\(i).swiglu")
        h = graph.addition(h, linear(graph, act, downW, "l\(i).down"), name: "l\(i).mr")

        return h
    }

    // MARK: - Ops

    private static func linear(
        _ g: MPSGraph, _ x: MPSGraphTensor, _ w: MPSGraphTensor, _ n: String
    ) -> MPSGraphTensor {
        g.matrixMultiplication(
            primary: x, secondary: g.transposeTensor(w, dimension: 0, withDimension: 1, name: nil),
            name: n)
    }

    private static func rmsNorm(
        _ g: MPSGraph, _ x: MPSGraphTensor, _ w: MPSGraphTensor, _ n: String
    ) -> MPSGraphTensor {
        let eps = g.constant(1e-5, dataType: .float16)
        let sq = g.multiplication(x, x, name: "\(n).sq")
        let mean = g.mean(of: sq, axes: [-1], name: "\(n).m")
        let inv = g.reverseSquareRoot(with: g.addition(mean, eps, name: "\(n).e"), name: "\(n).i")
        return g.multiplication(g.multiplication(x, inv, name: "\(n).n"), w, name: "\(n).o")
    }

    private static func toHeads(
        _ g: MPSGraph, _ x: MPSGraphTensor, _ heads: Int, _ hd: Int, _ n: String
    ) -> MPSGraphTensor {
        g.transposeTensor(
            g.reshape(x, shape: [1, 1, heads as NSNumber, hd as NSNumber], name: nil),
            dimension: 1, withDimension: 2, name: "\(n).h")
    }

    private static func sdpa(
        _ g: MPSGraph, _ q: MPSGraphTensor, _ k: MPSGraphTensor,
        _ v: MPSGraphTensor, _ hd: Int, _ n: String
    ) -> MPSGraphTensor {
        let s = g.constant(Double(1.0 / Float(hd).squareRoot()), dataType: .float16)
        let scores = g.multiplication(
            g.matrixMultiplication(
                primary: q, secondary: g.transposeTensor(k, dimension: 2, withDimension: 3, name: nil),
                name: "\(n).qk"),
            s, name: "\(n).sc")
        return g.matrixMultiplication(
            primary: g.softMax(with: scores, axis: -1, name: "\(n).sm"),
            secondary: v, name: "\(n).av")
    }
}
