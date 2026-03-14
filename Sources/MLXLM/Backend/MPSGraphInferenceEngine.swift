import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX

/// Transformer inference engine built on MPSGraph.
///
/// Compiles the full model into a single fused execution plan.
/// Accepts weights as `[String: Data]` (Float16 raw bytes keyed by parameter name).
///
/// The graph is built once at init, compiled into an `MPSGraphExecutable`,
/// and reused for every forward pass.
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
    let device: MTLDevice
    let graph: MPSGraph
    let commandQueue: MTLCommandQueue
    private(set) var inputPlaceholder: MPSGraphTensor!
    private(set) var outputTensor: MPSGraphTensor!

    /// Build and compile a transformer graph with the given weights.
    ///
    /// - Parameters:
    ///   - config: Model dimensions.
    ///   - weights: Weight data keyed by HF parameter name (e.g. "model.layers.0.self_attn.q_proj.weight").
    ///             Values are raw Float16 bytes in row-major layout.
    public init(config: Config, weights: [String: Data]) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Error.noMetalDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw Error.noCommandQueue
        }
        self.config = config
        self.device = device
        self.commandQueue = queue
        self.graph = MPSGraph()

        let D = config.hiddenSize
        let H = config.headCount
        let KVH = config.kvHeadCount
        let hd = config.headDim
        let I = config.intermediateSize
        let V = config.vocabSize

        // Resolve weight helper
        func weight(_ name: String, shape: [Int]) throws -> MPSGraphTensor {
            guard let data = weights[name] else {
                throw Error.missingWeight(name)
            }
            return graph.variable(
                with: data, shape: shape.map { $0 as NSNumber },
                dataType: .float16, name: name)
        }

        // Token embedding
        let embeddingTable = try weight("model.embed_tokens.weight", shape: [V, D])

        // Input: token IDs [1, T]
        let input = graph.placeholder(shape: [1, 1], dataType: .int32, name: "token_ids")
        self.inputPlaceholder = input

        // Embedding lookup
        var h = graph.gatherAlongAxis(
            0, updates: embeddingTable, indices: input, name: "embed")
        // h: [1, 1, D]

        // Transformer layers
        for i in 0..<config.layerCount {
            let prefix = "model.layers.\(i)"

            let n1W = try weight("\(prefix).input_layernorm.weight", shape: [D])
            let qW = try weight("\(prefix).self_attn.q_proj.weight", shape: [H * hd, D])
            let kW = try weight("\(prefix).self_attn.k_proj.weight", shape: [KVH * hd, D])
            let vW = try weight("\(prefix).self_attn.v_proj.weight", shape: [KVH * hd, D])
            let oW = try weight("\(prefix).self_attn.o_proj.weight", shape: [D, H * hd])
            let n2W = try weight("\(prefix).post_attention_layernorm.weight", shape: [D])
            let gateW = try weight("\(prefix).mlp.gate_proj.weight", shape: [I, D])
            let upW = try weight("\(prefix).mlp.up_proj.weight", shape: [I, D])
            let downW = try weight("\(prefix).mlp.down_proj.weight", shape: [D, I])

            // Attention sublayer
            let norm1 = Self.rmsNorm(graph: graph, x: h, weight: n1W, name: "l\(i).an")

            let q = Self.linear(graph: graph, x: norm1, weight: qW, name: "l\(i).q")
            let k = Self.linear(graph: graph, x: norm1, weight: kW, name: "l\(i).k")
            let v = Self.linear(graph: graph, x: norm1, weight: vW, name: "l\(i).v")

            let qH = Self.reshapeToHeads(graph: graph, x: q, heads: H, headDim: hd, name: "l\(i).q")
            let kH = Self.reshapeToHeads(graph: graph, x: k, heads: KVH, headDim: hd, name: "l\(i).k")
            let vH = Self.reshapeToHeads(graph: graph, x: v, heads: KVH, headDim: hd, name: "l\(i).v")

            let attnOut = Self.sdpa(graph: graph, query: qH, key: kH, value: vH,
                                     headDim: hd, name: "l\(i)")
            let flat = graph.reshape(
                graph.transposeTensor(attnOut, dimension: 1, withDimension: 2, name: nil),
                shape: [1, 1, (H * hd) as NSNumber], name: "l\(i).flat")
            let proj = Self.linear(graph: graph, x: flat, weight: oW, name: "l\(i).o")
            h = graph.addition(h, proj, name: "l\(i).attn.res")

            // MLP sublayer
            let norm2 = Self.rmsNorm(graph: graph, x: h, weight: n2W, name: "l\(i).mn")
            let gate = Self.linear(graph: graph, x: norm2, weight: gateW, name: "l\(i).gate")
            let up = Self.linear(graph: graph, x: norm2, weight: upW, name: "l\(i).up")
            let silu = graph.multiplication(
                gate, graph.sigmoid(with: gate, name: "l\(i).sig"), name: "l\(i).silu")
            let activated = graph.multiplication(silu, up, name: "l\(i).swiglu")
            let down = Self.linear(graph: graph, x: activated, weight: downW, name: "l\(i).down")
            h = graph.addition(h, down, name: "l\(i).mlp.res")
        }

        // Final norm
        let fnW = try weight("model.norm.weight", shape: [D])
        h = Self.rmsNorm(graph: graph, x: h, weight: fnW, name: "final")

        // LM head (tied to embedding)
        self.outputTensor = Self.linear(graph: graph, x: h, weight: embeddingTable, name: "lm_head")
    }

    /// Run a forward pass with token IDs.
    ///
    /// - Parameter tokenIDs: Int32 token IDs, shape [1, T].
    /// - Returns: Logits as MLXArray [1, T, vocab].
    public func forward(tokenIDs: [Int32]) -> MLXArray {
        let T = tokenIDs.count
        let inputData = tokenIDs.withUnsafeBufferPointer { ptr in
            MPSGraphTensorData(
                device: MPSGraphDevice(mtlDevice: device),
                data: Data(buffer: ptr),
                shape: [1, T as NSNumber],
                dataType: .int32)
        }

        let results = graph.run(
            with: commandQueue,
            feeds: [inputPlaceholder: inputData],
            targetTensors: [outputTensor],
            targetOperations: nil)

        let resultData = results[outputTensor]!
        let outputShape = resultData.shape.map { $0.intValue }
        let count = outputShape.reduce(1, *)
        var output = [Float16](repeating: 0, count: count)
        resultData.mpsndarray().readBytes(&output, strideBytes: nil)
        return MLXArray(output, outputShape)
    }

    // MARK: - Graph Ops

    private static func linear(
        graph: MPSGraph, x: MPSGraphTensor, weight: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        graph.matrixMultiplication(
            primary: x,
            secondary: graph.transposeTensor(weight, dimension: 0, withDimension: 1, name: nil),
            name: name)
    }

    private static func rmsNorm(
        graph: MPSGraph, x: MPSGraphTensor, weight: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        let eps = graph.constant(1e-5, dataType: .float16)
        let sq = graph.multiplication(x, x, name: "\(name).sq")
        let mean = graph.mean(of: sq, axes: [-1], name: "\(name).mean")
        let inv = graph.reverseSquareRoot(
            with: graph.addition(mean, eps, name: "\(name).eps"), name: "\(name).inv")
        return graph.multiplication(
            graph.multiplication(x, inv, name: "\(name).n"), weight, name: "\(name).out")
    }

    private static func reshapeToHeads(
        graph: MPSGraph, x: MPSGraphTensor, heads: Int, headDim: Int, name: String
    ) -> MPSGraphTensor {
        graph.transposeTensor(
            graph.reshape(x, shape: [1, 1, heads as NSNumber, headDim as NSNumber], name: nil),
            dimension: 1, withDimension: 2, name: "\(name).heads")
    }

    private static func sdpa(
        graph: MPSGraph, query: MPSGraphTensor, key: MPSGraphTensor,
        value: MPSGraphTensor, headDim: Int, name: String
    ) -> MPSGraphTensor {
        let scale = graph.constant(Double(1.0 / Float(headDim).squareRoot()), dataType: .float16)
        let scores = graph.multiplication(
            graph.matrixMultiplication(
                primary: query,
                secondary: graph.transposeTensor(key, dimension: 2, withDimension: 3, name: nil),
                name: "\(name).qk"),
            scale, name: "\(name).scaled")
        let w = graph.softMax(with: scores, axis: -1, name: "\(name).sm")
        return graph.matrixMultiplication(primary: w, secondary: value, name: "\(name).attn")
    }
}
