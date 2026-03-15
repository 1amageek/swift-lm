@preconcurrency import MLX
import MLXFast
import SwiftLM

/// Lowered short convolution module for LFM2 family with compile-time kernel selection.
///
/// Implements double-gated LIV depthwise convolution:
///   in_proj(D -> 3D) -> chunk(B, C, x) -> B*x -> depthwise_conv1d -> C*conv_out -> out_proj
///
/// ## Design
///
/// ShortConv is significantly simpler than DeltaNet:
/// - 2 projections (in_proj, out_proj) vs 5 (QKV, Z, B, A, out)
/// - No recurrent state — conv cache only
/// - Plain element-wise gating (no sigmoid/softplus)
/// - No dtype boundary management (no recurrence accumulation)
///
/// ## Compile-time optimizations
///
/// - `inProj` and `outProj` use `LoweredProjection` for compile-time
///   kernel selection (matmul vs quantizedMatmul)
/// - Cache state is externalized as `LoweredCacheState.recurrent`
///   for functional-style inference
///
/// Reference: `Lfm2ShortConv` in HuggingFace transformers.
public struct LoweredShortConv: @unchecked Sendable {

    /// Input projection (D -> 3D) with compile-time resolved kernel.
    public let inProj: LoweredProjection

    /// Output projection (D -> D) with compile-time resolved kernel.
    public let outProj: LoweredProjection

    /// Depthwise conv1d weight: [D, K] for SSM conv kernel (squeezed from [D, 1, K]).
    public let convWeight: MLXArray

    /// SSM conv Metal kernel — specialized dot product for short convolution.
    /// Much faster than generic conv1d for small kernel sizes (K=3..4).
    private let ssmConvKernel: MLXFast.MLXFastKernel

    /// Short convolution attributes (hiddenSize, kernelSize).
    public let attrs: ShortConvAttributes

    /// Compile-time resolved cache slot index.
    public let cacheSlotIndex: Int

    public init(
        inProj: LoweredProjection,
        outProj: LoweredProjection,
        convWeight rawWeight: MLXArray,
        attrs: ShortConvAttributes,
        cacheSlotIndex: Int
    ) {
        self.inProj = inProj
        self.outProj = outProj

        // Pre-compute weight layout at compile time.
        // Raw weight from safetensors: [D, 1, K]
        // SSM conv kernel uses [D, K] — squeezed for direct dot product.
        if rawWeight.ndim == 3 {
            self.convWeight = rawWeight.squeezed(axis: 1)  // [D, K]
        } else {
            self.convWeight = rawWeight  // already [D, K]
        }

        // Create SSM conv Metal kernel — llama.cpp-style dot product per channel.
        // Input: padded [B, K+T-1, D] (channels-last), Weight: [D, K]
        // Output: [B, T, D] where each output[b,t,d] = sum_{k=0}^{K-1} input[b,t+k,d] * weight[d,k]
        self.ssmConvKernel = MLXFast.metalKernel(
            name: "ssm_conv",
            inputNames: ["input", "weight"],
            outputNames: ["output"],
            source: """
                // SSM conv: depthwise causal convolution as dot product per channel.
                // grid: (D, T, B)
                uint d = thread_position_in_grid.x;  // channel index
                uint t = thread_position_in_grid.y;  // output token index
                uint b = thread_position_in_grid.z;  // batch index

                uint D = output_shape[2];  // hidden size
                uint T = output_shape[1];  // output sequence length
                uint K = weight_shape[1];  // kernel size

                float sum = 0.0f;
                for (uint k = 0; k < K; k++) {
                    // input[b, t + k, d] * weight[d, k]
                    uint in_idx = b * input_strides[0] + (t + k) * input_strides[1] + d;
                    uint w_idx = d * weight_strides[0] + k;
                    sum += float(input[in_idx]) * float(weight[w_idx]);
                }

                uint out_idx = b * output_strides[0] + t * output_strides[1] + d;
                output[out_idx] = sum;
            """,
            ensureRowContiguous: true
        )

        self.attrs = attrs
        self.cacheSlotIndex = cacheSlotIndex
    }

    /// Apply short convolution with external cache state.
    public func apply(_ x: MLXArray, caches: inout [LoweredCacheState]) -> MLXArray {
        let D = attrs.hiddenSize
        let K = attrs.kernelSize
        let B = x.dim(0)

        // in_proj: [B, T, D] -> [B, T, 3D] — compile-time kernel selection
        let bcx = inProj.apply(x)

        // Split into B-gate, C-gate, x-input (single op, zero-copy)
        let parts = bcx.split(parts: 3, axis: -1)
        let bGate = parts[0]
        let cGate = parts[1]
        let xInput = parts[2]

        // First gate: element-wise multiply (plain, not sigmoid)
        var bx = bGate * xInput  // [B, T, D]

        // Extract cache state or initialize with zeros
        var state: MLXArray?
        var cacheOffset: Int = 0
        switch caches[cacheSlotIndex] {
        case .recurrent(let cache):
            state = cache.convState
            cacheOffset = cache.offset
        default:
            break
        }
        if state == nil {
            state = MLXArray.zeros([B, K - 1, D], dtype: bx.dtype)
        }

        // Concatenate cache + input (zero-copy views)
        bx = concatenated([state!, bx], axis: -2)

        // Update cache: last K-1 timesteps
        // Cache stores K-1 elements. After concat with T new tokens, total length = (K-1) + T.
        // We need the last K-1 from that, so start = T.
        // For decode (T=1): start = 1, for prefill: start = T.
        let newConvState = bx[0..., x.dim(1)..., 0...]

        // SSM conv via custom Metal kernel — replaces generic conv1d(groups: D)
        let T = x.dim(1)
        let convOut = ssmConvKernel(
            [bx, convWeight],
            grid: (D, T, B),
            threadGroup: (min(D, 256), 1, 1),
            outputShapes: [[B, T, D]],
            outputDTypes: [bx.dtype]
        )[0]

        // Store updated cache
        caches[cacheSlotIndex] = .recurrent(LoweredRecurrentCache(
            convState: newConvState,
            recurrentState: nil,
            offset: cacheOffset + x.dim(1)
        ))

        // Second gate: element-wise multiply (plain, not sigmoid)
        let y = cGate * convOut

        // out_proj: [B, T, D] -> [B, T, D] — compile-time kernel selection
        return outProj.apply(y)
    }
}
