# Metal 4 Integration Design

## Background

### Current hybrid prefill status

Hybrid models with convolution or recurrent state are correctness-gated before
they can use sequence prefill. `MetalPrefillPlan.requiresSequentialPromptIngestion`
is no longer driven by state-buffer presence alone. The current correctness gate
is trace based: BF16 LFM short-convolution plans and BF16 Qwen DeltaNet/SSM plans
can use sequence prefill only while focused tests show the same first token and
short decode trace as decode-equivalent token-by-token ingestion.

Q3 prefill projection and embedding lookup also remain an explicit unsupported
sequence-prefill case. BF16 `conv1d_causal_seq` is enabled for LFM-style
short-convolution plans after matching decode-equivalent short traces. BF16
Qwen DeltaNet/SSM prompt ingestion is covered by `Qwen35PromptIngestionTests`
with `ENABLE_METAL_PROBES=1`.

Before any Metal 4 command-buffer reuse or MPP prefill work is treated as a
performance win for hybrid models, the stateful sequence path must produce the
same first token and short decode trace as `prefillPlan = nil` sequential
ingestion on the same prompt for the model family being claimed.

### Current decode submission status

Decode submission now reuses the resident `MTL4ArgumentTable` by default for
synchronous decode submissions. Command buffers are still created per
submission; replayable command-buffer reuse is intentionally not claimed until
prompt-state restore, real-bundle decode, and asynchronous submission
correctness are proven together.

| Item | Current status | Evidence / guard |
|---|---|---|
| Argument table reuse | Enabled for synchronous submissions | Falls back to a fresh table when `waitUntilCompleted == false` |
| Fresh-table diagnostic path | Available | `SWIFTLM_METAL_FRESH_ARGUMENT_TABLE=1` or `SWIFTLM_METAL_FRESH_SUBMISSION=1` |
| Command buffer replay | Pending | Not enabled; correctness risk remains higher than the measured benefit |

### Current quantized decode batching status

Quantized decode no longer decomposes every `BatchedProjection` back into
individual `LinearFragment` entries just because a sibling projection uses a
block-quantized weight format. The compiler can now emit batched quantized GEMV
kernels such as `batched_gemv3_q4_g64`, preserving Q/K/V and gate/up dispatch
reduction for decode.

This is kernel/planner support, not a broad model-quality claim. Synthetic
kernel tests cover Q3/Q4 batched GEMV correctness and planning tests verify that
Q4 sibling projections remain batched. Qwen3.5 quantized agreement remains a
separate real-bundle quality gate and must stay marked unsupported until that
suite is green.

| Path | Current status | Release claim |
|---|---|---|
| Dense BF16/FP16 decode batching | Supported | Yes, where real-bundle gates pass |
| Quantized batched decode kernels | Implemented for generic 2-4 projection GEMV | Kernel/planner evidence only |
| Qwen3.5 Q3/Q4 real-bundle decode quality | Still not release-ready | No |

### Profile Data (LFM2.5-1.2B, current implementation)

```
Decode (single token):  ~110 tok/s, 8.4 ms/tok
  GEMV:      94.1% (7889 us) — memory bandwidth bound (~516 GB/s)
  FlashAttn:  1.2%
  Fused Norm: 2.2%
  Other:      2.5%

Prefill (64 tokens):  historical snapshot; re-benchmark required
  GEMM: routes through Metal 4 MPP matmul2d where eligible, with direct
        quantized fallbacks for unsupported layouts.
```

### Key Constraints

- **Decode GEMV is at memory bandwidth ceiling.** Kernel-level optimization (threadgroup cache, vectorized loads) showed no improvement or regression. Apple Silicon L2 cache already handles input vector reuse.
- **Prefill GEMM uses Metal 4 MPP where eligible.** Dense BF16/FP16/FP32 and compatible
  dequantized quantized projections route through `matmul2d`; unsupported strides or
  schemes use explicit direct kernels/fallback reasons.
- **Short sequence tile selection matters.** MPP GEMM kernels are emitted with
  `_mtile16`, `_mtile32`, and `_mtile64` variants, and runtime dispatch selects
  the smallest available tile that covers the actual prompt length. `_mtile128`
  is intentionally not in the default set until benchmark data proves it helps.
- **Hybrid decode-equivalent prefill is not always MPP.** Qwen-style
  conv/recurrent sequence prefill intentionally keeps dense BF16/FP16
  projections on decode-equivalent sequence GEMV kernels so state seeding stays
  trace-gated. The batched sequence GEMV path now processes two output rows per
  threadgroup, preserving per-row SIMD reduction order while reducing grid
  width for Q/K/V and gate/up projections. These sequence GEMV kernels and
  synthesized fused sequence kernels write decode-rounded storage values
  directly, so the planner does not emit an immediate `round_bf16_seq_f32` /
  `round_f16_seq_f32` pass after them. A larger 1024-element sequence GEMV tile
  was tested and rejected because it regressed the focused Qwen prefill profile.
- **Metal 4 matmul2d** uses Apple's internal optimized paths (likely AMX). Supports BF16×BF16→float natively.

## Strategy: Metal 4 Prefill GEMM

The current implementation replaces eligible prefill `generateGEMM` dispatches
with Metal 4 `matmul2d_descriptor` + `tensor_inline`. This section remains the
design reference for extending that path, not a statement that all prefill GEMM
uses MPP.

### Why Prefill GEMM Only

| Path | Bottleneck | Metal 4 Benefit |
|---|---|---|
| **Decode GEMV** | Memory bandwidth (N=1) | None — matmul2d requires N≥32 tile. GEMV is bandwidth-bound regardless of compute. |
| **Prefill GEMM** | Compute (naive kernel) | **High** — matmul2d uses Apple's optimized AMX tiling. seqLen provides the N dimension. |

### API: matmul2d_descriptor

```metal
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// C[M×N] = A[M×K] × B[K×N]  (+ C for accumulate mode)
constexpr auto desc = matmul2d_descriptor(
    M_tile,              // output rows per threadgroup
    N_tile,              // output cols per threadgroup
    dynamic_length_v<int>, // K: read from tensor extents at runtime
    transpose_A,         // false: A is [K, M] column-major
    transpose_B,         // false: B is [N, K] column-major
    relaxed_precision,   // false for correctness, true for speed
    matmul2d_descriptor::mode::multiply  // C = A×B (not accumulate)
);

matmul2d<desc, execution_simdgroups<4>> op;
op.run(sliceA, sliceB, sliceC);
```

### Supported Type Combinations (from MPPTensorOpsMatMul2d.h)

```
A (input)    B (weight)   C (output)
---------    ----------   ----------
bfloat       bfloat       float      ← BF16 model, F32 prefill buffer ✓
half         half         float      ← F16 model, F32 prefill buffer
half         half         half       ← F16 model, F16 decode buffer
bfloat       bfloat       bfloat     ← BF16 model, BF16 output
```

### Current vs Metal 4 GEMM

**Current (MetalSourceGenerator.generateGEMM):**
```metal
kernel void gemm_bf16_f32s(
    device const float* input,     // [seqLen × inputDim]
    device const uint16_t* weight, // [outputDim × inputDim]
    device float* output,          // [seqLen × outputDim]
    constant uint& inputDim, constant uint& outputDim, constant uint& seqLen,
    uint2 gid, uint tiisg, uint sgitg
) {
    // 1 row per simdgroup, 2 simdgroups per threadgroup
    const uint row = gid.x * 2 + sgitg;
    const uint seqPos = gid.y;
    float sum = 0.0f;
    for (uint j = tiisg; j < inputDim; j += SIMD_WIDTH)
        sum += bf16_to_float(weight[row * inputDim + j]) * input[seqPos * inputDim + j];
    sum = simd_sum(sum);
    if (tiisg == 0) output[seqPos * outputDim + row] = float(sum);
}
```

**Metal 4 (proposed):**
```metal
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

kernel void gemm_mpp(
    device bfloat* input,          // [seqLen × inputDim] — need bfloat type
    device bfloat* weight,         // [outputDim × inputDim]
    device float* output,          // [seqLen × outputDim]
    constant uint& inputDim,
    constant uint& outputDim,
    constant uint& seqLen,
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Wrap raw buffers as tensor_inline (zero-copy)
    auto A = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>(inputDim, seqLen));
    auto B = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(outputDim, inputDim));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        output, dextents<int32_t, 2>(outputDim, seqLen));

    constexpr auto desc = matmul2d_descriptor(
        64, 32, dynamic_length_v<int>,
        false, true,  // A: not transposed, B: transposed (row-major weight)
        false,
        matmul2d_descriptor::mode::multiply);

    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(mA, mB, mC);
}
```

## Architecture Changes

### 1. MetalSourceGenerator: Metal 4 GEMM Variant

```
MetalSourceGenerator
  ├── generateGEMM(...)          ← existing, Metal 3 fallback
  └── generateMPPGEMM(...)       ← new, Metal 4 matmul2d
```

The compiler selects based on Metal 4 availability at compile time.

### 2. Compilation: Metal Language Version

```swift
// Current
compileOptions.languageVersion = .version3_0

// Metal 4 path
compileOptions.languageVersion = .version4_0

// Need framework search path for MetalPerformancePrimitives
compileOptions.preprocessorMacros = ["USE_MPP": NSNumber(value: 1)]
```

**Critical**: `MetalPerformancePrimitives.h` is a framework header, not stdlib. The Metal compiler needs `-framework MetalPerformancePrimitives` or equivalent include path. The example repo pre-compiles to metallib via `xcrun metal` to avoid JIT issues.

### 3. Pre-compiled Metallib vs JIT

Current swift-lm compiles MSL source at runtime via `device.makeLibrary(source:options:)`. For Metal 4 MPP kernels, there are two options:

**Option A: JIT with framework include path**
```swift
let options = MTLCompileOptions()
options.languageVersion = .version4_0
// Need to set include path for MetalPerformancePrimitives
options.preprocessorMacros = ["__HAVE_TENSOR__": NSNumber(value: 1)]
let library = try device.makeLibrary(source: source, options: options)
```

**Option B: Pre-compiled metallib (recommended)**
```bash
xcrun metal -std=metal4.0 -framework MetalPerformancePrimitives -c shader.metal -o shader.air
xcrun metallib -o mpp_kernels.metallib shader.air
```
Ship `mpp_kernels.metallib` as a resource. Load at runtime via `device.makeLibrary(URL:)`.

**Recommendation**: Option B. Pre-compilation avoids runtime header resolution issues and ensures the Apple-optimized paths are properly compiled. The metallib can be generated as a build step.

### 4. Dispatch Configuration

```
// Metal 4 matmul2d with tile size 64×32, 4 simdgroups:
let simdWidth = pipeline.threadExecutionWidth  // 32
let threadsPerThreadgroup = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
let threadgroups = MTLSize(
    width: (outputDim + 31) / 32,   // N tiles
    height: (seqLen + 63) / 64,     // M tiles
    depth: 1
)
```

### 5. Buffer Type Compatibility

**Problem**: Current prefill buffers are `device float*` (F32). Metal 4 matmul2d wants `device bfloat*` for input (BF16 model) and `device float*` for output.

**Solution**: The input to GEMM in prefill is the hidden state (F32 after norm), not raw weight. The weight is BF16 (from STAF). So:
- Input (hidden/scratch): F32 — matmul2d supports `float × bfloat → float`
- Weight (STAF): BF16 raw bytes — cast to `device bfloat*`
- Output (scratch/hidden): F32

This maps to the supported type combination: **float × bfloat → float** ✓

### 6. Tensor Memory Layout

matmul2d expects column-major layout (innermost dimension first in `tensor_inline` extents):
- A `[M×K]`: extents = `(K, M)` where K is inner (column stride)
- B `[K×N]`: extents = `(N, K)` where N is inner
- C `[M×N]`: extents = `(N, M)` where N is inner

Current GEMM buffer layout:
- input: `[seqLen × inputDim]` — row-major, stride = inputDim
- weight: `[outputDim × inputDim]` — row-major, stride = inputDim
- output: `[seqLen × outputDim]` — row-major, stride = outputDim

For tensor_inline, row-major `[M × K]` with stride K = column-major with extents `(K, M)`. This matches.

## Module Structure

```
MetalCompiler/
  ├── Fragments/
  │   ├── MetalSourceGenerator.swift     ← add generateMPPGEMM()
  │   └── Primitives/
  │       └── LinearFragment.swift       ← add Metal 4 kernel name variant
  ├── Metal4/                            ← new directory
  │   ├── MPPKernelSource.swift          ← Metal 4 MSL source for MPP matmul
  │   └── Metal4Availability.swift       ← runtime Metal 4 detection
  └── MetalInferenceCompiler.swift       ← select Metal 3/4 path
```

## Scope

### Phase 1: Prefill GEMM with matmul2d
- Implemented for eligible dense and dequantized prefill projections.
- Runtime sequence length adjusts grid height, and MPP tile variants are preserved
  through barrier optimization, resident-constant conversion, and runtime isolation.
- Remaining work: model-level Qwen/LFM/Gemma benchmark refresh after correctness gates.

### Phase 2: MTL4CommandBuffer for Decode
- Reuse command buffer across decode steps (eliminate per-step allocation)
- Use MTL4CommandAllocator with frame rotation

### Phase 3: Fused lm_head + Argmax
- Single kernel: matmul2d tile → cooperative_tensor local argmax → atomic global argmax
- Eliminates 128KB logits buffer write+read

## Risks

1. **MetalPerformancePrimitives header availability**: JIT compilation may fail if headers aren't found at runtime. Pre-compiled metallib mitigates this.
2. **BF16 type (`bfloat`)**: MSL `bfloat` requires Metal 4.0 language version. Current `uint16_t + bf16_to_float()` approach doesn't work with tensor_inline.
3. **Tile size constraints**: matmul2d requires K to be multiple of 32 (per the example repo's bug report). `dynamic_length_v<int>` handles non-aligned K but may have performance cost.
4. **Minimum seqLen**: matmul2d tile M=64 means seqLen < 64 will use partial tiles. For short prompts (5 tokens), the overhead of tensor setup may negate the benefit.
