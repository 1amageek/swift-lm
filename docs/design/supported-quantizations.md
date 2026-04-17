# Supported Quantizations

This document defines the **concrete quantization support surface** of `swift-lm`
today: which weight formats and which KV cache schemes are actually validated
per model family, and how they compose.

The broader architectural design for quantization lives in
[`quantization.md`](quantization.md). This document is the ground-truth catalog
for what the repository currently supports, not what it aspires to.

## Two Orthogonal Axes

Quantization in `swift-lm` is two independent axes:

| Axis | Controlled by | Source of truth |
|---|---|---|
| **Weight format** | safetensors + STAF converter | `QuantizationSchemeIdentifier` in `Sources/MetalCompiler/STAF/STAFFormat.swift` |
| **KV cache scheme** | `InferencePolicy.kvCache` at load time | `KVCachePolicy` / `SchemeSelection` |

Weight format is determined by the model bundle on disk. KV cache scheme is a
deployment decision made by the consumer.

## Weight Formats

Enumerated by `QuantizationSchemeIdentifier` and mapped to concrete GEMV/GEMM
kernels via `QuantizationFormatRegistry`.

| Scheme | ID | bits | groupSize | Swift `WeightFormat` |
|---|---|---|---|---|
| `fp16RowMajor` | 0x00 | 16 | — | `.float16` |
| `bf16RowMajor` | 0x01 | 16 | — | `.bfloat16` |
| `fp32RowMajor` | 0x02 | 32 | — | `.float32` |
| `q8Group32ScaleF16` | 0x10 | 8 | 32 | `.quantized8Bit(32)` |
| `q8Group64ScaleF16` | 0x11 | 8 | 64 | `.quantized8Bit(64)` |
| `q4Group64ScaleF16` | 0x40 | 4 | 64 | `.quantized4Bit(64)` |
| `q4Group128ScaleF16` | 0x41 | 4 | 128 | `.quantized4Bit(128)` |

Declared identifiers without a registry mapping (e.g., `q6_*`, `q5_*`,
`q3_*`, `q2_*`, `q4Group128ScaleF16Zero`) are **not supported at runtime** —
`QuantizationFormatRegistry.format(for:)` returns nil and loading fails loudly.

## KV Cache Schemes

Declared via `KVCachePolicy.keyScheme` / `KVCachePolicy.valueScheme` on
`InferencePolicy`.

| Scheme | ID | Purpose | Memory vs FP16 |
|---|---|---|---|
| `fp16RowMajor` | 0x00 | Default dense KV cache | 100% |
| `bf16RowMajor` | 0x01 | Selected automatically when weights are BF16 | 100% |
| `rotorQ8Group32ScaleF16` | 0x70 | Clifford rotor + Q8 block quant | 62.5% |
| `rotorQ4Group64ScaleF16` | 0x71 | Clifford rotor + Q4 block quant | 37.5% |

`SchemeSelection.automatic` resolves to `bf16RowMajor` if the weights are
BF16, otherwise `fp16RowMajor`. The loader can additionally default to
`rotorQ4Group64ScaleF16` for pure-attention models without stateful sequence
operators — see `Region.isRotorQuantDefaultCandidate`.

## Per-Model Support Matrix

Support means the scheme has a passing **real-bundle correctness test** or
**real-bundle embedding evaluation** at HEAD. Untested combinations are
considered unsupported — even if the loader accepts them — because there is no
regression guard preventing silent correctness loss.

### Gemma4 (text decoder)

Reference bundle: `google/gemma-4-E2B-it` (TestData: `gemma-4-E2B-it`)

Weight formats:

| Format | Status | Evidence |
|---|---|---|
| FP16 | ✅ Validated | `RotorQuantRealBundleBaselineTests` — token "Tokyo" |
| BF16 | ⚠ Loader-supported only | No real-bundle test |
| Q4 / Q8 | ❌ Not tested | No published bundle converted to STAF |

KV cache schemes (FP16 weights, Gemma4-E2B):

| Scheme | Status | Evidence |
|---|---|---|
| FP16 / FP16 | ✅ Validated | `RotorQuantRealBundleBaselineTests` |
| RotorQ8-K + FP16-V | ✅ Validated | `RotorQuantRealBundleKeyPathTests` |
| FP16-K + RotorQ8-V | ✅ Validated | `RotorQuantRealBundleValuePathTests` |
| RotorQ8 / RotorQ8 | ✅ Validated | `RotorQuantRealBundleFullTests` (RotorQ8) |
| RotorQ4 / RotorQ4 | ✅ Validated | `RotorQuantRealBundleFullTests` (RotorQ4) |

### EmbeddingGemma (encoder-only)

Reference bundles:
- `mlx-community/embeddinggemma-300m-bf16`
- `mlx-community/embeddinggemma-300m-4bit`

Weight formats:

| Format | Status | Evidence |
|---|---|---|
| BF16 | ✅ Validated | `EmbeddingGemmaVariantCompatibilityTests` + `EmbeddingGemmaPerformanceTests` (66.2 emb/s) |
| Q4g64 | ✅ Validated | Same tests (54.0 emb/s) |

KV cache: N/A — embedding model has no generative KV cache.

### LFM2 (hybrid DeltaNet + attention)

Reference bundle: `LiquidAI/LFM2.5-1.2B-Thinking` (TestData: `LFM2.5-1.2B-Thinking`)

Weight formats:

| Format | Status | Evidence |
|---|---|---|
| BF16 | ✅ Validated | `LFMOutputDiagnosticsTests`, `ReleaseSmokeOutputTests` |
| FP16 / Q4 / Q8 | ❌ Not tested | No published bundle |

KV cache schemes:

| Scheme | Status | Evidence |
|---|---|---|
| FP16 / FP16 | ✅ Validated | Default — all LFM2 tests |
| Rotor* | ❌ Excluded by loader | `ShortConvAttributes` present → `containsStatefulSequenceState` → auto-rotor disabled |

Rotor is structurally incompatible with LFM2's stateful conv layers in the
current implementation. Attempting to fix a rotor scheme on LFM2 loads but is
untested for correctness.

### Qwen3.5 (VLM: vision + language)

Reference bundle: `Qwen/Qwen3.5-0.8B-Base`

Weight formats:

| Format | Status | Evidence |
|---|---|---|
| BF16 | ✅ Validated (text path) | `QwenVisionRealBundleTextTests` |
| BF16 | ✅ Validated (vision path) | `QwenVisionRealBundleImageTests`, `QwenVisionRealBundleVideoTests` |
| Q4 / Q8 | ❌ Not tested | No published bundle |

KV cache schemes: FP16 default only; hybrid DeltaNet + Attention layers make
rotor application partial at best. Not tested with rotor schemes.

## Roll-up

|   | FP16 | BF16 | Q4 | Q8 | Rotor KV |
|---|---|---|---|---|---|
| Gemma4 text | ✅ | ⚠ | — | — | ✅ (Q4 / Q8 both K, V, or full) |
| EmbeddingGemma | — | ✅ | ✅ | — | N/A |
| LFM2 | — | ✅ | — | — | ❌ loader-excluded |
| Qwen3.5 VLM | — | ✅ | — | — | ❌ not tested |

Legend: ✅ real-bundle test passes · ⚠ loader accepts but untested · ❌ unsupported · — no bundle

## Rules When Adding a New Scheme or Bundle

1. **Register the scheme identifier** in `QuantizationSchemeIdentifier` with
   its numeric ID.
2. **Provide a `QuantizationFormat` struct** with kernel names and block
   geometry.
3. **Wire it into `QuantizationFormatRegistry.format(for:)`** — otherwise
   loading fails with "unknown scheme".
4. **Add a real-bundle correctness test** before marking the scheme as
   supported in this document. A passing test is the only evidence counted.
5. **Do not silently fall back** to a different scheme if the requested one is
   unavailable. Loader errors must be explicit (per repo rule: no silent
   fallback).

## Rules When Dropping a Scheme

1. Remove the registry entry.
2. Remove the Swift struct.
3. Update this document (strike the row, do not delete history).
4. The STAF scheme ID must not be reused for a different semantic scheme — IDs
   are an append-only contract.
