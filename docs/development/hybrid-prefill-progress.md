# Hybrid Prefill Fast Path Progress

Last updated: 2026-05-09

This file is the single progress ledger for the hybrid prefill fast-path work.
It tracks the current milestone, implementation status, validation evidence,
failed experiments, and next decisions.

## Goal

Move hybrid Qwen/LFM prefill from the current decode-equivalent sequence GEMV
path toward the final fastest design without losing production correctness.

```mermaid
flowchart TD
  A["Current decode-equivalent sequence GEMV"] --> B["M1 safe tiled sequence GEMV"]
  B --> C["M2 reference harness"]
  C --> D["M3 reference-equivalent MPP prefill"]
  D --> E["M4 fused hybrid block prefill"]
  E --> F["M5 benchmark-backed release claim"]
```

## Current Contract

| Area | Contract |
|---|---|
| Hybrid BF16 sequence prefill | Must match decode-equivalent prompt ingestion token trace |
| Q3 sequence prefill | Enabled only through reference-backed packed and batched Q3 sequence GEMV plus quantized sequence embedding |
| Speed claims | Forbidden unless the same path has green correctness evidence |
| Fallback | Silent fallback is not allowed |

## Milestone Status

| Milestone | Status | Notes |
|---|---|---|
| M1 safe tiled sequence GEMV | Done / not default | Tile4 kernels compile and pass decode-equivalence tests; Qwen profile regressed when defaulted, so production planner stays on base sequence GEMV |
| M2 reference harness | Done / hardened | LFM and Qwen3.5 both have reference dump scripts and manifest coverage; Qwen3.5 validates snapshot identity, prefill state, decode0 state, and KV cache |
| M3 reference-equivalent MPP prefill | Pending | Requires M2 before adopting non decode-equivalent math |
| M4 fused hybrid block prefill | Pending | Requires M2 and block-level reference probes |
| M5 benchmark-backed release claim | Pending | Requires real-bundle correctness gates first |

## M1 Design

The M1 kernel changes the scheduling unit but not the numeric unit:

```text
Numeric contract: one output row x one token keeps the same SIMD lane reduction.
Scheduling unit: one threadgroup may cover several tokens for the same row group.
```

| Parameter | Current | M1 target |
|---|---:|---:|
| Rows per threadgroup | 2 | 2 |
| Sequence tile | 1 | tile4 implemented as non-default experiment |
| Dot-product lane order | decode-equivalent | unchanged |
| Storage rounding | inside projection kernel | unchanged |
| MPP | not used for stateful hybrid sequence projection | unchanged |

## M2 Design

M2 changes the future correctness oracle from decode-only to reference-aware.
It does not yet authorize MPP or fused non-decode-equivalent kernels.

| Component | Status | Source |
|---|---|---|
| LFM reference dump script | Existing | `scripts/hf/dump_lfm2_reference.py` |
| LFM Swift reference comparison | Existing | `Tests/MetalCompilerTests/Models/LFM2/ReferenceComparisonTests.swift` |
| Qwen reference dump script | Done | `scripts/hf/dump_qwen35_reference.py`; schema v4 supports multiple reference cases plus version metadata |
| Qwen Swift reference comparison | Done | `Tests/MetalCompilerTests/Models/Qwen35/Qwen35ReferenceComparisonTests.swift`; validates every schema v4 case |
| Reference manifest | Done | `Tests/MetalCompilerTests/Core/ReferenceHarnessManifestTests.swift` |

## Confidence Audit

The current strategy is not treated as mathematically complete. It is treated as
release-safe only for the explicitly covered model, prompt, precision, and
optimizer path. Every known hole is either closed by a hard gate or listed as a
remaining scope limit.

```mermaid
flowchart TD
  A["HF reference dump"] --> B["Schema and identity gate"]
  B --> C["Metal sequence prefill gate"]
  C --> D["Prefill output/state/KV comparison"]
  D --> E["Decode0 output/state/KV comparison"]
  E --> F["Future M3/M4 kernel admission"]
```

| Potential hole | Status | Gate / decision |
|---|---|---|
| Reference snapshot generated from a different bundle | Closed | Snapshot stores `config.json` SHA-256 and the Swift test compares it with the local bundle |
| Swift prompt does not match reference prompt | Closed | Snapshot stores `ref.meta.input_tokens`; Swift test requires exact token equality |
| Sequence prefill silently falls back to sequential decode | Closed | Qwen reference test requires `requiresSequentialPromptIngestion == false` and no fallback reason |
| Only early linear states are checked | Closed | Qwen reference test checks every `linear_ordinal_*` conv and recurrent state |
| Decode post-state is not checked | Closed | Qwen reference test checks `decode_0` logits, next token, conv state, and recurrent state |
| KV cache corruption can pass | Closed | Qwen reference test checks prefill and `decode_0` key/value caches for all attention ordinals |
| Metal readback failure can look like a numeric match | Closed | Buffer readback now throws on allocation, command-buffer, or range failure; length mismatch produces infinite error |
| Single prompt / sequence length only | Closed | Qwen schema v4 stores multiple reference cases; Swift validates each case independently |
| Q3 sequence prefill | Closed for current Qwen3.5 Q3 path | Q3 G16/G32/G64 projection sequence GEMV and embedding lookup have CPU-reference tests; real Q3 prompt ingestion uses sequence prefill and matches sequential ingestion trace |
| HuggingFace backend variance | Closed for traceability | Qwen schema v4 records PyTorch version, Transformers version, and fast-backend availability |

## Evidence Log

| Date | Command / check | Result |
|---|---|---|
| 2026-05-07 | `swift build` | Pass before M1/M2 edits |
| 2026-05-07 | `MetalCompilerTests/Qwen35PromptIngestionTests` | Pass after reverting failed MPP gate/up experiment |
| 2026-05-07 | `MetalCompilerTests/Qwen35PrefillProfileTests` | Pass, current default remains sequence GEMV |
| 2026-05-07 | `swift build` | Pass after M1/M2 edits |
| 2026-05-07 | `MetalCompilerTests/SequenceProjectionEquivalenceTests` | Pass; includes base batched, tiled batched, tiled single sequence GEMV |
| 2026-05-07 | `MetalCompilerTests/ReferenceHarnessManifestTests` | Pass |
| 2026-05-07 | `MetalCompilerTests/GeneratedLibraryTests` | Pass |
| 2026-05-07 | `MetalCompilerTests/Qwen35PromptIngestionTests` with `ENABLE_METAL_PROBES=1` | Pass |
| 2026-05-07 | `MetalCompilerTests/Qwen35PrefillProfileTests` with `ENABLE_METAL_PROBES=1` | Pass; production planner uses base sequence GEMV |
| 2026-05-07 | `python3 scripts/hf/dump_qwen35_reference.py --output TestData/qwen35_reference.safetensors --decode-steps 2` | Pass; wrote schema v2 Qwen3.5 HF reference snapshot with config hash and input token metadata |
| 2026-05-07 | `MetalCompilerTests/Qwen35ReferenceComparisonTests` with `ENABLE_METAL_PROBES=1` | Pass; HF prefill/decode0 argmax, logits, final hidden, all linear states, and KV cache gates are green |
| 2026-05-07 | `MetalCompilerTests/Qwen35ReferenceComparisonTests` with `ENABLE_METAL_PROBES=1` | Pass after hardening Metal readback to throw instead of returning empty arrays |
| 2026-05-07 | `python3 scripts/hf/dump_qwen35_reference.py --output TestData/qwen35_reference.safetensors --decode-steps 2` | Pass; wrote schema v3 multi-case Qwen3.5 HF reference snapshot with 1328 tensors |
| 2026-05-07 | `MetalCompilerTests/Qwen35ReferenceComparisonTests` with `ENABLE_METAL_PROBES=1` | Pass; validates schema v3 case 0 and case 1 for prefill/decode0 logits, token trace, all linear states, and KV cache |
| 2026-05-07 | `python3 scripts/hf/dump_qwen35_reference.py --output TestData/qwen35_reference.safetensors --decode-steps 2` | Pass; wrote schema v4 multi-case Qwen3.5 HF reference snapshot with version and backend metadata |
| 2026-05-07 | `MetalCompilerTests/Qwen35ReferenceComparisonTests` with `ENABLE_METAL_PROBES=1` | Pass; validates schema v4 metadata plus both reference cases |
| 2026-05-07 | `MetalCompilerTests/QuantizationPlanningTests` | Pass; all Q3 prefill projection and embedding schemes require explicit sequential ingestion |
| 2026-05-07 | `MetalCompilerTests/Qwen35PromptIngestionTests` with `ENABLE_METAL_PROBES=1` | Pass; BF16 sequence path and Q3 explicit sequential ingestion trace gates are green |
| 2026-05-07 | `MetalCompilerTests/ReferenceHarnessManifestTests` | Pass after marking Qwen3.5 ready |
| 2026-05-07 | `swift build` | Pass after enabling Q3 sequence prefill |
| 2026-05-07 | `MetalCompilerTests/MetalSourceGeneratorTests` | Pass; Q3 G16/G32/G64 dequant→MPP and packed sequence GEMV match CPU references |
| 2026-05-07 | `MetalCompilerTests/QuantizedEmbeddingKernelTests` | Pass; Q3 G16/G32/G64 sequence embedding lookup matches affine CPU reference |
| 2026-05-07 | `MetalCompilerTests/QuantizationPlanningTests` | Pass; Q3 prefill projection and embedding schemes no longer require sequential ingestion |
| 2026-05-07 | `MetalCompilerTests/Qwen35PromptIngestionTests` with `ENABLE_METAL_PROBES=1` | Pass; Q3 real bundle sequence prefill trace matches decode-equivalent sequential ingestion |
| 2026-05-07 | `MetalCompilerTests/Qwen35BenchmarkTests` with `ENABLE_METAL_PROBES=1` | Pass; Q3 smoke benchmark confirms 186 packed Q3 sequence GEMV kernels and sequence prefill faster than sequential ingestion for lengths 16/64/128 |
| 2026-05-07 | `MetalCompilerTests/MetalSourceGeneratorTests` | Pass; batched Q3 sequence GEMV for counts 2/3/4 and groups 16/32/64 matches decode-rounded CPU references |
| 2026-05-07 | `MetalCompilerTests/Qwen35PromptIngestionTests` with `ENABLE_METAL_PROBES=1` | Pass; Q3 real bundle sequence prefill remains trace-equivalent after batched Q3 projection routing |
| 2026-05-07 | `MetalCompilerTests/Qwen35BenchmarkTests` with `ENABLE_METAL_PROBES=1` | Pass; Q3 smoke benchmark confirms 96 total Q3 sequence GEMV entries, 48 batched Q3 sequence GEMV entries, and sequence prefill faster than sequential ingestion for lengths 16/64/128 |
| 2026-05-09 | `MetalCompilerTests/Qwen35PrefillProfileTests` with `ENABLE_METAL_PROBES=1` (re-run) | Pass; projection share confirmed at 75.8-76.4%, single `gemv_seq_bf16_f32s` confirmed as 26.5% of projection time (48 dependent output projections), pass count remains 1 |
| 2026-05-09 | `MetalCompilerTests/SequenceProjectionEquivalenceTests` | Pass; 5 tests including `BF16 tile2 single sequence GEMV matches repeated decode GEMV` and `BF16 tiled single sequence GEMV matches repeated decode GEMV` |
| 2026-05-09 | `MetalCompilerTests/Qwen35ReferenceComparisonTests` with `SWIFTLM_PREFILL_BF16_SINGLE_TILE2=1` | Pass; 4/4 cases remain reference-equivalent with feature-flagged tile2 single sequence GEMV routing |
| 2026-05-09 | `MetalCompilerTests/Qwen35PrefillProfileTests` with `SWIFTLM_PREFILL_BF16_SINGLE_TILE2=1` | Pass; tile2 routed 48 BF16 single sequence GEMV dispatches but did not improve prefill timing, so default routing remains disabled |

## Failed Experiments

| Experiment | Result | Decision |
|---|---|---|
| Sequence GEMV `tileElements = 1024` | Regressed Qwen profile | Rejected |
| 4 / 8 output rows per threadgroup | Regressed or unstable Qwen profile | Rejected |
| MPP for 2-way gate/up while keeping stateful projections sequence GEMV | Broke BF16 Qwen token trace | Rejected |
| Default tile4 sequence GEMV | Correctness passed but Qwen seqLen 16/64/128 regressed to 114.827/190.303/374.976 ms | Kept as non-default experiment; production planner reverted to base sequence GEMV |
| Feature-flagged tile2 single sequence GEMV | Correctness passed, but Qwen seqLen 16/64/128 changed from 44.572/158.280/314.697 ms to 44.768/160.643/317.498 ms | Kept behind `SWIFTLM_PREFILL_BF16_SINGLE_TILE2=1`; default production planner stays on base sequence GEMV |

## Open Decisions

| Decision | Status |
|---|---|
| Best sequence tile size for M1 | tile4 is not acceptable as default; smaller or different tiling remains future work |
| Whether M1 should default to tile4 or stay opt-in | Stay non-default |
| Whether M2 should route tile2 for dependent single projections | Stay non-default; correctness passed but Qwen3.5 BF16 profile was noise/slower |
| Qwen reference dump schema for M2 | Implemented and validated as schema v4 multi-case |
| Q3 sequence prefill support | Implemented for current packed and batched Q3 projection paths plus Q3 embedding lookup |

## Current Production Prefill Profile

Latest focused Qwen profile (2026-05-09 re-run, two runs consistent within run-to-run noise),
with production planner using base decode-equivalent sequence GEMV:

| Sequence length | Total prefill time | Steps | Pass count |
|---:|---:|---:|---:|
| 16 | 43.333 ms | 293 | 1 |
| 64 | 159.418 ms | 293 | 1 |
| 128 | 311.291 ms | 293 | 1 |

Category share at seqLen=128:

| Category | Steps | Time | Share |
|---|---:|---:|---:|
| `projection` | 97 | 236.054 ms | 75.8% |
| `ssm_recurrence` | 18 | 68.859 ms | 22.1% |
| `attention` | 6 | 3.841 ms | 1.2% |
| `other` | 129 | 2.260 ms | 0.7% |
| remaining | 43 | 0.354 ms | 0.1% |

Kernel families confirmed in the plan:

| Kernel | Count |
|---|---:|
| `gemv_seq_bf16_f32s` | 48 |
| `batched_gemv2_seq_bf16_f32s` | 24 |
| `batched_gemv4_seq_bf16_f32s` | 18 |
| `batched_gemv3_seq_bf16_f32s` | 6 |
| `gemv_bf16_f32s` (output head) | 1 |

## Projection Bottleneck Breakdown (seqLen 128, 2026-05-09)

Aggregated from `.test-artifacts/prefill-profile/qwen35-prefill-steps-seq128.csv`.
Average per-dispatch time at seqLen 128 reveals which projection family dominates.

| Kernel | Role | n | avg µs | total µs | per-output µs | tg |
|---|---|---:|---:|---:|---:|---:|
| `batched_gemv2_seq_bf16_f32s` | MLP gate+up batched (gridWidth 3584) | 24 | 3657.6 | 87782.0 | 1.02 | 64 |
| `batched_gemv4_seq_bf16_f32s` | SSM in_proj batched (gridWidth 4112) | 18 | 4224.7 | 76044.4 | 1.03 | 64 |
| `gemv_seq_bf16_f32s` | dependent output projections (gridWidth 512) | 48 | 1360.2 | 65288.6 | 2.66 | 64 |
| `batched_gemv3_seq_bf16_f32s` | Attention Q+K+V batched (gridWidth 2560) | 6 | 2595.7 | 15574.4 | 1.01 | 64 |
| `gemv_bf16_f32s` | last-token output head (gridWidth 62080) | 1 | 1618.4 | 1618.4 | 0.03 | 128 |

Per-output time is `avg µs / gridWidth` and is a memory-bandwidth efficiency proxy.
The single-projection kernel is 2.6x less efficient per output element than the
batched variants, so input-staging amortization across tokens is the next lever
for the dependent-projection path.

### Single-projection entries are 100% dependent (no missed batching)

The 48 `gemv_seq_bf16_f32s` entries decompose into output projections that
cannot be batched with sibling projections because they consume an
intermediate activation produced by their own block:

| Projection role | Layers | Notes |
|---|---:|---|
| `mlp.down_proj` | 24 | Reads SwiGLU activation (intermediate dim 1792), writes hidden 2048 |
| `linear_attn.out_proj` | 18 | Reads SSM recurrence output, writes hidden 2048 |
| `self_attn.o_proj` | 6 | Reads attention output, writes hidden 2048 |

This count (24+18+6=48) and the entry pattern (every block ends with a
single output projection) match the per-entry profile data, so M1 missed
batching is exhausted: the planner already groups every batchable sibling
projection (gate+up, Q+K+V, SSM in_proj quartet) into a single dispatch.
Single output projections remain because their input is intermediate-only.

The next prefill speed lever is therefore **per-token amortization inside the
single-projection kernel** (M2 territory), not additional batching.

## Current Q3 Prefill Smoke

Latest focused Qwen3.5 Q3 smoke benchmark compares the enabled sequence prefill
path against the old sequential prompt-ingestion path in the same test run.
The run was intentionally short and should be treated as a regression smoke,
not a release-grade throughput claim.

| Sequence length | Sequence prefill | Sequential ingestion | Speedup |
|---:|---:|---:|---:|
| 16 | 206.76 ms | 322.11 ms | 1.56x |
| 64 | 769.38 ms | 1343.48 ms | 1.75x |
| 128 | 1510.55 ms | 2829.22 ms | 1.87x |

Kernel families confirmed in the Q3 plan:

| Kernel | Count |
|---|---:|
| `batched_gemv2_seq_q3_g64_f32s` | 24 |
| `batched_gemv3_seq_q3_g64_f32s` | 6 |
| `batched_gemv4_seq_q3_g64_f32s` | 18 |
| `gemv_seq_q3_g64_f32s` | 48 |

The remaining single Q3 sequence GEMV entries are dependent projections, not
missed sibling batches:

| Projection role | Count |
|---|---:|
| `linear_attn.out_proj` | 18 |
| `mlp.down_proj` | 24 |
| `self_attn.o_proj` | 6 |

This closes the current BatchedProjection routing gap for Q3 sequence prefill.
The next Q3 speed target is the single-projection kernel/fusion path rather
than additional sibling batching.

A Q3 single sequence tile4 kernel was added and reference-tested, then tried as
the production routing for the single projection entries. It stayed
decode-equivalent but did not improve the Qwen3.5 Q3 smoke profile:

| Sequence length | Base sequence prefill | Tile4 sequence prefill | Decision |
|---:|---:|---:|---|
| 16 | 206.76 ms | 207.48 ms | keep base |
| 64 | 769.38 ms | 773.49 ms | keep base |
| 128 | 1510.55 ms | 1523.47 ms | keep base |

The tile4 kernel remains in the generator as a tested experiment, but planner
routing stays on the base Q3 sequence GEMV path.

## Prefill Profile Harness

The profiling path is now shared instead of Qwen-specific. `MetalPrefillProfileHarness`
records both step-level and pass-level prefill timing with:

| Field group | Contents |
|---|---|
| Timing | GPU feedback time and wall-clock submit+wait time |
| Dispatch shape | grid, threadgroup size, threadgroup memory, estimated dispatch count |
| Routing | kernel name, category, mode, layer index, entry index, weight tensor name |
| Binding size | buffer binding count, inline constant bytes, unique bound buffer bytes |
| Artifacts | JSON and CSV under `.test-artifacts/prefill-profile/` |

The focused Qwen profile test writes one step profile and one pass profile per
sequence length. The artifacts are diagnostic evidence only; performance claims
still require a correctness-green run for the same model path.

## M1 Outcome

Tile4 sequence GEMV is now available and tested, but it is not selected by the
production planner. The profile showed it reduces grid height but loses enough
occupancy/locality that Qwen prefill regresses. This keeps the codebase honest:
correctness assets are retained, but runtime does not claim or ship a slower
path.

## M2 Status (2026-05-09)

Phase 1 + Phase 2 of the prefill speed plan are confirmed:

| Phase | Confirmed | Evidence |
|---|---|---|
| Phase 1 profile re-run | Yes | `Qwen35PrefillProfileTests` 2 runs, projection 75.8-76.4% |
| Phase 2 dependent-projection judgment | Yes | All 48 single GEMV entries are output projections (24 down + 18 ssm_out + 6 o) |
| Phase 4 reference equivalence (BF16 single tile2) | Yes | `BF16 tile2 single sequence GEMV matches repeated decode GEMV` passes |
| Phase 4 reference equivalence (BF16 single tile4) | Yes | `BF16 tiled single sequence GEMV matches repeated decode GEMV` passes |
| Phase 5 routing decision (tile2 / tile4) | Yes | tile4 already rejected at 2026-05-07; tile2 stays feature-flagged because real-shape Qwen3.5 profile was noise/slower |

The base sequence GEMV kernel uses one threadgroup per `(rowGroup, token)` pair
with `tileElements = 256` input staging. The tile2 kernel halves grid height by
covering 2 tokens per threadgroup, sharing input staging across the two tokens.
It is implemented and reference-equivalent, but the real Qwen3.5 dependent
projection shapes did not benefit.

| Sequence length | Baseline | Tile2 flag enabled | Delta | Decision |
|---:|---:|---:|---:|---|
| 16 | 44.572 ms | 44.768 ms | +0.44% | noise |
| 64 | 158.280 ms | 160.643 ms | +1.49% | slower |
| 128 | 314.697 ms | 317.498 ms | +0.89% | noise |

The planner keeps tile2 behind `SWIFTLM_PREFILL_BF16_SINGLE_TILE2=1`. The
structural `SequenceGEMVKernelSelection` refactor remains because it makes
future tiled experiments derive grid height, threadgroup shape, diagnostics,
and sequence-length policy from one explicit tile descriptor instead of parsing
kernel-name suffixes. Production default routing stays on the base sequence
GEMV path and no speed claim is made for tile2.
