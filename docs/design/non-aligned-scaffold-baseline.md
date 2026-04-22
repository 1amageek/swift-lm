# Non-aligned scaffold baseline (2026-04-22)

Baseline captured before optimizing the unified non-aligned GEMV scaffold,
plus post-optimization numbers for the same session.

## Gemma4-E2B, 3-run variance (cool-start within session, runs back-to-back)

### PREFILL throughput (tok/s, median of 3)

| Length | BF16 | Q4g64 | Q6g32 |
|---|---|---|---|
| 16 | 335.5 | 143.2 | 201.4 |
| 32 | 638.6 | 148.2 | 402.8 |
| 64 | 1232.7 | 151.2 | 770.4 |
| 128 | 1534.6 | 150.4 | 1127.4 |

### DECODE throughput (100 steps, median of 3)

| | BF16 | Q4g64 | Q6g32 |
|---|---|---|---|
| tok/s | 28.9 | **37.7** | **8.5** |
| ms/tok | 34.61 | 26.52 | **116.98** |

### Fusion counts (from endToEndBenchmark)

| | Unfused | Fused | Reduction |
|---|---|---|---|
| BF16 | 789 | 474 | 40.0% |
| Q4g64 | 789 | 539 | 31.7% |
| Q6g32 | 789 | 539 | 31.7% |

### Decode memory bandwidth (endToEndBenchmark, 1× weight read per token)

| | Bandwidth |
|---|---|
| BF16 | 617.0 GB/s (2× multiplier: 16-bit weights) |
| Q4g64 | 147.8 GB/s |
| Q6g32 | **35.1 GB/s** |

Apple Silicon M-series theoretical bandwidth: ~800 GB/s.
Q6 at 35 GB/s uses 4.4% of theoretical — compute-bound in bit-unpacking, not memory-bound.

## Observed bottleneck

Non-aligned scaffold (shared by Q3 / Q5 / Q6) uses a generic group-dequant helper
emitted per `QuantizationFormat.emitGroupDequant`. Q4/Q8 (aligned) have hand-tuned
direct kernels and run at 147–617 GB/s. The gap is kernel efficiency, not fusion.

## Target

Close the non-aligned → aligned gap on decode GEMV. Improvement to the scaffold
benefits Q3/Q5/Q6 simultaneously (no format-specific code).

## Change

Non-aligned formats (Q3/Q5/Q6) now expose `perWeightReadExpression` via a ternary
chain of MLX-compatible bit-unpacking patterns. The unified GEMV scaffold prefers
`perWeightReadExpression` over `emitGroupDequant`, so every non-aligned format
reaches the same work-efficient simdgroup path that aligned formats use:

- Before: 32 simdgroup threads each dequantized the full group into a 128 B
  thread-local `weights_f32[]`, then strided-read their slice (32× redundant
  dequant + stack spill).
- After: each thread dequantizes exactly the weight it multiplies, inline
  (no stack array, no redundant work).

The embedding lookup kernel was updated in the same way ("prefer perWeight,
fall back to group"). `emitGroupDequant` is retained as a fallback for any
future format that cannot express per-weight dequant in closed form.

## Result — Gemma4-E2B Q6 (same session, back-to-back)

### PREFILL throughput (tok/s, median of 3)

| Length | Before | After | Δ |
|---|---|---|---|
| 16 | 201.4 | 232.4 | +15.4% |
| 32 | 402.8 | 470.3 | +16.8% |
| 64 | 770.4 | 866.0 | +12.4% |
| 128 | 1127.4 | 1222.3 | +8.4% |

### DECODE throughput (100 steps, median of 3)

| | Before | After | Δ |
|---|---|---|---|
| tok/s | 8.5 | **19.6** | **+131%** |
| ms/tok | 116.98 | **50.92** | **-56%** |

### Decode memory bandwidth (endToEndBenchmark)

| | Before | After | Δ |
|---|---|---|---|
| Q6 | 35.1 GB/s | **84.3 GB/s** | **+140%** |

### Correctness

- `UnifiedGEMVBitLevelTests` (13 tests) — all pass (Q3G16/G32, Q5G32/G64, Q6G16/G32 included)
- `UnifiedGEMVMultiBlockTests` / `UnifiedGEMVMultiRowTests` / `STAFEndToEndGEMVTests` (39 total) — all pass
- `Gemma4Q6AgreementTests` — Q6 generates 23–29 unique tokens per 31 decode steps (healthy diversity, no argmax collapse)

### Bottleneck after the change

Q6 decode is still compute-bound (84 GB/s vs ~800 GB/s theoretical, ~10.5%
utilization). The scaffold is no longer doing 32× redundant work, but the
ternary-chain dequant is divergent across simdgroup threads — threads with
different `slot` values take different branches. Further improvement requires
either (a) a non-aligned-aware packing that keeps each simdgroup lane on a
single bit pattern, or (b) merging per-thread dequant work with the input-tile
multiply so arithmetic hides the unpacking latency. Q4's 147 GB/s remains the
next target.
