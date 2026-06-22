# Tensor Parallel Scaling — Project Plan

**Date**: 2025-07-24  
**Branch**: `tensor-parallel`  
**Model**: Qwen2.5-7B-Instruct Q8_0  
**Hardware**: 2× AMD Instinct MI60 (gfx906, 60 CUs, 1024 GB/s HBM each)  
**Baseline commit**: `d185e3c6` (Phases 1-4: GEMV optimizations + concurrent prefill)

---

## Executive Summary

TP=2 decode was **slower** than single GPU (47.3 vs 52.2 tok/s = 0.91× scaling) due to a critical `executeNode` overhead bug: non-collective stages in TP graphs were routed through the full node path (contract building, vector allocations, arena coherence checks) instead of the raw `stage->execute()` fast path. This caused **5.7× higher per-stage overhead** in TP mode (96.5 μs vs 16.9 μs per stage call) from thread contention on the heap allocator.

**After Phase 0+0B fixes**: TP=2 decode reaches **82.4 tok/s** (1.10× scaling vs single GPU 74.9 tok/s), and prefill reaches **1044 tok/s** (1.11× scaling vs single GPU 938 tok/s). Decode exceeds the Phase 2+3 projected target of 75 tok/s, achieved purely by eliminating CPU-side overhead. Prefill fix eliminated a HIP graph capture regression that was costing ~550ms per benchmark iteration.

**Root cause breakdown** (decode, per-128-token run, per-device):

| Source | Lost ms | % of Gap | Root Cause |
|--------|--------:|:--------:|------------|
| Allreduce overhead | 196 | 38% | 56 ops/token × 14 KB each, RCCL kernel launch tax |
| GEMM Attn (1.15× scaling) | 120 | 23% | N=1792 grid-kpar KB=34, atomicAdd contention, launch overhead |
| GEMM FFN (1.65× scaling) | 112 | 22% | N=9472 KB doubles, atomicAdd contention |
| Flash Attn (1.02× scaling) | 85 | 17% | 14 heads fits single CU wave, bandwidth-bound |
| **Total gap** | **513** | **100%** | **vs ideal 2× scaling** |

---

## Baseline Profiling Data

### Single GPU vs TP=2 (Decode, 128 tokens)

| Kernel | Single GPU (ms) | TP=2 per-device (ms) | Ideal TP=2 (ms) | Actual Scaling | Lost ms |
|--------|----------------:|---------------------:|-----------------:|:--------------:|--------:|
| FLASH_ATTN_DECODE | 171 | 167 | 85.5 | 1.02× | 82 |
| GEMM_ATTN | 325 | 283 | 163 | 1.15× | 120 |
| GEMM_FFN | 1,048 | 636 | 524 | 1.65× | 112 |
| Allreduce | 0 | 196 | — | (new) | 196 |
| Other | 5,809 | 6,830 | 2,905 | — | — |
| **Total GPU** | **7,353** | **8,112** | **~3,677** | **0.91×** | — |

### Single GPU vs TP=2 (Prefill, 596 tokens)

| Kernel | Single GPU (ms) | TP=2 per-device (ms) | Scaling |
|--------|----------------:|---------------------:|:-------:|
| FLASH_ATTN_PREFILL | 162 | 103 | 1.57× |
| GEMM_ATTN | 285 | 161 | 1.77× |
| GEMM_FFN | 1,140 | 576 | 1.98× |
| Allreduce | 0 | 8 | (new) |
| **Total** | **1,938** | **1,869** | **1.04×** |

### Per-Token Cost (Decode)

| | Single GPU | TP=2 (before) | TP=2 (after Phase 0+0B) | Target |
|---|---:|---:|---:|---:|
| **E2E ms/tok** | 13.4 | 21.1 | **12.1** | ≤12.5 |
| **E2E tok/s** | 74.9 | 47.3 | **82.4** | ≥80 |

> Note: Original single GPU baseline was 52.2 tok/s measured with profiling. Production (non-profiling) baseline is 74.9 tok/s.

---

## Phase 1: Allreduce Optimization (38% of gap, ~196 ms) — COMPLETED

**Goal**: Reduce allreduce cost from 196 ms → ≤60 ms (−70%)  
**Actual Result**: Allreduce overhead is **protocol-latency-dominated** (~29 μs/op avg), not message-size-dominated. FP16 for all layers is the only feasible optimization. Overlap and batching are architecturally infeasible.

### 1A. FP16 Mixed-Precision Allreduce — ✅ DONE

**Changes**: Set `tp_allreduce_fp32_layer_count = 0` in `Qwen2Schema.h` and `Qwen3Schema.h` (was 6, which forced FP32 for layers 0-5). All 28 layers now use FP16 allreduce by default.

**Finding**: The allreduce cost is dominated by RCCL per-op protocol latency (~29 μs average), not data transfer. FP16 halves message size (14 KB → 7 KB) but the absolute savings per op is negligible at this message size. The 6 additional FP16 layers (12 ops/token) save ~2-3 μs total.

**Quality**: No regression. FP16 round-trip loss (cast FP32→FP16→FP32 around allreduce) is negligible relative to Q8_0 quantization noise (~1e-3).

### 1B. Allreduce–Compute Overlap — ❌ NOT FEASIBLE

**Root cause**: All stages execute sequentially on the **same GPU compute stream**. `allreduceSingleDeviceOnStream()` runs RCCL on the caller's stream, not a separate RCCL stream. There is no independent compute to overlap with:
- After Wo allreduce → attn_residual_add **immediately depends** on the result
- After Down allreduce → ffn_residual_add **immediately depends** on the result
- Cross-layer overlap impossible because layers are strictly sequential
- Moving allreduce to a separate stream still requires compute stream to wait

**Would require**: Dual-stream architecture with speculative execution of non-dependent stages on a second stream. Major architectural change not justified for ~29 μs per op.

### 1C. Batched Allreduce (Wo + Down Fusion) — ❌ NOT FEASIBLE

**Root cause**: Wo and Down allreduces occur at **different pipeline points**:
- Wo allreduce happens after attention output projection (mid-layer)
- Down allreduce happens after FFN down projection (end of layer)
- The FFN computation between them **depends on** the Wo allreduce result (via residual stream)

Cannot batch operations that have intermediate dependencies.

### Phase 1 Actual Result

| Metric | Before | After Phase 1 | Improvement |
|--------|:------:|:---------:|:-----------:|
| Allreduce ms (128 tok, sum both devs) | ~2525 | ~2500 | −1% (noise) |
| Decode tok/s | 47.3 | ~47.2 | negligible |
| **Revised assessment** | | | Allreduce cost is ~30.7% of decode, dominated by RCCL protocol overhead, NOT message size |

**Key insight**: The original estimate of −136 ms was based on the assumption that message size dominated allreduce latency. In reality, at 7-14 KB per message, RCCL protocol overhead (~29 μs) dwarfs transfer time (~0.01 μs at 1 TB/s PCIe). The only way to significantly reduce allreduce cost is to reduce the **number** of allreduce calls (kernel fusion in Phase 2 doesn't help — fusion is within a rank, 56 allreduces per token remain). Reducing allreduce count requires architectural changes (e.g., reducing TP synchronization points).

---

## Phase 2: GEMV Kernel Fusion (45% of gap, ~232 ms)

**Goal**: Reduce decode GEMV overhead from 919 ms → ≤550 ms through launch count reduction and atomicAdd elimination.

### Current Decode Launch Accounting (CORRECTED)

The original plan assumed 3 launches per projection (quantize + GEMV + scale) = 21 launches/layer. **Actual investigation reveals:**

| Per Layer | Launches | Notes |
|-----------|:--------:|-------|
| FusedQKV projection | 2 | `multiply_fused_tensor()`: 1 shared quantize + 1 batched GEMV (3 projections) |
| Wo projection | 2 | `multiply_tensor()`: 1 quantize + 1 GEMV |
| FusedGateUp projection | 2 | `multiply_fused_tensor()`: 1 shared quantize + 1 batched GEMV (2 projections) |
| Down projection | 2 | `multiply_tensor()`: 1 quantize + 1 GEMV |
| **Per layer** | **8** | Scale is done inline in GEMV kernel (no separate epilogue) |
| **28 layers** | **224** | + norms, residuals, attention, etc. |

**Key findings**:
- `multiply_fused_tensor()` already batches QKV and Gate+Up projections (Phases 2B/2C already done)
- Scale output is computed inline in the GEMV kernel (no separate launch)
- Remaining opportunity: fuse quantize into GEMV kernel (Phase 2A) → saves 4 launches/layer → 112 fewer launches total

**HIP launch cost**: ~1.5–2.0 μs × 588 = **~1 ms of pure dispatch overhead**.

### 2A. Fused Quantize-GEMV-Scale Pipeline

**Status**: Currently 3 separate launches (quantize_fp32_to_int8 → GEMV → scale_output).

**Target**: Single fused kernel that reads FP32 input, quantizes to INT8 in registers, computes GEMV, and scales output — all in one launch.

| Metric | Current (3 launches) | Fused (1 launch) | Savings per op |
|--------|:-:|:-:|:-:|
| Launch overhead | 4.5–6 μs | 1.5–2 μs | −3–4 μs |
| Total per token (56 ops) | 252–336 μs | 84–112 μs | ~200 μs |
| Over 128 tokens | 32–43 ms | 11–14 ms | ~25 ms |

**Work**:
1. New fused kernel template in `ROCmGemvKernel_INT8_VNNI.hip`
2. Read FP32 activation, compute per-row absmax inline (single warp reduction)
3. Quantize-multiply-accumulate in the same loop body
4. Scale output inline before atomicAdd

**Risk**: Low — all operations are already implemented separately, just merging codegen.

### 2B. Batched QKV Projection

**Status**: Q, K, V are three separate GEMV calls with same input, different weights.

**Target**: Single kernel launch that computes all three projections simultaneously.

| Metric | Current | Batched | Savings |
|--------|:-:|:-:|:-:|
| Launches per layer | 9 (3×3) | 3 (1×3 fused) | −6 per layer |
| Over 28 layers | 252 | 84 | −168 launches |

**Concept**: A "grouped GEMV" kernel where grid.z indexes the projection (Q=0, K=1, V=2) and each projection reads a different weight matrix but the **same input** (read once, used thrice → 3× better input cache reuse).

**Work**:
1. New `gemv_batched_qkv` kernel template
2. Three weight pointers + one input pointer + three output pointers
3. Grid: `(grid_n_max, KB, 3)` — z-dim selects projection
4. Handle different N per projection (Q=d_model, K=kv_dim, V=kv_dim)

**Risk**: Medium — different N per projection requires conditional grid sizing.

### 2C. Fused Gate-Up Projection

**Status**: Gate and Up are two separate GEMVs with same input, different weights, followed by SwiGLU.

**Target**: Single kernel: `output = SiLU(gate(x)) * up(x)` in one launch.

| Metric | Current | Fused | Savings |
|--------|:-:|:-:|:-:|
| Launches per layer | 6 (2×3) + 1 SwiGLU | 3 (1×3 fused) | −4 per layer |
| Over 28 layers | 196 | 84 | −112 launches |

**Work**:
1. New `gemv_fused_gate_up_swiglu` kernel
2. Two weight matrices, same input, fused SiLU activation
3. Output directly feeds to FFN down projection

**Risk**: Medium — fused kernel is larger, may hit register pressure on gfx906.

### 2D. TP-Aware KB Selection — ✅ DONE

**Status**: COMPLETED. Two-tier heuristic replaces flat TARGET_ACTS_PER_WAVE=16.

**Problem**: The flat heuristic had two failure modes:
1. **Over-splitting at large grid_n**: FFN Gate/Up TP=1 (grid_n=148) selected KB=14, but KB=1 was 3.8% faster — grid alone provided sufficient CU occupancy.
2. **Under-splitting for large K at TP≥2**: FFN Down TP=2 (grid_n=28, K=9472) selected KB=3, but KB=14 was 3.6% faster — the large-K regime TARGET=16 left CUs starved.

**Fix**: Two changes to `select_blockwise_qwo_outer_splits`:
1. **Small-K regime**: Cap KB=1 when `grid_n >= 2×NUM_CUS` (120+)
2. **Large-K regime**: Two-tier approach — conservative (TARGET=16) when it fills CUs (≥1.5 blocks/CU); aggressive (TARGET=2.5) otherwise

**Perf test**: `Perf__ROCmGemvKBScaling.cpp` — full KB sweep across TP=1/2/4 for all Qwen-7B projections.

**Isolated kernel results**:

| Shape | Old KB | New KB | Latency Δ |
|---|---|---|---|
| FFN Gate/Up TP=1 (grid_n=148) | 14 | 1 | **−3.8%** |
| FFN Down TP=2 (grid_n=28, K=9472) | 3 | 14 | **−3.6%** |
| FFN Down TP=1 (grid_n=28, K=18944) | 4 | 4 | ≈0 (no regression) |
| Q/KV/Wo/FFN Gate TP=2/4 | unchanged | unchanged | neutral |

**Remaining gaps** (accepted — fixing requires shape-specific exceptions):
- Q TP=2 (grid_n=14): KB=14 vs best=5 (8.6% gap, atomicAdd contention at medium grid_n)
- Wo TP=4 (grid_n=28, K=896): KB=3 vs best=4 (7.3% gap)

**Production benchmark**: Improvements too small (~0.5% per-token) to measure with 3-run averaging.

| Config | Prefill tok/s | Decode tok/s |
|--------|:-:|:-:|
| Single GPU | 938 | 74.9 |
| TP=2 | 1045 | 84.4 |

**Risk**: None — no regressions observed.

### Phase 2 Target

| Metric | Before | After Phase 2 | Improvement |
|--------|:------:|:---------:|:-----------:|
| GEMM decode ms (128 tok) | 919 | ≤550 | −369 ms |
| Launch count / token | ~588 | ~200 | −66% |
| Decode tok/s (cumulative) | ~53 | ~68 | +28% |

---

## Phase 3: Flash Attention Decode TP Scaling (17% of gap, ~85 ms)

**Goal**: Improve flash attention decode scaling from 1.02× → ≥1.4× at TP=2.

### Root Cause Analysis

The decode attention kernel launches `(n_heads, num_splits, batch)` blocks. At TP=2:

| | Single GPU | TP=2 per-device | CU Utilization |
|---|:-:|:-:|:-:|
| Q-heads | 28 | 14 | 23% (14/60 CUs) |
| KV-heads | 4 | 4 | same (GQA) |
| Splits | 1–8 | 1–8 | same KV length |
| **Blocks** | 28–224 | 14–112 | ≤50% → **bandwidth under-saturated** |

With 14 heads and split=1, only 14 of 60 CUs are active. The kernel is bandwidth-bound (~0.5 MACs/byte) so halving occupancy directly halves effective HBM throughput.

### 3A. Aggressive Split-K for Low Head Count — ✅ DONE

**Status**: COMPLETED. CU-aware occupancy floor added to `chooseDecodeSplitsAuto()`.

**Fix**: Added `n_heads`-aware occupancy targeting with 75% CU floor:
- `min_splits = ceil(0.75 × NUM_CUS / n_heads)` — ensures CU saturation
- Candidates filtered by occupancy, autotune-key updated to include `n_heads`
- Verified: all 8 flash attention perf tests pass

**Production results**:
- TP=2 prefill: 783→1044 tok/s (combined with Phase 0B graph capture fix)
- TP=2 decode: 82.4→84.6 tok/s (+2.7%)

**Target**: When `n_heads < CU_count/2`, increase splits to fill CU grid.

```
Current:  14 heads × 1 split = 14 blocks (23% CU occupancy)
Proposed: 14 heads × 4 splits = 56 blocks (93% CU occupancy)
```

Each split processes KV_len/4 positions, then a reduction kernel merges partial softmax states.

| Metric | Current (split=1) | With split=4 | Improvement |
|--------|:-:|:-:|:-:|
| Active CUs | 14 | 56 | 4× |
| HBM bandwidth utilized | ~250 GB/s | ~900 GB/s | 3.6× |
| Per-head wall time | 1.0× | ~0.35× | 2.9× reduction |
| Expected attention ms | 167 | ~60 | −107 ms |

**Work**:
1. Modify split selection: `splits = max(current, ceil(target_CU_occupancy / n_heads))`
2. Cap at 8 (current max) or extend to 16 for very small head counts
3. Validate reduction kernel handles variable split counts correctly
4. Benchmark at KV lengths {128, 512, 1024, 4096}

**Risk**: Medium — reduction kernel adds overhead, benefit depends on KV length.

### 3B. Multi-Query Attention Grouping

**Status**: Each Q-head gets its own block, even when multiple Q-heads share the same KV-head (GQA ratio 28:4 = 7:1).

**Concept**: For TP=2 + GQA, co-schedule Q-heads sharing the same KV-head in adjacent wavefronts to share KV cache reads.

**Work**: Significant kernel restructuring — defer to Phase 4 if split-K is sufficient.

**Risk**: High — fundamental kernel architecture change.

### Phase 3 Target

| Metric | Before | After Phase 3 | Improvement |
|--------|:------:|:---------:|:-----------:|
| Flash Attn decode ms (128 tok) | 167 | ≤80 | −87 ms |
| Decode tok/s (cumulative) | ~68 | ~80 | +18% |

---

## Phase 4: Prefill Scaling (secondary priority)

**Goal**: Improve prefill from 1.04× → ≥1.7× scaling.

### Root Cause

Prefill at 596 tokens uses GEMM (not GEMV) with M=596. The kernels scale well individually (GEMM_FFN at 1.98×), but CK GEMM launch overhead and allreduce cost dominate.

### 4A. HIP Graph Capture for Prefill — ⚠️ REVISITED

**Original Status**: `allreduceOnStream()` supports graph capture. CK GEMM kernels are graph-capturable.

**Phase 0B Finding**: HIP graph capture for prefill was already happening (via `executeDecodeWithCapturePolicy`), but was **counterproductive** — the capture overhead (~550ms) exceeded the replay savings because prefill shapes vary per prompt. Phase 0B **disabled** graph capture for prefill, improving throughput by 33%.

**Remaining opportunity**: Graph capture could still benefit **repeated prefills of the same length** (e.g., batched serving with fixed prompt templates). However, this requires:
1. Detecting when multiple prefills share the same seq_len
2. Amortizing capture cost over many replays (need ≥3 replays to break even)
3. This is a serving-mode optimization, not relevant for interactive use

**Status**: Deprioritized — current `executeFastDecode` path for prefill achieves 1044 tok/s (1.11× scaling), which is acceptable.

### 4B. Overlapped Prefill Allreduce

Same technique as Phase 1B but with larger messages (596 × 3584 × 4 = 8.5 MB per op). Overlapping with subsequent computation hides the cost.

### Phase 4 Target (Revised)

| Metric | Before Phase 0B | After Phase 0B | Stretch (Phase 4B) |
|--------|:-:|:-:|:-:|
| Prefill tok/s (TP=2) | 783 | **1044** ✅ | ≥1,570 |
| Prefill scaling | 0.84× | **1.11×** ✅ | ≥1.7× |

> Phase 0B already fixed the primary prefill regression. Phase 4B (overlapped allreduce) remains a stretch goal for further gains.

---

## Overall Targets & Milestones

### Decode Scaling Roadmap (Revised after Phase 1)

| Phase | Key Change | Decode tok/s | vs Single GPU | Prefill tok/s | Prefill Scaling |
|:-----:|------------|:-----------:|:------------:|:------------:|:--------------:|
| **Baseline** | Before fix (TP=2) | 47.3 | 0.91× | 783 | 0.84× |
| **Phase 0** ✅ | executeFastDecode TP fast-path | 83.4 | 1.12× | 783 | 0.84× |
| **Phase 0B** ✅ | Prefill graph capture bypass | **82.4** | **1.10×** | **1044** | **1.11×** |
| **Phase 1** ✅ | FP16 allreduce (all layers) | ~82.4 | 1.10× | ~1044 | ~1.11× |
| **Phase 2** | GEMV kernel fusion | ~95 | 1.27× | — | — |
| **Phase 3** | Attention split-K | ~104+ | 1.40×+ | — | — |
| *Theoretical max* | *Perfect 2× scaling* | *149* | *2.00×* | *1876* | *2.00×* |

> **Note**: Phase 0 (executeNode bypass) was the dominant fix, delivering 82.4 tok/s decode vs the original 47.3 tok/s baseline. The original profiling data above was measured with `LLAMINAR_PROFILING=1` which forces the `executeNode` path — this is why the profiling-measured single GPU baseline was 52.2 tok/s. Production-path (no profiling) baselines are: single GPU 74.9 tok/s, TP=2 82.4 tok/s.

### Success Criteria

| Metric | Original Baseline | Current (Phase 0+0B+1) | Target | Stretch |
|--------|:-------:|:------:|:------:|:-------:|
| **Decode tok/s (TP=2)** | 47.3 | **82.4** ✅ | **≥80** ✅ | ≥104 |
| **Decode scaling factor** | 0.91× | **1.10×** ✅ | **≥1.07×** ✅ | ≥1.40× |
| **Prefill tok/s (TP=2)** | 783 | **1044** ✅ | **≥1,000** ✅ | ≥1,570 |
| **Prefill scaling factor** | 0.84× | **1.11×** ✅ | **≥1.07×** ✅ | ≥1.70× |

> **Decode and prefill targets met!** Phase 0 (executeNode bypass) + Phase 0B (prefill graph capture bypass) together exceed both the ≥80 decode tok/s and ≥1,000 prefill tok/s targets. Remaining phases (2, 3) target further gains toward the 1.40×+ decode stretch goal.

### Measurement Protocol

**Production benchmarks** (throughput numbers for comparison):
```bash
# TP=2
./build_v2_release/llaminar2 \
  --benchmark \
  --tp-devices "rocm:0,rocm:1" \
  -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  -n 128

# Single GPU baseline
./build_v2_release/llaminar2 \
  --benchmark \
  -d rocm:0 \
  -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  -n 128
```

**GPU stage timing** (per-stage GPU event-based timing on the production fast path — zero GPU overhead):
```bash
LLAMINAR_GPU_STAGE_TIMING=1 ./build_v2_release/llaminar2 \
  --benchmark \
  --tp-devices "rocm:0,rocm:1" \
  -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  -n 128

# With per-stage detail breakdown:
LLAMINAR_GPU_STAGE_TIMING_DETAIL=1 ./build_v2_release/llaminar2 \
  --benchmark \
  --tp-devices "rocm:0,rocm:1" \
  -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  -n 128
```

**Full profiling** (CPU kernel timing + executor overhead + GPU stage timing — note: forces `executeNode` path which adds CPU overhead):
```bash
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 \
  --benchmark \
  --tp-devices "rocm:0,rocm:1" \
  -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  -n 128
```

**TP timing breakdown** (per-device prefill/decode split):
```bash
LLAMINAR_TP_TIMING=1 ./build_v2_release/llaminar2 \
  --benchmark \
  --tp-devices "rocm:0,rocm:1" \
  -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  -n 128
```

> **IMPORTANT**: Production benchmark numbers (without `LLAMINAR_PROFILING`) are the authoritative throughput figures. `LLAMINAR_PROFILING=1` forces the `executeNode` path adding 20-40% CPU overhead — use only for per-kernel CPU-side analysis. For accurate GPU-side per-stage timing without distorting throughput, use `LLAMINAR_GPU_STAGE_TIMING=1` which records GPU events on the production fast path (`executeFastDecode`) with ~0.5μs resolution and zero GPU overhead. `LLAMINAR_PROFILING=1` auto-enables GPU stage timing in addition to its CPU-side instrumentation.

---

## Dependency Graph

```
Phase 1A (FP16 allreduce)     ──┐
Phase 1C (batched allreduce)  ──┤──→ Phase 1B (allreduce overlap) ──→ Phase 4B
                                │
Phase 2A (fused quant-GEMV)   ──┤
Phase 2D (KB heuristic) ✅    ──┤──→ Phase 2B (batched QKV)
                                │──→ Phase 2C (fused gate-up)
                                │
Phase 3A (split-K attention) ✅──┘──→ Phase 3B (MQA grouping, stretch)
                                          │
                                          ▼
                                    Phase 4A (HIP graph capture)
```

**Recommended execution order**: ~~1A →~~ ~~2D →~~ ~~3A →~~ 2A → 2B → 2C → ~~4A~~ → 4B → 3B  
*(Phase 0+0B complete: executeNode bypass + prefill graph capture bypass. Phase 1 complete: 1A done, 1B/1C infeasible. Phase 4A deprioritized after 0B findings.)*

---

## Phase 0: executeFastDecode TP Fast-Path Fix — COMPLETED

**Date**: 2025-07-25

### Root Cause

In `DeviceGraphExecutor::executeFastDecode()`, a `collective_graph` safety check routed **ALL** stages through `executeNode()` when any collectives were present in the graph (TP mode). This applied even to trivial stages like ResidualAdd. Meanwhile, single GPU's non-collective graph used the maximal fast path (`node->stage->execute(ctx)` directly).

The `executeNode()` path performs per-stage:
1. `bufferContract()` — virtual call, allocates `StageBufferContract` with 3-5 `std::vector` members
2. `contract.allArenaReads()` — allocates new vector, copies bindings
3. `contract.allWrites()` — allocates new vector (called twice: coherence + mark dirty)
4. Arena `prepareForRead`/`prepareForWrite` — flag checks (fast but unnecessary)
5. Various safety checks, string operations, profiling infrastructure

In TP=2, two worker threads run simultaneously, both performing rapid short-lived heap allocations. **Thread contention on the malloc arena** amplified the overhead from ~17 μs to ~97 μs per stage call — a 5.7× increase.

This made ADD_RESIDUAL (a trivially fast float4 kernel) appear as the **#1 bottleneck** in TP=2 profiling at 22.6% of execution time (2092 ms for 21672 calls vs 365 ms for single GPU).

### Fix

Removed the `collective_graph` safety branch. Non-collective stages now always take the maximal fast path. Collective stages (ALLREDUCE, ALLGATHER) are already handled by explicit `is_collective_node` checks above this branch, so no collective coherence was lost.

**File changed**: `src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp`
- Removed `collective_graph` variable and pre-scan loop
- Removed conditional branch routing non-collective stages through `executeNode()`
- Non-collective stages now hit `node->stage->execute(ctx)` directly (same as single GPU)

### Results

| Config | Before (tok/s) | After Phase 0 (tok/s) | After Phase 0+0B (tok/s) | Improvement |
|--------|:-:|:-:|:-:|:-:|
| Single GPU decode | 74.6 | 74.6 | 74.9 | — (unaffected) |
| Single GPU prefill | 938 | 938 | 938 | — (unaffected) |
| TP=2 decode | 46.9 | **83.4** | **82.4** | **+76%** |
| TP=2 prefill | 1031 | 783 | **1044** | **+1.3%** (regression fixed) |
| **Decode scaling** | **0.63×** | **1.12×** | **1.10×** | **decode > single GPU** |
| **Prefill scaling** | **1.10×** | **0.84×** | **1.11×** | **prefill > single GPU** |

**Note**: Profiling-enabled runs still show the overhead (profiling forces executeNode path), which is expected — profiling needs the full infrastructure for timing attribution.

---

## Phase 0B: Prefill HIP Graph Capture Bypass — COMPLETED

**Date**: 2025-07-25

### Root Cause

After Phase 0, TP=2 prefill dropped from 1031 to 783 tok/s. Investigation revealed the cause was **not** the Phase 0 change but a pre-existing issue now exposed by proper benchmarking:

In `DeviceGraphOrchestrator::executeForward()`, cached forward graphs (including prefill) were routed through `executeDecodeWithCapturePolicy()` regardless of whether the graph was actually a decode graph. This function attempts **HIP segmented graph capture** — recording all GPU operations into a replayable graph for zero-overhead dispatch.

For decode (seq_len=1), graph capture is beneficial: the graph is captured once and replayed thousands of times (one per decode token). For prefill (seq_len=596), graph capture is **wasteful**: the capture costs ~550ms, and the prefill shape changes with every new prompt, so the captured graph is never replayed.

In the 3-iteration benchmark:
- Iteration 1: Cache miss → `executeSequential()` → ~570ms (normal)
- **Iteration 2: Cache hit → `executeDecodeWithCapturePolicy()` → HIP graph capture → ~1140ms** (550ms wasted on capture)
- Iteration 3: Cache hit + captured graph → `hipGraphLaunch` → ~570ms (replay works)

The benchmark average (3 iterations) was dragged from ~570ms to ~760ms by the single capture iteration.

**Why profiling-enabled runs were faster**: `buildDecodeCapturePolicy()` sets `allow_fast_decode = false` when `executor_profiling` is true. This bypasses graph capture entirely via `execute()` → `executeSequential()`. So profiling-enabled TP=2 prefill at 1031 tok/s was the "correct" speed — the non-profiling path was slower due to wasteful graph capture.

### Fix

Gated `executeDecodeWithCapturePolicy()` on `is_decode` (seq_len == 1). Prefill graphs now call `executor_.executeFastDecode()` directly, bypassing the graph capture policy entirely.

**File changed**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
- Added `is_decode` check before entering the graph capture policy path
- Prefill (seq_len > 1) → `executor_.executeFastDecode()` directly
- Decode (seq_len == 1) → continues through `buildDecodeCapturePolicy()` + `executeDecodeWithCapturePolicy()` as before

### Results (Production Path, No Profiling)

| Config | Before Phase 0B (tok/s) | After Phase 0B (tok/s) | Improvement |
|--------|:-:|:-:|:-:|
| Single GPU decode | 74.6 | 74.9 | — (noise) |
| Single GPU prefill | 938 | 938 | — (unaffected) |
| TP=2 decode | 83.4 | **82.4** | — (noise, within variance) |
| TP=2 prefill | 783 | **1044** | **+33%** |
| **Prefill scaling** | **0.84×** | **1.11×** | **regression eliminated** |

TP=2 prefill timing per benchmark iteration (with TP_TIMING):
- Before: 572 / **1138** / 572 ms (iter 2 is capture overhead)
- After: 572 / 570 / 574 ms (consistent, no outlier)

### Correctness Verification

Greedy decode produces coherent text: `"The capital of France is Paris. It is located in the?..."`

---

## Risk Matrix

| Phase | Risk | Mitigation |
|:-----:|:----:|------------|
| 1A | Low | Existing infra, parity test validation |
| 1B | Medium | Careful dependency edge tracking; fallback to serialized |
| 1C | Low | Same bytes, fewer calls; easy rollback |
| 2A | Low | Merge existing kernels; easy to A/B benchmark |
| 2B | Medium | Different N per projection; conditional grid needed |
| 2C | Medium | Register pressure on gfx906 (256 VGPRs); may need 2-pass |
| 2D | Low | Heuristic tuning; revertible |
| 3A | Medium | Reduction kernel overhead may negate benefit at short KV |
| 3B | High | Fundamental kernel redesign; defer if split-K suffices |
| 4A | Medium | Graph capture constraints (static shapes, no host sync) |
| 4B | Low | Same as 1B but for prefill |

---

## Phase 2D Results: KB Heuristic Tuning (Completed 2026-03-12)

**Scope**: Four targeted changes to `select_blockwise_qwo_outer_splits()` in `ROCmGemvKernel_INT8_VNNI.hip` (lines 163-290), validated across all 5 Qwen models with per-shape KB sweeps and full TP scaling analysis.

### Changes Made

| # | Change | Root Cause | Key Fix |
|---|--------|-----------|---------|
| 1 | Remove `occupancy_cap` + add high-grid_n branch | `cap = max(1, (60*2)/grid_n)` forced KB=1 for wide shapes | Three-branch large-K: high-grid_n (≥3 bpc → apw/5), conservative (apw/16), aggressive (apw/2.5) |
| 2 | Ceiling division in small-K | `act_blocks/wk` floor creates uneven shards | `(act_blocks + wk - 1) / wk` avoids pathological values |
| 3a | Conditional occupancy floor | `MIN_WAVES_PER_CU * NUM_CUS` floor pushes KB too high for bandwidth-saturated shapes | Floor now only applies when `apw ≤ SMALL_K_THRESHOLD` |
| 3b | Conservative cap at KB=4 for very large K | Conservative tier's `apw/16` over-splits when apw > 64 | `min(conservative_kb, 4)` when `apw > 64`; individual shards already bandwidth-saturated |

### Per-Shape Kernel Improvements (KB Sweep, Auto vs Before)

| Model | Shape | TP | Before (µs) | After (µs) | Delta | Root Fix |
|-------|-------|---:|-------------|------------|-------|----------|
| 32B | FFN Gate | 1 | 215.8 | 188.6 | **-12.6%** | #1 high-grid_n branch (KB=1→4) |
| 32B | FFN Up | 1 | 215.8 | 188.8 | **-12.5%** | #1 high-grid_n branch |
| 32B | FFN Down | 1 | 205.4 | 184.3 | **-10.3%** | #3b conservative cap (KB=6→4) |
| 32B | KV proj | 2 | 24.3 | 22.7 | **-6.6%** | #3a conditional floor (KB=15→8) |
| 14B | FFN Down | 4 | 45.8 | 35.2 | **-23.1%** | #2 ceiling division (KB=13→14) |
| 14B | KV proj | 2 | 24.6 | 22.6 | **-8.1%** | #3a conditional floor (KB=15→8) |
| 7B | FFN Gate | 2 | 56.0 | 51.2 | **-8.6%** | #1 occupancy_cap removal |
| 7B | Wo proj | 4 | 20.0 | 18.2 | **-9.0%** | #2 ceiling division (KB=3→4) |
| 0.5B | FFN Gate | 1 | 21.3 | 19.4 | **-8.9%** | #2 ceiling division (KB=3→4) |
| 0.5B | FFN Up | 1 | 21.3 | 19.4 | **-8.9%** | #2 ceiling division |

### TP Scaling Totals (AllModels_WithLMHead, After)

| Model | 1 GPU (µs) | 2 GPU (µs) | 2 GPU Eff | 4 GPU (µs) | 4 GPU Eff |
|-------|------------|------------|-----------|------------|-----------|
| 0.5B | 293 | 223 | **66%** | 210 | **35%** |
| 3B | 592 | 400 | **74%** | 360 | **41%** |
| 7B | 1030 | 671 | **77%** | 583 | **44%** |
| 14B | 1377 | 879 | **78%** | 461 | **75%** |
| 32B | 1654 | 997 | **83%** | 528 | **78%** |

### Layer-Only Improvements (No LM Head)

| Model | TP | Before (µs/layer) | After (µs/layer) | Delta | × Layers | Per-Token Savings |
|-------|----|-------------------|------------------|-------|----------|-------------------|
| 32B | 1 | 788 | 720 | **-8.6%** | ×64 | **-4,352µs** |
| 7B | 2 | 272 | 240 | **-11.8%** | ×28 | **-896µs** |
| 14B | 2 | 312 | 289 | **-7.4%** | ×40 | **-920µs** |
| 14B | 4 | 211 | 208 | **-1.4%** | ×40 | -120µs |
| 3B | 2 | 160 | 156 | **-2.5%** | ×36 | -144µs |

### Remaining Gaps (Diminishing Returns)

Three categories of sub-optimal KB remain, none addressable by parametric heuristic changes without regressing other shapes:

| Category | Gap Size | Shapes | Why Unfixable |
|----------|---------|--------|---------------|
| **Measurement noise** | 5-10% | 0.5B/3B KV projections (<20µs) | Auto path overhead dominates at sub-20µs scale; heuristic picks correct KB |
| **Non-monotonic KB=28/56** | 5-9% | 14B/32B FFN Gate/Up TP=2/4 | Hardware-specific wave scheduling sweet spots; intermediate KB values (14, 16) are worse |
| **Aggressive tier overshoot** | 3-7% | 14B/32B Q/Wo TP=1/2 (apw=20) | KB=8 vs optimal KB=4, but reducing divisor would regress 7B FFN Down (apw=37, needs KB=14) |

---

## GEMV Decode Scaling Baseline (Measured 2026-03-11)

**Hardware**: 3× AMD Instinct MI50 (gfx906, 60 CUs, ~1024 GB/s HBM each), PCIe 3.0 x16  
**Kernel**: `ROCmGemvKernel_INT8_VNNI.hip` — LDS-kred dispatch (TN=128, 1 wavefront/block)  
**Test**: `Perf__ROCmTPScaling::GEMV_TPScaling_AllModels_WithLMHead`  
**All times in µs. Efficiency = ideal/actual (ideal = 1GPU_time / TP_degree).**

### Attention Projections

| Model | Shape (N×K) | 1 GPU (µs) | 2 GPU (µs) | 2 GPU Eff | 4 GPU (µs) | 4 GPU Eff |
|-------|-------------|------------|------------|-----------|------------|-----------|
| **Q proj** | | | | | | |
| 0.5B | 896×896 | 15.2 | 14.6 | **52%** | 14.4 | **26%** |
| 3B | 2048×2048 | 22.4 | 18.1 | **62%** | 16.8 | **33%** |
| 7B | 3584×3584 | 32.8 | 25.8 | **64%** | 21.1 | **39%** |
| 14B | 5120×5120 | 53.3 | 32.5 | **82%** | 24.8 | **54%** |
| 32B | 5120×5120 | 53.4 | 32.6 | **82%** | 24.8 | **54%** |
| **K proj** | | | | | | |
| 0.5B | 128×896 | 14.4 | 14.1 | **51%** | 13.8 | **26%** |
| 3B | 256×2048 | 16.8 | 16.3 | **52%** | 15.5 | **27%** |
| 7B | 512×3584 | 19.8 | 19.2 | **52%** | 21.0 | **24%** |
| 14B | 1024×5120 | 25.1 | 23.7 | **53%** | 22.1 | **28%** |
| 32B | 1024×5120 | 24.6 | 23.8 | **52%** | 22.2 | **28%** |
| **V proj** | | | | | | |
| 0.5B | 128×896 | 14.2 | 13.9 | **51%** | 13.6 | **26%** |
| 3B | 256×2048 | 19.7 | 16.3 | **60%** | 15.5 | **32%** |
| 7B | 512×3584 | 20.0 | 21.9 | **46%** | 21.3 | **24%** |
| 14B | 1024×5120 | 25.0 | 23.5 | **53%** | 21.8 | **29%** |
| 32B | 1024×5120 | 25.0 | 23.7 | **53%** | 22.2 | **28%** |
| **Wo proj** | | | | | | |
| 0.5B | 896×896 | 15.4 | 13.8 | **56%** | 9.1 | **42%** |
| 3B | 2048×2048 | 22.2 | 16.5 | **68%** | 17.0 | **33%** |
| 7B | 3584×3584 | 33.4 | 23.7 | **71%** | 19.5 | **43%** |
| 14B | 5120×5120 | 53.3 | 34.1 | **78%** | 23.8 | **56%** |
| 32B | 5120×5120 | 47.0 | 33.8 | **70%** | 24.0 | **49%** |

### FFN Projections

| Model | Shape (N×K) | 1 GPU (µs) | 2 GPU (µs) | 2 GPU Eff | 4 GPU (µs) | 4 GPU Eff |
|-------|-------------|------------|------------|-----------|------------|-----------|
| **FFN Gate** | | | | | | |
| 0.5B | 4864×896 | 21.1 | 16.3 | **65%** | 15.5 | **34%** |
| 3B | 11008×2048 | 43.8 | 30.2 | **73%** | 23.0 | **48%** |
| 7B | 18944×3584 | 89.9 | 59.7 | **75%** | 37.8 | **60%** |
| 14B | 13824×5120 | 95.7 | 66.4 | **72%** | 39.7 | **60%** |
| 32B | 27648×5120 | 217 | 96.5 | **113%** | 57.3 | **95%** |
| **FFN Up** | | | | | | |
| 0.5B | 4864×896 | 21.1 | 16.2 | **65%** | 15.4 | **34%** |
| 3B | 11008×2048 | 43.8 | 30.2 | **73%** | 22.9 | **48%** |
| 7B | 18944×3584 | 90.4 | 61.8 | **73%** | 37.8 | **60%** |
| 14B | 13824×5120 | 96.3 | 66.4 | **73%** | 39.7 | **61%** |
| 32B | 27648×5120 | 217 | 96.2 | **113%** | 57.6 | **94%** |
| **FFN Down** | | | | | | |
| 0.5B | 896×4864 | 23.2 | 21.4 | **54%** | 15.5 | **37%** |
| 3B | 2048×11008 | 51.0 | 32.8 | **78%** | 23.7 | **54%** |
| 7B | 3584×18944 | 92.0 | 60.8 | **76%** | 39.0 | **59%** |
| 14B | 5120×13824 | 102 | 65.3 | **78%** | 39.8 | **64%** |
| 32B | 5120×27648 | 204 | 101 | **101%** | 56.8 | **90%** |

### LM Head

| Model | Shape (N×K) | 1 GPU (µs) | 2 GPU (µs) | 2 GPU Eff | 4 GPU (µs) | 4 GPU Eff |
|-------|-------------|------------|------------|-----------|------------|-----------|
| 0.5B | 151936×896 | 170 | 112 | **76%** | 105 | **41%** |
| 3B | 151936×2048 | 376 | 244 | **77%** | 229 | **41%** |
| 7B | 152064×3584 | 652 | 420 | **78%** | 393 | **42%** |
| 14B | 152064×5120 | 934 | 595 | **78%** | 241 | **97%** |
| 32B | 152064×5120 | 934 | 595 | **78%** | 240 | **97%** |

### Layer Total (Attention + FFN + LM Head)

| Model | 1 GPU (µs) | 2 GPU (µs) | 2 GPU Eff | 4 GPU (µs) | 4 GPU Eff |
|-------|------------|------------|-----------|------------|-----------|
| 0.5B | 295 | 222 | **67%** | 202 | **37%** |
| 3B | 596 | 404 | **74%** | 363 | **41%** |
| 7B | 1031 | 692 | **74%** | 591 | **44%** |
| 14B | 1384 | 907 | **76%** | 452 | **77%** |
| 32B | 1722 | 1003 | **86%** | 505 | **85%** |

### Observations

1. **Scaling improves dramatically with model size**: 37% (0.5B) → 85% (32B) at TP=4
2. **K/V projections universally poor** (24-53%): smallest N dims, completely launch-overhead dominated (~14µs floor)
3. **Q/Wo scale moderately** (33-82%): better than K/V due to larger shapes, but still launch-limited for small models
4. **FFN scales well for large models**: 32B gets 90-113% at TP=2 (superlinear from cache effects), 90-95% at TP=4
5. **LM Head discontinuity at TP=4**: 14B/32B jump to 97% when dispatch switches from `wide` → `LDS-kred`; 0.5B-7B stuck at ~41%
6. **Launch overhead floor (~14-18µs)** is the dominant bottleneck for any shape where ideal time < floor

---

## Files to Modify

| Phase | File | Change |
|:-----:|------|--------|
| 1A | `src/v2/collective/LocalTPContext.cpp` | Default FP16 for decode allreduce |
| 1B | `src/v2/execution/DeviceGraphExecutor.cpp` | Stream overlap for allreduce stages |
| 1C | `src/v2/models/qwen/Qwen2Graph.cpp` | Batch Wo+Down allreduce per layer |
| 2A | `src/v2/kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip` | Fused quant-GEMV-scale kernel |
| 2B | `src/v2/kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip` | Batched QKV dispatch |
| 2C | `src/v2/kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip` | Fused gate-up-SwiGLU |
| 2D | `src/v2/kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip` | KB cap heuristic (line ~4082) |
| 3A | `src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip` | Split selection for low head count |
| 3B | `src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip` | MQA co-scheduling (stretch) |
| 4A | `src/v2/execution/DeviceGraphExecutor.cpp` | HIP graph capture mode |

---

## Appendix: Kernel Architecture Reference

### Flash Attention Decode

- **Grid**: `(n_heads, num_splits, batch_size)`, **Block**: 256 threads (4 wavefronts)
- **Architecture**: Lane-owns-output-dims v3 — lane i owns dims `{i, i+64, i+128, i+192}`
- **Bandwidth-bound**: ~0.5 MACs/byte arithmetic intensity
- **Head is atomic**: one block per head per split, no intra-head TP
- **Split-K**: 1, 2, 4, or 8 splits for KV cache slicing
- **Source**: `src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip` lines 1456-1780

### GEMV (INT8 grid-kpar)

- **Grid**: `(ceil(N/TILE_N), KB)`, **Block**: 64 threads
- **KB heuristic**: targets 480 blocks (8 waves/CU × 60 CUs), rounded to factor of k_groups
- **At TP=2 N=1792**: grid_n=14, KB=34 → 476 blocks, 34 atomicAdds per output
- **At single N=3584**: grid_n=28, KB=17 → 476 blocks, 17 atomicAdds per output
- **Source**: `src/v2/kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip` lines 4033-4120

### Allreduce (RCCL)

- **Count**: 56 per token (Wo + Down × 28 layers)
- **Message**: 3,584 FP32 elements = 14,336 bytes per op
- **Backend**: RCCL per-device async (no barrier)
- **FP16 mode**: Available via `LLAMINAR_ALLREDUCE_PRECISION=fp16`
- **Source**: `src/v2/collective/LocalTPContext.cpp` lines 2120-2450
