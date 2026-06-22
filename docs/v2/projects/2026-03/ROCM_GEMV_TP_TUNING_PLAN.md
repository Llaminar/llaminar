# ROCm GEMV TP Scaling Tuning Plan

**Created**: 2026-03-11  
**Status**: Phase 3 — COMPLETE (+8.4% 0.5B TP=2), moving to Phase 4  
**Goal**: Improve GEMV decode throughput at TP-sharded dimensions across all Qwen2 model sizes (0.5B–32B) at TP degrees 2, 4, 8, 16.

---

## Problem Statement

When tensor-parallel sharding halves (or quarters, etc.) the N dimension of GEMV weight matrices, the ROCm INT8-VNNI GEMV kernel dispatch selects suboptimal execution paths. The N≥2048 gate on the high-performance QWO-LDS k-reduce path is a **dispatch heuristic, not a correctness constraint** — the underlying kernel (`gemv_int8_int8_lds_kreduce_blockwise_scaled_kernel_t`) works for any N that is a multiple of 4.

### Root Cause

| Path | Gate | Performance | Issue at TP-sharded N |
|------|------|-------------|----------------------|
| **QWO-LDS** | N≥2048 | Best (LDS k-reduce, high occupancy) | Excluded when N/tp < 2048 |
| **grid-kpar** | N≥128, fallback | Medium (atomicAdd-based) | Falls here when QWO excluded |
| **scatter** | grid_n≤8 redirect | Worst (partial buffer + reduce) | Falls here when grid_n is small |
| **tiny-KV** | N≤512, K≥2048 | Good (same kernel as QWO, WK=8) | Only covers K/V projections |

### Measured Impact (Qwen-7B, TP=2)

| Projection | Full N | TP N | Full µs | TP µs | Efficiency | Dispatch Change |
|-----------|--------|------|---------|-------|------------|-----------------|
| Q proj | 3584 | 1792 | 32 | 44 | **36.5%** | QWO → grid-kpar |
| K proj | 512 | 256 | 23 | 23 | 50.0% | tiny-KV → tiny-KV |
| Wo proj | 3584 | 3584 | 32 | 23 | 70.1% | QWO → QWO (K halved) |
| FFN Gate | 9472 | 4736 | 95 | 57 | 83.7% | QWO → QWO |
| FFN Down | 3584 | 3584 | 92 | 62 | 74.7% | QWO → QWO (K halved) |
| **Total** | | | | | **67.5%** | |

---

## TP Shape Space

### Model Dimensions

| Model | H (d_model) | I (FFN intermediate) | Heads | KV Heads | Head Dim | Layers |
|-------|-------------|---------------------|-------|----------|----------|--------|
| 0.5B | 896 | 4864 | 14 | 2 | 64 | 24 |
| 3B | 2048 | 11008 | 16 | 2 | 128 | 36 |
| 7B | 3584 | 18944 | 28 | 4 | 128 | 28 |
| 14B | 5120 | 13824 | 40 | 8 | 128 | 48 |
| 32B | 5120 | 27648 | 40 | 8 | 128 | 64 |

### Sharding Rules

- **COLUMN_PARALLEL** (Q, K, V, FFN Gate, FFN Up, LM Head): N → N/tp, K unchanged
- **ROW_PARALLEL** (Wo, FFN Down): N unchanged, K → K/tp

### Critical Problem Shapes (flagged `***`)

Shapes that fall out of QWO-LDS into grid-kpar or scatter:

| Model | TP | Projection | N | K | grid_n | Dispatch |
|-------|----|-----------|---|---|--------|----------|
| 0.5B | 2 | Q proj | 448 | 896 | 4 | scatter |
| 0.5B | 2 | FFN Gate/Up | 2432 | 896 | 19 | QWO ✓ |
| 3B | 2 | Q proj | 1024 | 2048 | 8 | scatter (gn=8) |
| 7B | 2 | Q proj | 1792 | 3584 | 14 | grid-kpar |
| 7B | 4 | Q proj | 896 | 3584 | 7 | scatter |
| 7B | 4 | FFN Gate/Up | 4736→2368 | 3584 | 19 | QWO ✓ |
| 14B | 2 | Q proj | 2560 | 5120 | 20 | QWO ✓ |
| 14B | 4 | Q proj | 1280 | 5120 | 10 | grid-kpar |
| 14B | 8 | Q proj | 640 | 5120 | 5 | scatter |
| 14B | 8 | FFN Gate/Up | 1728 | 5120 | 14 | grid-kpar |
| 14B | 16 | Q proj | 320 | 5120 | 3 | scatter |
| 14B | 16 | FFN Gate/Up | 864 | 5120 | 7 | scatter |
| 32B | 4 | Q proj | 1280 | 5120 | 10 | grid-kpar |
| 32B | 8 | Q proj | 640 | 5120 | 5 | scatter |
| 32B | 8 | FFN Gate/Up | 3456 | 5120 | 27 | QWO ✓ |
| 32B | 16 | Q proj | 320 | 5120 | 3 | scatter |
| 32B | 16 | FFN Gate/Up | 1728 | 5120 | 14 | grid-kpar |

---

## Phased Plan

### Phase 1: Unify Dispatch — QWO-LDS as Universal Path ✅ COMPLETE

**Impact**: HIGH — directly fixes the worst regressions  
**Risk**: LOW — kernel already works at any N  
**File**: `src/v2/kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip`

**Changes Implemented**:

1. **Lowered the QWO-LDS entry threshold** from `N >= 2048` to `N >= 128` (GRID_KPAR_TILE_N)
   - Renamed to `use_lds_kreduce_blockwise` to reflect unified nature
   - Absorbed old tiny-KV path (N≤512, K≥2048) — identical kernel template

2. **Added KB cap to occupancy selector** `select_blockwise_qwo_outer_splits`
   - Caps KB at `act_blocks / wk` to prevent excessive splits at small grid_n
   - Without cap, grid_n=1 produced KB=60 (mostly empty waves, atomicAdd contention)
   - With cap, matches the old tiny-KV selector's behavior

3. **Removed entire tiny-KV code block** (~40 lines) — absorbed into unified path

4. **Grid-kpar/scatter now unreachable** for blockwise vec4 shapes — left as dead-code safety net

**Results (vs pre-Phase-1 baseline)**:

| Model | TP | Before Eff% | After Eff% | Delta |
|-------|----|------------|-----------|-------|
| Qwen-7B | 2 | 67.5% | **75.1%** | **+7.6pp** |
| Qwen-7B | 4 | 46.0% | **50.8%** | **+4.8pp** |
| Qwen-3B | 2 | 62.3% | **68.3%** | **+6.0pp** |

**Key shape improvements (Qwen-7B TP=2)**:
- Q proj (1792×3584): 44µs → **25.9µs** (61.4% eff, was 36.5%)
- K proj (256×3584): 23µs → **19.4µs** (52.1% eff, was 50.0%)
- FFN Down (3584×9472): 62µs → **54.2µs** (85.4% eff, was 74.7%)

**E2E benchmark**: 82.79 tok/s TP=2, no regression vs 82.91 baseline (within run-to-run variance)

**Known limitations**: Shapes with N < 128 (e.g. K/V at extreme TP on small models) fall to "fallback" — Phase 3 (fused QKV) addresses these.

---

### Phase 2: Adaptive WK for Small grid_n — INVESTIGATED, WK=16 NEUTRAL

**Impact**: NONE (experimentally verified)  
**Finding**: WK=16 produces identical total wave count to WK=8 (WK×KB is constant for a given shape), and atomicAdd overhead is NOT the bottleneck for small-grid_n shapes — the kernel launch latency floor (~15-17µs) dominates.

**Infrastructure Added** (kept for future use):

1. **Expanded static_asserts** from `WK <= 8` to `WK <= 16` in all 6 kernel templates
2. **Added `case 16:` to all 6 switch(wk) dispatch blocks** — WK=16 is reachable via override env var
3. **Override clamp** expanded to `wk_ov <= 16` (was 8)
4. **Default WK stays at 8** — adaptive auto-select reverted since no measurable benefit

**Experimental Results** (WK=16 auto-select for grid_n≤4):

| Config | Phase 1 Eff% | WK=16 Eff% | Delta |
|--------|-------------|-----------|-------|
| 7B TP=2 | 75.1% | 74.8% | -0.3pp (noise) |
| 7B TP=4 | 50.8% | 51.5% | +0.7pp (noise) |
| 3B TP=2 | 68.3% | 66.7% | -1.6pp (noise) |
| 0.5B TP=2 | 48.7% | 48.9% | +0.2pp (noise) |

**E2E**: 83.38 tok/s TP=2 (baseline 82.79). Single: 74.92 (baseline 74.50). No regression.

**Why WK=16 doesn't help**: For K/V at 7B TP=2 (N=256, K=3584, grid_n=2):
- WK=8:  KB=14, total blocks=28, total waves=224
- WK=16: KB=7,  total blocks=14, total waves=224  ← SAME
- AtomicAdd passes halved (14→7), but this saves <1µs at ~17µs total latency

**Key insight**: The small-shape bottleneck is **launch overhead + insufficient CU utilization**, not atomicAdd contention. Phase 3 (fused QKV) is the correct fix for these shapes.

**Previous Phase 2 design** (TN=64 for grid_n≤2): Deprioritized since the grid_n doubling effect would slightly help CU utilization but the launch latency floor still dominates for N<512 shapes.

---

### Phase 3: Pair Dispatch Gate Widening + KB Selector Unification ✅ COMPLETE

**Impact**: +8.4% E2E decode for 0.5B TP=2, neutral for 7B  
**Risk**: LOW — same kernel, just changes dispatch routing

**Investigation Finding**: The original Phase 3 plan (fused QKV projection) was based on the assumption of 3 individual GEMV launches per QKV stage. Investigation of `multiply_fused_tensor` revealed that for Q8_0 models (`has_native_vnni=false`), K+V are **already batched into a single pair kernel launch** via `rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched`. Production QKV decode was already 2 launches (1× Q individual, 1× K+V pair), not 3.

**Real Issue**: The `is_tiny_kv_pair` gate in the batched dispatch had a `K >= 2048` restriction that excluded small models (0.5B: K=896) from the LDS k-reduce pair kernel. These fell through to the inferior `grid_kpar_pair` path. Additionally, the pair dispatch function used the legacy `select_blockwise_tiny_kv_outer_splits` KB selector instead of Phase 1's unified `select_blockwise_qwo_outer_splits`.

**Changes Implemented**:

1. **Removed `K >= 2048` from `is_tiny_kv_pair` gate** (line ~5083 in ROCmGemvKernel_INT8_VNNI.hip)
   - LDS k-reduce pair kernel works for ALL K values
   - Remaining conditions (N≤512, no bias, beta=0) are sufficient safety checks
   - This routes 0.5B K/V pairs (K=896) to the high-performance LDS-kred pair kernel

2. **Replaced `select_blockwise_tiny_kv_outer_splits` with `select_blockwise_qwo_outer_splits`** in `rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair` (line ~4476)
   - Aligns pair dispatch KB selection with Phase 1's two-regime + occupancy floor + KB cap logic
   - For 7B TP=2 (K=3584, grid_n=4): both selectors produce KB=14 (neutral, as expected)
   - For 0.5B TP=2 (K=896): new selector provides better occupancy tuning

**E2E Results (A/B comparison)**:

| Model | Config | Pre-Phase 3 | Post-Phase 3 | Delta |
|-------|--------|------------|-------------|-------|
| **Qwen-0.5B** | **TP=2** | **138.6 tok/s** | **150.2 tok/s** | **+8.4%** |
| Qwen-0.5B | Single | 196.4 | 193.6 | neutral |
| Qwen-7B | TP=2 | 83.4 | 83.2 | neutral |
| Qwen-7B | Single | 74.9 | 74.8 | neutral |
| Qwen-3B | TP=2 | — | 89.3 | (no pre-baseline, K=2048 already passed old gate) |

**0.5B TP Efficiency**: 70.6% → **77.1%** (+6.5pp)

**Why 7B is neutral**: K=3584 ≥ 2048, so the pair gate was already passing before Phase 3. Both KB selectors produce identical KB=14 for the 7B pair shape.

**Why 0.5B benefits**: K=896 < 2048, so pairs were routed to `grid_kpar_pair` (inferior) before Phase 3. Now they hit the LDS-kred pair kernel.

---

### Phase 4: GEMM Tile Tuning for Small-K TP

**Impact**: LOW — GEMM (prefill) is less affected than GEMV (decode)  
**Risk**: LOW — CK config selection only

**Changes**:

1. **Add CK GEMM configs for small-K TP shapes** in ROCm GEMM kernel
   - At TP=4, Wo has K=896 (7B) or K=1280 (14B) — may benefit from different tile sizes
   - Current CK configs may not have optimal tiles for these K values

2. **Profile GEMM shapes at all TP degrees** using existing `Perf__ROCmQuantisedGemmKernel` infrastructure

---

## Test Infrastructure

### Performance Tests

- **File**: `tests/v2/performance/kernels/rocm/Perf__ROCmTPScaling.cpp`
- **Tests**: `GEMV_TPScaling_Qwen{7B,3B,05B}`, `GEMM_TPScaling_Qwen{7B,3B,05B}`
- **Run**: `ctest --test-dir build_v2_release -R "V2_Perf_ROCmTPScaling" -V`
- **Reports**: Per-shape timing, ideal time, efficiency %, dispatch path, bandwidth

### Correctness Gate

```bash
# Single device
./build_v2_release/llaminar2 -d rocm:0 -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  -p "The capital of Paris is" -n 30 -t 0

# Pipeline parallel
./build_v2_release/llaminar2 \
  --define-domain "stage0=rocm:0;backend=rccl" \
  --define-domain "stage1=rocm:1;backend=rccl" \
  --pp-stage "0=stage0:0-13" --pp-stage "1=stage1:14-27" \
  -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  -p "The capital of Paris is" -n 30 -t 0

# Tensor parallel
./build_v2_release/llaminar2 --tp-devices "rocm:0,rocm:1" \
  -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  -p "The capital of Paris is" -n 30 -t 0
```

### Benchmark

```bash
# Before/after benchmark
./build_v2_release/llaminar2 --benchmark --tp-devices "rocm:0,rocm:1" \
  -m models/Qwen2.5-7B-Instruct-Q8_0.gguf -n 128
```

---

## Key Technical Details

### Dispatch Code Location

- **Main dispatch**: `ROCmGemvKernel_INT8_VNNI.hip` line ~4503 (`rocmGemv_int8_int8_fp32_vnni_blockwise_scaled`)
- **QWO gate**: line ~4576 (`N >= 2048`)
- **Scatter redirect**: line ~4793 (`grid_n <= 8`)
- **Occupancy selectors**: lines 60–210
- **LDS k-reduce kernel**: line ~2252 (`gemv_int8_int8_lds_kreduce_blockwise_scaled_kernel_t`)

### Tile Constants

| Constant | Value | Used By |
|----------|-------|---------|
| `GRID_KPAR_TILE_N` | 128 | grid-kpar path |
| `SQUARE_TILE_N` | 256 | square path |
| `ACT_BLOCK_K` | 32 | All paths (K blocking) |
| `NUM_CUS` | 60 | Occupancy selectors (MI60) |
| `MIN_WAVES_PER_CU` | 8 | QWO/tiny-KV occupancy |
| `TARGET_ACTS_PER_WAVE` | 16 | QWO occupancy |

### Kernel Template

```
gemv_int8_int8_lds_kreduce_blockwise_scaled_kernel_t<TN, CPT, WK>
```
- `TN`: Tile N (128 for QWO, 128 for tiny-KV) — elements per block in N dimension
- `CPT`: Columns per thread (2) — vec4 pairs
- `WK`: Work-K splits per wave (8 for both QWO and tiny-KV currently)
- **No N≥2048 constraint in kernel** — works for any N that's a multiple of 4

---

## Baselines (pre-tuning)

### End-to-End (Qwen-7B Q8_0, 128 decode tokens)

| Config | Decode tok/s | TP Efficiency |
|--------|-------------|---------------|
| Single (rocm:0) | 78.54 | — |
| TP=2 (rocm:0,1) | 82.91 | 105.6% |
| Peak TP (historical) | 95.53 | 122.4% |

### Per-Shape Efficiency Baselines

| Model | TP | Total Layer Efficiency |
|-------|----|----------------------|
| Qwen-7B | 2 | 67.5% |
| Qwen-7B | 4 | 46.0% |
| Qwen-3B | 2 | 62.3% |
| Qwen-0.5B | 2 | 57.1% |
| Qwen-0.5B | 4 | 27.7% |
