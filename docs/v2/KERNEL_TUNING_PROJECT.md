# Kernel Tuning & Optimisation Project

**Started**: 2026-02-09  
**Model**: Qwen2.5-7B-Instruct IQ4_NL (28 layers, 3584 hidden dim, 28 heads)  
**Benchmark**: 596 prefill tokens + 128 decode tokens, 3 iterations averaged after warmup  

---

## Baseline Numbers (2026-02-09)

### Throughput

| Metric | ROCm (MI50 gfx906) | CUDA (RTX 3090 SM 8.6) | CUDA / ROCm |
|---|---|---|---|
| **Prefill** | 172.8 tok/s | 1,493.7 tok/s | 8.6x |
| **Decode** | 14.2 tok/s | 25.7 tok/s | 1.8x |
| **Wall-clock total** | 12,487 ms | 5,385 ms | 2.3x |

### GPU Kernel Time Breakdown — Decode Phase

| Kernel | ROCm (ms) | ROCm % | CUDA (ms) | CUDA % | ROCm / CUDA |
|---|---|---|---|---|---|
| **GEMM** (CK / CUTLASS) | 10,847 | 46% | 5,710 | 49% | 1.9x |
| **Flash Attn Decode** | 12,003 | 51% | 5,634 | 48% | 2.1x |
| RMS Norm | 414 | 1.7% | 153 | 1.3% | 2.7x |Le
| Residual Add | 156 | 0.7% | 93 | 0.8% | 1.7x |
| RoPE | 126 | 0.5% | 60 | 0.5% | 2.1x |
| SwiGLU | 81 | 0.3% | 47 | 0.4% | 1.7x |
| Embedding | 6 | 0.0% | 3 | 0.0% | 2.0x |
| **Decode GPU total** | **23,632** | | **11,700** | | **2.0x** |

### GPU Kernel Time Breakdown — Prefill Phase

| Kernel | ROCm (ms) | CUDA (ms) | ROCm / CUDA |
|---|---|---|---|
| **Flash Attn Prefill** | 7,840 | 861 | 9.1x |
| **GEMM** (CK / CUTLASS) | 2,306 | 111 | 20.8x |
| RoPE | 5 | 39 | 0.1x (CUDA slower?) |
| SwiGLU | 16 | 14 | 1.2x |
| RMS Norm | 10 | 5 | 2.0x |
| Residual Add | 8 | 6 | 1.4x |
| **Prefill GPU total** | **10,185** | **1,035** | **9.8x** |

### Per-Kernel Averages — Decode

| Kernel | ROCm avg (µs) | CUDA avg (µs) |
|---|---|---|
| GEMM per call | 300.8 | 76.4 |
| Flash Attn Decode per call | 1,116 | 524 |
| RMS Norm per call | 19.2 | 7.1 |
| RoPE per call | 12.1 | 9.2 |
| SwiGLU per call | 9.0 | 5.6 |
| Residual Add per call | 7.6 | 4.5 |

---

## Observations

1. **Decode is 97% GEMM + Attention** on both GPUs. Everything else is noise.
2. **Prefill on ROCm is catastrophically slow** — 9.8x slower than CUDA overall.
   - GEMM prefill is **20.8x slower** — likely dominated by VNNI→CK repack overhead.
   - Flash Attn prefill is **9.1x slower** — MI50 FA2 kernel vs optimised cuDNN/FA2 on Ampere.
3. **Decode GEMM on ROCm is only 1.9x slower** — reasonable given MI50 INT8 vs RTX 3090 INT8 hardware.
4. ROCm GEMM call count (43.7K) vs CUDA (76.2K) — different dispatch granularity.

---

## Tuning Roadmap

### Phase 1: ROCm Flash Attention Decode (COMPLETED)
- [x] **v2: float4 + parallel output** — 1,116 → 382 µs/call (2.92x, +40% decode tok/s)
- [x] **v3: wavefront-cooperative KV** — 382 → 57.6 µs/call (6.6x, +71% decode tok/s from baseline)
- [x] **exp2f hardware intrinsic** — 57.6 → 56.2 µs/call (2.4% kernel improvement, negligible end-to-end)
- [x] **GQA-aware blocks** — ATTEMPTED & REVERTED: 5.1× regression due to CU starvation (32 vs 224 blocks on 60 CUs)

### Phase 2: ROCm GEMM Decode (IN PROGRESS — 89% of decode time)
- [x] **Fused FP32×INT8 GEMV** — ATTEMPTED & REVERTED: 9.8% regression due to gfx906 float atomicAdd CAS loop + no v_dot4 in FP32 path
- [x] **v_dot4_i32_i8 intrinsic** — 303 → 286 µs/call avg (-5.6%), decode 30.31 → 32.10 tok/s (+5.9%)
- [x] **GEMV tile/occupancy tuning** — CPT=4→2 (full wavefront), KB formula tuned per shape; +0.4% decode (32.30→32.43 tok/s)
- [x] **Kill wide2_gemv + adaptive KB** — Route FFN Gate/Up through grid_kpar (was 17% peak BW, now 48-63%); adaptive KB for large-N vs small-N shapes; **+74.8% decode (32.43→56.69 tok/s)**
- [ ] **Multi-projection GEMV** — batch Q+K+V in one kernel launch to reduce launch overhead
- [ ] **GEMM prefill repack overhead** — VNNI→row-major repack runs PER CK GEMM call (~65-200µs each)
- [x] **Assembly sweep (rsqrtf, rintf)** — RMS Norm rsqrtf + Quantize rintf, +0.6% decode (32.10→32.30 tok/s)
- [ ] Ops kernels (RoPE, SwiGLU) — low priority, <2% of time

### Phase 2: CUDA Kernel Tuning  
- [ ] CUTLASS GEMM tile configuration
- [ ] Flash Attention decode optimisation
- [ ] Fused kernel opportunities

---

## Architecture Notes

### ROCm GEMM Kernel Dispatch

```
multiply_tensor() / multiply_fused_tensor()
  ├── M == 1  →  GEMV fast path (reads VNNI layout directly)
  │     ├── rocmQuantGemm_quantizeActivations()   FP32 → INT8 + scale
  │     ├── rocmGemv_int8_int8_int32_vnni()       INT8×INT8 GEMV
  │     └── rocmQuantGemm_applyScaling()          INT32 → FP32
  │
  └── M > 1  →  CK GEMM path (needs row-major)
        ├── rocmQuantGemm_quantizeActivations()   FP32 → INT8 + scale
        ├── rocmGemv_repackVNNI_to_rowmajor()     ⚠️ 65-200µs PER CALL
        ├── rocmQuantGemm_executeNoScale()         CK INT8×INT8→INT32
        └── rocmQuantGemm_applyScaling()           INT32 → FP32
```

### Weight Storage (Option B — VNNI-only on device)

- **VNNI layout** `[K/4 × N × 4]` — sole persistent device copy, optimal for GEMV decode
- **Row-major scratch** `d_B_rowmajor_scratch` — workspace buffer, repacked on-demand for CK
- Dual-layout would cost ~2× VRAM for weights but eliminate ~15-25ms repack per prefill

### Repack Cost Analysis (Qwen 7B, 28 layers)

Per layer in prefill:
- QKV fused (3 proj): `3584×3584` × 3 = 3 repacks
- Wo projection: `3584×3584` × 1 = 1 repack  
- GateUp fused (2 proj): `18944×3584` × 2 = 2 repacks (expensive!)
- Down projection: `3584×18944` × 1 = 1 repack (expensive!)

Per prefill: **28 × ~7 = ~196 repacks**  
At ~65-200µs each: **~15-35ms pure repack overhead**  
(Actual total prefill GEMM time on ROCm: 2,306ms — repack is ~1-1.5%, not dominant)

### Call Count Discrepancy

ROCm GEMM: 43.7K calls — profiled at `multiply_tensor()` entry  
CUDA GEMM: 76.2K calls — profiled per-CUTLASS-call inside projection loops  
Ratio: 1.74× matches average ~1.7 sub-projections per fused call (QKV=3, GateUp=2)

---

## Tuning Log

### Entry 10: Kill wide2_gemv + Adaptive KB — GEMV Dispatch Overhaul (2026-02-10)

**Target**: `wide2_gemv` kernel dispatch in `ROCmGemvKernel.hip` — FFN Gate/Up projections (2 of 7 per decode layer)

**Problem**: Profiling with rocprof/rocprofv3 (see Appendix A) revealed `wide2_gemv` was catastrophically underperforming at **17% of peak HBM bandwidth** (176 GB/s vs 1024 GB/s theoretical). For FFN Gate/Up (N=18944, K=3584):
- `wide2` created only `ceil(18944/256) = 74 blocks` with 128 threads = **148 waves for 60 CUs**
- Each thread processed the entire K dimension (896 k-groups) without K-splitting
- No K-parallel decomposition → insufficient wavefronts to hide HBM2 latency
- Meanwhile, `grid_kpar` on FFN Down (same N×K product, transposed) achieved **63% peak BW**

**Root Cause**: The dispatch condition `use_wide = (N >= 3*K)` steered all "wide" shapes away from `grid_kpar`. This was designed to avoid atomicAdd overhead, but atomicAdd is cheap (L2-cached for output arrays <8MB) while having too few wavefronts is fatal for bandwidth-bound GEMV.

**Changes**:

1. **Dispatch overhaul**: Eliminated `use_wide`/`use_wide2` — all shapes with N%4==0 now route through `grid_kpar`:
   ```
   Before: wide → wide2 (N<8K) or wide_vec4 (N≥8K)
           non-wide → grid_kpar
   After:  N≥8K → wide_vec4 (LM Head, already 67% peak)
           N%4==0 → grid_kpar (everything else, K-splitting)
           fallback → square → basic
   ```

2. **grid_kpar kernel optimizations**:
   - Vectorized `int2` B-loads for CPT=2: load two adjacent VNNI columns in one 8-byte memory transaction
   - Partial loop unrolling (`#pragma unroll 4`) for better memory-level parallelism
   - Precomputed B row pointer to reduce address arithmetic

3. **Adaptive KB heuristic** (three-tier):
   - K > 8192: `KB = max(4, min(32, K/512))` — FFN Down path, unchanged
   - K ≤ 8192, grid_n ≥ 64: `KB = k_groups/32` — cap KB for efficiency (28 for K=3584). Many N-tiles provide occupancy; fewer K-splits reduce atomicAdd contention.
   - K ≤ 8192, grid_n < 64: `KB = min(64, (K+63)/64)` — maximize K-splits for occupancy (56 for K=3584 Q/Wo). Few N-tiles need aggressive K-splitting.

**Kernel timing results** (rocprofv3, Qwen 7B shapes, warm):

| Shape | Before | After | Speedup |
|---|---|---|---|
| FFN Gate (18944×3584) | 387 us (wide2) | **121 us** (grid_kpar KB=28) | **3.2×** |
| FFN Up (18944×3584) | 385 us (wide2) | **123 us** (grid_kpar KB=28) | **3.1×** |
| Q/Wo proj (3584×3584) | 24 us (grid_kpar KB=56) | **24 us** (grid_kpar KB=56) | unchanged |
| FFN Down (3584×18944) | 105 us (grid_kpar KB=32) | **105 us** (grid_kpar KB=32) | unchanged |
| LM Head (152064×3584) | 797 us (wide_vec4) | **800 us** (wide_vec4) | unchanged |

**End-to-end results**:

| Metric | Baseline (Entry 9) | This Entry | Change |
|---|---|---|---|
| **Decode** | 32.43 tok/s | **56.69 tok/s** | **+74.8%** |
| **Prefill** | - | 173.0 tok/s | (not compared) |
| **Correctness** | ✓ | ✓ | All 16 shapes pass (0.5B + 7B), cosine sim ≥ 0.99999 |

**Why massive end-to-end impact**: FFN Gate and FFN Up are the two largest GEMV calls per layer (N=18944 > all other projections). With 28 layers in Qwen 7B, the per-layer savings of ~528 us × 28 layers = **14.8 ms/token**, cutting decode time from ~30.8 ms to ~18.3 ms per token.

### Entry 9: GEMV Tile/Occupancy Tuning — CPT + KB Optimization (2026-02-10)

**Target**: `grid_kpar_t<TN,CPT>` kernel in `ROCmGemvKernel.hip` — handles QKV, Wo, FFN Down projections (84 of 113 GEMV calls per decode token)

**Problem**: The production kernel used `TN=128, CPT=4` → `128/4 = 32 threads per block`, which is only **half a wavefront** on GCN (64-wide). This wastes 50% of SIMD lanes per block. Additionally, the KB formula `max(2, min(64, (K+63)/64))` gave KB=64 for FFN Down (K=18944), creating excessive atomicAdd contention with 64 competing writers per output element.

**Analysis**: Created `Perf__GemvTileOccupancy.hip` benchmark sweeping 11 TN/CPT configurations × 9 KB values across all 5 Qwen2.5-7B projection shapes. Assembly analysis confirmed:
- Inner loop: `global_load_dwordx4` (coalesced), `s_load_dword` (scalar A broadcast), 4× `v_dot4_i32_i8` — compiler already optimal
- Register usage: 13 VGPRs, 13 SGPRs — max occupancy regardless of config
- Main tuning variable: wavefront utilization and atomicAdd contention

**Microbenchmark results (isolated GEMV kernel, MI50)**:

| Shape | Old (128/4, KB=56/64) | Best (128/2, tuned KB) | Change |
|---|---|---|---|
| QKV (4608×3584) | 31.5 µs @ KB=56 | 29.5 µs @ KB=128 | **-6.3%** |
| Wo (3584×3584) | 27.0 µs @ KB=56 | 25.0 µs @ KB=96 | **-7.4%** |
| FFN Down (3584×18944) | 92.2 µs @ KB=64 | 91.5 µs @ KB=32 | **-0.8%** |

**Changes applied**:
1. **CPT: 4 → 2** — `GEMV_INT8_VNNI_GRID_KPAR_CPT = 2` gives 64 threads = 1 full wavefront per block
2. **KB formula**: Conservative two-branch heuristic:
   - K ≤ 8192: `max(32, min(64, (K+63)/64))` → KB=56 for QKV/Wo (unchanged)
   - K > 8192: `max(4, min(32, K/512))` → KB=32 for FFN Down (reduced from 64)

**End-to-end results**:

| Metric | Baseline (asm sweep) | Tile/Occupancy | Change |
|---|---|---|---|
| **Decode (non-profiled)** | 32.30 tok/s | **32.43 tok/s** | **+0.4%** |
| **GEMM_CK decode avg** | 285.3 µs | **285.4 µs** | ~0% (within noise) |
| **Parity** | ✓ | ✓ | Identical output |

**Why small end-to-end impact**: The GEMV kernel is only one of three kernels in the GEMM pipeline (quantize → GEMV → scale). The GEMV accounts for ~50-70% of the GEMM call, and only grid_kpar shapes (3 of 5 projections) are affected. Expected per-token savings of ~250µs out of ~31,000µs decode time = ~0.8% theoretical, of which ~0.4% was measured.

**Conclusion**: Modest but correct improvement. The half-wavefront issue was real but had limited end-to-end impact because the GEMV inner loop is already compute-efficient (v_dot4 saturated). The FFN Down KB reduction from 64→32 is the primary contributor, reducing atomicAdd contention. No further tile/occupancy tuning opportunities remain for the grid_kpar kernel — the bottleneck is now memory bandwidth (745 GB/s = 72% of theoretical 1024 GB/s on MI50).

### Entry 8: Assembly Sweep — rsqrtf + rintf Intrinsics (2026-02-10)

**Target**: All 9 ROCm HIP kernel files, compiled to assembly for systematic analysis

**Approach**: Compiled every `.hip` kernel to GCN assembly (`-S -O3 --offload-arch=gfx906`) and searched for suboptimal instruction patterns where the compiler failed to use available hardware instructions.

**Files analyzed**:
| File | Assembly Size | Status |
|---|---|---|
| ROCmGemvKernel.hip | 2.4K lines | Already optimized (Entry 7) |
| ROCmQuantisedGemmKernel_CK.hip | 60K lines | **FIX: rintf** |
| ROCmRMSNormKernels.hip | 1.2K lines | **FIX: rsqrtf** |
| ROCmSwiGLUKernels.hip | 800 lines | Clean — v_exp_f32 used |
| ROCmFlashAttentionKernels.hip | 3K lines | Clean — v_exp_f32 used |
| ROCmRoPEKernels.hip | 12K lines | Low priority — sin/cos in loading phase |
| ROCmResidualAddKernels.hip | 300 lines | Trivial — no opportunities |
| ROCmEmbeddingKernels.hip | 400 lines | Clean — v_rcp_f32 expected |

**Finding 1: RMS Norm `sqrtf()` + `1/rms` → `rsqrtf()`**

Three RMS Norm kernels (FP32, BF16, FP16) computed `float rms = sqrtf(x); float inv_rms = 1.0f / rms;` which compiled to:
- `v_sqrt_f32` (25 cycles) + `v_rcp_f32` (4 cycles + Newton-Raphson refinement) = ~35 cycles

Replaced with `float inv_rms = rsqrtf(x)` → single `v_rsq_f32` (4 cycles). Assembly verified: `v_sqrt: 3→0`, `v_rsq: 0→3`, `v_rcp: 6→3`.

Applied to both ROCm (3 kernels) and CUDA (3 kernels, maps to MUFU.RSQ).

**Finding 2: Quantize `roundf()` → `rintf()`**

The Q8 activation quantize kernel used `roundf(scaled)` which decomposed into 6 scalar instructions:
```
v_trunc_f32    ; truncate to integer
v_sub_f32      ; compute fraction = x - trunc(x)
v_cmp_ge_f32   ; compare |fraction| >= 0.5
v_cndmask_b32  ; select rounding direction
v_bfi_b32      ; copy sign bit
v_add_f32      ; add rounding correction
```

Replaced with `rintf(scaled)` → single `v_rndne_f32` (1 cycle). Loop body reduced from 8 to 4 compute instructions.

For INT8 quantization, the difference between round-half-away-from-zero (`roundf`) and round-to-nearest-even (`rintf`) affects only exact x.5 tie points — negligible quality impact. CUDA quantize already used `rintf`.

**Finding 3: Flash Attention `1/sqrtf(head_dim)` — NOT applicable**

Identified `1.0f / sqrtf(head_dim)` in flash attention, but this is in **host-side** launch functions (computed once per kernel launch, not per-thread). `rsqrtf()` is not available in HIP host compilation. No performance impact since this runs once on CPU.

**Results**:

| Metric | Baseline (v_dot4) | Assembly Sweep | Change |
|---|---|---|---|
| **Decode (non-profiled)** | 32.10 tok/s | **32.30 tok/s** | **+0.6%** |
| **GEMM_CK decode avg** | 286 µs | **285.3 µs** | -0.2% (noise) |
| **RMS Norm decode total** | ~425 ms | **413 ms** | **-2.8%** |
| **Parity** | ✓ | ✓ | Identical output |

**Conclusion**: Small but correct improvements. RMS Norm and Quantize are <4% of decode time combined, so even significant per-kernel speedups translate to minimal end-to-end gains. The sweep confirmed that the compiler is already doing a good job on the hot-path kernels (GEMV, SwiGLU, Flash Attention). No further assembly-level intrinsic opportunities remain in the current kernel set.

### Entry 7: v_dot4_i32_i8 Hardware Intrinsic for GEMV (2026-02-10)

**Target**: All 5 GEMV kernel variants in `ROCmGemvKernel.hip`

**Discovery**: `llvm-objdump` confirmed **zero** `v_dot4_i32_i8` instructions in the entire binary. The compiler was generating individual `v_mul_i32_i24` (24-bit multiply) instructions (~7 cycles per 4-element dot product) instead of using the gfx906 hardware `v_dot4_i32_i8` instruction (1 cycle for 4 multiply-adds).

**Root cause**: The compiler doesn't auto-generate `v_dot4_i32_i8` from manual INT8 unpack + multiply-add C++ code. The code was:
```cpp
const int32_t a0 = static_cast<int32_t>(d_A_int8[k_base + 0]);
// ... 4 individual byte loads, manual unpacking, 8 multiplies + adds
acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
```

**Fix**: Replace with `__builtin_amdgcn_sdot4` intrinsic in all 5 GEMV kernels:
```cpp
const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
const int32_t b_packed = *reinterpret_cast<const int32_t*>(d_B_int8_vnni + ...);
acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
```

**Kernels modified**: `gemv_int8_int8_vnni_kernel` (scalar), `wide_vec4`, `wide2`, `square`, `grid_kpar_t` — all use the same packed dot-product pattern now.

**Verification**: Compiled assembly confirms 16 `v_dot4_i32_i8` instructions (was 0).

**Results**:

| Metric | Baseline | v_dot4 | Change |
|---|---|---|---|
| **Decode (non-profiled)** | 30.31 tok/s | **32.10 tok/s** | **+5.9%** |
| **Decode (profiled)** | 24.29 tok/s | **25.55 tok/s** | **+5.2%** |
| **GEMM_CK avg** | 303 µs | **286 µs** | **-5.6%** |
| **GEMM_CK decode total (3 runs)** | ~10,944 ms | **10,224 ms** | **-6.6%** |
| **Parity** | ✓ | ✓ | Identical output |

**Post-v_dot4 decode breakdown** (profiled):

| Kernel | Time (ms, 3 runs) | % of decode |
|---|---|---|
| GEMM_CK | 10,224 | 88% |
| Flash Attn Decode | 605 | 5% |
| RMS Norm | 425 | 3% |
| Residual Add | 165 | 1% |
| RoPE | 128 | 1% |

**Conclusion**: Clean 5-6% decode improvement with zero risk — same math, same data types, same output. The hardware INT8 dot product instruction was available on gfx906 but the compiler never auto-vectorized to use it. Explicit intrinsic forces the optimal instruction.

### Entry 6: Fused FP32×INT8→FP32 GEMV — ATTEMPTED & REVERTED (2026-02-10)

**Target**: `ROCmGemvKernel.hip` + `ROCmQuantisedGemmKernel.cpp`

**Hypothesis**: Eliminating the 3-kernel quantize→GEMV→scale pipeline into a single fused kernel would save ~308 kernel launches per decode step and remove intermediate INT8/INT32 buffers from the memory path.

**Implementation**: Two new kernels (`gemv_fp32xi8_fused_kpar_vnni_kernel_t`, `gemv_fp32xi8_fused_wide2_vnni_kernel`) + dispatch wrapper + wired into all 3 M=1 decode paths.

**Results**: **9.8% regression — REVERTED**

| Metric | Baseline | Fused (reverted) | Change |
|---|---|---|---|
| **Decode (non-profiled)** | 30.31 tok/s | **27.35 tok/s** | **-9.8%** |
| **Decode (profiled)** | 24.29 tok/s | **22.44 tok/s** | **-7.6%** |
| **Parity** | ✓ | ✓ | Correct output |

**Root cause analysis**:
1. **gfx906 lacks native `global_atomic_add_f32`** — float `atomicAdd` uses a CAS loop, catastrophically slow with 56-way K-block contention per output element. The INT32 path uses native `atomicAdd` which is 10-50× faster.
2. **FP32×INT8 cannot use `v_dot4_i32_i8`** — needs individual int8→float conversions + FP32 FMA (~12 cycles) vs INT8×INT8 manual multiply-add (~7 cycles) or `v_dot4_i32_i8` (1 cycle).
3. **4× larger activation loads** — FP32 activations are 14KB vs 3.6KB INT8 for K=3584, causing L1 cache pressure.

**Lesson**: On gfx906, the separate quantize+INT8_GEMV+scale pipeline is optimal because: (a) INT32 atomicAdd is native, (b) INT8×INT8 can use v_dot4, (c) INT8 activations are 4× smaller. The quantize+scale kernel overhead (~2-3% of GEMM time) is much less than the compute regression from FP32 operations.

### Entry 5: exp2f Hardware Intrinsic (2026-02-09)

**Target**: `flash_decoding_mi50_kernel` in `ROCmFlashAttentionKernels.hip`

**Changes**:
1. **`fast_expf()` inline helper** — `exp2f(x * 1.4426950408889634f)` replaces all `expf()` calls
2. On GCN ISA, `exp2f` maps to `v_exp_f32` (single-cycle transcendental ALU instruction), vs `expf` which requires a multi-step polynomial (~50+ cycles)
3. Applied to: main loop softmax (2 calls/KV pos), inter-wavefront reduction (2 calls/wavefront pair)
4. Accuracy: ~2 ULP vs ~1 ULP for `expf`, acceptable for attention softmax

**Results**:

| Metric | v3 (expf) | v3+exp2f | Change |
|---|---|---|---|
| **Flash Attn Decode avg** | 57.6 µs | **56.2 µs** | **-2.4%** |
| **Flash Attn Decode total** | 619 ms | **604 ms** | -15 ms |
| **Decode throughput** | ~24.3 tok/s | ~24.3 tok/s | within noise |

**Conclusion**: Measurable 2.4% per-call improvement to the kernel. End-to-end impact negligible since flash attention is only ~2% of decode time. Kept as a zero-risk improvement.

### Entry 4: GQA-Aware Blocks — ATTEMPTED & REVERTED (2026-02-09)

**Target**: `flash_decoding_mi50_kernel` in `ROCmFlashAttentionKernels.hip`

**Architecture change**: Grid indexed by KV heads instead of Q heads. Each block processes all 7 Q heads that share a KV head, loading K/V once.

**Changes attempted**:
1. Grid changed from `(n_heads=28, num_splits=8, batch=1)` = 224 blocks to `(n_kv_heads=4, num_splits=8, batch=1)` = 32 blocks
2. All 7 Q vectors loaded into shared memory (3.5KB vs 512B)
3. K/V cached in lane-local registers, reused across 7 Q heads
4. 7× wavefrontReduceSum per KV position (dot product for each Q head)
5. Inter-wavefront reduction processed one Q head at a time (3 syncs × 7 heads)

**Results**: **5.1× regression — REVERTED**

| Metric | v3 | GQA (reverted) | Change |
|---|---|---|---|
| **Flash Attn Decode avg** | 57.6 µs | **296.1 µs** | **5.1× slower** |
| **Decode throughput** | ~24.3 tok/s | **20.29 tok/s** | **-16.5%** |

**Root cause**: CU starvation. MI50 has 60 CUs. With 32 blocks, only 53% of CUs are active (vs 224/60 = 3.7 blocks/CU in v3). The 7× K/V bandwidth savings cannot compensate for the massive parallelism loss. Additionally, `FD_MAX_GQA_RATIO=16` array sizes (64+16+16 = 96 VGPRs for arrays alone) likely caused register spilling, further degrading performance.

**Lesson**: For short kernels on wide GPUs (60 CUs), grid occupancy dominates bandwidth savings. GQA-aware blocking only wins when: (a) the kernel is bandwidth-bound AND (b) there are enough blocks to fill the GPU. With 4 KV heads × 8 splits = 32 blocks, MI50 is fundamentally under-utilised. Would need num_splits ≈ 56 to maintain 224 blocks, but at kv_len=600 that's only ~11 KV positions per split — too small for efficient wavefront-cooperative processing.

### Entry 3: Flash Attention Decode v3 — Wavefront-Cooperative (2026-02-09)

**Target**: `flash_decoding_mi50_kernel` in `ROCmFlashAttentionKernels.hip`

**Architecture change**: Complete repartitioning — lanes own output dims, not KV positions.

**Changes**:
1. **Wavefront-cooperative KV processing** — each wavefront cooperates on one KV position at a time (all 64 lanes load contiguous K/V elements). Was: each thread independently loaded from separate KV positions (uncoalesced)
2. **Lanes own output dims** — each lane holds only 2 floats of O (for head_dim=128), down from 128 floats per thread. Drops VGPR usage from ~140 to ~15
3. **Zero O reduction within wavefront** — since all lanes process the same KV positions cooperatively, no cross-lane O shuffling is needed. Was: 128 × wavefrontReduceSum = 768 `__shfl_xor` per lane
4. **Dot product via wavefront reduce** — only 6 shuffles per KV position (one `wavefrontReduceSum` for the partial dot)
5. **Coalesced memory access** — adjacent lanes load adjacent floats from K and V. Perfect coalescing vs the old strided-KV-position pattern
6. **`__launch_bounds__(256, 4)`** — higher occupancy hint (was 2) since register pressure dropped dramatically
7. **Correct `__syncthreads` placement** — barriers moved outside `if` blocks (old v2 had UB with sync inside `if (wavefront_id == 0)`)

**Results**:

| Metric | Baseline (v1) | v2 | v3 | v1→v3 |
|---|---|---|---|---|
| **Decode throughput** | 14.2 tok/s | 19.86 tok/s | **~24.3 tok/s** | **+71%** |
| **Flash Attn Decode total** | 12,003 ms | 4,107 ms | **619 ms** | **-94.8%** |
| **Flash Attn Decode avg** | 1,116 µs | 382 µs | **57.6 µs** | **19.4x** |
| **Flash Attn % of decode** | 51% | 15% | **2.7%** | Eliminated as bottleneck |
| **Decode wall-clock** | ~9,014 ms | ~6,446 ms | **~5,273 ms** | **-41.5%** |

**Post-v3 decode breakdown**:

| Kernel | Time (ms) | % of decode |
|---|---|---|
| GEMM_CK | 10,954 | 89% |
| Flash Attn Decode | 619 | 5% |
| RMS Norm | 412 | 3% |
| Residual Add | 158 | 1% |
| RoPE | 124 | 1% |
| SwiGLU | 80 | <1% |

**Conclusion**: Flash attention is no longer a meaningful optimisation target. GEMM is 89% of decode. Further attention work (GQA-aware blocks, exp2f) deprioritised — combined they'd yield <6% decode improvement.

### Entry 2: Flash Attention Decode Optimisation — ROCm (2026-02-09)

**Target**: `flash_decoding_mi50_kernel` in `ROCmFlashAttentionKernels.hip`

**Changes**:
1. **float4 vectorized K/V loads** — dot product and V accumulation now use `float4`, reducing memory transactions by 4×
2. **Deferred O_local rescale** — online softmax `O_local[128]` rescale fused with V accumulation (single `scale_old * o + p * v` pass)
3. **Efficient wavefront reduction** — new `wavefrontReduceML()` reduces ONLY m and l across 64 lanes (12 `__shfl_xor` calls instead of 780). Each thread then rescales its own O_local to the converged m
4. **Parallel output write** — all lanes in wavefront 0 write their portion of head_dim (was: thread 0 serial loop over 128 elements)
5. **`__launch_bounds__(256, 2)`** on decode kernel — guides compiler register allocation for better occupancy
6. **`__launch_bounds__(256, 4)`** on reduce kernel — reduce kernel is register-light, can support higher occupancy
7. **Hoisted KV pointer arithmetic** — `kv_head_offset` and `kv_stride` computed once outside loop

**Results**:

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| **Decode throughput** | 14.2 tok/s | 19.86 tok/s | **+40%** |
| **Flash Attn Decode total** | 12,003 ms | 4,107 ms | **-65.8%** |
| **Flash Attn Decode avg** | 1,116 µs | 382 µs | **2.92x faster** |
| **Decode GPU time share** | 51% | 15% | GEMM now dominant |
| **GEMM decode** | 10,847 ms | 10,904 ms | unchanged |
| **Decode wall-clock** | ~9,014 ms | ~6,446 ms | **-28.5%** |

**Post-optimisation decode breakdown**:

| Kernel | Time (ms) | % of decode |
|---|---|---|
| GEMM_CK | 10,904 | 69% |
| Flash Attn Decode | 4,107 | 26% |
| RMS Norm | 416 | 2.6% |
| Residual Add | 160 | 1.0% |
| RoPE | 124 | 0.8% |
| SwiGLU | 81 | 0.5% |

**Next target**: GEMM decode is now the clear bottleneck (69% of decode GPU time).

### Entry 1: Baseline (2026-02-09)
- Established baseline profiling with phase-separated GPU event timing
- Fixed ROCm kernel profiling (was using CPU chrono instead of HIP events)
- Added prefill/decode phase breakdown to profiling output

---

## Appendix A: ROCm GEMV Kernel Profiling

### Test Binary & Filters

```bash
# Test binary (Release build)
build_v2_release/tests/v2/v2_perf_rocm_gemv_kernel

# Available test filters
--gtest_filter=ROCmGemvPerfTest.Correctness_Qwen05B   # 0.5B shapes (N≤4864)
--gtest_filter=ROCmGemvPerfTest.Correctness_Qwen7B    # 7B shapes (N≤152064)
--gtest_filter=ROCmGemvPerfTest.Benchmark_Qwen7B_DecodeLayer  # Timed benchmark
```

### rocprofv3 Kernel Trace (timing + grid/register info)

```bash
# Kernel trace — writes SQLite DB to <hostname>/<pid>_results.db
rocprofv3 --kernel-trace -- ./build_v2_release/tests/v2/v2_perf_rocm_gemv_kernel \
  --gtest_filter=ROCmGemvPerfTest.Correctness_Qwen7B

# Query results (python3):
python3 -c "
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
tables = [t[0] for t in cur.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]
kd = [t for t in tables if 'kernel_dispatch' in t][0]
ks = [t for t in tables if 'kernel_symbol' in t][0]
rows = cur.execute(f'''
    SELECT ks.kernel_name, (kd.end-kd.start)/1000.0 as dur_us,
           kd.workgroup_size_x, kd.grid_size_x, kd.grid_size_y,
           ks.sgpr_count, ks.arch_vgpr_count
    FROM {kd} kd JOIN {ks} ks ON kd.kernel_id = ks.id ORDER BY kd.start
''').fetchall()
for r in rows:
    print(f'{r[1]:8.1f}us  grid={r[3]}x{r[4]}  block={r[2]}  SGPR={r[5]}  VGPR={r[6]}  {r[0][:60]}')
" <hostname>/<pid>_results.db
```

### rocprof v1 PMC Counter Collection

rocprofv3 PMC collection hangs on gfx906; use rocprof v1 (`/opt/rocm/bin/rocprof`).
gfx906 has limited single-pass counter capacity — collect in separate passes.

```bash
# Pass 1: Instruction mix (SQ counters)
cat > /tmp/rocprof_p1.txt << 'EOF'
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_INSTS_LDS
EOF
/opt/rocm/bin/rocprof --timestamp on -i /tmp/rocprof_p1.txt -o /tmp/results_p1.csv \
  ./build_v2_release/tests/v2/v2_perf_rocm_gemv_kernel \
  --gtest_filter=ROCmGemvPerfTest.Correctness_Qwen7B

# Pass 2: L2 cache hit/miss
cat > /tmp/rocprof_p2.txt << 'EOF'
pmc: TCC_HIT_sum TCC_MISS_sum
EOF
/opt/rocm/bin/rocprof --timestamp on -i /tmp/rocprof_p2.txt -o /tmp/results_p2.csv \
  ./build_v2_release/tests/v2/v2_perf_rocm_gemv_kernel \
  --gtest_filter=ROCmGemvPerfTest.Correctness_Qwen7B

# Pass 3: SALU/SMEM/FLAT instruction counts
cat > /tmp/rocprof_p3.txt << 'EOF'
pmc: SQ_INSTS_SALU SQ_INSTS_SMEM SQ_INSTS_FLAT
EOF
/opt/rocm/bin/rocprof --timestamp on -i /tmp/rocprof_p3.txt -o /tmp/results_p3.csv \
  ./build_v2_release/tests/v2/v2_perf_rocm_gemv_kernel \
  --gtest_filter=ROCmGemvPerfTest.Correctness_Qwen7B
```

**Counters NOT supported on gfx906**: `SQ_WAIT_ANY`, `SQ_WAIT_INST_LDS`, `TCP_TOTAL_CACHE_ACCESSES_sum`.
**Derived metrics** (`FETCH_SIZE`, `WRITE_SIZE`) exceed HW limits — use `TCC_HIT_sum`/`TCC_MISS_sum` instead.

### Analysis Script

```bash
python3 << 'PYEOF'
import csv

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def safe_int(v): return int(float(v))

p1 = load_csv("/tmp/results_p1.csv")
p2 = load_csv("/tmp/results_p2.csv")

for i, r in enumerate(p1):
    name = r['KernelName'].strip('"').replace('.kd','').split('(')[0]
    if 'copyBuffer' in name or 'fillBuffer' in name: continue
    dur_us = (int(r['EndNs']) - int(r['BeginNs'])) / 1000.0
    waves = safe_int(r['SQ_WAVES'])
    valu = safe_int(r['SQ_INSTS_VALU'])
    vmem_rd = safe_int(r['SQ_INSTS_VMEM_RD'])
    tcc_hit = safe_int(p2[i]['TCC_HIT_sum']) if i < len(p2) else 0
    tcc_miss = safe_int(p2[i]['TCC_MISS_sum']) if i < len(p2) else 0
    l2_total = tcc_hit + tcc_miss
    l2_pct = 100.0 * tcc_hit / l2_total if l2_total else 0
    miss_bytes = tcc_miss * 64  # 64B cache line on gfx906
    bw_gbps = miss_bytes / (dur_us * 1e-6) / 1e9 if dur_us > 0 else 0
    print(f"{dur_us:8.1f}us  waves={waves:5}  VALU/w={valu/waves if waves else 0:6.0f}"
          f"  VMEM_R/w={vmem_rd/waves if waves else 0:5.0f}  L2Hit={l2_pct:5.1f}%"
          f"  HBM_BW={bw_gbps:6.0f} GB/s  {name[:50]}")
PYEOF
```

### Profiling Results (2026-02-09, Qwen 7B shapes)

| Kernel | Shape (NxK) | Duration | Waves | L2 Hit% | HBM BW (GB/s) | % Peak (1024) |
|--------|-------------|----------|-------|---------|----------------|---------------|
| `grid_kpar_gemv` | Q/Wo proj (3584×3584) | 26 us | 1568 | 11% | **488** | 48% |
| `grid_kpar_gemv` | K/V proj (512×3584) | 12 us | 224 | 12% | 148 | 15% |
| `grid_kpar_gemv` | FFN Down (3584×18944) | 105 us | 896 | 1.7% | **649** | 63% |
| `wide2_gemv` | FFN Gate/Up (18944×3584) | 385 us | 148 | 0.3% | **176** | **17%** |
| `wide_vec4_gemv` | LM Head (152064×3584) | 797 us | 594 | 0.4% | **686** | 67% |

**Key findings**:
- `wide2_gemv` (FFN Gate/Up) is severely underperforming at 17% of peak BW — biggest optimization target
- All GEMV kernels use 0 LDS and only 12-20 VGPRs — occupancy is NOT register-limited
- L2 hit rates are very low (<12%), confirming pure HBM streaming workload
- Register usage: grid_kpar=12 VGPR, wide2=12 VGPR, wide_vec4=20 VGPR (out of 256)
