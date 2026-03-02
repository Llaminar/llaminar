# Native-VNNI: Lossless Q4/IQ4 GEMV for ROCm

## Executive Summary

Replace the **ratio-VNNI** encoding with a simpler, higher-accuracy **native-VNNI** format for ≤6-bit quantized weights on AMD GPUs. The new format preserves the VNNI memory coalescing benefits while achieving **lossless weight reconstruction** (cosine = 1.000000 vs 0.986 for ratio-VNNI grouped).

### Motivation

Ratio-VNNI was designed to fold per-block FP16 scales into an INT8 "ratio" that could ride alongside the 4-bit payload, enabling `sdot4` (v_dot4_i32_i8) to compute the dot product in a single fused operation. However, the `(ratio * d + 64) >> 7` truncation step introduces ~1.4–5% per-element reconstruction error that compounds through 24 transformer layers, yielding decode parity of only ~0.85 cosine vs PyTorch ground truth.

Native-VNNI eliminates this error source entirely by keeping per-block scales separate and decoding 4-bit values directly to INT8 (`d = q - 8`, which is exact). The `sdot4` instruction is still used — it just operates on losslessly-decoded INT8 weights instead of lossy ratio-encoded ones.

### Empirical Results (Python Playground — `playground_vnni.py`)

Tested on real Q4_0 weights from `qwen2.5-0.5b-instruct-q4_0.gguf`, layer 0 `attn_q` (896×896):

| Approach | Weight Cosine | GEMV Cosine | Eff. Bits/Element |
|---|---|---|---|
| Ratio-VNNI (grouped G=4) | 0.9862 | 0.9835 | 4.50 |
| Ratio-VNNI (per-row) | 0.9521 | 0.9471 | 4.29 |
| **Native-VNNI (FP16 scale)** | **1.000000** | **0.999961** | **4.50** |
| Native-VNNI (FP32 scale) | 1.000000 | 0.999961 | 5.00 |

FP16 and FP32 per-block scales produce identical results because Q4_0 scales were already FP16 in the original format. Native-VNNI at 4.50 bits/element achieves the same effective size as ratio-VNNI grouped — with perfect weight fidelity.

---

## Format Specification

### Native-VNNI Memory Layout (Q4_0 / IQ4_NL)

**Three separate device arrays per weight matrix [N × K]:**

```
1. d_payload:     uint8_t[blocks_per_row × N × 16]     — 4-bit quant nibbles in VNNI-coalesced order
2. d_block_scales: __half[blocks_per_row × N]           — FP16 per-block scale (native from Q4_0)
3. d_codebook_id: uint8_t                               — 0 = linear (Q4_0), 4 = IQ4 (IQ4_NL)
```

**Block layout** (32 elements per block):
- `blocks_per_row = K / 32`
- Payload is interleaved by N: block (b, n) stored at `d_payload[(b * N + n) * 16]`
- Scales are interleaved by N: scale for block (b, n) at `d_block_scales[b * N + n]`
- This gives coalesced reads when threads process adjacent output rows (n, n+1, …)

**Weight decode** (on GPU, per element):
```
Q4_0:   w_int8 = q - 8                     // q ∈ [0,15] → w ∈ [-8,+7], EXACT
IQ4_NL: w_int8 = iq4_lut[q]                // q ∈ [0,15] → w ∈ [-127,+113], EXACT
```

**GEMV compute** (M=1 decode):
```
for each block b:
    int32_block = Σ sdot4(a_int8[b*32..], w_int8[b*32..])    // 8 sdot4 calls per block
    f_acc += int32_block * block_scale[b]                      // ONE FP32 multiply per block
result = f_acc * scale_A                                       // ONE FP32 multiply per row
```

### Memory Footprint Comparison

| Format | Bits/Element | 896×896 Matrix | 4864×896 Matrix |
|--------|-------------|----------------|-----------------|
| Native Q4_0 (GGUF) | 4.50 | 451 KB | 2,451 KB |
| Full INT8-VNNI | 8.00 | 802 KB | 4,358 KB |
| Ratio-VNNI (grouped G=4) | 4.50 | 451 KB | 2,451 KB |
| **Native-VNNI (FP16 scales)** | **4.50** | **451 KB** | **2,451 KB** |

Breakdown: 4 bits (payload) + 16 bits / 32 elements (FP16 scale) = 4.50 bits/element — identical to the original Q4_0 format, just reordered for GPU coalescing.

---

## Scope

### In Scope (Sprint 1)
- Q4_0 and IQ4_NL weight formats
- Decode GEMV path (M=1) — the performance-critical token-by-token generation path
- Prefill GEMM path (M>1) — via CK (Composable Kernel) with on-the-fly expand to INT8 row-major
- Automatic detection and routing: if weight tensor is Q4_0 or IQ4_NL, pack to native-VNNI and route GEMV/GEMM calls to the new kernels
- Complete removal of all ratio-VNNI infrastructure

### Out of Scope (Future Sprints)
- Q5_0, Q5_1, Q4_1 formats (5-bit or formats with min side-channel)
- Q6_K and other K-quant formats
- Native-VNNI prefill kernel (currently using CK expand fallback, which already works)
- CPU-side native-VNNI (x86 VNNI uses different patterns)

---

## Architecture

### Data Flow

```
                    ┌────────────────────┐
                    │   GGUF Q4_0 Tensor │
                    └────────┬───────────┘
                             │
                   packNativeVNNI()
                   (host-side repack)
                             │
                    ┌────────▼───────────┐
                    │  ROCmPackedWeights  │
                    │  .native_vnni_payload│
                    │  .native_vnni_scales │
                    │  .native_vnni_codec  │
                    └────────┬───────────┘
                             │
                   H2D upload (async)
                             │
              ┌──────────────▼──────────────┐
              │    Device Memory             │
              │  d_native_vnni_payload       │
              │  d_native_vnni_scales        │
              └──────┬──────────────┬───────┘
                     │              │
            ┌────────▼────┐  ┌─────▼──────────┐
            │  M=1 GEMV   │  │  M>1 GEMM      │
            │  (new kernel│  │  (CK expand     │
            │   sdot4 +   │  │   fallback)     │
            │   FP16 mul) │  │                 │
            └─────────────┘  └────────────────┘
```

### GEMV Kernel Design (M=1 Decode)

```cpp
template <int TILE_N, int CPT, bool IQ4>
__global__ void gemv_native_vnni_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [K] quantized activations
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × 16] nibble-packed
    const __half*    __restrict__ d_block_scales,  // [blocks_per_row × N] FP16
    float*           __restrict__ d_C_fp32,        // [N] output (FP32)
    int N, int K, int kblocks)
{
    // Thread processes CPT output columns
    // Grid K-partitioned: blockIdx.y splits K-dimension across blocks
    // Per-block: decode 32 nibbles → 32 INT8, sdot4 → INT32, × block_scale → FP32
    // Accumulate all blocks within K-partition in FP32
    // AtomicAdd partial FP32 sums across K-partitions
}
```

**Key differences from ratio-VNNI GEMV:**
1. No ratio decode (`>>7` elimination) — nibbles decoded directly to INT8
2. FP32 accumulation per-block instead of per-group or per-row INT32
3. One FP16→FP32 multiply per block (32 elements) — same arithmetic intensity as grouped ratio-VNNI
4. Output is FP32 directly — no separate `applyScaling` or `applyScaleA` epilogue kernel needed

**Epilogue:**
The native-VNNI GEMV accumulates in FP32 and includes `scale_A` application inline. The result is written directly to `d_C_fp32`. No separate epilogue kernel is needed. This eliminates two kernel launches per GEMV call versus ratio-VNNI (which needed GEMV + applyScaling/applyScaleA).

### CK Prefill Path (M>1)

For prefill, we continue to use the CK (Composable Kernel) GEMM with on-the-fly expansion. The expand kernel is simplified:

```cpp
__global__ void expand_native_vnni_to_rowmajor_kernel(
    const uint8_t* __restrict__ d_payload,
    const __half*  __restrict__ d_block_scales,
    int8_t*        __restrict__ d_B_rowmajor,
    float*         __restrict__ d_scales_row,     // per-row scale for CK
    int N, int K, uint8_t codebook_id)
{
    // For each element (n, k):
    //   block = k / 32, nibble_idx = k % 32
    //   q = extract_nibble(d_payload[block * N + n], nibble_idx)
    //   d = (codebook == IQ4) ? iq4_lut[q] : (q - 8)
    //   w_int8 = round(d * block_scale / row_scale * 127)
    //   d_B_rowmajor[n * K + k] = w_int8
}
```

This is a lossy step (re-quantizing FP32 → INT8 with a per-row scale), but it only affects prefill, not the latency-critical decode path.

---

## Implementation Phases

### Phase 1: Host-Side Packing + Data Structures
**Goal:** Pack Q4_0/IQ4_NL weights into native-VNNI format on the host and upload to GPU.

**Tasks:**
1. Add native-VNNI fields to `ROCmPackedWeights` in `ROCmQuantisedGemmKernel.h`:
   - `std::vector<uint8_t> native_vnni_payload`
   - `std::vector<uint16_t> native_vnni_scales` (FP16 stored as uint16_t)
   - `uint8_t native_vnni_codebook_id`
   - Device pointers: `uint8_t* d_native_vnni_payload`, `__half* d_native_vnni_scales`

2. Add `Impl` fields for device pointers + metadata:
   - `uint8_t* d_weights_native_payload`
   - `void* d_weights_native_scales` (half*)
   - `uint8_t native_vnni_codebook_id`
   - `bool has_native_vnni`

3. Add `DeviceUpload` staging fields + pinned memory for async upload

4. Write `packNativeVNNI()` function in `ROCmQuantisedGemmKernel.cpp`:
   - Input: `TensorBase*` (Q4_0 or IQ4_NL)
   - Extracts per-block FP16 scales and 4-bit payload nibbles
   - Interleaves payload by N: `payload[(b * N + n) * 16 + byte]`
   - Interleaves scales by N: `scales[b * N + n]`
   - Called from `packWeightsToROCm()` where `packRatioVNNIPhase1` was previously called

5. Wire upload in `ensureWeightsConverted()`:
   - Allocate device buffers, async H2D copy, commit to `impl_->d_weights_native_*`

6. Destructor/cleanup: free device buffers and pinned staging

**Deliverable:** Native-VNNI packed weights on GPU, ready for kernel consumption. Build succeeds, existing INT8-VNNI paths still work.

### Phase 2: GEMV Decode Kernel
**Goal:** GPU kernel for M=1 decode using native-VNNI format.

**Tasks:**
1. Write `gemv_native_vnni_kernel_t` in `ROCmGemvKernel.hip`:
   - Template parameters: `TILE_N`, `CPT` (columns per thread), `IQ4` (codebook select)
   - Grid K-partitioned (same pattern as existing `gemv_int8_int8_grid_kpar_vnni_kernel_t`)
   - Per-block: decode 32 nibbles → INT8, 8× `sdot4` → INT32, × FP16 block scale → FP32
   - Accumulate partial FP32 sums, atomicAdd across K-partitions
   - Apply `scale_A` inline (no separate epilogue kernel)

2. Write dispatch function `rocmGemv_native_vnni_fp32()`:
   - Memset output to 0
   - Select tile configuration based on N, K
   - Launch kernel
   - Signature: `(stream, d_A_int8, d_payload, d_block_scales, d_C_fp32, N, K, kblocks, codebook_id)`

3. Forward declaration in `ROCmQuantisedGemmKernel.cpp`

**Deliverable:** Standalone GEMV kernel that can be called directly. Not yet wired into dispatch.

### Phase 3: Dispatch Routing
**Goal:** Route Q4_0/IQ4_NL GEMV calls to native-VNNI automatically in all 4 entry points.

**Tasks:**
1. Update decode (M=1) dispatch in all 4 entry points:
   - `multiply_tensor()`
   - `multiply_fused_tensor()`
   - `multiply_fp32_to_fp32()`
   - `multiply_fp32_to_fp32_with_bias()`

   The dispatch logic for each:
   ```cpp
   if (impl_->has_native_vnni) {
       // Quantize activations to INT8
       quantize_activations(...);
       // Native-VNNI GEMV: sdot4 + FP16 block scales → FP32 output
       rocmGemv_native_vnni_fp32(stream,
           d_A_int8, impl_->d_weights_native_payload,
           impl_->d_weights_native_scales, d_C_fp32,
           N, K, kblocks, impl_->native_vnni_codebook_id,
           scale_A_value);
       // Output is already FP32 with scale_A applied — no epilogue needed
   } else if (impl_->d_int8_data_vnni) {
       // INT8-VNNI path (for Q8_0 and formats that pack to full INT8)
       ...
   }
   ```

2. Update prefill (M>1) dispatch:
   - Write `expandNativeVNNI_to_rowmajor()` expand kernel
   - Wire into `ensureRepackedWeightsForCK()` for CK fallback path
   - Remove `RATIO_VNNI_NATIVE` prefill path (this was the direct ratio-VNNI prefill)

**Deliverable:** All GEMV/GEMM calls for Q4_0/IQ4_NL use native-VNNI automatically. Parity tests should show dramatic improvement.

### Phase 4: Remove Ratio-VNNI
**Goal:** Complete removal of all ratio-VNNI code.

**Files to delete entirely:**
- `src/v2/kernels/rocm/ROCmRatioVNNIAbi.h`
- `tests/v2/performance/kernels/rocm/Perf__ROCmRatioVNNIDecodeKernel.cpp`
- `tests/v2/performance/kernels/rocm/Perf__ROCmRatioVNNIPrefillKernel.cpp`
- `scripts/profile_ratio_vnni_rocprof.sh`
- `scripts/bench_ratio_iq4_variants.sh`

**Code to remove from `ROCmQuantisedGemmKernel.h` (~80 lines):**
- `DeviceUpload` fields: `d_ratio_vnni_*`, `d_scales_grouped`, all `startup_h2d_pinned_ratio_*` / `_scales_grouped`
- `ROCmPackedWeights` fields: `scales_grouped`, `ratio_vnni_*` (all 20+ fields), `d_ratio_vnni_*`, `d_scales_grouped`
- Move constructor/assignment: all ratio-vnni field copies
- `PrefillDispatchPath::RATIO_VNNI_NATIVE` enum value
- `selectPrefillDispatchPath()` declaration (if no longer needed)

**Code to remove from `ROCmQuantisedGemmKernel.cpp` (~1500+ lines):**
- `Impl` fields: `d_weights_ratio_*`, `d_scales_B_grouped`, pinned staging, 13 `ratio_vnni_*` metadata, repack cache `repack_cached_src_ratio_*`
- Destructor: ratio buffer frees
- Extern "C" declarations: `rocmGemv_ratio_vnni_*`, `rocmGemm_ratio_vnni_*` (7 symbols)
- Constants: `RATIO_VNNI_CODEBOOK_*`
- Functions: `packRatioVNNIPhase1()`, `buildBitwidth8PayloadV2FromVNNI()`
- `ensureWeightsConverted()`: ratio upload paths (both Path 1 packed and Path 2 legacy)
- `ensureRepackedWeightsForCK()`: ratio expand paths
- Commit section: `impl_->d_weights_ratio_*` wiring
- `selectPrefillDispatchPath()`: `RATIO_VNNI_NATIVE` case
- `tryPrefillNativeGemm()`: `RATIO_VNNI_NATIVE` case with ABI descriptor
- 4× decode dispatch: ratio-VNNI branches in multiply_tensor, multiply_fused_tensor, multiply_fp32_to_fp32, multiply_fp32_to_fp32_with_bias
- Error-path cleanup for ratio buffers

**Code to remove from `ROCmGemvKernel.hip` (~3000+ lines):**
- Device constants: `k_ratio_vnni_iq4_lut_i8`, `k_ratio_iq4_perm_lut_words`, `k_ratio_linear_perm_lut_words`, `k_ratio_linear_unsigned_perm_lut_words`
- LUT build functions: `build_ratio_iq4_perm_lut_table_host()`, `build_ratio_linear_perm_lut_table_host()`, `build_ratio_linear_unsigned_perm_lut_table_host()`
- LUT init functions: `rocmGemv_ratio_vnni_init_*_lut()`
- Decode functions: `decode_q4_pack4_low()`, `decode_q4_pack4_high()`, `decode_q4_block8_to_packed_i8()`, `decode_q4_block8_to_packed_i8_linear_with_min()`, `add_min_clamp_to_packed_i8x4()`
- Prefill kernel: `gemm_ratio_vnni_prefill_kernel`
- GEMV kernels: `gemv_ratio_vnni_grid_kpar_kernel_t`, `gemv_ratio_vnni_grouped_kernel_t`
- Expand kernels: `expand_ratio_vnni_to_rowmajor_q4_kernel`, `expand_ratio_vnni_to_rowmajor_q5_kernel`
- Dispatch functions: `rocmGemv_ratio_vnni_int8_int32()`, `rocmGemv_ratio_vnni_int8_fp32_grouped()`, `rocmGemv_expandRatioVNNI_to_rowmajor()`, `rocmGemm_ratio_vnni_int8_int32_prefill*()`, tuning overrides
- Note: keep `k_ratio_vnni_iq4_lut_i8` if native-VNNI IQ4 decode needs it (rename to `k_iq4_lut_i8`)

**Code to remove from `ROCmQuantisedGemmKernel.hip` (~110 lines):**
- `applyScaleA_m1_vec4_kernel`, `applyScaleA_full_kernel`, `rocmQuantGemm_applyScaleA_fp32()` — these were ratio-VNNI-specific epilogues

**Tests to update:**
- `tests/v2/integration/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp`:
  - Remove `PrefillNativeRatioVNNI_MatchesCKFallback` test
  - Remove negative assertions for ratio buffers (L618–619)
  - Remove ratio-only packing assertions (L948–953)
  - Add new native-VNNI packing + GEMV tests
- `tests/v2/CMakeLists.txt`:
  - Remove ratio-VNNI perf test targets and labels
  - Add native-VNNI test targets

**Deliverable:** Clean codebase with zero ratio-VNNI references. Build succeeds, all parity tests pass.

### Phase 5: Testing & Validation
**Goal:** Full parity test suite passes with native-VNNI.

**Tasks:**
1. Run decode parity test: `*DecodeParity/ROCm_KV_FP16`
   - Target: AvgCosine ≥ 0.99 (currently 0.85 with ratio-VNNI)
2. Run prefill parity test: `*PrefillParity/ROCm_KV_FP16`
   - Target: Layer 0 cosine ≥ 0.99, KL ≤ 1.0
3. Run full parity suite: `ctest -R "^V2_Integration_Parity_"`
4. Run integration tests: `ctest -R "^V2_Integration_"`
5. Run unit tests: `ctest -R "^V2_Unit_"`
6. Performance benchmark: `./build_v2_release/llaminar2 --benchmark -m models/Qwen2.5-7B-Instruct-Q8_0.gguf`

**Deliverable:** All tests pass, performance is competitive with ratio-VNNI.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Per-block FP32 multiply overhead | Low | Medium | One FP32 multiply per 8 sdot4 calls (~12% overhead); grouped ratio-VNNI had the same ratio |
| AtomicAdd FP32 contention | Low | Low | Same K-partitioning strategy as INT8-VNNI; well-proven |
| CK expand quality for prefill | Medium | Low | Row-scale + INT8 re-quantization is lossy, same as current CK path; decode is the critical path |
| Regression in Q5/Q4_1 formats | None | None | These formats are out of scope for Sprint 1; ratio-VNNI removal only affects Q4_0/IQ4_NL paths |

## IQ Shape Specialization (Post-LDS Kernel)

### Motivation

The IQ LDS kernel (`gemv_iq_lds_kernel_t`) gives 76-190% decode speedups by caching
grid lookup tables in LDS.  But the initial dispatch (simple `N > 2048` threshold) is
suboptimal for several shape categories.  Shape specialization applies different kernel
strategies based on the N/K ratio, inspired by the INT8 VNNI prefill kernel's V3/V7
dispatch (`K >= N → V3/LDS-pipeline`, `K < N → V7/safe-tile`).

### Benchmark Baseline (IQ3_XXS, representative)

| Shape | N | K | K/N | Category | Kernel | Eff% | Issue |
|-------|---|---|-----|----------|--------|------|-------|
| 0.5B_AttnOut | 896 | 896 | 1.0 | Small balanced | non-LDS TN=64 | 46% | Launch floor |
| 0.5B_QKV | 2688 | 896 | 0.33 | N-heavy (small) | LDS TN=256 | 34% | **Regression**: only 11 LDS blocks for 60 CUs |
| 0.5B_FFN_Up | 4864 | 896 | 0.18 | N-heavy (small K) | LDS TN=256 | 38% | Short K, LDS overhead not amortized |
| 0.5B_FFN_Dn | 896 | 4864 | 5.4 | K-heavy (small N) | non-LDS TN=64 | 53% | Good — K-loop benefits from constant$ |
| 0.5B_LM_Head | 151936 | 896 | 0.006 | Very N-heavy | LDS TN=256 | 61% | BW-limited, good |
| 3B_AttnOut | 2048 | 2048 | 1.0 | Balanced | non-LDS TN=64 | 47% | Fallback at threshold boundary |
| 3B_FFN_Up | 11008 | 2048 | 0.19 | N-heavy | LDS TN=256 | 52% | Good |
| 3B_FFN_Dn | 2048 | 11008 | 5.4 | K-heavy | non-LDS TN=64 | 40% | **Poor**: long K-loop + constant$ thrashing |
| 3B_LM_Head | 151936 | 2048 | 0.013 | Very N-heavy | LDS TN=256 | 75% | Near-optimal |
| 7B_QKV | 10752 | 3584 | 0.33 | N-heavy | LDS TN=256 | 50% | Good |
| 7B_FFN_Up | 18944 | 3584 | 0.19 | N-heavy | LDS TN=256 | 55% | Good |
| 7B_FFN_Dn | 3584 | 18944 | 5.3 | K-heavy | LDS TN=256 | 49% | Long K, moderate |

### Strategy 1: CU-Saturation Threshold (replaces N > 2048)

**Problem**: The LDS kernel uses TN=256, so `grid_n = N/256`.  For small N this
creates too few blocks.  0.5B_QKV (N=2688) gets only 11 blocks — not enough to
fill 60 CUs even with K-partitioning.

**Fix**: Replace the static `N > 2048` threshold with a hardware-relative CU-saturation check:
```
grid_n_lds = ceil(N / 256)
use_iq_lds = is_iq && grid_n_lds >= NUM_CUS / 2   (e.g., 60/2 = 30)
```
This ensures ~(CU_count/2) N-blocks × KB Y-blocks = enough work for full CU coverage.
For shapes below the threshold, the non-LDS TN=64 kernel creates 4× more N-blocks.
The threshold is hardware-relative, not a hardcoded count.

### Strategy 2: K-Heavy Aspect Ratio (FFN_Dn pattern)

**Problem**: K-heavy shapes (K >> N) have long K-loops with many grid lookups.
The non-LDS kernel relies on constant cache, which thrashes when 60+ CUs hammer
the same grid table.  3B_FFN_Dn (N=2048, K=11008) hits only 40% efficiency.

**Fix**: Force LDS kernel for K-dominated shapes using ratio-based checks:
```
is_k_heavy = (K >= 4.0 * N)                  // Aspect ratio: K-dominated shape
           && (grid_n_lds >= NUM_CUS / 10)    // Min parallelism for 256-thread WGs
use_iq_lds = is_iq && (grid_n_lds >= NUM_CUS/2 || is_k_heavy)
```
For K-heavy shapes, the LDS grid cache eliminates constant memory contention on
the long K-loop.  The aspect ratio threshold (K/N ≥ 4) captures the FFN_Dn pattern
regardless of absolute size.  The grid_n floor (~10% of CUs) ensures the LDS
kernel has workable parallelism — shapes with too few N-blocks (e.g., 0.5B FFN_Dn
with grid_n=4) stay on the more parallel TN=64 kernel.

All thresholds are **ratio-based or hardware-relative** (no hardcoded row/column
counts) so they scale to arbitrary model dimensions.

### Strategy 3: CPT=2 for N-Heavy Shapes (Future)

For very N-heavy shapes (LM_Head, FFN_Up), processing 2 output columns per thread
shares activation loads across both.  INT8 V7 uses N_TILE=128 (vs V3's 64) for the
same reason.

### Strategy 4: K-Loop Double Buffering (Future)

For K-heavy shapes, explicit software pipelining (load block b+1 while computing
block b) would overlap global memory latency with compute.  INT8 V3 uses LDS
double-buffering for the same effect.

### Implementation Priority

1. **Strategies 1+2 combined** — fix threshold + K-heavy override (this sprint)
2. **Strategy 3: CPT=2** — activation reuse for BW-bound shapes (next sprint)
3. **Strategy 4: K-loop pipeline** — overlap loads/compute for K-heavy (future)

---

## ISA-Level IQ Format Optimization Plan

### Context

ISA analysis (disassembly + instruction census) of the IQ format kernels reveals that
IQ1 and IQ2 performance (29–39% of INT8 baseline) is limited by three root causes,
not occupancy. VGPR counts are low (22–32) giving max occupancy (32–40 waves/CU).

### Root Causes

**RC1: L1 Cache Thrashing (IQ1)**
The `d_iq1s_grid` constant table is 16KB — exactly the vector L1 capacity on GFX9.
Grid lookups use `global_load_dwordx2` with per-lane scattered addresses, continuously
evicting payload/scale data. Actual memory traffic is ~4.25× the useful weight data
due to grid fetches. IQ2_S (8KB grid) fits with headroom; IQ3 (1–2KB) fits easily.

**RC2: Dependent-Load Latency Chain**
Q4_0 has a single load phase: payload → decode (pure VALU) → dot4. IQ formats have
*two* sequential load phases: payload load → wait → index extract → grid load → wait → dot4.
Critical path: ~508 cycles (IQ1) vs ~420 cycles (Q4_0).

**RC3: Double dot4 Count for IQ1 Asymmetric Correction**
The ISA shows two parallel accumulator chains: 8× `v_dot4_i32_i8` for `act × weight`
and 8× `v_dot4_i32_i8` for `act × 1` (computing `sum_a` for the asymmetric min
correction). 50% of dot4 throughput is pure overhead.

**RC4: Massive Decode VALU for IQ2/IQ3 Sign Application**
`iq_apply_signs_4()` expanded to ~15 VALU per call (4× BFE extract + shifts + ORs +
mask expand + XOR + add). Called 8× per block = ~120 decode VALU vs only 8 dot4.
**Fixed by O4**: SWAR multiply reduces to ~5 VALU per call (see O4 results below).

### ISA Census Summary

| Metric | Q4_0 | IQ1_S | IQ1_M | IQ2_S |
|--------|------|-------|-------|-------|
| Total instructions | 337 | 164 | 174 | 253 |
| VALU (%) | 181 (54%) | 74 (45%) | 82 (47%) | 159 (63%) |
| SALU (%) | 123 (37%) | 74 (45%) | 75 (43%) | 78 (31%) |
| VMEM loads | 22 | 10 | 11 | 10 |
| v_dot4_i32_i8 | 8 | 16 | 16 | 8 |
| VALU/VMEM ratio | 8.2:1 | 7.4:1 | 7.5:1 | 15.9:1 |

### Optimization Roadmap

| # | Optimization | Target | Expected Gain | Actual Gain | Status |
|---|-------------|--------|---------------|-------------|--------|
| ~~O1~~ | ~~Ternary-encoded compact grid for IQ1~~ | ~~IQ1_S/M~~ | ~~+30-40%~~ | Regression | **Reverted** |
| O2 | 2-block loop unrolling with grid prefetch | IQ2 non-LDS | +15-25% | +0.4-0.8% (noise) | **Done — marginal** |
| ~~O3~~ | ~~Cooperative block-sum precomputation via LDS~~ | ~~IQ1_S/M, Q2_K~~ | ~~+10-15%~~ | Regression (-3 to -11%) | **Reverted** |
| **O4** | **SWAR multiply-based sign expansion** | **IQ2/IQ3** | **+10-20%** | **+8.6-20.9%** | **Done — major win** |
| O5 | CPT=2 for IQ formats | All IQ | +5-10% | — | Planned |

### O2: 2-Block Loop Unrolling (Marginal)

Processes two K-blocks per iteration in the non-LDS kernel so the compiler can
overlap constant-memory grid lookup latencies between blocks. Block N+1's grid
loads are issued while block N's dot4 chain executes.

**Result**: Applied only to IQ2 non-LDS path (IQ1 regressed due to 16KB grid L1
pressure; LDS kernel already has ~20-cycle latency with 4 overlapping reads).
Benchmarks showed +0.4-0.8% improvement, within measurement noise. The compiler
was already scheduling loads well enough that manual 2-block unrolling added no
meaningful benefit. Kept in code (zero regression risk) but not impactful.

### O3: Cooperative Block-Sum Precomputation via LDS (Reverted)

Attempted to eliminate redundant `sum_a` computation (8 sdot4 per block per thread)
for asymmetric formats (IQ1_S, Q4_1, Q5_1) and dual-scale-asymmetric formats
(IQ1_M, Q2_K). All 64 threads in a wavefront compute IDENTICAL `sum_a` values
(pure cross-lane redundancy). Fix: distribute blocks across threads cooperatively
in a pre-loop, store to LDS, then read precomputed sums in the main loop.

**Implementation** (fully coded, tested, benchmarked, then reverted):
- Added `needs_block_sums` and `sums_per_block` compile-time traits to NVNNITraits
- Non-LDS kernel: cooperative pre-loop with `extern __shared__`, `__syncthreads()`
- LDS kernel: dynamic shared memory after static grid table + cooperative pre-loop
- Dispatch: computed and passed dynamic shared memory sizes
- All 4 unit tests passed (correct results)

**Benchmark Results** (vs O4 baseline — REGRESSION across all targets):

| Format | O4 Eff | O3 Eff | O4 BW | O3 BW | Change |
|--------|--------|--------|-------|-------|--------|
| Q2_K | 49% | 45% | 320.3 | 295.3 | **-7.8%** |
| IQ1_S | 30% | 28% | 215.7 | 192.6 | **-10.7%** |
| IQ1_M | 32% | 31% | 260.5 | 246.3 | **-5.5%** |
| Q4_1 | 85% | 84% | 352.0 | 347.3 | -1.3% (noise) |
| Q5_1 | 89% | 88% | 348.6 | 349.5 | ~0% |

**Root Cause Analysis**:

1. **ILP loss (non-LDS kernel)**: On gfx906, the inline `sum_a` sdot4 chain is
   independent of the main dot-product sdot4 chain. The compiler/SIMD scheduler
   interleaves them, making `sum_a` computation effectively **free** (hidden
   behind the main accumulation). Replacing free VALU work with LDS reads adds
   latency to the critical path without reducing total execution time.

2. **Occupancy degradation (LDS kernel)**: For IQ1_S/IQ1_M, the static grid
   table uses 16384 bytes of LDS. Adding even ~112-224 bytes of dynamic shared
   memory for block sums pushes total LDS past the 16384-byte boundary:
   `floor(65536/16608) = 3` vs `floor(65536/16384) = 4` WGs/CU → 25% occupancy
   drop (16 → 12 waves).

3. **Extra barrier overhead (LDS kernel)**: The 256-thread workgroup requires an
   additional `__syncthreads()` after the block-sum pre-loop, adding ~50-100
   cycles of inter-wavefront synchronization.

**Key Insight**: The cross-lane redundancy that O3 targets (all threads computing
identical `sum_a`) is NOT actually a bottleneck on AMD wavefront64 because all
64 lanes execute in lockstep — the redundant work takes the SAME number of cycles
as non-redundant work. The VALU pipe is wide enough that the sum_a chain hides
behind the main dot-product chain via instruction-level parallelism.

### O4: SWAR Multiply-Based Sign Expansion (Major Win)

Replaced the per-bit BFE extraction chain in `iq_apply_signs_4()` with a
multiply-based SWAR (SIMD Within A Register) bit-to-byte expansion:

```cpp
// OLD: 4× v_bfe_i32 + shifts + ORs ≈ 15 VALU per call
uint32_t mask =
    (uint32_t)(-(sign_lo4 & 1)) & 0xFF) |
    ((uint32_t)(-((sign_lo4 >> 1) & 1)) & 0xFF) << 8) | ...

// NEW: 2× v_mul_lo + 1× v_and ≈ 5 VALU per call
uint32_t spread = (uint32_t(sign_lo4 & 0xF) * 0x08040201u) & 0x01010101u;
uint32_t mask = spread * 255u;
return (grid4 ^ mask) + spread;
```

**Math**: `sign_lo4 * 0x08040201` scatters each sign bit to its own byte lane
(no cross-byte carry since max value 15 × 0x08040201 = 0x783C1E0F, all bytes < 256).
Then `spread * 255` expands 0x00→0x00, 0x01→0xFF per byte.

**ISA Impact** (verified via llvm-objdump):

| Kernel | v_bfe Before | v_bfe After | Total Insns Before | Total Insns After | Reduction |
|--------|-------------|-------------|-------------------|-------------------|-----------|
| IQ3_S LDS | 29 | 7 | 288 | 227 | **-21%** |
| IQ2_S LDS | 27 | 6 | 266 | 207 | **-22%** |
| IQ2_XXS LDS | 25 | 5 | — | — | **-20%** |

**Benchmark Results** (vs pre-O4 baseline):

| Format | Before (μs / eff%) | After (μs / eff%) | Speedup | Eff Δ |
|--------|-------------------|-------------------|---------|-------|
| IQ2_XXS | 58.9 / 36% | **46.6 / 44%** | **-20.9%** | **+8pp** |
| IQ2_S | 68.3 / 38% | **57.1 / 44%** | **-16.4%** | **+6pp** |
| IQ2_XS | 67.5 / 35% | **57.3 / 41%** | **-15.1%** | **+6pp** |
| IQ3_S | 71.0 / 50% | **61.6 / 57%** | **-13.2%** | **+7pp** |
| IQ3_XXS | 61.4 / 51% | **56.1 / 56%** | **-8.6%** | **+5pp** |
| All other formats | — | — | ±0.2% (noise) | 0 |

IQ1_S/M unchanged (they don't call `iq_apply_signs_4` — grid values are already signed).

### O5: CPT=2 for IQ Formats (Planned)

Process 2 output columns per thread, sharing activation loads across both columns.
Reduces activation bandwidth by 50%.

---

## Notes

- **IQ4_NL LUT**: The `k_ratio_vnni_iq4_lut_i8[16]` constant is still needed for IQ4_NL native-VNNI. Rename to `k_iq4_codebook_i8[16]` and keep it.
- **Q5_0 / Q5_1 / Q4_1**: These formats have additional structure (5th bit plane, min side-channel) that requires separate design. Not in Sprint 1.
- **Prefill native kernel**: A dedicated native-VNNI prefill kernel (avoiding CK expand) could improve prefill throughput. Future optimization.
- **Scale compression**: For very large K dimensions, per-block FP16 scales could be compressed (e.g., delta encoding). Not needed for current model sizes.
