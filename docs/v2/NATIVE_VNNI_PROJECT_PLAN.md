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

## Notes

- **IQ4_NL LUT**: The `k_ratio_vnni_iq4_lut_i8[16]` constant is still needed for IQ4_NL native-VNNI. Rename to `k_iq4_codebook_i8[16]` and keep it.
- **Q5_0 / Q5_1 / Q4_1**: These formats have additional structure (5th bit plane, min side-channel) that requires separate design. Not in Sprint 1.
- **Prefill native kernel**: A dedicated native-VNNI prefill kernel (avoiding CK expand) could improve prefill throughput. Future optimization.
- **Scale compression**: For very large K dimensions, per-block FP16 scales could be compressed (e.g., delta encoding). Not needed for current model sizes.
