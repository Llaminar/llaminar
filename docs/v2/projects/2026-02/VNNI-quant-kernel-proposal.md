# VNNI-Quant Kernel Proposal

## Overview

This document describes a new VNNI-style compact-weight kernel and repacker that:
- Preserves VNNI-style coalesced access patterns.
- Preserves the existing grid-level K-parallel reduction (int32 atomicAdd).
- Avoids expanding weights to full INT8 size for <= 4-bit formats.
- Supports Q4*, IQ4*, Q3*, IQ3*, Q2*, IQ2*, Q1*, IQ1*.

The main idea is to retain compact payload bits (1-4 bits per weight) and add a
per-block int8 ratio so that on-the-fly reconstructed int8 weights can be used
with sdot4 and int32 accumulation. The per-column floating scale is applied via
existing epilogue logic (applyScaling), exactly like the current INT8 VNNI path.

**Initial scope**: Phase 1 targets IQ4_NL and Q4_0 only (simple 32-element
blocks with a single FP16 scale). K-quant formats (Q4_K, Q6_K) and
lower-bitwidth formats (3/2/1-bit) are deferred to Phase 2 after the 4-bit path
is validated end-to-end.


### Relationship to existing compact IQ4_NL GEMV

The codebase already has `gemv_int8_iq4nl_fp32_kernel_t` which operates on
compact IQ4_NL blocks without INT8 expansion, using FP32 accumulation. The
ratio-VNNI kernel **supersedes** this path for decode because it provides:

- VNNI-interleaved coalescing (vs sequential block access)
- Grid-level K-parallelism via `blockIdx.y` (vs single-block)
- INT8 sdot4 accumulation (vs FP32 FMA)

Once ratio-VNNI is validated via parity tests, the existing
`gemv_int8_iq4nl_fp32_kernel_t` will be deprecated for M=1 decode. The compact
IQ4_NL block upload (`d_weights_native`) can also be removed from
`ROCmPackedWeights`, reducing device memory further.


### Bandwidth and memory savings

| Format | Current INT8 VNNI | Ratio-VNNI | VRAM Savings |
|--------|-------------------|------------|--------------|
| 4-bit (IQ4_NL, Q4_0) | 8 bits/weight | ~4.25 bits/weight (16B payload + 1B ratio / 32 elts) | **1.88×** |
| 3-bit | 8 bits/weight | ~3.25 bits/weight (12B + 1B / 32) | **2.46×** |
| 2-bit | 8 bits/weight | ~2.25 bits/weight (8B + 1B / 32) | **3.56×** |
| 1-bit | 8 bits/weight | ~1.25 bits/weight (4B + 1B / 32) | **6.40×** |

For a 7B IQ4_NL model on MI50 (16GB VRAM), weight VRAM drops from ~6.2GB
(INT8 VNNI) to ~3.3GB, freeing ~2.9GB for KV cache or larger models.


### Compute vs bandwidth analysis (MI50 gfx906)

Decode GEMV is bandwidth-bound. The ratio-VNNI kernel shifts the balance:

**Per sdot4 group (4 weights)**:
- INT8 VNNI: ~1 cycle (load → sdot4)
- Ratio-VNNI: ~8-10 cycles (decode nibbles + ratio multiply + pack + sdot4)

**3584×3584 projection (Wo/Q) at M=1**:
- Compute at ~8 cycles/4MADs: `3584 × 3584 / 4 × 8 / 1.5GHz ≈ 17μs`
- Bandwidth (ratio-VNNI): `~6.8MB / 480 GB/s ≈ 14μs`
- **Balanced** — ideal GPU utilization, not compute-bottlenecked

**18944×3584 projection (FFN Down) at M=1**:
- Compute: `~90μs`
- Bandwidth: `~73μs`
- Still bandwidth-dominated for large matrices

**Expected decode speedup**: ~1.8-1.9× for 4-bit formats (bandwidth reduction
translates nearly linearly since the kernel remains bandwidth-bound or balanced).


## Diagrams

### Dataflow (repacker + kernel)

```
Quantized weights
    (Q4/Q3/Q2/Q1, IQ*)
                    |
                    |  repack_ratio_vnni()
                    v
    +---------------------------+
    | Ratio-VNNI container      |
    | - payload[block][N][..]   |
    | - ratio[block][N] (int8)  |
    | - col_scale[N] (float)    |
    +---------------------------+
                    |
                    |  decode GEMV (M=1)
                    v
    +---------------------------+
    | grid_kpar kernel          |
    | - sdot4 -> int32 acc      |
    | - atomicAdd(int32)        |
    +---------------------------+
                    |
                    |  applyScaling
                    v
    +---------------------------+
    | FP32 output               |
    +---------------------------+
```

### VNNI-style interleave by N (coalesced reads)

```
payload layout (4-bit example, 32 weights per block)

    block b = 0
    +------------------------------------------------------------+
    | N=0 | N=1 | N=2 | ... | N=63 | N=64 | N=65 | ...           |
    | 16B | 16B | 16B |     | 16B  | 16B  | 16B  |               |
    +------------------------------------------------------------+

    ratio layout
    +------------------------------------------------------------+
    | N=0 | N=1 | N=2 | ... | N=63 | N=64 | N=65 | ...           |
    | 1B  | 1B  | 1B  |     | 1B   | 1B   | 1B   |               |
    +------------------------------------------------------------+

Each wavefront reads consecutive N values, producing coalesced loads.
```

### Kernel equivalence to INT8 VNNI grid_kpar

```
Existing INT8 VNNI grid_kpar:
    a_int8 + b_int8_vnni -> sdot4 -> int32 acc -> atomicAdd

Ratio-VNNI grid_kpar:
    a_int8 + (payload + ratio + LUT)
                    -> reconstructed int8 -> sdot4 -> int32 acc -> atomicAdd

Epilogue (both): applyScaling(d_C_int32, scaleA, col_scale)
```


## Part 1: Repacker

### 1a) Support for 4-bit quantized formats (Q4*, IQ4*)

Target block size is 32 weights. All 4-bit formats can be normalized to the same
payload layout: 16 bytes per block (two 4-bit values per byte). The repacker
will:

1) Extract the 16 nibble bytes for each 32-weight block, without expanding to
   INT8.
2) Compute a per-column scale S_col[n] and a per-block ratio r_b (int8).
3) Store payload and ratio in a VNNI-style interleaved layout by N.

Output layout:
- payload: [blocks][N][16]
- ratio:   [blocks][N]
- col_scale: [N]

Examples:
- Q4_0: payload is already nibble-packed (symmetric). Ratio is derived from
  block scale and S_col[n].
- IQ4_NL: payload is nibble indices into the IQ4 LUT. Ratio uses the block
  scale and S_col[n].
- Q4_1 (asymmetric): see Section 1e for zero-point handling.
- IQ4_XS, Q4_K: deferred to Phase 2 (K-quant sub-block scale extraction).


### 1b) Support for 3-bit quantized formats (Q3*, IQ3*)

3-bit formats are supported by packing 32 values into 12 bytes (96 bits) per
block. The repacker keeps the payload in compact 3-bit form and uses a ratio
byte to scale the decoded int8 values.

Output layout:
- payload: [blocks][N][12]
- ratio:   [blocks][N]
- col_scale: [N]

Format handling:
- Q3*: decode bits to signed integer levels (format-specific), then apply ratio
  and sdot4.
- IQ3*: decode bits to IQ3 LUT indices, then apply ratio and sdot4.


### 1c) Support for 2-bit quantized formats (Q2*, IQ2*)

2-bit formats use 8 bytes per 32-weight block. These can be stored as a compact
payload without expansion.

Output layout:
- payload: [blocks][N][8]
- ratio:   [blocks][N]
- col_scale: [N]

Format handling:
- Q2*: decode 2-bit levels to signed values (format-specific), then apply ratio.
- IQ2*: decode indices with IQ2 LUT and optional sign tables. Ratio then applies.

Some Q2_K/IQ2 formats use sub-block scale/mins. In that case, the repacker
stores per-subblock ratio and uses the correct sub-block during reconstruction.
This is still compact because the ratio is 1 byte per sub-block.


### 1d) Support for 1-bit quantized formats (Q1*, IQ1*)

1-bit formats pack 32 values into 4 bytes. The repacker keeps the payload in
compact form, and uses a ratio byte per block.

Output layout:
- payload: [blocks][N][4]
- ratio:   [blocks][N]
- col_scale: [N]

Format handling:
- Q1*: decode 1-bit to {-1,+1} or format-specific levels.
- IQ1*: decode with IQ1 LUT and sign rules. Ratio applies after decode.


### 1e) Asymmetric format handling (Q4_1, Q5_1)

Asymmetric formats like Q4_1 have a per-block `min` (zero-point) in addition to
`scale`. The weight reconstruction is:

$w[k] = scale_b \cdot decoded_k + min_b$

The ratio factorization handles the scale term as usual. The `min` term requires
a separate per-block int8 value `min_ratio` and a per-column `col_min`:

- `col_min[n] = max_b(|min_b|)` over blocks in column n
- `min_ratio_b = round(clamp(min_b / col_min[n], -1, 1) * 127)`

The kernel reconstructs `min` as `col_min[n] * min_ratio_b / 127` and adds it
to each decoded element before sdot4. This adds one extra int8 load per block
and one FP32 multiply in the epilogue.

**Extended container for asymmetric formats:**
- payload: `[blocks][N][payload_bytes]`
- ratio: `[blocks][N]` (scale ratio)
- min_ratio: `[blocks][N]` (zero-point ratio, only when `has_min=1`)
- col_scale: `[N]`
- col_min: `[N]` (only when `has_min=1`)

The `RatioVNNIHeader.has_min` flag gates the extra loads and epilogue cost. For
symmetric formats (Q4_0, IQ4_NL), `min_ratio` and `col_min` are not allocated.

**Note**: Asymmetric format support is deferred to Phase 2. Phase 1 targets
only symmetric formats (Q4_0, IQ4_NL).


### Common ratio scale computation

For each column n:
- Compute S_col[n] = max_b(|scale_b|) over blocks in that column.
- For each block b, compute:
  - r_b = round( clamp(scale_b / S_col[n], -1, 1) * 127 )
  - Store r_b as int8.

The ratio allows reconstruction of per-block scale on the fly, while keeping the
col_scale vector compatible with the existing applyScaling epilogue.

**MANDATORY: 128/127 compensation factor.** The kernel reconstructs the per-block
scale as `r_b / 128` (via `>> 7`), but `r_b` was computed with a denominator of
127. Without compensation, this introduces a systematic -0.78% scale bias per
block that compounds across 24+ layers. To correct this, the repacker applies:

$S_{col}[n] := S_{col}[n] \cdot (128 / 127)$

This ensures that `S_col[n] * (r_b >> 7) ≈ S_col_original[n] * r_b / 127`
exactly, cancelling the integer division mismatch. The compensation is applied
once during repacking and has zero runtime cost.


### Repacker pseudo-code (format-agnostic)

```cpp
struct RatioVNNIHeader {
    uint8_t bitwidth;       // 1,2,3,4
    uint8_t codebook_id;    // 0 = linear, 1 = IQ1, 2 = IQ2, 3 = IQ3, 4 = IQ4
    uint8_t has_min;        // 1 if asymmetric min/zero-point is used
    uint8_t block_size;     // 32 (simple formats) or sub-block size for K-quants
    uint16_t lut_entries;   // LUT size (0 for linear decode)
    uint16_t reserved;
};

void repack_ratio_vnni(
    const TensorBase* src,
    uint8_t* dst_payload,
    int8_t* dst_ratio,
    float* dst_col_scale,
    RatioVNNIHeader* hdr,
    int N, int K)
{
    // Determine bitwidth, codebook_id from src->native_type()
    // Compute blocks_per_row based on bitwidth and block size

    // Pass 1: compute col scales
    for (int n = 0; n < N; ++n) {
        float max_abs = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            float scale_b = get_block_scale(src, n, b);
            max_abs = std::max(max_abs, std::abs(scale_b));
        }
        dst_col_scale[n] = (max_abs > 0.0f) ? max_abs : 1.0f;
    }

    // Apply mandatory 128/127 compensation to col_scale
    for (int n = 0; n < N; ++n) {
        dst_col_scale[n] *= (128.0f / 127.0f);
    }

    // Pass 2: store payload + ratio interleaved by N
    // Use IQuantizedTileAccessor to access native blocks directly,
    // avoiding the FP32 dequant→requant round-trip.
    for (int n = 0; n < N; ++n) {
        const float inv_col = 127.0f / (dst_col_scale[n] * (127.0f / 128.0f));
        for (int b = 0; b < blocks_per_row; ++b) {
            PackedBlock pb;
            unpack_block_to_packed(src, n, b, pb);

            const int8_t r = static_cast<int8_t>(
                std::round(std::clamp(pb.scale * inv_col, -127.0f, 127.0f)));

            dst_ratio[b * N + n] = r;
            write_payload_interleaved(dst_payload, b, n, pb);
        }
    }
}
```


## Part 2: Kernel

### 2a) How it supports all formats and LUTs

The kernel supports all formats by decoding the compact payload according to
bitwidth + codebook_id. A single kernel template can branch on:
- bitwidth: 1, 2, 3, 4
- codebook_id: linear vs IQ LUT
- optional min/zero-point (asymmetric formats)

The ratio byte r_b is always applied to the decoded int8 value before sdot4.
Thus, all formats map to the same int8 sdot4 accumulation path.

Required LUTs:
- IQ4: kvalues_iq4nl_i8 (16 entries)
- IQ3: kvalues_iq3* (format-specific tables)
- IQ2: kvalues_iq2* (format-specific tables + optional sign tables)
- IQ1: kvalues_iq1* (format-specific tables)

These LUTs already exist in IQQuantTables.h and can be mirrored into HIP
constant memory for fast access.

**Constant memory budget (gfx906: 64KB `__constant__` cache)**:

| Format | LUT Size | Entries | Fits in L1 Constant? |
|--------|----------|---------|---------------------|
| IQ4_NL | 16 bytes | 16 × int8 | Yes (trivially) |
| IQ4_XS | 16 bytes | 16 × int8 | Yes |
| IQ3_S | 256 bytes | 256 × int8 | Yes |
| IQ3_XXS | 256 bytes | 256 × int8 | Yes |
| IQ2_XXS | ~2 KB | 256 × 8 grid + sign tables | Yes |
| IQ2_XS | ~2 KB | 256 × 8 grid + sign tables | Yes |
| IQ2_S | ~2 KB | 256 × 8 grid + sign tables | Yes |
| IQ1_S | ~4 KB | grid indices + sign permutation | Yes |
| IQ1_M | ~4 KB | grid indices + sign permutation | Yes |
| **Total (all formats loaded)** | **~15 KB** | | **Yes (< 64KB limit)** |

All IQ format LUTs fit comfortably within gfx906's constant cache. Phase 1
(IQ4_NL only) uses just 16 bytes.


### 2b) Algorithm and equivalence to grid_kpar int32 atomicAdd

#### High-level algorithm

For each block (split across blockIdx.y for K-parallelism):
1) Load activation int8 values (already quantized).
2) Load compact payload bits for weights (coalesced by N).
3) Decode payload bits into int8 values using LUT or linear mapping.
4) Apply per-block ratio r_b (int8) to decoded values to get int8 weights.
5) Accumulate dot products in int32 using sdot4.
6) Use atomicAdd to write into int32 output buffer.

This preserves the existing behavior of the INT8 VNNI grid_kpar kernel:
- Same block and grid structure.
- Same int32 accumulators.
- Same atomicAdd reduction.
- Same applyScaling epilogue with scaleA and col_scale.

#### Mathematical equivalence

Let decoded weight be:

$w[k] = scale_b \cdot \text{decode}(payload_k)$

We factor:

$scale_b = S_{col}[n] \cdot r_b / 127$

Then:

$w[k] \approx S_{col}[n] \cdot (r_b / 127) \cdot \text{decode}(payload_k)$

Define:

$w'[k] = \text{round}((r_b / 127) \cdot \text{decode}(payload_k))$

Then the kernel accumulates:

$acc[n] = \sum_k a_{int8}[k] \cdot w'[k]$

The epilogue applies:

$out[n] = acc[n] \cdot scaleA \cdot S_{col}[n]$

This matches the original quantization semantics, with only the expected
rounding error from the ratio scaling step (no worse than typical INT8
requantization).


### Kernel pseudo-code (grid_kpar)

```cpp
template<int TILE_N, int CPT>
__global__ void gemv_ratio_vnni_kpar_kernel(
    const int8_t*  d_A_int8,   // [K]
    const uint8_t* d_payload,  // compact bits (layout depends on bitwidth)
    const int8_t*  d_ratio,    // [blocks][N]
    int32_t*       d_C_int32,  // [N]
    int N, int K, int kblocks,
    RatioVNNIHeader hdr)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const int k_block = blockIdx.y;
    if (n_base >= N) return;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = k_block * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    for (int b = b_start; b < b_end; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(d_A_int8 + b * 32);

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n >= N) continue;

            const int8_t r = d_ratio[b * N + n];
            // Decode payload -> int8 weights (4 at a time)
            for (int g = 0; g < 8; ++g)
            {
                const int32_t a_packed = a4[g];
                const int32_t w_packed = decode_payload_to_packed_i8(
                    d_payload, hdr, b, n, g, r);
                acc[c] = __builtin_amdgcn_sdot4(a_packed, w_packed, acc[c], false);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
        if (n_base + c < N)
            atomicAdd(&d_C_int32[n_base + c], acc[c]);
}
```

#### decode_payload_to_packed_i8

For each bitwidth, decode 4 values from the payload into a packed int32 of 4
int8 weights. The function reads only the bytes needed for the current group of
4 elements, not the entire block.

**4-bit (Phase 1)**:
- Read 2 bytes from `payload[b][n][g*2 .. g*2+1]` (groups g=0..7, each 2 bytes
  covers 4 nibbles)
- Extract nibbles: `idx = (byte >> (4 * (i & 1))) & 0x0F`
- For IQ4_NL: LUT lookup → `decoded = kvalues_iq4nl_i8[idx]`
- For Q4_0 (linear): `decoded = (int8_t)(idx - 8)` (symmetric around 0)
- Apply ratio: `w = (int8_t)((ratio * decoded + 64) >> 7)`
- Pack 4 int8 into int32: `(w0) | (w1<<8) | (w2<<16) | (w3<<24)`

**3-bit (Phase 2)**: Read from 12-byte block payload. For group g (4 values
starting at element `g*4`), extract the specific 3-bit fields spanning at most
2 bytes. NOT reading all 12 bytes per group — only the 2-3 bytes that contain
the target 12 bits.

**2-bit (Phase 2)**: Read 1 byte per group (4 × 2-bit = 8 bits = 1 byte).

**1-bit (Phase 2)**: Read 1 byte per 2 groups (4 × 1-bit = 4 bits per group,
8 groups share 4 bytes total).

The ratio is applied as:

```
int8_t w = (int8_t)((ratio * decoded + 64) >> 7);
```

The `+ 64` provides rounding (half of 128). The `>> 7` divides by 128, which
combined with the mandatory 128/127 compensation on `S_col`, exactly
reconstructs the original per-block scale.

This keeps weights in int8 and preserves int32 accumulation.


## Part 3: Prefill Strategy (M > 1)

The ratio-VNNI layout is optimized for decode (M=1 GEMV). For prefill (M > 1),
CK GEMM requires row-major INT8 weights. Three strategies were considered:

| Strategy | Description | Memory | Prefill Overhead |
|----------|-------------|--------|-----------------|
| A: Dual storage | Keep ratio-VNNI + INT8 VNNI on device | No savings during prefill | None |
| B: Decode-on-fly | Ratio-VNNI → expand to row-major INT8 in workspace scratch | Full savings | ~65-200μs per projection |
| C: Ratio-VNNI GEMM | Batch sdot4 with compact weights for M>1 | Full savings | Major kernel complexity |

**Selected: Strategy B (decode-on-fly)**.

The existing `d_B_rowmajor_scratch` workspace (sized to `max(N×K)` across all
kernels, ~65MB for FFN Gate/Up) is reused. A new device kernel
`expand_ratio_vnni_to_rowmajor()` decodes compact payload + ratio into INT8
row-major format in the workspace buffer before each CK GEMM call.

The expansion cost is amortized over the prefill sequence length (typically
128-2048 tokens), making it negligible per-token. For a 3584×3584 projection,
the expansion produces ~12.3MB of INT8 data — well within the 65MB scratch
budget.

```
Prefill path:
    ratio-VNNI (on device)
        |
        | expand_ratio_vnni_to_rowmajor() [device kernel]
        v
    INT8 row-major (in workspace scratch)
        |
        | CK GEMM (existing path)
        v
    FP32 output
```

This preserves the full VRAM savings of ratio-VNNI while reusing the existing
CK GEMM infrastructure for prefill without modification.


## Summary

- The repacker is format-agnostic: it outputs the same ratio-VNNI container for
  any <=4-bit format.
- The kernel is grid_kpar compatible and keeps int32 atomicAdd.
- The formats differ only in how their payload bits decode to int8 values, which
  is handled by LUTs or linear decode paths.
- No full INT8 expansion is required at repack time.
- Prefill uses Strategy B: expand ratio-VNNI to row-major INT8 in workspace
  scratch, then CK GEMM as before.
- The existing compact IQ4_NL GEMV (`gemv_int8_iq4nl_fp32_kernel_t`) is
  deprecated once ratio-VNNI is validated.
- The mandatory 128/127 compensation factor eliminates systematic scale bias.


## Implementation Plan

### Phase 1: IQ4_NL + Q4_0 (symmetric 4-bit, 32-element blocks)

1) **Define the ratio-VNNI container**
    - Add `RatioVNNIHeader` with `bitwidth`, `codebook_id`, `has_min`,
      `block_size`, `lut_entries`.
    - Extend `ROCmPackedWeights` with new fields:
      ```cpp
      std::vector<uint8_t> ratio_vnni_payload;   // [blocks × N × payload_bytes]
      std::vector<int8_t>  ratio_vnni_ratio;      // [blocks × N]
      uint8_t* d_ratio_vnni_payload = nullptr;    // device
      int8_t*  d_ratio_vnni_ratio = nullptr;      // device
      RatioVNNIHeader ratio_vnni_header;
      ```
    - `col_scale` reuses the existing `d_scales` (same semantic: per-column FP32
      scale applied in `applyScaling` epilogue).

2) **Add format adapters for IQ4_NL and Q4_0**
    - Implement `unpack_block_to_packed()` for each:
      - IQ4_NL: copy 16 nibble bytes as-is, extract FP16 `d` as block scale.
      - Q4_0: copy 16 nibble bytes as-is, extract FP16 `d` as block scale.
    - Use `IQuantizedTileAccessor::get_raw_block()` and
      `get_block_scale()` to access native blocks directly, avoiding the
      FP32 dequant→requant round-trip in current `packWeightsToROCm()`.

3) **Implement the repacker**
    - Pass 1: compute `S_col[n] = max_b(|scale_b|)` via `get_block_scale()`.
    - Apply mandatory 128/127 compensation: `S_col[n] *= 128.0f / 127.0f`.
    - Pass 2: compute `ratio = round(scale_b / (S_col[n] * 127/128) * 127)`.
    - Interleave payload and ratio by N for coalesced reads.
    - Store `S_col` as the per-column scale for `applyScaling`.

4) **Build the ratio-VNNI decode kernel**
    - Template on `<TILE_N=128, CPT=2>` matching existing grid_kpar.
    - Implement `decode_payload_to_packed_i8()` for 4-bit with:
      - Linear path (Q4_0): `decoded = (int8_t)(nibble - 8)`
      - LUT path (IQ4_NL): `decoded = kvalues_iq4nl_i8[nibble]`
    - Keep grid_kpar structure and int32 atomicAdd.
    - KB heuristic from existing grid_kpar (target ≥64 k-groups/wave).

5) **Build the prefill expansion kernel**
    - `expand_ratio_vnni_to_rowmajor()`: decode ratio-VNNI payload + ratio
      into INT8 row-major in `d_B_rowmajor_scratch`.
    - Grid: `ceil(N * blocks_per_row / 256)` blocks, 256 threads.
    - Each thread expands one (n, block) pair = 32 elements.

6) **Integrate dispatch in `ROCmQuantisedGemmKernel::multiply_tensor()`**
    - Dispatch priority for M=1 decode:
      ```
      Priority 0: Ratio-VNNI GEMV
                  (if d_ratio_vnni_payload != nullptr && M==1 && K%32==0)
      Priority 1: Fused FP32→INT8 GEMV (INT8 VNNI fallback)
      Priority 2: 3-kernel INT8 GEMV pipeline (INT8 VNNI fallback)
      ```
    - Dispatch for M>1 prefill:
      ```
      If ratio-VNNI available:
          expand_ratio_vnni_to_rowmajor() → d_B_rowmajor_scratch
          CK GEMM with d_B_rowmajor_scratch
      Else:
          repack_vnni_to_rowmajor() → d_B_rowmajor_scratch (existing path)
          CK GEMM
      ```
    - Gate: `ROCmPackedWeights::d_ratio_vnni_payload != nullptr`.
    - Fallback: INT8 VNNI when K not divisible by 32 or format not yet
      supported by ratio-VNNI.

7) **Validation (Phase 1)**
    - **Unit test (round-trip)**: repack IQ4_NL/Q4_0 to ratio-VNNI, then
      decode via `decode_payload_to_packed_i8()` and compare against
      `decode_block_at()` output. Target: max absolute error ≤ 1 LSB per int8.
    - **Parity test (GEMV)**: compare ratio-VNNI GEMV output against FP32
      reference GEMM using existing `ParityTestBase` framework. Target:
      cosine similarity > 0.999 per layer.
    - **Parity test (end-to-end)**: greedy decode with ratio-VNNI vs INT8 VNNI.
      Target: identical token predictions for first 50 tokens.
    - **Bandwidth profiling**: `rocprof --stats` to measure actual bandwidth
      utilization. Target: ≥ 55% of peak HBM2 bandwidth (> 480 × 0.55 ≈
      264 GB/s).
    - **Prefill parity**: CK GEMM via expanded ratio-VNNI vs CK GEMM via
      existing VNNI repack. Target: bitwise-identical INT8 row-major weights.

8) **Deprecate existing compact IQ4_NL GEMV**
    - Once parity tests pass, remove `gemv_int8_iq4nl_fp32_kernel_t` dispatch.
    - Remove `d_weights_native` (compact IQ4_NL blocks) from
      `ROCmPackedWeights` and device uploads.
    - Remove host-side `native_data` vector from `ROCmPackedWeights`.
    - Update `packWeightsToROCm()` to skip INT8 VNNI packing when ratio-VNNI
      is available (saves both host memory and H2D upload time).


### Phase 2: Extended format support (after Phase 1 validation)

9) **Asymmetric formats (Q4_1, Q5_1)**
    - Implement `min_ratio` + `col_min` path (Section 1e).
    - Add `has_min=1` branch in kernel decode.
    - Separate parity tests for asymmetric formats.

10) **K-quant formats (Q4_K, Q6_K, IQ4_XS)**
    - Implement sub-block scale extraction for 256-element super-blocks.
    - Q4_K: 8 sub-blocks × 32 elements, 6-bit scales from `d`, `dmin`,
      `scales[12]`.
    - Q6_K: complex bit-packed `ql[128]` + `qh[64]` + `scales[16]`.
    - Each sub-block gets its own ratio byte.

11) **3-bit formats (Q3_K, IQ3_S, IQ3_XXS)**
    - Implement 3-bit payload decode in kernel.
    - LUT tables for IQ3 variants (256 entries each, 256 bytes constant mem).

12) **2-bit and 1-bit formats**
    - Implement 2-bit and 1-bit payload decode.
    - IQ2/IQ1 LUT tables (~2-4KB each, all fit in constant cache).
    - For IQ1_S: grid indices + sign permutation decode (~20 instructions
      per 8 weights — verify this doesn't push the kernel compute-bound).

13) **Validation (Phase 2)**
    - Run same validation suite as Phase 1 for each new format.
    - Additional error budget analysis: for 1-2 bit formats where decoded
      values are {-1, 0, +1}, the ratio multiply is essentially
      sign+magnitude and error should be negligible.
    - End-to-end perplexity comparison on WikiText-2 for each format.



## C. 2026-02-13 Mode3 const-LUT 10-run A/B (Release)

### C.1 Setup

- Binary: `build_v2_release/tests/v2/v2_perf_rocm_ratio_vnni_kernel`
- Test filter: `ROCmRatioVNNIPerfTest.Phase1Q4AndIQ4SpeedupVsInt8VNNI`
- Fixed env:
  - `LLAMINAR_RATIO_IQ4_DECODE_MODE=3`
  - `LLAMINAR_RATIO_IQ4_CPT=1`
  - `LLAMINAR_RATIO_IQ4_PREFETCH_NEXT=1`
- A/B variable:
  - A: `LLAMINAR_RATIO_IQ4_MODE3_CONST_LUT=0`
  - B: `LLAMINAR_RATIO_IQ4_MODE3_CONST_LUT=1`
- Logs: `/tmp/iq4_mode3_constlut_ab_10run_20260213_150931`


### C.2 Results (Global ratio/int8 speedup)

- A (`MODE3_CONST_LUT=0`):
  - mean `1.37030x`, median `1.37360x`, min/max `1.32466x / 1.46412x`, std `0.03723`
  - pass count: `10/10`
- B (`MODE3_CONST_LUT=1`):
  - mean `1.41430x`, median `1.41512x`, min/max `1.38818x / 1.44810x`, std `0.01468`
  - pass count: `10/10`

- Median uplift (B vs A): **`+3.02%`**


### C.3 Takeaway

- The const-LUT mode3 path improves release median throughput and reduces run-to-run variance in this 10-run sample.
- Keeping mode3 const-LUT enabled by default is supported by this A/B evidence.


### C.4 2026-02-13 Codegen-guided pass: register pressure + prefetch A/B

Release codegen audit artifacts:
- `/tmp/rocm_codegen_audit_20260213_155905/rocmgemv_release_gfx906.syms`
- `/tmp/rocm_codegen_audit_20260213_155905/rocmgemv_release_gfx906.disasm`

Hot mode3 (`IQ4_DECODE_MODE=3`, `CPT=1`) symbol metadata from `.syms`:
- `...ELi3ELb0ELb1...` (`PREFETCH_NEXT=0`, `MODE3_CONST_LUT=1`): `vgpr=23`, `sgpr=24`, `private=0`
- `...ELi3ELb1ELb1...` (`PREFETCH_NEXT=1`, `MODE3_CONST_LUT=1`): `vgpr=28`, `sgpr=50`, `private=0`

This indicates prefetch-on materially increases register pressure for the mode3
const-LUT variant.

Release 10-run A/B (fixed: `LLAMINAR_RATIO_IQ4_CPT=1`, `LLAMINAR_RATIO_IQ4_DECODE_MODE=3`, `LLAMINAR_RATIO_IQ4_MODE3_CONST_LUT=1`):
- `LLAMINAR_RATIO_IQ4_PREFETCH_NEXT=0`:
  - mean `1.42418x`, median `1.42506x`, std `0.02088`, pass `10/10`
- `LLAMINAR_RATIO_IQ4_PREFETCH_NEXT=1`:
  - mean `1.40387x`, median `1.39667x`, std `0.01512`, pass `10/10`

Median delta (`prefetch=1` vs `prefetch=0`): **`-1.99%`**.

Logs:
- `/tmp/iq4_mode3_prefetch_ab10_20260213_155937`

Conclusion:
- For current mode3 const-LUT + dual-acc path, `PREFETCH_NEXT=0` performs better and aligns with lower VGPR/SGPR usage.


## Appendix A: Raw Binary Disassembly & ISA Audit Quick Reference

This appendix captures the exact workflow used to verify release-build codegen
for ratio-VNNI kernels on gfx906.

### A.1 Why this detour

When performance regresses or stalls, verify that the hot path is still:
- emitting `v_dot4_i32_i8` in the inner loop,
- using vector-width global loads where expected (`global_load_dwordx4`), and
- not unexpectedly scalarized into byte-at-a-time compute.

This check should be done on **release artifacts**, not debug/integration builds.


### A.2 End-to-end command sequence (release build)

From repository root:

```bash
# 1) Locate host HIP object for ROCm GEMV kernel file
OBJ=build_v2_release/CMakeFiles/rocm_backend.dir/kernels/rocm/ROCmGemvKernel.hip.o

# 2) Extract HIP fatbin payload embedded in the host object
/opt/rocm/llvm/bin/llvm-objcopy \
  --dump-section .hip_fatbin=/tmp/rocmgemv_release.hip_fatbin "$OBJ"

# 3) (Optional) list fatbin bundles
/opt/rocm/llvm/bin/clang-offload-bundler \
  --type=o --list --input=/tmp/rocmgemv_release.hip_fatbin

# 4) Unbundle gfx906 device object
/opt/rocm/llvm/bin/clang-offload-bundler \
  --type=o --unbundle \
  --input=/tmp/rocmgemv_release.hip_fatbin \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx906 \
  --output=/tmp/rocmgemv_release_gfx906.o

# 5) Produce disassembly + symbol table
/opt/rocm/llvm/bin/llvm-objdump -d /tmp/rocmgemv_release_gfx906.o > /tmp/rocmgemv_release_gfx906.disasm
/opt/rocm/llvm/bin/llvm-objdump -t /tmp/rocmgemv_release_gfx906.o > /tmp/rocmgemv_release_gfx906.syms
```


### A.3 Hot symbol names to inspect

Current ratio/int8 kernels typically appear as:

```text
_Z38gemv_int8_int8_grid_kpar_vnni_kernel_tILi128ELi2EEvPKaS1_Piiii
_Z34gemv_ratio_vnni_grid_kpar_kernel_tILi256ELi1ELb1EEvPKaPKhS1_Piiiih
_Z34gemv_ratio_vnni_grid_kpar_kernel_tILi128ELi1ELb1EEvPKaPKhS1_Piiiih
_Z34gemv_ratio_vnni_grid_kpar_kernel_tILi128ELi2ELb0EEvPKaPKhS1_Piiiih
```

Find them quickly:

```bash
grep -n "gemv_ratio_vnni_grid_kpar_kernel_t\|gemv_int8_int8_grid_kpar_vnni_kernel_t" /tmp/rocmgemv_release_gfx906.syms
grep -n "v_dot4_i32_i8" /tmp/rocmgemv_release_gfx906.disasm | head -n 80
```


### A.4 What “healthy” codegen looks like

For each hot kernel, expect:
- repeated `v_dot4_i32_i8` in the inner loop,
- at least one `global_load_dwordx4` in payload-heavy decode paths,
- no broad replacement of dot products by scalar arithmetic in the reduction loop.

For ratio-IQ4, some scalar setup is expected (index/scale reconstruction). The
goal is to ensure this setup does not replace vectorized dot-product execution.


### A.5 Resource metadata sanity check

The symbol table includes compiler-emitted metadata keys:
- `.num_vgpr`
- `.numbered_sgpr`
- `.private_seg_size`

Quick extraction pattern:

```bash
grep -n "num_vgpr\|numbered_sgpr\|private_seg_size" /tmp/rocmgemv_release_gfx906.syms
```

If ratio kernels carry much higher VGPR/SGPR than INT8 baseline, occupancy may
drop and pipeline stalls can increase even when `v_dot4_i32_i8` is present.


### A.6 Common pitfalls

- Disassembling the host object directly yields x86-64 assembly and can hide
  the device ISA. Extract `.hip_fatbin` first.
- Running this workflow on non-release builds can mislead optimization analysis.
- `clang-offload-bundler --list` may print no visible lines in some shells;
  extracting `.hip_fatbin` and listing that payload is more reliable.


### A.7 Artifacts generated by this workflow

- `/tmp/rocmgemv_release.hip_fatbin`
- `/tmp/rocmgemv_release_gfx906.o`
- `/tmp/rocmgemv_release_gfx906.disasm`
- `/tmp/rocmgemv_release_gfx906.syms`

These files are safe to delete after analysis.


## Appendix B: Comprehensive rocprof Bottleneck Workflow (Repeatable)

This appendix captures the exact profiling workflow used to diagnose whether
ratio-VNNI kernels are memory-bound, compute-bound, or occupancy-limited on
gfx906. It is designed to be copy/paste repeatable for every tuning iteration.

### B.0 One-command wrapper (recommended)

Use the automation script when you want the full workflow in one command:

```bash
./scripts/profile_ratio_vnni_rocprof.sh
```

Optional output root override:

```bash
./scripts/profile_ratio_vnni_rocprof.sh /tmp/rocprof_ratio_vnni_run_custom
```

The script runs:
- full timing pass (`--stats`),
- split counter passes under rocprof HW limits,
- merged per-kernel summary generation (`kernel_bottleneck_summary.txt` and `.md`).

The remaining sections document the same workflow step-by-step.

### B.1 Scope and intent

Use this workflow when micro-optimizations produce noisy perf changes and you
need hard evidence of the current bottleneck.

Goals:
- capture kernel time share across INT8 baseline and ratio kernels,
- gather compute-vs-memory counters for those kernels,
- consolidate all metrics into one per-kernel summary table.


### B.2 Canonical benchmark command (release)

Always profile the same release test target and filter:

```bash
HSA_VISIBLE_DEVICES=0 ./build_v2_release/tests/v2/v2_perf_rocm_ratio_vnni_kernel \
  --gtest_filter=ROCmRatioVNNIPerfTest.Phase1Q4AndIQ4SpeedupVsInt8VNNI \
  --gtest_color=no
```


### B.3 Pass 1: Full timing dataset (`--stats`)

```bash
OUT=/tmp/rocprof_ratio_full_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

HSA_VISIBLE_DEVICES=0 rocprof --stats -o "$OUT/rocprof.csv" \
  ./build_v2_release/tests/v2/v2_perf_rocm_ratio_vnni_kernel \
  --gtest_filter=ROCmRatioVNNIPerfTest.Phase1Q4AndIQ4SpeedupVsInt8VNNI \
  --gtest_color=no > "$OUT/test_stdout.log" 2>&1 || true

ls -la "$OUT"
```

Primary files:
- `rocprof.stats.csv`: kernel-level total and average duration
- `rocprof.csv`: per-dispatch records
- `test_stdout.log`: benchmark table for context


### B.4 Pass 2: Counter profiling with kernel filter

`rocprof` v1 can reject large metric sets due to HW block limits. Use split
passes and kernel filtering:

```bash
BASE=/tmp/rocprof_ratio_counters_split_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BASE"

run_set() {
  name="$1"
  metrics="$2"
  out="$BASE/$name"
  mkdir -p "$out"

  cat > "$out/metrics.txt" << EOF
gpu: 0
kernel: gemv_ratio_vnni_grid_kpar_kernel_t gemv_int8_int8_grid_kpar_vnni_kernel_t
pmc: $metrics
EOF

  HSA_VISIBLE_DEVICES=0 rocprof -i "$out/metrics.txt" --stats -o "$out/rocprof.csv" \
    ./build_v2_release/tests/v2/v2_perf_rocm_ratio_vnni_kernel \
    --gtest_filter=ROCmRatioVNNIPerfTest.Phase1Q4AndIQ4SpeedupVsInt8VNNI \
    --gtest_color=no > "$out/test_stdout.log" 2>&1 || true
}

# Instruction/ALU mix (fits HW limit)
run_set g_instr_main "VFetchInsts FlatVMemInsts VALUBusy SALUBusy VALUInsts SALUInsts Wavefronts"

# Scalar fetch in separate pass
run_set g_sfetch "SFetchInsts"

# Memory behavior split by rocprof HW limits
run_set g_mem_main "FetchSize MemUnitBusy MemUnitStalled ALUStalledByLDS"
run_set g_writes   "WriteSize"
run_set g_l2hit    "L2CacheHit"

# Misc utilization counters
run_set g_misc "GPUBusy WriteUnitStalled VALUUtilization"

find "$BASE" -maxdepth 2 -type f | sort
```


### B.5 Consolidate split passes into one summary

The script below merges all split-pass `rocprof.csv` files and prints one
summary per kernel.

```bash
python3 - << 'PY'
import csv,glob,statistics

base = sorted(glob.glob('/tmp/rocprof_ratio_counters_split_*'))[-1]
csvs = glob.glob(base + '/*/rocprof.csv')

metrics = [
  'GPUBusy','MemUnitBusy','MemUnitStalled','WriteUnitStalled','L2CacheHit',
  'VALUUtilization','VALUBusy','SALUBusy','Wavefronts','VALUInsts','SALUInsts',
  'VFetchInsts','SFetchInsts','FlatVMemInsts','FetchSize','WriteSize','ALUStalledByLDS'
]

per_kernel = {}
for p in csvs:
    with open(p, newline='') as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        present = [m for m in metrics if m in cols]
        for row in r:
            k = row.get('KernelName', '')
            if 'gemv_ratio_vnni_grid_kpar_kernel_t' not in k and 'gemv_int8_int8_grid_kpar_vnni_kernel_t' not in k:
                continue
            d = per_kernel.setdefault(k, {
                'count': 0,
                'dur': [],
                'arch_vgpr': row.get('arch_vgpr', ''),
                'sgpr': row.get('sgpr', ''),
                'wgr': row.get('wgr', ''),
            })
            d['count'] += 1
            if row.get('DurationNs'):
                d['dur'].append(float(row['DurationNs']))
            for m in present:
                v = row.get(m, '')
                if v != '':
                    d.setdefault(m, []).append(float(v))

print('BASE', base)
for k, v in sorted(per_kernel.items(), key=lambda kv: -sum(kv[1].get('dur', []))):
    total_ms = sum(v.get('dur', [])) / 1e6
    avg_us = statistics.mean(v.get('dur', [])) / 1e3 if v.get('dur') else float('nan')
    print('\nKERNEL:', k)
    print(f"  calls={v['count']} total_ms={total_ms:.3f} avg_us={avg_us:.3f} arch_vgpr={v['arch_vgpr']} sgpr={v['sgpr']} wgr={v['wgr']}")
    for m in metrics:
        vals = v.get(m)
        if vals:
            print(f"  {m}={statistics.mean(vals):.3f}")
PY
```


### B.6 Quick interpretation guide

Use these heuristics for decode GEMV kernels:

- **Memory-bound** signature:
  - high `MemUnitBusy` (often 60-90%+),
  - low `VALUBusy`,
  - throughput sensitivity to memory-layout/coalescing changes.

- **Compute/decode-bound** signature:
  - low-to-moderate `MemUnitBusy` (~10-30%),
  - high `VALUBusy`, high `VALUInsts`,
  - regressions from extra decode arithmetic even if loads are reduced.

- **Occupancy/register pressure clues**:
  - high `arch_vgpr`/`sgpr` plus reduced effective wave scheduling,
  - larger tile variants not improving despite similar memory pressure.


### B.7 Common pitfalls and notes

- `rocprof --metrics` is not valid in this v1 tool mode; use `-i metrics.txt`.
- Large metric lists can exceed HW limits; split into multiple passes.
- Keep command line, device, and build type fixed across iterations.
- Profiling overhead changes absolute timings; use profiles for bottleneck
  directionality, not for final speedup claims.


### B.8 Artifact checklist

For each pass directory, keep:
- `metrics.txt` (exact counters used)
- `rocprof.csv` (raw per-dispatch counters)
- `rocprof.stats.csv` (time-share summary)
- `test_stdout.log` (benchmark context)

This is sufficient to reproduce and compare future tuning iterations.


### B.9 Hot-loop disassembly findings (mode3 baseline)

Targeted ISA extraction was performed from the embedded ROCm fatbin in the
release binary for the mode3 decode hot family (`<128,1,true,...>`). The
representative symbol disassembled was:

- `_Z34gemv_ratio_vnni_grid_kpar_kernel_tILi128ELi1ELb1EEvPKaPKhS1_Piiiih`

Key static observations from the back-edge loop window (roughly `0x8500-0x8d30`):

- Total decoded instructions in window: `373`
- Scalar (`s_*`) instructions in loop window: `11`
- Vector (`v_*`) instructions in loop window: `360`
- Global memory ops in loop window: `2` (`global_load_*`)

Top opcode families in the loop window:

- `v_and_b32_e32` (90)
- `v_cndmask_b32_e32` (64)
- `v_lshrrev_b64` (32)
- `v_lshrrev_b32_e32` (18)
- `v_mul_i32_i24_e32` (16)
- `v_cmp_gt_u16_e32` (16)
- `v_lshlrev_b32_e32` (16)

This indicates the hot loop is dominated by nibble/lane decode and pack logic,
not by scalar control instructions alone.

Loop-carried scalar/control sequence still present each iteration:

- `s_ashr_i32` + `s_add_u32` + `s_addc_u32` (address update)
- `s_load_dwordx8` (per-iteration scalar table/load block)
- `s_add_i32` (loop counters)
- `s_cmp_lt_i32` + `s_cbranch_scc1` (back-edge control)

#### Elimination shortlist from ISA evidence

1. **Reduce vector bit-manip chain in mode3 decode path**
   - Target: repeated `v_and`/`v_cndmask`/`v_lshrrev`/`v_cmp` clusters.
   - Approach: fold more decode into fewer permute-table operations (or wider
     packed transforms) to lower total vector integer op count.

2. **Trim loop-carried scalar address/control overhead**
   - Target: per-iteration `s_ashr/s_add/s_addc/s_cmp/s_cbranch` sequence.
   - Approach: unroll by 2 blocks (if register-safe) and amortize scalar pointer
     and branch work over more decode-dot work.

3. **Hoist/compact scalar setup where legality permits**
   - Target: scalar constants/index setup that can be reused across iterations.
   - Approach: keep invariant scalar setup outside the inner loop and reuse SGPR
     state instead of regenerating equivalent forms.

Cross-check with counter data from this same session:

- Prefetch-on increased `SALUInsts` and `SALUBusy` materially, while runtime
  did not improve.
- Memory pressure remained moderate (`MemUnitBusy` around ~50%, low
  `MemUnitStalled`), reinforcing decode/control efficiency as next lever.

Quick follow-up experiment (`2026-02-13`):

- Tried a low-risk mode3 `CPT=1` inner-loop unroll-by-2 variant (release build).
- Hot-kernel check (`<128,1,true,3,...>`): `avg_us` moved from ~`35.61` to
  ~`36.13` (regression), so the change was reverted.
- Signal: reducing scalar back-edge overhead alone did not compensate for the
  added vector/register-side pressure in this path.

Second follow-up experiment (`2026-02-13`):

- Hoisted loop-invariant address math in the inner `b` loop using running
  pointers (`a4`, `payload16`, `ratio`) rather than recomputing linear offsets
  each iteration.
- Targeted hot-kernel profile (`<128,1,true,3,...>`) moved from ~`35.61us` to
  ~`35.36us` (about `-0.7%`), with lower scalar pressure than the previous
  unroll attempt.
- This variant was later reverted in source after interleaved A/B validation.

Interleaved validation pass (`2026-02-13`, 10 paired runs, release):

- Method: alternating `A,B,A,B,...` sequence with fixed env (`DECODE_MODE=3`,
  `CPT=1`, `MODE3_CONST_LUT=1`, `PREFETCH_NEXT=0`, `MODE3_KB_TUNE=0`).
- Binaries:
  - A = current hoist variant
  - B = temporary baseline (hoist reverted)
- Artifacts: `/tmp/iq4_interleaved_ab/run_20260213_165821`
- Result (global ratio/int8 speedup):
  - A mean/median: `1.42931x` / `1.42213x`
  - B mean/median: `1.43380x` / `1.42796x`
  - Paired delta (A-B): mean `-0.00450x` (`-0.31%`), median `-0.00580x`
    (`-0.41%`)
- Conclusion: under interleaved pairing, the hoist change is neutral-to-slightly
  negative on this benchmark; no high-confidence throughput gain observed.

### B.10 Addendum: post-cleanup noise check (`2026-02-13`)

After canonicalizing the IQ4 mode3 path (removing experimental decode-mode/CPT
branches in dispatch), we ran additional repeats to check whether a single
`1.3899x` result was regression or noise.

Six repeated runs (same release binary / same affinity and MPI settings):

- `1.42254x`, `1.41783x`, `1.43116x`, `1.44540x`, `1.44748x`, `1.44725x`
- Mean: `1.43528x`
- Median: `1.43828x`
- Stddev: `0.01210x`
- Range: `[1.41783x, 1.44748x]`

Comparison points:

- Versus one-off `1.38996x`: `+3.26%`
- Versus earlier 3-run post-revert baseline (`1.40580x`, `1.41667x`,
  `1.41372x`, mean `1.41206x`): `+1.64%`

Interpretation: the `1.3899x` value is most likely a low outlier/noise run,
not evidence of sustained regression after cleanup.

#### Repro command (6-run repeat + summary)

```bash
python3 - << 'PY'
import os, re, statistics, subprocess

bin_path = '/workspaces/llaminar/build_v2_release/tests/v2/v2_perf_rocm_ratio_vnni_kernel'
env = os.environ.copy()
env.update({
  'LLAMINAR_LOG_LEVEL': 'INFO',
  'HWLOC_COMPONENTS': '-gl,-opencl',
  'OMP_NUM_THREADS': '28',
  'OMP_PLACES': 'sockets',
  'OMP_PROC_BIND': 'close',
  'OMP_NESTED': 'false',
  'OMP_DYNAMIC': 'false',
  'KMP_AFFINITY': 'granularity=fine,compact,1,0',
  'KMP_BLOCKTIME': '0',
  'OPENBLAS_NUM_THREADS': '28',
  'GOTO_NUM_THREADS': '28',
  'MKL_NUM_THREADS': '28',
  'MKL_DYNAMIC': 'false',
  'OMPI_MCA_mpi_leave_pinned': '1',
  'OMPI_MCA_btl_vader_single_copy_mechanism': 'none',
  'OMPI_MCA_btl_openib_allow_ib': '1',
  'HSA_OVERRIDE_GFX_VERSION': '9.0.6',
})

cmd = [
  'mpirun', '-np', '1', '--bind-to', 'socket', '--map-by', 'socket',
  '--mca', 'mpi_leave_pinned', '1',
  '--mca', 'btl_vader_single_copy_mechanism', 'none',
  bin_path,
]
pat = re.compile(r'Average ratio/int8 speedup:\\s*([0-9.]+)x')

vals = []
for i in range(1, 7):
  p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  m = pat.search(p.stdout)
  if not m:
    print(f'run={i:02d} speedup=NaN rc={p.returncode}')
    continue
  v = float(m.group(1))
  vals.append(v)
  print(f'run={i:02d} speedup={v:.6f} rc={p.returncode}')

if vals:
  print('values=' + ','.join(f'{v:.6f}' for v in vals))
  print(f'mean={sum(vals)/len(vals):.6f}')
  print(f'median={statistics.median(vals):.6f}')
  print(f'stdev={statistics.pstdev(vals) if len(vals) > 1 else 0.0:.6f}')
  print(f'min={min(vals):.6f}')
  print(f'max={max(vals):.6f}')
PY
```

### B.11 Addendum: Q4_0 canonicalization + post-prune verification (`2026-02-13`)

We switched focus to the `Q4_0` path and made the same permute-based decode
approach canonical there:

- Added a linear-codebook ratio→perm LUT table (`k_ratio_linear_perm_lut_words`).
- Routed `Q4_0` decode through the `__builtin_amdgcn_perm` low/high pair path.
- Canonicalized ratio-VNNI kernel instantiation to `CPT=1` for both IQ4 and
  linear codebooks.
- Pruned dead Q4 legacy decode branches/helpers (old LUT64/map helper path and
  stale CPT>1 logic in the ratio kernel body).

Post-prune regression check (5 repeated runs, same release harness/env):

- Global speedup mean/median: `1.60633x` / `1.60211x`
- `Q4_0` means:
  - Q/Wo `3584x3584`: `1.24197x`
  - FFN Down `3584x18944`: `1.78673x`
  - FFN Gate `18944x3584`: `1.82266x`
  - 3-shape Q4 average mean/median: `1.61712x` / `1.61897x`

Conclusion: no regression after pruning; performance is consistent with (and
slightly above) the pre-prune tuned Q4_0 runs.

#### Repro command (5-run Q4_0 + global summary)

```bash
python3 - << 'PY'
import os, re, subprocess, statistics

bin_path = '/workspaces/llaminar/build_v2_release/tests/v2/v2_perf_rocm_ratio_vnni_kernel'
env = os.environ.copy()
env.update({
  'LLAMINAR_LOG_LEVEL': 'INFO',
  'HWLOC_COMPONENTS': '-gl,-opencl',
  'OMP_NUM_THREADS': '28',
  'OMP_PLACES': 'sockets',
  'OMP_PROC_BIND': 'close',
  'OMP_NESTED': 'false',
  'OMP_DYNAMIC': 'false',
  'KMP_AFFINITY': 'granularity=fine,compact,1,0',
  'KMP_BLOCKTIME': '0',
  'OPENBLAS_NUM_THREADS': '28',
  'GOTO_NUM_THREADS': '28',
  'MKL_NUM_THREADS': '28',
  'MKL_DYNAMIC': 'false',
  'OMPI_MCA_mpi_leave_pinned': '1',
  'OMPI_MCA_btl_vader_single_copy_mechanism': 'none',
  'OMPI_MCA_btl_openib_allow_ib': '1',
  'HSA_OVERRIDE_GFX_VERSION': '9.0.6',
})

cmd = [
  'mpirun', '-np', '1', '--bind-to', 'socket', '--map-by', 'socket',
  '--mca', 'mpi_leave_pinned', '1',
  '--mca', 'btl_vader_single_copy_mechanism', 'none',
  bin_path,
]

pat_global = re.compile(r'Average ratio/int8 speedup:\\s*([0-9.]+)x')
pat_qwo = re.compile(r'Q4_0\\s+│\\s+Q/Wo 3584x3584\\s+│[^\\n]*?│\\s*([0-9.]+)\\s*║')
pat_down = re.compile(r'Q4_0\\s+│\\s+FFN Down 3584x18944\\s+│[^\\n]*?│\\s*([0-9.]+)\\s*║')
pat_gate = re.compile(r'Q4_0\\s+│\\s+FFN Gate 18944x3584\\s+│[^\\n]*?│\\s*([0-9.]+)\\s*║')

gvals, qwo_vals, down_vals, gate_vals, qavg_vals = [], [], [], [], []

for i in range(1, 6):
  p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  out = p.stdout

  mg = pat_global.search(out)
  mq = pat_qwo.search(out)
  md = pat_down.search(out)
  mge = pat_gate.search(out)

  gv = float(mg.group(1)) if mg else float('nan')
  qwo = float(mq.group(1)) if mq else float('nan')
  qdown = float(md.group(1)) if md else float('nan')
  qgate = float(mge.group(1)) if mge else float('nan')

  qvals = [v for v in (qwo, qdown, qgate) if v == v]
  qavg = sum(qvals) / len(qvals) if qvals else float('nan')

  if gv == gv: gvals.append(gv)
  if qwo == qwo: qwo_vals.append(qwo)
  if qdown == qdown: down_vals.append(qdown)
  if qgate == qgate: gate_vals.append(qgate)
  if qavg == qavg: qavg_vals.append(qavg)

  print(f'run={i:02d} global={gv:.6f} q4_qwo={qwo:.6f} q4_down={qdown:.6f} q4_gate={qgate:.6f} q4_avg={qavg:.6f} rc={p.returncode}')

if gvals:
  print(f'global_mean={sum(gvals)/len(gvals):.6f} global_median={statistics.median(gvals):.6f}')
if qwo_vals:
  print(f'q4_qwo_mean={sum(qwo_vals)/len(qwo_vals):.6f}')
if down_vals:
  print(f'q4_down_mean={sum(down_vals)/len(down_vals):.6f}')
if gate_vals:
  print(f'q4_gate_mean={sum(gate_vals)/len(gate_vals):.6f}')
if qavg_vals:
  print(f'q4_avg_mean={sum(qavg_vals)/len(qavg_vals):.6f} q4_avg_median={statistics.median(qavg_vals):.6f}')
PY
```

### B.12 Addendum: Q/Wo autosweep defaults (`2026-02-13`)

To stabilize Q/Wo performance, we added a local autosweep harness for square-ish
shapes and promoted the winning tuning into dispatch defaults.

#### What changed

- Added runtime tuning override hooks for ratio-VNNI dispatch:
  - `rocmGemv_ratio_vnni_set_tuning_overrides(linear_tn, linear_kb, iq4_tn, iq4_kb)`
  - `rocmGemv_ratio_vnni_reset_tuning_overrides()`
- Added env-gated autosweep test (`QWoAutoSweepTnKb`) over:
  - `TN ∈ {128, 256}`
  - `KB ∈ {8, 10, 12, 14, 16}`
- Promoted square-ish defaults in production dispatch for both codebooks:
  - `TN = 128`
  - `KB = 16`

#### Autosweep winners (Q/Wo 3584x3584)

- `Q4_0`: `TN=128`, `KB=16`, speedup `1.42474x`
- `IQ4_NL`: `TN=128`, `KB=16`, speedup `1.42478x`

#### Full perf table after promoting defaults

Measured from `V2_Perf_ROCmRatioVNNIKernel` (release build, full table run):

| Format | Shape | INT8 ms | Ratio ms | Speedup |
|---|---|---:|---:|---:|
| Q4_0 | Q/Wo 3584x3584 | 0.0305759 | 0.0237546 | 1.28716x |
| Q4_0 | FFN Down 3584x18944 | 0.101738 | 0.0570932 | 1.78197x |
| Q4_0 | FFN Gate 18944x3584 | 0.101706 | 0.0573652 | 1.77296x |
| IQ4_NL | Q/Wo 3584x3584 | 0.0305813 | 0.0232159 | 1.31725x |
| IQ4_NL | FFN Down 3584x18944 | 0.103424 | 0.0578558 | 1.78761x |
| IQ4_NL | FFN Gate 18944x3584 | 0.104213 | 0.0568799 | 1.83216x |

Summary metrics from the same run:

- Global average speedup: `1.62985x`
- Q4_0 average speedup: `1.61403x`
- IQ4_NL average speedup: `1.64568x`

Interpretation:

- The autosweep winner improved Q/Wo versus the prior square-ish default path,
  while preserving strong FFN performance and healthy overall averages.

#### Repro commands

Autosweep run:

```bash
LLAMINAR_RUN_QWO_AUTOSWEEP=1 \
ctest --test-dir /workspaces/llaminar/build_v2_release \
  -R '^V2_Perf_ROCmRatioVNNIKernel$' --output-on-failure -V
```

Full table run (defaults only):

```bash
ctest --test-dir /workspaces/llaminar/build_v2_release \
  -R '^V2_Perf_ROCmRatioVNNIKernel$' --output-on-failure -V
```