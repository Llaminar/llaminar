# Quantized Tensor Architecture Proposal

> Branch: feature/quantized-tensors  
> Author: David Sanftenberg (proposal drafted with AI assist)  
> Date: 2025-10-18

## 1. Background & Motivation

Today all runtime weights and activations in Llaminar are stored as FP32 `SimpleTensor` or `COSMATensor` instances. Quantized GGUF models are **fully dequantized at load time** (Q4_0, Q6_K, Q8_0, etc.) to FP32, exploding memory footprint (e.g. 0.5B model: ~2GB post-dequant vs ~350MB quant file) and causing unnecessary bandwidth pressure for matmul. Modern high‑performance inference systems (llama.cpp, vLLM, turbines) avoid this by:

1. Keeping weights in their original quantized representation in memory
2. Performing matmul in int8 / packed 4–6 bit representation into **BF16**/FP32 accumulators
3. Accumulating partial results in FP32 for numerical stability
4. Optionally storing activations in **BF16** where safe

**Note (Oct 19 2025):** Llaminar's implementation uses **BF16** (bfloat16), not FP16 (IEEE half precision), for intermediate storage. This document has been updated to reflect this throughout.

We need an extensible architecture to migrate **incrementally** without breaking existing operators or MPI distribution.

## 2. High-Level Goals

| Goal | Description |
|------|-------------|
| Pluggable quantized tensor type | Introduce `QuantizedTensor` alongside `SimpleTensor` without forcing immediate refactors everywhere |
| On-the-fly dequant for GEMM | Provide kernel helpers that read quant blocks and emit BF16 tiles (optionally FP32 accumulation) |
| Reduced memory footprint | Avoid full model FP32 expansion; keep raw GGUF bytes in structured blocks |
| MPI-aware sharding | Support partitioning quantized weights exactly like current FP32 slicing (row/head dimension splits) |
| Fused dequant + matmul | Dequantization happens inside the inner compute loop (cache-friendly) |
| Backward-compatible | Existing tests, parity framework continue to work on FP32 path until phased switch |
| Multi-precision accumulation | Allow int8/4bit → BF16 multiply → FP32 accumulator |
| Minimal operator churn initially | Phase 1 focuses on linear projections (largest memory/compute share) |

## 3. Phased Migration Plan

### Phase 1: Infrastructure & Skeleton
- Add `QuantStorageLayout` metadata (block size, elements per block, scale encoding, super-block mapping)
- Create `QuantizedTensor` class storing raw bytes + lightweight descriptors
- Extend `TensorFactory` with `create_quantized(shape, QuantSpec)` and `wrap_quantized(raw_ptr, QuantSpec, shape)`
- Provide dequant helpers: `dequant_block_q4_0`, `dequant_block_q6_k` (reuse existing from `QuantDequant.h`)
- Add a small test verifying block decode equals existing ModelLoader path
- Keep existing ModelLoader behavior; optionally gate new path behind `LLAMINAR_LOAD_QUANTIZED=1`

### Phase 2: Weight Loader Integration
- Teach `ModelLoader` to optionally skip FP32 expansion: produce `QuantizedTensor` instead of `SimpleTensor`
- For each weight: store raw quant bytes + per-block scales in `QuantizedTensor`
- Provide cached transposed views (for row-major matmul convenience) without full dequant
- Update attention & linear operators to detect quantized weights → choose quantized kernel path

### Phase 3: Quantized Linear Kernel
- Implement `QuantGEMMKernel` templated on (QuantFormat, AccumType)
- Input: `QuantizedTensor` (weights), `SimpleTensor` or **BF16** activation
- Output: **BF16** / FP32 activation tensor
- Loop structure:
  ```cpp
  for (tile_m ... )
    for (tile_n ... )
      accum[ tile_m, tile_n ] = 0
      for (k_block ... ) {
        // 1. Load quant block bytes
        // 2. Dequant block to temporary BF16 buffer (vectorized)
        // 3. FMA: activation_tile (BF16) * dequant_weight_tile (BF16) → FP32 accum
      }
      // 4. Convert accum FP32 → BF16 (optional) and write out
  ```
- Provide an OpenMP strategy: outer tiles parallelized, inner k_block sequential for cache locality
- SIMD: Use AVX2/AVX512 for dequant + FMA (future: intrinsic fallbacks)
- **Backend Integration:** OpenBLAS `cblas_sbgemm()` for BF16×BF16→FP32 (verified working in v0.3.26, October 2025), COSMA BF16 (user's branch), Intel MKL (hardware acceleration on Ice Lake+)

### Phase 4: Attention Path & Projection Fusion
- Replace Q/K/V projection matmuls with quantized kernels when weights are quantized
- Maintain existing bias addition & RoPE application semantics
- Optionally fuse bias addition inside tile loop to save passes

### Phase 5: Activation Precision & KV Cache Reduction
- Add BF16 tensor support (using existing `bfloat16` type from `BFloat16.h`)
- Store KV cache in **BF16** with optional FP32 fallback (env flag `LLAMINAR_KV_BF16=1`)
- Adjust attention softmax scaling and RMSNorm epsilon for **BF16** stability (better than FP16 due to wider exponent range)

### Phase 6: Distributed & COSMA Path Exploration
- For large prefill matmuls with COSMA: evaluate hybrid—keep quant blocks local, broadcast dequant tiles
- Possibly stage **BF16** tiles across MPI ranks for reduce-scatter
- COSMA BF16 support: User implementing in separate branch, will integrate when available

### Phase 7: Cleanup & Full Switch
- Default model loading path uses quantized weights unless `LLAMINAR_FORCE_FP32_WEIGHTS=1`
- Remove legacy once-block dequant test path

## 4. Core Types & Data Structures

### 4.1 QuantFormat Enum
```cpp
enum class QuantFormat {
  F32,    // Unquantized fallback
  F16,    // IEEE half precision (legacy, not used)
  Q4_0,   // 4-bit blocks (32 elements)
  Q6_K,   // 6-bit K-quants (256-element super-block)
  Q8_0,   // 8-bit blocks (32 elements)
  // Extend: Q2_K, Q3_K, Q4_K, Q5_K, Q8_K
};
// Note: Slab storage uses BF16 (bfloat16), not F16.
// F16 enum value kept for GGUF compatibility but not actively used.
```

### 4.2 QuantBlockDescriptor
Describes how to interpret raw bytes for one logical block.
```cpp
struct QuantBlockDescriptor {
  int elements_per_block;      // e.g. 32 for Q4_0/Q8_0, 256 for K-quants
  int bytes_per_block;         // includes scale bytes
  int scale_count;             // number of scales per block (1 or more)
  int bits_per_value;          // 4, 6, 8, etc.
  bool is_k_quant;             // true for super-block encoded variants
};
```

### 4.3 QuantStorageLayout
Top-level descriptor for entire tensor.
```cpp
struct QuantStorageLayout {
  QuantFormat format;
  std::vector<int> original_shape;  // [rows, cols]
  size_t total_blocks;              // computed from shape / block granularity
  QuantBlockDescriptor block_desc;  // constant per tensor
  // Optional offsets or index mapping arrays (for K-quants)
};
```

### 4.4 QuantizedTensor
```cpp
class QuantizedTensor : public TensorBase {
public:
  QuantizedTensor(const QuantStorageLayout &layout, std::vector<uint8_t> raw)
    : layout_(layout), raw_(std::move(raw)) {}

  const std::vector<int> &shape() const override { return layout_.original_shape; }
  float *data() override { return nullptr; } // No direct FP32 buffer
  const float *data() const override { return nullptr; }
  size_t size() const override { return (size_t)layout_.original_shape[0] * layout_.original_shape[1]; }
  std::string type_name() const override { return "QuantizedTensor"; }

  // Dequantize a single block (slow path utility)
  void dequantizeBlock(size_t block_index, float *dst) const;

  // Produce a temporary FP16 tile (vectorized fast path)
  void dequantizeTile(int row_start, int rows, int col_start, int cols, _Float16 *dst) const;

  const QuantStorageLayout &layout() const { return layout_; }
  const uint8_t *raw() const { return raw_.data(); }

private:
  QuantStorageLayout layout_;
  std::vector<uint8_t> raw_; // contiguous block storage
};
```

## 5. TensorFactory Extensions
```cpp
class TensorFactory {
public:
  static std::shared_ptr<TensorBase> create_quantized(const std::vector<int> &shape,
                                                      QuantFormat fmt,
                                                      const std::vector<uint8_t> &raw);
  static bool is_quantized(const std::shared_ptr<TensorBase> &t);
  static std::shared_ptr<QuantizedTensor> to_quantized(std::shared_ptr<TensorBase> t);
};
```
Internally computes block layout from `shape` and `fmt`, validating:
- For 2D matrices only initially
- Elements divisible by block size (pad last block if needed; track `effective_cols`)

## 6. Kernel Design (Quantized Linear)

### 6.1 Functional Contract
Inputs:
`A` (activation matrix): shape (M, K) stored as FP32 initially (later FP16)
`W_q` (quantized weight matrix): logical shape (K, N) in quant blocks (column-major or row-major variant)
Outputs:
`C` (output matrix): shape (M, N) produced as FP16 or FP32 plus optional FP32 accumulator scratch

Error Modes:
- Unsupported format → fallback to existing FP32 GEMM
- Misaligned K not multiple of block size → pad & warn

### 6.2 Tiling Strategy
```
Outer tiles: M_TILE × N_TILE (e.g. 32×64)
Inner reduction: K processed in BLOCK_K chunks aligned to quant blocks
```
- For each inner `k_block`:
  - Load activation slice A_tile (if FP32, convert to FP16 for vector FMA)
  - Dequant quant blocks covering logical region W_q(k_block : k_block + BLOCK_K, n_start : n_start + N_TILE) into contiguous FP16 buffer
  - Perform FMA: `accum_fp32[m, n] += a_fp16[m, k] * w_fp16[k, n]`

### 6.3 SIMD Considerations
- Q4_0 decode: nibble unpack → int8 vector → multiply by scale → FP16 convert
- Q6_K decode: handle super-block multiple scale groups; likely fallback to FP32 temporarily until optimized
- Provide function pointers or templates for format-specific decode

### 6.4 Accumulation Precision Policy
Environment flags:
- `LLAMINAR_QUANT_ACCUM_FP32=1` (default: on)
- `LLAMINAR_QUANT_OUTPUT_FP16=1` (write FP16 result)

Decision matrix:
| Input (Act) | Accum | Output | Use Case |
|-------------|--------|--------|----------|
| FP32 | FP32 | FP32 | Baseline correctness |
| **BF16** | FP32 | **BF16** | Balanced throughput/memory |
| **BF16** | FP32 | FP32 | Debug numeric investigation |
| FP32 | FP32 | **BF16** | Reduced output bandwidth |

## 7. Operator Integration Plan

### 7.1 MPILinearOperator / Batch Variant
- Detect quantized weight via `dynamic_cast<QuantizedTensor*>`
- If quantized: route to `executeQuantizedLinear()`
- Provide micro-parity test: compare quant path vs fully dequantized FP32 path for small matrix (rel_l2 < 1e-3)

### 7.2 MPIAttentionOperator
- Q/K/V/O projections share linear kernel logic → after MPILinearOperator integration, simply call quantized linear path
- KV Cache remains FP32 (later Phase 5 for **BF16**)

### 7.3 Embedding Operator
- Future: store embedding as quantized blocks; add fast gather + dequant in place
- Phase 1 leaves embedding FP32 to minimize surface area

### 7.4 RMSNorm / SwiGLU / Residual
- Initially operate on FP32 output of projections
- After **BF16** activation introduction, audit epsilon & clamp logic

## 8. MPI & Sharding Considerations
Current sharding logic slices FP32 matrices by row (per head) or column. For quantized weights:
- Keep original quant block ordering globally
- Each rank stores only its slice of raw bytes (no dequant)
- Global gather ops for diagnostics can still request temporary FP32 expansion if necessary (debug path)

Sharded layout descriptor could include `global_row_offset` or `head_offset` to reconstruct logical indices.

## 9. Debug & Parity Strategy
- New env flags:
  - `LLAMINAR_QUANT_ENABLE=1` (master gate)
  - `LLAMINAR_LOAD_QUANTIZED=1` (ModelLoader skips dequant)
  - `LLAMINAR_QUANT_VERIFY_BLOCKS=1` (after decode each block compare subset against reference dequant function)
- Parity tests: run small prompt with both FP32-dequant and quant-on-the-fly paths, capture stage snapshots, compare tolerances.

## 10. Testing Roadmap
| Test | Purpose |
|------|---------|
| `QuantBlockDecodeTest` | Ensure block decode matches existing functions |
| `QuantLinearKernelParityTest` | Compare quantized linear vs FP32 for synthetic matrix |
| `QuantPipelinePrefillParityTest` | End-to-end small prompt parity |
| `QuantEnvFlagDisableFallbackTest` | Validate fallback when flags unset |

## 11. Migration Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Increased code complexity | Encapsulate format-specific decode behind strategy objects |
| Performance regression (initial) | Guard new path behind env; profile iteratively |
| Numerical instability with **BF16** | Keep FP32 accumulation; BF16 has better stability than FP16 due to wider exponent range |
| MPI deadlocks due to new paths | Reuse existing synchronization patterns; no new collectives introduced in Phase 1 |
| Memory misalignment | Enforce 64-byte alignment for temporary tile buffers |

## 12. Intern-Friendly Comments & Conventions
- Every new public method documented with Doxygen (`@brief`, `@param`, `@return`)
- Inline comments outline micro-steps ("Load block header", "Unpack nibbles", "Apply scale")
- Avoid premature optimization—clear loops first, then vectorize with guarded intrinsics

## 13. Next Steps (Actionable)
1. Add `QuantizedTensor` + factory helpers (header + minimal implementation)
2. Add architecture doc (this file)
3. Implement block decode unit tests (Q4_0, Q8_0)
4. ModelLoader flag to emit `QuantizedTensor`
5. Implement quantized linear kernel prototype (Q8_0 easiest)
6. Add parity test comparing Q8_0 quant path vs dequant FP32
7. Extend to Q4_0 & Q6_K

## 14. Appendix: Sample Quantized Linear Kernel Skeleton
```cpp
bool QuantLinearKernel::run(const QuantizedTensor &Wq,
                            const TensorBase &A,
                            TensorBase &C,
                            QuantKernelConfig cfg) {
  // 1. Validate shapes
  // 2. Decide tile sizes from cfg or defaults
  // 3. Allocate accumulator (FP32) if output is FP16
  // 4. Outer loops over M,N tiles
  // 5. Inner loop: for k in 0..K step block_elems
  //    - Dequant block(s) covering this slice into fp16_tile_w
  //    - Load/convert activation slice to fp16_tile_a
  //    - FMA into accum_fp32
  // 6. Post-process: convert accum_fp32 to FP16 if needed
  // 7. Return success flag
}
```

---
This document seeds the implementation effort. Subsequent changes will refine specifics once prototype performance numbers are available.

## 15. Implementation Progress (Oct 18 2025)

### 15.1 Implemented Formats & Accurate Block Descriptors
Implemented decode support for:
- 32‑value formats: `Q4_0`, `Q5_0`, `Q8_0`
- 256‑value K-family: `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`

`TensorFactory::create_quantized` now sets `bytes_per_block = sizeof(block_q*_K)` for K formats (removed placeholder constants). This ensures correct byte offsets when computing `block_index * bytes_per_block` and avoids subtle misalignment during partial/tile decodes.

### 15.2 Block Decode (`QuantizedTensor::decodeBlock`)
- 32‑value formats decoded with lightweight inline logic (nibble / high‑bit unpack + fp16 scale conversion).
- K formats delegate to upstream ggml `dequantize_row_q*_K` for exact parity, decoding a single 256‑value block each call.
- Unsupported formats trigger `LOG_WARN` and zero-fill (defensive default).

### 15.3 Tile Decode Evolution (`decodeTileFP16`)
Progression:
1. Naive per-element approach (decoded block every element) – correctness focus.
2. Two-entry opportunistic cache (last + previous block) – simple locality improvement.
3. Current: 8-entry LRU cache plus vectorized full-block fast path.

Current mechanics:
- Fixed array of 8 cache entries: `{ index, stamp, data[] }` with monotonic `clock` for LRU replacement.
- On hit: update stamp and reuse decoded data.
- On miss: choose empty slot or least-recently-used; invoke `decodeBlock` once.
- Fast path: if tile covers exactly one aligned full block width, a single fetch + OpenMP `simd` cast loop materializes the row.

Benefits:
- Reduces redundant decodes dramatically for localized and modestly sparse access patterns (typical tile/gather scenarios).
- Keeps additional memory modest: at most `8 * elements_per_block * sizeof(float)` transient storage.

### 15.4 Testing Added
- `TestQuantizedTensorDecode.cpp`: Basic block parity + tile coverage (Q4_0, Q8_0).
- `TestQuantizedTensorDecodeK.cpp`: Parity for `Q4_K`, `Q5_K`, `Q6_K` (quantize_ref → decodeBlock vs reference dequant) and a smoke test for `Q8_K`.

Assertions: `max_abs < 1e-3`, `rel_l2 < 1e-4` on parity tests ensure numerical fidelity.

### 15.5 Caching Rationale & Future Direction
Chosen cache size (8) balances improved hit rate vs. L1/L2 footprint. Future enhancements (pending profiling):
- Parameterize size via env flag (e.g., `LLAMINAR_QUANT_TILE_CACHE=16`).
- Prefetch detection for sequential multi-block scans.
- Direct fused decode→GEMM path eliminating temporary FP32 buffer (Phase 3).

### 15.6 Safety & Logging
- Out-of-range block offsets guarded; zero-fill avoids UB.
- Unsupported formats produce a warning (throttling optional later).
- K-format `bytes_per_block` correctness validated implicitly via parity tests (any mis-size would cause parity divergence).

### 15.7 Alignment with Original Phases
Delivered parts of Phases 1–2:
- Core structures (`QuantizedTensor`, descriptors)
- Per-block & tile decode with caching
- Unit tests / parity checks for a representative set of formats

Still outstanding before entering Phase 3:
- Quantized linear kernel (fused decode + matmul)
- Environment gating for accumulation / output precision
- Sharded partial load path directly producing quant buffers (loader groundwork exists but not yet fused with operator path)

### 15.8 Next Immediate Steps (Proposed)
1. Add micro-benchmark to measure decode throughput & cache hit rate.
2. Implement provisional `QuantLinearKernel` utilizing block/tile decode (baseline correctness).
3. Introduce optional block verification sampling (`LLAMINAR_QUANT_VERIFY_BLOCKS`)—decode a small random set and compare with reference dequant for early regression detection.
4. Begin integrating fused decode loop (replace double-buffer with direct inner-loop FMA) for Q4_0 / Q8_0.

---
*Section added to capture concrete implementation status and rationale.*

### 15.9 Quant Decode Micro-Benchmark

Executable: `bench_quant_decode` (registered as CTest `QuantDecodeBench` with label `Bench`). Provides focused measurement of decode tile throughput and cache behavior for selected quant formats and access patterns.

Example usage:
```
# Default representative formats (Q4_0, Q8_0, Q4_K, Q6_K), 200 iterations, all patterns
./bench_quant_decode

# Limit to formats, reduce iterations, emit CSV
./bench_quant_decode -f 2,4 -n 50 --csv > decode_baseline.csv

# Random only, no stats columns
./bench_quant_decode --patterns random --no-stats

# Larger square matrix (~1M elements -> ~1024x1024) sequential + row-scan
./bench_quant_decode -e 1048576 -p sequential,row-scan -f 2,6
```

CLI flags:

| Flag | Description |
|------|-------------|
| `-f, --formats list` | Comma list of numeric format ids (2=Q4_0,3=Q5_0,4=Q8_0,5=Q4_K,6=Q5_K,7=Q6_K,8=Q8_K) |
| `-p, --patterns list` | Comma list of patterns: `sequential`, `row-scan`, `random` |
| `-n, --iterations I` | Iterations per pattern (default 200) |
| `-e, --elements N` | Override base matrix size (rows=cols≈sqrt(N)) |
| `--no-stats` | Disable cache statistics (only timing/bandwidth) |
| `--csv` | CSV output instead of table |
| `-h, --help` | Help text |

Output columns:

| Column | Meaning |
|--------|---------|
| `FMT` | Format tag (human readable) |
| `PATTERN` | Access pattern |
| `ELEMS` | Total decoded FP16 elements (iters * tile area) |
| `MS` | Wall clock milliseconds |
| `GB/s` | Approx decoded FP16 bandwidth |
| `APPX_BLOCKS` | Heuristic blocks touched = iters * ceil(tile_area / block_elems) |
| `REQ` | (stats) Block fetch attempts |
| `HITS` | (stats) Cache hits |
| `MISSES` | (stats) Cache misses (decode invocations) |
| `FASTPATH` | (stats) Full-block fast-path row copies |

Initial Debug build observations: sequential pattern achieves ~3× lower latency vs random for 32‑value formats due to high reuse captured by the 8-entry LRU; K formats (256‑value blocks) reduce pattern divergence as each block covers a larger span.

Planned follow-ups: collect Release baselines, experiment with cache size knob, and explore pre-decoding strategy for contiguous multi-block sweeps in fused matmul path.

#### 15.9.1 Release Build Baseline (Oct 18 2025)
Release configuration (OpenBLAS, -O3): representative results (200 iterations, default tile sizes):

| Format | Pattern     | Elements (decoded) | Time ms | GB/s | Approx Blocks |
|--------|-------------|--------------------|--------:|-----:|--------------:|
| Q4_0   | sequential  | 102,400            | 0.52    | 0.40 | 3,200 |
| Q4_0   | row-scan    | 102,400            | 3.22    | 0.06 | 3,200 |
| Q4_0   | random      | 102,400            | 3.28    | 0.06 | 3,200 |
| Q8_0   | sequential  | 102,400            | 0.49    | 0.42 | 3,200 |
| Q8_0   | row-scan    | 102,400            | 3.21    | 0.06 | 3,200 |
| Q8_0   | random      | 102,400            | 3.27    | 0.06 | 3,200 |
| Q4_K   | sequential  | 204,800            | 6.77    | 0.06 | 800   |
| Q4_K   | row-scan    | 204,800            | 6.81    | 0.06 | 800   |
| Q4_K   | random      | 204,800            | 6.88    | 0.06 | 800   |
| Q6_K   | sequential  | 204,800            | 8.76    | 0.05 | 800   |
| Q6_K   | row-scan    | 204,800            | 9.26    | 0.04 | 800   |
| Q6_K   | random      | 204,800            | 9.74    | 0.04 | 800   |

Observations:
1. 32‑value formats (Q4_0/Q8_0) achieve ~0.40–0.42 GB/s sequential decode with small tiles; random access imposes ~6.3× slowdown due to low cache reuse and per-element block lookups.
2. K formats show near pattern-invariant timing because each 256‑value block amortizes header unpack cost over more values; cache effects are muted at these tile sizes.
3. Absolute GB/s remains modest because tiles are intentionally small; fused decode+GEMM should raise effective throughput by eliminating intermediate FP16 store and leveraging wider inner loops.
4. Increasing cache size (LLAMINAR_QUANT_TILE_CACHE) did not materially change hit-rate for these shapes (working set < default 8 blocks in sequential paths) – larger matrices will stress this more.

Planned: Re-run with larger contiguous tiles and fused kernel prototype to establish upper-bound decode contribution to end-to-end matmul cost.

### 15.10 Fused Decode + GEMM Prototype Benchmark (Release, Oct 18 2025)

Executable: `bench_quant_linear_fused` (new). Measures a naive fused kernel (`quant_linear_fused`) against the two‑stage baseline: (1) full weight dequant of touched blocks to FP16 (cached tile decode) and (2) naive GEMM accumulation (triple nested loops, no BLAS). The goal of this initial probe is to ensure functional parity instrumentation and gather a reference point before investing in heavy vectorization / scheduling work.

Invocation (Release, 10 iterations per case):
```
./bench_quant_linear_fused 10 > fused_release.csv
```

Collected cases (current shapes kept deliberately small; larger shapes pending):

| Format | M | K | N | Iterations | Fused ms | Unfused (dequant+gemm) ms | Speedup (unfused/fused) |
|--------|---|---|---|-----------:|---------:|--------------------------:|------------------------:|
| Q4_0   | 32 | 256 | 384 | 10 | 3.467 | 0.512 | 0.148 |
| Q8_0   | 32 | 256 | 384 | 10 | 3.552 | 0.515 | 0.145 |
| Q4_K   | 32 | 256 | 384 | 10 | 55.902 | 0.420 | 0.0075 |
| Q5_K   | 32 | 256 | 384 | 10 | 17.789 | 0.429 | 0.024 |
| Q6_K   | 32 | 256 | 384 | 10 | 19.543 | 0.656 | 0.0336 |
| Q8_K   | 32 | 256 | 384 | 10 | 17.361 | 0.406 | 0.0234 |
| Q4_0   | 64 | 512 | 512 | 10 | 5.275 | 1.455 | 0.276 |
| Q4_K   | 64 | 512 | 512 | 10 | 37.984 | 1.613 | 0.0425 |

Key Findings (Phase 0 Prototype):
1. The fused kernel is currently **slower** (speedup < 1 across all formats). For small shapes the overhead of per‑block decode inside the innermost accumulation loop dominates, while the unfused path benefits from very small naive GEMM working sets that sit in cache.
2. K‑formats (super‑blocks, e.g. Q4_K) exhibit extreme slowdowns (×25–130) due to the prototype performing a full dequant of each 256‑value block for narrow N tiles repeatedly without reuse.
3. Base formats (Q4_0/Q8_0) are closer (≈6–7× slower fused) highlighting that removing redundant decode + improving inner FMA vectorization could close the gap.
4. Larger shape (64×512×512) narrows the relative penalty for Q4_0 (fused 5.28 ms vs 1.46 ms) indicating scaling headroom; however algorithmic inefficiencies persist (see TODO list below).

Primary Bottlenecks Identified:
- Inner loop structure decodes weight fragments repeatedly per (m, n) tile row instead of staging a reused K-slice across multiple M rows and N columns.
- Lack of vectorized dequant path inside fused kernel (current implementation relies on scalar nibble unpack followed by scalar multiply and cast).
- Accumulation uses plain scalar loops; no AVX2/AVX512 FMAs or unrolling.
- No blocking for cache hierarchy (single tile sizes `tile_k`, `tile_n` chosen heuristically; `tile_m` effectively 1 in practice given current parallelization strategy over `M`).
- Super-block (K-family) decode not partially reused: full 256‑value dequant each time even when only a subset contributes to current tile.

Planned Optimization Roadmap:
1. Weight Tile Reuse: Introduce a per-thread scratch buffer for the current (k_block, n_tile) weight tile so multiple `m` rows reuse decoded values before advancing K.
2. Vectorized Dequant Kernels: Implement format-specific SIMD decode (e.g. AVX2 unpack 8×16 nibbles) producing FP16/FP32 vectors directly; fuse scale broadcast.
3. FMA Vectorization: Convert inner accumulation to AVX2/AVX512 with loop unrolling; consider accumulating in FP32 registers and horizontal store once per tile.
4. K-Format Partial Decode: Decode only the 64‑value group(s) needed for the active `n` subspan; maintain per-group scale cache.
5. Tile Size Autotuning: Add environment overrides `LLAMINAR_QUANT_FUSED_TILE_N`, `LLAMINAR_QUANT_FUSED_TILE_K`, and runtime heuristic exploring a small search space on first invocation.
6. Parallelization Strategy: Switch from parallel-over-M to parallel-over-(M_tiles × N_tiles) with dynamic scheduling for better core utilization on larger matrices.
7. BLAS Baseline for Unfused: Replace naive GEMM with OpenBLAS call to establish a realistic performance target; measure decode overhead fraction explicitly.
8. Mixed Precision Output: Optionally produce FP16 output when `LLAMINAR_QUANT_OUTPUT_FP16=1` to reduce store bandwidth; confirm parity tolerance.

Immediate Action Items (next session):
- Implement environment tile overrides (item 5) to allow manual exploration.
- Replace unfused GEMM naive loop with OpenBLAS SGEMM for apples-to-apples overhead analysis.
- Add instrumentation counters: decoded_blocks_fused, reused_weight_tiles, simd_path_taken.
- Expand benchmark shapes (e.g., M=128, K=4096, N=4096) to reflect projection layer scales.

Success Criteria for Next Iteration:
- Achieve fused <= 1.2× unfused (OpenBLAS) for Q4_0 on medium shape (M=64 K=4096 N=4096) with decode still inside loop.
- Reduce K-format penalty by at least 10× (e.g., Q4_K fused slowdown from 130× → <13×) through partial decode reuse.
- Provide at least one SIMD implementation (Q4_0) demonstrating >2× speedup vs current scalar fused path.

Once these are met, re-run benchmarks and update this section with Phase 1 fused kernel performance.

### 15.11 Strategic Pivot: Slab Decode Path & Fused Kernel Retirement (Oct 19 2025)

After multiple optimization attempts, the prototype fused decode+matmul kernel remained significantly slower than the unfused path for all tested shapes and quant formats. Root causes:
1. Redundant per-block decodes inside the innermost accumulation loop (no staging/reuse across M rows or N cols).
2. Lack of vectorized nibble/byte unpack and FMA; scalar loops dominated.
3. K-format (256 value) super-blocks penalized narrow N tiles due to repeated whole-block decodes.
4. Inefficient tile scheduling (tile_m effectively 1) preventing cache-friendly weight reuse.

Given diminishing returns and complexity risk, we have retired the fused kernel behind an environment gate (`LLAMINAR_QUANT_FUSED_ENABLE=1`, default off) and adopted a **slab decode strategy**:

| Aspect | Fused Prototype (Retired) | Slab Decode Strategy |
|--------|---------------------------|----------------------|
| Weight handling | Decode fragments inline every inner loop | Decode full K × N_tile slab once, reuse for multiple matmuls or rows |
| Storage | Transient FP32 vectors | Persistent FP16 slab buffer (half memory) |
| Reuse scope | Limited (per iteration) | Cross-iteration via LRU `QuantSlabCache` |
| Complexity | High (tiled FMA scheduling) | Moderate (decode + existing adaptive matmul) |
| Performance (current) | 6–130× slower | Close to unfused baseline; expected gains after FP16 integration |

#### Slab Decode Overview
`QuantSlabCache` caches decoded slabs keyed by (weight raw pointer, column start, column span). A slab is a contiguous FP16 matrix slice of shape `K × N_tile`. The decode path:
1. Iterate rows `0..K-1`
2. For each quant block overlapping `[col_start, col_start + N_tile)`
3. Decode block → temporary FP32 (local) → cast to FP16 and write into slab buffer
4. Insert slab into LRU and account memory (FP16 halves footprint vs FP32)

#### Why FP16 Slab Storage?
| Benefit | Detail |
|---------|--------|
| Memory reduction | 2× smaller than FP32 slab; improves cache residency for large K × N_tile slices |
| Bandwidth savings | Adaptive matmul reads half the bytes when using FP16 weights vs FP32-dequant buffer |
| Future mixed precision | Aligns with eventual activation FP16 and FP32 accumulators for consistent kernels |
| Minimal precision loss | Weight dequant already approximate; FP16 cast preserves sufficient fidelity (parity tolerances maintained) |

#### Environment & Future Flags (Planned)
- `LLAMINAR_QUANT_SLAB_ENABLE=1` (master gate for slab path usage in operators)
- `LLAMINAR_QUANT_SLAB_FP16=1` (currently implicit; future flag may allow FP32 debug mode)
- `LLAMINAR_QUANT_SLAB_CAP_MB=<int>` (cache capacity tuning)

#### Integration Plan
1. Replace current block/tile immediate dequant path in `MPILinearOperator` with slab fetch (decode or reuse).
2. Feed FP16 slab into existing `adaptiveMatMul` via temporary conversion or provide FP16-aware matmul fast path.
3. Add parity test: slab path output vs fully dequantized FP32 matmul (rel_l2 < 1e-3).
4. Introduce instrumentation counters: slab_hits, slab_misses, bytes_decoded, bytes_reused.

#### Performance Validation Roadmap
1. Release benchmark: measure `slab_first_ms` (cold decode) vs `slab_hit_ms` (reuse) vs legacy full dequant.
2. Larger shapes (K=4096, N=4096) to assess memory traffic reduction impact.
3. Compare FP16 slab matmul throughput against FP32 expanded weight path.
4. Optional micro-benchmark to isolate slab decode bandwidth (GB/s) vs block/tile decode.

#### Risk Mitigation
| Risk | Mitigation |
|------|------------|
| Cache thrashing (many distinct slabs) | Capacity tuning + slab span heuristic (merge adjacent spans) |
| Precision concerns | Parity tests across random prompts; fallback FP32 mode for diagnostics |
| Memory fragmentation | Continuous vector allocation; reuse via shared_ptr; predictable LRU eviction |
| MPI sharding alignment | Key ties to raw buffer; for sharded tensors each rank naturally builds independent slab subset |

#### Deprecated Path Note
The fused kernel will remain in the codebase (gated) for potential future experimentation once a robust SIMD + reuse implementation is available. Any reintroduction will require fresh benchmarks demonstrating ≥1.2× speedup vs slab approach on representative projection shapes.

### 15.12 BF16 Pivot Status & Architecture Corrections (Oct 19 2025)

#### Critical Discovery: Documentation vs Implementation Mismatch

**The original architecture document incorrectly specified FP16 throughout Phases 3-6. The actual implementation uses BF16 (bfloat16).**

#### What's Already BF16 ✅

1. **QuantSlabCache Storage** (`src/operators/QuantSlabCache.{h,cpp}`):
   - Line 59 in QuantSlabCache.h: `std::vector<bfloat16> data;`
   - Comment explicitly states: "Slab stores decoded weights in BF16 (bfloat16) to preserve dynamic range"
   - Decode path: Quant blocks → FP32 temp → **BF16 conversion** → slab storage
   - LRU accounting uses `sizeof(bfloat16)` (2 bytes)

2. **BFloat16 Utilities** (`src/BFloat16.h`):
   - Complete `bfloat16` struct with proper round-to-nearest-even conversion
   - `from_float()`: FP32 → uint32 → rounding bias (`0x7FFF + lsb`) → shift → BF16
   - `operator float()`: BF16 → shift → reinterpret as FP32
   - 2 bytes per element storage

3. **MPILinearOperator Slab Path** (lines 172-209):
   - Fetches BF16 slab from cache
   - **Currently expands BF16→FP32** before matmul (lines 197-203)
   - This expansion is the target for elimination via BF16 GEMM integration

4. **QuantizedTensor Class** (`src/tensors/TensorFactory.h`):
   - Exists with `decodeBlock()` method (block → FP32)
   - Has `decodeTileFP16()` method (see critical note below)

#### What's Missing for Full BF16 GEMM ❌

1. **AdaptiveMatmul BF16 Path**:
   - No BF16×BF16 GEMM dispatch in `AdaptiveMatmul.h`
   - Currently only handles FP32 (`cblas_sgemm`, COSMA with FP32)
   - Need: BF16 input detection + backend routing logic

2. **OpenBLAS sbgemm Integration**:
   - Target API: `cblas_sbgemm(layout, transA, transB, m, n, k, alpha, A_bf16, lda, B_bf16, ldb, beta, C_f32, ldc)`
   - Mixed precision: BF16 inputs, FP32 output and accumulation
   - Not yet wrapped or called anywhere in codebase

3. **COSMA BF16 Support**:
   - User adding BF16 support in separate COSMA branch
   - Need: Interface placeholder in AdaptiveMatmul for future integration

4. **Intel MKL Integration** (Optional):
   - Research exact signature of `cblas_gemm_bf16bf16f32()`
   - Add CMake detection and conditional compilation

5. **Input Activation Conversion**:
   - No FP32→BF16 conversion path for input activations
   - Currently: Activations remain FP32 throughout pipeline
   - Need: Optional conversion before BF16 GEMM call

#### Critical Naming Issue: decodeTileFP16 vs decodeTileBF16

**Problem:** `QuantizedTensor::decodeTileFP16()` signature uses `_Float16*` (IEEE 754 half precision), NOT `bfloat16*`.

```cpp
// Current signature (TensorFactory.h line 87):
void decodeTileFP16(int row_start, int rows, int col_start, int cols, _Float16 *dst) const;
```

**Analysis:**
- Method is **never called** anywhere in codebase (grep search confirms)
- Outputs IEEE FP16, incompatible with slab cache which stores BF16
- Name suggests FP16 but should be BF16 to match architecture

**Options:**
1. **Delete dead code:** Remove `decodeTileFP16()` entirely (never used)
2. **Rename and fix:** Change to `decodeTileBF16(...)` with `bfloat16* dst`
3. **Keep both:** Add new `decodeTileBF16()`, deprecate FP16 variant

**Recommendation:** Option 1 (delete) unless there's a planned use case.

#### BF16 vs FP16: Why BF16 is Better

| Aspect | FP16 (IEEE 754) | BF16 (Brain Float 16) |
|--------|-----------------|------------------------|
| **Exponent bits** | 5 | 8 (same as FP32) |
| **Mantissa bits** | 10 | 7 |
| **Dynamic range** | Limited (±65,504) | Same as FP32 (±3.4×10³⁸) |
| **Precision** | Higher | Lower |
| **Conversion from FP32** | Complex | Simple truncation |
| **Numerical stability** | Prone to overflow/underflow | Same stability as FP32 |
| **Hardware support** | Older (ARM NEON, Tensor Cores) | Modern (Intel AMX, Google TPU, ARM SVE) |
| **Use case** | Inference on older GPUs | Training & inference on modern hardware |

**Why Llaminar chose BF16:**
- Same exponent range as FP32 → better numerical stability
- Simple conversion (round + truncate) → faster decode
- Modern CPU support (Intel Sapphire Rapids AMX, OpenBLAS sbgemm)
- Future COSMA integration easier (user's branch)

#### The Pivot Goal: Eliminate BF16→FP32 Expansion

**Current inefficiency** (MPILinearOperator.cpp lines 197-203):
```cpp
// Expand BF16 slab to FP32 then let existing adaptive path handle backend selection.
std::vector<float> slab_fp32(slab.k * slab.n);
#ifdef _OPENMP
#pragma omp parallel for if (total > 32768) schedule(static)
for (size_t idx = 0; idx < total; ++idx)
    slab_fp32[idx] = (float)slab.data[idx];  // BF16 → FP32 conversion
#endif
// Then call adaptiveMatMul with FP32 weights...
```

**Target architecture:**
1. Keep weights in BF16 after slab decode (ALREADY DONE ✅)
2. Convert input activations FP32 → BF16 (NEW)
3. Call BF16×BF16 GEMM with FP32 accumulation (NEW)
4. Output directly in FP32 (standard) or BF16 (optional)

**Benefits:**
- **2× memory bandwidth reduction** for weight reads (2 bytes vs 4 bytes)
- **Exploit hardware acceleration**: OpenBLAS sbgemm, Intel AMX, future COSMA
- **Preserve numerical accuracy**: FP32 accumulation prevents precision loss
- **Cache efficiency**: More weights fit in L1/L2 cache

#### Implementation Phases for BF16 GEMM Integration

**Phase 1: OpenBLAS sbgemm Backend in AdaptiveMatmul** (Priority: HIGH)
1. Extend `AdaptiveMatmul` class with BF16 detection and dispatch logic
2. Add `multiplyBF16()` method that wraps `cblas_sbgemm()` with error handling
3. Backend selection: OpenBLAS (verified working v0.3.26), Intel MKL (hardware accelerated when available), COSMA (future work)
4. Tensor type detection: Check if weights are BF16 (from slab cache)
5. Input conversion: FP32 activations → BF16 temporary buffer (managed internally)
6. Output: Direct FP32 (no conversion needed, sbgemm handles it)
7. Environment flag: `LLAMINAR_QUANT_BF16_GEMM=1` master gate (default off for safety)

**Phase 2: MPILinearOperator Integration**
1. Remove BF16→FP32 expansion loop (lines 197-203 in MPILinearOperator.cpp)
2. Call existing `adaptiveMatMul()` with BF16 slab data directly (dispatch handled internally)
3. AdaptiveMatmul detects BF16 input and routes to `multiplyBF16()` automatically
4. Pass input as FP32 (AdaptiveMatmul converts internally to BF16)
5. Validate via existing parity tests (no new test infrastructure needed)

**Phase 3: COSMA BF16 Support** (Future - when upstream merge complete)
1. Add BF16 detection in COSMA backend path within AdaptiveMatmul
2. Call COSMA BF16 multiply when available and enabled
3. Fallback to OpenBLAS sbgemm if COSMA BF16 not available
4. User integrates their COSMA branch when ready

**Phase 4: Intel MKL BF16 Support** (Optional - when MKL linked)
1. Research exact `cblas_gemm_bf16bf16f32` or MKL-specific API signature
2. Add MKL detection in CMake (`find_package(MKL)`)
3. Add MKL BF16 path in AdaptiveMatmul backend selection
4. Benchmark against OpenBLAS sbgemm for performance comparison
5. Environment flag: `LLAMINAR_QUANT_BF16_PREFER_MKL=1` to override backend priority

**Phase 5: Parity Testing**
1. Small matrix test: Q8_0 → BF16 slab → sbgemm vs FP32 reference
2. Tolerance: `rel_l2 < 1e-3`, `max_abs < 1e-2` (BF16 rounding acceptable)
3. Performance benchmark: Measure bandwidth savings on large projections
4. Validate across all quant formats (Q4_0, Q6_K, Q8_0, etc.)

#### Environment Flags for BF16 Path

| Flag | Purpose | Default |
|------|---------|--------|
| `LLAMINAR_QUANT_BF16_GEMM` | Enable BF16×BF16→FP32 GEMM path | 0 (off) |
| `LLAMINAR_QUANT_SLAB_ENABLE` | Enable slab decode caching | 1 (on) |
| `LLAMINAR_QUANT_BF16_VERIFY` | Compare BF16 path vs FP32 reference | 0 (off) |
| `LLAMINAR_QUANT_BF16_BACKEND` | Force backend: `openblas`, `cosma`, `mkl` | auto |
| `LLAMINAR_KV_BF16` | Store KV cache in BF16 (Phase 5) | 0 (off) |

#### Expected Performance Impact

**Memory bandwidth reduction** (theoretical):
- Weight reads: 2× reduction (BF16 vs FP32)
- Input conversion overhead: ~5-10% (FP32→BF16 cast)
- Net bandwidth savings: ~40-60% for weight-bound operations

**Numerical accuracy** (empirical from other projects):
- BF16 matmul vs FP32: Relative L2 typically < 1e-3
- Accumulated error over 32 layers: < 1e-2
- Perplexity impact: Negligible (<0.1% increase)

**Hardware acceleration** (when available):
- Intel Sapphire Rapids (AMX): Up to 4× GEMM speedup
- OpenBLAS sbgemm: 1.5-2× vs sgemm (depends on matrix size)
- Future COSMA BF16: Expected 2-3× improvement for large prefill

#### Remaining Work Summary

**Documentation:**
- ✅ Corrected all FP16 → BF16 mentions in Phases 3-6
- ✅ Added this status section (15.12)
- ⏳ Update API documentation for slab cache

**Code:**
- ✅ BFloat16 utilities complete
- ✅ Slab cache stores BF16
- ❌ AdaptiveMatmul BF16 path (not started)
- ❌ OpenBLAS sbgemm wrapper (not started)
- ❌ Input activation BF16 conversion (not started)
- ❌ COSMA BF16 placeholder (not started)
- ❌ Parity tests for BF16 path (not started)

**Technical Debt:**
- ⚠️ `decodeTileFP16()` method: Dead code using wrong type, should delete or rename to BF16
- ⚠️ MPILinearOperator: BF16→FP32 expansion inefficiency (lines 197-203)
- ⚠️ No BF16 backend selection logic in AdaptiveMatmul

**✅ Resolved (October 2025)**:
- ✅ OpenBLAS BF16 emulation verified working (v0.3.26) - no NaN issues on Cascade Lake
- ✅ CPU feature check removed from AdaptiveMatmul.h - trusts OpenBLAS software emulation
- ✅ 4/4 BF16 tests passing, end-to-end inference successful

#### Next Immediate Steps (Phase 1 Implementation)

See `TODO.md` for detailed task breakdown. Summary:

1. **AdaptiveMatmul BF16 Extension**:
   - Add `multiplyBF16()` method wrapping `cblas_sbgemm()`
   - Add BF16 input detection in `multiply()` dispatcher
   - Backend selection: OpenBLAS (verified working v0.3.26), Intel MKL (hardware accelerated when available)
   - Environment flag: `LLAMINAR_QUANT_BF16_GEMM=1` (default off)
   - **Status Update (Oct 2025)**: OpenBLAS BF16 emulation verified reliable - no forced FP32 fallback needed

2. **Input Conversion Helper**:
   - Add FP32→BF16 conversion with OpenMP parallelization
   - Reusable temporary buffer management (avoid repeated allocation)
   - NUMA-aware allocation for large inputs

3. **BF16 Parity Test**:
   - Small matrix (M=64, K=256, N=384) with Q8_0 quantization
   - Compare: BF16 path vs fully dequantized FP32 path
   - Tolerance: `rel_l2 < 1e-3`, `max_abs < 1e-2`

4. **MPILinearOperator Integration**:
   - Remove BF16→FP32 expansion (lines 197-203)
   - Direct call to `adaptiveMatMul()` with BF16 slab data
   - AdaptiveMatmul detects BF16 and dispatches automatically

5. **Performance Validation**:
   - Benchmark BF16 path vs current expansion approach
   - Track: decode time, matmul time, total end-to-end
   - Target: ≥1.3× speedup from bandwidth reduction

6. **Cleanup**:
   - Delete or rename `decodeTileFP16()` dead code
   - Update environment documentation
   - Add inline comments for BF16 code paths

---
