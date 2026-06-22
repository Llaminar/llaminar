# TurboQuant KV Cache Quantization — Project Plan

**Date**: March 25, 2026
**Status**: Planning
**Target**: CPU-first implementation, GPU follow-up
**Paper**: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)

---

## 1. Overview

TurboQuant is a data-oblivious online vector quantization method from Google Research that achieves near-optimal distortion rates (within ~2.7× of information-theoretic lower bounds). Applied to KV cache quantization, it delivers **quality-neutral** compression at 3.5 bits/element and marginal degradation at 2.5 bits — dramatically better than scalar quantization at comparable bit-widths.

**User-facing modes**:
| Mode | Bits/element | CLI flag | Description |
|------|-------------|----------|-------------|
| **TurboQuant 4-bit** | 4 | `--kv-precision tq4` | Default. Quality-neutral with full precision. |
| **TurboQuant 3-bit** | 3 | `--kv-precision tq3` | Aggressive compression, minor quality loss. |

**Scope**: CPU-first. Prove correctness and quality via parity tests, then port to GPU.

---

## 2. Paper Summary

### 2.1 Algorithm — TurboQuant_mse (MSE-Optimized)

The core algorithm operates per-vector (each KV cache entry is one head-dim vector):

1. **Pre-compute** a random orthogonal rotation matrix Π ∈ ℝ^(d×d) via QR decomposition of a random Gaussian matrix. One matrix per model, deterministic given a seed.
2. **Pre-compute** Lloyd-Max optimal codebook centroids for Beta(d/2-1, d/2-1) distribution (converges to N(0, 1/√d) for large d). For b-bit quantization, this is 2^b centroids.
3. **Quantize**: Normalize x to unit sphere, rotate y ← Π·(x/‖x‖), then for each coordinate j store idx_j = argmin_k |y_j - c_k| (b-bit index per element).
4. **Dequantize**: Look up centroids from indices, rotate back: x̃ ← ‖x‖ · Π^T · ỹ.

**MSE distortion**: ≤ (√3·π/2) · 1/4^b for unit-norm vectors.

| Bits | MSE Upper Bound | Centroids |
|------|----------------|-----------|
| 2 | 0.117 | 4 values |
| 3 | 0.030 | 8 values |
| 4 | 0.009 | 16 values |

### 2.2 Why It Works

- **Random rotation** decorrelates input dimensions, making each coordinate approximately i.i.d. Gaussian — this is the key insight that enables per-element scalar quantization to approach vector quantization optimality.
- **Lloyd-Max centroids** are optimal for the resulting Gaussian distribution.
- **Data-oblivious**: No calibration pass needed. Works on any input distribution because the rotation randomizes coordinates.
- **Orthogonal Π**: Π^T = Π^{-1}, so dequantization is just a transpose matmul, no matrix inversion needed.

### 2.3 Outlier Channel Splitting

The paper achieves fractional bit-widths (2.5, 3.5) by splitting channels into outlier and non-outlier sets:
- Identify top-N outlier channels (by activation magnitude variance across a calibration set or online statistics)
- Quantize outlier channels at (b+1) bits, remaining at b bits
- Example: 32 outlier channels at 4-bit + 96 regular at 3-bit = 3.5 effective bits for head_dim=128

**Decision**: We defer outlier splitting to a future phase. The integer bit-width modes (3-bit, 4-bit) are the initial targets.

### 2.4 Paper KV Cache Results (LongBench, Llama-3.1-8B)

| Method | KV bits | LongBench Avg |
|--------|---------|---------------|
| Full cache | 16 | 50.06 |
| KIVI (scalar) | 3 | 48.50 |
| PolarQuant | 3.9 | 49.78 |
| **TurboQuant** | **2.5** | **49.44** |
| **TurboQuant** | **3.5** | **50.06** ← matches full precision |

Needle-in-a-Haystack test: TurboQuant at 4× compression scores 0.997 (identical to full-precision 0.997).

---

## 3. Current Llaminar KV Cache Architecture

### 3.1 Existing Precision Modes

```cpp
enum class KVCachePrecision { AUTO, FP32, FP16, Q8_1, Q16_1 };
// AUTO = Q16_1 on CPU, FP16 on GPU
```

### 3.2 Quantization Block Structures

| Format | Block size | Bytes/block | Bits/element | Key fields |
|--------|-----------|-------------|--------------|------------|
| Q8_1Block | 32 | 36 | 9.0 | FP16 scale + INT16 sum + 32×int8 |
| Q16_1Block | 32 | 72 | 18.0 | FP32 scale + INT32 sum + 32×int16 |
| Q16_1Block_64 | 64 | 136 | 17.0 | FP32 scale + INT32 sum + 64×int16 |
| Q16_1Block_128 | 128 | 264 | 16.5 | FP32 scale + INT32 sum + 128×int16 |

### 3.3 CPU Attention Flow

1. Model computes Q, K, V as FP32 activations
2. K, V are quantized (e.g., to Q16_1) and stored in KV cache
3. At attention time, K, V are dequantized back to FP32
4. `CPUAttentionKernelT<FP32>` performs standard FP32 attention: Q@K^T → softmax → scores@V

TurboQuant slots into steps 2 and 3 — different quantize/dequantize routines, same FP32 attention kernel.

---

## 4. TurboQuant Storage Format

### 4.1 Block Structure

TurboQuant stores b-bit indices per element, plus a single FP32 norm per vector.

```cpp
// 4-bit TurboQuant block (one per head-dim vector)
// Each element gets a 4-bit index into a 16-entry codebook.
// Indices are nibble-packed: 2 indices per byte.
struct TQ4Block {
    float norm;                             // L2 norm of original vector (4 bytes)
    uint8_t indices[BLOCK_SIZE / 2];        // Nibble-packed 4-bit codebook indices
    static constexpr int BITS = 4;
};
// head_dim=64:  4 + 32 = 36 bytes  (4.50 bits/element)
// head_dim=128: 4 + 64 = 68 bytes  (4.25 bits/element)

// 3-bit TurboQuant block (one per head-dim vector)
// 3-bit indices packed into bytes (8 indices per 3 bytes).
struct TQ3Block {
    float norm;                             // L2 norm of original vector (4 bytes)
    uint8_t indices[PACKED_SIZE];           // Bit-packed 3-bit codebook indices
    static constexpr int BITS = 3;
};
// head_dim=64:  4 + 24 = 28 bytes  (3.50 bits/element)
// head_dim=128: 4 + 48 = 52 bytes  (3.25 bits/element)
```

### 4.2 Memory Comparison (per KV vector, head_dim=128)

| Format | Bytes | Bits/elem | Compression vs FP32 |
|--------|-------|-----------|---------------------|
| FP32 | 512 | 32.0 | 1.0× |
| Q16_1_128 | 264 | 16.5 | 1.9× |
| Q8_1 (4 blocks) | 144 | 9.0 | 3.6× |
| **TQ4** | **68** | **4.25** | **7.5×** |
| **TQ3** | **52** | **3.25** | **9.8×** |

### 4.3 Shared Pre-computed Data (Per Model)

| Data | Size (head_dim=128) | Lifetime |
|------|-------------------|----------|
| Rotation matrix Π | 128×128×4 = 64 KB | Model load (once) |
| 4-bit codebook | 16 floats = 64 B | Compile-time constant |
| 3-bit codebook | 8 floats = 32 B | Compile-time constant |

The rotation matrix is generated deterministically from a fixed seed + head_dim, so it's reproducible across runs.

---

## 5. Implementation Plan — CPU First

### Phase 1: Foundation (TurboQuant math + block types)

**Goal**: Standalone quantize/dequantize working with unit tests, no KV cache integration yet.

**Files to create**:

| File | Purpose |
|------|---------|
| `src/v2/tensors/TurboQuantBlock.h` | Block structures (TQ4Block, TQ3Block) |
| `src/v2/kernels/cpu/turboquant/TurboQuantCodebook.h` | Lloyd-Max codebook centroids (constexpr tables) |
| `src/v2/kernels/cpu/turboquant/TurboQuantRotation.h` | Rotation matrix generation + apply/unapply (AVX-512) |
| `src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h` | FP32 → TQ4/TQ3 quantization (normalize + rotate + nearest centroid) |
| `src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h` | TQ4/TQ3 → FP32 dequantization (lookup + inverse rotate + rescale) |
| `tests/v2/unit/Test__TurboQuantCodebook.cpp` | Codebook centroid verification |
| `tests/v2/unit/Test__TurboQuantRoundtrip.cpp` | Quantize → dequantize MSE validation against paper bounds |

**Key implementation details**:

1. **Rotation matrix generation**: QR decomposition of N(0,1) random matrix. Use a fixed seed per model (e.g., hash of head_dim + model name). Use LAPACK `dgeqrf`/`dorgqr` via OpenBLAS (already linked).

2. **AVX-512 rotation kernel**: The rotation is a dense d×d matvec. For head_dim=128, this is 128 dot products of length 128. Each dot product maps to 2 iterations of 64-wide FMA. The full rotation is ~16K FMAs — comfortably under 1μs on AVX-512.

3. **Nearest centroid**: For b=4 (16 centroids), use `_mm512_min_ps` reduction to find nearest codebook entry per element. Process 16 elements per ZMM register, compute |y_j - c_k| for all k simultaneously. For b=3 (8 centroids), even simpler.

4. **Bit-packing**: 4-bit indices are nibble-packed (standard, same as Q4_0). 3-bit indices use the same packing as GGML Q3_K (3 bits per element, 8 elements packed into 3 bytes).

**Unit test acceptance criteria**:
- Roundtrip MSE for random unit-norm vectors ≤ paper's upper bound (0.009 for 4-bit, 0.030 for 3-bit)
- MSE improves with higher bit-width (TQ4 < TQ3)
- Deterministic: same input + same seed → same output

### Phase 2: Tensor Type + KV Cache Integration

**Goal**: TurboQuant as a selectable KV cache precision mode, end-to-end inference working.

**Files to modify**:

| File | Change |
|------|--------|
| `src/v2/tensors/TensorType.h` | Add `TQ4`, `TQ3` to `TensorType` enum |
| `src/v2/execution/config/RuntimeConfig.h` | Add `TQ4`, `TQ3` to `KVCachePrecision` enum |
| `src/v2/config/OrchestrationConfigParser.cpp` | Parse `--kv-precision tq4` / `tq3` |

**New files**:

| File | Purpose |
|------|---------|
| `src/v2/tensors/TQ4Tensor.h/cpp` | TQ4 tensor class (TurboQuantBlock storage, quantize/dequant interface) |
| `src/v2/tensors/TQ3Tensor.h/cpp` | TQ3 tensor class |

**KV cache integration**:

The graph builder (Qwen2Graph) allocates KV cache tensors based on `KVCachePrecision`. Currently it creates Q16_1Tensor or FP16Tensor. We add a branch for TQ4/TQ3:

```
KVCachePrecision::TQ4 → allocate TQ4Tensor (head_dim elements per vector)
KVCachePrecision::TQ3 → allocate TQ3Tensor
```

The KV cache write stage quantizes: FP32 → TQ4/TQ3.
The attention stage dequantizes: TQ4/TQ3 → FP32, then runs standard FP32 attention.

**Files to modify (KV cache flow)**:

| File | Change |
|------|--------|
| `src/v2/pipelines/qwen/Qwen2Graph.cpp` | KV cache tensor allocation for TQ4/TQ3 |
| `src/v2/execution/compute_stages/stages/KVCacheStage.cpp` | Quantize activations to TQ4/TQ3 on write |
| `src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp` | Dequantize TQ4/TQ3 before attention (or wherever K/V are consumed) |

**Rotation matrix lifecycle**: Generated once at model load time, stored in a `TurboQuantContext` object held by the graph orchestrator. Passed to quantize/dequantize stages via stage params.

### Phase 3: Parity Testing

**Goal**: Validate quality against full-precision inference.

**New files**:

| File | Purpose |
|------|---------|
| `tests/v2/integration/Test__TurboQuantInference.cpp` | End-to-end: run Qwen2.5-0.5B with TQ4 KV cache, verify token output matches FP32 baseline |
| `tests/v2/integration/Test__TurboQuantMSE.cpp` | Per-layer KV cache MSE measurement, verify within paper bounds |

**Acceptance criteria**:
- TQ4: Top-1 token prediction matches FP32 for at least 95% of decode steps (greedy, temperature=0)
- TQ3: Top-1 match rate ≥ 85%
- Per-vector MSE within 2× paper upper bound (allowing for non-uniform real activations vs unit-sphere theory)

### Phase 4: AVX-512 Optimization

**Goal**: Optimize the rotation and quantize/dequantize hot paths for production decode throughput.

**Key optimizations**:

1. **Fused rotation + quantize**: Single pass over the vector — rotate each element via FMA, immediately find nearest centroid and pack index. Avoids writing intermediate rotated vector to memory.

2. **Fused dequantize + inverse rotation**: Lookup centroids, immediately multiply by Π^T rows. Single pass output.

3. **Prefetch rotation matrix rows**: The d×d matrix (64KB for head_dim=128) fits in L1. Prefetch next row during current row's computation.

4. **OpenMP worksharing**: Use `OMP_WORKSHARE_REGION` for multi-head quantize/dequantize (each head is independent).

**Performance target**: Quantize + dequantize overhead < 5% of total decode time for Qwen2.5-0.5B.

### Phase 5: GPU Implementation (Future)

**Goal**: Port TurboQuant to CUDA/ROCm for GPU KV cache.

**Approach**: The rotation is a dense matvec — can use cuBLAS `sgemv`. Nearest centroid is a simple kernel. Bit-packing is standard.

**Why GPU benefits more**: GPU KV cache is currently FP16 (16 bits). TQ4 at 4.25 bits is a 3.8× reduction — directly translates to fitting longer contexts in VRAM. On CPU, the improvement over Q8_1 is also significant but CPU memory is more abundant.

---

## 6. Computational Cost Analysis

### 6.1 Per-Vector Operations

| Operation | FLOPs (head_dim=128) | Notes |
|-----------|---------------------|-------|
| Normalize to unit sphere | 128 FMA + 1 sqrt + 128 mul | ‖x‖ then x/‖x‖ |
| Rotation Π·x | 16,384 FMA | 128×128 matvec |
| Nearest centroid (4-bit) | 128 × 16 comparisons | 16 centroids per element |
| Inverse rotation Π^T·ỹ | 16,384 FMA | 128×128 matvec |
| Rescale by ‖x‖ | 128 mul | Output scaling |
| **Total per vector** | **~33K FMA** | |

### 6.2 Per-Token Cost (Qwen2.5-7B, 4 KV heads × 28 layers)

| Operation | FLOPs | vs GEMM budget |
|-----------|-------|---------------|
| Quantize K+V (2 vectors × 4 heads × 28 layers) | 224 × 33K = 7.4M | <0.01% |
| Dequantize K+V for attention | same | <0.01% |
| **Total TurboQuant overhead** | **~15M FLOPs** | **negligible** |

The GEMM operations in a 7B model are on the order of billions of FLOPs per token. TurboQuant overhead is unmeasurable.

### 6.3 Memory Savings (Qwen2.5-7B, 4 KV heads, head_dim=128, 2048 context)

| Format | KV cache size | Savings vs FP32 |
|--------|--------------|-----------------|
| FP32 | 2048 × 4 × 128 × 4B × 28 layers × 2 (K+V) = 231 MB | baseline |
| Q16_1_128 | ~119 MB | 1.9× |
| Q8_1 | ~65 MB | 3.6× |
| **TQ4** | **~31 MB** | **7.5×** |
| **TQ3** | **~24 MB** | **9.8×** |

---

## 7. CLI Interface

```bash
# TurboQuant 4-bit KV cache (default TurboQuant mode)
./llaminar2 --kv-precision tq4 -m model.gguf -p "Hello" -n 50

# TurboQuant 3-bit KV cache
./llaminar2 --kv-precision tq3 -m model.gguf -p "Hello" -n 50

# Existing modes still work
./llaminar2 --kv-precision auto -m model.gguf -p "Hello" -n 50   # Q16_1 on CPU
./llaminar2 --kv-precision q8_1 -m model.gguf -p "Hello" -n 50   # Q8_1
```

---

## 8. File Index

### New files (Phase 1-2)

```
src/v2/tensors/TurboQuantBlock.h                           # Block structures
src/v2/kernels/cpu/turboquant/TurboQuantCodebook.h          # Codebook centroids
src/v2/kernels/cpu/turboquant/TurboQuantRotation.h          # Rotation matrix gen + AVX-512 apply
src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h          # FP32 → TQ4/TQ3
src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h        # TQ4/TQ3 → FP32
src/v2/tensors/TQ4Tensor.h                                  # TQ4 tensor class (header)
src/v2/tensors/TQ4Tensor.cpp                                # TQ4 tensor class (impl)
src/v2/tensors/TQ3Tensor.h                                  # TQ3 tensor class (header)
src/v2/tensors/TQ3Tensor.cpp                                # TQ3 tensor class (impl)
tests/v2/unit/Test__TurboQuantCodebook.cpp                  # Codebook tests
tests/v2/unit/Test__TurboQuantRoundtrip.cpp                 # Roundtrip MSE tests
tests/v2/integration/Test__TurboQuantInference.cpp          # End-to-end quality tests
tests/v2/integration/Test__TurboQuantMSE.cpp                # Per-layer MSE measurement
```

### Modified files (Phase 2)

```
src/v2/tensors/TensorType.h                                 # Add TQ4, TQ3 enums
src/v2/execution/config/RuntimeConfig.h                     # Add KVCachePrecision::TQ4, TQ3
src/v2/config/OrchestrationConfigParser.cpp                 # Parse --kv-precision tq4/tq3
src/v2/pipelines/qwen/Qwen2Graph.cpp                        # KV cache allocation for TQ4/TQ3
src/v2/execution/compute_stages/stages/KVCacheStage.cpp     # Quantize on write
src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp  # Dequantize on read
```

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Rotation matrix overhead at long contexts | Low | Low | Profile; 33K FMA per vector is tiny vs GEMM |
| MSE worse than paper (real activations ≠ unit sphere) | Medium | Medium | Paper handles non-unit via norm storage; test with real model activations |
| 3-bit packing complexity | Low | Low | Reuse GGML Q3_K bit-packing pattern |
| Rotation matrix size grows with head_dim² | Low | Low | 64KB for head_dim=128, fits comfortably in L1 |
| Numerical precision of rotation | Low | Medium | Use double-precision for QR decomposition, store result as FP32 |

---

## 10. Success Criteria

| Metric | TQ4 Target | TQ3 Target |
|--------|-----------|-----------|
| Top-1 token match rate vs FP32 (greedy) | ≥ 95% | ≥ 85% |
| Per-vector MSE (head_dim=128) | ≤ 0.02 | ≤ 0.06 |
| Decode throughput overhead | < 5% | < 5% |
| KV cache memory reduction vs FP32 | 7.5× | 9.8× |
| KV cache memory reduction vs Q16_1 | 3.9× | 5.1× |
