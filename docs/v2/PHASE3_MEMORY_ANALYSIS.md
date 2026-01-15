/**
 * @file PHASE3_MEMORY_ANALYSIS.md
 * @brief Memory analysis for slab-based FP16 GEMM
 *
 * Detailed calculations comparing full FP16 conversion vs slab-based approach.
 */

# Phase 3: Slab-Based FP16 GEMM Memory Analysis

## 1. Target Use Case: Qwen2.5-7B FFN Layers

### Model Dimensions

```
Qwen2.5-7B Parameters:
  - hidden_size: 3584
  - intermediate_size: 18944 (5.3× hidden)
  - num_hidden_layers: 28

FFN Layers (per transformer block):
  - ffn_gate: [18944 × 3584] - Gate projection (up)
  - ffn_up:   [18944 × 3584] - Up projection  
  - ffn_down: [3584 × 18944] - Down projection
```

### GEMM Dimensions for FFN

```
For ffn_down (largest memory pressure):
  Input:  [M × K] where M = seq_len, K = 18944
  Weight: [K × N] = [18944 × 3584] (transposed)
  Output: [M × N] = [M × 3584]

Typical M values:
  - Prefill (prompt processing): M = 128 to 2048
  - Decode (token generation): M = 1 to 8
```

## 2. Full FP16 Conversion Memory

### Formula

```
Full FP16 memory = A_fp16 + B_fp16 + C_fp16
                 = M×K×2 + K×N×2 + M×N×2 bytes

Where:
  - M = batch size (rows in activation)
  - K = input features  
  - N = output features
```

### Calculations for Qwen2.5-7B FFN Down (K=18944, N=3584)

| M (batch) | A_fp16 (M×K×2) | B_fp16 (K×N×2) | C_fp16 (M×N×2) | **Total** |
|-----------|---------------|----------------|----------------|-----------|
| 1         | 37 KB         | 136 MB         | 7 KB           | **136 MB** |
| 8         | 296 KB        | 136 MB         | 56 KB          | **136 MB** |
| 64        | 2.4 MB        | 136 MB         | 448 KB         | **139 MB** |
| 128       | 4.7 MB        | 136 MB         | 896 KB         | **142 MB** |
| 256       | 9.5 MB        | 136 MB         | 1.8 MB         | **147 MB** |
| 512       | 19 MB         | 136 MB         | 3.6 MB         | **159 MB** |
| 1024      | 38 MB         | 136 MB         | 7.2 MB         | **181 MB** |
| 2048      | 76 MB         | 136 MB         | 14.4 MB        | **226 MB** |

**Key Insight**: B_fp16 (weight) dominates at ~136MB regardless of batch size.

### For FFN Gate/Up (K=3584, N=18944)

| M (batch) | A_fp16 (M×K×2) | B_fp16 (K×N×2) | C_fp16 (M×N×2) | **Total** |
|-----------|---------------|----------------|----------------|-----------|
| 1         | 7 KB          | 136 MB         | 37 KB          | **136 MB** |
| 128       | 896 KB        | 136 MB         | 4.7 MB         | **142 MB** |
| 512       | 3.6 MB        | 136 MB         | 19 MB          | **159 MB** |

**Same B_fp16 size** - weights dominate in both directions.

## 3. Slab-Based Memory with 64MB Budget

### Optimal Slab Configuration

Using `SlabGemmConfig::fromBudget(64MB, ...)`:

```cpp
// For FFN down: M=512, K=18944, N=3584
SlabGemmConfig config = {
    .slab_m = 256,   // Process 256 rows at a time
    .slab_k = 512,   // 512 inner dimension
    .slab_n = 3584   // Full output width (fits!)
};

Memory calculation:
  slab_a: 256 × 512 × 2 = 256 KB
  slab_b: 512 × 3584 × 2 = 3.6 MB
  slab_c: 256 × 3584 × 2 = 1.8 MB
  Total: 5.7 MB (with alignment: ~6 MB)
```

### Comparison

| Configuration | Memory | Reduction |
|--------------|--------|-----------|
| Full FP16 (M=512) | 159 MB | 1× (baseline) |
| Slab (64MB budget) | 6 MB | **26.5×** |
| Slab (32MB budget) | ~3 MB | **53×** |

### Iteration Count

```
For FFN down (M=512, K=18944, N=3584) with slab_m=256, slab_k=512, slab_n=3584:

  m_iters = ceil(512/256) = 2
  k_iters = ceil(18944/512) = 37
  n_iters = ceil(3584/3584) = 1

  Total iterations = 2 × 37 × 1 = 74 slab GEMMs

For FFN gate/up (M=512, K=3584, N=18944) with slab_m=256, slab_k=512, slab_n=4096:

  m_iters = ceil(512/256) = 2
  k_iters = ceil(3584/512) = 7
  n_iters = ceil(18944/4096) = 5

  Total iterations = 2 × 7 × 5 = 70 slab GEMMs
```

## 4. Performance Impact Analysis

### hipBLAS hgemm Overhead Model

```
T_gemm(M, N, K) ≈ T_launch + M×N×K / GFLOPS

Where:
  - T_launch ≈ 5-10 μs per kernel (PCIe/HBM latency)
  - GFLOPS ≈ 50-100 TFLOPS for MI50 FP16 MFMA (theoretical: 106 TFLOPS)
  - Actual GFLOPS depends on tile utilization
```

### Full FP16 Timing (M=512, K=18944, N=3584)

```
Compute: 512 × 3584 × 18944 × 2 / 100e12 = 0.69 ms
Conversion A: 512 × 18944 × 2 bytes / 1000 GB/s = 0.02 ms
Conversion B: 18944 × 3584 × 2 bytes / 1000 GB/s = 0.14 ms
Conversion C: 512 × 3584 × 2 bytes / 1000 GB/s = 0.004 ms
Total: ~0.85 ms
```

### Slab FP16 Timing (74 iterations)

```
Per slab (256×3584×512):
  Compute: 256 × 3584 × 512 × 2 / 100e12 = 0.009 ms
  Launch: 0.01 ms
  Conversion A: 256 × 512 × 2 / 1000 GB/s = 0.0003 ms
  Conversion B: 512 × 3584 × 2 / 1000 GB/s = 0.004 ms
  Accumulate C: 256 × 3584 × 2 / 1000 GB/s = 0.002 ms
  Per slab total: ~0.025 ms

Total (74 slabs): 74 × 0.025 = 1.85 ms

Overhead vs full: 1.85 / 0.85 = 2.2× slower
```

### Decode Performance (M=1)

```
For decode, M=1, so slab_m=32 (minimum) covers it.
We only iterate over K dimension.

With slab_m=32, slab_k=512, slab_n=3584:
  k_iters = ceil(18944/512) = 37
  Total iterations = 37

But decode uses INT8 path (M < 128), not FP16, so no impact.
```

## 5. Memory Budget Recommendations

### For Different GPU Configurations

| GPU Memory | Weight Size (7B) | KV Cache | Available Workspace | Recommendation |
|------------|------------------|----------|---------------------|----------------|
| 16 GB (MI50) | ~4 GB | ~2 GB | ~2-4 GB | 64-128 MB budget |
| 32 GB (MI60) | ~4 GB | ~4 GB | ~8-16 GB | 256-512 MB budget |
| 80 GB (MI250) | ~4 GB | ~10 GB | ~40 GB | Full FP16 (no slab) |

### Configuration by Use Case

| Use Case | Batch Size | Recommended Config | Memory | Iterations |
|----------|-----------|-------------------|--------|------------|
| Interactive (low latency) | M=1-8 | INT8 path | N/A | N/A |
| Small prefill | M=64-128 | Full FP16 | ~142 MB | 1 |
| Medium prefill | M=256-512 | Slab (64MB) | ~6 MB | 70-74 |
| Large prefill | M=1024-2048 | Slab (128MB) | ~12 MB | 37 |
| Throughput batch | M=2048+ | Slab (256MB) | ~24 MB | 19 |

## 6. Summary

### Key Findings

1. **B matrix dominates memory**: Weight conversion takes ~136MB for 7B FFN layers
2. **Slab reduces by 26×**: 6MB workspace vs 159MB full conversion
3. **Performance overhead: 2-3×** slower due to kernel launch overhead
4. **Decode unaffected**: M=1 uses INT8 path (threshold M=128)

### Recommendation

- **Default**: Enable slab FP16 when workspace budget < required full FP16
- **Threshold**: Use slab if `K×N×2 > budget × 0.8` (80% of budget)
- **Minimum budget**: 32MB (covers most model sizes with reasonable iteration count)
- **Optimal budget**: 64-128MB (balances memory savings vs performance)

### Trade-off Summary

| Approach | Memory | Performance | Use Case |
|----------|--------|-------------|----------|
| Full FP16 | High (150+ MB) | Fastest | Plenty of VRAM |
| Slab FP16 | Fixed (~6 MB) | 2-3× slower | Memory constrained |
| INT8 CK | Low (~32 MB) | Medium | Decode, small M |
