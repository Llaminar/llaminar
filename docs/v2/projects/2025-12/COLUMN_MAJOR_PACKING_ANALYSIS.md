# Column-Major Weight Packing: Analysis and Benefits

## Executive Summary

**UPDATE (Benchmark Results)**: For FP32 Wo GEMV (decode, M=1), **column-major packing is not a consistent win**.

- The best-performing column-major implementation in our microbench is now the **compiler-guided** variant (“ColMaj CG”), which closes much of the gap versus row-major.
- The full single-thread size sweep shows **small sizes can slightly favor column-major**, but **larger sizes tend to favor row-major**.
- At `d_model=3584` (Qwen 7B class), row-major AVX-512 MR=4 is still faster in the sweep: **8.4 vs 8.0 GFLOP/s** (~0.96×).

---

## Benchmark Results (d_model=3584, single-threaded)

| Variant | GFLOP/s | vs. RowMaj AVX |
|---------|---------|----------------|
| **Row-major AVX512 (MR=4)** | **8.4** | 1.00× |
| Column-major compiler-guided (ColMaj CG) | 8.0 | 0.96× |
| Column-major naive | 8.0 | 0.95× |
| Column-major AVX512 (NR=64) | 4.6 | 0.55× |
| Column-major tuned (aligned + K unroll4) | 5.1 | 0.61× |
| Column-major AVX512+prefetch | 4.3 | 0.51× |
| Row-major naive | 1.9 | 0.22× |

**Key Insight**: For this kernel shape, scalar broadcast (`set1_ps`) and reduced ILP often outweigh the theoretical cache-streaming advantage. The compiler-guided column-major version can narrow the gap substantially.

---

## Size Sweep (single-threaded)

Measured with `OMP_NUM_THREADS=1` via `v2_perf_wo_transpose`.

| d_model | RowMaj AVX (GFLOP/s) | ColMaj AVX (GFLOP/s) | ColMaj Tun (GFLOP/s) | ColMaj CG (GFLOP/s) | ColMaj Nve (GFLOP/s) | Speedup (Best ColMaj / RowMaj) | Model |
|--------:|----------------------:|----------------------:|----------------------:|---------------------:|----------------------:|------------------------------:|:------|
| 896  | 12.7 | 13.3 | 13.0 | 13.5 | 13.6 | 1.07× | Qwen 0.5B |
| 1536 | 13.1 | 13.3 | 13.2 | 13.4 | 12.8 | 1.02× | Qwen 1.5B |
| 2048 | 13.0 | 12.7 | 12.9 | 13.4 | 13.4 | 1.03× | Qwen 3B |
| 3584 | 8.4  | 4.6  | 5.1  | 8.0  | 8.0  | 0.96× | Qwen 7B |
| 4096 | 6.7  | 3.5  | 3.8  | 7.1  | 7.1  | 1.07× | Llama 8B |
| 5120 | 6.3  | 4.0  | 4.0  | 6.5  | 6.5  | 1.03× | Qwen 14B |
| 6144 | 6.2  | 3.6  | 3.7  | 6.2  | 6.2  | 1.00× | Mistral Lg |
| 6656 | 6.5  | 3.4  | 3.5  | 6.0  | 6.0  | 0.93× | Llama 30B |
| 8192 | 6.0  | 3.4  | 3.3  | 5.5  | 5.6  | 0.92× | Qwen 72B |

---

## Current Weight Layout Analysis

### 1. QuantisedGemmJit (Main GEMM Kernel)

**Current Layout**: VNNI-packed `[N/64][K/4][64][4]`

```
Weights B: [N, K] → Packed: [N/64][K/4][64][4]
                            ↑      ↑    ↑   ↑
                            |      |    |   └─ 4 K values for vpdpbusd
                            |      |    └───── 64 N columns per block
                            |      └────────── K dimension in 4-element groups
                            └───────────────── N dimension in 64-column tiles
```

**Access Pattern** (M=1 GEMV):
```cpp
for (n = 0; n < N; n += 64) {           // Outer: N columns
    acc[0:64] = 0;
    for (k = 0; k < K; k += 32) {        // Inner: K reduction
        a_vals = load A[k:k+32]          // 36 bytes
        for (i = 0; i < 8; i++) {        // 8 iterations × 4 values = 32
            b_vals = load B_packed[n/64][k/4+i][64][4]  // 256 bytes (64×4)
            acc += vpdpbusd(a_vals[i*4:i*4+4], b_vals)
        }
    }
    store C[n:n+64]
}
```

**Locality Analysis**:
- ✅ Good K-dimension locality (B loaded in 256-byte sequential chunks)
- ✅ 64 output columns per N-tile amortizes A loading
- ⚠️ Each N-tile start jumps to a new section of B

### 2. Fused Attention Wo Projection (GEMV)

**Current Layout**: Row-major FP32 `[d_model, d_model]`

```
Wo: [rows=d_model, cols=d_model] row-major
    output[i] = sum_j(context[j] × Wo[i, j])
```

**Access Pattern**:
```cpp
for (i = 0; i < d_model; i += 4) {      // Outer: 4 output rows at a time (MR=4)
    acc[0:4] = 0;
    for (k = 0; k < d_model; k += 16) { // Inner: K reduction (16 floats per ZMM)
        ctx = load context[k:k+16]       // 64 bytes
        wo0 = load Wo[i+0, k:k+16]       // 64 bytes, row stride = d_model*4
        wo1 = load Wo[i+1, k:k+16]       // 64 bytes
        wo2 = load Wo[i+2, k:k+16]       // 64 bytes
        wo3 = load Wo[i+3, k:k+16]       // 64 bytes
        acc[0:4] += ctx × [wo0,wo1,wo2,wo3]
    }
    store output[i:i+4]
}
```

**Locality Analysis**:
- ✅ Context vector reused across 4 output rows
- ⚠️ Wo rows are non-contiguous (stride = d_model × 4 bytes)
- ⚠️ Each K iteration touches 4 cache lines from different rows

---

## Column-Major Benefits for GEMV

### The Core Insight

For GEMV (M=1), we compute: `output[i] = sum_j(A[j] × B[i,j])`

**Row-major B**: Iterating over j (reduction) means strided access through rows
**Column-major B**: Iterating over j means sequential access through columns

### Memory Access Comparison (d_model=3584)

| Metric | Row-Major (Current) | Column-Major |
|--------|---------------------|--------------|
| **Wo size** | 3584 × 3584 × 4B = 49MB | Same |
| **Access per K-iter** | 4 rows × 64B = 256B (strided) | 4 columns × 64B = 256B (sequential) |
| **Cache lines touched** | 4 cache lines from different 14KB-apart locations | 4 contiguous cache lines |
| **Prefetch efficiency** | Poor (can't predict row jumps) | Excellent (sequential stream) |
| **HW prefetcher help** | Limited | Full benefit |

### Theoretical Expectation (Disproven for Wo FP32 GEMV)

For memory-bound GEMV on modern CPUs:
- Row-major: ~8-15 GFLOP/s (limited by TLB misses, prefetch failures)
- Column-major: ~25-40 GFLOP/s (streaming at memory bandwidth)

**Estimated improvement: 2-3×** for Wo projection in attention.

**UPDATE**: Empirically, this expectation does not hold for this workload on this platform; row-major MR=4 remains highly competitive and often faster.

---

## Column-Major Benefits for Other Components

### 1. Q/K/V Projections (GEMM with M>1)

For prefill with M=seq_len tokens:

**Current**: Row-major works well because we're doing GEMM not GEMV:
```
C[M,N] = A[M,K] × B[K,N]
```

With M>1, we can tile over both M and N, achieving good reuse regardless of B layout.

**Recommendation**: Keep row-major for prefill GEMM. Column-major doesn't help.

### 2. FFN Projections (Gate, Up, Down)

Same as Q/K/V: These are full GEMM operations during prefill, GEMV during decode.

| Operation | Prefill (M=seq_len) | Decode (M=1) |
|-----------|---------------------|--------------|
| **FFN Gate** | GEMM (row-major OK) | GEMV (column-major helps) |
| **FFN Up** | GEMM (row-major OK) | GEMV (column-major helps) |
| **FFN Down** | GEMM (row-major OK) | GEMV (column-major helps) |

**Recommendation**: Dual layout or decode-specific column-major packing.

### 3. LM Head (Logit Projection)

The final `hidden → vocab` projection:
- Shape: [hidden_dim × vocab_size] (~4K × 128K for Qwen 7B)
- Always GEMV during decode (M=1)
- **Column-major would help significantly** (vocab is huge, stride is massive)

### 4. Embedding Lookup

Not applicable (no matrix multiply, just gather).

---

## Implementation Options

### Option A: Dual Packing (Prefill + Decode)

Store weights in both layouts:
```cpp
struct DualPackedWeights {
    QuantisedPackedWeights row_major;    // For prefill GEMM
    QuantisedPackedWeights col_major;    // For decode GEMV
};
```

**Pros**: Optimal for both modes
**Cons**: 2× memory for weights

### Option B: Runtime Transpose

Transpose weights on-the-fly for decode:
```cpp
// At decode kernel entry:
if (M == 1 && weights_are_row_major) {
    transpose_to_column_major_cache(weights);
}
```

**Pros**: No memory overhead
**Cons**: Transpose overhead (can be amortized over many decode steps)

### Option C: Column-Major Only

Store all weights column-major, adjust GEMM kernel:
```cpp
// GEMM with column-major B requires different tiling
C[M,N] = A[M,K] × B^T[N,K]  // Treat B as transposed
```

**Pros**: Single layout, good for decode
**Cons**: Suboptimal for prefill (but prefill is less latency-sensitive)

### Option D: Selective Packing (DEPRECATED)

**UPDATE**: Benchmarking showed column-major does NOT benefit Wo projection. Row-major AVX-512 with MR=4 is already optimal.

---

## ~~Recommended Implementation~~ (DEPRECATED - Benchmarks Disprove Theory)

### Benchmark Results (d_model=3584, single-threaded)

| Variant | GFLOP/s | vs. Best |
|---------|---------|----------|
| **Row-major AVX512 (MR=4)** | **9.31** | 1.00× |
| Column-major naive | 7.96 | 0.86× |
| Column-major AVX512 (NR=64) | 6.51 | 0.70× |
| Column-major AVX512+prefetch | 4.40 | 0.47× |
| Interleaved layout | 3.22 | 0.35× |
| Row-major naive | 1.87 | 0.20× |

### Why Row-Major AVX-512 Wins

1. **Vector context loading**: `_mm512_loadu_ps(context + k)` loads 16 context elements in one instruction
2. **Scalar broadcast is expensive**: Column-major requires `_mm512_set1_ps(context[k])` per K element
3. **Instruction-Level Parallelism**: 4 independent FMA accumulator streams execute in parallel
4. **Modern prefetchers work**: Intel CPUs track 4+ independent streams efficiently

### ~~Phase 1: Wo Projection~~ (NO BENEFIT)

Benchmarking showed **no benefit** from column-major Wo projection. Keep current row-major layout.

### Phase 2: FFN Projections (INVESTIGATE)

FFN projections use INT8 quantized weights with VNNI. May have different characteristics - needs separate benchmarking.

### Phase 3: LM Head (INVESTIGATE)

LM head is a large GEMV during decode. May benefit from different strategies.

---

## Updated Trade-off Summary

| Component | Current (Row-Major) | Column-Major Benefit | Recommendation |
|-----------|--------------------|--------------------|----------------|
| **Wo Projection** | **RowMaj MR=4 is strong** | Mixed by size (see sweep) | **No change (for now)** |
| **FFN Gate/Up/Down** | INT8 VNNI | Unknown | Investigate |
| **LM Head** | FP32 GEMV | Unknown | Investigate |
| **Q/K/V Projections** | Prefill OK | No benefit | No |
| **Attention Q·K** | N/A (not matrix) | N/A | No |

---

## Conclusion

**Column-major packing is not a consistent win for FP32 Wo GEMV.** The theoretical advantage of sequential memory access is often outweighed by:

1. **Scalar broadcast overhead** in the inner loop
2. **Loss of ILP** from single accumulator vs. 4 independent streams
3. **Efficient vector loads** of context in row-major pattern

**Lessons Learned**:
- Always benchmark before implementing - theory predicted 2-3×, reality is mixed and often < 1× at larger `d_model`
- Modern CPU prefetchers handle multiple strided streams well
- Instruction-level parallelism often matters more than memory access patterns
- The overhead of scalar operations (`set1_ps`) should not be underestimated

**Next steps**: Focus optimization efforts on:
1. INT8 GEMV for FFN projections (different memory/compute balance)
2. Batch size > 1 decode (multiple tokens amortize overhead)
3. KV cache optimization (likely larger impact than Wo)
