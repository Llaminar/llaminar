# Layer-Level Parallel Region Fusion Plan

**Date:** December 15, 2025  
**Author:** David Sanftenberg  
**Goal:** Reduce OpenMP overhead from 29% to <5% by consolidating parallel regions

## Executive Summary

Perf profiling shows 29% of inference time is spent in OpenMP thread management (`libgomp`). 
Root cause: **482 parallel region launches per token** during decode.

Solution: Consolidate to **~50 parallel region entries per token** via layer-level fusion.

## Current Architecture (Problem)

```
Per-token decode: 482 parallel region entries

Per Layer (×48):
├── RMSNorm (attn)    : 0 regions (disabled for seq_len=1) ✓
├── QKV Fused GEMM    : 4 regions
│   ├── quantize_activations():             1 #pragma omp parallel
│   ├── multiply_with_precomputed Q:        1 #pragma omp parallel
│   ├── multiply_with_precomputed K:        1 #pragma omp parallel  
│   └── multiply_with_precomputed V:        1 #pragma omp parallel
├── RoPE              : 0 regions (disabled for seq_len=1) ✓
├── JIT Attention+Wo  : 0 regions (pure JIT code) ✓
├── Residual Add      : 0 regions (SIMD only) ✓
├── RMSNorm (FFN)     : 0 regions (disabled for seq_len=1) ✓
├── Gate/Up Fused GEMM: 3 regions
│   ├── quantize_activations():             1 #pragma omp parallel
│   ├── multiply_with_precomputed Gate:     1 #pragma omp parallel
│   └── multiply_with_precomputed Up:       1 #pragma omp parallel
├── SwiGLU            : 1 region (always parallel - BUG!)
├── Down GEMM         : 2 regions
│   ├── quantize_activations():             1 #pragma omp parallel
│   └── multiply_with_precomputed Down:     1 #pragma omp parallel
└── Residual Add      : 0 regions (SIMD only) ✓

Per-layer total: 10 parallel region entries
LM Head: 2 parallel region entries

Grand total: 48×10 + 2 = 482 parallel region entries per token
```

## Target Architecture (Solution)

```
Per-token decode: ~50 parallel region entries

Per Layer (×48):
├── #pragma omp parallel  ← ONE parallel region entry per layer
│   ├── #pragma omp for (QKV quantize)
│   ├── #pragma omp for (Q GEMM)
│   ├── #pragma omp for (K GEMM)
│   ├── #pragma omp for (V GEMM)
│   ├── [JIT attention - single-threaded]
│   ├── #pragma omp for (Gate/Up quantize)
│   ├── #pragma omp for (Gate GEMM)
│   ├── #pragma omp for (Up GEMM)
│   ├── #pragma omp for (SwiGLU)
│   ├── #pragma omp for (Down quantize)
│   └── #pragma omp for (Down GEMM)
└── (end parallel region)

Per-layer total: 1 parallel region entry
LM Head: 1 parallel region entry
Embedding lookup: 1 entry

Grand total: 48 + 1 + 1 = 50 parallel region entries per token
```

**Expected improvement: ~10× reduction in parallel region overhead**

## Implementation Phases

### Phase 1: Kernel-Level `omp_in_parallel()` Guards

Each kernel checks if it's already inside a parallel region:
- If NOT in parallel → create `#pragma omp parallel { #pragma omp for }`
- If IN parallel → just use `#pragma omp for` (worksharing)

#### Files to Modify:

**1. QuantisedGemmKernel.h** (src/v2/kernels/cpu/gemm_v4/)

Functions to modify:
- `quantize_activations()` (line ~1290)
- `multiply_with_precomputed_q8_1()` (line ~1417)
- `multiply_with_precomputed_q8_1_to_q8_1()` (line ~1625)
- `multiply()` (line ~689)

Pattern:
```cpp
// BEFORE:
#pragma omp parallel
{
    // ... work ...
}

// AFTER:
#ifdef _OPENMP
bool already_parallel = omp_in_parallel();
#else
bool already_parallel = false;
#endif

if (already_parallel) {
    // Worksharing only - thread pool already exists
    #pragma omp for schedule(static)
    for (...) { /* work */ }
} else {
    // Create new parallel region
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (...) { /* work */ }
    }
}
```

**2. SwiGLUPrimitives.cpp** (src/v2/kernels/cpu/primitives/)

Functions to modify:
- `compute_swiglu()` (line ~208)
- `compute_swiglu_q8_1_avx2()` (line ~270)

Add `omp_in_parallel()` check and threshold for small inputs.

**3. RMSNormPrimitives.cpp** 
Already has `omp_in_parallel()` checks! ✓

**4. RoPEPrimitives.cpp**
Already disabled for seq_len=1. Add guard for safety.

### Phase 2: Layer-Level Parallel Region

Add outer `#pragma omp parallel` in pipeline:

**File:** `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`

**Location:** `transformer_layer()` method

```cpp
bool Qwen2Pipeline::transformer_layer(int layer_idx, int effective_seq_len)
{
    LOG_TRACE("Processing layer " << layer_idx);
    auto &layer = getLayerWeights(layer_idx);

    // =========================================================
    // LAYER-LEVEL PARALLEL REGION
    // All parallelizable operations execute as worksharing
    // within this single parallel region.
    // =========================================================
    bool success = true;
    
#pragma omp parallel
    {
        // Set thread-local success flag
        bool thread_success = true;
        
        #pragma omp single
        {
            // Attention block (contains worksharing loops)
            if (!attention_block(layer, layer_idx, effective_seq_len))
            {
                thread_success = false;
            }
            
            // FFN block (contains worksharing loops)
            if (thread_success && !ffn_block(layer, layer_idx, effective_seq_len))
            {
                thread_success = false;
            }
        }
        
        // Reduce success across threads
        #pragma omp atomic update
        success &= thread_success;
    }
    
    return success;
}
```

**Alternative (simpler):** Just wrap the layer body in `#pragma omp parallel` and let
kernels detect via `omp_in_parallel()`.

### Phase 3: Thread-Local Scratch Buffers

Some operations allocate thread-local scratch buffers inside parallel regions.
With layer-level fusion, these allocations happen once per layer instead of per-kernel.

**Potential optimization:** Pre-allocate scratch buffers at pipeline construction.

### Phase 4: Verification and Benchmarking

1. **Correctness:** Run full test suite
   ```bash
   ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure --parallel
   ctest --test-dir build_v2_e2e_release -R "^V2_E2E_" --output-on-failure
   ```

2. **Performance:** Run benchmark before/after
   ```bash
   ./run_llaminar.sh -- --benchmark -m models/qwen2.5-14b-instruct.q8_0.gguf -n 50
   ```

3. **Profiling:** Verify parallel region reduction
   ```bash
   perf record -F 999 -e cpu-clock -g ./run_llaminar.sh ...
   perf report
   ```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Correctness regression | Medium | High | Extensive testing, parity checks |
| Nested parallelism issues | Low | Medium | `omp_in_parallel()` guards prevent nesting |
| Thread-local buffer races | Medium | High | Ensure buffers are per-thread or protected |
| Load imbalance | Low | Low | Static scheduling, similar work per thread |
| JIT kernel interference | Low | Medium | JIT runs single-threaded, no conflict |

## Expected Results

| Metric | Before | After (Target) |
|--------|--------|----------------|
| Parallel regions/token | 482 | ~50 |
| libgomp overhead | 29% | <5% |
| Decode throughput | 1.0x | 1.3-1.5x |

## Implementation Checklist

- [ ] Phase 1a: Add `omp_in_parallel()` to `QuantisedGemmKernel::quantize_activations()`
- [ ] Phase 1b: Add `omp_in_parallel()` to `QuantisedGemmKernel::multiply_with_precomputed_q8_1()`
- [ ] Phase 1c: Add `omp_in_parallel()` to `QuantisedGemmKernel::multiply_with_precomputed_q8_1_to_q8_1()`
- [ ] Phase 1d: Add `omp_in_parallel()` to `SwiGLUPrimitives::compute_swiglu()`
- [ ] Phase 1e: Add `omp_in_parallel()` to `SwiGLUPrimitives::compute_swiglu_q8_1_avx2()`
- [ ] Phase 2: Add layer-level `#pragma omp parallel` in `transformer_layer()`
- [ ] Phase 3: Run unit tests
- [ ] Phase 4: Run E2E parity tests
- [ ] Phase 5: Benchmark and profile

## Appendix: Code Snippets

### A. QuantisedGemmKernel Pattern

```cpp
bool quantize_activations(const float *A, void *q8_1_buffer, int m, int k) override
{
    if (!A || !q8_1_buffer) return false;

    int k_blocks = (k + 31) / 32;
    Q8_1Block *all_blocks = reinterpret_cast<Q8_1Block *>(q8_1_buffer);
    const bool k_aligned = (k % 32 == 0);

#ifdef _OPENMP
    bool in_parallel = omp_in_parallel();
#else
    bool in_parallel = false;
#endif

    auto quantize_work = [&](int start_row, int end_row) {
        for (int i = start_row; i < end_row; ++i) {
            const float *a_row = A + i * k;
            Q8_1Block *row_blocks = all_blocks + i * k_blocks;
            for (int k_blk = 0; k_blk < k_blocks; ++k_blk) {
                int valid_elements = std::min(32, k - k_blk * 32);
                simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], valid_elements);
            }
        }
    };

    if (in_parallel) {
        // Already in parallel region - use worksharing
        #pragma omp for schedule(static)
        for (int i = 0; i < m; ++i) {
            quantize_work(i, i + 1);
        }
    } else {
        // Create new parallel region
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; ++i) {
            quantize_work(i, i + 1);
        }
    }

    return true;
}
```

### B. SwiGLU Pattern

```cpp
void compute_swiglu_q8_1_avx2(const void *gate, const void *up, void *output, int size)
{
    const Q8_1Block *g_blocks = static_cast<const Q8_1Block *>(gate);
    const Q8_1Block *u_blocks = static_cast<const Q8_1Block *>(up);
    Q8_1Block *o_blocks = static_cast<Q8_1Block *>(output);
    int num_blocks = size / Q8_1Block::BLOCK_SIZE;

#ifdef _OPENMP
    bool in_parallel = omp_in_parallel();
    // Also add threshold: don't parallelize tiny work
    bool should_parallelize = !in_parallel && num_blocks > 64;
#else
    bool should_parallelize = false;
    bool in_parallel = false;
#endif

    if (in_parallel) {
        // Worksharing within existing parallel region
        #pragma omp for schedule(static)
        for (int b = 0; b < num_blocks; ++b) {
            // ... SwiGLU computation ...
        }
    } else if (should_parallelize) {
        // Create new parallel region
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < num_blocks; ++b) {
            // ... SwiGLU computation ...
        }
    } else {
        // Sequential (small work or nested)
        for (int b = 0; b < num_blocks; ++b) {
            // ... SwiGLU computation ...
        }
    }
}
```

### C. Layer-Level Fusion in Pipeline

```cpp
bool Qwen2Pipeline::transformer_layer(int layer_idx, int effective_seq_len)
{
    LOG_TRACE("Processing layer " << layer_idx);
    auto &layer = getLayerWeights(layer_idx);

    // ================================================================
    // LAYER-LEVEL PARALLEL REGION
    // 
    // This single parallel region encompasses ALL parallelizable work
    // in the transformer layer. Individual kernels detect they are
    // already in a parallel region via omp_in_parallel() and use
    // worksharing (#pragma omp for) instead of creating new regions.
    //
    // This reduces parallel region entries from ~10 per layer to 1.
    // ================================================================
    
    bool attention_success = false;
    bool ffn_success = false;
    
#pragma omp parallel
    {
        #pragma omp single nowait
        {
            // Attention block - all GEMMs use worksharing
            attention_success = attention_block(layer, layer_idx, effective_seq_len);
        }
        
        // Implicit barrier here - all threads sync before FFN
        
        if (attention_success) {
            #pragma omp single nowait
            {
                // FFN block - all GEMMs use worksharing
                ffn_success = ffn_block(layer, layer_idx, effective_seq_len);
            }
        }
    }
    
    if (!attention_success) {
        LOG_ERROR("Attention block failed in layer " << layer_idx);
        return false;
    }
    
    if (!ffn_success) {
        LOG_ERROR("FFN block failed in layer " << layer_idx);
        return false;
    }
    
    return true;
}
```

## References

- OpenMP 5.0 Specification: `omp_in_parallel()` runtime library routine
- Intel Threading Building Blocks: Work-stealing vs worksharing analysis
- LLVM OpenMP Runtime: Thread pool management overhead
