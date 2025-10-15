# NUMA-Aware Memory Allocation Issue in ModelLoader

**Date:** October 15, 2025  
**Author:** David Sanftenberg  
**Type:** Critical Performance Issue  
**Status:** Identified - Needs Fix

## Executive Summary

**❌ CRITICAL ISSUE: Model weights are NOT NUMA-local by default**

When MPI ranks bind to different NUMA nodes, model weights loaded by `ModelLoader` may be allocated on the wrong NUMA node, causing severe performance degradation due to remote memory access.

## Problem Description

### Current Behavior

```cpp
// ModelLoader.cpp line 736
data_f32.resize(n_elems);  // ❌ Allocated on master thread's NUMA node
std::memcpy(data_f32.data(), raw.data(), n_elems * sizeof(float));
```

**What happens:**
1. Model loading occurs on **main thread** (typically rank 0's NUMA node)
2. `std::vector::resize()` allocates memory on **whatever NUMA node** happens to service the page fault
3. Memory pages are **not guaranteed** to be local to the MPI rank that will use them
4. When rank 1 (on NUMA node 1) accesses weights allocated on node 0 → **remote memory access penalty**

### NUMA Penalty

Remote NUMA memory access is **2-3x slower** than local:
- **Local access**: ~100ns latency
- **Remote access**: ~200-300ns latency
- **Bandwidth**: 50% reduction for remote access

For compute-bound operations (matmul, attention), this kills performance.

## Root Cause Analysis

### Memory Allocation Paths

**1. F32 Direct Copy (line 736):**
```cpp
data_f32.resize(n_elems);  // ❌ Master thread allocation
std::memcpy(data_f32.data(), raw.data(), n_elems * sizeof(float));
```

**2. F16 Conversion (line 770):**
```cpp
data_f32.resize(n_elems);  // ❌ Master thread allocation

#pragma omp parallel for schedule(static)
for (long long i = 0; i < (long long)n_elems; ++i) {
    // Conversion happens in parallel, but memory already allocated
    data_f32[i] = ggml_compute_fp16_to_fp32(h);
}
```

**3. Quantized Dequantization (line 1605):**
```cpp
std::vector<float> out(n_elements, 0.0f);  // ❌ Single-thread allocation
dequantize_row_q8_0(..., out.data(), ...);  // Then fill in parallel
```

### Why This Is Broken

Linux kernel uses **first-touch policy** for NUMA allocation:
1. `resize()` or `std::vector(size)` reserves virtual memory (cheap)
2. First **write** to a page triggers page fault
3. Kernel allocates physical page on **NUMA node of the faulting thread**

**Problem:** 
- Allocation happens on main thread → wrong NUMA node
- Even with OpenMP parallelization for filling, pages already allocated on wrong node
- MPI rank binding is **irrelevant** if pages are already on wrong node

## System Configuration

From `run_llaminar.sh` and MPI binding:
```bash
# 2 sockets, 28 cores per socket
SOCKETS=2
CORES_PER_SOCKET=28

# MPI binding (correct)
mpirun -np $SOCKETS --bind-to socket --map-by socket

# Rank 0 → NUMA node 0 (socket 0)
# Rank 1 → NUMA node 1 (socket 1)
```

**Expected:** Each rank's weights local to its NUMA node  
**Actual:** Weights may be on node 0 (or unpredictable), accessed remotely by rank 1

## Evidence

### OpenMP Parallelization Is Not Enough

```cpp
// Line 776 - F16 conversion with OpenMP
#pragma omp parallel for schedule(static)
for (long long i = 0; i < (long long)n_elems; ++i) {
    data_f32[i] = ggml_compute_fp16_to_fp32(h);  // ✅ Parallel write
}
```

**This writes in parallel but doesn't allocate in parallel!**
- `resize()` happened **before** the parallel region
- Pages already allocated on master thread's node
- Parallel writes just fill pre-allocated pages

## Proposed Solution

### First-Touch Parallel Allocation

Replace single-threaded allocation with parallel first-touch:

```cpp
// BEFORE (broken)
data_f32.resize(n_elems);
#pragma omp parallel for
for (size_t i = 0; i < n_elems; ++i) {
    data_f32[i] = compute_value(i);
}

// AFTER (NUMA-aware)
data_f32.resize(n_elems);  // Reserve virtual address space

// Parallel first-touch: Each thread initializes its chunk
#pragma omp parallel
{
    size_t chunk_size = (n_elems + omp_get_num_threads() - 1) / omp_get_num_threads();
    size_t start = omp_get_thread_num() * chunk_size;
    size_t end = std::min(start + chunk_size, n_elems);
    
    // First touch: Allocate pages on this thread's NUMA node
    for (size_t i = start; i < end; ++i) {
        data_f32[i] = 0.0f;  // Trigger page fault on local thread
    }
    
    // Implicit barrier here
}

// Now fill with actual values (pages already allocated locally)
#pragma omp parallel for schedule(static)
for (long long i = 0; i < (long long)n_elems; ++i) {
    data_f32[i] = compute_value(i);
}
```

### Implementation Points

**1. F32 Path (direct copy):**
```cpp
data_f32.resize(n_elems);

// Parallel first-touch
#pragma omp parallel
{
    const size_t nthreads = omp_get_num_threads();
    const size_t tid = omp_get_thread_num();
    const size_t chunk = (n_elems + nthreads - 1) / nthreads;
    const size_t start = tid * chunk;
    const size_t end = std::min(start + chunk, n_elems);
    
    // First touch
    std::fill(&data_f32[start], &data_f32[end], 0.0f);
}

// Then copy (can be parallel or serial, pages already local)
std::memcpy(data_f32.data(), raw.data(), n_elems * sizeof(float));
```

**2. F16 Path (with conversion):**
```cpp
data_f32.resize(n_elems);
const uint8_t *src_bytes = raw.data();

#pragma omp parallel
{
    const size_t nthreads = omp_get_num_threads();
    const size_t tid = omp_get_thread_num();
    const size_t chunk = (n_elems + nthreads - 1) / nthreads;
    const size_t start = tid * chunk;
    const size_t end = std::min(start + chunk, n_elems);
    
    // Combined first-touch + conversion (single pass)
    for (size_t i = start; i < end; ++i) {
        uint16_t h;
        std::memcpy(&h, src_bytes + 2 * i, 2);
        data_f32[i] = ggml_compute_fp16_to_fp32(h);
    }
}
```

**3. Quantized Path (dequantizers):**
```cpp
std::vector<float> out(n_elements);  // Don't initialize yet!

// Parallel first-touch
#pragma omp parallel
{
    const size_t nthreads = omp_get_num_threads();
    const size_t tid = omp_get_thread_num();
    const size_t chunk = (n_elements + nthreads - 1) / nthreads;
    const size_t start = tid * chunk;
    const size_t end = std::min(start + chunk, n_elements);
    
    std::fill(&out[start], &out[end], 0.0f);
}

// Then dequantize (ggml functions may not be thread-safe, verify!)
dequantize_row_q8_0(..., out.data(), ...);
```

### Environment Control

Add environment variable to control behavior:

```cpp
// DebugEnv.h
struct LoaderEnv {
    bool numa_first_touch = true;  // LLAMINAR_NUMA_FIRST_TOUCH (default ON)
    bool verify_locality = false;  // LLAMINAR_VERIFY_NUMA_LOCALITY
};
```

## Performance Impact

### Expected Improvements

For 700MB model on 2-socket system:
- **Current**: 50% of accesses remote (rank 1 accessing node 0 memory)
- **With fix**: >95% local access (assuming proper OpenMP binding)

**Estimated speedup:**
- Prefill (memory-bound): **1.3-1.5x faster**
- Decode (compute-bound): **1.1-1.2x faster**
- Overall throughput: **20-40% improvement**

### Microbenchmark

Test NUMA locality impact:
```bash
# Before fix
./run_llaminar.sh --benchmark -m model.gguf -p "Test" -n 50
# Measure: prefill tok/s, decode tok/s

# After fix
export LLAMINAR_NUMA_FIRST_TOUCH=1
./run_llaminar.sh --benchmark -m model.gguf -p "Test" -n 50
# Expect: 20-40% improvement in prefill
```

## Validation Strategy

### 1. NUMA Locality Verification

```bash
# Check memory distribution across NUMA nodes
numastat -p $(pgrep llaminar)

# Expected after fix:
# Node 0: ~350MB (rank 0's weights)
# Node 1: ~350MB (rank 1's weights)

# Current (broken):
# Node 0: ~700MB (all weights)
# Node 1: ~0MB
```

### 2. Performance Regression Testing

```bash
# Smoke test with NUMA verification
export LLAMINAR_VERIFY_NUMA_LOCALITY=1
./run_llaminar.sh -m model.gguf -v

# Should log warnings if remote memory detected
```

### 3. A/B Performance Comparison

```bash
# Baseline (broken)
export LLAMINAR_NUMA_FIRST_TOUCH=0
./run_llaminar.sh --benchmark -m model.gguf -p "Test" -n 100

# Fixed
export LLAMINAR_NUMA_FIRST_TOUCH=1
./run_llaminar.sh --benchmark -m model.gguf -p "Test" -n 100

# Compare prefill/decode tok/s
```

## Implementation Checklist

- [ ] Add `numa_first_touch` flag to DebugEnv
- [ ] Implement parallel first-touch for F32 path (line 736)
- [ ] Implement parallel first-touch for F16 path (line 770)
- [ ] Implement parallel first-touch for Q8_0 (line 1605)
- [ ] Implement parallel first-touch for Q4_0, Q4_K, Q6_K (similar pattern)
- [ ] Add NUMA locality verification (optional, debug mode)
- [ ] Update documentation (README, copilot-instructions)
- [ ] Benchmark before/after on 2-socket system
- [ ] Test with weight sharding enabled (combine with previous fix)
- [ ] Verify correctness (parity tests should still pass)

## Related Issues

### Interaction with Weight Sharding

**Good news:** Weight sharding (when enabled) partially mitigates this:
- Each rank loads only its column slice
- Smaller memory footprint per rank
- Still needs NUMA-aware allocation for the sliced data

**Fix applies to both cases:**
- Replicated weights: Need NUMA locality per rank
- Sharded weights: Need NUMA locality for the slice

### OpenMP Thread Binding

From `run_llaminar.sh`:
```bash
export OMP_PLACES=sockets
export OMP_PROC_BIND=close
```

**This is correct** - threads stay on their socket. But it's useless if memory is remote!

**First-touch allocation + OMP binding = optimal NUMA performance**

## Testing Plan

### Phase 1: Implement and Verify

1. Add environment flag
2. Implement first-touch for F32, F16, Q8_0
3. Run smoke tests (verify correctness)
4. Check `numastat` output (verify locality)

### Phase 2: Performance Validation

1. Baseline benchmark (NUMA_FIRST_TOUCH=0)
2. Fixed benchmark (NUMA_FIRST_TOUCH=1)
3. Measure improvement across:
   - Different model sizes
   - Different quantization formats
   - Prefill vs decode performance

### Phase 3: Rollout

1. Enable by default if >20% improvement observed
2. Document in README and copilot-instructions
3. Add to canonical script defaults

## References

- Linux NUMA first-touch policy: https://www.kernel.org/doc/html/latest/vm/numa.html
- OpenMP memory affinity: https://www.openmp.org/spec-html/5.0/openmpsu58.html
- ModelLoader allocation: `src/ModelLoader.cpp` lines 736, 770, 1605

## Conclusion

**Critical NUMA issue identified:**
- Model weights allocated on wrong NUMA node
- Up to 2-3x memory latency penalty
- 20-40% performance loss estimated

**Solution is straightforward:**
- Parallel first-touch allocation before filling
- Combine with existing OpenMP thread binding
- Minimal code changes, large performance gain

**Next steps:**
1. Implement parallel first-touch for all allocation paths
2. Add environment controls and verification
3. Benchmark and validate improvement
4. Document and enable by default

This fix is **essential** for multi-socket performance and should be prioritized alongside weight sharding implementation.
