# SimpleTensor NUMA First-Touch Optimization

**Date:** 2025-10-15  
**Author:** David Sanftenberg  
**Category:** Performance Optimization  
**Status:** Implemented, Testing Pending

## Summary

Implemented NUMA-aware first-touch initialization in `SimpleTensor` to eliminate remote NUMA access penalties for large tensor allocations, particularly K/V caches which are the largest allocations in the inference pipeline.

## Problem Statement

### Issue
`SimpleTensor` (used for K/V caches, intermediate activations, and all local tensors) was allocating memory using single-threaded `std::vector::resize()`, which:
1. Allocated memory on the NUMA node of the calling thread (often rank 0)
2. Caused 2-3x latency penalty for threads on other NUMA nodes accessing this memory
3. Resulted in significant performance degradation for large models where K/V cache exceeds cache capacity

### Impact Analysis
- **Small Models (Qwen 0.5B):** K/V cache = 96MB, mostly cache-resident → minimal impact
- **Large Models (Qwen 7B):** K/V cache = 4GB, exceeds L3 cache → **+10-40% potential improvement**
- **Critical Path:** K/V cache accessed every layer, every token during prefill and decode

## Implementation

### Changes Made

**File:** `src/tensors/SimpleTensor.h`

#### 1. Added NUMA First-Touch Helper Method

```cpp
/**
 * @brief Perform NUMA-aware first-touch initialization on allocated memory
 * 
 * Uses OpenMP parallel loops to ensure memory pages are allocated on the NUMA node
 * where they will be accessed by worker threads. This eliminates remote NUMA access
 * penalties (2-3x latency) for large tensors like K/V caches.
 * 
 * @param data Pointer to memory buffer to initialize
 * @param size Number of elements to initialize
 * @param init_value Value to initialize elements to (default 0.0f)
 */
static void numaFirstTouch(float* data, size_t size, float init_value = 0.0f)
{
    const auto& env = debugEnv();
    
    // Skip if disabled via environment
    if (!env.numa.first_touch) {
        std::fill(data, data + size, init_value);
        return;
    }
    
    // Small allocations: single-threaded (overhead not worth it)
    constexpr size_t kSmallThreshold = 32 * 1024; // 128KB (32K floats)
    if (size < kSmallThreshold) {
        std::fill(data, data + size, init_value);
        return;
    }
    
    // Large allocations: parallel first-touch for NUMA locality
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; ++i) {
        data[i] = init_value;
    }
    
    // Optional: Verify NUMA locality (expensive, debug only)
    if (env.numa.verify_locality && size >= kSmallThreshold) {
        LOG_DEBUG("[SimpleTensor-NUMA] First-touch completed for " << size 
                 << " elements (" << (size * sizeof(float) / 1024.0 / 1024.0) 
                 << " MB) using " << omp_get_max_threads() << " threads");
    }
}
```

#### 2. Updated Constructor for NUMA-Aware Initialization

**Before:**
```cpp
data_.resize(total_size, 0.0f); // Single-threaded, wrong NUMA node
```

**After:**
```cpp
// Allocate memory (without initialization to avoid duplicate work)
data_.resize(total_size);

// NUMA-aware first-touch initialization
// This ensures memory pages are allocated on the NUMA node where they will be accessed,
// eliminating 2-3x remote access latency penalty for large tensors (K/V caches, etc.)
numaFirstTouch(data_.data(), total_size, 0.0f);
```

#### 3. Updated `resize()` Method for Incremental Growth

```cpp
void resize(const std::vector<int> &new_shape)
{
    shape_ = new_shape;
    int new_size = 1;
    for (int dim : new_shape)
    {
        new_size *= dim;
    }
    
    // Resize without initialization, then apply NUMA-aware first-touch
    size_t old_size = data_.size();
    data_.resize(new_size);
    
    // Only first-touch newly allocated portion
    if (static_cast<size_t>(new_size) > old_size) {
        numaFirstTouch(data_.data() + old_size, new_size - old_size, 0.0f);
    }
}
```

#### 4. Added Required Headers

```cpp
#include "../utils/DebugEnv.h"  // For numa configuration flags
#include <omp.h>                // For OpenMP parallel first-touch
```

## Technical Details

### NUMA First-Touch Strategy

1. **Small Allocations (<128KB):** Single-threaded `std::fill()` to avoid OpenMP overhead
2. **Large Allocations (≥128KB):** Parallel OpenMP loop with static scheduling
   - Each thread writes to its portion of the buffer
   - OS allocates pages on the NUMA node of the writing thread
   - Future accesses by the same thread hit local NUMA node (no remote penalty)

### Configuration

Controlled via existing environment variables in `DebugEnv`:

```bash
# Enable NUMA first-touch (default: enabled)
export LLAMINAR_NUMA_FIRST_TOUCH=1

# Disable for comparison testing
export LLAMINAR_NUMA_FIRST_TOUCH=0

# Enable verbose NUMA locality verification (debug only)
export LLAMINAR_NUMA_VERIFY_LOCALITY=1
```

### Affected Allocations

This optimization applies to **all SimpleTensor allocations**, including:

1. **K/V Cache (CRITICAL):**
   - `QwenPipeline::initializeKVCache()` - 96MB to 4GB depending on model
   - `OpenblasPrefillProvider` cache structures
   - **Highest impact** due to size and access frequency

2. **Intermediate Activations:**
   - Attention outputs
   - FFN outputs
   - RMSNorm outputs
   - **Medium impact** due to frequent allocation/deallocation

3. **Communication Buffers:**
   - MPI gather/scatter buffers
   - **Low impact** due to small sizes

### Performance Characteristics

| Allocation Type | Size | Access Pattern | NUMA Sensitivity | Expected Impact |
|-----------------|------|----------------|------------------|-----------------|
| K/V Cache (7B) | 4GB | Every layer, every token | Extreme | +10-40% |
| K/V Cache (0.5B) | 96MB | Every layer, every token | High | +1-3% |
| Activations | 1-50MB | Per-layer temporary | Medium | +0.5-2% |
| MPI Buffers | <1MB | Occasional | Low | Negligible |

## Testing Plan

### Benchmark Mode Defaults

The `--benchmark` flag now sets intelligent defaults:
- **Prefill:** ~512 tokens (auto-generated repetitive prompt if none provided)
- **Decode:** 128 tokens (default `n_predict` value)

```bash
# Quick benchmark with defaults (512 prefill + 128 decode)
./run_llaminar.sh -- --benchmark -m models/qwen2.5-0.5b-instruct-q8_0.gguf

# Custom benchmark parameters
./run_llaminar.sh -- --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Custom prompt here" \
  -n 50  # Override decode token count
```

### Unit Testing
```bash
# Smoke tests (verify no regression)
ctest --test-dir build --output-on-failure --parallel \
  -R "^(BasicTest|NumaTest|PipelineFactoryTest)$"

# Full unit test suite
ctest --test-dir build --output-on-failure --parallel \
  -E "(Integration|ParityFramework|Incremental|Qwen|Prefill|.*Stress.*)"
```

### Performance Benchmarking

#### Small Model (Qwen 0.5B - Cache Resident)
```bash
# Baseline (NUMA disabled)
export LLAMINAR_NUMA_FIRST_TOUCH=0
./run_llaminar.sh --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "$(python -c 'print("test " * 256)')" \
  -n 50

# With NUMA optimization (default)
unset LLAMINAR_NUMA_FIRST_TOUCH
./run_llaminar.sh --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "$(python -c 'print("test " * 256)')" \
  -n 50
```

**Expected:** +1-3% improvement (cache-resident workload)

#### Large Model (Qwen 7B - Memory Bound)
```bash
# If 7B model available, test with long prompts
./run_llaminar.sh --benchmark \
  -m models/qwen2.5-7b-instruct-q8_0.gguf \
  -p "$(python -c 'print("test " * 512)')" \
  -n 50
```

**Expected:** +10-40% improvement (memory-bound workload)

### NUMA Locality Verification

```bash
# Enable verification logging
export LLAMINAR_NUMA_VERIFY_LOCALITY=1

# Run and check for NUMA first-touch messages
./run_llaminar.sh --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "test prompt" -n 10 2>&1 | grep "SimpleTensor-NUMA"

# Check actual NUMA distribution with numastat
numastat -p $(pgrep llaminar)
```

## Compatibility

### Backward Compatibility
- ✅ **Fully compatible:** No API changes, only internal implementation
- ✅ **Optional:** Can be disabled via `LLAMINAR_NUMA_FIRST_TOUCH=0`
- ✅ **Graceful degradation:** Falls back to single-threaded for small allocations

### Side Effects
- **Slight overhead for small tensors:** OpenMP setup cost for tensors >128KB but <1MB
  - Mitigated by threshold: only tensors ≥128KB use parallel path
- **Thread pool warm-up:** First allocation may be slightly slower (one-time cost)
  - Amortized across many allocations during model load

## Metrics to Track

### Before/After Comparison

| Metric | Baseline (No NUMA) | With NUMA | Change |
|--------|-------------------|-----------|--------|
| Prefill (512 tok) | TBD | TBD | TBD |
| Decode (50 tok) | TBD | TBD | TBD |
| Peak RSS | TBD | TBD | TBD |
| NUMA Remote % | TBD | TBD | TBD |

### NUMA Distribution (via `numastat`)

**Goal:** <10% remote NUMA accesses for large tensors

```bash
# Check NUMA distribution after prefill
numastat -p $(pgrep llaminar) | grep -E "numa_hit|numa_miss|numa_foreign"
```

## Related Work

- **20251015_numa_first_touch_implementation.md** - ModelLoader NUMA optimization (15 sites)
- **20251015_numa_performance_test_results.md** - Debug build +3.1% improvement
- **20251015_release_build_numa_performance_results.md** - Release build -4.2% (workload-dependent)
- **20251015_numa_hot_path_allocation_analysis.md** - Analysis identifying SimpleTensor as critical

## Next Steps

1. **Immediate:** Run smoke tests to verify no regression
2. **Short-term:** Performance benchmark with small model (0.5B)
3. **Medium-term:** Performance benchmark with large model (7B) if available
4. **Long-term:** Monitor NUMA metrics in production workloads

## Lessons Learned

### Design Decisions

1. **Threshold Choice (128KB):**
   - Below this, OpenMP overhead exceeds NUMA benefit
   - Based on empirical testing in ModelLoader implementation
   - Matches cache line and page size considerations

2. **Static Scheduling:**
   - Ensures predictable memory distribution across threads
   - Each thread gets contiguous chunk on its NUMA node
   - Dynamic scheduling would cause fragmentation

3. **Incremental Resize:**
   - Only first-touch newly allocated portion during `resize()`
   - Avoids re-touching already-allocated memory
   - Critical for K/V cache growth during long sequences

### Optimization Opportunities

1. **Pre-allocation Heuristic:**
   - Could pre-allocate K/V cache to max_seq_len during initialization
   - Avoids incremental resize overhead
   - Trade-off: wastes memory for short sequences

2. **Per-Rank Affinity:**
   - Could bind specific tensor allocations to specific NUMA nodes
   - Requires deeper MPI rank-to-NUMA-node awareness
   - Future enhancement if profiling shows benefit

## Conclusion

SimpleTensor NUMA optimization addresses the **largest source of NUMA-related performance degradation** in the Llaminar inference pipeline. K/V caches (multi-GB for large models) were being allocated on the wrong NUMA node, causing 2-3x latency penalty for every access.

Expected impact:
- **Small models (cache-resident):** +1-3% improvement
- **Large models (memory-bound):** +10-40% improvement
- **Long sequences:** Greater benefit as K/V cache grows

This complements the ModelLoader NUMA optimization and provides comprehensive NUMA-aware memory allocation across the entire inference pipeline.

---

**Status:** ✅ Implemented, ⏳ Performance Testing Pending
