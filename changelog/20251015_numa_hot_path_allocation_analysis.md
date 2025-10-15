# NUMA Hot Path Allocation Analysis

**Date:** 2025-10-15  
**Author:** David Sanftenberg  
**Category:** Performance Optimization  
**Status:** Analysis Complete, Implementation Pending

## Executive Summary

Comprehensive analysis of allocation sites beyond ModelLoader that could benefit from NUMA-aware first-touch allocation. Identified 5 categories of allocations in prefill/decode hot paths, prioritized by size, access frequency, and NUMA sensitivity.

## Analysis Methodology

Searched codebase for allocation patterns:
- `std::vector::resize()` / `reserve()`
- `std::vector<float>` declarations
- `new[]` / `malloc` / `calloc`
- Focus on pipeline operators, providers, and kernel infrastructure

## Findings: NUMA Optimization Candidates

### 1. **RoPE Frequency Tables** (HIGH PRIORITY ⭐⭐⭐)

**Location:** `src/operators/MPIRoPEOperator.cpp:348-349`

```cpp
void MPIRoPEOperator::precomputeFrequencyTables() {
    int freq_dims = head_dim_ / 2;
    cos_table_.resize(max_seq_len_ * freq_dims);  // ← NUMA candidate
    sin_table_.resize(max_seq_len_ * freq_dims);  // ← NUMA candidate
    
    // Precompute for all positions and frequency dimensions
    for (int pos = 0; pos < max_seq_len_; ++pos) {
        for (int dim = 0; dim < freq_dims; ++dim) {
            float freq = 1.0f / std::pow(theta_, static_cast<float>(2 * dim) / head_dim_);
            float angle = pos * freq;
            
            int idx = pos * freq_dims + dim;
            cos_table_[idx] = std::cos(angle);
            sin_table_[idx] = std::sin(angle);
        }
    }
}
```

**Characteristics:**
- **Size:** `max_seq_len * (head_dim / 2) * sizeof(float)` per table
  - Example: 2048 seq × 64 dims × 4 bytes = 512KB per table (1MB total)
  - Larger models (7B/13B): 8192 seq × 128 dims × 4 bytes = 4MB per table (8MB total)
- **Access Pattern:** Read-intensive, accessed every token in prefill and decode
- **NUMA Sensitivity:** **VERY HIGH** - remote access would add 2-3x latency to every RoPE operation
- **Lifecycle:** Allocated once during initialization, never freed
- **Current Allocation:** Default `std::vector::resize()` (no first-touch)

**Recommendation:**
- **Action:** Implement NUMA first-touch during precomputation loop
- **Approach:** The existing initialization loop already touches every element, so allocation is likely already NUMA-local
- **Verification Needed:** Confirm with `numastat` whether allocation happens on correct node
- **Expected Impact:** Low (likely already optimal), but worth verifying for future reference

**Implementation Note:**
The precomputation loop already performs first-touch naturally:
```cpp
cos_table_[idx] = std::cos(angle);  // Write touches memory
sin_table_[idx] = std::sin(angle);  // Write touches memory
```
**Verdict:** Probably already NUMA-optimal due to initialization pattern. Low priority unless `numastat` shows issues.

---

### 2. **K/V Cache Structures** (HIGH PRIORITY ⭐⭐⭐)

**Location:** Multiple files

#### 2a. QwenPipeline Cache Initialization
`src/QwenPipeline.cpp:1523-1537`

```cpp
void QwenPipeline::initializeKVCache(int seq_len) {
    k_cache_.resize(n_layers);
    v_cache_.resize(n_layers);
    for (int l = 0; l < config_.getLayerConfig().n_layers; ++l) {
        bool recreated_k = false;
        bool recreated_v = false;
        
        if (!k_cache_[l] || k_cache_[l]->shape()[0] < seq_len) {
            k_cache_[l] = createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim});
            recreated_k = true;
        }
        if (!v_cache_[l] || v_cache_[l]->shape()[0] < seq_len) {
            v_cache_[l] = createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim});
            recreated_v = true;
        }
    }
}
```

#### 2b. OpenblasPrefillProvider Cache Allocation
`src/OpenblasPrefillProvider.cpp:50-51`

```cpp
OpenBLASPrefillProvider::OpenBLASPrefillProvider(
    const ModelConfig &config, const MPIContext &mpi_ctx)
    : PrefillProviderBaseImpl(config, mpi_ctx) {
    
    int n_layers = config.getLayerConfig().n_layers;
    k_cache_.resize(n_layers);  // ← Vector-of-pointers
    v_cache_.resize(n_layers);  // ← Vector-of-pointers
}
```

**Characteristics:**
- **Size:** **VERY LARGE** - Multiple GB for larger models
  - Per layer: `seq_len × n_head_kv × head_dim × sizeof(float)`
  - Example (Qwen 0.5B): 2048 seq × 8 heads × 64 dim × 4 bytes = 4MB per layer × 24 layers = **96MB total**
  - Example (Qwen 7B): 8192 seq × 32 heads × 128 dim × 4 bytes = 128MB per layer × 32 layers = **4GB total**
- **Access Pattern:** Read/write intensive during prefill, read-only during decode
- **NUMA Sensitivity:** **EXTREME** - multi-GB structures, critical path operations
- **Lifecycle:** Allocated during first prefill, grows as needed, retained throughout inference
- **Current Allocation:** Via `TensorFactory::create_local_tensor()` - needs investigation

**Recommendation:**
- **Action:** Investigate `createLocalTensor()` implementation to determine if NUMA-aware
- **Priority:** **CRITICAL** - largest allocations in the system
- **Expected Impact:** Potentially **+10-40% performance** for larger models if not already NUMA-aware
- **Verification:** Must check if `TensorFactory::create_local_tensor()` uses NUMA allocation

**Investigation Required:**
```bash
# Check TensorFactory implementation
grep -n "create_local_tensor\|createLocalTensor" src/tensors/*.{h,cpp}
```

---

### 3. **MPI Communication Buffers** (MEDIUM PRIORITY ⭐⭐)

**Location:** `src/operators/MPIAttentionOperator.cpp:1999-2000`

```cpp
// First call or non-predictable growth (prefill): gather counts
recvcounts_kv.resize(world_size);  // ← Small allocation
displs_kv.resize(world_size);      // ← Small allocation
```

**Characteristics:**
- **Size:** `world_size × sizeof(int)` = Typically 8 bytes (2 ranks) to 64 bytes (16 ranks)
- **Access Pattern:** Read-only during MPI collectives (Allgather)
- **NUMA Sensitivity:** **LOW** - tiny allocations fit in cache
- **Lifecycle:** Allocated during first attention operation, cached for decode steps
- **Current Allocation:** Default `std::vector::resize()`

**Recommendation:**
- **Action:** No optimization needed - too small to benefit from NUMA
- **Rationale:** Allocations are <1KB, completely cache-resident
- **Priority:** **SKIP**

---

### 4. **RMSNorm Thread-Local Scratch Buffers** (MEDIUM PRIORITY ⭐⭐)

**Location:** `src/operators/common/RmsnormCore.cpp:419-420`

```cpp
void rmsnorm_execute(const float *src, const float *gamma, float *dst,
                     std::size_t rows, std::size_t cols, float epsilon,
                     GammaMode mode, std::size_t gamma_offset) {
    // ...
    thread_local RMSNormScratch tls_scratch;
    if (env.scratch_prealloc_rows > 0 && tls_scratch.row_sumsq.capacity() < (size_t)env.scratch_prealloc_rows) {
        tls_scratch.row_sumsq.reserve(env.scratch_prealloc_rows);  // ← TLS allocation
        tls_scratch.inv.reserve(env.scratch_prealloc_rows);        // ← TLS allocation
    }
    tls_scratch.ensure(rows);
    // ...
}
```

**Characteristics:**
- **Size:** `env.scratch_prealloc_rows × sizeof(float)` per thread
  - Default: 0 (disabled), configurable via `LLAMINAR_RMSNORM_SCRATCH_PREALLOC`
  - If enabled: ~128-512 rows × 4 bytes = 512 bytes - 2KB per thread
- **Access Pattern:** Read/write intensive during RMSNorm computation
- **NUMA Sensitivity:** **MEDIUM** - thread-local, benefits from core affinity
- **Lifecycle:** Thread-local static, allocated lazily on first use per thread
- **Current Allocation:** `std::vector::reserve()` + `std::vector::resize()`

**Recommendation:**
- **Action:** Potentially beneficial if `scratch_prealloc_rows` is large (>1024)
- **Approach:** Thread-local allocations are likely already NUMA-local due to thread affinity
- **Priority:** **LOW-MEDIUM** - only matters if scratch buffers are enabled and large
- **Note:** Thread-local storage inherently benefits from core affinity if OpenMP threads are pinned

**Verdict:** Likely already NUMA-optimal due to thread affinity. Skip unless profiling shows issues.

---

### 5. **SwiGLU Activation Temporary Buffers** (LOW PRIORITY ⭐)

**Location:** `src/QwenPipeline.cpp:1700` (within layer execution)

```cpp
// SwiGLU activation: out = silu(gate) * up
auto activation_fn = [](const std::vector<float>& gate, 
                       const std::vector<float>& up) -> std::vector<float> {
    std::vector<float> out;
    out.resize(up.size());  // ← Temporary buffer
    for (size_t i = 0; i < up.size(); ++i) {
        float silu = gate[i] / (1.0f + std::exp(-gate[i]));
        out[i] = silu * up[i];
    }
    return out;
};
```

**Characteristics:**
- **Size:** `hidden_dim × sizeof(float)` = Varies by model
  - Qwen 0.5B: 1536 × 4 bytes = 6KB
  - Qwen 7B: 11008 × 4 bytes = 44KB
- **Access Pattern:** Single-use temporary, immediately discarded
- **NUMA Sensitivity:** **LOW** - short-lived, likely cache-resident
- **Lifecycle:** Allocated/deallocated per layer per forward pass
- **Current Allocation:** Default `std::vector::resize()`

**Recommendation:**
- **Action:** No optimization needed - too small and short-lived
- **Rationale:** Temporary buffers are cache-resident and deallocated immediately
- **Alternative:** Consider reusing a pre-allocated buffer if profiling shows allocation overhead
- **Priority:** **SKIP**

---

## Summary Table

| Allocation Site | Location | Size | Access Freq | NUMA Sensitivity | Priority | Action |
|-----------------|----------|------|-------------|------------------|----------|--------|
| **RoPE Frequency Tables** | MPIRoPEOperator.cpp:348-349 | 1-8MB | Every token | Very High | ⭐⭐⭐ | **Verify (likely already optimal)** |
| **K/V Cache Structures** | QwenPipeline.cpp:1523<br/>OpenblasPrefillProvider.cpp:50 | 96MB-4GB | Every layer | Extreme | ⭐⭐⭐ | **Investigate TensorFactory** |
| **MPI Comm Buffers** | MPIAttentionOperator.cpp:1999 | <1KB | Per prefill | Low | ⭐ | Skip (too small) |
| **RMSNorm TLS Scratch** | RmsnormCore.cpp:419 | 512B-2KB | Every norm | Medium | ⭐⭐ | Skip (thread-affinity handles) |
| **SwiGLU Temp Buffers** | QwenPipeline.cpp:1700 | 6-44KB | Every layer | Low | ⭐ | Skip (short-lived) |

---

## Recommendations

### Immediate Actions

1. **Investigate K/V Cache Allocation (CRITICAL):**
   ```bash
   # Check if TensorFactory::create_local_tensor uses NUMA allocation
   grep -rn "create_local_tensor\|createLocalTensor" src/tensors/
   ```
   - If not NUMA-aware, this is **the highest impact optimization** (potentially +10-40%)
   - K/V cache is the largest allocation in the system (multi-GB for large models)

2. **Verify RoPE Table Locality (LOW EFFORT):**
   ```bash
   # Run with NUMA verification enabled
   export LLAMINAR_NUMA_VERIFY_LOCALITY=1
   ./run_llaminar.sh --benchmark -m model.gguf -p "test" -n 10
   numastat -p $(pgrep llaminar)
   ```
   - Check if `cos_table_` and `sin_table_` are on correct NUMA nodes
   - Likely already optimal, but worth confirming

### Future Optimizations (if profiling indicates)

3. **Consider K/V Cache Pre-warming:**
   - If K/V cache allocation shows NUMA locality issues, implement parallel first-touch:
   ```cpp
   #pragma omp parallel for
   for (size_t i = 0; i < cache_size; ++i) {
       k_cache_data[i] = 0.0f;
       v_cache_data[i] = 0.0f;
   }
   ```

4. **Monitor TLS Scratch Buffer Affinity:**
   - Thread-local buffers should automatically benefit from OpenMP thread pinning
   - Verify with `numastat` if RMSNorm shows performance issues

### Skip (Not Worth Optimizing)

- MPI communication buffers (too small)
- SwiGLU temporary buffers (short-lived, cache-resident)

---

## Expected Performance Impact

### Small Models (Qwen 0.5B, 645MB)
- **Current:** K/V cache = 96MB, mostly cache-resident in Release builds
- **Expected Impact:** +1-3% (minimal, already fast)
- **Rationale:** Cache-resident, NUMA overhead minimal

### Large Models (Qwen 7B, 13B)
- **Current:** K/V cache = 2-4GB, exceeds L3 cache significantly
- **Expected Impact:** +10-40% (substantial, if not NUMA-aware)
- **Rationale:** Multi-GB cache thrashing, remote NUMA access = 2-3x latency penalty

---

## Implementation Priority

1. **Phase 1 (CRITICAL):** Investigate and fix K/V cache NUMA locality
   - Check `TensorFactory::create_local_tensor()` implementation
   - Apply NUMA first-touch if needed
   - Test with 7B/13B models for maximum impact

2. **Phase 2 (VERIFICATION):** Confirm RoPE table locality
   - Run `numastat` validation
   - Document findings (likely already optimal)

3. **Phase 3 (OPTIONAL):** Re-evaluate TLS scratch buffers
   - Only if profiling shows RMSNorm performance issues
   - Likely already optimal due to thread affinity

---

## Next Steps

```bash
# 1. Investigate TensorFactory implementation
grep -rn "create_local_tensor\|createLocalTensor" src/tensors/

# 2. If not NUMA-aware, implement first-touch in K/V cache allocation
# 3. Test with large model (7B) to measure impact
# 4. Run numastat validation on RoPE tables

# 5. Benchmark before/after with large prefill
./run_llaminar.sh --benchmark -m qwen-7b.gguf \
  -p "$(python -c 'print("test " * 512)')" -n 50
```

---

## Related Work

- **20251015_numa_first_touch_implementation.md** - ModelLoader NUMA optimization (15 sites)
- **20251015_numa_performance_test_results.md** - Debug build +3.1% improvement
- **20251015_release_build_numa_performance_results.md** - Release build -4.2% (workload-dependent)

---

## Conclusion

**K/V cache allocation is the critical path for NUMA optimization beyond ModelLoader.** Investigating `TensorFactory::create_local_tensor()` should be the immediate next step, as it manages the largest allocations (multi-GB) with the highest access frequency (every layer, every token). Other allocations are either:
- Already NUMA-optimal (RoPE tables, TLS buffers due to initialization patterns and thread affinity)
- Too small to matter (MPI buffers, temporary activations)

Expected impact: **+1-3% for small models, +10-40% for large models** if K/V cache is not currently NUMA-aware.
