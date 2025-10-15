# Performance Investigation: Single-Threading Bottleneck Analysis
**Date:** October 15, 2025  
**Issue:** Operations appear to run single-threaded despite OMP_NUM_THREADS=28

## System Configuration

### Detected Topology (from run_llaminar.sh)
```
System: 2 sockets, 28 cores/socket, 2 NUMA nodes
Topology: 56 physical cores, 112 logical cores
Hyperthreading: Yes (2 threads/core)
OpenMP: 28 threads/socket, sockets placement, close binding
MPI: 2 processes
BLAS: OpenBLAS threads=28 policy=match_omp
```

**Environment Variables Set:**
- `OMP_NUM_THREADS=28` ✅
- `OMP_PLACES=sockets` ✅
- `OMP_PROC_BIND=close` ✅
- `OPENBLAS_NUM_THREADS=28` ✅

## Root Causes Identified

### 🔴 **CRITICAL: OpenBLAS Thread Cap at 8**

**Location**: `src/ProductionAdaptiveMatmul.h:233`

```cpp
bool multiply_multi_threaded_openblas(...) {
    // Use optimal thread count (usually number of cores)
    int optimal_threads = std::min(omp_get_max_threads(), 8); // ⚠️ Cap at 8 threads
    int old_threads = openblas_get_num_threads();
    openblas_set_num_threads(optimal_threads);
    // ...
}
```

**Impact**: 
- Even though `OMP_NUM_THREADS=28`, OpenBLAS is capped at **8 threads maximum**
- On a 56-core system, this leaves **48 physical cores idle** during GEMM operations
- This affects ALL multi-threaded OpenBLAS paths

**Why This Exists**: 
Likely added as a conservative default to prevent oversubscription in early testing, but never updated for production systems with high core counts.

---

### 🟡 **HIGH: Operation Size Thresholds May Be Too Conservative**

**Location**: `src/MatmulBackendSelection.h:108-112`

```cpp
static constexpr size_t TINY_OP_THRESHOLD = 8192;            // < 8K elements = single-threaded
static constexpr size_t SMALL_OP_THRESHOLD = 1048576;        // < 1M elements = multi-threaded local  
static constexpr size_t PREFILL_COSMA_SEQ_THRESHOLD = 4096;  // Min seq length for COSMA
static constexpr size_t LARGE_PREFILL_THRESHOLD = 8388608;   // >= 8M elements for distributed
```

**Impact**:
- Operations < 8K elements (e.g., 1x896x896 = 803,456 elements) → **single-threaded**
- Operations between 8K-1M elements → multi-threaded **but capped at 8 threads**
- Only operations > 8M elements considered for distributed execution

**Typical Decode Shapes**:
- QKV projections: 1 × 896 × 896 = **803,456 elements** → **SINGLE-THREADED** ⚠️
- Attention output: 1 × 896 × 896 = **803,456 elements** → **SINGLE-THREADED** ⚠️
- FFN gate/up: 1 × 896 × 4864 = **4,357,544 elements** → multi-threaded (but capped at 8)

**Analysis**: The 8K threshold is **way too high** for modern hardware. On a 56-core system, we can efficiently parallelize much smaller operations.

---

### 🟢 **MEDIUM: RMSNorm Parallel Threshold**

**Location**: `src/operators/common/RmsnormCore.h:31`

```cpp
struct RMSNormExecOptions {
    bool allow_parallel = true;
    bool force_scalar = false;
    std::size_t parallel_threshold_elems = 8192;  // rows*cols threshold
};
```

**Impact**:
- RMSNorm operations < 8K elements run single-threaded
- Typical decode: 1 row × 896 cols = **896 elements** → **SINGLE-THREADED**
- Even 8-token batch: 8 × 896 = 7,168 elements → **SINGLE-THREADED**

**Heuristic Logic** (`src/operators/common/RmsnormCore.cpp:22-32`):
```cpp
static inline bool want_parallel(std::size_t rows, std::size_t cols, const RMSNormExecOptions &opts) {
    if (opts.force_scalar || !opts.allow_parallel)
        return false;
    std::size_t elems = rows * cols;
    if (rows <= 1)  // ⚠️ Single-row path always serial!
        return false;
    #ifdef _OPENMP
    if (omp_in_parallel())  // Avoid nested parallelism
        return false;
    #endif
    return elems >= opts.parallel_threshold_elems;
}
```

**Additional Issue**: The `rows <= 1` check **always disables parallelism for decode** (which processes 1 token at a time).

**Analysis**: This makes sense for tiny operations, but the threshold should be lower (e.g., 1024 or 2048), and the single-row restriction should allow parallelization over columns for large models.

---

## Threading Audit Results

### OpenBLAS Thread Management
| Function | Thread Count | Source File | Notes |
|----------|--------------|-------------|-------|
| `multiply_single_threaded_openblas()` | **1** | ProductionAdaptiveMatmul.h:202 | Explicit single-threading |
| `multiply_multi_threaded_openblas()` | **min(omp_get_max_threads(), 8)** | ProductionAdaptiveMatmul.h:233 | ⚠️ CAPPED AT 8 |
| `multiply_distributed_openblas()` | Varies | ProductionAdaptiveMatmul.h:302 | MPI-distributed |

### Backend Selection Decision Tree

**For Typical Decode (1 token, d_model=896)**:

```
Input: m=1, n=896, k=896 (QKV projection)
Total elements: 1 × 896 × 896 = 803,456

1. Distributed partition? → NO
2. MPI size? → 2 ranks
3. Total elements < TINY_OP_THRESHOLD (8192)? → NO (803,456 > 8192)
4. Total elements < SMALL_OP_THRESHOLD (1M)? → YES (803,456 < 1M)
5. Stage = Decode
6. Decision: MULTI_THREADED_OPENBLAS

→ Calls multiply_multi_threaded_openblas()
→ Sets openblas_set_num_threads(min(28, 8)) = 8
→ **Only uses 8 threads instead of 28!**
```

---

## Recommended Fixes (Priority Order)

### 1. **CRITICAL: Remove OpenBLAS 8-Thread Cap**

**File**: `src/ProductionAdaptiveMatmul.h:233`

**Current**:
```cpp
int optimal_threads = std::min(omp_get_max_threads(), 8); // Cap at 8 threads
```

**Proposed**:
```cpp
// Use all available threads (already set via OPENBLAS_NUM_THREADS env var)
// If we need a cap, make it environment-configurable
int max_blas_threads = env.adaptive.max_blas_threads > 0 
    ? env.adaptive.max_blas_threads 
    : omp_get_max_threads();
int optimal_threads = std::min(omp_get_max_threads(), max_blas_threads);
```

**Impact**: Immediate 3.5x parallelism increase (8 → 28 threads on our system)

---

### 2. **HIGH: Lower TINY_OP_THRESHOLD**

**File**: `src/MatmulBackendSelection.h:108`

**Current**:
```cpp
static constexpr size_t TINY_OP_THRESHOLD = 8192;  // Total elements
```

**Proposed**:
```cpp
static constexpr size_t TINY_OP_THRESHOLD = 2048;  // ~1.4K × 1.4K matrix
```

**Rationale**: 
- Modern CPUs can efficiently parallelize operations as small as 2K elements
- 8K threshold (89×89 matrix) is too conservative for 56-core systems
- 2K still avoids threading overhead for truly tiny operations

---

### 3. **MEDIUM: Reduce RMSNorm Parallel Threshold**

**File**: `src/operators/common/RmsnormCore.h:31`

**Current**:
```cpp
std::size_t parallel_threshold_elems = 8192;
```

**Proposed**:
```cpp
std::size_t parallel_threshold_elems = 2048;  // Lower for better parallelization
```

**Rationale**: Matches the adjusted TINY_OP_THRESHOLD, allows 8-token batches to parallelize.

---

### 4. **MEDIUM: Allow Column Parallelization for Single-Row RMSNorm**

**File**: `src/operators/common/RmsnormCore.cpp:want_parallel()`

**Current**:
```cpp
if (rows <= 1)
    return false;  // Decode always single-threaded
```

**Proposed**:
```cpp
if (rows <= 1) {
    // For single-row decode, allow column parallelization if d_model is large
    return cols >= 2048;  // e.g., Qwen-7B has d_model=4096
}
```

**Rationale**: Large language models have d_model ≥ 2048. Even single-row normalization can benefit from column-wise parallelization.

---

### 5. **LOW: Add Runtime Threading Diagnostics**

Add logging to show actual thread counts during key operations:

```cpp
// In MatMulBackendSelector::executeBackend()
LOG_DEBUG("GEMM[" << ctx.stage_name() << "] " << m << "×" << n << "×" << k 
          << " -> " << backendName(decision.backend)
          << " threads=" << openblas_get_num_threads() 
          << " omp_threads=" << omp_get_max_threads());
```

**Rationale**: Makes threading behavior visible for debugging and validation.

---

## Validation Plan

### Phase 1: Quick Win - Remove 8-Thread Cap
1. Modify `ProductionAdaptiveMatmul.h` to remove/raise cap
2. Rebuild: `cmake --build build --parallel`
3. Run single decode: `./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello" --no-warmup -n 5`
4. Monitor with `top` or `htop` - should see ~28 threads active during GEMM operations
5. Benchmark: Compare tokens/sec before vs after

### Phase 2: Threshold Tuning
1. Lower TINY_OP_THRESHOLD to 2048
2. Lower RMSNorm threshold to 2048
3. Run decode benchmark suite
4. Measure: tokens/sec improvement, CPU utilization %

### Phase 3: Single-Row RMSNorm Parallelization
1. Add column parallelization logic
2. Run parity tests (ensure no numerical regressions)
3. Profile decode performance

### Phase 4: Add Diagnostics
1. Implement thread count logging
2. Run full inference
3. Analyze logs to find remaining bottlenecks

---

## Expected Performance Impact

### Current State (Pessimistic Estimate)
- **OpenBLAS**: 8 threads (28% of available cores)
- **RMSNorm**: 1 thread (decode) or 28 threads (prefill with rows > 1)
- **Overall Utilization**: ~15-30% of CPU capacity

### After Fix #1 (Remove 8-Thread Cap)
- **OpenBLAS**: 28 threads (100% of socket)
- **Expected Speedup**: **2.5-3.5x** for GEMM-dominated workloads
- **Tokens/sec**: Likely **3-4x improvement** for decode

### After All Fixes
- **OpenBLAS**: 28 threads
- **RMSNorm**: Parallelized where beneficial
- **Expected Speedup**: **4-5x** overall for single-token decode
- **Batch Decode**: Even larger improvements (more operations hit parallel paths)

---

## Next Steps

1. ✅ Document findings (this file)
2. **Implement Fix #1** (remove 8-thread cap) - **highest priority**
3. Benchmark before/after
4. If successful, proceed with threshold tuning
5. Add comprehensive threading diagnostics
6. Create performance baseline document

---

## Additional Notes

### Why Wasn't This Caught Earlier?

1. **Tests run with small models**: Most unit tests use tiny matrices where single-threading is actually appropriate
2. **Parity tests prioritize correctness over speed**: Pass/fail based on numerical accuracy, not performance
3. **No automated performance regression tests**: We don't currently track tokens/sec in CI

### Monitoring Recommendations

**During Development**:
```bash
# Watch thread activity in real-time
htop -p $(pgrep -d',' llaminar)

# Or use perf to see CPU utilization per function
perf record -g ./build/llaminar -m models/... -p "test"
perf report
```

**For Benchmarking**:
```bash
# Run with timing instrumentation
LLAMINAR_LOG_LEVEL=INFO ./run_llaminar.sh -m models/qwen... -p "benchmark" -n 100

# Extract tokens/sec from logs
grep "tokens/sec" llaminar.log
```

---

## References

- OpenBLAS threading docs: https://github.com/xianyi/OpenBLAS/wiki/faq#multi-threaded
- Intel MKL threading: https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/techniques-to-set-the-number-of-threads.html
- OpenMP best practices: https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf
