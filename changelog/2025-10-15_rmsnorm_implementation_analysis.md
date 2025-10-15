# RMSNorm Implementation Analysis: RmsnormT5 vs RmsnormCore
**Date:** October 15, 2025  
**Analysis for:** Potential harmonization of dual RMSNorm implementations

## Executive Summary

The codebase currently has **two distinct RMSNorm implementations** serving different purposes:
- **RmsnormT5**: Simple, PyTorch-compatible implementation (179 lines)
- **RmsnormCore**: Advanced, production-optimized implementation (419 lines)

**Recommendation**: **DO NOT MERGE** - They serve complementary roles. Consider refactoring RmsnormT5 to use RmsnormCore internally to reduce code duplication while preserving the simple API.

---

## Implementation Comparison

### RmsnormT5.cpp/h (179 lines)
**Location**: `src/operators/common/RmsnormT5.{cpp,h}`

**Purpose**: Reference implementation matching HuggingFace Transformers T5LayerNorm exactly

**API Surface**:
```cpp
// Single-precision accumulation (matches PyTorch default)
void rmsnorm_t5_forward(const float *input, const float *weight, float *output,
                       size_t rows, size_t cols, float eps = 1e-6f, 
                       bool use_parallel = true);

// Double-precision accumulation (improved numerical stability)
void rmsnorm_t5_forward_double_acc(const float *input, const float *weight, float *output,
                                   size_t rows, size_t cols, float eps = 1e-6f,
                                   bool use_parallel = true);
```

**Characteristics**:
- ✅ **Simple**: Single-function call, minimal API surface
- ✅ **PyTorch-compatible**: Matches T5LayerNorm formula exactly
- ✅ **Self-contained**: No external state or configuration
- ❌ **Limited optimization**: Basic OpenMP parallelization only
- ❌ **No SIMD**: No vectorization beyond compiler auto-vectorization
- ❌ **Fixed algorithm**: Cannot customize threading/execution strategy

**Implementation Details**:
```cpp
// T5 formula: output = weight * input / sqrt(mean(input^2) + eps)
float sum_sq = 0.0f;
for (size_t c = 0; c < cols; ++c) {
    float val = input_row[c];
    sum_sq += val * val;
}
float variance = sum_sq / static_cast<float>(cols);
float inv_rms = 1.0f / std::sqrt(variance + eps);
for (size_t c = 0; c < cols; ++c) {
    output_row[c] = weight[c] * input_row[c] * inv_rms;
}
```

---

### RmsnormCore.cpp/h (419 lines)
**Location**: `src/operators/common/RmsnormCore.{cpp,h}`

**Purpose**: Production-grade, highly optimized RMSNorm primitives with modular design

**API Surface**:
```cpp
// Modular primitives (composable)
void rmsnorm_compute_row_sumsq(const float *src, size_t rows, size_t cols,
                               double *row_sumsq, const RMSNormExecOptions &opts = {});

void rmsnorm_compute_inv(const double *row_sumsq, size_t rows, size_t cols,
                         float epsilon, float *inv_out);

void rmsnorm_apply(const float *src, const float *gamma, const float *inv,
                   size_t rows, size_t cols, float *dst,
                   GammaMode mode = GammaMode::REPLICATED,
                   size_t gamma_offset = 0, const RMSNormExecOptions &opts = {});

// Convenience fused API
void rmsnorm_row_major_fused(const float *src, const float *gamma, float *dst,
                             size_t rows, size_t cols, float epsilon,
                             GammaMode mode = GammaMode::REPLICATED,
                             size_t gamma_offset = 0,
                             const RMSNormExecOptions &opts = {});

// With external scratch buffer (avoids allocations)
void rmsnorm_row_major_fused(const float *src, const float *gamma, float *dst,
                             size_t rows, size_t cols, float epsilon,
                             RMSNormScratch &scratch, /* ... */);
```

**Characteristics**:
- ✅ **Highly optimized**: AVX2/AVX512 SIMD paths with multi-accumulator unrolling
- ✅ **Modular**: Separate primitives for sum-of-squares, scaling, application
- ✅ **Flexible**: Configurable threading, SIMD strategies, gamma sharding modes
- ✅ **Memory-efficient**: Scratch buffer reuse to avoid repeated allocations
- ✅ **MPI-aware**: Supports sharded gamma weights for distributed execution
- ✅ **Debug instrumentation**: Environment-controlled SIMD selection, validation hooks
- ❌ **Complex**: 419 lines, multiple configuration options, learning curve
- ❌ **Over-engineered for simple cases**: Overkill when basic RMSNorm suffices

**Optimization Features**:
1. **AVX512 path**: Processes 64 elements per iteration with 4-way accumulator unrolling
2. **AVX2 fallback**: 32 elements per iteration for older CPUs
3. **Double precision accumulation**: Reduces numerical errors on long sequences
4. **Adaptive parallelization**: Heuristics to avoid OpenMP overhead on small inputs
5. **SIMD selection**: Runtime environment control (`LLAMINAR_RMSNORM_VEC_IMPL`)
6. **Gamma sharding**: `GammaMode::SHARDED` for feature-partitioned weights

**Configuration Options**:
```cpp
struct RMSNormExecOptions {
    bool allow_parallel = true;                  // Enable OpenMP
    bool force_scalar = false;                   // Force scalar path
    size_t parallel_threshold_elems = 8192;      // Parallelization threshold
};

enum class GammaMode {
    REPLICATED,  // Standard: all ranks have full gamma
    SHARDED      // Distributed: gamma partitioned across MPI ranks
};
```

---

## Usage Analysis

### Where RmsnormT5 is Used

#### 1. **MPIRMSNormOperator** (Production)
**File**: `src/operators/MPIRMSNormOperator.cpp:508`  
**Context**: Distributed RMSNorm computation across MPI ranks  
```cpp
// Use double precision accumulation for distributed normalization
kernels::rmsnorm_t5_forward_double_acc(
    local_input, weight, local_output,
    local_seq_len, hidden_size, epsilon_, use_parallel);
```
**Rationale**: Uses T5 variant for guaranteed PyTorch parity in distributed context

#### 2. **CosmaPrefillManager** (Prefill Path)
**File**: `src/CosmaPrefillManager.cpp:983`  
**Context**: Fused RMSNorm + QKV projection for COSMA-accelerated prefill  
```cpp
llaminar::kernels::rmsnorm_t5_forward(
    activation_row_major, gamma, normalized->data(),
    seq_len, hidden_size, eps, true);
```
**Rationale**: Simple, proven primitive for activation normalization before distributed matmuls

#### 3. **Architecture Documentation**
**File**: `.github/instructions/llaminar-architecture.instructions.md:2099`  
**Context**: Reference implementation example  

**Summary**: Used in **2 production locations** where PyTorch compatibility is critical

---

### Where RmsnormCore is Used

#### 1. **MPIRMSNormOperator** (Sharded Path)
**File**: `src/operators/MPIRMSNormOperator.cpp:258`  
**Context**: Feature-sharded RMSNorm requiring cross-rank reduction  
```cpp
kernels::rmsnorm_compute_row_sumsq(in_ptr, local_seq_len, feat_dim, 
                                   local_row_sumsq.data(), opts);
// ... MPI_Allreduce ...
// Then apply with sharded gamma
```
**Rationale**: Modular API allows MPI reduction between sum-of-squares and application phases

#### 2. **QwenPipeline Fallback Path**
**File**: `src/QwenPipeline.cpp:1661`  
**Context**: CPU-only fallback inference (non-COSMA path)  
```cpp
auto rmsnorm = [&](std::vector<float> &mat, const float *wn) {
    kernels::RMSNormExecOptions opts;
    kernels::rmsnorm_row_major_fused(mat.data(), wn, mat.data(),
        (size_t)seq_len, (size_t)config_.getLayerConfig().d_model,
        config_.getLayerConfig().eps, kernels::GammaMode::REPLICATED, 0, opts);
};
```
**Rationale**: High-performance local computation with SIMD optimization

#### 3. **Unit Tests**
**File**: `tests/TestRmsnormCoreCorrectness.cpp:81`  
**Context**: Correctness validation tests  
```cpp
rmsnorm_row_major_fused(src.data(), gamma.data(), dst.data(), 
                        rows, cols, 1e-5f, GammaMode::REPLICATED, 0, {});
```

**Summary**: Used in **3 production locations** requiring advanced features (MPI, SIMD, sharding)

---

## Functional Overlap and Divergence

### Overlap
Both implementations compute the same mathematical operation:
```
output[i,j] = gamma[j] * input[i,j] / sqrt(mean(input[i,:]^2) + eps)
```

### Key Differences

| Aspect | RmsnormT5 | RmsnormCore |
|--------|-----------|-------------|
| **Lines of code** | 179 | 419 |
| **API complexity** | 2 functions | 7 functions |
| **SIMD support** | None (compiler auto-vec) | AVX2/AVX512 manual |
| **Accumulator precision** | Float32 or Double | Always Double for sum-of-squares |
| **Gamma sharding** | ❌ No | ✅ Yes (MPI distribution) |
| **Scratch buffer reuse** | ❌ No | ✅ Yes (avoid allocations) |
| **Modular primitives** | ❌ No | ✅ Yes (3-stage pipeline) |
| **Configuration** | `use_parallel` flag | `RMSNormExecOptions` struct |
| **Thread safety** | Basic OpenMP | Nested parallel detection |
| **Debug hooks** | Minimal logging | Environment-controlled SIMD |
| **Primary use case** | PyTorch parity | Production performance |

---

## Harmonization Options

### Option 1: Keep Both (Current State) ✅ RECOMMENDED
**Pros**:
- ✅ RmsnormT5 remains simple for validation/testing
- ✅ RmsnormCore optimized for production without compromise
- ✅ Clear separation of concerns (compatibility vs performance)

**Cons**:
- ❌ Code duplication (~179 lines)
- ❌ Maintenance burden (two implementations to update)

### Option 2: Reimplement RmsnormT5 Using RmsnormCore 🤔 VIABLE
**Approach**: 
```cpp
void rmsnorm_t5_forward(const float *input, const float *weight, float *output,
                       size_t rows, size_t cols, float eps, bool use_parallel) {
    RMSNormExecOptions opts;
    opts.allow_parallel = use_parallel;
    rmsnorm_row_major_fused(input, weight, output, rows, cols, eps, 
                           GammaMode::REPLICATED, 0, opts);
}
```

**Pros**:
- ✅ Eliminates duplication
- ✅ RmsnormT5 API preserved (no caller changes)
- ✅ Automatically benefits from RmsnormCore optimizations
- ✅ Single implementation to maintain

**Cons**:
- ⚠️ RmsnormCore's double precision accumulation may differ from T5's float32 default
- ⚠️ Introduces dependency (T5 now depends on Core)
- ⚠️ May change numerical results slightly (needs validation)

### Option 3: Deprecate RmsnormT5, Migrate All Callers ❌ NOT RECOMMENDED
**Pros**:
- ✅ Single implementation

**Cons**:
- ❌ Breaks PyTorch parity guarantee (double vs float accumulation)
- ❌ Requires updating all callers with new API
- ❌ More complex API for simple use cases
- ❌ Loses reference implementation for validation

---

## Recommendation

### Short Term: **Keep Both Implementations**
Rationale:
1. **Different design goals**: T5 for compatibility, Core for performance
2. **Limited overlap**: Only 2 call sites for T5 vs 3 for Core
3. **Risk of regression**: Changing T5 may break PyTorch parity tests
4. **Low maintenance cost**: 179 lines is trivial, rarely changes

### Medium Term: **Refactor T5 to Use Core Internally**
Implementation plan:
1. Add `rmsnorm_t5_compat_mode` flag to `RMSNormExecOptions`
2. When enabled, use float32 accumulation to match PyTorch exactly
3. Reimplement `rmsnorm_t5_forward()` as thin wrapper around `rmsnorm_row_major_fused()`
4. Validate with parity tests (no numerical changes)
5. Keep simple T5 API surface for callers

**Benefits**:
- Maintains PyTorch compatibility guarantees
- Reduces duplication to ~20 lines (just API wrapper)
- Inherits SIMD optimizations from Core (performance win!)
- Single code path to debug/maintain

**Code Sketch**:
```cpp
// In RmsnormCore.h
struct RMSNormExecOptions {
    bool allow_parallel = true;
    bool force_scalar = false;
    size_t parallel_threshold_elems = 8192;
    bool t5_compat_mode = false;  // NEW: Use float32 accumulation
};

// In RmsnormT5.cpp (simplified to wrapper)
void rmsnorm_t5_forward(const float *input, const float *weight, float *output,
                       size_t rows, size_t cols, float eps, bool use_parallel) {
    RMSNormExecOptions opts;
    opts.allow_parallel = use_parallel;
    opts.t5_compat_mode = true;  // Force float32 for PyTorch parity
    rmsnorm_row_major_fused(input, weight, output, rows, cols, eps,
                           GammaMode::REPLICATED, 0, opts);
}
```

### Long Term: **Document Usage Patterns**
Add comments to each call site explaining why T5 vs Core was chosen:
```cpp
// Use T5 variant: guarantees PyTorch parity for distributed normalization
kernels::rmsnorm_t5_forward_double_acc(local_input, weight, local_output, ...);

// Use Core variant: needs sharded gamma for MPI feature distribution
kernels::rmsnorm_compute_row_sumsq(in_ptr, local_seq_len, ...);
```

---

## Migration Safety Checklist

If proceeding with Option 2 (wrapper refactor):

- [ ] Add `t5_compat_mode` to `RMSNormExecOptions`
- [ ] Implement float32 accumulation path in RmsnormCore
- [ ] Rewrite `rmsnorm_t5_forward()` as 5-line wrapper
- [ ] Run **ParityFrameworkTest** (all 6 tests must pass)
- [ ] Run **RMSNormCoreParity** unit tests
- [ ] Compare outputs on 100-token sequence (max diff < 1e-7)
- [ ] Profile performance: ensure no regression vs current T5
- [ ] Update documentation to explain wrapper design
- [ ] Tag release with clear migration notes

---

## Conclusion

**Current state is acceptable**: Two implementations serve distinct purposes with minimal overlap.

**Best path forward**: Refactor RmsnormT5 to be a thin compatibility wrapper around RmsnormCore, preserving PyTorch parity while eliminating code duplication and inheriting SIMD optimizations.

**Do NOT merge blindly**: RmsnormT5's float32 accumulation is intentional for PyTorch compatibility. Any harmonization must preserve this semantic guarantee.
