# Q8_1 Activation Performance Fix - Handover Document

**Date**: December 24, 2024  
**Branch**: `feature/typed-residuals`  
**Issue**: Q8_1 decode was 9.1x slower than FP32 (4.3 tok/s vs 39.69 tok/s)

---

## Problem Summary

When using `--activation-precision q8_1`, decode throughput was severely degraded compared to FP32. Initial profiling showed only ~45ms of kernel time but decode took ~1150ms per iteration—96% of time was unaccounted for in kernel profiling.

---

## Root Cause

**SwiGLU stage was processing the entire pre-allocated buffer (2048 elements) instead of the actual token count (1 for decode).**

In `ComputeStage.cpp`, `SwiGLUStage::execute()` was using `params_.gate->rows()` to determine `seq_len`, which returns the buffer capacity (2048) rather than the actual sequence length being processed.

For decode (single token generation), this meant:
- SwiGLU processed 2048x more elements than necessary
- Per-layer time: **12-13ms** instead of **0.02-0.04ms**
- 24 layers × 12ms = ~288ms wasted per decode step

---

## Files Modified

### 1. `src/v2/execution/ComputeStage.cpp` (CRITICAL FIX)

**Before (BUG):**
```cpp
const int seq_len = static_cast<int>(params_.gate->rows());
```

**After (FIX):**
```cpp
const int seq_len = (params_.seq_len > 0)
                        ? params_.seq_len
                        : static_cast<int>(params_.gate->rows());
```

This ensures SwiGLU uses the explicit `seq_len` parameter when provided, falling back to buffer dimensions only when `seq_len` is not set.

### 2. `src/v2/pipelines/qwen/GraphOrchestrator.cpp`

Wired up `LLAMINAR_EXECUTOR_PROFILING` environment variable to DeviceGraphExecutor. Both constructors now include:

```cpp
exec_config.enable_profiling = graph_builder_->config().enable_profiling 
                             || env.execution.executor_profiling;
```

### 3. `src/v2/kernels/cpu/attention/FusedAttentionWoKernel.h`

Added kernel profiling for ATTENTION operations:
```cpp
#include "utils/KernelProfiler.h"
// ... in execute():
KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);
```

### 4. `tests/v2/integration/Test__Q8_1_FusedAttentionWo_Prefill_TP.cpp` (NEW)

Created integration test for Q8_1 prefill with tensor parallelism. Tests 5 sequence lengths (8/32/64/128/256 tokens) with 2 MPI ranks, validates cosine similarity >0.99 against FP32 reference.

---

## Environment Variables for Profiling

| Variable | Purpose |
|----------|---------|
| `LLAMINAR_PROFILE_KERNELS=1` | Per-kernel timing breakdown (GEMM, ATTENTION, etc.) |
| `LLAMINAR_EXECUTOR_PROFILING=1` | Per-stage timing in DeviceGraphExecutor (swiglu, rope, etc.) |
| `LLAMINAR_LOG_LEVEL=DEBUG` | Detailed logging output |

### Example Profiling Commands

```bash
# Kernel-level profiling
LLAMINAR_PROFILE_KERNELS=1 ./build_v2_release/llaminar2 --benchmark \
  --activation-precision q8_1 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -n 5 -t 0

# Stage-level profiling (found the SwiGLU bug)
LLAMINAR_EXECUTOR_PROFILING=1 ./build_v2_release/llaminar2 --benchmark \
  --activation-precision q8_1 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -n 5 -t 0

# Combined profiling
LLAMINAR_PROFILE_KERNELS=1 LLAMINAR_EXECUTOR_PROFILING=1 ./build_v2_release/llaminar2 \
  --benchmark --activation-precision q8_1 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -n 5 -t 0
```

---

## Performance Results

### Before Fix
| Metric | FP32 | Q8_1 | Ratio |
|--------|------|------|-------|
| Decode | 39.69 tok/s | 4.3 tok/s | 9.1x slower |
| Prefill | ~210 tok/s | ~170 tok/s | 1.2x slower |

### After Fix
| Metric | FP32 | Q8_1 | Ratio |
|--------|------|------|-------|
| Decode | 43.97 tok/s | 12.65 tok/s | 3.5x slower |
| Prefill | 210.36 tok/s | 412.86 tok/s | 2x faster |

**Improvement**: Q8_1 decode went from 4.3 tok/s to 12.65 tok/s (~3x speedup)

---

## Test Status

```bash
ctest --test-dir build_v2_release -R "Q8_1" --output-on-failure -j4
```

- **20 Q8_1 tests executed**
- **19 passed**
- **1 pre-existing failure**: `V2_Perf_QuantisedGemmKernel_Q8_1_OnlineSoftmax.DEBUG_MinimalCase` (unrelated to changes)

New integration test passes:
```bash
ctest --test-dir build_v2_release -R "V2_Integration_Q8_1_FusedAttentionWo_Prefill_TP" --output-on-failure
# 100% tests passed, 1 tests passed out of 1
```

---

## Next Steps

### 1. Investigate Remaining Q8_1 Decode Overhead

Q8_1 decode is still 3.5x slower than FP32. Remaining hotspots from profiling:

| Stage | Time per Layer | Notes |
|-------|----------------|-------|
| `down_proj` | ~2ms | Quantized GEMM |
| `rope` | ~0.4ms | Could benefit from Q8_1 path |
| `attention` | ~0.3ms | JIT kernel |

**Action**: Profile `down_proj` GEMM to understand why it's slower than FP32.

### 2. Check Other Stages for seq_len Bug

The SwiGLU bug pattern (using buffer dimensions instead of actual seq_len) may exist in other stages. Audit:
- `RMSNormStage`
- `ResidualAddStage`
- `RoPEStage`
- `FFNDownStage`

### 3. Fix Pre-existing Test Failure

`V2_Perf_QuantisedGemmKernel_Q8_1_OnlineSoftmax.DEBUG_MinimalCase` has a pre-existing failure. Investigate separately.

### 4. Validate Q8_1 Output Quality

Run E2E parity tests to ensure Q8_1 outputs are numerically correct:
```bash
ctest --test-dir build_v2_e2e_release -R "E2E" --output-on-failure
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/v2/execution/ComputeStage.cpp` | Stage implementations (SwiGLU, RMSNorm, etc.) |
| `src/v2/pipelines/qwen/GraphOrchestrator.cpp` | Graph construction and executor config |
| `src/v2/kernels/cpu/attention/FusedAttentionWoKernel.h` | Attention + Wo projection kernel |
| `src/v2/utils/KernelProfiler.h` | Kernel profiling macros |
| `src/v2/utils/DebugEnv.h` | Environment variable definitions |
| `tests/v2/integration/Test__Q8_1_FusedAttentionWo_Prefill_TP.cpp` | Q8_1 prefill integration test |

---

## Build Commands

```bash
# Release build (for benchmarking)
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Run Q8_1 inference
./build_v2_release/llaminar2 --activation-precision q8_1 \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello, world!" -n 50 -t 0
```
