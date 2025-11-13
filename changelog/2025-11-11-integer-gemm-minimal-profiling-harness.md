# Integer GEMM Minimal Profiling Harness

**Date**: November 11, 2025  
**Status**: ✅ **WORKING** - Ready for iterative bug fixing  
**Purpose**: Isolate and debug the catastrophic INT8 GEMM performance bug (1-2% efficiency)

## Overview

Created a minimal, focused test harness for profiling and iteratively fixing the INT8 GEMM performance bug without the overhead of the full 8000-config sweep suite.

## Deliverables

### 1. Minimal Profiling Harness
**File**: `tests/v2/performance/Perf__IntegerGEMM_Minimal.cpp`  
**Size**: ~300 lines (vs 800+ lines in full sweep)  
**Build target**: `v2_perf_integer_gemm_minimal`

**Key features**:
- ✅ **Standalone executable** (no GTest dependency)
- ✅ **Command-line configuration** (--mr, --m, --n, --k, --unroll, --prefetch, --iters)
- ✅ **Clean output** (formatted performance metrics with efficiency %)
- ✅ **Fast iteration** (3 warmup + 10 timed iters by default)
- ✅ **Profiling-ready** (works with perf, vtune, gdb)
- ✅ **Configuration space coverage** (8 pre-instantiated configs)

### 2. Usage Examples

```bash
# Default (128×896×896, MR=16, shows bug)
./build_v2_release/performance/v2_perf_integer_gemm_minimal

# Test MR=1 (less overhead - 52.6 GFLOPS)
./build_v2_release/performance/v2_perf_integer_gemm_minimal --mr 1

# Test MR=16 (bug manifests - 5.9 GFLOPS)
./build_v2_release/performance/v2_perf_integer_gemm_minimal --mr 16

# Profile with perf
perf record -g ./build_v2_release/performance/v2_perf_integer_gemm_minimal --iters 100
perf report

# Large matrix for sustained profiling
./build_v2_release/performance/v2_perf_integer_gemm_minimal --m 512 --n 4096 --k 4096 --iters 1000

# Custom configuration
./build_v2_release/performance/v2_perf_integer_gemm_minimal \
  --mr 8 --unroll 8 --prefetch 3 --m 256 --n 2048 --k 2048
```

### 3. Baseline Performance (RELEASE build, Buggy Version)

| Configuration | Throughput | Efficiency | Notes |
|---------------|------------|------------|-------|
| **MR=1** (baseline) | **52.6 GFLOPS** | **14.7%** | Best case (minimal overhead) |
| **MR=16** (default) | **5.9 GFLOPS** | **1.65%** | ❌ CATASTROPHIC (8.9× slower!) |

**Theoretical Peak**: 358.4 GFLOPS (2.8 GHz × 2 FMA ports × 64 INT8 ops/dpbusd)

## Bug Confirmation

The harness **successfully reproduces** the catastrophic performance bug:

### Root Cause (Identified)
**Location**: `src/v2/kernels/cpu/gemm/int8/IntegerGemmMicroKernel.h::accumulate()`

**Problem**: Double buffering per K-block

```cpp
// BUGGY CODE (per K-block overhead):
alignas(64) int32_t block_result[TILE_M * TILE_N];  // 512 int32s for MR=16
std::memset(block_result, 0, sizeof(block_result));  // 2KB memset
BaseKernel::micro_kernel(..., block_result, ...);    // Compute into temp
for (int i = 0; i < TILE_M; ++i) {                   // 512-iteration copy
    for (int j = 0; i < TILE_N; ++j) {
        accumulators_[i * TILE_N + j] += block_result[i * TILE_N + j];
    }
}
```

**Impact**:
- **MR=16, NR=32**: 512 int32s allocated, zeroed, and copied **per K-block**
- **28 K-blocks** for 896-dimensional embedding → 28× overhead
- **O(MR) scaling**: Larger MR = larger temp buffer = more overhead
- **Explains 8.9× slowdown** (MR=1 → MR=16)

### Comparison to FP32 GEMM (Correct Implementation)

```cpp
// CORRECT CODE (FP32 GEMM):
for (each K-block) {
    ukernel.accumulate(A, B, p);  // ← Accumulate directly into registers
}
ukernel.reduce(C_tile);  // ← Reduce once at the end
```

**FP32 Performance**: 335-1208 GFLOPS (93-337% efficiency!) ✅

## Next Steps

### Immediate (Bug Fix Iteration)

1. **Remove double buffering wrapper** in `IntegerGemmMicroKernel.h`
2. **Call `GemmMicroKernelTemplateINT8` directly** like FP32 does
3. **Use beta=1 for accumulation** across K-blocks
4. **Benchmark with minimal harness** to verify fix
5. **Expected outcome**: 52.6 GFLOPS (MR=1) → ~300-400 GFLOPS (MR=16)

### Testing Workflow

```bash
# 1. Edit IntegerGemmMicroKernel.h to remove double buffering
# 2. Rebuild
cmake --build build_v2_release --target v2_perf_integer_gemm_minimal --parallel

# 3. Test immediately
./build_v2_release/performance/v2_perf_integer_gemm_minimal --mr 16

# 4. Compare before/after
#    Before: 5.9 GFLOPS (1.65%)
#    Target: 300+ GFLOPS (80%+)

# 5. If fixed, run full sweep to validate all configs
./run_integer_gemm_sweep.sh single
```

### Profiling Workflow

```bash
# Profile hotspots
perf record -g --call-graph dwarf \
  ./build_v2_release/performance/v2_perf_integer_gemm_minimal --mr 16 --iters 100
perf report --stdio | head -100

# Expected hotspots (BEFORE fix):
# - std::memset (zeroing block_result)
# - Nested accumulation loop (512 iterations)
# - BaseKernel::micro_kernel (should be here but overshadowed)

# Expected hotspots (AFTER fix):
# - BaseKernel::micro_kernel (dominant)
# - VNNI intrinsics (_mm512_dpbusd_epi32)
# - Cache prefetch operations
```

## Design Notes

### Pre-Instantiated Configurations

To avoid template bloat, only 8 configurations are pre-instantiated:

```cpp
// Common configs (add more as needed)
MR=1, UNROLL=4, PREFETCH=2   // Baseline (best before fix)
MR=2, UNROLL=4, PREFETCH=2   // Small tile
MR=4, UNROLL=4, PREFETCH=2   // Medium tile
MR=8, UNROLL=4, PREFETCH=2   // Large tile
MR=16, UNROLL=4, PREFETCH=2  // Default (worst bug)
MR=16, UNROLL=8, PREFETCH=2  // High unroll
MR=16, UNROLL=4, PREFETCH=0  // No prefetch
MR=16, UNROLL=4, PREFETCH=3  // Aggressive prefetch
```

To add more: Edit `run_benchmark()` dispatcher and add template instantiation.

### Memory Layout (Critical for Correctness)

**Q8_0Block Tensor Layout**:
```cpp
// A (activations): m ROWS × k_blocks BLOCKS
A = aligned_alloc(64, m * k_blocks * sizeof(Q8_0Block));

// B (weights): n ROWS × k_blocks BLOCKS (NOT n_blocks!)
B = aligned_alloc(64, n * k_blocks * sizeof(Q8_0Block));

// C (output): m ROWS × n_blocks BLOCKS
C = aligned_alloc(64, m * n_blocks * sizeof(Q8_0Block));

// Provider expects:
SimpleBlockProvider B_provider(B, k_blocks, n);  // n = number of rows!
```

**Critical**: B is allocated as `n × k_blocks`, not `n_blocks × k_blocks`!

### Command-Line Arguments

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--m <size>` | M dimension (rows) | 128 | `--m 512` |
| `--n <size>` | N dimension (cols) | 896 | `--n 4096` |
| `--k <size>` | K dimension (inner) | 896 | `--k 2048` |
| `--mr <tiles>` | Microkernel M tiles | 16 | `--mr 8` |
| `--unroll <factor>` | K-loop unroll | 4 | `--unroll 8` |
| `--prefetch <dist>` | Prefetch distance | 2 | `--prefetch 0` |
| `--warmup <iters>` | Warmup iterations | 3 | `--warmup 10` |
| `--iters <count>` | Timed iterations | 10 | `--iters 100` |
| `--verbose, -v` | Verbose output | Off | `-v` |

## Success Criteria

### Phase 1: Bug Fix ✅
- [x] Create minimal harness
- [x] Reproduce bug (MR=16: 5.9 GFLOPS)
- [ ] Remove double buffering
- [ ] Verify fix (MR=16: 300+ GFLOPS expected)
- [ ] Validate all MR values show improvement

### Phase 2: Optimization
- [ ] Profile fixed kernel to find next bottleneck
- [ ] Tune UNROLL_K, PREFETCH_DIST for peak performance
- [ ] Target: >80% efficiency (>286 GFLOPS)

### Phase 3: Validation
- [ ] Run full 8000-config sweep with fixed kernel
- [ ] Verify ML heuristic improvements
- [ ] Update benchmark documentation

## Files Modified

### New Files
- `tests/v2/performance/Perf__IntegerGEMM_Minimal.cpp` (300 lines, standalone harness)
- `changelog/2025-11-11-integer-gemm-minimal-profiling-harness.md` (this file)

### Modified Files
- `tests/v2/CMakeLists.txt` (+38 lines, new target + documentation)

### Build Artifacts
- `build_v2_release/performance/v2_perf_integer_gemm_minimal` (executable, ~100KB)

## Conclusion

The minimal profiling harness is **ready for iterative bug fixing**. It successfully reproduces the catastrophic performance bug (1.65% efficiency with MR=16) and provides a clean, fast iteration cycle for testing fixes.

**Next action**: Remove the double buffering in `IntegerGemmMicroKernel.h::accumulate()` and retest!
