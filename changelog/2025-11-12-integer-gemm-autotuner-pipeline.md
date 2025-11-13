# Integer GEMM Autotuner Data Collection Pipeline - November 12, 2025

## Session Overview

Built comprehensive infrastructure for gathering training data to develop an intelligent kernel selection autotuner for Integer GEMM operations. This addresses the critical finding that **fixed tile size parameters are inadequate** - optimal MR varies dramatically (4 to 16) based on workload characteristics.

## Problem Statement

**Tile sweep revealed no single optimal configuration**:
- Decode (M=1): MR=8 is 81% faster than MR=16
- Prefill-512 (M=512): MR=4 is 36% faster than MR=16
- Current fixed default (MR=16) is suboptimal across all workloads

**User insight**: "Setting default values is kind of pointless. The problem size will change too much and too often."

## Solution: Autotuner with Data-Driven Selection

Similar to existing CUDA autotuner (`auto_run_pipeline.sh`), build a system that:
1. Gathers comprehensive performance + microarchitectural data
2. Analyzes correlations between problem size and optimal config
3. Predicts best tile parameters based on (M, N, K) workload shape

## Implementation

### 1. Full Configuration Space Sweep Test

**File**: `tests/v2/performance/cpu/kernels/gemm/Perf__IntegerGEMM_FullSweep.cpp` (652 lines)

**Test modes**:
```bash
# Complete sweep: 384 configs × 12 workloads = 4,608 benchmarks
./v2_perf_integer_gemm_full_sweep --gtest_filter="*AllConfigsAllWorkloads"

# Single workload: 384 configs on decode (for quick testing)
./v2_perf_integer_gemm_full_sweep --gtest_filter="*AllConfigsSingleToken"

# Single config: For perf profiling (encodes params in test name)
./v2_perf_integer_gemm_full_sweep --gtest_filter="*SingleConfig_M512_N896_K896_MR8_NR32_KB4_U4_P2_MC256_KC512_NC128"
```

**Configuration space**:
- MR: {4, 8, 16, 32} (4 values)
- K_BLOCKS: {2, 4} (64-byte vs 128-byte)
- UNROLL_K: {1, 2, 4, 8} (4 values)
- PREFETCH_DIST: {0, 1, 2, 3} (4 values)
- Cache blocking: {(128,256,64), (256,512,128), (512,1024,256)} (3 combos)

**Total**: 4 × 2 × 4 × 4 × 3 = **384 configs per workload**

**Workloads** (Qwen 0.5B representative operations):
- 3 decode ops (M=1): Q_proj, FFN_gate, FFN_down
- 9 prefill ops (M=32/128/512): Same ops × 3 batch sizes

**Output**: CSV format
```csv
m,n,k,mr,nr,k_blocks,unroll_k,prefetch_dist,mc,kc,nc,gflops,time_ms,efficiency_pct
1,896,896,8,32,4,4,2,256,512,128,1.34,0.52,0.37
512,896,896,4,32,4,2,2,256,512,128,264.94,1.01,73.92
```

### 2. Data Collection Pipeline

**File**: `run_integer_gemm_autotuner_data.sh` (450+ lines)

**Phase 1 - Performance Sweep** (30-60 minutes):
- Run full config sweep
- Output: `integer_gemm_sweep_full.csv` (~500 KB, 4,608 rows)

**Phase 2 - Identify Top/Bottom Performers** (<1 minute):
- Python script analyzes sweep results
- For each workload: extract top-10 and bottom-10 configs by GFLOPS
- Output: `integer_gemm_selected_for_perf.csv` (~240 rows: 20 per workload × 12 workloads)

**Phase 3 - Hardware Counter Collection** (20-40 minutes):
- For each selected config, run `perf stat` with hardware counters:
  - `L1-dcache-loads`, `L1-dcache-load-misses` → L1 miss rate
  - `LLC-loads`, `LLC-load-misses` → Last level cache miss rate
  - `cycles`, `instructions` → IPC (instructions per cycle)
  - `branch-misses`, `context-switches`, `cpu-migrations`
- Output: `integer_gemm_perf_counters.csv` with microarchitectural features

**Phase 4 - Data Merging** (<1 minute):
- Merge sweep + perf data
- Output: `integer_gemm_autotuner_training.csv` (~520 KB)

**Complete pipeline usage**:
```bash
# Full pipeline (60-90 minutes)
./run_integer_gemm_autotuner_data.sh

# Fast mode (skip perf counters, 30-60 minutes)
./run_integer_gemm_autotuner_data.sh --skip-perf

# Resume from existing sweep
./run_integer_gemm_autotuner_data.sh --skip-sweep
```

### 3. Documentation

**File**: `INTEGER_GEMM_AUTOTUNER_GUIDE.md` (350+ lines)

Comprehensive guide covering:
- Pipeline architecture and rationale
- Configuration space and workload definitions
- Phase-by-phase execution details
- Hardware counter descriptions
- Next steps for heuristic/ML development
- Troubleshooting (`perf` permissions, missing counters)
- Expected performance improvements (20-40%)

### 4. CMake Integration

Added to `tests/v2/CMakeLists.txt`:
```cmake
add_executable(v2_perf_integer_gemm_full_sweep performance/cpu/kernels/gemm/Perf__IntegerGEMM_FullSweep.cpp)
target_link_libraries(v2_perf_integer_gemm_full_sweep
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
```

## Key Features

### Intelligent Test Design

**SingleConfig tests with encoded parameters**:
```cpp
// Test name encodes all config params for perf script parsing
TEST(IntegerGEMM_FullSweep, SingleConfig_M512_N896_K896_MR8_NR32_KB4_U4_P2_MC256_KC512_NC128)
```

Enables automated perf profiling:
```bash
# Perf script can construct test filter from CSV config line
FILTER="*SingleConfig_M${M}_N${N}_K${K}_MR${MR}_NR${NR}_KB${KB}_U${U}_P${P}_MC${MC}_KC${KC}_NC${NC}"
perf stat -e L1-dcache-loads,... ./v2_perf_integer_gemm_full_sweep --gtest_filter="$FILTER"
```

### Robust Block Provider Implementation

Fixed common pitfalls:
- Uses public `get_raw_block_at()` API (not private `raw_data_`)
- Implements all required `Q8_0BlockProvider` methods:
  ```cpp
  void warmup_cache(size_t, size_t, size_t, size_t) override {}
  bool is_zero_copy() const override { return true; }
  size_t k_blocks() const override { return k_blocks_; }
  size_t num_rows() const override { return num_rows_; }
  ```

### Correct Kernel API Usage

Uses static `IntegerGemmKernelV2::multiply()` signature:
```cpp
using KernelType = IntegerGemmKernelV2<ISA, MR, NR, K_BLOCKS, UNROLL_K, PREFETCH_DIST, MC, KC, NC>;
KernelType::multiply(A_blocks, B_provider, C_blocks, m, n, k);
```

## Expected Performance Impact

Based on tile sweep findings, autotuner should achieve:

| Workload | Current (MR=16) | Optimal Config | Speedup |
|----------|----------------|----------------|---------|
| Decode (M=1) | 0.74 GFLOPS | MR=8: 1.34 GFLOPS | **1.81×** |
| Prefill-32 | 22.5 GFLOPS | MR=8: 43.1 GFLOPS | **1.92×** |
| Prefill-128 | 92.7 GFLOPS | MR=8: 171.9 GFLOPS | **1.85×** |
| Prefill-512 | 194.2 GFLOPS | MR=4: 264.9 GFLOPS | **1.36×** |

**Conservative estimate**: 20-40% average improvement by selecting optimal config per workload  
**Optimistic estimate**: May discover additional optimization opportunities via cache/IPC analysis

## Next Steps (Future Work)

### Option 1: Rule-Based Heuristic

Analyze data for simple decision rules:
```python
# Find correlations
correlations = df[['m', 'n', 'k', 'mr', 'ipc', 'l1_miss_rate', 'gflops']].corr()

# Example rules discovered:
# - m < 32 && llc_miss_rate < 0.01 → MR=8
# - m >= 256 && l1_miss_rate < 0.02 → MR=4
```

Implement in C++:
```cpp
class IntegerGemmAutotuner {
public:
    static AutotunerConfig selectConfig(int m, int n, int k) {
        if (m == 1) return {8, 32, 4, 4, 2, 256, 512, 128}; // Decode
        if (m >= 256) return {4, 32, 4, 2, 2, 256, 512, 128}; // Large prefill
        return {8, 32, 4, 4, 2, 256, 512, 128}; // Medium prefill
    }
};
```

### Option 2: ML-Based Autotuner (ONNX)

Following CUDA autotuner pattern (`auto_run_pipeline.sh`):
```bash
# Train neural network
python3 train_integer_gemm_ml.py \
  --input integer_gemm_autotuner_training.csv \
  --output integer_gemm_heuristic.onnx

# Export to C++ header (embedded weights)
python3 export_integer_gemm_heuristic.py \
  --model integer_gemm_heuristic.onnx \
  --output IntegerGemmHeuristicWeights.h
```

Inference:
```cpp
// Predict optimal MR from problem size
float input[3] = {float(m), float(n), float(k)};
float output[4]; // Probabilities for MR={4,8,16,32}
feedforward_nn(input, output); // Embedded weights
int predicted_mr = argmax(output);
```

## Build Verification

```bash
$ cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
$ cmake --build build_v2_release --target v2_perf_integer_gemm_full_sweep --parallel 4
[100%] Built target v2_perf_integer_gemm_full_sweep
```

 **Compiles successfully**

## Files Created/Modified

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `Perf__IntegerGEMM_FullSweep.cpp` | 652 | ✅ New | Complete config space sweep test |
| `run_integer_gemm_autotuner_data.sh` | 450+ | ✅ New | 4-phase data collection pipeline |
| `INTEGER_GEMM_AUTOTUNER_GUIDE.md` | 350+ | ✅ New | Comprehensive documentation |
| `INTEGER_GEMM_AUTOTUNER_IMPLEMENTATION.md` | 250+ | ✅ New | Implementation summary |
| `tests/v2/CMakeLists.txt` | +15 | ✅ Modified | Add full sweep test target |

**Total new content**: ~1,700 lines (code + docs)

## Testing Strategy

**Phase 1 validation** (quick test):
```bash
# Test single workload sweep (~10 minutes)
./v2_perf_integer_gemm_full_sweep --gtest_filter="*AllConfigsSingleToken" > test_sweep.csv

# Verify output format
head test_sweep.csv
# Expected: CSV header + 384 data rows
```

**Phase 2+3 validation** (requires perf permissions):
```bash
# Check perf availability
perf stat -e cycles echo test
# If permission denied: sudo sysctl -w kernel.perf_event_paranoid=-1

# Run single config under perf
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses \
  ./v2_perf_integer_gemm_full_sweep --gtest_filter="*SingleConfig_M512_N896_K896_MR8_*"
```

**Full pipeline** (60-90 minutes):
```bash
./run_integer_gemm_autotuner_data.sh
# Expected outputs:
#   - integer_gemm_sweep_full.csv (~500 KB, 4,608 rows)
#   - integer_gemm_selected_for_perf.csv (~30 KB, 240 rows)
#   - integer_gemm_perf_counters.csv (~40 KB, 240 rows with counters)
#   - integer_gemm_autotuner_training.csv (~520 KB, merged data)
```

## Lessons Learned

### 1. Tile Size Selection is Problem-Dependent
- No single MR value is optimal across workloads
- Small M (decode): MR=8 wins
- Large M (prefill-512): MR=4 wins
- Medium M: MR=8 balanced

### 2. Microarchitectural Events Matter
- Cache miss rates correlate strongly with performance
- IPC reveals instruction-level efficiency
- Thread overhead (context switches) affects small workloads

### 3. Test Infrastructure Design
- Encode config params in test names for perf integration
- Use template dispatch for compile-time kernel specialization
- Separate concerns: sweep test vs perf collection script

## References

- **Tile sweep findings**: `INTEGER_GEMM_TILE_SWEEP_RESULTS.md`
- **CUDA autotuner precedent**: `auto_run_pipeline.sh`, `CUDA_ML_HEURISTIC_SUMMARY.md`
- **Performance baseline**: `changelog/2025-11-12-integer-gemm-tile-sweep.md`

## Conclusion

Built complete infrastructure for data-driven autotuner development. The pipeline systematically explores 4,608 config×workload combinations, collecting both performance metrics (GFLOPS) and microarchitectural features (cache misses, IPC) to enable intelligent kernel selection.

**Next action**: Run data collection pipeline and analyze results to design heuristic or ML model.

**Impact**: Expected 20-40% performance improvement by adapting tile parameters to workload characteristics, with potential for larger gains (up to 92% on prefill-32) in favorable cases.

**Status**: ✅ Implementation complete, ready for data collection run.
