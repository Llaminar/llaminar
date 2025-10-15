# Prefill Performance Benchmark - User Guide
**Date:** October 15, 2025  
**Author:** David Sanftenberg

## Overview

The Prefill Performance Benchmark is a comprehensive test harness designed to measure and analyze the parallelization efficiency of prefill operations in Llaminar. It tests both OpenBLAS and COSMA backends across various problem sizes and thread configurations, providing detailed metrics on:

- **Throughput** (GFLOPS)
- **Memory bandwidth** (GB/s)
- **Parallel efficiency** (actual vs ideal speedup)
- **CPU utilization** (percentage)
- **Strong scaling** (fixed problem size, varying threads)
- **Weak scaling** (problem size scales with threads)

## Quick Start

### Run All Benchmarks
```bash
./run_performance_bench.sh
```

### Run Specific Test Suite
```bash
# Model shape performance analysis
./run_performance_bench.sh --filter '*ModelShapes*'

# Strong scaling analysis
./run_performance_bench.sh --filter '*StrongScaling*'

# Thread utilization analysis
./run_performance_bench.sh --filter '*ThreadUtilization*'

# Weak scaling analysis
./run_performance_bench.sh --filter '*WeakScaling*'
```

### Direct Test Execution
```bash
# Via CTest
ctest --test-dir build -R PrefillPerformanceBench --output-on-failure -V

# Direct MPI execution
mpirun -np 2 --bind-to socket ./build/test_prefill_performance_bench
```

## Test Suites

### 1. Strong Scaling Test
**Purpose:** Measures how well performance scales when adding more threads to a fixed problem size.

**Metrics:**
- Ideal: Linear speedup (2x threads = 2x performance)
- Good: 70-90% efficiency
- Reality: Usually 60-80% due to synchronization overhead

**Test Cases:**
- `OpenBLAS_StrongScaling_Prefill` - 2048 token prefill (typical size)
- `OpenBLAS_StrongScaling_LargePrefill` - 8192 token prefill (large batch)

**Example Output:**
```
Backend     MPI  OMP   Dims          Time(ms)    GFLOPS    BW(GB/s)  Efficiency  CPU%
-------     ---  ---   ----          --------    ------    --------  ----------  ----
OpenBLAS     2    1   2048x896x896    245.32     14.82      3.21        100.0%    1.8%
OpenBLAS     2    2   2048x896x896    128.45     28.28      6.13         95.4%   17.1%
OpenBLAS     2    4   2048x896x896     66.72     54.43     11.79         91.9%   33.0%
OpenBLAS     2    8   2048x896x896     35.21     103.21    22.34         87.1%   62.3%
OpenBLAS     2   28   2048x896x896     12.89     281.78    61.03         77.6%   87.3%
```

**Interpretation:**
- Efficiency drops slightly with more threads (overhead)
- CPU% shows actual core utilization
- GFLOPS should increase near-linearly for good parallelization

### 2. Weak Scaling Test
**Purpose:** Measures how well performance maintains when problem size and thread count scale proportionally.

**Ideal:** Time remains constant (perfect weak scaling)  
**Reality:** Time increases slightly due to communication/synchronization

**Test Cases:**
- `OpenBLAS_WeakScaling` - Base: 256 tokens per thread

**Example Output:**
```
Backend     MPI  OMP   Dims          Time(ms)    GFLOPS    BW(GB/s)  Efficiency  CPU%
-------     ---  ---   ----          --------    ------    --------  ----------  ----
OpenBLAS     2    1    256x896x896     30.12     14.35      3.11        100.0%  100.0%
OpenBLAS     2    2    512x896x896     31.45     27.41      5.93         95.8%   95.8%
OpenBLAS     2    4   1024x896x896     33.89     50.78     10.99         88.9%   88.9%
OpenBLAS     2    8   2048x896x896     38.21     90.12     19.50         78.8%   78.8%
OpenBLAS     2   28   7168x896x896     52.34    261.45     56.59         57.5%   57.5%
```

**Interpretation:**
- Time should stay relatively constant
- Efficiency > 80% indicates good weak scaling
- Degradation shows overhead of parallelization

### 3. Model Shape Performance
**Purpose:** Tests real-world Qwen model shapes (decode and prefill operations).

**Test Cases:**
- Single token decode (1×896×896)
- Small batches (8, 64 tokens)
- Medium prefill (512, 2048 tokens)
- Large prefill (4096, 8192 tokens)

**Example Output:**
```
  Qwen 0.5B - Single token Q proj         OpenBLAS  2  28  1x896x896      0.42  8651.23  1872.76   15.4%   15.4%
  Qwen 0.5B - Single token KV proj        OpenBLAS  2  28  1x896x2304     1.05  3929.18   850.76   14.0%   14.0%
  Qwen 0.5B - Batch-8 Q proj              OpenBLAS  2  28  8x896x896      2.89  2512.34   543.67   85.0%   85.0%
  Qwen 0.5B - Batch-64 Q proj             OpenBLAS  2  28  64x896x896    18.34  3139.45   679.42   85.0%   85.0%
  Qwen 0.5B - Batch-512 prefill           OpenBLAS  2  28  512x896x896  134.56  2698.89   584.12   85.0%   85.0%
  Qwen 0.5B - Batch-2048 prefill          OpenBLAS  2  28 2048x896x896  512.34  2821.67   610.45   85.0%   85.0%
  Qwen 0.5B - Batch-4096 prefill          OpenBLAS  2  28 4096x896x896 1024.12  2815.23   609.06   85.0%   85.0%
```

**Key Insights:**
- Single-token operations have lower efficiency (small problem size)
- Batch operations achieve higher efficiency (better amortization)
- GFLOPS should remain relatively stable across batch sizes

### 4. Thread Utilization Analysis
**Purpose:** Detailed analysis of actual vs theoretical peak performance across thread counts.

**Metrics:**
- Single-thread baseline (100% efficiency)
- Multi-thread efficiency vs single-thread
- CPU utilization percentage

**Example Output:**
```
Backend     MPI  OMP   Dims          Time(ms)    GFLOPS    BW(GB/s)  Efficiency  CPU%
-------     ---  ---   ----          --------    ------    --------  ----------  ----
OpenBLAS     2    1   4096x896x896   1024.56     14.12      3.06        100.0%    3.6%
OpenBLAS     2    2   4096x896x896    518.34     27.92      6.04         98.9%   71.3%
OpenBLAS     2    4   4096x896x896    264.12     54.78     11.86         97.1%   88.4%
OpenBLAS     2    8   4096x896x896    137.45    105.23     22.78         93.5%   85.2%
OpenBLAS     2   14   4096x896x896     82.34    175.67     38.01         88.6%   89.1%
OpenBLAS     2   28   4096x896x896     45.12    320.45     69.34         81.4%   81.4%
```

**Efficiency Guide:**
- **> 90%:** Excellent parallelization, minimal overhead
- **70-90%:** Good parallelization, acceptable overhead
- **50-70%:** Moderate parallelization, some bottlenecks
- **< 50%:** Poor parallelization, major bottlenecks

## Performance Metrics Explained

### GFLOPS (Giga Floating-Point Operations Per Second)
- Measures computational throughput
- For GEMM: `FLOPS = 2 × m × n × k` (multiply + add)
- Higher is better
- Compare against theoretical peak (e.g., 1 TFLOPS for high-end CPU)

### Memory Bandwidth (GB/s)
- Measures data transfer rate
- For GEMM: `bytes = sizeof(float) × (m×k + k×n + m×n)`
- Limited by DRAM speed (~50-200 GB/s per socket)
- Memory-bound operations show low GFLOPS but high BW%

### Parallel Efficiency
- `Efficiency = (actual_speedup) / (ideal_speedup)`
- `Speedup = baseline_time / current_time`
- `Ideal_speedup = num_threads × num_ranks`
- 100% = perfect linear scaling
- Reality: 60-85% typical for large operations

### CPU Utilization (%)
- Percentage of available CPU cores actively computing
- 100% = all cores fully utilized
- < 50% suggests bottlenecks (memory, synchronization, I/O)
- Use `htop` during benchmark to visually confirm

## Interpreting Results

### Good Parallelization Pattern
```
Threads    Time(ms)    GFLOPS    Efficiency
-------    --------    ------    ----------
   1        1000.0      14.5        100%
   2         510.0      28.4         98%
   4         262.0      55.3         95%
   8         138.0     105.1         91%
  28          45.0     322.2         82%
```
- Near-linear GFLOPS scaling
- High efficiency (> 80%)
- Time reduces proportionally

### Poor Parallelization Pattern
```
Threads    Time(ms)    GFLOPS    Efficiency
-------    --------    ------    ----------
   1        1000.0      14.5        100%
   2         750.0      19.3         67%
   4         650.0      22.3         38%
   8         580.0      25.0         22%
  28         520.0      27.9          7%
```
- GFLOPS plateau
- Low efficiency (< 50%)
- Diminishing returns with more threads
- **Root cause:** Synchronization, memory bandwidth, or lock contention

## Troubleshooting

### Low GFLOPS Across All Tests
**Symptoms:** All operations show < 10 GFLOPS  
**Possible Causes:**
- OpenBLAS not using threads (`OPENBLAS_NUM_THREADS=1`)
- CPU throttling (thermal or power)
- Running on low-performance cores

**Solutions:**
```bash
# Check OpenBLAS configuration
./run_performance_bench.sh --filter '*ThreadUtilization*'

# Monitor CPU frequency
watch -n1 "cat /proc/cpuinfo | grep MHz"

# Disable CPU throttling (requires root)
sudo cpupower frequency-set --governor performance
```

### Low Efficiency Despite High GFLOPS
**Symptoms:** GFLOPS increase but efficiency < 50%  
**Explanation:** Amdahl's Law - sequential portions limit speedup  
**Expected:** Small operations always have lower efficiency

### Memory Bandwidth Saturation
**Symptoms:** GFLOPS plateau, BW approaching DRAM limit  
**Example:** 180 GB/s on system with 200 GB/s max DRAM bandwidth  
**Explanation:** Memory-bound, not compute-bound  
**Solution:** Use larger caches, better data locality, or distributed memory

### MPI Binding Warnings
**Symptoms:**
```
WARNING: Open MPI tried to bind a process but failed.
  Error message: failed to bind memory
```
**Cause:** Docker/container environment lacks memory binding permissions  
**Impact:** Performance degradation (NUMA locality lost)  
**Solution:**
```bash
# Run container with --cap-add=SYS_NICE --cap-add=IPC_LOCK
# Or disable binding checks:
export OMPI_MCA_hwloc_base_binding_policy=none
```

## Integration with CI/CD

### Smoke Test (Fast, < 30s)
```bash
ctest --test-dir build -R PrefillPerformanceBench_ThreadUtilization
```

### Full Benchmark (Slow, ~5 minutes)
```bash
ctest --test-dir build -R PrefillPerformanceBench --output-on-failure
```

### Regression Detection
Compare GFLOPS against baseline:
```bash
# Store baseline
./run_performance_bench.sh --filter '*ModelShapes*' > baseline_gflops.txt

# After optimization, compare
./run_performance_bench.sh --filter '*ModelShapes*' > optimized_gflops.txt
diff baseline_gflops.txt optimized_gflops.txt
```

## Advanced Usage

### Custom Problem Sizes
Modify `test_prefill_performance_bench.cpp` and add test cases:
```cpp
TEST_F(PrefillPerformanceBench, CustomShape) {
    test_model_shapes("OpenBLAS", false);
    // Add custom shapes to the shapes vector
}
```

### COSMA vs OpenBLAS Comparison
Enable COSMA tests (currently disabled):
```cpp
// Remove DISABLED_ prefix in test_prefill_performance_bench.cpp
TEST_F(PrefillPerformanceBench, COSMA_StrongScaling_Prefill) { ... }
TEST_F(PrefillPerformanceBench, ComparativePerformance) { ... }
```

### Profiling Integration
```bash
# Profile with perf
perf record -g mpirun -np 2 ./build/test_prefill_performance_bench --gtest_filter='*ModelShapes*'
perf report

# Profile with VTune (Intel)
vtune -collect hotspots -r vtune_results -- mpirun -np 2 ./build/test_prefill_performance_bench
```

## Expected Performance Targets

### Qwen 0.5B (d_model=896)
| Operation | Size | Target GFLOPS | Target Efficiency |
|-----------|------|---------------|-------------------|
| Single token | 1×896×896 | 8-15 | 10-20% |
| Batch-8 | 8×896×896 | 50-100 | 70-85% |
| Batch-64 | 64×896×896 | 200-400 | 75-88% |
| Prefill 2K | 2048×896×896 | 250-350 | 78-90% |
| Prefill 8K | 8192×896×896 | 280-380 | 80-92% |

### System-Specific Baselines
**2-socket, 28 cores/socket, DDR4-3200:**
- Peak theoretical: ~1.5 TFLOPS (sustained ~800 GFLOPS)
- Memory bandwidth: ~180 GB/s total
- Expected large-op efficiency: 75-85%

## Related Documentation

- **Threading Optimizations:** `changelog/2025-10-15_threading_optimizations_complete.md`
- **Performance Analysis:** `changelog/2025-10-15_performance_threading_analysis.md`
- **Copilot Instructions:** `.github/copilot-instructions.md`

## Contributing

To add new benchmark tests:
1. Add test method to `PrefillPerformanceBench` class
2. Register with `TEST_F(PrefillPerformanceBench, YourTestName)`
3. Add to CMakeLists.txt with appropriate timeout
4. Update this documentation

---

**Author:** David Sanftenberg  
**Date:** October 15, 2025  
**Version:** 1.0
