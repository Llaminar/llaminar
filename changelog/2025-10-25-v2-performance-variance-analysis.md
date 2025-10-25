# V2 Performance Test: Multi-Trial Variance Analysis

**Date**: October 25, 2025  
**Author**: GitHub Copilot (with user dbsanfte)  
**Session**: OpenMP Integration → Performance Analysis → Variance Characterization

## Summary

Enhanced V2 performance benchmarks to run **5 independent trials** for each test case and report **mean ± standard deviation** with coefficient of variation (CV%). This provides insight into measurement stability and repeatability.

## Performance Results (5 trials each)

### Test Configuration
- **MPI Ranks**: 2 (bind-to-socket, map-by-socket)
- **OpenMP Threads**: 28 per rank (56 total cores)
- **Model**: Qwen 2.5 0.5B IQ4_NL quantized
- **Trials**: 5 independent runs per test case
- **Build**: Release mode with `-march=native`

### Benchmark Results

| Test Case | Batch Size | Time (ms) | CV% | Throughput (GFLOPS) | Variance |
|-----------|------------|-----------|-----|---------------------|----------|
| **Small Batch** | 32 | 0.52 ± 0.01 | **1.2%** | 97.90 ± 1.13 | Very stable |
| **Medium Batch** | 128 | 0.75 ± 0.02 | **3.3%** | 273.30 ± 9.05 | Stable |
| **Large Batch** | 512 | 6.18 ± 0.03 | **0.6%** | 132.97 ± 0.73 | Very stable |
| **Q-Proj 1024** | 1024 | 4.08 ± 0.09 | **2.1%** | **402.89 ± 8.65** | Stable |
| **Single Token** | 1 | 1.08 ± 0.05 | **4.7%** | 1.48 ± 0.07 | Moderate variance |

**Key Findings:**
- ✅ **Most tests show excellent stability**: CV < 3.5%
- ✅ **Q-Proj 1024 maintains 400+ GFLOPS**: 402.89 ± 8.65 (2.1% CV)
- ✅ **Large batch very consistent**: 0.6% CV despite 6ms runtime
- ⚠️ **Single token shows higher variance**: 4.7% CV (likely due to very short runtime ~1ms)

### Virtual Dispatch vs Template Comparison

| Implementation | Time (ms) | CV% | Throughput (GFLOPS) |
|----------------|-----------|-----|---------------------|
| **Virtual Dispatch** | 3.98 ± 0.14 | **3.5%** | 412.96 ± 14.63 |
| **Template** | 4.07 ± 0.06 | **1.5%** | 403.53 ± 6.09 |

**Surprising Result**: Virtual dispatch is **2% faster** (412.96 vs 403.53 GFLOPS)!
- Template has **lower variance** (1.5% vs 3.5% CV) → more predictable
- Performance difference is within measurement noise (~2.3% difference)
- **Conclusion**: Virtual dispatch overhead negligible for this workload (dominated by compute, not dispatch)

## Implementation Details

### Changes Made

**File**: `tests/v2/performance/Perf__IQ4_NL_GEMM.cpp`

1. **Added BenchmarkStats struct**:
   ```cpp
   struct BenchmarkStats {
       double mean_ms;        ///< Mean time per iteration (ms)
       double stddev_ms;      ///< Standard deviation (ms)
       double min_ms;         ///< Minimum time (ms)
       double max_ms;         ///< Maximum time (ms)
       double mean_gflops;    ///< Mean throughput (GFLOPS)
       double stddev_gflops;  ///< Standard deviation of throughput
   };
   ```

2. **Enhanced BenchmarkConfig**:
   ```cpp
   struct BenchmarkConfig {
       // ... existing fields ...
       int num_trials;        ///< Number of independent trials (default: 5)
   };
   ```

3. **Multi-trial benchmark loop**:
   ```cpp
   for (int trial = 0; trial < config.num_trials; ++trial) {
       // Run bench_iters iterations (timed)
       MPI_Barrier(MPI_COMM_WORLD);
       auto start = std::chrono::high_resolution_clock::now();
       
       for (int i = 0; i < config.bench_iters; ++i) {
           gemm.multiply(...);  // Timed operation
       }
       
       MPI_Barrier(MPI_COMM_WORLD);
       auto end = std::chrono::high_resolution_clock::now();
       
       trial_times_ms.push_back(elapsed_ms / config.bench_iters);
   }
   ```

4. **Statistical calculations**:
   ```cpp
   // Mean
   stats.mean_ms = sum(trial_times_ms) / num_trials;
   
   // Standard deviation
   stats.stddev_ms = sqrt(sum((t - mean)²) / num_trials);
   
   // Coefficient of variation
   double cv_percent = (stddev_ms / mean_ms) * 100.0;
   ```

5. **Enhanced output format**:
   ```
   ╔════════════════════════════════════════════════════════════════╗
   ║ Performance (mean ± stddev):                                   ║
   ║   Time per iter:    4.08    ± 0.09  ms (CV: 2.1 %)            ║
   ║   Throughput:       402.89  ± 8.65  GFLOPS                      ║
   ╠════════════════════════════════════════════════════════════════╣
   ║ Range:                                                         ║
   ║   Min time:         3.97       ms                                   ║
   ║   Max time:         4.22       ms                                   ║
   ╚════════════════════════════════════════════════════════════════╝
   ```

## Analysis: Variance Patterns

### Very Stable (CV < 1.5%)
- **Large Batch (512 tokens)**: 0.6% CV
  - Long runtime (6.18ms) → averaging effect
  - Cache-warm steady state
  - Predictable memory access pattern

- **Small Batch (32 tokens)**: 1.2% CV
  - Short but consistent runtime
  - Good microkernel efficiency

### Stable (1.5% < CV < 3.5%)
- **Q-Proj 1024**: 2.1% CV
  - Good stability for large workload
  - 4.08ms ± 0.09ms → 6% max variation
  - **402.89 GFLOPS peak performance**

- **Medium Batch (128 tokens)**: 3.3% CV
  - Moderate variance acceptable
  - Transitional batch size

### Moderate Variance (CV > 4%)
- **Single Token**: 4.7% CV
  - Very short runtime (1.08ms) → sensitive to noise
  - System jitter becomes significant
  - 0.99ms to 1.12ms range (13% span)

## Insights

### Why Variance Matters
1. **Repeatability**: Low CV% means results are reproducible
2. **Benchmarking**: High variance indicates measurement noise or system instability
3. **Optimization**: Stable baselines required for A/B testing

### Recommendations
1. ✅ **Use Q-Proj 1024 as canonical benchmark**: 400+ GFLOPS, 2.1% CV, matches V1 workload
2. ✅ **Report mean ± stddev** for all performance claims
3. ✅ **Increase trials for short tests**: Single-token needs 10+ trials for CV < 3%
4. ⚠️ **Be cautious with micro-benchmarks**: Sub-millisecond tests inherently noisy

## Comparison to Previous Session

**Before (single measurement)**:
- Q-Proj 1024: 400.82 GFLOPS (no variance data)
- Could have been outlier or representative?

**After (5 trials)**:
- Q-Proj 1024: **402.89 ± 8.65 GFLOPS** (2.1% CV)
- High confidence in 400 GFLOPS performance
- Range: 394-412 GFLOPS (consistent)

**Validation**: Previous 400 GFLOPS measurement was representative, not anomaly!

## Next Steps

### Immediate
- [x] Multi-trial statistics implemented
- [x] Variance characterized across workloads
- [ ] Extend to 10 trials for single-token tests (reduce CV from 4.7% → ~2%)

### Future Work
1. **Automated Regression Detection**: Alert if performance degrades >3σ
2. **Confidence Intervals**: Report 95% CI instead of just ±1σ
3. **Outlier Removal**: Implement RANSAC or median-based statistics
4. **Cross-Session Tracking**: Compare results across builds/commits
5. **Batch Size Sweep**: Test 2⁰, 2¹, 2², ..., 2¹⁰ with variance at each point

## Files Modified

1. `tests/v2/performance/Perf__IQ4_NL_GEMM.cpp`
   - Added `BenchmarkStats` struct
   - Enhanced `BenchmarkConfig` with `num_trials`
   - Rewrote `benchmarkFP32()` for multi-trial execution
   - Updated `printResults()` to show variance
   - Applied to all 6 test cases

**Lines changed**: ~300 (mostly printResults and benchmarkFP32 rewrite)

## Session Timeline

1. **Context**: Previous session discovered OpenMP missing → 10× speedup
2. **Goal**: "Can we report variance between runs?"
3. **Implementation**: Multi-trial framework with statistics
4. **Result**: All tests now report mean ± stddev with CV%
5. **Validation**: Q-Proj 1024 confirmed at **402.89 ± 8.65 GFLOPS** (2.1% CV)

---

**Status**: ✅ Complete  
**Performance**: ✅ Validated (400+ GFLOPS with 2.1% variance)  
**Documentation**: ✅ This changelog
