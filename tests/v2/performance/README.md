# Llaminar V2 Performance Test Suite

This directory contains performance benchmarks for V2 components. These tests measure **throughput**, **latency**, and **resource utilization** under production-like conditions.

## Overview

Performance tests differ from unit/integration tests:

| Aspect | Unit/Integration Tests | Performance Tests |
|--------|----------------------|-------------------|
| **Build Type** | Debug (`build_v2/`) | Release (`build_v2_release/`) |
| **Purpose** | Correctness validation | Performance measurement |
| **Optimizations** | `-O0 -g` | `-O3 -DNDEBUG -march=native` |
| **Thread Settings** | Default | Optimal (core pinning, NUMA) |
| **Execution Time** | Fast (~seconds) | Slower (~minutes) |
| **Frequency** | Every commit | Before releases, after optimizations |

## Running Performance Tests

### Quick Start

```bash
# From workspace root
cd build_v2
ctest -L Performance --verbose

# Or with specific test filter
ctest -L Performance -R "IQ4_NL_GEMM" --verbose
```

**What happens:**
1. CMake checks if `build_v2_release/` exists
2. If not, automatically builds Release configuration
3. Launches performance tests with optimal MPI/OpenMP settings
4. Reports GFLOPS, bandwidth, and timing metrics

### Manual Release Build

If you prefer to build Release manually:

```bash
# From workspace root
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel
cd build_v2
ctest -L Performance --verbose
```

### Running Individual Benchmarks

```bash
# From build_v2_release/tests/v2/performance/
./v2_perf_iq4nl_gemm

# Or with MPI (matches CTest configuration)
mpirun -np 1 --bind-to socket ./v2_perf_iq4nl_gemm
```

## Optimal Launch Settings

Performance tests are automatically launched with settings from `run_benchmark.sh`:

### CPU Topology Detection

```bash
# Auto-detected at CMake configure time
SOCKETS=$(lscpu | grep 'Socket(s):' | awk '{print $2}')
CORES_PER_SOCKET=$(lscpu | grep 'Core(s) per socket:' | awk '{print $4}')
```

### OpenMP Configuration

```bash
export OMP_NUM_THREADS=$CORES_PER_SOCKET   # Physical cores per socket
export OMP_PLACES=sockets                  # Place threads on sockets
export OMP_PROC_BIND=close                 # Bind threads close together
export OMP_NESTED=false                    # Disable nested parallelism
export OMP_DYNAMIC=false                   # Disable dynamic adjustment
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0                     # Reduce blocking time
```

### BLAS Threading

```bash
export OPENBLAS_NUM_THREADS=$CORES_PER_SOCKET
export GOTO_NUM_THREADS=$CORES_PER_SOCKET
export MKL_NUM_THREADS=$CORES_PER_SOCKET
export MKL_DYNAMIC=false
```

### MPI Optimizations

```bash
mpirun -np 1 \                             # Single rank for pure performance
    --bind-to socket \                     # Bind to socket
    --map-by socket \                      # Map by socket
    --mca mpi_leave_pinned 1 \             # Keep memory pinned
    --mca btl_vader_single_copy_mechanism none \
    --report-bindings \                    # Verify core pinning
    ./benchmark_executable
```

**Why these settings matter:**
- **Core pinning**: Prevents OS from migrating threads (reduces cache misses)
- **NUMA awareness**: Allocates memory local to execution socket
- **Thread binding**: Maximizes cache locality
- **Consistent numbers**: Same settings = reproducible results

## Available Benchmarks

### Perf__IQ4_NL_GEMM

**Purpose**: Benchmark IQ4_NL quantized GEMM performance

**What it tests:**
- IQ4_NL quantized matrix multiplication (4-bit weights)
- FP32 activation path (standard precision)
- BF16 activation path (reduced memory, hardware acceleration)

**Test cases:**
1. **SingleToken_Decode**: 1 token (autoregressive decode)
   - Measures single-token latency
   - Typical for real-time inference
   
2. **SmallBatch_StandardDims**: 32 tokens (small batch)
   - Measures small batch throughput
   - Common for multi-user serving
   
3. **MediumBatch_WideProjection**: 128 tokens (wider projection)
   - Tests attention projection performance
   - Combined Q/K/V projection (896 → 2304)
   
4. **LargeBatch_Prefill**: 512 tokens (prefill scenario)
   - Large batch throughput
   - Typical for initial prompt processing

**Metrics reported:**
- **Time per iteration** (ms)
- **Throughput** (GFLOPS) = `(2 * m * n * k) / time / 1e9`
- **Memory bandwidth** (GB/s) = `(weight_compressed + activation + output) / time`
- **Speedup** (BF16 vs FP32)

**Example output:**
```
╔════════════════════════════════════════════════════════════════╗
║ Small Batch (32 tokens, 896x896)                              ║
╠════════════════════════════════════════════════════════════════╣
║ Configuration:                                                 ║
║   Sequence Length:  32                                         ║
║   Input Features:   896                                        ║
║   Output Features:  896                                        ║
║   MPI Ranks:        1                                          ║
╠════════════════════════════════════════════════════════════════╣
║ FP32 Activation Path:                                          ║
║   Time per iter:    0.523 ms                                   ║
║   Throughput:       98.45 GFLOPS                               ║
║   Bandwidth:        24.32 GB/s                                 ║
╠════════════════════════════════════════════════════════════════╣
║ BF16 Activation Path:                                          ║
║   Time per iter:    0.412 ms                                   ║
║   Throughput:       124.87 GFLOPS                              ║
║   Bandwidth:        30.89 GB/s                                 ║
╠════════════════════════════════════════════════════════════════╣
║ Speedup (BF16 vs FP32): 1.27x                                  ║
╚════════════════════════════════════════════════════════════════╝
```

## Adding New Performance Tests

### Step 1: Create Benchmark File

```cpp
// tests/v2/performance/Perf__MyFeature.cpp
#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>

class MyFeature_Perf : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        // Load models, initialize resources
    }
    
    void TearDown() override {
        // Cleanup
    }
    
    double benchmarkMyFeature() {
        // Warmup iterations
        for (int i = 0; i < warmup_iters; ++i) {
            performOperation();
        }
        
        // Timed benchmark
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < bench_iters; ++i) {
            performOperation();
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    int rank_;
};

TEST_F(MyFeature_Perf, StandardConfiguration) {
    double time_ms = benchmarkMyFeature();
    
    // Calculate metrics
    double throughput = calculateThroughput(time_ms);
    
    if (rank_ == 0) {
        std::cout << "Throughput: " << throughput << " ops/s\n";
    }
    
    EXPECT_GT(time_ms, 0.0);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
```

### Step 2: Add to CMake

Edit `tests/v2/CMakeLists.txt`:

```cmake
# Build executable
add_executable(v2_perf_my_feature performance/Perf__MyFeature.cpp)
target_link_libraries(v2_perf_my_feature 
    llaminar2_core 
    GTest::gtest 
    GTest::gtest_main
)

# Add performance test
add_v2_perf_test(V2_Perf_MyFeature
    COMMAND v2_perf_my_feature
    LABELS "V2;Performance;MyComponent;SpecificFeature"
    MPI_PROCS 1  # Or 2+ if testing MPI distribution
)
```

### Step 3: Build and Run

```bash
cd build_v2
cmake --build build_v2_release --target v2_perf_my_feature --parallel
ctest -L Performance -R "MyFeature" --verbose
```

## Best Practices

### Benchmark Design

1. **Warmup iterations**: Always include warmup to stabilize caches
   ```cpp
   for (int i = 0; i < warmup_iters; ++i) {
       performOperation();
   }
   ```

2. **MPI barriers**: Synchronize before and after timed section
   ```cpp
   MPI_Barrier(MPI_COMM_WORLD);
   auto start = now();
   // ... benchmark ...
   MPI_Barrier(MPI_COMM_WORLD);
   auto end = now();
   ```

3. **Realistic data**: Use realistic input values, not all zeros
   ```cpp
   for (size_t i = 0; i < n; ++i) {
       data[i] = (float)(i % 1000) / 1000.0f - 0.5f; // [-0.5, 0.5]
   }
   ```

4. **Multiple configurations**: Test small/medium/large workloads
   ```cpp
   TEST_F(MyPerf, SmallWorkload) { /* ... */ }
   TEST_F(MyPerf, MediumWorkload) { /* ... */ }
   TEST_F(MyPerf, LargeWorkload) { /* ... */ }
   ```

5. **Report metrics**: Calculate and display relevant metrics
   ```cpp
   double gflops = (2.0 * m * n * k) / time_ms / 1e6;
   double bandwidth_gb = total_bytes / time_ms / 1e6;
   ```

### Common Pitfalls

❌ **DON'T**:
- Run performance tests on Debug builds (10-100x slower)
- Skip warmup iterations (cold caches give inconsistent results)
- Test trivial workloads (overhead dominates)
- Use all-zero inputs (unrealistic, may enable optimizations)
- Compare across different machines without documenting specs

✅ **DO**:
- Always use Release builds (`-O3 -DNDEBUG -march=native`)
- Include sufficient warmup iterations (3-10 depending on operation)
- Test realistic workload sizes
- Use realistic activation ranges
- Document hardware specs when reporting numbers
- Use MPI barriers for accurate timing
- Report multiple metrics (time, GFLOPS, bandwidth)

## Interpreting Results

### GFLOPS (Giga Floating Point Operations Per Second)

**What it measures**: Raw computational throughput

**Formula**: `GFLOPS = (2 * m * n * k) / time_ms / 1e6`

**Typical values** (Intel Xeon, single socket):
- Small GEMM (32x896x896): 50-150 GFLOPS
- Medium GEMM (128x896x2304): 200-500 GFLOPS
- Large GEMM (512x896x896): 400-800 GFLOPS

**What to look for**:
- Larger workloads → higher GFLOPS (better cache utilization)
- Speedup with BF16 vs FP32 (hardware acceleration)
- Comparison with theoretical peak (e.g., AVX-512 FMA peak)

### Memory Bandwidth (GB/s)

**What it measures**: Data transfer rate

**Formula**: `BW = (weight_bytes + activation_bytes + output_bytes) / time_ms / 1e6`

**Typical values** (DDR4-3200):
- Single socket: 20-60 GB/s
- Dual socket: 40-100 GB/s

**What to look for**:
- Small ops: Memory-bound (low GFLOPS, high BW utilization)
- Large ops: Compute-bound (high GFLOPS, lower BW utilization)
- IQ4_NL: Reduced weight memory (4 bits/weight vs 32 bits)

### Speedup (BF16 vs FP32)

**What it measures**: Performance improvement from BF16

**Expected values**:
- No hardware acceleration: ~1.0x (same performance)
- With AVX-512 BF16: ~1.2-2.0x (hardware speedup)
- Small ops: Lower speedup (overhead dominates)
- Large ops: Higher speedup (compute dominates)

## Troubleshooting

### Performance tests are slow

**Symptom**: GFLOPS much lower than expected

**Check**:
1. Build type: `cmake -B build_v2_release -DCMAKE_BUILD_TYPE=Release`
2. Optimization flags: Should see `-O3 -DNDEBUG -march=native`
3. Core pinning: `--report-bindings` should show socket binding
4. NUMA: Check memory is local to execution socket

### Inconsistent results

**Symptom**: GFLOPS varies significantly between runs

**Check**:
1. Background processes: Close unnecessary programs
2. CPU governor: Set to `performance` mode
3. Hyperthreading: Disable or set `OMP_NUM_THREADS` to physical cores
4. Warmup: Increase warmup iterations
5. Core pinning: Verify with `--report-bindings`

### Out of memory

**Symptom**: Allocation failure in large benchmarks

**Solutions**:
- Reduce batch size or matrix dimensions
- Check available memory: `free -h`
- Ensure NUMA allocation is balanced

## Related Documentation

- **V2 Architecture**: `.github/instructions/llaminar-v2-architecture.instructions.md`
- **Benchmark Runner**: `BENCHMARK_RUNNER_GUIDE.md` (V1 but same principles)
- **CTest Labels**: `tests/v2/CMakeLists.txt` (label conventions)
- **Development Guidelines**: `.github/copilot-instructions.md`

## Contributing

When adding new performance tests:

1. **Follow naming convention**: `Perf__ComponentName.cpp`
2. **Use GTest framework**: Consistent with other V2 tests
3. **Include warmup**: Always stabilize caches first
4. **Calculate metrics**: Report GFLOPS, bandwidth, latency
5. **Test multiple sizes**: Small/medium/large workloads
6. **Document purpose**: What does this test measure and why?
7. **Label appropriately**: Use CTest label hierarchy

## Example Session

```bash
# Build Release if needed
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Run all performance tests
cd build_v2
ctest -L Performance --verbose

# Run specific benchmark
ctest -L Performance -R "IQ4_NL_GEMM" --verbose

# Run single test case
cd build_v2_release/tests/v2/performance
mpirun -np 1 --bind-to socket ./v2_perf_iq4nl_gemm --gtest_filter="*.SmallBatch*"
```

---

**Questions?** See `.github/copilot-instructions.md` or ask in project discussions.
