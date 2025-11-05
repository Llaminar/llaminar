# CUDA Kernel Tuning and Profiling Guide

**Purpose**: This guide documents the complete workflow for profiling, analyzing, and optimizing CUDA kernels in Llaminar V2, with a focus on the IQ4_NL GEMM kernel used in Phase 5 parity testing.

**Last Updated**: November 4, 2025

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Organization](#file-organization)
3. [Performance Testing Workflow](#performance-testing-workflow)
4. [NCU Profiling Commands](#ncu-profiling-commands)
5. [Interpreting NCU Metrics](#interpreting-ncu-metrics)
6. [Common Optimization Patterns](#common-optimization-patterns)
7. [Case Study: Column-Major Memory Layout](#case-study-column-major-memory-layout)

---

## Architecture Overview

### CUDA GEMM System Design

The V2 CUDA GEMM system uses a **JIT compilation pipeline** with runtime optimization:

```
Weight Loading → Kernel Analysis → Compilation → Caching → Execution
     ↓                ↓                 ↓            ↓          ↓
 IQ4_NL blocks   Shape analysis    NVRTC JIT    Hash-based   Profiling
                 Heuristics        PTX gen      lookup       NCU reports
```

**Key Components**:
1. **Kernel Templates**: Parameterized CUDA source code
2. **JIT Compiler**: Runtime compilation via NVRTC
3. **Heuristic System**: Automatically selects optimal kernel parameters
4. **Cache**: Hash-based compiled kernel lookup
5. **Profiling Integration**: NCU and performance testing framework

---

## File Organization

### Core CUDA Kernel Files

**Location**: `src/v2/kernels/cuda/`

| File | Purpose | Lines | Key Contents |
|------|---------|-------|--------------|
| `CudaGemmJITPhase5.h` | JIT compilation interface | ~150 | Kernel cache, compilation API |
| `CudaGemmJITPhase5.cu` | Host code (JIT compiler) | ~400 | NVRTC compilation, module loading, launcher |
| `CudaGemmKernelTemplatePhase5.h` | **KERNEL SOURCE TEMPLATE** | ~574 | CUDA device code, optimization hotspot |
| `IQ4_NL_BlockDecoder.h` | IQ4_NL dequantization | ~150 | Block structure, dequant lookup tables |
| `IQ4_NL_BlockDecoder.cu` | Decoder implementation | ~100 | Device functions for block decode |
| `CudaGemmAutoTuner.h` | Auto-tuning framework | ~200 | ML heuristics, config selection |
| `CudaGemmAutoTuner.cu` | Auto-tuner implementation | ~300 | Neural network heuristic |

**Key File**: `CudaGemmKernelTemplatePhase5.h` contains the actual CUDA kernel template code (as a C++ string) that gets JIT-compiled and executed on the GPU.

### Testing and Profiling Files

**Location**: `tests/v2/cuda/` and `tests/v2/performance/`

| File | Purpose | Test Name | Runtime |
|------|---------|-----------|---------|
| `Test__Phase5Parity.cpp` | Phase 5 JIT kernel parity tests | `Phase5ParityTest.*` | ~12s compile + run |
| `Perf__IQ4_NL_GEMM.cpp` | Standalone GEMM benchmarks | `IQ4NLGemmPerfTest.*` | ~1-2s |
| `Perf__GemmAutoTuner.cpp` | Auto-tuner validation | `GemmAutoTunerTest.*` | Variable |

**Key Test**: `Phase5ParityTest.Phase5A_Baseline_Config` in `Test__Phase5Parity.cpp` is the primary performance benchmark.

### How Files Work Together

```
┌─────────────────────────────────────────────────────────────┐
│ Test File: Test__Phase5Parity.cpp (tests/v2/cuda/)         │
│   - Loads IQ4_NL weights from GGUF                          │
│   - Calls CudaGemmJITPhase5::compileAndLaunch()             │
│   - Measures throughput (TFLOPS)                            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Host Code: CudaGemmJITPhase5.cu                             │
│   1. Analyze kernel shape (M, N, K)                         │
│   2. Query auto-tuner for optimal config                    │
│   3. Check cache for compiled kernel                        │
│   4. If miss: JIT compile template with params              │
│   5. Launch kernel on GPU                                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Kernel Template: CudaGemmKernelTemplatePhase5.h             │
│   - Template parameters: TILE_M, TILE_N, TILE_K, SUB_K      │
│   - Device code: __global__ void iq4nl_gemm_phase5_kernel() │
│   - Memory access patterns, shared memory, CuTe MMA ops     │
│   **THIS IS WHERE YOU OPTIMIZE**                            │
└─────────────────────────────────────────────────────────────┘
```

**Compilation Flow**:
1. Test calls `CudaGemmJITPhase5::compileAndLaunch(A, B_iq4nl, C, m, n, k, config)`
2. Host code generates kernel source from template (substitutes ${TILE_M}, ${TILE_N}, etc.)
3. NVRTC compiles CUDA source → PTX → cubin
4. Kernel hash computed and cached for future reuse
5. Kernel launched with grid/block configuration
6. Results returned to host

**First Run**: ~11-12 seconds (JIT compilation overhead)  
**Cached Runs**: ~0.001 ms (hash lookup, 7.8M× speedup)

---

## Performance Testing Workflow

### Build Configuration

**Always use Release builds** for accurate performance measurement:

```bash
# From workspace root
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build_v2_release --target v2_test_phase5_parity --parallel
```

**CRITICAL**: Do NOT profile Debug builds - performance is 5-10× slower.

### Running Performance Tests

**Primary Test Command**:
```bash
cd /workspaces/llaminar/build_v2_release/tests/v2

# Run baseline configuration test (M=1024, K=896, N=896)
./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"
```

**Expected Output**:
```
[----------] 1 test from Phase5ParityTest
[ RUN      ] Phase5ParityTest.Phase5A_Baseline_Config

=== IQ4_NL GEMM JIT Compilation ===
Compilation time: 11,514.54 ms

=== Cache Lookup Performance ===
Cached kernel lookup time: 0.001468 ms
Speedup vs compilation: 7,841,748×

=== Performance Metrics ===
Timing 50 iterations...
Mean kernel time: 0.1875 ms
Throughput: 8.77 TFLOPS

=== Comparison to Phase 5A Baseline ===
JIT kernel:    8.77 TFLOPS
Phase 5A:      8.86 TFLOPS
Difference:    -0.09 TFLOPS (-1.00%)
Status:        ✓ PASS (within ±15% tolerance)
```

**Key Metrics**:
- **Compilation time**: First run only (11-12s typical)
- **Cached lookup**: 0.001-0.002 ms (subsequent runs)
- **Kernel time**: Mean execution time per iteration
- **Throughput**: TFLOPS (higher is better)
- **Comparison**: vs baseline Phase 5A implementation

### Alternative Test Configurations

```bash
# Run all Phase 5 parity tests (multiple configurations)
./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.*"

# Run standalone GEMM benchmarks (smaller overhead)
./v2_perf_iq4nl_gemm --gtest_filter="IQ4NLGemmPerfTest.*"

# Run with verbose output
./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config" --verbose
```

---

## NCU Profiling Commands

### ⚠️ CRITICAL: Always Use `sudo`

NCU requires elevated privileges for hardware performance counter access. **All `ncu` commands must be prefixed with `sudo`**.

### Creating NCU Profiles

**Basic Profiling** (recommended starting point):
```bash
cd /workspaces/llaminar/build_v2_release/tests/v2

# Profile the test and save to .ncu-rep file
sudo /usr/local/cuda/bin/ncu \
  --set full \
  --force-overwrite \
  -o phase5_baseline \
  ./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"
```

**Options Explained**:
- `--set full`: Collect comprehensive metrics (memory, compute, occupancy)
- `--force-overwrite`: Overwrite existing .ncu-rep file
- `-o <name>`: Output filename (creates `<name>.ncu-rep`)
- Final arg: Command to profile (test executable + filter)

**Targeted Profiling** (faster, specific sections):
```bash
# Profile only memory workload
sudo /usr/local/cuda/bin/ncu \
  --section MemoryWorkloadAnalysis \
  --section SourceCounters \
  -o phase5_memory_only \
  ./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"

# Profile only compute workload
sudo /usr/local/cuda/bin/ncu \
  --section ComputeWorkloadAnalysis \
  --section Occupancy \
  -o phase5_compute_only \
  ./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"
```

**Profile Kernel Launches** (limit to specific kernel):
```bash
# Profile only kernels matching name pattern
sudo /usr/local/cuda/bin/ncu \
  --kernel-name iq4nl_gemm_phase5_kernel \
  --set full \
  -o phase5_kernel_only \
  ./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"
```

**Expected Runtime**: 
- Full profiling: ~5-10 minutes (70 kernel launches × 40 passes each)
- Section-specific: ~1-3 minutes
- Kernel-specific: ~30 seconds

### Viewing NCU Reports

**Default Summary Report**:
```bash
# View comprehensive summary of all metrics
sudo /usr/local/cuda/bin/ncu --import phase5_baseline.ncu-rep 2>&1 | less
```

**Specific Sections**:
```bash
# Memory workload analysis
sudo /usr/local/cuda/bin/ncu \
  --import phase5_baseline.ncu-rep \
  --page details \
  --section MemoryWorkloadAnalysis

# Compute workload analysis
sudo /usr/local/cuda/bin/ncu \
  --import phase5_baseline.ncu-rep \
  --page details \
  --section ComputeWorkloadAnalysis

# Occupancy metrics
sudo /usr/local/cuda/bin/ncu \
  --import phase5_baseline.ncu-rep \
  --page details \
  --section Occupancy

# Source code annotations (uncoalesced accesses, bank conflicts)
sudo /usr/local/cuda/bin/ncu \
  --import phase5_baseline.ncu-rep \
  --page details \
  --section SourceCounters
```

### Exporting Metrics to CSV

**Export All Metrics**:
```bash
# Export raw metrics for scripting/analysis
sudo /usr/local/cuda/bin/ncu \
  --import phase5_baseline.ncu-rep \
  --page raw \
  --csv > ncu_metrics.csv
```

**CSV Structure**:
```
ID,Process ID,Process Name,Host Name,Kernel Name,...
,,,,,"ID","Process ID",...   # Units row (skip when parsing)
0,315028,v2_test_phase5_parity,127.0.0.1,iq4nl_gemm_phase5_kernel,...
1,315028,v2_test_phase5_parity,127.0.0.1,iq4nl_gemm_phase5_kernel,...
```

**Parsing Example** (Python):
```python
import csv

with open('ncu_metrics.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)  # First row: column names
    units = next(reader)     # Second row: units (skip)
    
    # Find metric column indices
    global_load_idx = headers.index('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
    ideal_sectors_idx = headers.index('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_second')
    
    for row in reader:
        if not row[0]:  # Skip summary rows
            continue
        
        global_sectors = int(row[global_load_idx])
        print(f"Kernel {row[0]}: {global_sectors:,} global load sectors")
```

**Query Specific Metrics**:
```bash
# List all available metrics
sudo /usr/local/cuda/bin/ncu --query-metrics

# Export only specific metrics
sudo /usr/local/cuda/bin/ncu \
  --import phase5_baseline.ncu-rep \
  --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum \
  --csv > specific_metrics.csv
```

---

## Interpreting NCU Metrics

### Top-Level Summary Sections

When you run `sudo /usr/local/cuda/bin/ncu --import <file>.ncu-rep`, you'll see these sections:

#### 1. GPU Speed Of Light Throughput

```
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
Memory Throughput                 %        61.52
Compute (SM) Throughput           %        53.89
```

**What This Tells You**:
- **Memory Throughput %**: How close to peak memory bandwidth (higher = more memory-bound)
- **Compute Throughput %**: How close to peak compute (higher = more compute-bound)

**Rule of Thumb**:
- Both ~50-60%: **Balanced** (need to optimize both)
- Memory > 80%: **Memory-bound** (optimize memory access)
- Compute > 80%: **Compute-bound** (optimize math operations)
- Both < 50%: **Underutilized** (occupancy or stall issues)

**NCU Guidance**:
```
INF   Compute and Memory are well-balanced: To reduce runtime, 
      both computation and memory traffic must be reduced. Check 
      both the Compute Workload Analysis and Memory Workload 
      Analysis sections.
```

#### 2. Compute Workload Analysis

```
Metric Name          Metric Unit Metric Value
-------------------- ----------- ------------
Executed Ipc Active   inst/cycle         0.80
SM Busy                        %        24.96
Issue Slots Busy               %        19.94
```

**Key Metrics**:
- **SM Busy %**: Percentage of time SMs are executing instructions
  - <25%: Low utilization, check occupancy
  - 25-50%: Moderate, likely stalls
  - >75%: High utilization, compute-bound
  
- **Executed IPC (Active)**: Instructions per cycle during active execution
  - <0.5: Very low throughput
  - 0.5-1.0: Typical for memory-bound kernels
  - >1.0: Good instruction-level parallelism

**NCU Guidance Example**:
```
INF   Tensor is the highest-utilized pipeline (25.0%) based on 
      active cycles. It is well-utilized, but should not be a 
      bottleneck.
```

This tells you that tensor cores (WMMA operations) are being used but aren't saturated.

#### 3. Memory Workload Analysis

```
Metric Name                  Metric Unit Metric Value
---------------------------- ----------- ------------
Memory Throughput                Gbyte/s        28.65
Mem Busy                               %        61.52
Max Bandwidth                          %        53.89
L1/TEX Hit Rate                        %        88.88
L2 Hit Rate                            %        94.27
```

**Key Metrics**:
- **L1/TEX Hit Rate**: Percentage of L1 cache hits
  - >85%: Excellent cache locality
  - 60-85%: Good
  - <60%: Poor locality, check access patterns
  
- **L2 Hit Rate**: Percentage of L2 cache hits
  - >90%: Excellent
  - 70-90%: Good
  - <70%: High memory traffic
  
- **Max Bandwidth %**: Achieved bandwidth vs theoretical peak
  - >80%: Memory-bound
  - 50-80%: Balanced
  - <50%: Not memory-bound

**Critical**: High cache hit rates but low bandwidth → not memory-bound, look elsewhere.

#### 4. Memory Workload Analysis Tables (Optimization Hints)

NCU automatically identifies optimization opportunities:

```
OPT   Est. Speedup: 65.84%
      This kernel has uncoalesced global accesses resulting in 
      18,278,400 excessive sectors (90% of total 20,299,776).
```

**What "Excessive Sectors" Means**:
- **Ideal sectors**: Minimum number of 32-byte sectors needed to load requested data
- **Actual sectors**: Total sectors actually loaded from memory
- **Excessive sectors**: Actual - Ideal (wasted bandwidth)
- **Excessive %**: (Excessive / Actual) × 100

**Example**:
```
Global load sectors:     20,040,123
Ideal sectors:            2,021,376
Excessive sectors:       18,018,747
Excessive %:                    90%
```

This means **90% of memory traffic is wasted** due to poor coalescing!

**Shared Memory Bank Conflicts**:
```
OPT   Est. Speedup: 58.91%
      The memory access pattern for shared stores might not be 
      optimal and causes on average a 9.6-way bank conflict 
      across all 501,760 shared store requests.
```

A **9.6-way bank conflict** means 9.6 threads are accessing the same memory bank simultaneously, serializing the accesses (9.6× slower than ideal).

#### 5. Occupancy

```
Metric Name                     Metric Unit Metric Value
------------------------------- ----------- ------------
Theoretical Occupancy                     %        33.33
Achieved Occupancy                        %        19.97
Block Limit Registers                 block            4
Block Limit Shared Mem                block            4
```

**Key Metrics**:
- **Theoretical Occupancy**: Maximum occupancy given resource limits
- **Achieved Occupancy**: Actual occupancy during execution
- **Block Limit**: What resource limits block count
  - Registers: Register pressure (reduce per-thread usage)
  - Shared Memory: Shared memory usage (reduce or optimize)
  - Warps: Warp limit (not usually the issue)

**Occupancy Impact**:
- <20%: Very low, likely poor performance
- 20-40%: Low, but may be acceptable for memory-bound kernels
- 40-60%: Moderate
- >60%: Good

**Gap Between Theoretical and Achieved**:
```
OPT   Est. Speedup: 38.48%
      The difference between theoretical (33.3%) and achieved 
      (20.0%) occupancy can be the result of warp scheduling 
      overheads or workload imbalances.
```

Large gap (>10%) suggests:
- Warp divergence (if/else causing different execution paths)
- Load imbalance between thread blocks
- Long-latency operations causing stalls

#### 6. Scheduler Statistics

```
Metric Name                  Metric Unit Metric Value
---------------------------- ----------- ------------
One or More Eligible                   %        19.94
No Eligible                            %        80.06
Issued Warp Per Scheduler                        0.20
Active Warps Per Scheduler          warp         2.40
Eligible Warps Per Scheduler        warp         0.26
```

**Critical Metric**: **No Eligible %**
- This is the percentage of cycles where **no warp was ready to execute**
- 80% means the GPU is idle 80% of the time waiting for data/dependencies!

**Why Warps Are Not Eligible**:
- Memory latency (waiting for global/shared loads)
- Dependency chains (waiting for previous instructions)
- Execution dependencies (waiting for other warps)

**Issued Warp Per Scheduler**:
- Ideal: 1.0 (one instruction issued per cycle)
- 0.20: Only issuing 1 instruction every 5 cycles (20% efficiency)

#### 7. Source Counters (Most Actionable)

```
OPT   Est. Speedup: 65.84%
      This kernel has uncoalesced global accesses resulting in 
      18,278,400 excessive sectors (90% of total 20,299,776).
      Check the L2 Theoretical Sectors Global Excessive table.
```

**This section provides**:
- **Estimated speedup** if you fix the issue
- **Root cause** (uncoalesced accesses, bank conflicts, divergence)
- **Source location** (which lines of code to fix)

**Common Issues Reported**:

1. **Uncoalesced Global Accesses**:
   ```
   18,278,400 excessive sectors (90% of total)
   Est. Speedup: 65.84%
   ```
   Fix: Ensure consecutive threads access consecutive memory addresses

2. **Uncoalesced Shared Accesses**:
   ```
   4,214,784 excessive wavefronts (38% of total)
   Est. Speedup: 34.24%
   ```
   Fix: Avoid bank conflicts with padding or access pattern changes

3. **Branch Divergence**:
   ```
   Avg. Divergent Branches: 2.4
   Branch Efficiency: 45%
   ```
   Fix: Minimize if/else within warps, use predication

### Metric Priority for Optimization

**Prioritize by Est. Speedup** (NCU tells you!):

1. **Uncoalesced global accesses** (65.84% speedup) → Fix memory layout
2. **Shared memory bank conflicts** (58.91% speedup) → Pad shared arrays
3. **Global load pattern** (55.49% speedup) → Improve tile sizes
4. **Scheduler stalls** (38.48% speedup) → Improve occupancy
5. **Occupancy gap** (38.48% speedup) → Reduce registers/shared memory

**Don't optimize blindly** - follow NCU's estimated speedup rankings!

---

## Common Optimization Patterns

### 1. Fixing Uncoalesced Global Memory Accesses

**Problem**: Threads in a warp access non-consecutive memory addresses.

**Example Bad Pattern** (row-major A-matrix, column-major access):
```cuda
// Each thread loads different row (strided by M)
float a_val = A[gk * M + gm];  // gk same for all threads, gm varies
// Thread 0: A[gk * M + 0]
// Thread 1: A[gk * M + 1]
// ...
// Thread 31: A[gk * M + 31]
// If M >> 32, these are far apart in memory!
```

**Solution 1**: Change memory layout (column-major → row-major):
```cuda
// Transpose A so consecutive threads access consecutive elements
// Now A is row-major: A[K][M] → A[M][K]
float a_val = A[gm * K + gk];  // gm varies, consecutive in memory
```

**Solution 2**: Change access pattern (use shared memory):
```cuda
// Cooperatively load tile into shared memory with coalesced access
__shared__ float As[TILE_K][TILE_M];

// Coalesced load (consecutive threads → consecutive addresses)
As[ty][tx] = A[block_k * TILE_K + ty][block_m * TILE_M + tx];
__syncthreads();

// Now access As with any pattern (already in shared memory)
float a_val = As[k_local][m_local];
```

### 2. Fixing Shared Memory Bank Conflicts

**Problem**: Multiple threads in a warp access the same memory bank simultaneously.

**Background**: Shared memory has 32 banks. Addresses map to banks as:
```
bank = (address / 4) % 32  // 4-byte words, 32 banks
```

**Example Bad Pattern** (column-major shared array):
```cuda
__shared__ float As[TILE_K][TILE_M];  // TILE_M = 128

// All threads in warp access same column (same K index)
for (int k = 0; k < TILE_K; k++) {
    float a_val = As[k][tx];  // tx = 0..31, all access As[k][0..31]
}

// Bank assignment:
// As[k][0] → bank 0
// As[k][1] → bank 1
// ...
// As[k][31] → bank 31
// As[k][32] → bank 0  ← CONFLICT with As[k][0]!
```

**Solution**: Pad shared memory to avoid conflicts:
```cuda
// Pad by +1 to offset bank mapping
__shared__ float As[TILE_K][TILE_M + 1];  // Extra column

// Now bank assignment:
// As[k][0] → bank 0
// As[k][1] → bank 1
// ...
// As[k][32] → bank 1  ← No conflict! (shifted by padding)
```

**Alternative**: Transpose access pattern:
```cuda
// Access rows instead of columns
for (int m = 0; m < TILE_M; m++) {
    float a_val = As[tx][m];  // Consecutive threads access consecutive rows
}
```

### 3. Improving Occupancy

**Problem**: Low occupancy due to high register or shared memory usage.

**Check Limiters**:
```bash
# View occupancy section
sudo /usr/local/cuda/bin/ncu \
  --import profile.ncu-rep \
  --section Occupancy
```

**Example Output**:
```
Block Limit Registers:  4 blocks/SM
Block Limit Shared Mem: 4 blocks/SM
Block Limit Warps:      12 blocks/SM
Theoretical Occupancy:  33.33%
```

**Solution 1**: Reduce register usage:
```cuda
// Before: Many intermediate variables (high register pressure)
float a1 = ..., a2 = ..., a3 = ...;
float b1 = ..., b2 = ..., b3 = ...;
float c1 = ..., c2 = ..., c3 = ...;

// After: Reuse variables, compute on-the-fly
float temp;
temp = ...; c1 += temp;
temp = ...; c2 += temp;
```

**Solution 2**: Reduce shared memory:
```cuda
// Before: Large shared memory tiles
__shared__ float As[128][128];  // 64 KB
__shared__ float Bs[128][128];  // 64 KB
// Total: 128 KB → limits to 1 block/SM on some GPUs

// After: Smaller tiles
__shared__ float As[64][64];   // 16 KB
__shared__ float Bs[64][64];   // 16 KB
// Total: 32 KB → allows 4 blocks/SM
```

**Solution 3**: Adjust block size:
```cuda
// Before: Large blocks (high resource usage per block)
dim3 block(256, 1, 1);  // 256 threads/block

// After: Smaller blocks (more blocks can fit)
dim3 block(128, 1, 1);  // 128 threads/block
```

**Check with CUDA Occupancy Calculator**:
```bash
# Estimate occupancy for different configurations
nvcc --resource-usage kernel.cu

# Or use NCU's occupancy section
```

### 4. Reducing Scheduler Stalls

**Problem**: High "No Eligible %" means warps are waiting for dependencies.

**Root Causes**:
1. **Memory latency**: Warps waiting for global/shared loads
2. **Dependency chains**: Instructions waiting for previous results
3. **Synchronization**: `__syncthreads()` causing all warps to wait

**Solution 1**: Increase occupancy (more warps hide latency):
```cuda
// More active warps mean some can execute while others wait
// See "Improving Occupancy" above
```

**Solution 2**: Software pipelining (overlap compute and memory):
```cuda
// Before: Load → Sync → Compute → Repeat (stalls)
for (int tile = 0; tile < num_tiles; tile++) {
    // Load tile
    As[ty][tx] = A_global[...];
    __syncthreads();
    
    // Compute on tile
    for (int k = 0; k < TILE_K; k++) {
        acc += As[k][tx] * Bs[k][ty];
    }
    __syncthreads();
}

// After: Load next while computing current (overlap)
// Pre-load first tile
As[ty][tx] = A_global[tile=0];
__syncthreads();

for (int tile = 0; tile < num_tiles - 1; tile++) {
    // Start loading next tile (async if possible)
    float next_As = A_global[tile+1];  // Start load
    
    // Compute on current tile (while next loads)
    for (int k = 0; k < TILE_K; k++) {
        acc += As[k][tx] * Bs[k][ty];
    }
    
    // Store next tile
    __syncthreads();
    As[ty][tx] = next_As;
    __syncthreads();
}

// Compute last tile
for (int k = 0; k < TILE_K; k++) {
    acc += As[k][tx] * Bs[k][ty];
}
```

**Solution 3**: Use `__pipeline` primitives (SM 8.0+):
```cuda
#include <cuda/pipeline>

// Asynchronous global → shared memory copy
__pipeline_memcpy_async(&As[ty][tx], &A_global[...], sizeof(float));
__pipeline_commit();

// Compute while async copy happens
// ...

// Wait for async copy to complete
__pipeline_wait_prior(0);
__syncthreads();
```

### 5. Optimizing WMMA (Tensor Core) Usage

**Problem**: Low tensor core utilization despite using WMMA.

**Check WMMA Coverage**:
```bash
sudo /usr/local/cuda/bin/ncu \
  --import profile.ncu-rep \
  --section ComputeWorkloadAnalysis
```

Look for:
```
INF   Tensor is the highest-utilized pipeline (25.0%)
```

**If Tensor Utilization < 50%**:

1. **Ensure proper tile sizes** (WMMA requires specific dimensions):
   ```cuda
   // WMMA requires multiples of 16 for M, N, K
   // Optimal: 16, 32, 64, 128
   #define WMMA_M 16
   #define WMMA_N 16
   #define WMMA_K 16
   ```

2. **Maximize WMMA operations per thread block**:
   ```cuda
   // More WMMA ops per block = better tensor core utilization
   // Increase TILE_M and TILE_N (must be multiples of 16)
   #define TILE_M 128  // 8 WMMA tiles (128/16)
   #define TILE_N 128  // 8 WMMA tiles
   ```

3. **Reduce non-WMMA overhead**:
   ```cuda
   // Minimize scalar operations between WMMA calls
   // Use WMMA for accumulation, not manual loops
   
   // Before: Manual accumulation (slow)
   for (int k = 0; k < K; k++) {
       acc += A[k] * B[k];
   }
   
   // After: WMMA accumulation (fast)
   wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
   ```

---

## Case Study: Column-Major Memory Layout

### Motivation

**Baseline Performance** (row-major A-matrix, row-major access):
- Throughput: 8.86 TFLOPS
- Excessive sectors (L1): Unknown (not profiled initially)
- Problem: Suspected memory coalescing issues

**Hypothesis**: Column-major activations (A-matrix) would improve coalescing because threads naturally access consecutive M indices.

### Implementation

**Change Made** (in `CudaGemmKernelTemplatePhase5.h`):

```cuda
// BEFORE: Row-major A-matrix (A[M][K])
// Thread access pattern: A[gm][gk]
//   Thread 0: A[0][gk]
//   Thread 1: A[1][gk]
//   ...
//   Thread 31: A[31][gk]
// Memory addresses: A[0][gk], A[1][gk], ..., A[31][gk]
// Stride: K elements (likely >> 32, poor coalescing)

// AFTER: Column-major A-matrix (A[K][M])
// Thread access pattern: A[gk][gm]
//   Thread 0: A[gk][0]
//   Thread 1: A[gk][1]
//   ...
//   Thread 31: A[gk][31]
// Memory addresses: A[gk][0], A[gk][1], ..., A[gk][31]
// Stride: 1 element (perfect coalescing!)
```

**Test Data Preparation** (also changed to column-major in `Test__Phase5Parity.cpp`):
```cpp
// In test file: Transpose A-matrix before kernel launch
std::vector<float> A_colmajor(m * k);
for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
        A_colmajor[j * m + i] = A_rowmajor[i * k + j];  // Transpose
    }
}
```

### Profiling Results

**NCU Profiling Command**:
```bash
cd /workspaces/llaminar/build_v2_release/tests/v2

sudo /usr/local/cuda/bin/ncu \
  --set full \
  --force-overwrite \
  -o phase5_colmajor_final \
  ./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"
```

**Extracting Metrics**:
```bash
# Export to CSV for analysis
sudo /usr/local/cuda/bin/ncu \
  --import phase5_colmajor_final.ncu-rep \
  --page raw \
  --csv > ncu_metrics.csv

# View memory workload analysis
sudo /usr/local/cuda/bin/ncu \
  --import phase5_colmajor_final.ncu-rep \
  --page details \
  --section MemoryWorkloadAnalysis
```

**Key Findings**:

| Metric | Baseline (Row-Major) | Column-Major | Change |
|--------|---------------------|--------------|--------|
| **Throughput** | 8.86 TFLOPS | 8.77 TFLOPS | -1.0% ❌ |
| **L1 Excessive Sectors** | ~81% (estimated) | **0.0%** | -81pp ✅ |
| **L2 Excessive Sectors** | Unknown | **90%** | ❌ |
| **Global Load Sectors** | ~10.4M | 20.0M | +92.7% |
| **L1 Hit Rate** | Unknown | 88.88% | ✅ |
| **L2 Hit Rate** | Unknown | 94.27% | ✅ |
| **Memory Bandwidth** | Unknown | 53.89% | (Not saturated) |

### Analysis

**What Worked** ✅:
- L1 coalescing: **Perfect** (0% excessive sectors at L1)
- L1 hit rate: **Excellent** (88.88%)
- L2 hit rate: **Excellent** (94.27%)

**What Didn't Work** ❌:
- Performance: **No improvement** (-1.0%)
- L2 excessive sectors: **Worse** (90% excessive at L2!)
- Total sectors loaded: **2× more** (20M vs 10.4M)

**Root Cause**:
- Fixing L1 coalescing **created L2 coalescing problems**
- Column-major layout requires reading entire columns (M=1024 elements)
- Each column spans many cache lines → more L2 misses
- Memory bandwidth only 54% utilized → **not memory-bound!**

**Key Lesson**: 
> "Memory coalescing at L1 is NOT the same as overall memory efficiency. Optimizing one level can hurt another. Always profile the FULL memory hierarchy."

### Next Steps After This Discovery

Based on NCU source counters, the **real bottlenecks** were identified:

1. **Shared memory bank conflicts** (58.91% potential speedup)
   - 9.6-way conflicts on shared stores
   - 87.43% of wavefronts have conflicts
   
2. **Uncoalesced global accesses at L2** (65.84% potential speedup)
   - 90% excessive sectors at L2
   - Poor byte utilization (3/32 bytes used per sector)

3. **Scheduler stalls** (38.48% potential speedup)
   - 80% of cycles with no eligible warps
   - Only issuing 1 instruction per 5 cycles

**Recommended Next Optimization**: Fix shared memory bank conflicts (highest NCU-estimated speedup).

### Reverting Changes

Since column-major didn't improve performance:

```bash
# Revert kernel template to row-major
git checkout src/v2/kernels/cuda/CudaGemmKernelTemplatePhase5.h

# Revert test data preparation
git checkout tests/v2/cuda/Test__Phase5Parity.cpp
```

**Lesson**: Not all optimizations improve performance. Profile first, optimize based on data, verify results.

---

## Iteration Checklist

When optimizing a CUDA kernel, follow this workflow:

### 1. Baseline Measurement
- [ ] Build Release configuration
- [ ] Run performance test (record TFLOPS)
- [ ] Profile with NCU (`--set full`)
- [ ] Export metrics to CSV
- [ ] Identify top 3 optimization opportunities (by Est. Speedup)

### 2. Hypothesis Formation
- [ ] Choose optimization based on NCU recommendations
- [ ] Understand root cause (coalescing, bank conflicts, occupancy)
- [ ] Predict expected improvement
- [ ] Document hypothesis in changelog/markdown

### 3. Implementation
- [ ] Modify kernel template (`IQ4_NL_GemmKernel_Template.cu`)
- [ ] Update test data if needed (e.g., transpose matrices)
- [ ] Rebuild: `cmake --build build_v2_release --target v2_test_phase5_parity`

### 4. Verification
- [ ] Run performance test (measure new TFLOPS)
- [ ] Profile with NCU (same options as baseline)
- [ ] Compare metrics (CSV diff or NCU comparison)
- [ ] Check if optimization achieved expected speedup

### 5. Analysis
- [ ] Did performance improve? (Yes/No)
- [ ] Did target metric improve? (e.g., excessive sectors ↓)
- [ ] Did any other metrics degrade? (e.g., L2 hit rate ↓)
- [ ] What is the new bottleneck? (check NCU Est. Speedup rankings)

### 6. Decision
- [ ] **If improved**: Commit changes, document in changelog
- [ ] **If degraded**: Revert changes, document lesson learned
- [ ] **If neutral**: Investigate why, revert or iterate

### 7. Documentation
- [ ] Update changelog with results
- [ ] Document any unexpected findings
- [ ] Note next optimization target

---

## Troubleshooting

### NCU Fails with "Permission Denied"

**Problem**: `ncu: error: unable to open performance counters`

**Solution**: Always use `sudo`:
```bash
sudo /usr/local/cuda/bin/ncu ...
```

### NCU Reports "No Metrics to Show"

**Problem**: Requesting chart/graph sections in CLI

**Solution**: Charts only available in GUI. Use `--page details --section <name>` instead:
```bash
# Wrong (fails)
sudo /usr/local/cuda/bin/ncu --section MemoryWorkloadAnalysis_Chart ...

# Right (works)
sudo /usr/local/cuda/bin/ncu --section MemoryWorkloadAnalysis ...
```

### Test Hangs During Profiling

**Problem**: NCU profiling takes very long (>10 minutes)

**Causes**:
- `--set full` profiles 70 kernel launches × 40 passes = 2800 iterations
- Full metric collection is comprehensive but slow

**Solutions**:
```bash
# Option 1: Profile specific sections only
sudo /usr/local/cuda/bin/ncu \
  --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis \
  -o profile.ncu-rep \
  ./test

# Option 2: Profile fewer kernel launches
sudo /usr/local/cuda/bin/ncu \
  --kernel-name iq4nl_gemm_phase5_kernel \
  --launch-count 10 \
  -o profile.ncu-rep \
  ./test

# Option 3: Use --metrics instead of --set full
sudo /usr/local/cuda/bin/ncu \
  --metrics smsp__cycles_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  -o profile.ncu-rep \
  ./test
```

### Compilation Time is Too Long

**Problem**: First test run takes 11-12 seconds

**Explanation**: This is **JIT compilation overhead** (normal behavior)

**Verification**:
- First run: ~11-12 seconds (compiling)
- Second run: ~0.001 ms (cached lookup, 7.8M× faster)

**If compilation fails**:
```bash
# Check NVRTC availability
nvcc --version

# Check CUDA toolkit path
ls -l /usr/local/cuda/bin/nvcc

# Verify CUDA_HOME
echo $CUDA_HOME
```

### Performance Degrades After Optimization

**Problem**: Throughput decreased after applying "optimization"

**Analysis**:
1. Compare NCU metrics side-by-side (baseline vs optimized)
2. Check if target metric improved (e.g., excessive sectors)
3. Look for regressions in other metrics (e.g., cache hit rates)
4. Review NCU Est. Speedup rankings (may have changed)

**Example** (Column-Major Case Study):
- L1 excessive: 81% → 0% ✅ (improved)
- L2 excessive: 0% → 90% ❌ (regressed!)
- Net effect: -1.0% performance ❌

**Lesson**: Some optimizations trade one bottleneck for another. Always verify with profiling.

---

## Quick Reference

### Essential Commands

```bash
# Build Release
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --target v2_test_phase5_parity --parallel

# Run performance test
cd build_v2_release/tests/v2
./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"

# Profile with NCU (full)
sudo /usr/local/cuda/bin/ncu --set full -o profile ./v2_test_phase5_parity --gtest_filter="..."

# View NCU report
sudo /usr/local/cuda/bin/ncu --import profile.ncu-rep 2>&1 | less

# Export metrics
sudo /usr/local/cuda/bin/ncu --import profile.ncu-rep --csv > metrics.csv
```

### Key Files

| File | Purpose |
|------|---------|
| `src/v2/kernels/cuda/CudaGemmKernelTemplatePhase5.h` | **Kernel source template (optimize here)** |
| `src/v2/kernels/cuda/CudaGemmJITPhase5.cu` | Host code (JIT, cache, launch) |
| `tests/v2/cuda/Test__Phase5Parity.cpp` | Phase 5 parity tests |
| `tests/v2/performance/Perf__IQ4_NL_GEMM.cpp` | Standalone GEMM benchmarks |
| `changelog/YYYY-MM-DD-*.md` | Document findings here |

### NCU Sections

| Section | Purpose |
|---------|---------|
| `MemoryWorkloadAnalysis` | Memory throughput, cache hits, bandwidth |
| `ComputeWorkloadAnalysis` | SM utilization, IPC, pipeline usage |
| `Occupancy` | Theoretical vs achieved, resource limits |
| `SourceCounters` | **Optimization hints (start here!)** |
| `SchedulerStatistics` | Warp scheduling, stalls |

### Metric Interpretation Quick Guide

| Metric | Good | Bad | Action |
|--------|------|-----|--------|
| Memory Throughput | <60% | >80% | If high: optimize memory |
| Compute Throughput | >60% | <40% | If low: check occupancy |
| L1 Hit Rate | >85% | <60% | If low: improve locality |
| Excessive Sectors | <10% | >50% | If high: fix coalescing |
| Achieved Occupancy | >50% | <30% | If low: reduce resources |
| No Eligible % | <40% | >70% | If high: hide latency |
| SM Busy % | >60% | <30% | If low: improve occupancy |

---

## References

- **NVIDIA NCU Documentation**: https://docs.nvidia.com/nsight-compute/
- **CUDA C++ Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Best Practices Guide**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **Nsight Compute CLI**: https://docs.nvidia.com/nsight-compute/NsightComputeCli/

---

## Changelog

- **2025-11-04**: Initial version based on Phase 5 column-major profiling session
  - Documented NCU profiling workflow
  - Added metric interpretation guide
  - Included column-major case study
  - Created optimization pattern library

