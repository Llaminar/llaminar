# Canonical Script Performance Comparison

**Date**: 2025-10-15  
**Build**: Release with `-march=native`  
**Model**: qwen2.5-0.5b-instruct-q8_0.gguf (Q8_0, 645MB)  
**System**: 2 sockets, 28 cores/socket, 112 logical cores (HT enabled)

## Executive Summary

Using the `run_llaminar.sh` canonical script with optimized OpenMP/MPI settings provides:
- **62% faster prefill** (8.83 → 14.28 tok/s for 8 tokens)
- **12% faster decode** (2.20 → 2.46 tok/s for 50 tokens)
- **7.9x faster prefill** for longer sequences (3.38 → 26.86 tok/s for 19 tokens)
- **13% overall improvement** in total throughput

## Detailed Performance Comparison

### Short Prompt Benchmark (8 tokens → 50 generated)

| Configuration | Prefill tok/s | Decode tok/s | Total tok/s | Prefill Speedup | Decode Speedup |
|---------------|---------------|--------------|-------------|-----------------|----------------|
| **Manual MPI** | 8.83 | 2.20 | 2.45 | baseline | baseline |
| **Canonical Script** | **14.28** | **2.46** | **2.78** | **1.62x** | **1.12x** |

**Prompt**: "Explain machine learning in simple terms."

#### Manual MPI Execution
```
Prefill:  906.36 ms (8 tokens)  →  8.83 tok/s
Decode:   22764.76 ms (50 tok) →  2.20 tok/s
Total:    23671.12 ms (58 tok) →  2.45 tok/s
```

#### Canonical Script
```
Prefill:  560.11 ms (8 tokens)  → 14.28 tok/s  ✅ 38% faster
Decode:   20310.64 ms (50 tok) →  2.46 tok/s  ✅ 12% faster
Total:    20870.76 ms (58 tok) →  2.78 tok/s  ✅ 13% faster
```

### Long Prompt Benchmark (19 tokens → 100 generated)

| Configuration | Prefill tok/s | Decode tok/s | Total tok/s | Prefill Speedup | Decode Speedup |
|---------------|---------------|--------------|-------------|-----------------|----------------|
| **Manual MPI** | 3.38 | 2.15 | 2.29 | baseline | baseline |
| **Canonical Script** | **26.86** | **2.33** | **2.73** | **7.95x** 🚀 | **1.08x** |

**Prompt**: "Write a detailed explanation of how neural networks work, including backpropagation and gradient descent algorithms."

#### Manual MPI Execution
```
Prefill:  5618.22 ms (19 tokens)  →  3.38 tok/s
Decode:   46431.96 ms (100 tok) →  2.15 tok/s
Total:    52050.18 ms (119 tok) →  2.29 tok/s
```

#### Canonical Script
```
Prefill:  707.40 ms (19 tokens)   → 26.86 tok/s  ✅ 7.9x faster! 🚀
Decode:   42950.57 ms (100 tok)  →  2.33 tok/s  ✅ 8% faster
Total:    43657.97 ms (119 tok)  →  2.73 tok/s  ✅ 19% faster
```

## Configuration Differences

### Manual MPI Command
```bash
mpirun -np 2 --bind-to socket --map-by socket \
  ./build/llaminar --benchmark ...
```

**Settings:**
- MPI: 2 processes, socket binding
- OpenMP: Default (likely uncontrolled)
- BLAS: Default threading
- No explicit CPU affinity

### Canonical Script
```bash
./run_llaminar.sh -- --benchmark ...
```

**Optimized Settings:**
- **OpenMP**: 28 threads/socket (physical cores)
  - `OMP_PLACES=sockets`
  - `OMP_PROC_BIND=close`
  - `OMP_NESTED=false`
  - `OMP_DYNAMIC=false`
- **BLAS**: 28 OpenBLAS threads (match_omp policy)
- **MPI**: Proper socket binding with thread affinity
  - `--report-bindings` shows correct pinning
- **KMP Settings**: 
  - `KMP_AFFINITY=granularity=fine,compact,1,0`
  - `KMP_BLOCKTIME=0`
- **Environment Flags**:
  - `LLAMINAR_OMP_USE_PHYSICAL=1`
  - `LLAMINAR_BIND_PER_SOCKET=1`

## Analysis

### Why Such Dramatic Prefill Improvement?

The **7.9x speedup** in prefill for longer sequences is due to:

1. **Full Thread Utilization**: 28 OpenMP threads per socket vs uncontrolled default
2. **BLAS Parallelism**: 28 OpenBLAS threads for matrix operations
3. **Proper CPU Affinity**: Threads bound to physical cores, avoiding migration
4. **Cache Locality**: `OMP_PROC_BIND=close` keeps threads near data
5. **No Hyperthreading Interference**: Physical core restriction prevents oversubscription

### Prefill Scaling Analysis

| Tokens | Manual (ms) | Canonical (ms) | Speedup | Manual tok/s | Canonical tok/s |
|--------|-------------|----------------|---------|--------------|-----------------|
| 8 | 906 | 560 | 1.62x | 8.83 | 14.28 |
| 19 | 5618 | 707 | **7.95x** | 3.38 | **26.86** |

**Key Insight**: The speedup increases dramatically with sequence length, suggesting:
- Longer sequences benefit more from parallelization
- Matrix operations (O(n²) attention) scale better with proper threading
- Fixed overhead amortized over larger computation

### Decode Improvement Modest but Consistent

Decode improved by **8-12%** across tests:
- Decode is memory-bound (KV cache access)
- Less parallelizable than prefill
- Thread affinity still helps with cache coherency
- Consistent ~430ms/token with canonical script

### Per-Token Latency

| Phase | Configuration | Latency (ms/token) |
|-------|---------------|-------------------|
| Prefill (8 tok) | Manual | 113.3 |
| Prefill (8 tok) | Canonical | **70.0** ✅ |
| Prefill (19 tok) | Manual | 295.7 |
| Prefill (19 tok) | Canonical | **37.2** ✅ |
| Decode (50 tok) | Manual | 455.3 |
| Decode (50 tok) | Canonical | **406.2** ✅ |
| Decode (100 tok) | Manual | 464.3 |
| Decode (100 tok) | Canonical | **429.5** ✅ |

## Recommendations

### Always Use Canonical Script

The `run_llaminar.sh` script should be the **default** way to run Llaminar:

```bash
# Recommended
./run_llaminar.sh -- --benchmark -m model.gguf -p "prompt" -n 100

# Also good (non-benchmark)
./run_llaminar.sh -m model.gguf -v --print-topology
```

### When to Override Settings

The script provides options for experimentation:

```bash
# Force specific thread count
./run_llaminar.sh --force-threads=16 -- --benchmark ...

# Adjust OpenBLAS policy for decode-heavy workloads
./run_llaminar.sh --openblas-policy=single -- --benchmark ...

# Use more MPI processes
./run_llaminar.sh --mpi-procs=4 -- --benchmark ...
```

### Documentation Updates

Both README.md and copilot-instructions.md now include:
- ✅ Benchmark mode usage with canonical script
- ✅ Performance expectations and tips
- ✅ Examples with `--` separator for clarity

## Conclusions

1. **Canonical script is essential**: 1.6-7.9x prefill speedup, 8-12% decode improvement
2. **Prefill scales beautifully**: Longer sequences benefit dramatically from parallelization
3. **Thread affinity matters**: Proper core binding and thread placement critical for performance
4. **BLAS threading key**: Matching BLAS threads to OpenMP prevents bottlenecks
5. **Decode remains bottleneck**: ~430ms/token regardless of optimizations (memory-bound)

### Next Optimization Targets

Based on these results:
1. **Decode phase**: Profile memory access patterns, optimize KV cache
2. **COSMA backend**: Test distributed GEMM for very large prefills
3. **Kernel fusion**: Reduce memory traffic in decode path
4. **FP16/BF16**: Consider mixed precision for bandwidth reduction

## System Details

**Detected Topology:**
- Sockets: 2
- Cores per socket: 28 physical
- Total logical cores: 112 (2x hyperthreading)
- NUMA nodes: 2

**Thread Bindings (verified via --report-bindings):**
- Rank 0: Socket 0, cores 0-27 (all hwt)
- Rank 1: Socket 1, cores 28-55 (all hwt)

**Build:**
- CMake Release mode
- `-O3 -DNDEBUG -march=native`
- GCC compiler with full optimizations
