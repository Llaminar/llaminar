# MPI Kernel OpenMP Parallelization Plan

## Executive Summary

Currently, Llaminar runs single-threaded within each MPI process, leaving significant compute resources unused. This document outlines a comprehensive plan to implement hybrid MPI+OpenMP parallelization to utilize all available cores within each socket.

## Current State Analysis

### Threading Architecture
- **Current**: 2 MPI processes × 1 thread = 2 cores utilized per socket
- **Target**: 2 MPI processes × 8 threads = 16 cores utilized per socket
- **OpenMP Status**: ✅ Available and linked (`libgomp.so.1`) but not used in our code
- **COSMA Integration**: ✅ COSMA library already uses OpenMP internally

### Performance Bottlenecks

#### 1. Element-wise Operations (High Impact - 5% of total time)
**Location**: `src/mpi_transformer_pipeline.cpp`
```cpp
// Residual connections (single-threaded)
for (size_t i = 0; i < input->total_elements(); ++i) {
    residual_data[i] = input_data[i] + attn_data[i];
}

// SwiGLU activation (single-threaded)
for (size_t i = 0; i < gate_out->total_elements(); ++i) {
    float x = up_data[i];
    float silu = x / (1.0f + std::exp(-x));
    gate_data[i] *= silu;
}
```

#### 2. Attention Computation (High Impact)
**Location**: `src/kernels/MPIAttentionKernel.cpp`
```cpp
// Attention scores computation (nested loops)
for (int head = 0; head < local_heads; ++head) {
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            for (int d = 0; d < head_dim_; ++d) {
                score += q_row[d] * k_row[d];
            }
        }
    }
}
```

#### 3. RoPE Computation (Medium Impact)
**Location**: `src/kernels/MPIAttentionKernel.cpp`
```cpp
// Rotary embeddings (single-threaded)
for (int head = 0; head < local_heads; ++head) {
    for (size_t seq = 0; seq < seq_len; ++seq) {
        for (int dim_pair = 0; dim_pair < head_dim_ / 2; ++dim_pair) {
            // Rotation computations
        }
    }
}
```

#### 4. Data Movement and Bias Addition (Medium Impact)
**Location**: `src/kernels/MPILinearKernel.cpp`
```cpp
// Bias addition (single-threaded)
for (size_t i = 0; i < seq_len; ++i) {
    for (size_t j = 0; j < local_output_size; ++j) {
        output[i * local_output_size + j] += bias[j];
    }
}
```

## Implementation Strategy

### Phase 1: Element-wise Operations (Immediate Impact)

**Target Files:**
- `src/mpi_transformer_pipeline.cpp`
- `src/kernels/MPILinearKernel.cpp`
- `src/kernels/MPIRMSNormKernel.cpp`

**Implementation:**
```cpp
// Residual connections with OpenMP
#pragma omp parallel for simd
for (size_t i = 0; i < input->total_elements(); ++i) {
    residual_data[i] = input_data[i] + attn_data[i];
}

// SwiGLU activation with OpenMP
#pragma omp parallel for simd
for (size_t i = 0; i < gate_out->total_elements(); ++i) {
    float x = up_data[i];
    float silu = x / (1.0f + std::exp(-x));
    gate_data[i] *= silu;
}

// Bias addition with OpenMP
#pragma omp parallel for
for (size_t i = 0; i < seq_len; ++i) {
    #pragma omp simd
    for (size_t j = 0; j < local_output_size; ++j) {
        output[i * local_output_size + j] += bias[j];
    }
}
```

**Expected Gains:** 4-8x speedup for element-wise operations

### Phase 2: Attention Parallelization (High Impact)

**Target Files:**
- `src/kernels/MPIAttentionKernel.cpp`

**Implementation:**
```cpp
// Parallel attention computation
#pragma omp parallel for collapse(2)
for (int head = 0; head < local_heads; ++head) {
    for (size_t i = 0; i < seq_len; ++i) {
        #pragma omp simd
        for (size_t j = 0; j < seq_len; ++j) {
            // Vectorized dot product computation
        }
    }
}

// Parallel softmax computation
#pragma omp parallel for
for (int head = 0; head < local_heads; ++head) {
    for (size_t i = 0; i < seq_len; ++i) {
        // Per-head, per-position softmax
    }
}
```

**Expected Gains:** 2-4x speedup for attention computation

### Phase 3: RoPE and Data Movement (Medium Impact)

**Target Files:**
- `src/kernels/MPIAttentionKernel.cpp`
- `src/kernels/MPILinearKernel.cpp`

**Implementation:**
```cpp
// Parallel RoPE computation
#pragma omp parallel for collapse(3)
for (int head = 0; head < local_heads; ++head) {
    for (size_t seq = 0; seq < seq_len; ++seq) {
        for (int dim_pair = 0; dim_pair < head_dim_ / 2; ++dim_pair) {
            // Parallel dimension rotation
        }
    }
}

// Parallel weight distribution
#pragma omp parallel for
for (size_t i = 0; i < d_model; ++i) {
    // Parallel data copying and distribution
}
```

**Expected Gains:** 2-3x speedup for RoPE and data movement

## Technical Considerations

### Thread Safety Requirements

1. **MPI Thread Support**: Already configured with `MPI_THREAD_MULTIPLE`
2. **COSMA Integration**: COSMA is already thread-safe and OpenMP-optimized
3. **Tensor Access**: SimpleTensor operations are inherently thread-safe for read-only and independent writes

### NUMA Optimization Strategy

```cpp
// Thread binding for NUMA awareness
void setupNUMAThreadAffinity() {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int mpi_rank = getRank();
        
        // Bind threads to cores within same NUMA node as MPI process
        int socket_id = mpi_rank;  // Assuming 1 MPI process per socket
        bindThreadToSocket(thread_id, socket_id);
    }
}
```

### Performance Monitoring

```cpp
// Add OpenMP timing instrumentation
double start_time = omp_get_wtime();
#pragma omp parallel for
for (...) {
    // Parallel computation
}
double end_time = omp_get_wtime();
LOG_DEBUG("OpenMP section took: " << (end_time - start_time) * 1000 << " ms");
```

## Expected Performance Improvements

### Overall Pipeline Impact
- **Element-wise operations**: 4-8x speedup (perfect parallel efficiency)
- **Attention computation**: 2-4x speedup (memory bandwidth limited)
- **Overall inference pipeline**: 15-30% improvement

### Resource Utilization
- **Current**: 2 cores per socket (12.5% utilization on 16-core systems)
- **Target**: 16 cores per socket (100% utilization)
- **Memory bandwidth**: Better NUMA locality and parallel memory access

## Implementation Phases

### Phase 1: Foundation (Week 1)
1. ✅ Add OpenMP includes and pragmas to element-wise operations
2. ✅ Implement parallel residual connections
3. ✅ Implement parallel SwiGLU activation
4. ✅ Implement parallel bias addition
5. ✅ Add performance timing instrumentation

### Phase 2: Core Computation (Week 2)
1. ❌ Parallelize attention score computation
2. ❌ Parallelize softmax computation  
3. ❌ Parallelize attention application to values
4. ❌ Add thread-safe attention caching

### Phase 3: Advanced Optimization (Week 3)
1. ❌ Parallelize RoPE computation
2. ❌ Optimize data distribution loops
3. ❌ Implement NUMA-aware thread placement
4. ❌ Add adaptive thread count based on workload

### Phase 4: Validation and Tuning (Week 4)
1. ❌ Performance benchmarking and comparison
2. ❌ Thread scaling analysis
3. ❌ Memory bandwidth optimization
4. ❌ Production readiness validation

## Code Quality Guidelines

### OpenMP Best Practices
1. **Use `simd` directives** for vectorizable loops
2. **Use `collapse` clauses** for nested loop optimization
3. **Avoid false sharing** with proper data alignment
4. **Use `firstprivate` and `private` clauses** appropriately
5. **Add `if` clauses** for small loop thresholds

### Error Handling
```cpp
// Check OpenMP availability at runtime
if (omp_get_max_threads() < 2) {
    LOG_WARN("OpenMP not available or single-threaded - using sequential execution");
}

// Graceful degradation for memory-constrained environments
int num_threads = std::min(omp_get_max_threads(), 
                          static_cast<int>(available_memory_gb * 2));
omp_set_num_threads(num_threads);
```

### Testing Strategy
1. **Correctness verification**: Compare single-threaded vs multi-threaded outputs
2. **Performance validation**: Measure speedups across different thread counts
3. **Memory usage monitoring**: Track memory consumption with OpenMP
4. **NUMA affinity testing**: Validate thread placement on multi-socket systems

## Success Metrics

### Performance Targets
- **Element-wise operations**: Minimum 4x speedup on 8-core systems
- **Overall pipeline**: Minimum 15% improvement in end-to-end inference
- **CPU utilization**: Achieve >90% core utilization during computation phases
- **Memory efficiency**: No significant increase in memory consumption

### Quality Targets
- **Numerical accuracy**: Maintain bit-exact results vs single-threaded execution
- **Stability**: No race conditions or deadlocks under load testing
- **Scalability**: Linear performance scaling up to socket core count
- **Maintainability**: Clean, readable OpenMP code with proper documentation

## Risks and Mitigation

### Technical Risks
1. **Memory bandwidth saturation**: Monitor and tune thread count accordingly
2. **NUMA effects**: Implement proper thread affinity and data placement
3. **Load balancing**: Ensure work distribution across threads is optimal
4. **Race conditions**: Thorough testing and code review for thread safety

### Mitigation Strategies
1. **Incremental rollout**: Implement phase by phase with validation
2. **Fallback mechanisms**: Maintain single-threaded code paths
3. **Performance monitoring**: Continuous benchmarking throughout development
4. **Code reviews**: Mandatory peer review for all OpenMP code

## Conclusion

The hybrid MPI+OpenMP parallelization strategy will significantly improve Llaminar's performance by utilizing all available cores within each socket. The phased implementation approach ensures manageable development with measurable progress and reduced risk of introducing regressions.

**Key Benefits:**
- 15-30% overall performance improvement
- Better resource utilization (2 cores → 16 cores per socket)
- Maintained MPI distributed architecture
- Minimal architectural changes required

**Implementation Priority:**
1. **High Impact**: Element-wise operations (Phase 1)
2. **Medium Impact**: Attention computation (Phase 2)  
3. **Low Impact**: RoPE and data movement (Phase 3)

This plan provides a clear roadmap for implementing hybrid parallelization while maintaining code quality and system stability.