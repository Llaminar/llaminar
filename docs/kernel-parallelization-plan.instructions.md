# 🚀 **MPI + COSMA Kernel Parallelization Plan**

## 📋 **Phase 1: Basic MPI Support (Todos 1-6)**

### **Phase 1A: Foundation (Todo 1)**
**Objective**: Establish MPI infrastructure for all kernels

**Deliverables**:
```cpp
// 1. MPIKernelBase class with:
//    - MPI communicator management
//    - Rank/size tracking  
//    - Common distribution utilities
//    - Error handling patterns

// 2. Distribution strategy enums:
//    - ROW_WISE, COL_WISE, BLOCK_WISE
//    - HEAD_WISE, VOCAB_WISE, SEQUENCE_WISE

// 3. Communication pattern templates:
//    - AllGather, AllReduce, ReduceScatter
//    - Async communication helpers
```

**Acceptance Criteria**:
- ✅ MPI initialization/finalization works correctly
- ✅ All kernels can inherit from MPIKernelBase
- ✅ Basic distribution utilities function properly

---

### **Phase 1B: Matrix Operations (Todos 2-3)**
**Objective**: Parallelize core matrix and normalization operations

**LinearKernel (Todo 2)**: (Historical) The standalone non-MPI LinearKernel has been removed; MPILinearKernel plus adaptiveMatMul now cover all projection paths.
```cpp
// Row-wise distribution pattern:
// Input:  [seq_len, d_model] → distribute by seq_len
// Weight: [d_model, d_ff]    → replicated on all ranks  
// Output: [seq_len, d_ff]    → gather by seq_len

// Communication: MPI_Allgather for final assembly
```

**RMSNormKernel (Todo 3)**:
```cpp
// Sequence-wise distribution + reduction pattern:
// Each rank: subset of sequences
// Global operation: MPI_Allreduce for RMS statistics
// Local operation: normalization per sequence
```

**Acceptance Criteria**:
- ✅ Linear operations distribute correctly across ranks
- ✅ RMSNorm produces identical results to serial version
- ✅ Communication overhead is reasonable

---

### **Phase 1C: Lookup & Attention Operations (Todos 4-6)**
**Objective**: Parallelize remaining transformer components

**EmbeddingKernel (Todo 4)**:
```cpp
// Vocabulary sharding pattern:
// Embedding table: [vocab_size, embed_dim] → shard by vocab_size
// Token lookups: MPI_Allgather for distributed access
```

**AttentionKernel (Todo 5)**:
```cpp
// Head-wise parallelism pattern:
// Q, K, V projections: each rank handles subset of heads
// Attention computation: parallel across heads
// Global coordination: MPI_Allreduce for softmax normalization
```

**MLPKernel (Todo 6)**:
```cpp
// Pipeline pattern:
// Gate/Up: column-wise distribution
// SwiGLU: local element-wise operation
// Down: row-wise distribution with MPI_Allreduce
```

**Acceptance Criteria**:
- ✅ All kernels produce correct results with MPI
- ✅ Strong scaling demonstrates parallelization benefits
- ✅ Memory usage scales appropriately with rank count

---

## 📋 **Phase 2: Advanced Distribution (Todos 7-10)**

### **Phase 2A: COSMA Interface (Todo 7)**
**Objective**: Create clean interface for COSMA integration

**Deliverables**:
```cpp
// 1. CosmaMatrixAdapter class:
//    - Convert Tensor → CosmaMatrix
//    - Handle data layout differences
//    - Manage strategy generation

// 2. CosmaKernelMixin template:
//    - Drop-in replacement for matrix operations  
//    - Automatic strategy optimization
//    - Error handling and fallbacks

// 3. Integration utilities:
//    - Memory pool management
//    - Communication buffer optimization
//    - Performance profiling hooks
```

**Acceptance Criteria**:
- ✅ Seamless conversion between Tensor and CosmaMatrix
- ✅ Strategy generation works for common matrix sizes
- ✅ No performance regression from interface overhead

---

### **Phase 2B: Matrix Operation Replacement (Todos 8-10)**
**Objective**: Replace all matrix multiplications with COSMA

**LinearKernel (Todo 8)**: (Historical) Skipped — legacy kernel retired after MPI unification.
```cpp
// Replace: computeLinear() triple loop
// With: cosma::multiply() call
// Strategy: optimal for [seq_len, d_model] × [d_model, d_ff]
```

**AttentionKernel (Todo 9)**:
```cpp
// Replace: Q, K, V, output projections  
// With: 4 separate cosma::multiply() calls
// Strategy: optimized for multi-head patterns
```

**MLPKernel (Todo 10)**:
```cpp
// Replace: gate, up, down projections
// With: 3 separate cosma::multiply() calls
// Strategy: optimized for MLP dimension patterns
```

**Acceptance Criteria**:
- ✅ All matrix operations use COSMA
- ✅ Performance improves over naive implementations
- ✅ Numerical accuracy maintained

---

## 📋 **Phase 3: COSMA Integration (Todos 11-15)**

### **Phase 3A: GPU Acceleration (Todo 11)**
**Objective**: Enable COSMA's GPU backends

**Deliverables**:
```cpp
// 1. GPU backend detection:
//    - CUDA availability check
//    - ROCm/HIP availability check
//    - Fallback to CPU if needed

// 2. Memory management:
//    - GPU tensor allocation
//    - Host-device transfers
//    - Memory pool optimization

// 3. Kernel configuration:
//    - GPU-aware MPI setup
//    - NCCL/RCCL integration
//    - Hybrid CPU-GPU strategies
```

**Acceptance Criteria**:
- ✅ Automatic GPU backend detection works
- ✅ Memory transfers are optimized
- ✅ GPU performance exceeds CPU performance

---

### **Phase 3B: Compute Grid Optimization (Todo 12)**
**Objective**: Use COSMA's compute grid for optimal scheduling

**Deliverables**:
```cpp
// 1. Grid layout optimization:
//    - Automatic process grid generation
//    - Load balancing across heterogeneous hardware
//    - NUMA topology awareness

// 2. Kernel scheduling:
//    - Dependency-aware kernel ordering
//    - Resource utilization optimization
//    - Communication overlap opportunities

// 3. Performance monitoring:
//    - Real-time performance metrics
//    - Bottleneck identification
//    - Adaptive strategy adjustment
```

**Acceptance Criteria**:
- ✅ Process grid is optimally configured
- ✅ Kernel scheduling minimizes idle time
- ✅ Resource utilization is maximized

---

### **Phase 3C: Testing & Validation (Todos 13-15)**
**Objective**: Comprehensive testing and benchmarking

**MPI Testing (Todo 13)**:
```cpp
// 1. Multi-rank test execution
// 2. Correctness validation across ranks  
// 3. Edge case testing (rank=1, uneven distributions)
```

**COSMA Testing (Todo 14)**:
```cpp
// 1. Matrix multiplication correctness
// 2. GPU backend validation
// 3. Strategy optimization testing
```

**Performance Benchmarking (Todo 15)**:
```cpp
// 1. Scaling studies (1-16+ ranks)
// 2. Performance comparison matrices
// 3. Memory usage profiling
```

**Acceptance Criteria**:
- ✅ All tests pass with 100% success rate
- ✅ Performance benchmarks show expected scaling
- ✅ Memory usage stays within acceptable bounds

---

## 🎯 **Success Metrics**

### **Phase 1 Success Criteria**:
- [ ] 2-4x speedup with 4 MPI ranks vs serial
- [ ] Linear scaling up to 8 ranks
- [ ] <10% memory overhead per rank

### **Phase 2 Success Criteria**:
- [ ] 4-8x speedup with COSMA vs naive matrix multiplication
- [ ] Numerical accuracy maintained (relative error <1e-6)
- [ ] <5% performance overhead from interface layer

### **Phase 3 Success Criteria**:
- [ ] 8-16x speedup with GPU acceleration
- [ ] 90%+ GPU utilization during matrix operations
- [ ] Optimal scaling across heterogeneous hardware

## 📅 **Timeline Estimate**

- **Phase 1**: 2-3 weeks (Foundation + basic MPI)
- **Phase 2**: 2-3 weeks (COSMA interface + integration)
- **Phase 3**: 2-3 weeks (GPU backends + optimization)

**Total**: 6-9 weeks for complete implementation

---

## 🛠 **Implementation Notes**

### **Key Dependencies**:
- OpenMPI or MPICH installation
- COSMA library with MPI support
- Optional: CUDA/ROCm for GPU acceleration

### **Build System Integration**:
- CMake FindMPI module
- COSMA library linking
- MPI compiler wrapper usage

### **Testing Strategy**:
- Unit tests with mpirun execution
- Correctness validation across rank counts
- Performance regression detection

This plan provides a clear path from basic MPI parallelization to full COSMA integration with GPU acceleration, ensuring kernels can scale like COSMA's matrix multiplication operations!