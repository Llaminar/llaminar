# Transformer Kernel Refactoring Plan

> Deprecation Context (2025-10): This historical plan predates consolidation under `DistributedTransformerPipeline` (formerly `MPITransformerPipeline`). Modern work should target `DistributedTransformerPipeline` directly; adapter layers referenced here were removed.

## Overview

This document outlines the architectural refactoring of the Llaminar transformer pipeline to achieve proper separation of concerns through modular kernel composition. The current implementation violates architectural principles by embedding kernel logic directly in the pipeline, preventing reusability and proper MPI distribution.

## Current Architecture Problems

### Identified Issues
1. **Embedded Kernel Logic**: (Historical) SwiGLU, RoPE, and other operations were hardcoded in `MPITransformerPipeline` (now `DistributedTransformerPipeline`) instead of being composable kernels
2. **Architectural Violations**: Pipeline implements operations rather than orchestrating them
3. **Code Duplication**: Activation functions and math operations scattered throughout codebase
4. **Limited Reusability**: Cannot reuse operations outside transformer context
5. **MPI Distribution Gaps**: Some operations not properly distributed across ranks
6. **Threading Inconsistency**: OpenMP parallelization applied inconsistently

### Current Hardcoded Implementations
- **SwiGLU Activation** (lines 314-321): Embedded in `executeTransformerLayer`
- **RoPE (Rotary Position Embedding)**: Missing proper implementation
- **Residual Connections**: Manual tensor addition loops
- **Tensor Broadcasting**: Ad-hoc MPI communication patterns

## Target Architecture

### Design Principles
1. **Separation of Concerns**: Pipelines orchestrate, kernels compute
2. **Composability**: All operations as reusable MPI-aware kernels
3. **Modularity**: Clean interfaces enable easy testing and replacement
4. **Performance**: OpenMP + MPI hybrid parallelization throughout
5. **Maintainability**: Single implementation per operation type

### Core Components

#### 1. PipelineBase Class
```cpp
class PipelineBase : public MPIKernelBase {
protected:
    // Kernel registry for composition
    std::unordered_map<std::string, std::unique_ptr<MPIKernelBase>> kernels_;
    
    // MPI-aware tensor creation utilities
    virtual std::shared_ptr<TensorBase> createBroadcastTensor(const std::vector<size_t>& shape);
    virtual std::shared_ptr<TensorBase> createLocalTensor(const std::vector<size_t>& shape);
    virtual std::shared_ptr<TensorBase> createDistributedTensor(const std::vector<size_t>& shape);
    
    // Kernel composition utilities
    bool registerKernel(const std::string& name, std::unique_ptr<MPIKernelBase> kernel);
    MPIKernelBase* getKernel(const std::string& name);
    
    // Synchronization utilities
    void syncAllRanks(const std::string& operation_name);
    bool broadcastTensor(std::shared_ptr<TensorBase>& tensor, int root_rank);
    
public:
    PipelineBase();
    virtual ~PipelineBase() = default;
    
    // Pure virtual for pipeline-specific logic
    virtual bool execute(const std::vector<std::shared_ptr<TensorBase>>& inputs,
                        std::vector<std::shared_ptr<TensorBase>>& outputs) = 0;
};
```

#### 2. Missing MPI Kernels

##### MPISwiGLUKernel
```cpp
class MPISwiGLUKernel : public MPIKernelBase {
private:
    // OpenMP threading configuration
    int num_threads_;
    
public:
    MPISwiGLUKernel();
    
    // SwiGLU: gate * silu(up) with MPI distribution and OpenMP parallelization
    bool execute(const std::vector<std::shared_ptr<TensorBase>>& inputs,
                std::vector<std::shared_ptr<TensorBase>>& outputs) override;
                
    // inputs: [gate_projection, up_projection]
    // outputs: [swiglu_result]
};
```

##### MPIRoPEKernel  
```cpp
class MPIRoPEKernel : public MPIKernelBase {
private:
    int max_seq_len_;
    int head_dim_;
    float theta_;
    
    // Precomputed frequency tables
    std::vector<float> cos_table_;
    std::vector<float> sin_table_;
    
public:
    MPIRoPEKernel(int max_seq_len, int head_dim, float theta = 10000.0f);
    
    // Rotary Position Embedding with MPI distribution
    bool execute(const std::vector<std::shared_ptr<TensorBase>>& inputs,
                std::vector<std::shared_ptr<TensorBase>>& outputs) override;
                
    // inputs: [q_or_k, position_ids]
    // outputs: [rotated_q_or_k]
};
```

##### MPIResidualKernel
```cpp
class MPIResidualKernel : public MPIKernelBase {
public:
    MPIResidualKernel();
    
    // Residual connection: input + residual with OpenMP SIMD
    bool execute(const std::vector<std::shared_ptr<TensorBase>>& inputs,
                std::vector<std::shared_ptr<TensorBase>>& outputs) override;
                
    // inputs: [input, residual]
    // outputs: [input + residual]
};
```

#### 3. (Historical) Refactored MPITransformerPipeline (Deprecated)
```cpp
// DEPRECATED: Example retained only for historical context.
class MPITransformerPipeline : public PipelineBase {
private:
    LayerConfig config_;
    
    // Kernel instances (registered in constructor)
    // - "rmsnorm": MPIRMSNormKernel
    // - "attention": MPIAttentionKernel  
    // - "linear": MPILinearKernel
    // - "swiglu": MPISwiGLUKernel
    // - "rope": MPIRoPEKernel
    // - "residual": MPIResidualKernel
    // - "embedding": EmbeddingKernel (rank 0 only)
    
public:
    MPITransformerPipeline(const LayerConfig& config);
    
    // High-level execution using kernel composition
    bool execute(const std::vector<int>& token_ids,
                const ModelWeights& weights,
                std::shared_ptr<TensorBase>& output);
                
protected:
    // Decomposed layer execution using kernel composition
    bool executeTransformerLayer(int layer_idx,
                                std::shared_ptr<TensorBase>& input,
                                const ModelWeights& weights,
                                std::shared_ptr<TensorBase>& output);
};
```

## Implementation Strategy

### Phase 1: Extract and Create Missing Kernels

1. **Extract SwiGLU Implementation**
   - Move lines 314-321 from `executeTransformerLayer` to `MPISwiGLUKernel`
   - Add OpenMP SIMD parallelization
   - Add MPI rank-aware work distribution
   - Implement proper input/output validation

2. **Create RoPE Kernel**
   - Implement rotary position embedding mathematics
   - Precompute frequency tables for efficiency
   - Add position-aware tensor rotation
   - Distribute computation across MPI ranks

3. **Create Residual Kernel**
   - Extract manual tensor addition loops
   - Implement OpenMP SIMD parallelization
   - Add shape validation and broadcasting support

### Phase 2: Implement PipelineBase

1. **Core Infrastructure**
   - Kernel registry with string-based lookup
   - MPI-aware tensor factory methods
   - Synchronization and communication utilities

2. **Composition Utilities**
   - Kernel registration and retrieval
   - Input/output tensor routing between kernels
   - Error handling and validation chains

### Phase 3: Refactor MPITransformerPipeline

1. **Constructor Changes**
   - Register all required kernels in constructor
   - Remove direct kernel instantiation
   - Move to composition-based approach

2. **Method Refactoring**
   - `executeTransformerLayer`: Use kernel composition instead of hardcoded operations
   - Remove embedded SwiGLU, residual, and other operation code
   - Replace with kernel execution calls

3. **Pipeline Orchestration**
   - Chain kernel executions with proper tensor flow
   - Handle MPI synchronization between kernel stages
   - Maintain performance optimizations

### Phase 4: OpenMP + MPI Hybrid Parallelization

#### Threading Strategy
```cpp
// Per-kernel OpenMP configuration
void configureOpenMPForKernel(const std::string& kernel_name, size_t tensor_size) {
    // Small operations: single-threaded to avoid overhead
    if (tensor_size < 8192) {
        omp_set_num_threads(1);
        return;
    }
    
    // Large operations: use physical cores per NUMA node
    int cores_per_node = detectCoresPerSocket();
    omp_set_num_threads(cores_per_node);
    
    // Configure thread affinity for NUMA locality
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
}
```

#### MPI Work Distribution
```cpp
// Distribute tensor operations across MPI ranks
void distributeTensorWork(const TensorBase* tensor, int& start_row, int& end_row) {
    int total_rows = tensor->shape()[0];
    int rank = getRank();
    int size = getSize();
    
    int rows_per_rank = total_rows / size;
    int remainder = total_rows % size;
    
    start_row = rank * rows_per_rank;
    end_row = start_row + rows_per_rank;
    
    // Distribute remainder rows to first ranks
    if (rank < remainder) {
        start_row += rank;
        end_row += rank + 1;
    } else {
        start_row += remainder;
        end_row += remainder;
    }
}
```

#### SIMD Parallelization Patterns
```cpp
// Standard SIMD pattern for element-wise operations
#pragma omp parallel for simd aligned(input_data, output_data:32) schedule(static)
for (size_t i = start_idx; i < end_idx; ++i) {
    output_data[i] = compute_operation(input_data[i]);
}

// Matrix operation pattern with cache optimization
#pragma omp parallel for collapse(2) schedule(static)
for (int i = start_row; i < end_row; ++i) {
    for (int j = 0; j < cols; j += SIMD_WIDTH) {
        #pragma omp simd aligned(A, B, C:32)
        for (int k = j; k < std::min(j + SIMD_WIDTH, cols); ++k) {
            C[i * cols + k] = compute_matmul_element(A, B, i, k);
        }
    }
}
```

### Phase 5: Cleanup Non-MPI Components

> [STATUS UPDATE - COMPLETED] All non-MPI kernels and pipelines referenced in this phase (LinearKernel, AttentionKernel, RMSNormKernel, MatMulKernel, legacy non-distributed pipelines) have been removed. Remaining references here are historical planning notes. New development must target MPIKernelBase-derived implementations only; add any new linear/attention variants as MPI-aware from inception.

#### Files to Remove
1. **Non-MPI Kernels**: All kernels not inheriting from MPIKernelBase
2. **Non-MPI Pipelines**: Any pipeline not designed for MPI distribution
3. **Legacy Components**: Superseded by new modular architecture

#### Files to Keep
1. **Core Infrastructure**: `kernel_base.h`, `mpi_kernel_base.h`
2. **MPI Kernels**: All `MPI*Kernel` implementations
3. **Utilities**: Tensor factory, performance timer, debug utilities
4. **Tests**: Update to test new modular architecture

## Performance Considerations

### Memory Management
- **NUMA-Aware Allocation**: Allocate tensors on local NUMA nodes
- **Cache Optimization**: Align data structures to cache line boundaries
- **Memory Reuse**: Pool intermediate tensors to reduce allocation overhead

### Communication Optimization
- **Asynchronous MPI**: Use non-blocking communication where possible
- **Communication Hiding**: Overlap computation with communication
- **Bandwidth Optimization**: Minimize data movement between ranks

### Threading Efficiency
- **Work Stealing**: Balance load across OpenMP threads
- **False Sharing Prevention**: Align thread-local data appropriately  
- **Vectorization**: Ensure compiler can auto-vectorize inner loops

## Validation Strategy

### Correctness Testing
1. **Kernel Unit Tests**: Individual kernel validation with known inputs/outputs
2. **Pipeline Integration Tests**: End-to-end transformer execution
3. **Numerical Precision Tests**: Compare against reference implementations
4. **MPI Consistency Tests**: Verify identical results across different rank counts

### Performance Benchmarking
1. **Kernel Microbenchmarks**: Individual operation performance
2. **Pipeline Macrobenchmarks**: Full transformer inference timing
3. **Scaling Tests**: Performance across different sequence lengths and batch sizes
4. **Memory Usage Profiling**: Monitor memory efficiency improvements

## Success Criteria

1. **Architectural Compliance**: Clean separation between pipelines and kernels
2. **Performance Maintenance**: No regression in inference speed
3. **Code Reusability**: Kernels usable in multiple contexts
4. **Memory Efficiency**: Reduced memory footprint through better management
5. **Maintainability**: Cleaner, more testable codebase
6. **MPI Scalability**: Consistent performance across rank configurations

## Implementation Timeline

1. **Week 1**: Extract kernels and implement PipelineBase
2. **Week 2**: Refactor MPITransformerPipeline for composition
3. **Week 3**: Add comprehensive OpenMP parallelization
4. **Week 4**: Performance optimization and validation
5. **Week 5**: Cleanup non-MPI components and final testing

This refactoring will transform Llaminar into a truly modular, high-performance distributed inference engine with clean architectural boundaries and optimal MPI + OpenMP hybrid parallelization.