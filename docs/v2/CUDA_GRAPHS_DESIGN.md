# CUDA Graphs Integration Design

## Overview

This document describes how to integrate CUDA Graphs with Llaminar's ComputeStage abstraction to eliminate kernel launch overhead during decode (single-token inference).

## Problem Statement

**Current State**: Each decode iteration launches 100+ individual CUDA kernels:
- ~24 layers × ~4-6 kernels/layer = ~100-150 kernel launches
- Kernel launch overhead: ~5-10μs per launch
- Total overhead: ~0.5-1.5ms per token (significant for 5 tok/s = 200ms/token)

**Solution**: CUDA Graphs capture the entire kernel sequence and replay it with a **single** API call.

## Architecture Analysis

### Current Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     ModelExecutor                                │
│  - Manages prefill/decode phases                                │
│  - Calls DeviceGraphExecutor.execute() per iteration                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DeviceGraphExecutor                                │
│  - Iterates over ComputeGraph nodes in topological order        │
│  - Calls executeNode() for each stage                           │
│  - Handles coherence (ensureOnDevice, markOutputsDirty)         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼ (per stage)
┌─────────────────────────────────────────────────────────────────┐
│                   IComputeStage.execute()                        │
│  - Stage-specific logic                                         │
│  - Calls kernel via KernelFactory                               │
│  - Kernel launches CUDA kernels on cudaStream                   │
└─────────────────────────────────────────────────────────────────┘
```

### Proposed CUDA Graph Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     ModelExecutor                                │
│  - DECODE phase: Use CUDA Graph execution path                  │
│  - PREFILL phase: Use standard execution (variable batch size)  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │ Decode (batch=1)                      │ Prefill (batch=N)
        ▼                                       ▼
┌───────────────────────┐           ┌───────────────────────┐
│  CUDAGraphExecutor    │           │  DeviceGraphExecutor        │
│  (new component)      │           │  (existing)           │
└───────────┬───────────┘           └───────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│              CUDA Graph Lifecycle                                │
│                                                                  │
│  1. CAPTURE PHASE (first decode iteration):                     │
│     cudaStreamBeginCapture(stream)                              │
│     → Execute all stages normally (kernels get captured)        │
│     cudaStreamEndCapture(stream, &graph)                        │
│     cudaGraphInstantiate(&graphExec, graph)                     │
│                                                                  │
│  2. REPLAY PHASE (subsequent iterations):                       │
│     cudaGraphLaunch(graphExec, stream)  // ONE API call!        │
│     cudaStreamSynchronize(stream)                               │
│                                                                  │
│  3. UPDATE PHASE (if inputs change shape):                      │
│     cudaGraphExecUpdate(graphExec, newGraph)                    │
│     OR: Discard and re-capture                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Stream Management

**Current**: Each kernel uses `cudaStream_t stream = 0` (default stream) or creates its own.

**Required**: All kernels must use the **same stream** for graph capture to work.

**Solution**: Add stream propagation through the execution stack:

```cpp
// In IBackend.h - add stream accessor
class IBackend {
public:
    // Existing methods...
    
    // NEW: Stream management for CUDA Graphs
    virtual void* getStream(int device_id) const = 0;
    virtual void setStream(void* stream, int device_id) = 0;
};

// In CUDABackend.cu
class CUDABackendImpl {
    cudaStream_t streams_[MAX_DEVICES];  // One stream per device
public:
    void* getStream(int device_id) const override {
        return static_cast<void*>(streams_[device_id]);
    }
};
```

### 2. DeviceContext Stream Propagation

**Add stream to IDeviceContext**:

```cpp
// In DeviceContext.h
class IDeviceContext {
public:
    virtual void* getStream() const = 0;  // NEW
    // ... existing methods
};
```

### 3. ComputeStage Stream Usage

**Stages must use the context's stream**:

```cpp
// In GEMMStage.cpp
bool GEMMStage::execute(IDeviceContext* ctx) {
    cudaStream_t stream = static_cast<cudaStream_t>(ctx->getStream());
    
    // Pass stream to kernel
    kernel->multiply_tensor(A, B, C, m, n, k, stream);
    return true;
}
```

### 4. Kernel Stream Parameters

**Update kernel interfaces**:

```cpp
// In ITensorGemm.h
class ITensorGemm {
public:
    virtual bool multiply_tensor(
        const ITensor* A, 
        ITensor* C, 
        int m, int n, int k,
        void* stream = nullptr  // NEW: optional stream override
    ) = 0;
};
```

## CUDAGraphExecutor Design

### Class Definition

```cpp
/**
 * @file CUDADeviceGraphExecutor.h
 * @brief CUDA Graph-based executor for decode phase
 */

#pragma once

#include "IDeviceGraphExecutor.h"
#include <memory>

namespace llaminar2 {

class CUDAGraphExecutor : public IGraphExecutor {
public:
    struct Config {
        int device_id = 0;
        bool enable_graph_update = true;  // Try update vs re-capture
        size_t max_graph_cache_size = 10; // Cache multiple graph shapes
    };
    
    explicit CUDAGraphExecutor(const Config& config);
    ~CUDAGraphExecutor() override;
    
    // IGraphExecutor interface
    bool execute(ComputeGraph& graph, IDeviceContext* ctx) override;
    
    // Graph lifecycle management
    bool isGraphCaptured() const { return graph_captured_; }
    bool needsRecapture(const ComputeGraph& graph) const;
    void invalidateGraph();
    
    // Statistics
    size_t captureCount() const { return capture_count_; }
    size_t replayCount() const { return replay_count_; }
    
private:
    enum class GraphState {
        UNCAPTURED,      // No graph yet
        CAPTURING,       // Currently capturing
        CAPTURED,        // Ready for replay
        INVALID          // Needs recapture (shape change)
    };
    
    // Capture a new graph
    bool captureGraph(ComputeGraph& graph, IDeviceContext* ctx);
    
    // Replay existing graph
    bool replayGraph(IDeviceContext* ctx);
    
    // Check if graph is compatible with current execution
    bool isGraphCompatible(const ComputeGraph& graph) const;
    
    // Implementation details hidden in .cu file
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    Config config_;
    GraphState state_ = GraphState::UNCAPTURED;
    bool graph_captured_ = false;
    size_t capture_count_ = 0;
    size_t replay_count_ = 0;
    
    // Cached graph signature for compatibility checking
    size_t cached_graph_hash_ = 0;
    std::vector<size_t> cached_input_shapes_;
};

} // namespace llaminar2
```

### Implementation Sketch

```cpp
// CUDAGraphExecutor.cu

struct CUDAGraphExecutor::Impl {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaStream_t stream = nullptr;
    
    ~Impl() {
        if (graphExec) cudaGraphExecDestroy(graphExec);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
    }
};

bool CUDAGraphExecutor::execute(ComputeGraph& graph, IDeviceContext* ctx) {
    // Check if we need to (re)capture
    if (state_ == GraphState::UNCAPTURED || needsRecapture(graph)) {
        if (!captureGraph(graph, ctx)) {
            LOG_WARN("[CUDAGraphExecutor] Capture failed, falling back to direct execution");
            return DeviceGraphExecutor::executeSequential(graph, ctx);
        }
    }
    
    // Replay the captured graph
    return replayGraph(ctx);
}

bool CUDAGraphExecutor::captureGraph(ComputeGraph& graph, IDeviceContext* ctx) {
    state_ = GraphState::CAPTURING;
    
    // Destroy old graph if exists
    if (impl_->graphExec) {
        cudaGraphExecDestroy(impl_->graphExec);
        impl_->graphExec = nullptr;
    }
    if (impl_->graph) {
        cudaGraphDestroy(impl_->graph);
        impl_->graph = nullptr;
    }
    
    // Begin capture
    cudaError_t err = cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        LOG_ERROR("[CUDAGraphExecutor] cudaStreamBeginCapture failed: " << cudaGetErrorString(err));
        state_ = GraphState::INVALID;
        return false;
    }
    
    // Execute graph normally - kernels get captured, not executed
    graph.reset();
    auto order = graph.getExecutionOrder();
    
    for (const auto& name : order) {
        auto* node = graph.getNode(name);
        if (!node || !node->stage) continue;
        
        // Execute stage (kernels captured into graph)
        if (!executeNodeForCapture(*node, ctx)) {
            cudaStreamEndCapture(impl_->stream, &impl_->graph);  // Must end capture
            state_ = GraphState::INVALID;
            return false;
        }
        graph.markCompleted(name);
    }
    
    // End capture
    err = cudaStreamEndCapture(impl_->stream, &impl_->graph);
    if (err != cudaSuccess) {
        LOG_ERROR("[CUDAGraphExecutor] cudaStreamEndCapture failed: " << cudaGetErrorString(err));
        state_ = GraphState::INVALID;
        return false;
    }
    
    // Instantiate executable graph
    err = cudaGraphInstantiate(&impl_->graphExec, impl_->graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        LOG_ERROR("[CUDAGraphExecutor] cudaGraphInstantiate failed: " << cudaGetErrorString(err));
        state_ = GraphState::INVALID;
        return false;
    }
    
    state_ = GraphState::CAPTURED;
    graph_captured_ = true;
    capture_count_++;
    
    // Cache graph signature for compatibility checking
    cacheGraphSignature(graph);
    
    LOG_INFO("[CUDAGraphExecutor] Graph captured with " << order.size() << " stages");
    return true;
}

bool CUDAGraphExecutor::replayGraph(IDeviceContext* ctx) {
    if (!impl_->graphExec) {
        LOG_ERROR("[CUDAGraphExecutor] No graph to replay");
        return false;
    }
    
    // Single API call replays entire graph!
    cudaError_t err = cudaGraphLaunch(impl_->graphExec, impl_->stream);
    if (err != cudaSuccess) {
        LOG_ERROR("[CUDAGraphExecutor] cudaGraphLaunch failed: " << cudaGetErrorString(err));
        return false;
    }
    
    // Synchronize (required for correctness)
    err = cudaStreamSynchronize(impl_->stream);
    if (err != cudaSuccess) {
        LOG_ERROR("[CUDAGraphExecutor] cudaStreamSynchronize failed: " << cudaGetErrorString(err));
        return false;
    }
    
    replay_count_++;
    return true;
}
```

## Coherence Considerations

### Problem: Coherence Operations Are Host-Side

Current coherence management in DeviceGraphExecutor:
```cpp
// This is HOST code - runs on CPU
if (!cohereInputs(inputs, target_device)) { ... }  // Host decision
markOutputsDirty(outputs);                          // Host bookkeeping
```

**These operations CANNOT be captured** in a CUDA graph because they're CPU operations.

### Solution: Pre-Coherence for Decode

For decode phase, tensors are **already on GPU** after prefill. The coherence state is stable:

```cpp
bool CUDAGraphExecutor::execute(ComputeGraph& graph, IDeviceContext* ctx) {
    // BEFORE capture/replay: Ensure all decode tensors are coherent
    if (state_ == GraphState::UNCAPTURED) {
        ensureDecodeCoherence(graph);  // One-time setup
    }
    
    // Capture or replay (GPU-only operations)
    // ...
}

void CUDAGraphExecutor::ensureDecodeCoherence(ComputeGraph& graph) {
    // For decode, inputs are already on GPU from prefill
    // Just verify all tensors have valid GPU buffers
    for (const auto& node : graph.nodes()) {
        auto dump_info = node.stage->getDumpInfo();
        
        // Verify inputs have GPU buffers
        for (const auto& input : dump_info.inputs) {
            if (input.tensor) {
                LLAMINAR_ASSERT(input.tensor->hasDeviceBuffer(),
                    "Decode tensor must be on GPU: " << input.name);
            }
        }
    }
}
```

## When CUDA Graphs Work (and Don't)

### ✅ Graph-Compatible Operations

- **Fixed-size GEMMs**: Decode always has `batch_size=1`
- **Attention with fixed sequence length**: After each token, seq_len increments predictably
- **Element-wise operations**: SwiGLU, RMSNorm, residual add
- **Memory-bound kernels**: Most decode operations

### ❌ Graph-Incompatible Operations

| Operation | Why Incompatible | Solution |
|-----------|------------------|----------|
| **Variable batch size** | Grid dimensions change | Use only for decode (batch=1) |
| **Host-side branching** | CPU conditionals can't be captured | Remove host-side logic from kernel paths |
| **cudaMalloc during execution** | Memory allocation is host-side | Pre-allocate all buffers |
| **MPI collectives** | Cross-GPU communication | Execute outside graph, or use CUDA-aware MPI with P2P |
| **Dynamic kernel selection** | Host-side dispatch | Pre-select kernel during capture |

### KV Cache Growth

The KV cache grows by 1 position each decode step. This changes memory access patterns.

**Solution**: Use **ring buffer KV cache** (already implemented in `CUDARingKVCache`):
- Fixed memory layout
- Position index wraps around
- No reallocation needed

## Integration with ModelExecutor

```cpp
// In ModelExecutor.cpp

class ModelExecutor {
    std::unique_ptr<DeviceGraphExecutor> standard_executor_;      // For prefill
    std::unique_ptr<CUDAGraphExecutor> decode_executor_;    // For decode
    
public:
    bool decode(/* ... */) {
        if (use_cuda_graphs_ && batch_size == 1) {
            return decode_executor_->execute(decode_graph_, device_ctx_);
        } else {
            return standard_executor_->execute(decode_graph_, device_ctx_);
        }
    }
    
    bool prefill(/* ... */) {
        // Always use standard executor (variable batch size)
        return standard_executor_->execute(prefill_graph_, device_ctx_);
    }
};
```

## Implementation Phases

### Phase 1: Stream Unification (Foundation)
- [ ] Add `getStream()` to IBackend
- [ ] Propagate stream through DeviceContext
- [ ] Update all CUDA kernels to accept stream parameter
- [ ] Verify all kernels use same stream

### Phase 2: Basic Graph Capture
- [ ] Implement CUDAGraphExecutor class
- [ ] Add capture/replay logic
- [ ] Handle graph instantiation errors
- [ ] Add fallback to standard execution

### Phase 3: Coherence Integration
- [ ] Pre-coherence verification for decode
- [ ] Remove host-side coherence from capture path
- [ ] Verify tensor states before replay

### Phase 4: Production Hardening
- [ ] Graph caching for multiple shapes
- [ ] Automatic invalidation on shape change
- [ ] Performance metrics and logging
- [ ] Environment variable to enable/disable

### Phase 5: Optimization
- [ ] Graph update vs re-capture heuristics
- [ ] Persistent kernel graphs (CUDA 11.4+)
- [ ] Multi-stream graphs for parallelism

## Expected Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Kernel launches/token | ~120 | 1 | 120× reduction |
| Launch overhead/token | ~1.2ms | ~0.01ms | ~100× reduction |
| Decode throughput | 5.3 tok/s | ~5.8 tok/s | ~10% (optimistic) |

**Note**: Actual improvement depends on how much of total time is launch overhead vs compute time. For small models like 0.5B, launch overhead is proportionally larger.

## References

- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [llama.cpp CUDA Graph Implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu#L3703-L3778)
- [NVIDIA Blog: Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
