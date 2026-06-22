# Pipeline Parallelism Research Report for Llaminar V2

**Date**: January 20, 2026  
**Author**: AI Research Assistant  
**Status**: Research Complete / Implementation Ready

---

## Executive Summary

This report analyzes the feasibility and requirements for implementing Pipeline Parallelism in Llaminar V2. The current architecture already has significant infrastructure that can be reused, including layer-level graph building, MPI context management, and execution orchestration. However, **no explicit pipeline parallelism support exists today**—this would be a new capability.

### Key Findings

| Aspect | Status | Complexity |
|--------|--------|------------|
| Layer Range Assignment | Requires new code | Medium |
| Activation Passing (P2P) | **No existing support** | High |
| Weight Loading by Layer Range | Partially supported | Low |
| KV Cache Partitioning | Per-layer already | Low |
| Micro-batch Scheduling | **No existing support** | High |
| ComputeGraph Changes | Moderate additions | Medium |

**Bottom Line**: Pipeline parallelism is achievable but requires **new point-to-point communication infrastructure** and **micro-batch scheduling logic** that don't exist today.

---

## 1. Current Architecture Analysis

### 1.1 Layer Construction ([Qwen2Graph.cpp](../../src/v2/models/qwen/Qwen2Graph.cpp))

**Current Approach**: Layers are built sequentially in a single loop, creating one monolithic graph:

```cpp
// buildFullForwardGraph() - lines 217-286
for (int layer = 0; layer < config_.n_layers; ++layer)
{
    // Get layer weights
    Qwen2LayerWeights layer_weights = weights_.get_layer_weights(layer);

    // Build attention graph for this layer
    ComputeGraph attn_graph = buildAttentionGraph(
        layer_weights, buffers_.layer_buffers, layer, input.seq_len,
        input.batch_size, input.kv_cache, position_ids, device,
        input.sequence_lengths);

    // Merge attention graph, connecting to previous node
    graph.merge(std::move(attn_graph), prev_node);

    // Build FFN graph for this layer
    ComputeGraph ffn_graph = buildFFNGraph(
        layer_weights, buffers_.layer_buffers, layer, input.seq_len,
        input.batch_size, device);

    // Merge FFN graph, connecting to attention residual
    graph.merge(std::move(ffn_graph), attn_last);

    prev_node = ffn_last;
}
```

**For Pipeline Parallel**: Need to modify to only build layers in `[layer_start, layer_end)` range based on rank.

### 1.2 Execution Orchestration ([GraphOrchestrator.cpp](../../src/v2/execution/GraphOrchestrator.cpp))

**Current Flow**:
1. `executeForward()` builds full forward graph
2. `DeviceGraphExecutor::execute()` runs stages in topological order
3. No inter-rank synchronization except `AllreduceStage` and `AllGatherStage` for tensor parallelism

**Key Structures**:
- `InferenceState`: Owns `hidden`, `logits`, `kv_cache`, activation buffers
- `LayerGraphCache`: Caches per-layer graphs for decode mode (reusable for PP)
- `device_contexts_`: Per-device execution contexts

**Limitation**: Current orchestrator assumes all layers execute on the same rank.

### 1.3 Existing Parallelism Concepts

From [MPIStrategy.h](../../../../src/v2/utils/MPIStrategy.h):

```cpp
enum class MPIStrategy {
    None,             // Single rank
    TensorParallel,   // Split heads/features (IMPLEMENTED)
    PipelineParallel, // Split layers (DEFINED BUT NOT IMPLEMENTED)
    SequenceParallel, // Split tokens (NOT IMPLEMENTED)
    Hybrid            // Combined (NOT IMPLEMENTED)
};
```

**Important**: `PipelineParallel` strategy is **defined** but marked as NOT IMPLEMENTED:
> "Communication: Point-to-point activation passing between ranks"
> "Requirements: n_layers % world_size == 0"

---

## 2. Required Changes for Pipeline Parallel

### 2.1 Layer Range Assignment

**What's Needed**: Configuration to assign layer ranges to ranks.

**Proposed Structure**:
```cpp
// New structure for GraphOrchestrator or Qwen2GraphConfig
struct PipelineParallelConfig {
    bool enabled = false;
    int pipeline_stages = 1;           // Number of pipeline stages (usually == world_size)
    
    // Computed per rank
    int layer_start = 0;               // First layer for this rank
    int layer_end = -1;                // Last layer (exclusive), -1 = all layers
    
    // Role flags
    bool is_first_stage = true;        // Has embedding
    bool is_last_stage = true;         // Has LM head
    
    // Micro-batch configuration
    int micro_batches = 1;             // For GPipe-style pipelining
};
```

**Implementation in Qwen2Graph**:
```cpp
ComputeGraph Qwen2Graph::buildForwardGraph_PipelineParallel(
    const Qwen2ForwardInput& input,
    Qwen2ForwardOutput& output,
    const PipelineParallelConfig& pp_config)
{
    ComputeGraph graph;
    std::string prev_node;
    
    // Stage 1: Embedding (only on first pipeline stage)
    if (pp_config.is_first_stage) {
        // ... add embedding stage ...
        prev_node = "embedding";
    } else {
        // Input comes from previous rank via ReceiveActivationsStage
        graph.addNode("recv_activations", 
            ComputeStageFactory::createReceiveActivations(recv_params));
        prev_node = "recv_activations";
    }
    
    // Stage 2: Transformer Layers (only assigned range)
    for (int layer = pp_config.layer_start; layer < pp_config.layer_end; ++layer) {
        // ... build attention + FFN as before ...
    }
    
    // Stage 3: Send to next rank OR final norm + LM head
    if (!pp_config.is_last_stage) {
        graph.addNode("send_activations",
            ComputeStageFactory::createSendActivations(send_params));
        graph.addDependency("send_activations", prev_node);
    } else {
        // Add final_norm + lm_head as before
    }
    
    return graph;
}
```

### 2.2 Activation Passing Between Pipeline Stages

**Current Gap**: No point-to-point communication primitives exist in `MPIContext`.

**Required MPIContext Extensions**:
```cpp
class MPIContext {
public:
    // EXISTING: Collective operations
    void allreduce_sum_inplace(float* data, size_t count) const;
    void allgather(const float* send, float* recv, size_t count) const;
    
    // NEW: Point-to-point operations for pipeline parallel
    
    /**
     * @brief Blocking send to destination rank
     * @param data Buffer to send
     * @param count Number of elements
     * @param dest Destination rank
     * @param tag Message tag (for matching)
     */
    void send(const float* data, size_t count, int dest, int tag = 0) const;
    void send_bytes(const void* data, size_t bytes, int dest, int tag = 0) const;
    
    /**
     * @brief Blocking receive from source rank
     * @param data Buffer to receive into
     * @param count Number of elements
     * @param source Source rank
     * @param tag Message tag (for matching)
     */
    void recv(float* data, size_t count, int source, int tag = 0) const;
    void recv_bytes(void* data, size_t bytes, int source, int tag = 0) const;
    
    // NEW: Non-blocking operations for overlapped compute/communication
    
    /**
     * @brief Non-blocking send
     * @return Request handle for wait/test
     */
    MPI_Request isend(const float* data, size_t count, int dest, int tag = 0) const;
    MPI_Request isend_bytes(const void* data, size_t bytes, int dest, int tag = 0) const;
    
    /**
     * @brief Non-blocking receive
     * @return Request handle for wait/test
     */
    MPI_Request irecv(float* data, size_t count, int source, int tag = 0) const;
    MPI_Request irecv_bytes(void* data, size_t bytes, int source, int tag = 0) const;
    
    /**
     * @brief Wait for async operation to complete
     */
    void wait(MPI_Request& request) const;
    bool test(MPI_Request& request) const;  // Non-blocking check
};
```

**New Compute Stages**:
```cpp
// SendActivationsStage: Send hidden states to next pipeline stage
class SendActivationsStage : public IComputeStage {
public:
    struct Params {
        TensorBase* hidden;      // [seq_len, d_model]
        int dest_rank;           // Next pipeline stage
        int tag;                 // MPI tag (layer index)
        MPIContext* mpi_ctx;
        bool async = false;      // Use non-blocking send
    };
    
    bool execute(void* ctx) override {
        mpi_ctx_->send(hidden_->fp32_data(), hidden_->numel(), dest_rank_, tag_);
        return true;
    }
};

// ReceiveActivationsStage: Receive hidden states from previous pipeline stage
class ReceiveActivationsStage : public IComputeStage {
public:
    struct Params {
        TensorBase* hidden;      // Output buffer [seq_len, d_model]
        int source_rank;         // Previous pipeline stage
        int tag;                 // MPI tag
        MPIContext* mpi_ctx;
        bool async = false;
    };
    
    bool execute(void* ctx) override {
        mpi_ctx_->recv(hidden_->mutable_fp32_data(), hidden_->numel(), source_rank_, tag_);
        return true;
    }
};
```

### 2.3 ComputeGraph/DeviceGraphExecutor Changes

**Current Limitation**: `DeviceGraphExecutor::execute()` is synchronous—no support for interleaved execution with communication.

**Required Changes**:

1. **Add `ExecutionMode::PIPELINED`** (already defined but not implemented):
```cpp
// DeviceGraphExecutor.cpp line 365
case ExecutionMode::PIPELINED:
    LOG_WARN("[DeviceGraphExecutor] Pipelined mode not yet implemented, using sequential");
    return executeSequential(graph, ctx);
```

2. **Implement `executePipelined()`**:
```cpp
bool DeviceGraphExecutor::executePipelined(ComputeGraph& graph, IDeviceContext* ctx) {
    // For GPipe-style execution:
    // 1. Process micro-batches in order
    // 2. Overlap communication with computation where possible
    // 3. Handle warmup and cooldown phases
    
    // For 1F1B (one-forward-one-backward) style:
    // 1. Interleave forward and backward passes
    // 2. Minimize bubble time
    
    // This is complex and requires micro-batch tracking
    throw std::runtime_error("Pipelined mode not implemented");
}
```

3. **ComputeGraph Extensions**:
```cpp
class ComputeGraph {
public:
    // NEW: Support for pipeline parallel graph composition
    
    /**
     * @brief Set pipeline stage metadata
     */
    void setPipelineStage(int stage_id, int total_stages);
    
    /**
     * @brief Get nodes that perform cross-rank communication
     */
    std::vector<std::string> getCommunicationNodes() const;
    
    /**
     * @brief Split graph for micro-batch execution
     * @param micro_batch_count Number of micro-batches
     * @return Vector of subgraphs, one per micro-batch
     */
    std::vector<ComputeGraph> splitForMicroBatches(int micro_batch_count) const;
};
```

---

## 3. Cross-Rank Communication Infrastructure

### 3.1 Existing MPI Infrastructure ([MPIContext.h](../../../../src/v2/utils/MPIContext.h))

**Available Primitives**:
| Method | Type | Use Case |
|--------|------|----------|
| `allreduce_sum()` | Collective | Tensor parallel GEMM output |
| `allreduce_sum_inplace()` | Collective | In-place reduction |
| `allreduce_q8_1_inplace()` | Collective | Quantized reduction |
| `allgather()` | Collective | LM head output gather |
| `allgather_bytes()` | Collective | Raw byte gather |
| `broadcast()` | Collective | Root-to-all |
| `barrier()` | Collective | Synchronization |
| `get_local_slice()` | Helper | Work distribution |

**Missing for Pipeline Parallel**:
| Method | Type | Use Case |
|--------|------|----------|
| `send()` | Point-to-Point | Send activations to next stage |
| `recv()` | Point-to-Point | Receive activations from prev stage |
| `isend()` | Async P2P | Non-blocking send |
| `irecv()` | Async P2P | Non-blocking receive |
| `wait()` | Async | Wait for async operation |
| `test()` | Async | Check if async operation complete |

### 3.2 Point-to-Point Implementation

**Implementation Approach**:
```cpp
void MPIContext::send(const float* data, size_t count, int dest, int tag) const {
    MPI_Send(data, count, MPI_FLOAT, dest, tag, comm_);
}

void MPIContext::recv(float* data, size_t count, int source, int tag) const {
    MPI_Recv(data, count, MPI_FLOAT, source, tag, comm_, MPI_STATUS_IGNORE);
}

MPI_Request MPIContext::isend(const float* data, size_t count, int dest, int tag) const {
    MPI_Request req;
    MPI_Isend(data, count, MPI_FLOAT, dest, tag, comm_, &req);
    return req;
}

MPI_Request MPIContext::irecv(float* data, size_t count, int source, int tag) const {
    MPI_Request req;
    MPI_Irecv(data, count, MPI_FLOAT, source, tag, comm_, &req);
    return req;
}

void MPIContext::wait(MPI_Request& request) const {
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

bool MPIContext::test(MPI_Request& request) const {
    int flag;
    MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
    return flag != 0;
}
```

### 3.3 Async Communication Support

**For Optimal Pipeline Efficiency**: Overlap communication with computation.

```cpp
// Pipeline stage execution with async communication
class PipelineStageExecutor {
public:
    void execute(ComputeGraph& graph, int micro_batch_id) {
        // 1. Post receive for next micro-batch (async)
        MPI_Request recv_req;
        if (!is_first_stage_ && micro_batch_id < num_micro_batches_ - 1) {
            recv_req = mpi_ctx_->irecv(recv_buffer_, hidden_size_, prev_rank_, micro_batch_id + 1);
        }
        
        // 2. Execute local computation
        executor_.execute(graph, ctx_);
        
        // 3. Send result to next stage (async)
        MPI_Request send_req;
        if (!is_last_stage_) {
            send_req = mpi_ctx_->isend(send_buffer_, hidden_size_, next_rank_, micro_batch_id);
        }
        
        // 4. Wait for receives needed for this micro-batch
        // ...
    }
};
```

---

## 4. Scheduling Considerations

### 4.1 Micro-batching

**Current State**: No micro-batch concept exists. Each `forward()` call processes `[batch_size, seq_len]` tokens as a unit.

**For Pipeline Parallel**: Must split input into micro-batches to fill pipeline.

**Proposed Changes to `GraphOrchestrator::forward()`**:
```cpp
TensorBase* GraphOrchestrator::forward(const int* tokens, int seq_len) {
    if (!pp_config_.enabled) {
        // Current path: single forward pass
        return forwardSingle(tokens, seq_len);
    }
    
    // Pipeline parallel path: micro-batch execution
    const int micro_batch_size = seq_len / pp_config_.micro_batches;
    std::vector<MPI_Request> pending_sends;
    std::vector<MPI_Request> pending_recvs;
    
    // GPipe-style schedule: all forwards, then wait
    for (int mb = 0; mb < pp_config_.micro_batches; ++mb) {
        const int* mb_tokens = tokens + mb * micro_batch_size;
        
        // Post async receive from previous stage (if not first)
        if (!pp_config_.is_first_stage) {
            pending_recvs.push_back(
                mpi_ctx_->irecv(recv_buffers_[mb], hidden_size_, prev_rank_, mb)
            );
        }
        
        // Wait for input from previous stage
        if (!pp_config_.is_first_stage) {
            mpi_ctx_->wait(pending_recvs[mb]);
        }
        
        // Execute local layers
        ComputeGraph graph = buildGraphForMicroBatch(mb_tokens, micro_batch_size);
        executor_.execute(graph, ctx_);
        
        // Post async send to next stage (if not last)
        if (!pp_config_.is_last_stage) {
            pending_sends.push_back(
                mpi_ctx_->isend(send_buffers_[mb], hidden_size_, next_rank_, mb)
            );
        }
    }
    
    // Wait for all sends to complete
    for (auto& req : pending_sends) {
        mpi_ctx_->wait(req);
    }
    
    // Only last stage returns logits
    return pp_config_.is_last_stage ? state_.logits.get() : nullptr;
}
```

### 4.2 Iteration-Level Pipelining for Decode

**Challenge**: Decode generates 1 token at a time. With pipeline depth D, throughput is limited by latency, not compute.

**Options**:

1. **Batch multiple sequences** (recommended for high throughput):
   - Process multiple independent sequences in parallel
   - Each sequence at different decode position
   - Requires multi-sequence KV cache (already supported)

2. **Speculative decoding** (advanced):
   - Predict multiple future tokens on one stage
   - Verify on other stages
   - Complex to implement

3. **Simple round-robin** (baseline):
   - Just process token through pipeline stages sequentially
   - High latency, low throughput
   - Simple to implement

### 4.3 Buffer Management Changes

**Current**: Single set of activation buffers, reused across layers:
```cpp
struct InferenceState {
    std::shared_ptr<TensorBase> hidden;      // Single hidden buffer
    std::shared_ptr<TensorBase> normalized;  // Single norm buffer
    // ... other single buffers
};
```

**For Micro-batching**: Need multiple buffer sets:
```cpp
struct PipelineBuffers {
    // Per micro-batch buffers for overlapped execution
    std::vector<std::shared_ptr<TensorBase>> hidden_buffers;
    std::vector<std::shared_ptr<TensorBase>> send_buffers;
    std::vector<std::shared_ptr<TensorBase>> recv_buffers;
    
    // Activation buffers can still be shared (only one micro-batch in flight per stage)
    std::shared_ptr<TensorBase> normalized;
    std::shared_ptr<TensorBase> Q, K, V;
    // ...
};
```

---

## 5. Weight and KV Cache Placement

### 5.1 WeightManager Capabilities ([WeightManager.h](../../../../src/v2/loaders/WeightManager.h))

**Current Support**:
- `WeightDistributionStrategy::REPLICATED`: Full copy per rank (default)
- `WeightDistributionStrategy::SHARDED`: Tensor-parallel sharding (rows/columns)
- `getWeight()`: Loads by name, supports device placement

**Key Insight**: Weight loading is **by name**, and layer weights have predictable names:
```
blk.0.attn_q.weight    // Layer 0 Q weights
blk.0.attn_k.weight    // Layer 0 K weights
...
blk.27.ffn_down.weight // Layer 27 FFN down
```

**For Pipeline Parallel**: Easy to load only assigned layers:
```cpp
// In Qwen2Graph or GraphOrchestrator
void loadWeightsForPipelineStage(WeightManager& mgr, int layer_start, int layer_end) {
    // Only first stage needs embedding
    if (is_first_stage_) {
        mgr.getWeight("token_embd.weight", device_);
    }
    
    // Load assigned layers only
    for (int l = layer_start; l < layer_end; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        mgr.getWeight(prefix + "attn_q.weight", device_);
        mgr.getWeight(prefix + "attn_k.weight", device_);
        mgr.getWeight(prefix + "attn_v.weight", device_);
        mgr.getWeight(prefix + "attn_output.weight", device_);
        mgr.getWeight(prefix + "attn_norm.weight", device_);
        mgr.getWeight(prefix + "ffn_gate.weight", device_);
        mgr.getWeight(prefix + "ffn_up.weight", device_);
        mgr.getWeight(prefix + "ffn_down.weight", device_);
        mgr.getWeight(prefix + "ffn_norm.weight", device_);
    }
    
    // Only last stage needs final norm + LM head
    if (is_last_stage_) {
        mgr.getWeight("output_norm.weight", device_);
        mgr.getWeight("output.weight", device_);  // LM head
    }
}
```

### 5.2 KV Cache Per-Layer Support ([IKVCache.h](../../../../src/v2/kernels/IKVCache.h))

**Current Implementation**: KV cache is **already per-layer**:
```cpp
class IKVCache {
public:
    // Get KV for specific layer
    virtual bool get_kv(int layer, int seq_idx,
                        ITensor** out_k, ITensor** out_v,
                        int* out_kv_len = nullptr) = 0;
    
    virtual int get_cached_tokens(int layer, int seq_idx = 0) const = 0;
    virtual int n_layers() const = 0;
};
```

**For Pipeline Parallel**: Each rank only needs KV cache for its layers:
```cpp
// Create KV cache only for assigned layers
std::unique_ptr<IKVCache> createKVCacheForStage(int layer_start, int layer_end) {
    int num_layers = layer_end - layer_start;
    auto cache = std::make_unique<CPUKVCache>(
        num_layers,  // Only assigned layers
        n_kv_heads_,
        head_dim_,
        max_seq_len_,
        precision_
    );
    return cache;
}

// When accessing, map global layer index to local
int local_layer = global_layer - layer_start_;
kv_cache_->get_kv(local_layer, seq_idx, &k, &v, &kv_len);
```

---

## 6. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                         Pipeline Parallel Execution (2 Stages)                               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

  Rank 0 (Layers 0-13)                                    Rank 1 (Layers 14-27)
  ─────────────────────                                   ─────────────────────
  
  ┌─────────────────┐                                     
  │   EMBEDDING     │  tokens → hidden[0]                 
  └────────┬────────┘                                     
           │                                              
           ▼                                              
  ┌─────────────────┐                                     
  │   Layer 0       │                                     
  │   (Attn + FFN)  │                                     
  └────────┬────────┘                                     
           │                                              
          ...                                             
           │                                              
  ┌─────────────────┐                                     
  │   Layer 13      │                                     
  │   (Attn + FFN)  │                                     
  └────────┬────────┘                                     
           │                                              
           │  MPI_Send(hidden, rank=1)                    
           └──────────────────────────────────────────────►┌─────────────────┐
                                                           │  MPI_Recv       │
                                                           └────────┬────────┘
                                                                    │
                                                           ┌────────▼────────┐
                                                           │   Layer 14      │
                                                           │   (Attn + FFN)  │
                                                           └────────┬────────┘
                                                                    │
                                                                   ...
                                                                    │
                                                           ┌────────▼────────┐
                                                           │   Layer 27      │
                                                           │   (Attn + FFN)  │
                                                           └────────┬────────┘
                                                                    │
                                                           ┌────────▼────────┐
                                                           │   FINAL_NORM    │
                                                           └────────┬────────┘
                                                                    │
                                                           ┌────────▼────────┐
                                                           │    LM_HEAD      │
                                                           └────────┬────────┘
                                                                    │
                                                                    ▼
                                                               logits[vocab]


┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                      GPipe Micro-batch Schedule (4 micro-batches, 2 stages)                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

Time ──────────────────────────────────────────────────────────────────────────────────►

Rank 0:  │ F(mb0) │ F(mb1) │ F(mb2) │ F(mb3) │ ════════ BUBBLE ═══════ │
         │   ↓    │   ↓    │   ↓    │   ↓    │
         │ SEND   │ SEND   │ SEND   │ SEND   │
         │        │        │        │        │

Rank 1:  │ ═══ BUBBLE ═══ │ RECV   │ RECV   │ RECV   │ RECV   │
         │                │   ↓    │   ↓    │   ↓    │   ↓    │
         │                │ F(mb0) │ F(mb1) │ F(mb2) │ F(mb3) │

Legend: F(mbX) = Forward pass for micro-batch X
        SEND/RECV = Point-to-point activation transfer
        BUBBLE = Pipeline idle time (inefficiency)
```

---

## 7. Risk Areas and Complexity Estimates

### 7.1 High Risk / High Complexity

| Area | Risk | Complexity | Mitigation |
|------|------|------------|------------|
| **Point-to-point communication** | Deadlock, ordering bugs | High | Careful tag management, extensive testing |
| **Micro-batch scheduling** | Pipeline bubbles, memory overhead | High | Start with simple GPipe, optimize later |
| **Async overlap** | Race conditions, data corruption | High | Careful synchronization, buffer management |

### 7.2 Medium Risk / Medium Complexity

| Area | Risk | Complexity | Mitigation |
|------|------|------------|------------|
| **Layer range assignment** | Off-by-one errors, uneven splits | Medium | Thorough unit tests |
| **Buffer management** | Memory bloat with many micro-batches | Medium | Limit micro-batch count, measure memory |
| **Graph building changes** | Breaking existing TP code | Medium | Feature flag, extensive regression testing |

### 7.3 Low Risk / Low Complexity

| Area | Risk | Complexity | Mitigation |
|------|------|------------|------------|
| **Weight loading by layer** | Minimal—names are predictable | Low | Already supported by WeightManager |
| **KV cache partitioning** | Minimal—already per-layer | Low | Just allocate fewer layers |
| **Config parsing** | Minimal | Low | Standard CLI/env var handling |

---

## 8. Suggested Phased Implementation

### Phase 1: Foundation (1-2 weeks)
**Goal**: Point-to-point communication and basic layer range execution

1. **Add P2P to MPIContext** (2 days)
   - `send()`, `recv()` synchronous versions
   - Unit tests for P2P communication

2. **Create SendActivationsStage and ReceiveActivationsStage** (2 days)
   - Compute stages for activation transfer
   - Integration with ComputeGraph

3. **Modify Qwen2Graph for layer ranges** (3 days)
   - Add `PipelineParallelConfig`
   - Modify `buildFullForwardGraph()` to respect layer ranges
   - Conditional embedding/LM head based on stage position

4. **Basic 2-rank test** (2 days)
   - Simple end-to-end test with 2 pipeline stages
   - Verify correctness against single-rank execution

### Phase 2: Micro-batching (2-3 weeks)
**Goal**: GPipe-style micro-batch scheduling for better utilization

1. **Add micro-batch buffer management** (3 days)
   - Multiple hidden/send/recv buffers per stage
   - Memory-efficient allocation strategy

2. **Implement GPipe schedule** (5 days)
   - Forward all micro-batches on each stage
   - Proper send/recv ordering
   - Handle pipeline warmup/cooldown

3. **Add async P2P operations** (3 days)
   - `isend()`, `irecv()`, `wait()`, `test()`
   - Overlap communication with computation

4. **Performance benchmarking** (3 days)
   - Measure pipeline efficiency
   - Profile bubble time
   - Optimize buffer sizes

### Phase 3: Advanced Features (2-3 weeks)
**Goal**: Production-ready pipeline parallelism

1. **1F1B scheduling** (optional, 5 days)
   - Interleaved forward/backward for memory efficiency
   - Complex but better memory footprint

2. **Hybrid TP+PP** (5 days)
   - Tensor parallel within pipeline stages
   - Combine AllreduceStage with SendActivationsStage

3. **Decode optimization** (3 days)
   - Batched decode with multiple sequences
   - Per-sequence position tracking across pipeline

4. **Documentation and examples** (2 days)
   - Usage guide
   - Configuration examples
   - Performance tuning tips

---

## 9. Existing Code That Can Be Reused

| Component | Location | How to Reuse |
|-----------|----------|--------------|
| `MPIContext` | `src/v2/utils/MPIContext.h` | Extend with P2P methods |
| `Qwen2Graph::buildAttentionGraph/buildFFNGraph` | `src/v2/models/qwen/Qwen2Graph.cpp` | Already per-layer, reuse directly |
| `LayerGraphCache` | `src/v2/execution/GraphOrchestrator.h` | Cache per-layer graphs per stage |
| `ComputeGraph::merge()` | `src/v2/execution/DeviceGraphExecutor.cpp` | Compose stage subgraphs |
| `WeightManager::getWeight()` | `src/v2/loaders/WeightManager.h` | Load by layer name |
| `IKVCache` per-layer interface | `src/v2/kernels/IKVCache.h` | Allocate partial cache |
| `AllreduceStage`, `AllGatherStage` | `src/v2/execution/compute_stages/` | Pattern for new P2P stages |
| `MPIStrategy::PipelineParallel` | `src/v2/utils/MPIStrategy.h` | Strategy enum already defined |

---

## 10. New Components Needed

| Component | Description | Estimated Effort |
|-----------|-------------|------------------|
| `MPIContext::send/recv` | Synchronous P2P | 1 day |
| `MPIContext::isend/irecv/wait` | Async P2P | 1 day |
| `SendActivationsStage` | Send hidden to next rank | 1 day |
| `ReceiveActivationsStage` | Recv hidden from prev rank | 1 day |
| `PipelineParallelConfig` | PP configuration struct | 0.5 days |
| `Qwen2Graph::buildForwardGraph_PP` | PP-aware graph building | 3 days |
| `GraphOrchestrator::forward_PP` | Micro-batch scheduling | 5 days |
| `PipelineBuffers` | Multi-buffer management | 2 days |
| `DeviceGraphExecutor::executePipelined` | Pipelined execution mode | 3 days |
| Unit/integration tests | Test coverage | 3 days |

**Total Estimate**: 3-4 weeks for basic PP, 6-8 weeks for production-ready

---

## 11. Conclusion

Pipeline Parallelism is **feasible** in Llaminar V2 with moderate architectural changes. The key gaps are:

1. **Point-to-point MPI operations** (not complex, just missing)
2. **Micro-batch scheduling logic** (complex, the bulk of the work)
3. **Buffer management for overlapped execution** (moderate complexity)

The existing infrastructure provides a solid foundation:
- Per-layer graph building ✓
- Per-layer KV cache ✓
- Layer weight loading by name ✓
- MPI context abstraction ✓
- Strategy enum defined ✓

**Recommendation**: Start with Phase 1 (basic 2-rank PP without micro-batching) to prove the concept, then iterate on scheduling for better utilization.

---

## References

- [Qwen2Graph.cpp](../../src/v2/models/qwen/Qwen2Graph.cpp) - Layer graph building
- [GraphOrchestrator.cpp](../../src/v2/execution/GraphOrchestrator.cpp) - Execution orchestration
- [MPIContext.h](../../../../src/v2/utils/MPIContext.h) - MPI primitives
- [MPIStrategy.h](../../../../src/v2/utils/MPIStrategy.h) - Strategy definitions
- [WeightManager.h](../../../../src/v2/loaders/WeightManager.h) - Weight loading
- [IKVCache.h](../../../../src/v2/kernels/IKVCache.h) - KV cache interface
- [v2_mpi_parallelization_design.md](../../../v2_mpi_parallelization_design.md) - Existing MPI design doc
