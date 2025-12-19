# GraphExecutor Buffer Management Plan

**Author:** David Sanftenberg  
**Date:** December 2025  
**Status:** Phase 4 Complete (Phases 1-4 Done, Phase 5 Pending)

## Overview

Transfer buffer management from `PipelineBase`/`Qwen2Pipeline` to the `GraphExecutor` framework with a standardized, type-safe system for buffer classification and zero hot-path allocations.

## Goals

1. **Clear Buffer Semantics**: Every buffer has a formal role (INPUT, OUTPUT, INOUT, SCRATCH, WEIGHT)
2. **Zero Hot-Path Allocations**: All buffers pre-allocated before graph execution
3. **Buffer Reuse**: Liveness analysis enables SCRATCH buffer sharing
4. **Memory Budget Tracking**: Know memory requirements before execution
5. **Gradual Migration**: Support both pipeline-owned and graph-owned modes

## Non-Goals (Out of Scope)

- Automatic differentiation / backward pass
- Dynamic shape inference (shapes known at graph build time)
- Cross-graph buffer sharing

---

## Architecture

### Buffer Role Classification

```cpp
enum class BufferRole {
    INPUT,   // Read-only, consumed by stage
    OUTPUT,  // Write-only, produced by stage
    INOUT,   // Read-modify-write (e.g., residual accumulation)
    SCRATCH, // Temporary workspace, undefined after execute()
    WEIGHT,  // Read-only model parameters
};

// Type-safe tensor type for buffer allocation
enum class BufferTensorType {
    FP32,
    BF16,
    FP16,
    Q8_0,
    Q8_1,
    Q4_0,
    Q4_1,
    IQ4_NL,
    OTHER,
};
```

### Component Hierarchy

```
GraphExecutor
├── GraphBufferManager          [COMPLETE] Buffer allocation & lifetime
│   ├── BufferDescriptor        [COMPLETE] Single buffer spec
│   ├── StageBufferRequirements [COMPLETE] All buffers for a stage
│   └── LivenessAnalyzer        [COMPLETE] SCRATCH reuse optimization
├── ComputeGraph                [EXISTING] DAG of stages
└── ComputeNode                 [EXISTING] Stage + metadata
```

### Stage Contract

Every `IComputeStage` must:
1. Declare buffer requirements via `getBufferRequirements()` ✅ All 16 stages implemented
2. Use `TensorBase*` for buffer parameters (type-safe introspection) ✅ Migrated
3. Perform ZERO allocations in `execute()`

---

## Implementation Phases

### Phase 1: Buffer Role Infrastructure ✅ COMPLETE

**Goal**: Add type system without changing behavior.

**Files Created**:
- `src/v2/execution/BufferRole.h` - Enum, descriptor types, helper functions

**Files Modified**:
- `src/v2/execution/ComputeStage.h` - Added `getBufferRequirements()` virtual

**Deliverables**:
- [x] `BufferRole` enum (INPUT, OUTPUT, INOUT, SCRATCH, WEIGHT)
- [x] `BufferTensorType` enum (FP32, BF16, FP16, Q8_0, Q8_1, Q4_0, Q4_1, IQ4_NL, OTHER)
- [x] `BufferDescriptor` struct with name, role, shape, tensor_type
- [x] `StageBufferRequirements` struct with helper methods (addInput, addOutput, etc.)
- [x] `toBufferTensorType()` helper for TensorType → BufferTensorType conversion
- [x] `IComputeStage::getBufferRequirements()` (default empty)
- [x] Unit tests: 8 tests in Test__BufferRole.cpp

**Breaking Changes**: None

---

### Phase 2: GraphBufferManager Core ✅ COMPLETE

**Goal**: Implement buffer allocation without reuse optimization.

**Files Created**:
- `src/v2/execution/GraphBufferManager.h`
- `src/v2/execution/GraphBufferManager.cpp`
- `tests/v2/unit/Test__GraphBufferManager.cpp`

**Deliverables**:
- [x] `GraphBufferManager` class
- [x] `registerBuffer()` - register named buffer requirements
- [x] `allocateAll()` - allocate all registered buffers
- [x] `getBuffer()` - retrieve named buffer as TensorBase*
- [x] `releaseAll()` - cleanup all allocations
- [x] `getTotalMemoryBytes()` - memory budget tracking
- [x] `getBufferMemoryBytes()` - per-buffer memory query
- [x] Device-aware allocation (device_idx parameter)
- [x] Unit tests: 32 tests in Test__GraphBufferManager.cpp

**Breaking Changes**: None (new API alongside existing)

---

### Phase 3: Stage Migration ✅ COMPLETE

**Goal**: Update all stages to declare requirements with type-safe TensorBase* parameters.

**All 16 Stages Migrated**:

| Stage | Category | Status | Tests |
|-------|----------|--------|-------|
| `GEMMStage` | GEMM | ✅ | 2 |
| `FusedQKVGEMMStage` | GEMM | ✅ | 2 |
| `FusedGateUpGEMMStage` | GEMM | ✅ | 2 |
| `RMSNormStage` | Norm | ✅ | 2 |
| `RoPEStage` | Transform | ✅ | 2 |
| `SwiGLUStage` | Activation | ✅ | 2 |
| `ResidualAddStage` | Residual | ✅ | 2 |
| `EmbeddingStage` | Model | ✅ | 2 |
| `LMHeadStage` | Model | ✅ | 3 |
| `AttentionWithKVCacheStage` | Attention | ✅ | 2 |
| `AttentionComputeStage` | Attention | ✅ | 2 |
| `KVCacheAppendStage` | KV Cache | ✅ | 2 |
| `AllreduceStage` | MPI | ✅ | 2 |
| `MoERouterStage` | MoE | ✅ | 2 |
| `MoEExpertStage` | MoE | ✅ | 2 |
| `MoECombineStage` | MoE | ✅ | 2 |

**Key Design Changes**:
- Migrated `AllreduceStage` from `void* buffer + size_t count` to `TensorBase* buffer`
- Migrated MoE stages from `void*` to `TensorBase*` for full type introspection
- Removed deprecated `AttentionStage` class (~170 lines)

**Test Coverage**: 35 tests in Test__StageBufferRequirements.cpp

**Breaking Changes**: None (getBufferRequirements has default impl)

---

### Phase 4: Liveness Analysis ✅ COMPLETE

**Goal**: Enable SCRATCH buffer sharing for memory efficiency.

**Files Created**:
- `src/v2/execution/LivenessAnalyzer.h`
- `src/v2/execution/LivenessAnalyzer.cpp`
- `tests/v2/unit/Test__LivenessAnalyzer.cpp`

**Algorithm**:
1. Build execution order from ComputeGraph (topological sort)
2. For each buffer, track first-use and last-use stage indices
3. Buffers with non-overlapping lifetimes can alias (share memory)
4. Use greedy interval graph coloring for optimal aliasing assignment

**Key Design**:
```cpp
struct BufferLiveness {
    std::string buffer_name;
    std::string stage_name;     // Stage where declared
    size_t first_use_idx;       // Stage index where buffer is first accessed
    size_t last_use_idx;        // Stage index where buffer is last accessed
    BufferRole role;            // Only SCRATCH buffers eligible for aliasing
    BufferTensorType tensor_type;
    std::vector<size_t> shape;
    size_t size_bytes;
    
    // Check if two lifetimes overlap
    bool overlaps(const BufferLiveness& other) const;
};

struct AliasingGroup {
    std::vector<std::string> buffer_names;  // Buffers that share memory
    size_t max_size_bytes;                   // Allocation size (max of all)
    BufferTensorType tensor_type;
};

class LivenessAnalyzer {
public:
    // Analyze a graph and return buffer lifetimes
    std::vector<BufferLiveness> analyze(const ComputeGraph& graph);
    
    // Compute aliasing groups (buffers that can share memory)
    std::vector<AliasingGroup> computeAliasingGroups(
        const std::vector<BufferLiveness>& lifetimes);
    
    // Get memory usage (original vs optimized)
    std::pair<size_t, size_t> computeMemoryUsage(
        const std::vector<BufferLiveness>& lifetimes,
        const std::vector<AliasingGroup>& groups);
        
    // Get savings percentage
    double computeSavingsPercent(
        const std::vector<BufferLiveness>& lifetimes,
        const std::vector<AliasingGroup>& groups);
    
    // Check if two buffers can alias
    static bool canAlias(const BufferLiveness& a, const BufferLiveness& b);
    
    // Filter only SCRATCH buffers
    static std::vector<BufferLiveness> filterScratchBuffers(
        const std::vector<BufferLiveness>& lifetimes);
};
```

**Deliverables**:
- [x] `BufferLiveness` struct with overlaps() method
- [x] `AliasingGroup` struct
- [x] `LivenessAnalyzer` class
- [x] `analyze()` - compute buffer lifetimes from graph
- [x] `computeAliasingGroups()` - greedy interval coloring
- [x] `computeMemoryUsage()` - original vs optimized bytes
- [x] `computeSavingsPercent()` - percentage savings
- [x] `canAlias()` - aliasing compatibility check
- [x] `filterScratchBuffers()` - extract SCRATCH-only buffers
- [x] Type compatibility (FP32/BF16/FP16 can alias; quantized only same-type)
- [x] Unit tests: 28 tests in Test__LivenessAnalyzer.cpp

**Memory Savings Demo**: Integration test shows ~24% savings for attention+FFN pattern

**Breaking Changes**: None (optimization only)

---

### Phase 5: Qwen2Graph Integration ← NEXT

**Goal**: Qwen2Graph uses GraphBufferManager instead of external buffers.

**Files to Modify**:
- `src/v2/pipelines/qwen/Qwen2Graph.h` - Add buffer spec
- `src/v2/pipelines/qwen/Qwen2Graph.cpp` - Use managed buffers
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - Config flag for graph mode

**Deliverables**:
- [ ] `Qwen2BufferSpec` - formal buffer specification
- [ ] `Qwen2Graph::initializeBuffers()` - setup via manager
- [ ] `Qwen2Pipeline` config: `use_graph_buffer_management`
- [ ] Integration tests comparing both modes
- [ ] Performance benchmark

**Breaking Changes**: Opt-in via config flag

---

### Phase 6: Full Migration (Future)

**Goal**: Make graph-managed buffers the default.

**Deliverables**:
- [ ] Remove `ActivationBuffers` from `PipelineBase`
- [ ] Deprecate `setBuffers()` pattern in Qwen2Graph
- [ ] Update all tests
- [ ] Documentation

**Breaking Changes**: Yes (major version bump)

---

## Buffer Specification: Qwen2

### Per-Layer Buffers

| Name | Role | Shape | Type | Notes |
|------|------|-------|------|-------|
| `hidden_input` | INPUT | [seq, d_model] | FP32/Q8_1 | From embedding or prev layer |
| `hidden_output` | OUTPUT | [seq, d_model] | FP32/Q8_1 | To next layer or LM head |
| `residual` | INOUT | [seq, d_model] | FP32/Q8_1 | Accumulator stream |
| `normalized` | SCRATCH | [seq, d_model] | FP32 | Post-norm, reused attn/FFN |
| `Q` | SCRATCH | [seq, n_heads*head_dim] | FP32 | Query projection |
| `K` | SCRATCH | [seq, n_kv_heads*head_dim] | FP32 | Key projection |
| `V` | SCRATCH | [seq, n_kv_heads*head_dim] | FP32 | Value projection |
| `attn_scores` | SCRATCH | [n_heads, seq, kv_len] | FP32 | Attention weights |
| `attn_context` | SCRATCH | [seq, n_heads*head_dim] | FP32 | Post-attention |
| `attn_proj` | SCRATCH | [seq, d_model] | FP32 | After Wo projection |
| `gate` | SCRATCH | [seq, d_ff] | FP32 | FFN gate projection |
| `up` | SCRATCH | [seq, d_ff] | FP32 | FFN up projection |
| `ffn_inter` | SCRATCH | [seq, d_ff] | FP32 | Post-SwiGLU |
| `ffn_output` | SCRATCH | [seq, d_model] | FP32 | After down projection |

### Buffer Aliasing Opportunities

Non-overlapping lifetimes enable reuse:
- `Q` ↔ `gate` (attention done before FFN)
- `K` ↔ `up` (attention done before FFN)
- `V` ↔ `ffn_inter` (attention done before FFN)
- `attn_context` ↔ `ffn_output` (sequential use)

### Model-Level Buffers

| Name | Role | Shape | Type | Notes |
|------|------|-------|------|-------|
| `embedding_output` | OUTPUT | [seq, d_model] | FP32 | From embedding lookup |
| `final_hidden` | INOUT | [seq, d_model] | FP32 | After all layers |
| `logits` | OUTPUT | [seq, vocab_size] | FP32 | Final predictions |

---

## Testing Strategy

### Unit Tests
- `Test__BufferRole.cpp` - Enum and descriptor tests
- `Test__GraphBufferManager.cpp` - Allocation, retrieval, release
- `Test__LivenessAnalyzer.cpp` - Aliasing correctness

### Integration Tests
- `Test__GraphExecutorBuffers.cpp` - Full graph with managed buffers
- `Test__Qwen2GraphBuffers.cpp` - Qwen2-specific buffer flows

### Parity Tests
- Compare pipeline-owned vs graph-owned buffer results
- Verify numerical equivalence

### Performance Tests
- Memory usage comparison
- Allocation overhead measurement

---

## Open Questions

1. **KV Cache Ownership**: Keep in pipeline (persists across calls) or add PERSISTENT role?
   - **Decision**: Keep in pipeline for now, revisit in Phase 6

2. **Multi-Device Aliasing**: Per-device or global?
   - **Decision**: Per-device (cross-device buffers can't alias)

3. **Threadlocal Scratch**: Track or exempt?
   - **Decision**: Exempt from tracking, document as exception

---

## Success Metrics

1. **Zero hot-path allocations** in graph execution mode
2. **Memory reduction** via SCRATCH aliasing (target: 30-40%)
3. **No performance regression** vs current implementation
4. **Clear buffer contracts** - every stage documents its buffers

---

## Timeline

| Phase | Estimated Effort | Status |
|-------|-----------------|--------|
| Phase 1 | 1 day | ✅ Complete |
| Phase 2 | 2 days | ✅ Complete |
| Phase 3 | 3-4 days | ✅ Complete |
| Phase 4 | 2 days | ✅ Complete |
| Phase 5 | 2 days | 🔄 Next |
| Phase 6 | TBD | Future |

**Completed**: Phases 1-4 (~8-9 days, 103 unit tests total)
- Phase 1: 8 tests (BufferRole, BufferDescriptor)
- Phase 2: 32 tests (GraphBufferManager)
- Phase 3: 35 tests (StageBufferRequirements)
- Phase 4: 28 tests (LivenessAnalyzer)

**Remaining**: Phases 5-6 (~2-4 days)
