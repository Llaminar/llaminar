# Column-Parallel Tensor Sharding Implementation Plan

**Date**: December 3, 2025  
**Updated**: December 21, 2025  
**Status**: Phase 4 Complete ✅  
**Author**: David Sanftenberg

## Executive Summary

This document outlines the implementation plan for column-parallel weight sharding in Llaminar V2. Column-parallel sharding reduces memory usage and compute per rank by splitting weight matrices along the output dimension. This is Phase 4b of the tensor parallelism roadmap.

## Current State

### What's Already Implemented

1. **Row-Parallel Sharding** (Phase 4a - COMPLETE)
   - Weights: `attn_output.weight` (Wo), `ffn_down.weight` (Down)
   - Implementation: `TensorSlice` with `loadTensorRowSlice()` for memory-efficient loading
   - Allreduce: `project_row_parallel()` sums partial results across ranks
   - Status: ✅ Working with quantized tensors (preserves IQ4_NL, Q4_0, etc.)

2. **TensorSlice Infrastructure** (COMPLETE)
   - `SliceMetadata`: Contains original dimensions, slice range, rank, world_size, `inner_is_presliced`
   - `SliceMode`: `FULL`, `ROW_PARALLEL`, `COLUMN_PARALLEL`
   - `TensorSlice::createGemm()`: Creates sliced GEMM kernel for row-parallel weights

3. **Weight Distribution Strategy** (COMPLETE)
   - `WeightDistributionStrategy::SHARDED` enables sharding
   - `WeightManager::determineShardingMode()` decides per-weight sharding mode
   - `WeightManager::getShardedWeight()` loads sharded weights

4. **Column-Parallel Q/K/V (Phase 3 - COMPLETE)** ✅ December 21, 2025
   - Weights: `attn_q.weight`, `attn_k.weight`, `attn_v.weight` + biases
   - Implementation: `ShardingMode::COLUMN_PARALLEL` in WeightManager
   - Head distribution: `MPIContext::get_local_slice()` divides heads across ranks
   - Buffer allocation: `Qwen2BufferSpecBuilder` with local head counts
   - Graph config: `Qwen2GraphConfig` extended with `head_start`, `local_n_heads`, `local_n_kv_heads`, `qkv_column_parallel`
   - Unit tests: 6 new tests in `Test__Qwen2BufferSpec.cpp` for local head scenarios
   - Integration tests: 5 tests in `Test__MPI_ColumnParallelQKV.cpp` (all passing)

5. **Column-Parallel FFN Gate/Up (Phase 4 - COMPLETE)** ✅ December 21, 2025
   - Weights: `ffn_gate.weight`, `ffn_up.weight` (column-parallel), `ffn_down.weight` (input-parallel)
   - Implementation: `ShardingMode::COLUMN_PARALLEL` for Gate/Up, `ShardingMode::INPUT_PARALLEL` for Down
   - Dimension calculation: `d_ff_local = d_ff / world_size` (4864/2 = 2432 for 2 ranks)
   - Graph config: `Qwen2GraphConfig` extended with `d_ff_local`, `ffn_column_parallel`
   - Buffer allocation: FFN buffers sized to `d_ff_local` when sharding enabled
   - Integration tests: 7 tests in `Test__MPI_ColumnParallelFFN.cpp` (all passing)

### What's Next (Phase 5: Full System Integration Testing)

With both QKV and FFN column-parallel sharding complete, the next phase focuses on:

```cpp
// Column-parallel weights: DISABLED for Phase 1
// TODO: Enable when attention infrastructure supports local-only Q/K/V
// if (name.find("attn_q.weight") != std::string::npos ||
//     name.find("attn_k.weight") != std::string::npos ||
//     name.find("attn_v.weight") != std::string::npos ||
//     name.find("ffn_gate.weight") != std::string::npos ||
//     name.find("ffn_up.weight") != std::string::npos)
// {
//     return ShardingMode::COLUMN_PARALLEL;
// }
```

## Architecture Analysis

### Column-Parallel vs Row-Parallel

| Aspect | Row-Parallel | Column-Parallel |
|--------|--------------|-----------------|
| Weight split | Along input dimension (K) | Along output dimension (N) |
| GEMM output | Partial sum (needs allreduce) | Local slice (no allreduce) |
| Memory per rank | `[N, K/world_size]` | `[N/world_size, K]` |
| Communication | Allreduce after GEMM | None (or allgather if full output needed) |
| Example weights | Wo, Down | Q, K, V, Gate, Up |

### Transformer FFN Flow

```
Input [seq, d_model]
    │
    ├──► Gate: [d_ff, d_model].T → [seq, d_ff]      ◄── Column-parallel candidate
    │
    ├──► Up:   [d_ff, d_model].T → [seq, d_ff]      ◄── Column-parallel candidate
    │
    ▼
SwiGLU(gate, up) → [seq, d_ff]
    │
    ▼
Down: [d_model, d_ff].T → [seq, d_model]            ◄── Row-parallel (DONE)
    │
    ▼
Allreduce → [seq, d_model]
```

With column-parallel Gate/Up:
- Each rank produces `[seq, d_ff_local]` where `d_ff_local = d_ff / world_size`
- SwiGLU operates on local slices
- Down projection receives `[seq, d_ff_local]` (matches its row-sliced input dimension)
- Final allreduce after Down sums partial results

### Transformer Attention Flow

```
Input [seq, d_model]
    │
    ├──► Q: [n_heads * head_dim, d_model].T → [seq, n_heads * head_dim]
    │
    ├──► K: [n_kv_heads * head_dim, d_model].T → [seq, n_kv_heads * head_dim]
    │
    ├──► V: [n_kv_heads * head_dim, d_model].T → [seq, n_kv_heads * head_dim]
    │
    ▼
MpiAttentionOrchestrator (extracts local heads internally)
    │
    ▼
Wo: [d_model, n_heads * head_dim].T → [seq, d_model]   ◄── Row-parallel (DONE)
    │
    ▼
Allreduce → [seq, d_model]
```

**Problem**: `MpiAttentionOrchestrator::compute_tensor_parallel()` currently:
1. Receives FULL Q/K/V tensors
2. Extracts local heads internally via `copy_head_slice()`
3. Performs local attention computation
4. Writes to local portion of output

Column-parallel Q/K/V would require the orchestrator to accept pre-sliced tensors.

## Implementation Strategy

Given the complexity difference, we'll implement in two phases:

### Phase 4b-1: FFN Column-Parallel (Gate/Up) - LOWER RISK

**Why start here:**
- FFN is a simple sequential pipeline (no complex orchestration)
- Down projection already expects sharded input (row-parallel)
- No changes to attention code
- Immediate memory/compute savings

**Changes required:**

1. **WeightManager.cpp** - Enable column-parallel for Gate/Up
2. **WeightManager.cpp** - Implement `loadTensorColumnSlice()` or use FP32 fallback
3. **Qwen2Pipeline.cpp** - Add `d_ff_local_` member, compute as `d_ff_ / world_size`
4. **Qwen2Pipeline.cpp** - Allocate FFN buffers with local dimensions
5. **Qwen2Pipeline.cpp** - Update FusedGEMM to use local output dimension
6. **Qwen2Pipeline.cpp** - Update SwiGLU to use local dimension
7. **TensorSlice.h** - Add column-parallel GEMM kernel creation (if not using FP32 fallback)

### Phase 4b-2: Attention Column-Parallel (Q/K/V) - HIGHER RISK

**Why more complex:**
- Attention orchestrator expects full Q/K/V
- GQA (Grouped Query Attention) complicates slicing
- KV cache dimensions may need adjustment
- Multi-rank synchronization in attention

**Changes required:**

1. **MpiAttentionOrchestrator.cpp** - Accept pre-sliced Q/K/V
2. **MpiAttentionOrchestrator.cpp** - Skip internal head slicing when input is pre-sliced
3. **Qwen2Pipeline.cpp** - Allocate Q/K/V buffers with local dimensions
4. **KVCache** - May need local head count awareness
5. **WeightManager.cpp** - Enable column-parallel for Q/K/V

## Detailed Implementation: Phase 4b-1 (FFN Gate/Up)

### Step 1: Enable Column-Parallel Mode for Gate/Up

**File**: `src/v2/loaders/WeightManager.cpp`

```cpp
ShardingMode WeightManager::determineShardingMode(const std::string &name) const
{
    if (strategy_ != WeightDistributionStrategy::SHARDED)
    {
        return ShardingMode::REPLICATE;
    }

    // Row-parallel weights (existing)
    if (name.find("attn_output.weight") != std::string::npos ||
        name.find("ffn_down.weight") != std::string::npos)
    {
        return ShardingMode::ROW_PARALLEL;
    }

    // Column-parallel weights (NEW - FFN only for Phase 4b-1)
    if (name.find("ffn_gate.weight") != std::string::npos ||
        name.find("ffn_up.weight") != std::string::npos)
    {
        return ShardingMode::COLUMN_PARALLEL;
    }

    // Everything else replicated
    return ShardingMode::REPLICATE;
}
```

### Step 2: Implement Column Slice Loading

**Option A: Memory-Efficient Column Slice (Preferred)**

Add `loadTensorColumnSlice()` to `ModelLoader` that reads only the needed rows from GGUF.

**File**: `src/v2/loaders/ModelLoader.h`

```cpp
/**
 * @brief Load a column slice of a tensor (rows [row_start, row_end))
 * 
 * For column-parallel sharding where output dimension is split.
 * Weight [out_dim, in_dim] → slice [out_local, in_dim]
 */
std::shared_ptr<TensorBase> loadTensorColumnSlice(
    const std::string& name,
    size_t row_start,
    size_t row_end,
    int device_idx = 0,
    WeightPrecision precision = WeightPrecision::NATIVE);
```

**Implementation**: Similar to `loadTensorRowSlice()`, but slices along first dimension.

**Option B: FP32 Fallback (Current Implementation)**

Already implemented in `getShardedWeight()` - loads full tensor, converts to FP32, slices columns:

```cpp
else if (mode == ShardingMode::COLUMN_PARALLEL)
{
    // Load full tensor as FP32
    auto fp32_tensor = loader_.loadTensor(name, device_idx, WeightPrecision::CONVERT_TO_FP32);
    auto sliced = sliceColumns(fp32_tensor, rank, world_size);
    return sliced;
}
```

**Tradeoff**: Option B works but loses quantization benefits. Option A preserves quantization.

### Step 3: Add Local FFN Dimension

**File**: `src/v2/pipelines/qwen/Qwen2Pipeline.h`

```cpp
class Qwen2Pipeline : public PipelineBase {
protected:
    // Existing
    int d_ff_;                    ///< Full FFN intermediate size
    
    // NEW
    int d_ff_local_;              ///< Local FFN size per rank (d_ff_ / world_size)
    bool ffn_column_parallel_;    ///< Whether FFN uses column-parallel sharding
};
```

**File**: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` (in constructor)

```cpp
// After d_ff_ is set
if (mpi_ctx_ && mpi_ctx_->world_size() > 1 && 
    model_ctx_->strategy() == WeightDistributionStrategy::SHARDED)
{
    int world_size = mpi_ctx_->world_size();
    if (d_ff_ % world_size != 0)
    {
        LOG_ERROR("d_ff (" << d_ff_ << ") not divisible by world_size (" << world_size << ")");
        throw std::runtime_error("FFN dimension not divisible by world_size");
    }
    d_ff_local_ = d_ff_ / world_size;
    ffn_column_parallel_ = true;
    LOG_INFO("FFN column-parallel enabled: d_ff_local=" << d_ff_local_ << " per rank");
}
else
{
    d_ff_local_ = d_ff_;
    ffn_column_parallel_ = false;
}
```

### Step 4: Allocate Local-Sized FFN Buffers

**File**: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` (in `allocate_activation_buffers`)

```cpp
// FFN buffers - use local dimension if column-parallel
int ffn_dim = ffn_column_parallel_ ? d_ff_local_ : d_ff_;

buffers.gate = createActivation(
    {static_cast<size_t>(effective_max), static_cast<size_t>(ffn_dim)});
buffers.up = createActivation(
    {static_cast<size_t>(effective_max), static_cast<size_t>(ffn_dim)});
```

### Step 5: Update FFN Block

**File**: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` (in `ffn_block`)

```cpp
bool Qwen2Pipeline::ffn_block(const LayerWeights &layer, int layer_idx, int effective_seq_len)
{
    // ... existing device setup ...

    // Use local FFN dimension for column-parallel
    int ffn_output_dim = ffn_column_parallel_ ? d_ff_local_ : d_ff_;

    // 2. Fused Gate/Up projections
    if (!layer.gate_up_fused)
    {
        layer.gate_up_fused = std::make_unique<FusedGEMM>(
            layer.gate_proj.get(), layer.up_proj.get());
    }

    VALIDATE_OP(layer.gate_up_fused->execute(
                    buffers.normalized->data(),
                    {{buffers.gate->mutable_data(), nullptr, ffn_output_dim, "gate"},
                     {buffers.up->mutable_data(), nullptr, ffn_output_dim, "up", nullptr, false}},
                    effective_seq_len, d_model_,
                    mpi_ctx_.get(), ffn_device),
                "Fused Gate/Up projection");

    // Capture with local dimensions
    capture_snapshot(layer_prefix + "_FFN_GATE", buffers.gate.get(), effective_seq_len, ffn_output_dim);
    capture_snapshot(layer_prefix + "_FFN_UP", buffers.up.get(), effective_seq_len, ffn_output_dim);

    // 3. Apply SwiGLU (uses local dimension)
    TRY_OP(swiglu(
        buffers.gate.get(), buffers.up.get(), buffers.up.get(),
        effective_seq_len, ffn_output_dim,
        layer_prefix + "_FFN_SWIGLU", ffn_device));

    // 4. Down projection (row-parallel, receives local d_ff input)
    // Input is [seq, d_ff_local], Down weight is [d_model, d_ff_local] (row-sliced)
    // Output is [seq, d_model] partial sum, then allreduced
    TRY_OP(project_row_parallel(
        buffers.up.get(), layer.down_proj.get(), buffers.ffn_output.get(),
        effective_seq_len, d_model_, ffn_output_dim,  // Note: using local FFN dim
        layer_prefix + "_FFN_DOWN", ffn_device));

    // ... rest unchanged ...
}
```

### Step 6: Column-Parallel TensorSlice GEMM (Optional Enhancement)

For preserving quantization, add column-parallel kernel creation:

**File**: `src/v2/tensors/TensorSlice.h`

```cpp
std::unique_ptr<ITensorGemm> TensorSlice::createGemm(...) const override
{
    if (meta_.mode == SliceMode::COLUMN_PARALLEL && !meta_.inner_is_presliced)
    {
        // Create kernel that outputs to local slice of full output
        // Output layout: write to [n_local] portion of [n_full] output
        return std::make_unique<ColumnSlicedGemmKernel>(
            inner_.get(), meta_, mpi_ctx, device_idx);
    }
    else if (meta_.mode == SliceMode::ROW_PARALLEL && !meta_.inner_is_presliced)
    {
        // Existing row-parallel handling
        return std::make_unique<QuantisedGemmKernel>(...);
    }
    // ...
}
```

## Testing Strategy

### Unit Tests

1. **Test__WeightManager_ColumnParallel.cpp**
   - Verify `determineShardingMode()` returns COLUMN_PARALLEL for Gate/Up
   - Test column slice loading produces correct dimensions
   - Verify slice metadata is correct

2. **Test__FFNColumnParallel.cpp**
   - Test FFN block with 2-rank column-parallel
   - Verify output matches single-rank reference
   - Test dimension calculations

### Integration Tests

1. **Test__Qwen2Pipeline_ColumnParallel.cpp**
   - Full pipeline with column-parallel FFN
   - Verify numerical correctness
   - Test with 2 and 4 ranks

### E2E Tests

1. Update existing parity tests to work with column-parallel
2. Compare outputs against PyTorch reference

## Memory Savings Analysis

For Qwen 2.5 0.5B (d_ff = 4864, d_model = 896):

| Weight | Full Size | Per-Rank (2 ranks) | Savings |
|--------|-----------|-------------------|---------|
| Gate | 4864 × 896 | 2432 × 896 | 50% |
| Up | 4864 × 896 | 2432 × 896 | 50% |
| **Total FFN column-parallel** | 8.7 MB | 4.35 MB | **50%** |

Combined with existing row-parallel (Wo, Down):
- **Total savings per layer**: ~35% of weight memory

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FP32 fallback loses quantization benefits | High | Medium | Implement native column slice loading |
| Dimension mismatch in FusedGEMM | Medium | High | Add assertions, unit tests |
| Buffer size mismatch | Medium | High | Centralize dimension calculation |
| Down projection input mismatch | Low | High | Row-parallel already expects sharded input |

## Timeline Estimate

| Task | Effort | Dependencies |
|------|--------|--------------|
| Enable column-parallel mode | 1 hour | None |
| Add d_ff_local tracking | 1 hour | None |
| Update buffer allocation | 1 hour | d_ff_local |
| Update FFN block | 2 hours | Buffer allocation |
| Unit tests | 2 hours | FFN block |
| Integration tests | 2 hours | Unit tests |
| Native column slice loading (optional) | 4 hours | None |
| **Total (with FP32 fallback)** | **9 hours** | |
| **Total (with native loading)** | **13 hours** | |

## Future Work: Phase 4b-2 (Attention Q/K/V)

After FFN column-parallel is stable:

1. **Refactor MpiAttentionOrchestrator**
   - Add `pre_sliced` flag to config
   - Skip internal `copy_head_slice()` when pre-sliced
   - Accept Q/K/V with local head count

2. **Update KV Cache**
   - Store only local KV heads
   - Adjust cache dimensions

3. **Update Qwen2Pipeline**
   - Allocate local-sized Q/K/V buffers
   - Enable column-parallel for attention weights

4. **Testing**
   - Extensive attention parity tests
   - Multi-rank synchronization verification

## Appendix: Key Files

| File | Purpose |
|------|---------|
| `src/v2/loaders/WeightManager.cpp` | Weight sharding logic |
| `src/v2/loaders/ModelLoader.cpp` | GGUF loading, slice loading |
| `src/v2/tensors/TensorSlice.h` | Slice metadata, sliced GEMM |
| `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` | FFN execution |
| `src/v2/pipelines/PipelineBase.cpp` | `project_row_parallel()` |
| `src/v2/pipelines/attention/MpiAttentionOrchestrator.cpp` | Tensor-parallel attention |

## Appendix: Verification Commands

```bash
# Build with sharding enabled
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Run with sharding
./run_llaminar.sh -- --shard-weights -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello" -n 10

# Run unit tests
ctest --test-dir build_v2 -R "ColumnParallel" --output-on-failure

# Run integration tests with MPI
mpirun -np 2 ./build_v2_release/tests/v2/v2_test_column_parallel
```
