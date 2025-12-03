# Weight Sharding Implementation - Phase 1 (Row-Parallel Only)

**Date:** 2025-06-30
**Author:** David Sanftenberg

## Summary

Implemented weight sharding for tensor parallelism with Phase 1 focusing on row-parallel weights only. This provides memory savings without requiring changes to the attention infrastructure.

## Changes

### Core Implementation

**`src/v2/loaders/WeightManager.h`** and **`.cpp`**:
- Added `ShardingMode` enum: `{REPLICATE, COLUMN_PARALLEL, ROW_PARALLEL}`
- Added static methods for tensor slicing (made public for testability):
  - `sliceColumns()`: Extract `[out_local, in_dim]` from `[out_dim, in_dim]`
  - `sliceRows()`: Extract `[out_dim, in_local]` from `[out_dim, in_dim]`
  - `determineShardingMode()`: Pattern match weight names to sharding strategy
- Added instance methods:
  - `isWeightSharded()`: Check if weight uses sharded strategy
  - `getShardingMode()`: Get sharding mode for a weight name
- Added `getShardedWeight()` implementation that:
  1. Loads full tensor
  2. Determines sharding mode from weight name
  3. Slices tensor based on mode and rank
  4. Frees original tensor, returns sliced

**Phase 1 Strategy (Row-Parallel Only)**:
- `attn_output.weight` → ROW_PARALLEL (split input dim)
- `ffn_down.weight` → ROW_PARALLEL (split input dim)
- Everything else → REPLICATE (QKV, Gate/Up, norms, embeddings)

**Rationale for Phase 1 limitations**:
Column-parallel sharding (QKV, Gate/Up) is disabled because:
1. Current attention infrastructure expects full Q/K/V tensors
2. `MpiAttentionOrchestrator::compute_tensor_parallel()` extracts local heads internally
3. Changing to accept pre-sharded Q/K/V would require larger refactor

Row-parallel provides immediate benefits:
- Memory reduction for largest weights (Down projection = d_ff × d_model)
- Allreduce already implemented in `project_row_parallel()`

### CLI Integration

**`src/v2/utils/ArgParser.h`** and **`.cpp`**:
- Added `--shard-weights` / `--shard` flag
- Auto-enables `CONVERT_TO_FP32` weight precision (required for slicing)
- Updated help text to explain Phase 1 limitations

**`src/v2/Main.cpp`**:
- Integrated `WeightDistributionStrategy::SHARDED` with CLI flag
- Passes strategy to `ModelContext::create()`

### Pipeline Integration

**`src/v2/pipelines/PipelineBase.cpp`**:
- `project_row_parallel()` already handles both paths:
  - **Sharded weights**: True row-parallel with local input slice + allreduce
  - **Replicated weights**: Full GEMM with scale + allreduce workaround

### Kernel Dimension Accessors

**`src/v2/tensors/TensorKernels.h`** (ITensorGemm interface):
- Added virtual `get_n()` and `get_k()` methods (return 0 by default)
- Enables callers to query actual weight dimensions for sharded weights

**`src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h`**:
- Implemented `get_n()` and `get_k()` to return `packed_weights_.N` and `packed_weights_.K`

### Tests

**`tests/v2/unit/loaders/Test__WeightManagerSharding.cpp`** (new):
- 13 unit tests covering:
  - Sharding mode determination (row-parallel vs replicate)
  - Column slicing (rank 0/1, remainder handling)
  - Row slicing (rank 0/1, remainder handling)
  - Single-rank edge case

## Usage

```bash
# Enable weight sharding (auto-sets --weight-precision fp32)
./run_llaminar.sh -m model.gguf --shard-weights -p "Hello"

# Explicit with MPI
mpirun -np 2 ./build_v2_release/llaminar2 \
    -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --shard-weights \
    -p "Hello world" -n 10
```

## Memory Savings

With 2 MPI ranks and `--shard-weights`:
- Wo weight: 50% reduction (896×896 → 896×448 per rank)
- Down weight: 50% reduction (4864×896 → 4864×448 per rank)
- Overall: ~25-30% memory reduction for these weights

## Future Work (Phase 2)

To enable column-parallel sharding:
1. Modify `MpiAttentionOrchestrator` to accept pre-sharded Q/K/V
2. Update buffer allocation in Qwen2Pipeline for local dimensions
3. Update FusedGEMM to pass local N dimensions to kernel
4. Enable column-parallel patterns in `determineShardingMode()`

## Files Modified

- `src/v2/loaders/WeightManager.h` - Added sharding API
- `src/v2/loaders/WeightManager.cpp` - Implemented sharding logic
- `src/v2/utils/ArgParser.h` - Added shard_weights flag
- `src/v2/utils/ArgParser.cpp` - Added flag parsing and help
- `src/v2/Main.cpp` - Integrated with ModelContext
- `src/v2/tensors/TensorKernels.h` - Added dimension accessors
- `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h` - Implemented accessors
- `tests/v2/CMakeLists.txt` - Added new test
- `tests/v2/unit/loaders/Test__WeightManagerSharding.cpp` - New test file
