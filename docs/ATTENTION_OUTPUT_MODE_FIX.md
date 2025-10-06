## Attention Output Mode Configuration - Summary & Prevention

### What We Fixed

**Root Cause:** `MPIAttentionKernel` was using the default `LocalHeads` output mode with row-partitioned weight matrices in multi-rank execution, causing it to return only partial results without summing contributions across MPI ranks.

**Impact:** 98.6% parity test failure rate (145/147 checks failing) due to systematic negative bias that cascaded through all downstream layers.

**Fix:** Explicitly set `AttentionOutputMode::GatherHeadsPostProjection` mode when creating `MPIAttentionKernel` instances to enable `MPI_Allreduce(MPI_SUM)` across ranks.

### Why LocalHeads is the Default

`LocalHeads` mode was designed for **future tensor-parallel (TP) head-sharding** where:
- Each MPI rank owns a **subset of attention heads** (head-wise partitioning)
- Weights are pre-sharded along the head dimension
- Each rank returns only its local head slice without gathering
- This avoids unnecessary communication in true TP scenarios

However, our **current implementation uses row-partitioned weights**:
- Each rank owns a **subset of rows** in the output projection matrix W_o
- All ranks compute partial contributions to ALL output dimensions
- Results MUST be summed via `MPI_Allreduce` to get the correct output
- `LocalHeads` mode silently returns incomplete results in this configuration

### Files Modified

1. **`src/qwen_pipeline.cpp`** (lines ~589-615)
   - Added comprehensive explanatory comment
   - Set `GatherHeadsPostProjection` mode before kernel registration
   
2. **`src/openblas_prefill_provider.cpp`** (lines ~60-85)
   - Added comprehensive explanatory comment
   - Set `GatherHeadsPostProjection` mode before kernel registration

3. **`src/kernels/MPIAttentionKernel.cpp`** (lines ~168-189)
   - Added runtime assertion to detect invalid configuration
   - Throws clear error if `LocalHeads` mode used with non-sharded weights on multiple ranks
   
4. **`src/kernels/MPIAttentionKernel.h`** (line 287)
   - Updated default mode comment to warn about TP-specific design

5. **`tests/test_attention_output_mode_validation.cpp`** (new file)
   - Regression test that validates the assertion catches the bug
   - Tests configuration validation (not full execution)

6. **`CMakeLists.txt`** (lines ~512-535)
   - Added new test target with 2-rank MPI execution

### Prevention Mechanisms

#### 1. Runtime Assertion (Primary Defense)
```cpp
// In MPIAttentionKernel::execute()
if (!pre_sharded && getSize() > 1 && output_mode_ == AttentionOutputMode::LocalHeads)
{
    throw std::runtime_error(
        "MPIAttentionKernel: LocalHeads mode requires head-sharded weights. "
        "For replicated/row-partitioned weights with multiple ranks, use GatherHeadsPostProjection mode.");
}
```

**Triggers when:**
- Weights are NOT head-sharded (detected via `ShardedSimpleTensor` check)
- Multiple MPI ranks present (`getSize() > 1`)
- Output mode is `LocalHeads` (the dangerous default)

**Result:** Immediate failure with actionable error message instead of silent incorrectness.

#### 2. Unit Test (Regression Protection)
```bash
# Run the validation test
ctest --test-dir build -R AttentionOutputModeValidation
```

**Test Coverage:**
- ✅ Verifies `LocalHeads` mode throws error with non-sharded weights on 2 ranks
- ✅ Verifies `GatherHeadsPostProjection` mode can be set correctly
- ✅ Verifies single-rank execution doesn't trigger false positive

**Integration:** Runs automatically in CI as part of test suite.

#### 3. Comprehensive Comments
Added detailed comments at both kernel creation sites explaining:
- Why the mode must be set explicitly
- What happens without the fix (partial results, bias, cascading errors)
- Link to investigation documentation
- Reference to parity test failure rates

### How to Create MPIAttentionKernel Correctly

```cpp
// ✅ CORRECT - Explicitly set output mode for row-partitioned weights
auto attention_kernel = std::make_unique<MPIAttentionKernel>(
    n_head, n_head_kv, head_dim, rope_freq_base);

// CRITICAL: Set output mode to gather heads post-projection
// This ensures all MPI ranks' head contributions are summed together
attention_kernel->setOutputMode(
    MPIAttentionKernel::AttentionOutputMode::GatherHeadsPostProjection);

registerKernel("attention", std::move(attention_kernel));
```

```cpp
// ❌ WRONG - Using default LocalHeads mode with non-sharded weights
auto attention_kernel = std::make_unique<MPIAttentionKernel>(
    n_head, n_head_kv, head_dim, rope_freq_base);

// Missing setOutputMode() call!
// Will throw runtime error on multi-rank execution

registerKernel("attention", std::move(attention_kernel));
```

### Output Mode Reference

| Mode | Use Case | Requirements | Behavior |
|------|----------|--------------|----------|
| `LocalHeads` (default) | Future head-sharded TP | Weights pre-sharded by heads | Returns local head slice only, no gather |
| `GatherHeadsPostProjection` | Current row-partitioned W_o | Replicated/row-partitioned weights | MPI_Allreduce to sum all ranks' contributions |
| `GatherHeadsPreProjection` | Future alternative TP | (Not yet implemented) | Gather before output projection |
| `Replicated` | Debug/legacy | Any | Force fully replicated output |

### Testing Checklist

When modifying attention code:

- [ ] If creating a new `MPIAttentionKernel`, set `GatherHeadsPostProjection` mode
- [ ] Run `AttentionOutputModeValidation` test to verify no regression
- [ ] Run parity tests to validate numerical correctness
- [ ] Check that multi-rank execution doesn't throw configuration error
- [ ] Verify output is identical across all ranks after reduction

### Related Documentation

- **Investigation Report:** `docs/OPENBLAS_PREFILL_ROOT_CAUSE_ANALYSIS.md`
- **Parity Framework:** `PARITY_FRAMEWORK_SUMMARY.md`
- **Test Suite:** `tests/test_attention_output_mode_validation.cpp`
- **Kernel Implementation:** `src/kernels/MPIAttentionKernel.{h,cpp}`

### Future Work

**Consider changing the default mode to `GatherHeadsPostProjection`** since:
- It's the only mode that works with current weight distribution
- Head-sharded TP is not yet implemented
- Would prevent this bug from recurring
- Could auto-detect based on weight shard spec and adjust

**Alternative:** Auto-detect mode based on weight tensor metadata:
```cpp
if (is_head_sharded(wo)) {
    output_mode_ = AttentionOutputMode::LocalHeads;
} else if (getSize() > 1) {
    output_mode_ = AttentionOutputMode::GatherHeadsPostProjection;
}
```

This would eliminate the need for manual configuration entirely.
