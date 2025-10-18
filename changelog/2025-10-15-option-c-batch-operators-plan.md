# Option C: Batch Operator Migration Plan

**Date**: 2025-10-15  
**Author**: David Sanftenberg  
**Status**: 🚀 Starting Implementation  
**Timeline**: 1.5-2 weeks (Oct 15 - Oct 31, 2025)  
**Approach**: Incremental operator migration with side-by-side validation

## Executive Summary

Implement new batch-aware operator variants alongside existing operators, enabling:
- **Zero-risk migration**: Old operators unchanged, new operators tested in isolation
- **Comprehensive validation**: Direct parity tests prove correctness
- **Incremental progress**: Switch operators one at a time with clear checkpoints
- **Future flexibility**: Eventually use batch operators in both pipelines

This hybrid approach combines the safety of Option B with the architectural cleanliness of Option A.

---

## Architecture Overview

### Current State (Phase 4.1)
```
BatchQwenPipeline uses existing operators with workarounds:
- MPILinearOperator: Flatten [B,T,D] → [B*T,D], reshape back
- MPIAttentionOperator: Empty KV cache (no reuse)
- MPIRMSNormOperator: Works element-wise (inefficient)
- MPISwiGLUOperator: Flatten/reshape approach

Performance: 48.5× prefill, functional decode (no KV cache opt)
```

### Target State (Option C Complete)
```
BatchQwenPipeline uses new batch-native operators:
- MPILinearBatchOperator: Native [B,T,D] processing
- MPIAttentionBatchOperator: Batch KV cache, per-sequence n_past
- MPIRMSNormBatchOperator: Per-sequence statistics
- MPISwiGLUBatchOperator: Native batch activation

QwenPipeline unchanged (uses old operators)

Performance: 48.5× prefill, 2-3× faster decode with KV cache
```

### Migration Path
```
Week 1: Create batch operators + parity tests
Week 2: Integrate into BatchQwenPipeline + optimize
Optional future: Migrate QwenPipeline to batch operators (batch=1)
```

---

## Implementation Plan by Week

### Week 1: Batch Operator Development

#### Day 1-2 (Oct 15-16): MPILinearBatchOperator

**Objective**: Create and validate batch-aware linear projection operator.

**Files**:
- `src/operators/MPILinearBatchOperator.h` (new, ~150 lines)
- `src/operators/MPILinearBatchOperator.cpp` (new, ~250 lines)
- `tests/test_linear_batch_operator.cpp` (new, ~200 lines)

**Interface**:
```cpp
class MPILinearBatchOperator : public MPIKernelBase {
public:
    MPILinearBatchOperator();
    
    bool execute(
        const std::vector<std::shared_ptr<TensorBase>>& inputs,
        std::vector<std::shared_ptr<TensorBase>>& outputs) override;
    
    // Inputs:
    //   [0] = input tensor [batch, seq_len, d_in]
    //   [1] = weight matrix [d_out, d_in]
    //   [2] = bias (optional) [d_out]
    //
    // Outputs:
    //   [0] = output tensor [batch, seq_len, d_out]
    
private:
    bool executeBatch(
        const std::shared_ptr<TensorBase>& input,
        const std::shared_ptr<TensorBase>& weight,
        const std::shared_ptr<TensorBase>& bias,
        std::shared_ptr<TensorBase>& output);
};
```

**Implementation Strategy**:
```cpp
// Approach: Reshape to [B*T, D_in], use existing BLAS, reshape back
// This leverages proven matmul while adding batch-native interface

bool MPILinearBatchOperator::executeBatch(...) {
    int B = input->shape()[0];
    int T = input->shape()[1];
    int D_in = input->shape()[2];
    int D_out = weight->shape()[0];
    
    // Flatten batch dimension
    auto flat_input = reshape(input, {B*T, D_in});
    auto flat_output = create_tensor({B*T, D_out});
    
    // Use BLAS: [B*T, D_in] @ [D_out, D_in]^T = [B*T, D_out]
    cblas_sgemm(...);
    
    // Add bias if present
    if (bias) { add_bias_broadcast(flat_output, bias); }
    
    // Reshape to batch format
    output = reshape(flat_output, {B, T, D_out});
    return true;
}
```

**Tests**:
```cpp
TEST(LinearBatchOperator, ParityWithSingleSequence) {
    // Compare batch=1 against MPILinearOperator
    auto old_op = std::make_unique<MPILinearOperator>();
    auto new_op = std::make_unique<MPILinearBatchOperator>();
    
    auto single_input = create_tensor({seq_len, hidden});
    auto batch_input = create_tensor({1, seq_len, hidden});
    
    auto result_old = old_op->execute(single_input, weight);
    auto result_new = new_op->execute(batch_input, weight);
    
    EXPECT_TENSORS_NEAR(result_old, result_new[0], 1e-6);
}

TEST(LinearBatchOperator, BatchEquivalence) {
    // Verify batch=N equals N× single sequence
    auto new_op = std::make_unique<MPILinearBatchOperator>();
    
    // Run 3 sequences separately
    std::vector<Tensor> individual_results;
    for (int i = 0; i < 3; ++i) {
        auto result = new_op->execute(batch_1_inputs[i], weight);
        individual_results.push_back(result);
    }
    
    // Run as batch
    auto batch_result = new_op->execute(batch_3_input, weight);
    
    for (int i = 0; i < 3; ++i) {
        EXPECT_TENSORS_NEAR(batch_result[i], individual_results[i], 1e-6);
    }
}

TEST(LinearBatchOperator, ShapeValidation) {
    // Test various batch sizes and sequence lengths
    for (int B : {1, 2, 4, 8, 16}) {
        for (int T : {1, 8, 32, 128}) {
            auto input = create_random({B, T, hidden});
            auto result = op->execute(input, weight);
            
            EXPECT_EQ(result->shape(), std::vector<int>{B, T, d_out});
        }
    }
}
```

**Deliverables**:
- ✅ MPILinearBatchOperator implemented
- ✅ Parity test passes (batch=1 == old operator)
- ✅ Equivalence test passes (batch=N == N× single)
- ✅ All shape validation tests pass

**Time**: 2 days (12-16 hours)

---

#### Day 3-4 (Oct 17-18): MPIRMSNormBatchOperator

**Objective**: Batch-aware RMSNorm with per-sequence statistics.

**Files**:
- `src/operators/MPIRMSNormBatchOperator.h` (new, ~100 lines)
- `src/operators/MPIRMSNormBatchOperator.cpp` (new, ~200 lines)
- `tests/test_rmsnorm_batch_operator.cpp` (new, ~150 lines)

**Interface**:
```cpp
class MPIRMSNormBatchOperator : public MPIKernelBase {
public:
    MPIRMSNormBatchOperator();
    
    bool execute(
        const std::vector<std::shared_ptr<TensorBase>>& inputs,
        std::vector<std::shared_ptr<TensorBase>>& outputs) override;
    
    // Inputs:
    //   [0] = input tensor [batch, seq_len, hidden]
    //   [1] = weight [hidden]
    //
    // Outputs:
    //   [0] = normalized tensor [batch, seq_len, hidden]
};
```

**Implementation Strategy**:
```cpp
bool MPIRMSNormBatchOperator::execute(...) {
    int B = input->shape()[0];
    int T = input->shape()[1];
    int D = input->shape()[2];
    
    output = create_tensor({B, T, D});
    
    // Parallelize across batch and sequence
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            // Compute RMS for this position
            float rms = compute_rms(input, b, t, D);
            
            // Normalize and apply weight
            for (int d = 0; d < D; ++d) {
                int idx = (b * T + t) * D + d;
                output->data()[idx] = input->data()[idx] / rms * weight->data()[d];
            }
        }
    }
    return true;
}
```

**Tests**: Similar parity and equivalence pattern as LinearBatch.

**Time**: 2 days (10-12 hours)

---

#### Day 5 (Oct 19): MPISwiGLUBatchOperator

**Objective**: Batch-aware SwiGLU activation (simplest operator).

**Files**:
- `src/operators/MPISwiGLUBatchOperator.h` (new, ~80 lines)
- `src/operators/MPISwiGLUBatchOperator.cpp` (new, ~120 lines)
- `tests/test_swiglu_batch_operator.cpp` (new, ~100 lines)

**Implementation**: Element-wise operation, straightforward batching.

**Time**: 1 day (6-8 hours)

---

### Week 2: Attention & Integration

#### Day 6-7 (Oct 20-21): MPIAttentionBatchOperator

**Objective**: Most complex - batch attention with native KV cache support.

**Files**:
- `src/operators/MPIAttentionBatchOperator.h` (new, ~300 lines)
- `src/operators/MPIAttentionBatchOperator.cpp` (new, ~500 lines)
- `tests/test_attention_batch_operator.cpp` (new, ~300 lines)

**Key Features**:
```cpp
class MPIAttentionBatchOperator : public MPIKernelBase {
public:
    bool execute(
        const std::vector<std::shared_ptr<TensorBase>>& inputs,
        std::vector<std::shared_ptr<TensorBase>>& outputs) override;
    
    // Inputs:
    //   [0] = input [batch, seq_len, hidden]
    //   [1-7] = wq, wk, wv, wo, bq, bk, bv (weights/biases)
    //   [8] = k_cache [batch, n_heads, cache_len, head_dim] (or empty)
    //   [9] = v_cache [batch, n_heads, cache_len, head_dim] (or empty)
    //
    // Outputs:
    //   [0] = attention output [batch, seq_len, hidden]
    //   [1] = updated k_cache [batch, n_heads, new_cache_len, head_dim]
    //   [2] = updated v_cache [batch, n_heads, new_cache_len, head_dim]
    
    // Per-sequence position tracking
    void setSequencePositions(const std::vector<int>& n_past_batch);
    
private:
    std::vector<int> n_past_batch_;  // Position per sequence
};
```

**Implementation Strategy**:
```cpp
// For prefill (empty cache):
// - Process all sequences together (current behavior)
// - Return updated cache for each sequence

// For decode (with cache):
// - Loop over batch dimension
// - For each sequence:
//   - Set n_past from n_past_batch_[b]
//   - Run attention with that sequence's cache
//   - Update that sequence's cache
// - Concatenate results back to batch tensor
```

**Tests**: Critical parity and equivalence tests, plus KV cache correctness.

**Time**: 2 days (14-18 hours)

---

#### Day 8 (Oct 22): Integration into BatchQwenPipeline

**Objective**: Replace operator registration with batch variants.

**Files Modified**:
- `src/BatchQwenPipeline.cpp` (~20 lines changed)
- `src/BatchQwenPipeline.h` (minimal changes)

**Changes**:
```cpp
// Old registration
registerOperator("linear", std::make_unique<MPILinearOperator>());
registerOperator("rmsnorm", std::make_unique<MPIRMSNormOperator>());
registerOperator("attention", std::make_unique<MPIAttentionOperator>());
registerOperator("swiglu", std::make_unique<MPISwiGLUOperator>());

// New registration
registerOperator("linear", std::make_unique<MPILinearBatchOperator>());
registerOperator("rmsnorm", std::make_unique<MPIRMSNormBatchOperator>());
registerOperator("attention", std::make_unique<MPIAttentionBatchOperator>());
registerOperator("swiglu", std::make_unique<MPISwiGLUBatchOperator>());
```

**Testing**: Run all existing BatchQwenPipeline tests - should pass unchanged.

**Time**: 1 day (4-6 hours)

---

#### Day 9 (Oct 23): KV Cache Optimization

**Objective**: Use MPIAttentionBatchOperator's native batch KV cache support.

**Changes in decodeBatch**:
```cpp
// Set sequence positions before running layers
auto* attn_op = dynamic_cast<MPIAttentionBatchOperator*>(
    getOperator("attention"));
attn_op->setSequencePositions(sequence_lengths_);

// In runBatchedLayers (decode mode):
// - Pass actual KV cache from BatchedKVCache to attention operator
// - Receive updated cache in outputs[1], outputs[2]
// - Store back to BatchedKVCache per sequence
```

**Testing**: Add KV cache growth validation test.

**Time**: 1 day (6-8 hours)

---

#### Day 10 (Oct 24): End-to-End Testing & Benchmarking

**Objective**: Validate all improvements and measure performance.

**Tests**:
1. All existing tests still pass
2. Parity tests for all operators pass
3. Batch equivalence tests pass
4. KV cache correctness tests pass
5. Performance benchmarks show expected improvements

**Benchmarks**:
```bash
# Prefill (should maintain 48.5× speedup)
./run_batch_performance.sh --filter '*Prefill*'

# Decode (should show 2-3× improvement over Phase 4.1)
./run_batch_performance.sh --filter '*Decode*'
```

**Time**: 1 day (6-8 hours)

---

## Success Criteria

### Correctness (All Must Pass)
- ✅ All parity tests pass (batch=1 == old operator, tolerance 1e-6)
- ✅ All equivalence tests pass (batch=N == N× single, tolerance 1e-6)
- ✅ All existing BatchQwenPipeline tests pass
- ✅ No regression in single-sequence QwenPipeline
- ✅ KV cache correctness validated

### Performance (Targets)
- ✅ Prefill maintains 48.5× speedup @ batch=32
- ✅ Decode shows 2-3× improvement with KV cache
- ✅ No additional memory overhead vs Phase 4.1
- ✅ Batch operators show ≤5% overhead vs flatten approach (acceptable)

### Code Quality
- ✅ All operators documented (Doxygen comments)
- ✅ Comprehensive test coverage (≥90% for new code)
- ✅ No breaking changes to existing code
- ✅ Clear migration path documented

---

## Risk Assessment & Mitigation

### Risk 1: Performance Regression
**Likelihood**: Low  
**Impact**: Medium  
**Mitigation**: 
- Benchmark each operator individually
- Compare against flatten/reshape baseline
- If regression found, optimize or keep old approach
- Parity tests ensure correctness maintained

### Risk 2: Parity Test Failures
**Likelihood**: Medium (numerical precision differences)  
**Impact**: Medium  
**Mitigation**:
- Use reasonable tolerance (1e-6 for float32)
- Test with various input sizes and values
- Document any known precision differences
- Can adjust tolerance if behavior is mathematically equivalent

### Risk 3: Integration Issues
**Likelihood**: Low  
**Impact**: Medium  
**Mitigation**:
- Integrate operators one at a time
- Test after each integration
- Keep old operators as fallback
- Clear rollback procedure

### Risk 4: Timeline Overrun
**Likelihood**: Low-Medium  
**Impact**: Low  
**Mitigation**:
- Simpler operators (Linear, RMSNorm) can be done quickly
- Most time in Attention operator (expected)
- Can ship partial implementation (some batch operators)
- Option B still available as fallback

---

## File Checklist

### New Operator Files (8 files)
- [ ] `src/operators/MPILinearBatchOperator.h`
- [ ] `src/operators/MPILinearBatchOperator.cpp`
- [ ] `src/operators/MPIRMSNormBatchOperator.h`
- [ ] `src/operators/MPIRMSNormBatchOperator.cpp`
- [ ] `src/operators/MPISwiGLUBatchOperator.h`
- [ ] `src/operators/MPISwiGLUBatchOperator.cpp`
- [ ] `src/operators/MPIAttentionBatchOperator.h`
- [ ] `src/operators/MPIAttentionBatchOperator.cpp`

### New Test Files (4 files)
- [ ] `tests/test_linear_batch_operator.cpp`
- [ ] `tests/test_rmsnorm_batch_operator.cpp`
- [ ] `tests/test_swiglu_batch_operator.cpp`
- [ ] `tests/test_attention_batch_operator.cpp`

### Modified Files (2 files)
- [ ] `src/BatchQwenPipeline.cpp` (~20 lines)
- [ ] `src/BatchQwenPipeline.h` (minimal)

### Documentation (2 files)
- [x] `changelog/2025-10-15-option-c-batch-operators-plan.md` (this file)
- [ ] `changelog/2025-10-31-option-c-batch-operators-complete.md` (after completion)

**Total New Code**: ~2500-3000 lines  
**Modified Code**: ~20 lines  
**Test Code**: ~800-1000 lines

---

## Daily Progress Tracking

### Week 1 Progress
- [x] Day 1 (Oct 15): MPILinearBatchOperator started ✅
  - Created operator header/implementation (~570 lines)
  - Created comprehensive test suite (9 test cases, ~600 lines)
  - Discovered and analyzed gather bug (documented in day1 status)
  - 4/9 tests passing (shape validation, error handling)
  - **Next**: Fix gatherOutput() data layout issue
- [ ] Day 2 (Oct 16): MPILinearBatchOperator complete + tests passing
- [ ] Day 3 (Oct 17): MPIRMSNormBatchOperator started
- [ ] Day 4 (Oct 18): MPIRMSNormBatchOperator complete + tests passing
- [ ] Day 5 (Oct 19): MPISwiGLUBatchOperator complete + tests passing

### Week 2 Progress
- [ ] Day 6 (Oct 20): MPIAttentionBatchOperator started
- [ ] Day 7 (Oct 21): MPIAttentionBatchOperator complete + tests passing
- [ ] Day 8 (Oct 22): BatchQwenPipeline integration complete
- [ ] Day 9 (Oct 23): KV cache optimization complete
- [ ] Day 10 (Oct 24): All tests passing, benchmarks complete

---

## Next Steps

**Immediate** (Today, Oct 15):
1. Create MPILinearBatchOperator.h skeleton
2. Implement basic execute() method
3. Create test file with parity test structure
4. Get first parity test passing

**Tomorrow** (Oct 16):
1. Complete LinearBatch implementation
2. Add equivalence tests
3. Add shape validation tests
4. Verify all tests pass

**This Week**:
- Complete all 4 batch operators
- Comprehensive test coverage
- Document any findings

**Next Week**:
- Integration
- KV cache optimization
- Final validation and benchmarks

---

## References

- Phase 4.1 changelog: `changelog/2025-10-15-batch-qwen-phase4-decode-implementation.md`
- Prefill benchmarks: `changelog/2025-10-15-batch-qwen-phase3-benchmarks.md`
- Option comparison: This discussion thread
- Existing operators: `src/operators/MPI*Operator.{h,cpp}`
- Parity test examples: `tests/test_incremental_decode_parity.cpp`

---

**Status**: 📝 Plan complete, ready to begin implementation  
**Next Action**: Create MPILinearBatchOperator  
**Estimated Completion**: October 31, 2025
