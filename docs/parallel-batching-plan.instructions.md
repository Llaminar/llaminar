# Parallel Batching Implementation Plan

## Overview

Implement true parallel batching to process multiple independent sequences simultaneously, leveraging the system's 281 GB/s memory bandwidth (currently utilizing <1%).

**Goal**: Achieve 20-30× aggregate throughput improvement by processing batch_size=32 sequences in parallel.

## System Capabilities

- **Hardware**: Dual Xeon 6238R, 768GB DDR4-2933, 6 channels/socket
- **Theoretical Bandwidth**: 281.52 GB/s total (140.76 GB/s per socket)
- **Current Utilization**: ~1.5 GB/s (0.6%)
- **Current Performance**: 13.25 tok/s single sequence
- **Target Performance**: 288-320 tok/s aggregate (batch=32)

## Implementation Phases

### Phase 1: Foundation (Week 1) - Tensor & Interface Updates

**Objective**: Add batch dimension support to tensor system and pipeline interfaces.

#### 1.1 SimpleTensor Batch Support
**Files**: `src/tensors/SimpleTensor.h`, `src/tensors/SimpleTensor.cpp`

- [ ] Add batch dimension accessors:
  ```cpp
  size_t batch_size() const;
  size_t seq_len() const;
  std::shared_ptr<SimpleTensor> reshape(const std::vector<size_t>& new_shape);
  ```
- [ ] Implement batch slicing:
  ```cpp
  std::shared_ptr<SimpleTensor> slice_batch(size_t batch_idx) const;
  ```
- [ ] Implement batch stacking:
  ```cpp
  static std::shared_ptr<SimpleTensor> stack_batch(
      const std::vector<std::shared_ptr<SimpleTensor>>& sequences);
  ```
- [ ] Update NUMA first-touch for 3D+ tensors

#### 1.2 AbstractPipeline Interface
**Files**: `src/AbstractPipeline.h`

- [ ] Add batch-aware methods:
  ```cpp
  virtual bool prefillBatch(
      const std::vector<std::vector<int>>& token_batches,
      std::shared_ptr<TensorBase>& out_logits) = 0;
  
  virtual bool decodeBatch(
      const std::vector<int>& next_tokens,
      std::shared_ptr<TensorBase>& out_logits) = 0;
  
  virtual void resetBatch(size_t batch_size) = 0;
  virtual size_t currentBatchSize() const = 0;
  ```
- [ ] Keep single-sequence methods for backward compatibility

#### 1.3 Batch Padding Utilities
**Files**: `src/BatchPaddingUtils.h`, `src/BatchPaddingUtils.cpp` (new)

- [ ] Create `PaddedBatch` struct:
  ```cpp
  struct PaddedBatch {
      std::shared_ptr<TensorBase> tokens;
      std::vector<int> actual_lengths;
      std::vector<int> padding_mask;
      size_t max_length;
  };
  ```
- [ ] Implement padding functions:
  ```cpp
  PaddedBatch createPaddedBatch(
      const std::vector<std::vector<int>>& sequences,
      int pad_token_id = 0);
  ```

**Tests**: 
- [ ] `tests/test_tensor_batch_ops.cpp` - Verify reshape, slice, stack
- [ ] `tests/test_batch_padding.cpp` - Verify padding correctness

---

### Phase 2: Kernel Updates (Week 2) - Core Operations

**Objective**: Update all kernels to support batch dimension processing.

#### 2.1 Embedding Kernel
**Files**: `src/kernels/EmbeddingKernel.h`, `src/kernels/EmbeddingKernel.cpp`

- [ ] Add batch embedding method:
  ```cpp
  bool executeBatch(
      const std::shared_ptr<TensorBase>& token_ids,  // [batch, seq_len]
      const float* embedding_weights,
      std::shared_ptr<TensorBase>& output);          // [batch, seq_len, d_model]
  ```
- [ ] Parallelize across batch with OpenMP:
  ```cpp
  #pragma omp parallel for
  for (size_t b = 0; b < batch_size; ++b) {
      // Embed sequence b
  }
  ```
- [ ] Handle padding tokens (skip embedding zeros)

#### 2.2 Linear Operator (Minimal Changes)
**Files**: `src/operators/MPILinearOperator.h`, `src/operators/MPILinearOperator.cpp`

- [ ] Add reshape-based batch execution:
  ```cpp
  // [batch, seq_len, d_in] -> [batch*seq_len, d_in]
  // Matmul: [batch*seq_len, d_in] × [d_out, d_in]^T
  // Output: [batch*seq_len, d_out] -> [batch, seq_len, d_out]
  ```
- [ ] Verify weight distribution works with larger batch*seq_len
- [ ] Test with adaptive backend selection

#### 2.3 RMSNorm Operator
**Files**: `src/operators/MPIRMSNormOperator.h`, `src/operators/MPIRMSNormOperator.cpp`

- [ ] Add batch RMSNorm:
  ```cpp
  #pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; ++b) {
      for (size_t t = 0; t < seq_len; ++t) {
          // Apply RMSNorm independently
      }
  }
  ```
- [ ] Keep existing single-sequence path for compatibility

#### 2.4 Attention Operator (Most Complex)
**Files**: `src/operators/MPIAttentionOperator.h`, `src/operators/MPIAttentionOperator.cpp`

- [ ] Add batch attention execution:
  ```cpp
  bool executeBatch(
      const std::shared_ptr<TensorBase>& hidden,          // [batch, seq_len, d_model]
      const std::shared_ptr<TensorBase>& k_cache_batch,   // [batch, n_heads, cache_len, head_dim]
      const std::shared_ptr<TensorBase>& v_cache_batch,
      const std::vector<int>& positions_batch,
      const std::vector<int>& seq_lengths_batch,
      std::shared_ptr<TensorBase>& output);
  ```
- [ ] Implement per-sequence attention processing:
  ```cpp
  #pragma omp parallel for
  for (size_t b = 0; b < batch_size; ++b) {
      // Process attention for sequence b
  }
  ```
- [ ] Add padding mask support:
  ```cpp
  void applyPaddingMask(scores, actual_seq_len);
  ```
- [ ] Update KV cache management for batch dimension

**Tests**:
- [ ] `tests/test_batch_embedding.cpp` - Verify batch embedding correctness
- [ ] `tests/test_batch_linear.cpp` - Verify reshape strategy works
- [ ] `tests/test_batch_attention.cpp` - Verify attention with padding masks
- [ ] `tests/test_batch_rmsnorm.cpp` - Verify batch normalization

---

### Phase 3: Pipeline Integration (Week 3) - QwenPipeline

**Objective**: Implement batch-aware pipeline execution.

#### 3.1 QwenPipeline State Management
**Files**: `src/QwenPipeline.h`, `src/QwenPipeline.cpp`

- [ ] Add batch state members:
  ```cpp
  size_t batch_size_;
  std::vector<int> n_past_batch_;  // [batch_size]
  std::vector<std::shared_ptr<TensorBase>> k_cache_batch_;  // Per layer
  std::vector<std::shared_ptr<TensorBase>> v_cache_batch_;
  ```
- [ ] Implement `resetBatch()`:
  ```cpp
  void resetBatch(size_t batch_size) {
      batch_size_ = batch_size;
      n_past_batch_.assign(batch_size, 0);
      allocateKVCacheBatch();
  }
  ```

#### 3.2 Batch Prefill Implementation
**Files**: `src/QwenPipeline.cpp`

- [ ] Implement `prefillBatch()`:
  1. Pad token sequences to max length
  2. Embed batch: [batch, max_len, d_model]
  3. Process through transformer layers
  4. Apply final layer norm
  5. Project to logits: [batch, max_len, vocab_size]
  6. Update n_past_batch_ positions

#### 3.3 Batch Decode Implementation
**Files**: `src/QwenPipeline.cpp`

- [ ] Implement `decodeBatch()`:
  1. Embed next tokens: [batch, 1, d_model]
  2. Process through transformer layers (seq_len=1)
  3. Project to logits: [batch, vocab_size]
  4. Increment n_past_batch_

#### 3.4 Transformer Layer Batch Processing
**Files**: `src/QwenPipeline.cpp`

- [ ] Update `executeTransformerLayer()` for batching:
  ```cpp
  auto executeTransformerLayerBatch(
      size_t layer,
      std::shared_ptr<TensorBase> hidden,  // [batch, seq_len, d_model]
      size_t actual_seq_len) -> std::shared_ptr<TensorBase>;
  ```
- [ ] Update FFN processing for batch dimension
- [ ] Maintain residual connections across batch

**Tests**:
- [ ] `tests/test_qwen_batch_prefill.cpp` - Verify batch prefill correctness
- [ ] `tests/test_qwen_batch_decode.cpp` - Verify batch decode correctness
- [ ] `tests/test_batch_state_isolation.cpp` - Verify sequences don't interfere

---

### Phase 4: Benchmarking & Validation (Week 4)

**Objective**: Validate correctness and measure performance.

#### 4.1 Batch Benchmark Runner
**Files**: `src/BenchmarkRunner.h`, `src/BenchmarkRunner.cpp`

- [ ] Create `BatchBenchmarkMetrics` struct:
  ```cpp
  struct BatchBenchmarkMetrics {
      size_t batch_size;
      size_t total_prefill_tokens;
      size_t total_decode_tokens;
      double prefill_time_ms;
      double decode_time_ms;
      double prefill_throughput;    // tokens/s
      double decode_throughput;
      double batching_efficiency;   // vs batch_size × single_throughput
      double estimated_bandwidth;   // GB/s
  };
  ```
- [ ] Implement `runBatchBenchmark()`:
  ```cpp
  BatchBenchmarkMetrics runBatchBenchmark(
      AbstractPipeline& pipeline,
      const QwenModelWeights& weights,
      chat::TokenizerInterface& tokenizer,
      const std::vector<std::string>& prompts,
      int n_predict);
  ```
- [ ] Add command-line integration:
  ```bash
  ./run_llaminar.sh -- --benchmark --batch-size 32 \
    -p "prompt1" -p "prompt2" ... -n 50
  ```

#### 4.2 Correctness Tests
**Files**: `tests/test_batch_correctness.cpp`

- [ ] Test: batch=2 same prompts vs 2× batch=1
  ```cpp
  TEST_CASE("Batch equivalence") {
      auto result_batch = run_batch({prompt, prompt});
      auto result1 = run_single(prompt);
      auto result2 = run_single(prompt);
      REQUIRE(allclose(result_batch[0], result1));
      REQUIRE(allclose(result_batch[1], result2));
  }
  ```
- [ ] Test: Different length sequences with padding
- [ ] Test: Padding doesn't affect non-padded positions
- [ ] Test: State isolation between sequences in batch

#### 4.3 Performance Benchmarks
**Files**: `scripts/run_batch_benchmarks.sh` (new)

- [ ] Benchmark suite:
  ```bash
  # Batch size sweep
  for BS in 1 2 4 8 16 32 64; do
      ./run_llaminar.sh -- --benchmark --batch-size $BS -p "test" -n 50
  done
  ```
- [ ] Measure:
  - Aggregate throughput (tokens/s)
  - Per-sequence latency
  - Memory bandwidth utilization
  - Batching efficiency
  - Memory usage

**Deliverables**:
- [ ] Correctness validation report
- [ ] Performance benchmark results
- [ ] Comparison: batch vs single-sequence

---

### Phase 5: Optimization (Week 5)

**Objective**: Optimize for production performance.

#### 5.1 Bucketing Strategy
**Files**: `src/BatchBucketing.h`, `src/BatchBucketing.cpp` (new)

- [ ] Implement sequence length bucketing:
  ```cpp
  std::vector<PaddedBatch> bucketSequencesByLength(
      const std::vector<std::string>& prompts,
      size_t batch_size,
      const std::vector<size_t>& bucket_boundaries = {8, 16, 32, 64, 128, 256, 512});
  ```
- [ ] Minimize padding waste by grouping similar lengths

#### 5.2 Kernel Fusion Opportunities
**Files**: Various operator files

- [ ] Fuse RMSNorm + Linear where possible
- [ ] Fuse QKV projection (already done for single-sequence)
- [ ] Investigate attention score computation fusion

#### 5.3 Memory Layout Optimization
**Files**: `src/tensors/SimpleTensor.cpp`

- [ ] Optimize NUMA placement for batch tensors
- [ ] Consider AoS vs SoA layouts for batch dimension
- [ ] Profile cache utilization

#### 5.4 Dynamic Batching
**Files**: `src/DynamicBatcher.h`, `src/DynamicBatcher.cpp` (new, stretch goal)

- [ ] Implement continuous batching like vLLM:
  - Accept new sequences as slots free up
  - Remove finished sequences from batch
  - Maintain high GPU/CPU utilization

**Deliverables**:
- [ ] Optimized performance metrics
- [ ] Memory usage analysis
- [ ] Profiling reports

---

## Testing Strategy

### Unit Tests
- Tensor operations (reshape, slice, stack)
- Padding utilities
- Individual kernel batch operations
- State isolation

### Integration Tests
- Full pipeline batch prefill
- Full pipeline batch decode
- Multi-layer processing
- KV cache correctness

### Correctness Tests
- Batch equivalence (batch=N vs N× single)
- Padding doesn't affect results
- Different sequence lengths
- Edge cases (batch=1, max batch size)

### Performance Tests
- Throughput scaling (batch 1→2→4→8→16→32)
- Memory bandwidth utilization
- Latency vs throughput trade-off
- Batching efficiency

---

## Success Criteria

### Correctness
- ✅ Batch=2 produces identical results to 2× batch=1 (tolerance: 1e-5)
- ✅ Padding masks correctly exclude padding tokens
- ✅ State properly isolated between sequences
- ✅ All unit tests pass

### Performance (at batch_size=32)
- ✅ Aggregate throughput: ≥250 tok/s (19× improvement over 13 tok/s)
- ✅ Memory bandwidth: ≥30 GB/s (from current 1.5 GB/s)
- ✅ Batching efficiency: ≥70% (actual vs ideal scaling)
- ✅ Memory usage: <100 GB total (plenty of headroom with 768GB)

### Code Quality
- ✅ Backward compatible (single-sequence still works)
- ✅ Well documented (architecture docs, code comments)
- ✅ Comprehensive tests (>80% coverage for batch code)
- ✅ No performance regression for single-sequence path

---

## Risk Mitigation

### Risk: State Corruption Between Sequences
**Mitigation**: 
- Thorough testing of state isolation
- Clear separation of batch state (n_past_batch_[b])
- Comprehensive correctness tests

### Risk: Memory Overflow with Large Batches
**Mitigation**:
- KV cache size validation before allocation
- Configurable max batch size
- Graceful degradation if memory insufficient

### Risk: Poor Scaling Due to Padding Overhead
**Mitigation**:
- Implement bucketing to group similar lengths
- Measure and report padding efficiency
- Consider variable-length attention kernels (stretch)

### Risk: Performance Regression for Single Sequence
**Mitigation**:
- Keep separate code paths (don't force batch=1 through batch code)
- Benchmark single-sequence before/after
- Profile to identify any introduced overhead

---

## File Checklist

### New Files
- [ ] `src/BatchPaddingUtils.{h,cpp}`
- [ ] `src/BatchBucketing.{h,cpp}` (Phase 5)
- [ ] `src/DynamicBatcher.{h,cpp}` (Phase 5, stretch)
- [ ] `tests/test_tensor_batch_ops.cpp`
- [ ] `tests/test_batch_padding.cpp`
- [ ] `tests/test_batch_embedding.cpp`
- [ ] `tests/test_batch_linear.cpp`
- [ ] `tests/test_batch_attention.cpp`
- [ ] `tests/test_batch_rmsnorm.cpp`
- [ ] `tests/test_qwen_batch_prefill.cpp`
- [ ] `tests/test_qwen_batch_decode.cpp`
- [ ] `tests/test_batch_state_isolation.cpp`
- [ ] `tests/test_batch_correctness.cpp`
- [ ] `scripts/run_batch_benchmarks.sh`

### Modified Files
- [ ] `src/tensors/SimpleTensor.{h,cpp}`
- [ ] `src/AbstractPipeline.h`
- [ ] `src/QwenPipeline.{h,cpp}`
- [ ] `src/QwenPipelineAdapter.{h,cpp}`
- [ ] `src/kernels/EmbeddingKernel.{h,cpp}`
- [ ] `src/operators/MPILinearOperator.{h,cpp}`
- [ ] `src/operators/MPIRMSNormOperator.{h,cpp}`
- [ ] `src/operators/MPIAttentionOperator.{h,cpp}`
- [ ] `src/BenchmarkRunner.{h,cpp}`
- [ ] `src/ArgumentParser.{h,cpp}` (already has --batch-size)
- [ ] `src/Main.cpp` (integrate batch benchmark)
- [ ] `CMakeLists.txt` (add new source files)

---

## Documentation Deliverables

- [x] Architecture design document (`changelog/20251015_parallel_batching_architecture.md`)
- [x] Quick reference guide (`changelog/20251015_batching_quick_reference.md`)
- [x] Implementation example (`changelog/20251015_batching_implementation_example.md`)
- [ ] API documentation (Doxygen comments in headers)
- [ ] User guide (how to use batch mode)
- [ ] Performance report (benchmark results)
- [ ] Final changelog entry

---

## Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Foundation | Tensor batch support, pipeline interface, padding utils |
| 2 | Kernels | Embedding, linear, RMSNorm, attention batch operations |
| 3 | Pipeline | QwenPipeline batch prefill/decode, state management |
| 4 | Validation | Correctness tests, performance benchmarks, metrics |
| 5 | Optimization | Bucketing, fusion, memory layout, dynamic batching |

**Total**: 5 weeks to production-ready parallel batching

---

## References

- **Architecture docs**: `changelog/20251015_parallel_batching_architecture.md`
- **Quick reference**: `changelog/20251015_batching_quick_reference.md`
- **Code example**: `changelog/20251015_batching_implementation_example.md`
- **Hardware analysis**: Dual Xeon 6238R, 281 GB/s theoretical bandwidth
- **Similar systems**: vLLM, TensorRT-LLM, Text-Generation-Inference
