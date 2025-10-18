# Batch Parallelization Investigation Plan

**Date**: January 15, 2025  
**Author**: David Sanftenberg  
**Priority**: 🔴 **CRITICAL** - Blocks Phase 4.4  
**Issue**: Zero throughput scaling with batch size (0× instead of 22× target speedup)

## Problem Statement

Performance benchmarks reveal that batching provides **no performance benefit**:

```
Batch Size | Wall Time | Throughput | Expected | Actual Speedup
-----------+-----------+------------+----------+---------------
    1      |  826 ms   | 9.68 tok/s | baseline | 1.0×
    4      | 3355 ms   | 9.54 tok/s | 4.0×     | 0.98× ❌
   32      |  ~27 sec  | ~9.5 tok/s | 22.0×    | 0.0× ❌
```

**Expected Behavior**: Throughput should increase with batch size (more parallel work)  
**Actual Behavior**: Time scales linearly → sequences processed sequentially

## Investigation Steps

### Step 1: Verify Batch Processing Path (30 minutes)

**Objective**: Confirm whether sequences are processed sequentially or in parallel

**Actions**:
1. Add timing instrumentation to `runBatchPrefill()` in test_batch_performance.cpp
2. Log per-sequence timing to detect sequential iteration
3. Check if `pipeline.prefill()` is called once (batch) or multiple times (sequential)

**Expected Code Pattern (Sequential - BAD)**:
```cpp
for (size_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    auto start = steady_clock::now();
    bool success = pipeline.prefill(batch_inputs[seq_idx], ...);
    auto end = steady_clock::now();
    LOG_DEBUG("Sequence " << seq_idx << ": " << duration(end - start));
}
```

**Desired Code Pattern (Parallel - GOOD)**:
```cpp
// Single call processes entire batch in parallel
auto start = steady_clock::now();
bool success = pipeline.prefill_batch(all_batch_inputs, ...);
auto end = steady_clock::now();
LOG_DEBUG("Entire batch (" << batch_size << "): " << duration(end - start));
```

**Test**:
```bash
# Add instrumentation to test
git diff tests/test_batch_performance.cpp

# Run with batch=4
./run_batch_performance.sh --filter '*Prefill*'

# Analyze per-sequence timing
grep "Sequence" /tmp/batch_perf_release.log
```

**Success Criteria**:
- If logs show individual sequence timings → **Sequential processing confirmed**
- If single batch timing → Need deeper investigation at operator level

---

### Step 2: Examine QwenPipeline Batch Implementation (1 hour)

**Objective**: Understand how batch dimension flows through pipeline

**Actions**:
1. Search for batch handling in QwenPipeline::prefill()
2. Check if operators receive batch dimension or process sequences one-by-one
3. Verify attention mechanism handles batch correctly

**Investigation**:
```bash
# Search for batch-related code in pipeline
cd /workspaces/llaminar
grep -n "batch" src/qwen_pipeline.cpp | head -30

# Check for loops over batch dimension
grep -A 5 "for.*batch" src/qwen_pipeline.cpp

# Examine prefill method signature
grep -A 10 "prefill(" src/qwen_pipeline.h
```

**Key Questions**:
1. Does `prefill()` accept batch dimension in input tensor?
2. Are operators called once per batch or once per sequence?
3. Is attention score computation batched?

**Expected Findings**:

**If Sequential**:
```cpp
// Suspected current implementation
bool QwenPipeline::prefill(const std::vector<int>& tokens, ...) {
    // tokens.size() = sequence_length (no batch dimension)
    for (int layer = 0; layer < num_layers; ++layer) {
        // Process single sequence through each layer
    }
}
```

**If Parallel** (desired):
```cpp
// Desired implementation
bool QwenPipeline::prefill(const Tensor& batch_tokens, ...) {
    // batch_tokens.shape() = [batch_size, seq_len]
    for (int layer = 0; layer < num_layers; ++layer) {
        // Process all sequences in parallel through each layer
    }
}
```

---

### Step 3: Operator-Level Batch Support Analysis (2 hours)

**Objective**: Verify if individual operators (attention, linear, etc.) support batching

**Actions**:
1. Check attention operator for batch dimension handling
2. Verify linear projection processes batch in parallel
3. Examine RMSNorm and other operators

**Attention Operator Investigation**:
```bash
# Check MPIAttentionKernel batch handling
grep -A 20 "class MPIAttentionKernel" src/kernels/MPIAttentionKernel.h

# Look for batch dimension in attention computation
grep -n "batch" src/kernels/MPIAttentionKernel.cpp | head -20

# Check attention score computation (Q @ K^T)
grep -A 10 "score" src/kernels/MPIAttentionKernel.cpp
```

**Linear Projection Investigation**:
```bash
# Check MPILinearKernel batch support
grep -A 20 "execute" src/kernels/MPILinearKernel.cpp

# Verify matmul dimensions
grep -n "cblas_sgemm\|cosma::multiply" src/kernels/MPILinearKernel.cpp
```

**Key Questions**:
1. Do operators accept input shape `[batch, seq, hidden]` or just `[seq, hidden]`?
2. Is batch dimension explicitly handled or implicit in outer loop?
3. Are matmul dimensions correct for batching? `(batch*seq, hidden) @ (hidden, out)`?

---

### Step 4: Memory Layout Analysis (1 hour)

**Objective**: Verify tensors are shaped for batch processing

**Actions**:
1. Add logging to print tensor shapes at each stage
2. Verify batch dimension flows through pipeline
3. Check for shape mismatches or reshaping

**Instrumentation**:
```cpp
// Add to QwenPipeline::prefill() at key points
LOG_DEBUG("Input shape: [" << input.shape(0) << ", " << input.shape(1) << "]");
LOG_DEBUG("After embedding: [" << embedded.shape(0) << ", " << embedded.shape(1) << "]");
LOG_DEBUG("After attention: [" << attn_out.shape(0) << ", " << attn_out.shape(1) << "]");
```

**Expected Shapes**:
```
Input tokens:        [batch_size, seq_len]
After embedding:     [batch_size, seq_len, hidden_size]
After attention:     [batch_size, seq_len, hidden_size]
After linear:        [batch_size, seq_len, hidden_size]
Final logits:        [batch_size, seq_len, vocab_size]
```

**Red Flags**:
- Missing batch dimension at any stage
- Shape reshaping from `[batch, seq, dim]` to `[seq, dim]`
- Loops iterating over batch dimension

---

### Step 5: Profiling and Bottleneck Identification (2 hours)

**Objective**: Identify where time is being spent and if it's parallelizable

**Actions**:
1. Use `perf` to profile batch=1 vs batch=4 execution
2. Compare operator-level timing breakdown
3. Identify serialization points (locks, barriers, sequential sections)

**Profiling**:
```bash
# Profile batch=1
perf record -g --call-graph dwarf \
  ./build/test_batch_performance --gtest_filter='*Prefill*' \
  --gtest_filter_batch_size=1

# Profile batch=4
perf record -g --call-graph dwarf \
  ./build/test_batch_performance --gtest_filter='*Prefill*' \
  --gtest_filter_batch_size=4

# Compare profiles
perf report --stdio > perf_batch1.txt
perf report --stdio > perf_batch4.txt
diff perf_batch1.txt perf_batch4.txt
```

**Analysis Questions**:
1. Does batch=4 show 4× more time in same functions? (sequential)
2. Or same time with higher parallelism? (parallel)
3. Are there unexpected serialization points? (locks, MPI_Barrier)
4. Is memory allocation a bottleneck? (malloc, new)

---

### Step 6: KV Cache Capacity Check (1 hour)

**Objective**: Verify cache warnings are not limiting batch size

**Context**: Logs show cache recreation warnings:
```
[CACHE_INIT_DEBUG] Layer 0: Recreated cache tensors! This will WIPE existing cache data!
```

**Actions**:
1. Check KV cache capacity configuration
2. Verify cache is sized for max batch size
3. Examine cache allocation patterns

**Investigation**:
```bash
# Find cache capacity logic
grep -n "cache_capacity\|max_batch" src/qwen_pipeline.cpp

# Check cache allocation
grep -A 10 "k_cache\|v_cache" src/qwen_pipeline.cpp | head -50

# Look for cache recreation triggers
grep -B 5 -A 5 "Recreated cache tensors" src/qwen_pipeline.cpp
```

**Expected Configuration**:
```cpp
// Cache should be sized for max_batch_size
size_t cache_capacity = max_batch_size * max_seq_len * num_heads * head_dim;
k_cache.resize({num_layers, cache_capacity});
v_cache.resize({num_layers, cache_capacity});
```

**Red Flags**:
- Cache sized for batch=1 only
- Recreation on every batch size change
- Running out of capacity

---

## Root Cause Scenarios

### Scenario A: Sequential Iteration (Most Likely)

**Evidence**:
- Time scales linearly (826ms → 3355ms for 1 → 4 sequences)
- Throughput constant (~9.5 tok/s)
- No parallelization benefit

**Diagnosis**:
```cpp
// Test code iterates sequentially through batch
for (size_t i = 0; i < batch_size; ++i) {
    pipeline.prefill(sequences[i]);  // Called batch_size times
}
```

**Fix**: Implement true batch API
```cpp
// Single call with all sequences
pipeline.prefill_batch(all_sequences);  // Called once
```

**Estimated Effort**: 2-3 days
- Modify pipeline API to accept batch dimension
- Update operators to process batch in parallel
- Verify correctness with existing tests

---

### Scenario B: Batch Dimension Not Propagated

**Evidence**:
- Batch API exists but dimension dropped internally
- Operators process sequences one-by-one

**Diagnosis**:
```cpp
// Pipeline accepts batch but iterates internally
bool prefill(const Tensor& batch_input) {
    for (size_t seq = 0; seq < batch_input.shape(0); ++seq) {
        Tensor single_seq = batch_input.slice(seq);
        process_single_sequence(single_seq);  // Sequential!
    }
}
```

**Fix**: Propagate batch dimension through all operators
```cpp
bool prefill(const Tensor& batch_input) {
    // batch_input.shape() = [batch, seq, hidden]
    // Pass entire batch to operators
    for (int layer = 0; layer < num_layers; ++layer) {
        attention(batch_input);  // Processes all sequences in parallel
        ffn(batch_input);
    }
}
```

**Estimated Effort**: 3-5 days
- Update all operators to handle batch dimension
- Verify tensor shapes at each stage
- Test correctness and performance

---

### Scenario C: Operator-Level Bottleneck

**Evidence**:
- Batch dimension propagated correctly
- But operators have sequential sections (locks, barriers, serial math)

**Diagnosis**:
```cpp
// Operator has serialization point
bool MPIAttentionKernel::execute() {
    for (size_t seq = 0; seq < batch_size; ++seq) {  // Sequential!
        compute_attention_scores(seq);
    }
}
```

**Fix**: Parallelize operator implementations
```cpp
bool MPIAttentionKernel::execute() {
    // Compute all sequences in parallel
    // Q @ K^T for shape [batch, num_heads, seq, head_dim]
    batched_matmul(Q, K_transpose);  // Parallel across batch
}
```

**Estimated Effort**: 1-2 weeks
- Audit all operators for sequential sections
- Implement batch-parallel versions
- Optimize memory access patterns

---

### Scenario D: Memory Allocation Overhead

**Evidence**:
- Test hangs after batch=4
- MPI warning: "failed to bind memory"
- Cache recreation warnings

**Diagnosis**:
```cpp
// Allocating new tensors for each sequence
for (size_t seq = 0; seq < batch_size; ++seq) {
    Tensor temp(large_size);  // Expensive allocation!
    process(temp);
}
```

**Fix**: Preallocate batch-sized buffers
```cpp
// Allocate once for entire batch
Tensor batch_buffer({batch_size, seq_len, hidden_size});
process_batch(batch_buffer);  // Reuse same memory
```

**Estimated Effort**: 2-4 days
- Identify allocation hotspots
- Implement tensor pooling/reuse
- Fix cache capacity issues

---

## Success Criteria

### Must Achieve
1. **Throughput Scaling**: batch=4 should show ~4× throughput vs batch=1
2. **No Hangs**: Test completes all batch sizes (1, 2, 4, 8, 16, 32)
3. **Target Performance**: batch=32 achieves ≥22× speedup (≥220 tok/s)

### Validation Tests
```bash
# After fixes, verify scaling
./run_batch_performance.sh --filter '*Prefill*'

# Expected results:
# batch=1:   ~10 tok/s  (baseline)
# batch=4:   ~40 tok/s  (4× speedup)
# batch=8:   ~80 tok/s  (8× speedup)
# batch=16: ~160 tok/s (16× speedup)
# batch=32: ~220 tok/s (22× speedup) ✅ TARGET MET
```

---

## Timeline Estimate

| Step | Duration | Depends On |
|------|----------|------------|
| 1. Verify batch path | 30 min | - |
| 2. Examine pipeline | 1 hour | Step 1 |
| 3. Operator analysis | 2 hours | Step 2 |
| 4. Memory layout | 1 hour | Step 3 |
| 5. Profiling | 2 hours | Steps 1-4 |
| 6. Cache check | 1 hour | Step 2 |
| **Total Investigation** | **~8 hours** (1 day) | |
| | | |
| Fix implementation | 2-14 days | Root cause |
| Testing & validation | 1-2 days | Fix complete |
| **Total Time to Resolution** | **3-17 days** | Scenario dependent |

**Best Case** (Scenario A): 3-4 days (simple iteration fix)  
**Worst Case** (Scenario C): 2-3 weeks (operator redesign)  
**Most Likely** (Scenario B): 5-7 days (propagate batch dimension)

---

## Next Actions

**Immediate** (Next 2 hours):
1. ✅ Document findings (this document)
2. 🔄 Execute Step 1: Add instrumentation to verify sequential vs parallel
3. 🔄 Execute Step 2: Examine QwenPipeline batch handling
4. 📊 Create summary of findings for decision point

**Tomorrow** (Day 2):
1. Complete investigation Steps 3-6
2. Identify root cause scenario (A, B, C, or D)
3. Create detailed fix plan with effort estimate
4. Get approval for implementation approach

**This Week**:
1. Implement fix based on root cause
2. Validate with performance tests
3. Document performance improvement
4. Resume Phase 4.4 if successful

---

## Communication Plan

**Stakeholders**: Development team, project management  
**Status Updates**: Daily until resolved  
**Decision Points**: 
- After investigation (8 hours): Root cause identified
- After fix (3-17 days): Performance validated

**Escalation Trigger**: If fix exceeds 2 weeks, consider:
- Parallel batching as Phase 5 stretch goal
- Focus on single-sequence optimization for Phase 4.4
- Defer 22× speedup target to future release

---

**Status**: Investigation pending  
**Assigned**: David Sanftenberg  
**Priority**: 🔴 CRITICAL  
**Blocks**: Phase 4.4, Phase 5
