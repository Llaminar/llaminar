# Investigation Results: Sequential Batching Confirmed

**Date**: October 15, 2025  
**Author**: David Sanftenberg  
**Status**: 🔴 **ROOT CAUSE IDENTIFIED**  
**Investigation Time**: 2 hours

## Executive Summary

Instrumentation has **conclusively proven** that the batching implementation processes sequences **sequentially, not in parallel**. This explains the 0× speedup (0.97×) instead of the target 22× speedup.

### The Smoking Gun

**Evidence from instrumentation**:
```
[BATCH_SEQUENTIAL] Processing 32 sequences SEQUENTIALLY (not parallel!)
[BATCH_SEQUENTIAL] Sequence 0/32: 837.817 ms (8 tokens)
[BATCH_SEQUENTIAL] Sequence 1/32: 835.694 ms (8 tokens)
[BATCH_SEQUENTIAL] Sequence 2/32: 834.827 ms (8 tokens)
...
[BATCH_SEQUENTIAL] Sequence 31/32: 844.201 ms (8 tokens)
[BATCH_SEQUENTIAL] SUMMARY: batch_size=32
[BATCH_SEQUENTIAL]   Total time: 26757.3 ms
[BATCH_SEQUENTIAL]   Sum of sequence times: 26750.9 ms
[BATCH_SEQUENTIAL]   Overhead: 6.46075 ms  (negligible!)
[BATCH_SEQUENTIAL]   ⚠️  NO PARALLELIZATION - sequences processed one-by-one!
```

**Mathematical Proof**:
- Total time: 26,757 ms
- Sum of individual sequence times: 26,751 ms  
- Overhead: **6.5 ms** (0.024% of total)
- **Conclusion**: 99.98% of time is sequential iteration through the batch

## Performance Results Summary

| Batch Size | Expected Speedup | Actual Speedup | Gap | Sequential Time Match |
|------------|------------------|----------------|-----|----------------------|
| 1          | 1.0× (baseline)  | 1.00×          | 0×  | ✅ 813ms (baseline)  |
| 2          | 2.0×             | 0.99×          | 1.01× | ✅ 1,648ms ≈ 2×813ms |
| 4          | 4.0×             | 0.98×          | 3.02× | ✅ 3,314ms ≈ 4×813ms |
| 8          | 8.0×             | 0.98×          | 7.02× | ✅ 6,643ms ≈ 8×813ms |
| 16         | 16.0×            | 0.98×          | 15.02× | ✅ 13,200ms ≈ 16×813ms |
| 32         | **22.0×** (target) | **0.97×**    | **21.03×** | ✅ 26,757ms ≈ 32×813ms |

**Pattern**: Time scales **exactly linearly** with batch size → sequences processed sequentially

## Root Cause: Sequential For-Loop

**Location**: `/workspaces/llaminar/src/QwenPipeline.cpp:2075-2142`

**Problematic Code**:
```cpp
bool QwenPipeline::prefillBatch(...) {
    // Initialize batch state
    ensureBatchState(batch_size);
    
    std::vector<std::vector<float>> batch_logits;
    
    // THE PROBLEM: Sequential iteration through batch
    for (int i = 0; i < batch_size; ++i)  // ❌ SEQUENTIAL!
    {
        // Restore this sequence's state
        n_past_ = n_past_batch_[i];
        k_cache_ = k_cache_batch_[i];
        v_cache_ = v_cache_batch_[i];
        
        // Call single-sequence prefill
        bool success = prefill(token_batches[i], ...);  // ❌ One sequence at a time
        
        // Save this sequence's state
        n_past_batch_[i] = n_past_;
        k_cache_batch_[i] = k_cache_;
        v_cache_batch_[i] = v_cache_;
        
        // Extract logits for this sequence
        batch_logits.push_back(...);
    }
    
    // Concatenate results into batch tensor
    return true;
}
```

**Why This Fails**:
1. ❌ **Loop iterates sequentially** through batch (lines 2090-2142)
2. ❌ **Calls single-sequence `prefill()`** for each iteration
3. ❌ **No tensor batch dimension** - each call processes `[seq_len, hidden]` not `[batch, seq_len, hidden]`
4. ❌ **State swapping overhead** - saving/restoring KV cache for each sequence
5. ❌ **Zero parallelism** - no parallel processing of multiple sequences

## Instrumentation Details

### Test-Level Instrumentation

**Added to** `/workspaces/llaminar/tests/test_batch_performance.cpp`:

```cpp
// Before timed run
if (rank == 0) {
    std::cout << "\n[BATCH_INSTRUMENTATION] Starting prefill for batch_size=" 
              << batch_size << std::endl;
}

// After timed run
if (batch_size > 1) {
    double expected_sequential_ms = 826.0 * batch_size;
    double ratio = duration_ms / expected_sequential_ms;
    std::cout << " [seq_ratio=" << ratio << "]";
}
```

**Output Example**:
```
[BATCH_INSTRUMENTATION] Starting prefill for batch_size=4
Prefill [batch= 4, tokens= 8]:  3313.69 ms, 9.66 tok/s [seq_ratio=1.00]
                                                         ^^^^^^^^^^^^^
                                                         Ratio ≈1.0 proves sequential!
```

### Pipeline-Level Instrumentation

**Added to** `/workspaces/llaminar/src/QwenPipeline.cpp`:

```cpp
// Track sequential processing
auto batch_start = std::chrono::high_resolution_clock::now();
std::vector<double> seq_timings;

if (getRank() == 0) {
    LOG_WARN("[BATCH_SEQUENTIAL] Processing " << batch_size 
             << " sequences SEQUENTIALLY (not parallel!)");
}

for (int i = 0; i < batch_size; ++i) {
    auto seq_start = std::chrono::high_resolution_clock::now();
    
    // ... process sequence i ...
    
    auto seq_end = std::chrono::high_resolution_clock::now();
    double seq_ms = duration(seq_end - seq_start).count();
    seq_timings.push_back(seq_ms);
    
    LOG_WARN("[BATCH_SEQUENTIAL] Sequence " << i << "/" << batch_size 
             << ": " << seq_ms << " ms");
}

// Report summary
auto batch_end = std::chrono::high_resolution_clock::now();
double batch_total_ms = duration(batch_end - batch_start).count();
double sum_seq_ms = 0.0;
for (double t : seq_timings) sum_seq_ms += t;

LOG_WARN("[BATCH_SEQUENTIAL] SUMMARY:");
LOG_WARN("  Total time: " << batch_total_ms << " ms");
LOG_WARN("  Sum of sequence times: " << sum_seq_ms << " ms");
LOG_WARN("  Overhead: " << (batch_total_ms - sum_seq_ms) << " ms");
LOG_WARN("  ⚠️  NO PARALLELIZATION - sequences processed one-by-one!");
```

**Output for batch=32**:
```
[BATCH_SEQUENTIAL] Processing 32 sequences SEQUENTIALLY (not parallel!)
[BATCH_SEQUENTIAL] Sequence 0/32: 837.817 ms (8 tokens)
[BATCH_SEQUENTIAL] Sequence 1/32: 835.694 ms (8 tokens)
...
[BATCH_SEQUENTIAL] Sequence 31/32: 844.201 ms (8 tokens)
[BATCH_SEQUENTIAL] SUMMARY: batch_size=32
[BATCH_SEQUENTIAL]   Total time: 26757.3 ms
[BATCH_SEQUENTIAL]   Sum of sequence times: 26750.9 ms  ← 99.98% of total!
[BATCH_SEQUENTIAL]   Overhead: 6.46 ms  ← negligible
[BATCH_SEQUENTIAL]   ⚠️  NO PARALLELIZATION - sequences processed one-by-one!
[BATCH_SEQUENTIAL]   ⚠️  To fix: Implement true parallel batching (tensor batch dimension)
```

## Key Findings

### 1. Sequential Timing Evidence

**Per-Sequence Timing** (batch=32):
```
Sequence 0:  837.8 ms
Sequence 1:  835.7 ms
Sequence 2:  834.8 ms
...
Sequence 31: 844.2 ms

Average: ~835 ms per sequence
Total: 26,751 ms (sum of all sequences)
Overhead: 6.5 ms (0.024%)
```

**Interpretation**: 
- Each sequence takes ~835ms (consistent with single-sequence baseline)
- Total time = sum of individual times
- **Zero parallelization benefit**

### 2. Linear Scaling Confirmation

**Ratio of actual time to expected sequential time**:
```
batch=2:  1648ms / (2 × 813ms) = 1.01  ← Perfect linear scaling
batch=4:  3314ms / (4 × 813ms) = 1.02  ← Perfect linear scaling
batch=8:  6643ms / (8 × 813ms) = 1.02  ← Perfect linear scaling
batch=16: 13200ms / (16 × 813ms) = 1.01  ← Perfect linear scaling
batch=32: 26757ms / (32 × 813ms) = 1.03  ← Perfect linear scaling
```

**All ratios ≈1.0** → Confirms sequences processed one-by-one, no parallel overlap

### 3. Negligible Overhead

**Overhead breakdown**:
- batch=1: 0.28 ms (state management for single sequence)
- batch=2: 0.49 ms (state management for 2 sequences)
- batch=4: 0.86 ms (state management for 4 sequences)
- batch=32: 6.46 ms (state management for 32 sequences)

**Analysis**: 
- Overhead grows linearly with batch size (~0.2ms per sequence)
- Dominated by state swapping (saving/restoring KV cache)
- **Not the bottleneck** - the sequential loop itself is the problem

### 4. Throughput Plateau

**Throughput across batch sizes**:
```
batch=1:  9.81 tok/s  (baseline)
batch=2:  9.71 tok/s  (-1%)
batch=4:  9.66 tok/s  (-1.5%)
batch=8:  9.58 tok/s  (-2.3%)
batch=16: 9.57 tok/s  (-2.4%)
batch=32: 9.56 tok/s  (-2.5%)
```

**Interpretation**:
- Throughput **constant** (minor degradation due to overhead)
- No speedup from batching
- Each additional sequence adds proportional time

## Why This Happened

### Design Decisions Leading to Sequential Implementation

1. **Per-Sequence State Management**:
   ```cpp
   // Separate state vectors for each sequence
   std::vector<int> n_past_batch_;
   std::vector<std::vector<SimpleTensor>> k_cache_batch_;
   std::vector<std::vector<SimpleTensor>> v_cache_batch_;
   ```
   - State swapping approach inherently sequential
   - Each sequence needs isolated state

2. **Single-Sequence Pipeline Reuse**:
   ```cpp
   bool success = prefill(token_batches[i], ...);  // Reuse single-sequence API
   ```
   - Easy to implement (reuses existing code)
   - But forces sequential iteration
   - No tensor batch dimension support

3. **KV Cache Architecture**:
   - Designed for single-sequence processing
   - No batch dimension in cache tensors
   - State swapping required for multi-sequence

### Why It's Not a Simple Fix

**Naive Approach Would Break**:
```cpp
// This won't work without deeper changes
#pragma omp parallel for  // ❌ Data races on pipeline state!
for (int i = 0; i < batch_size; ++i) {
    prefill(token_batches[i], ...);  // ❌ Shared n_past_, k_cache_, v_cache_
}
```

**Problems**:
- Pipeline state (`n_past_`, `k_cache_`, `v_cache_`) is shared
- Operators designed for single sequence
- KV cache doesn't have batch dimension
- Attention mechanism expects `[seq_len, hidden]` not `[batch, seq_len, hidden]`

## Path Forward

### Required Fix: True Parallel Batching

**Architectural Changes Needed**:

1. **Add Batch Dimension to Tensors** (5-7 days):
   ```cpp
   // Current: [seq_len, hidden_size]
   // Required: [batch_size, seq_len, hidden_size]
   ```

2. **Update All Operators** (5-10 days):
   - Embedding: Batch lookup
   - Linear: Batched matmul `(batch*seq, hidden) @ (hidden, out)`
   - RMSNorm: Batch normalization
   - Attention: Batched Q@K^T, softmax, @V
   - FFN: Batched operations

3. **Redesign KV Cache** (3-5 days):
   ```cpp
   // Current: [num_layers, capacity_tokens, hidden]
   // Required: [num_layers, batch_size, max_seq_len, hidden]
   ```

4. **Update Pipeline Flow** (2-3 days):
   ```cpp
   bool prefillBatch(...) {
       // Stack all sequences into single batch tensor
       Tensor batch_input = stack_sequences(token_batches);  // [batch, seq, ...]
       
       // Single pass through pipeline with batch dimension
       for (int layer = 0; layer < num_layers; ++layer) {
           embedding(batch_input);      // Process all sequences together
           attention(batch_input);      // Parallel attention
           ffn(batch_input);            // Parallel FFN
       }
       
       return true;
   }
   ```

**Estimated Total Effort**: 15-25 days (3-5 weeks)

### Alternative: Simplified Batching (2-3 days)

If full batching is too complex, we could implement operator-level parallelization:

```cpp
bool prefillBatch(...) {
    // Still iterate through batch, but parallelize operators
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        // Each thread gets its own pipeline instance
        thread_local_pipeline[i].prefill(token_batches[i]);
    }
}
```

**Limitations**:
- Not true batching (no tensor batch dimension)
- Still memory-inefficient (multiple pipeline instances)
- Speedup limited by thread count (~8-16× max)
- Won't achieve 22× target

## Immediate Actions Taken

1. ✅ **Added instrumentation** to prove sequential processing
2. ✅ **Identified root cause** in `QwenPipeline::prefillBatch`
3. ✅ **Measured impact**: 0.97× speedup vs 22× target (21× gap)
4. ✅ **Documented evidence** in this investigation report

## Next Steps

**Decision Point**: Choose implementation strategy

**Option A: Full Parallel Batching** (Recommended)
- Timeline: 3-5 weeks
- Achieves 22× target
- Proper architecture for future scaling
- **Decision**: Proceed to detailed design phase

**Option B: Thread-Level Parallelization** (Quick Fix)
- Timeline: 2-3 days
- Achieves 8-16× speedup (partial)
- Simpler implementation
- **Decision**: Only if time-constrained

**Option C: Defer Batching** (Minimal)
- Timeline: 0 days
- No speedup
- Document limitation
- **Decision**: Not recommended (defeats Phase 4 purpose)

## Recommendation

**Implement Option A: Full Parallel Batching**

**Rationale**:
1. Properly solves the problem (not just a workaround)
2. Enables future optimizations (COSMA, tensor parallelism)
3. Aligns with Phase 4 objectives (production batching)
4. 22× speedup achievable with proper implementation

**Proposed Timeline**:
- Week 1: Design review, operator interface updates
- Week 2-3: Implement batched operators (embedding, linear, attention)
- Week 4: KV cache redesign, pipeline integration
- Week 5: Testing, validation, performance verification

---

**Status**: Investigation complete, ready for design phase  
**Blocker Removed**: Root cause identified with proof  
**Next Phase**: Detailed design for parallel batching implementation
