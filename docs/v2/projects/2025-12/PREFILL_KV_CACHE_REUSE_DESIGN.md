# Prefill KV Cache Reuse Design

## Status

**Current State**: Infrastructure implemented, optimization pending.

- ✅ `AttentionMode` enum added (`DECODE`, `PREFILL`, `AUTO`)
- ✅ `JitAttentionConfig::effectiveMode()` for automatic mode selection
- ✅ Kernel cache separates decode vs prefill kernels
- ✅ Design document with implementation plan
- ⏳ Prefill kernel falls back to decode kernel (correct, not optimized)
- ❌ H→KV→Q loop reordering not yet implemented

## Problem Statement

The current `JitFusedAttentionWo` kernel achieves **~87 GFLOPS for decode** but only **~25 GFLOPS for prefill** (about 3.5x lower). The root cause is the loop structure:

### Current Loop Order (Decode-Optimized)

```
For each q in [0, seq_len_q):           // Loop Q (outer)
    For each head h in [0, num_heads):   // Loop H
        For each kv in [0, seq_len_kv):  // Loop KV (inner)
            score = Q[q,h] · K[kv, kv_h]
            online_softmax_update()
            context[h] += weight * V[kv, kv_h]
        normalize_context()
    Wo_projection(context → output[q])
```

**Why this is good for decode:**
- Decode has `seq_len_q = 1`, so there's no Q loop iteration overhead
- K/V cache is loaded once per head, used once
- Optimal memory access pattern for single-token generation

**Why this is bad for prefill:**
- When `seq_len_q = 128`, we reload the **entire K/V cache 128 times**
- K/V memory bandwidth becomes the bottleneck
- Example: Qwen 7B prefill 128 tokens:
  - K cache size: `128 × 4 × 4 × 36 = 73,728 bytes` per head
  - Total K reloads: `128 queries × 28 heads × 73KB ≈ 256 MB`
  - Could be: `28 heads × 73KB = 2 MB` with reuse

## Proposed Solution: Loop Reordering for Prefill

### New Loop Order (Prefill-Optimized)

```
For each head h in [0, num_heads):           // Loop H (outermost)
    For each kv in [0, seq_len_kv):          // Loop KV (load K/V once)
        K_block = load(K[kv, kv_h])
        V_block = load(V[kv, kv_h])
        For each q in [0, seq_len_q):        // Loop Q (inner)
            if (kv <= q + position_offset):  // Causal mask check
                score = Q[q,h] · K_block
                online_softmax_update[q]()
                context[q,h] += weight * V_block
    For each q in [0, seq_len_q):
        normalize_context[q,h]()
For each q in [0, seq_len_q):
    Wo_projection(context[q] → output[q])
```

### Key Differences

| Aspect | Decode Mode | Prefill Mode |
|--------|-------------|--------------|
| Outer Loop | Q position | Head |
| K/V Cache Access | Load per Q | Load once per head |
| Memory Traffic | `seq_q × heads × seq_kv × block_size` | `heads × seq_kv × block_size` |
| Softmax State | Single state per head | Array of states `[seq_q]` |
| Context Buffer | `[num_heads × head_dim]` | `[seq_q × num_heads × head_dim]` |

### Memory Savings for Qwen 7B Prefill (128 tokens)

| Resource | Current | With Reuse | Savings |
|----------|---------|------------|---------|
| K/V Memory Traffic | ~256 MB | ~2 MB | **128x** |
| Context Buffer | 14 KB | 1.8 MB | Larger |
| Softmax State | 32 bytes | 4 KB | Larger |

The tradeoff: We use more stack space for per-query softmax state and context buffers, but save **128x memory bandwidth** on K/V access.

## Implementation Strategy

### Option A: Dual-Path Kernel (Recommended)

Generate two different JIT kernels based on `seq_len_q`:

```cpp
enum class AttentionMode {
    DECODE,   // seq_len_q == 1, optimize for latency
    PREFILL   // seq_len_q > 1, optimize for K/V reuse
};

class JitFusedAttentionWoGenerator {
    void generate() {
        if (config_.batch_size == 1) {
            generate_decode_kernel();   // Current implementation
        } else {
            generate_prefill_kernel();  // New implementation
        }
    }
};
```

**Pros:**
- Each path fully optimized for its use case
- No runtime mode checks in hot loops
- Can tune tile sizes independently

**Cons:**
- Code duplication (microkernels can be shared)
- Larger cache footprint (two kernels per config)

### Option B: Tiled Hybrid Kernel

Single kernel with tiled loop structure that works for both modes:

```cpp
constexpr int Q_TILE_SIZE = 4;  // Process 4 queries at a time

For each q_tile in [0, seq_len_q, Q_TILE_SIZE):
    For each head h:
        For each kv in [0, seq_len_kv):
            K_block = load(K[kv, kv_h])
            V_block = load(V[kv, kv_h])
            For each q_local in [0, min(Q_TILE_SIZE, seq_len_q - q_tile)):
                q = q_tile + q_local
                // ... attention computation
```

**Pros:**
- Single unified kernel
- Good locality for both modes
- Tile size can be tuned

**Cons:**
- More complex code generation
- May not be optimal for either extreme

### Option C: Runtime Dispatch (Simplest)

Keep current kernel, add separate prefill-optimized kernel, dispatch at runtime:

```cpp
void compute(...) {
    if (seq_len_q == 1) {
        decode_kernel_(...);
    } else {
        prefill_kernel_(...);
    }
}
```

**Pros:**
- Simplest implementation
- No changes to existing decode path
- Easy to A/B test

**Cons:**
- Cache stores two kernels per config

## Detailed Design: Option A (Dual-Path)

### New Data Structures

```cpp
// Per-query online softmax state (for prefill mode)
struct alignas(64) SoftmaxStateArray {
    float max[MAX_SEQ_Q];   // Running max per query
    float sum[MAX_SEQ_Q];   // Running sum per query
};

// Stack layout for prefill mode
struct PrefillStackLayout {
    int q_blocks_offset;           // [num_blocks × 64] - Q blocks for current head
    int softmax_state_offset;      // [seq_len_q × 8] - max/sum per query
    int context_buffer_offset;     // [seq_len_q × head_dim × 4] - FP32 context
    int spill_area_offset;         // [seq_len_q × spill_size] - accumulator spills
};
```

### Prefill Kernel Code Generation

```cpp
void generate_prefill_kernel() {
    // Prologue: save registers, allocate stack
    push_callee_saved();
    int stack_size = calculate_prefill_stack_size();
    sub(rsp, stack_size);
    
    // Initialize all softmax states to (max=-∞, sum=0)
    emit_init_softmax_state_array();
    
    // Initialize all context accumulators to zero
    emit_init_context_array();
    
    // Main loop: H → KV → Q
    for (int h = 0; h < config_.num_heads; ++h) {
        emit_prefill_head_loop(h);
    }
    
    // Normalize all contexts
    for (int h = 0; h < config_.num_heads; ++h) {
        emit_normalize_context_array(h);
    }
    
    // Wo projection for all queries
    emit_wo_projection_batch();
    
    // Epilogue
    add(rsp, stack_size);
    pop_callee_saved();
    ret();
}

void emit_prefill_head_loop(int head_idx) {
    int kv_head_idx = head_idx / heads_per_kv;
    
    // Copy Q[*, h] blocks for all queries (or tile)
    emit_copy_q_head_for_all_queries(head_idx);
    
    // Loop over KV positions
    L("kv_loop_h" + std::to_string(head_idx));
    cmp(reg_kv_idx, reg_seq_len_kv);
    jge("kv_end_h" + std::to_string(head_idx));
    
    // Load K[kv, kv_h] block ONCE
    emit_load_k_block(reg_kv_idx, kv_head_idx);
    
    // Load V[kv, kv_h] block ONCE  
    emit_load_v_block(reg_kv_idx, kv_head_idx);
    
    // Inner loop over Q positions (with causal masking)
    emit_q_inner_loop(head_idx);
    
    inc(reg_kv_idx);
    jmp("kv_loop_h" + std::to_string(head_idx));
    
    L("kv_end_h" + std::to_string(head_idx));
}

void emit_q_inner_loop(int head_idx) {
    // For causal attention: only process q where kv <= q + position_offset
    // This means: q >= kv - position_offset
    // So q_start = max(0, kv - position_offset)
    
    Reg64 reg_q_start = calculate_q_start_for_causal();
    
    L("q_loop");
    cmp(reg_q_idx, reg_seq_len_q);
    jge("q_end");
    
    // Score = Q[q, h] · K_block (K already in registers)
    emit_dot_with_cached_k(reg_q_idx, head_idx);
    
    // Online softmax update for query q
    emit_softmax_update_indexed(reg_q_idx);
    
    // Context[q, h] += weight * V_block (V already in registers)
    emit_v_accum_indexed(reg_q_idx, head_idx);
    
    inc(reg_q_idx);
    jmp("q_loop");
    
    L("q_end");
}
```

### Register Allocation for Prefill

| Register | Decode Use | Prefill Use |
|----------|------------|-------------|
| zmm0-3 | Context accumulators | Context accumulators (per-query, cycled) |
| zmm4-5 | Softmax max/sum | Temp during Q loop |
| zmm6 | Scale | Scale |
| zmm7 | Weight | Weight |
| zmm8 | Correction | Correction |
| zmm9 | Constant 128 | Constant 128 |
| zmm10-15 | Scratch | **K block cache** |
| zmm16-19 | V accum scratch | **V block cache** |
| zmm20-27 | Scratch | Per-query softmax state (max/sum pairs) |

**Key insight**: In prefill mode, we keep K and V blocks resident in registers across the Q inner loop.

### Causal Masking in Prefill Mode

With loop reordering, causal masking becomes:
- For a given `kv` position, only queries where `kv <= q + position_offset` should attend
- Equivalently: `q >= kv - position_offset`

```cpp
// Calculate starting Q index for causal attention
// q_start = max(0, kv - position_offset)
void emit_calculate_q_start_causal() {
    mov(reg_q_start, reg_kv_idx);
    sub(reg_q_start, ptr[rsp + position_offset_spill]);
    // Clamp to 0 if negative
    xor_(reg_tmp, reg_tmp);
    cmp(reg_q_start, 0);
    cmovl(reg_q_start, reg_tmp);
}
```

This is more efficient than the current approach of checking inside the loop because we skip iterations entirely.

## Performance Projections

### Expected Prefill Throughput (Qwen 7B, 128 tokens)

| Metric | Current | With KV Reuse | Speedup |
|--------|---------|---------------|---------|
| K/V Memory BW | 256 MB | 2 MB | 128x |
| Effective GFLOPS | ~25 | ~70-80 | ~3x |
| Limiting Factor | Memory BW | Compute | - |

### When to Use Which Mode

```cpp
bool should_use_prefill_mode(int seq_len_q, int seq_len_kv) {
    // Heuristic: prefill mode benefits when K/V cache is reused enough
    // Break-even point depends on stack overhead vs memory savings
    constexpr int PREFILL_THRESHOLD = 4;
    return seq_len_q >= PREFILL_THRESHOLD;
}
```

## Implementation Phases

### Phase 1: Infrastructure ✅ (Completed December 13, 2025)
- [x] Add `AttentionMode` enum to `JitAttentionConfig`
- [x] Add `effectiveMode()` for automatic mode selection
- [x] Update hash function to separate decode/prefill kernels
- [x] Add fallback in `generate_prefill_kernel()` to decode kernel
- [x] Update benchmarks to pass correct batch_size

### Phase 2: Prefill Kernel (TODO)
- [ ] Implement `generate_prefill_kernel()` with H→KV→Q loop order
- [ ] Implement K/V block caching on stack
- [ ] Implement per-query softmax state arrays
- [ ] Implement per-query context accumulation
- [ ] Handle causal masking efficiently (skip Q iterations)

### Phase 3: Integration (TODO)
- [ ] Add mode dispatch in `JitFusedAttentionWo::compute()`
- [ ] Add specific prefill benchmarks
- [ ] Tune tile sizes and thresholds

### Phase 4: Optimization (TODO)
- [ ] Profile and optimize inner loop
- [ ] Consider Q tiling for very long sequences
- [ ] Consider prefetching for K/V loads

## Testing Strategy

1. **Unit Tests**: Verify prefill kernel matches reference implementation
2. **Parity Tests**: Compare prefill mode output against decode mode (process one query at a time)
3. **Benchmark Tests**: Measure throughput improvement vs current implementation
4. **Stress Tests**: Large sequences (512, 1024, 2048 tokens)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Stack overflow for large seq_q | High | Dynamic allocation or tiling |
| Register pressure | Medium | Spill less-critical state |
| Code complexity | Medium | Share microkernels between modes |
| Incorrect causal masking | High | Thorough parity testing |

## Open Questions

1. **Tile size for Q loop**: Should we process all Q at once or tile?
   - All at once: Maximum K/V reuse, but large stack
   - Tiled: Bounded stack, slightly more K/V traffic

2. **Register allocation trade-off**: Cache all K/V blocks or reload per Q iteration?
   - For `head_dim=128` (4 blocks), we need 8 ZMM registers for K+V
   - This leaves 24 ZMM registers for accumulators and scratch

3. **Wo projection batching**: Process all Q at once or one by one?
   - All at once could enable GEMM-style optimizations
   - Current per-query approach is simpler

## References

- Current implementation: `src/v2/kernels/cpu/jit/q8_1/JitFusedAttentionWo.h`
- Microkernel emitters: `src/v2/kernels/cpu/jit/q8_1/Jit*.h`
- Benchmark results: Session from December 13, 2025
