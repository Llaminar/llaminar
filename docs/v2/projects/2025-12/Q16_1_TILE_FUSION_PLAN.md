# Q16_1 Fusion in `emit_prefill_wo_q8_1_tile` - Implementation Plan

## Current Function Structure

```
emit_prefill_wo_q8_1_tile:
  for i in [0, d_model) step 2:           // OUTER: output rows
    for k_blk in [0, num_blocks):          // INNER: weight blocks  
      for q in [0, tile_size):             // UNROLLED: queries
        acc[q][i] += context[q][k] * Wo[i][k]
        acc[q][i+1] += context[q][k] * Wo[i+1][k]
    
    for q in [0, tile_size):               // STORE PHASE
      output[(tile_start+q)*d_model + i] = hsum(acc[q][i])
      output[(tile_start+q)*d_model + i+1] = hsum(acc[q][i+1])
```

## Q16_1 Block Layout
```cpp
struct Q16_1Block {  // 72 bytes, covers 32 elements
    float d;           // 4 bytes: scale
    int32_t sum_qs;    // 4 bytes: sum of qs for fast dot
    int16_t qs[32];    // 64 bytes: quantized values
};
```

## Challenge

The current loop iterates over output rows `i` in steps of 2, storing individual FP32 scalars.
Q16_1 requires operating on 32 elements at a time (one block).

## Solution: Block-Aligned Outer Loop

Restructure the outer loop to process 32 output rows at a time (one Q16_1 block worth):

```
emit_prefill_wo_q8_1_tile_with_q16_1_fusion:
  for i_block in [0, d_model/32):          // OUTER: Q16_1 blocks
    for q in [0, tile_size):               // For each query
      // Accumulate 32 output elements for this block
      fp32_block[32] = {0}
      
      for i_local in [0, 32) step 2:       // Output rows within block
        for k_blk in [0, num_blocks):      // Weight blocks
          acc[i_local] += context[q][k] * Wo[i_block*32 + i_local][k]
          acc[i_local+1] += context[q][k] * Wo[i_block*32 + i_local+1][k]
        fp32_block[i_local] = hsum(acc[i_local])
        fp32_block[i_local+1] = hsum(acc[i_local+1])
      
      // Now fuse with Q16_1 residual
      residual_ptr = output + (tile_start + q) * q16_1_row_stride + i_block * 72
      emit_q16_1_residual_fusion_block(residual_ptr, fp32_block)
```

## Problem: Register Pressure

The current implementation uses:
- zmm0-7: acc[q] for row i (8 queries)
- zmm8-15: acc[q] for row i+1 (8 queries)
- zmm16-21: Weight dequant temps
- zmm22-23: Context load temps

To accumulate 32 FP32 values per query, we'd need 32 scalar accumulators or 2 ZMM registers per query.

## Better Solution: Fuse After Full Row Computation

Instead of restructuring the entire loop, we can:

1. **Keep the current loop structure** that iterates `i` in steps of 2
2. **Store to a temp FP32 buffer** (32 floats per query, on stack)
3. **Every 32 rows**, emit inline Q16_1 fusion for the completed block

This is a hybrid approach:
- Minimal changes to the hot inner loop
- Small temp buffer (32 × tile_size × 4 = ~1KB for tile_size=8)
- Fusion happens once per 32 rows, amortizing overhead

## Implementation Steps

### Step 1: Add Stack Space for Block Accumulators

Before the outer loop:
```cpp
// When Q16_1 fusion enabled, allocate temp space for one block per query
// Size: 32 floats × tile_size = 128 bytes × tile_size
int block_accum_offset = ...; // New stack slot
int block_accum_size = 32 * 4 * tile_size; // 32 floats per query
```

### Step 2: Modify Store Logic

In the STORE PHASE, instead of:
```cpp
vmovss(ptr[r10 + r8 * 4], tile_accum_xmm(q));
```

Do:
```cpp
if (Q16_1_fusion_enabled) {
    // Store to block accumulator: block_accum[q * 32 + i_within_block]
    int i_within_block = r8 % 32;
    vmovss(ptr[rsp + block_accum_offset + q * 128 + i_within_block * 4], tile_accum_xmm(q));
    
    // If this completes a 32-element block, emit fusion
    if ((r8 + 2) % 32 == 0 || r8 + 2 >= d_model) {
        // Emit fusion for completed block
        emit_q16_1_block_fusion(q, i_block_idx);
    }
}
```

### Step 3: Implement `emit_q16_1_block_fusion`

This function:
1. Loads 32 FP32 values from block_accum
2. Loads Q16_1 residual block from output
3. Dequantizes residual
4. Adds FP32 values
5. Requantizes and stores back

## Alternative: Inline Block Fusion

Actually, even simpler: We can check at runtime when `i % 32 == 30` (about to finish a block) and emit the fusion inline.

```cpp
// After storing i and i+1
mov(rax, r8);
add(rax, 2);         // rax = i + 2 (next iteration's i)
and_(rax, 31);       // rax = (i + 2) % 32
test(rax, rax);
jnz(".skip_fusion_" + std::to_string(q));

// Block complete! Emit Q16_1 fusion for this block
// i_block = (r8 + 1) / 32 (current block index)
emit_inline_q16_1_block_fusion(q, block_index);

L(".skip_fusion_" + std::to_string(q));
```

## Conclusion

The cleanest implementation:
1. Add 32×tile_size FP32 stack buffer for block accumulators
2. Store to buffer instead of output when Q16_1 enabled
3. After each i += 2, check if block complete
4. When block complete, call existing `emit_q16_1_residual_fusion` with proper offsets

This reuses the existing fusion code and minimizes changes to the hot path.
