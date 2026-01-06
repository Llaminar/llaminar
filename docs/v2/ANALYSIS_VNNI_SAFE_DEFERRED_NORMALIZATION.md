# VNNI-Safe Deferred Normalization: Chunked P×V Accumulation

## The Problem

With deferred normalization, we accumulate:
```
context_accum[d] += w[k] * V[k][d]   for all k in [0, kv_len)
```

Without periodic normalization, this needs int64_t to avoid overflow. But **VPDPWSSD only has a 32-bit accumulator**.

## Key Insight: Two-Level Accumulation

We can use VNNI for the inner loop (chunk accumulation into INT32), then periodically dump to INT64:

```
int64_t context_accum[head_dim];  // Final accumulator
int32_t chunk_accum[head_dim];    // VNNI accumulator

for each chunk of CHUNK_SIZE positions:
    // VNNI-accelerated inner loop → INT32
    for k in chunk:
        for d in 0..head_dim:
            chunk_accum[d] += w[k] * V[k][d]  // VPDPWSSD
    
    // Dump to INT64 (scalar, but infrequent)
    for d in 0..head_dim:
        context_accum[d] += chunk_accum[d]
        chunk_accum[d] = 0
```

## Safe Chunk Size Calculation

### For P×V Accumulation

VNNI accumulates pairs: `acc += (w[0]*V[0]) + (w[1]*V[1])` per instruction.

**Per-instruction worst case:**
- w (scaled to 15 bits): max 32767
- V (INT16): max 32767  
- Two products: 2 × (32767 × 32767) = 2 × 2^30 ≈ 2^31

This is exactly INT32_MAX! So even ONE VNNI instruction can overflow with worst-case values.

**But weights are normalized!** After softmax:
- Σw[k] ≤ 32767 (total weight sum)
- Individual w[k] typically << 32767 (sparse attention)

**Conservative bound with realistic weights:**

If we assume weights are scaled such that max individual weight ≈ 16384 (14 bits):
- Per product: 16384 × 32767 ≈ 2^29
- Two products per VNNI: 2^30
- Safe accumulations before overflow: 2^31 / 2^30 = 2 products = 1 position

That's still very limited.

**Better approach: Scale weights down more aggressively**

Current code uses:
```cpp
int32_t w_scaled = weights_scratch[k] >> (state.lut_value_bits - 15);  // 30 - 15 = 15 bit shift
```

What if we use >> 22 instead of >> 15?
```cpp
int32_t w_scaled = weights_scratch[k] >> 22;  // 8-bit weights, max 255
```

With 8-bit weights:
- Per product: 255 × 32767 ≈ 2^23
- Two products per VNNI: 2^24
- Safe accumulations: 2^31 / 2^24 = 128 positions!

**This is workable!** We can process 128 KV positions before dumping to INT64.

### Trade-off: Weight Precision

Reducing weights from 15 bits to 8 bits loses 7 bits of precision in the attention distribution.

But this may be acceptable because:
1. Softmax is inherently approximate in integer domain anyway
2. The dominant attention weights (high values) lose relatively less precision
3. Low weights (which lose more relative precision) contribute less to output

## Recommended Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `WEIGHT_SCALE_BITS` | 8 | 8-bit weights for VNNI safety |
| `WEIGHT_SHIFT` | `lut_value_bits - 8` = 22 | From 30-bit LUT to 8-bit |
| `PV_CHUNK_SIZE` | 128 | Safe positions per INT32 chunk |
| `INT64_DUMP_FREQ` | Every 128 positions | Dump to int64 accumulator |

## Alternative: Keep 15-bit Weights, Smaller Chunks

If precision is critical:

| Weight bits | Chunk size | Dump frequency |
|-------------|------------|----------------|
| 15 | 4 | Every 4 positions |
| 12 | 32 | Every 32 positions |
| 10 | 128 | Every 128 positions |
| 8 | 512 | Every 512 positions |

With 15-bit weights and chunk size 4:
- For 596 tokens: 149 dumps to INT64 (not ideal but not catastrophic)
- Each dump is O(head_dim) scalar adds

## Implementation

### Chunked P×V with VNNI (Reference)

```cpp
template <typename BlockType>
void q16_pv_accumulate_chunked_int64(
    const int32_t *weights_raw,  // Full-precision weights from softmax
    const BlockType *V,
    int64_t *context_accum,      // INT64 accumulator
    int kv_len,
    int head_dim,
    int blocks_per_row,
    int weight_shift,            // How much to shift weights (e.g., 22 for 8-bit)
    int chunk_size)              // Safe chunk size (e.g., 128)
{
    constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
    
    // VNNI chunk accumulator
    alignas(64) int32_t chunk_accum[256];  // Max head_dim
    
    for (int chunk_start = 0; chunk_start < kv_len; chunk_start += chunk_size)
    {
        int chunk_end = std::min(chunk_start + chunk_size, kv_len);
        
        // Zero chunk accumulator
        std::memset(chunk_accum, 0, head_dim * sizeof(int32_t));
        
        // VNNI-friendly inner loop (INT32 accumulation)
        for (int k = chunk_start; k < chunk_end; ++k)
        {
            // Scale weight to fit VNNI safety bound
            int32_t w = weights_raw[k] >> weight_shift;
            if (w == 0) continue;
            
            for (int b = 0; b < blocks_per_row; ++b)
            {
                const int16_t *v_data = V[k * blocks_per_row + b].qs;
                int start = b * BLOCK_SIZE;
                int count = std::min(BLOCK_SIZE, head_dim - start);
                
                // This vectorizes to VPDPWSSD
                for (int i = 0; i < count; ++i)
                {
                    chunk_accum[start + i] += static_cast<int32_t>(w) * static_cast<int32_t>(v_data[i]);
                }
            }
        }
        
        // Dump chunk to INT64 (infrequent, scalar is fine)
        for (int d = 0; d < head_dim; ++d)
        {
            context_accum[d] += static_cast<int64_t>(chunk_accum[d]);
        }
    }
}
```

### Finalization

```cpp
void q16_pv_finalize(
    const int64_t *context_accum,
    int32_t *output,
    int64_t total_weight_sum,  // Σw[k] at full precision (before shift)
    int head_dim,
    int weight_shift)
{
    // Compute effective denominator (total weight sum at scaled precision)
    // Since we shifted weights by weight_shift, the sum is also shifted
    int64_t scaled_sum = total_weight_sum >> weight_shift;
    
    for (int d = 0; d < head_dim; ++d)
    {
        // Integer division for final normalization
        output[d] = static_cast<int32_t>(context_accum[d] / scaled_sum);
    }
}
```

## Overflow Proof

**Given:**
- Weight shift = 22 (8-bit weights, max 255)
- V values: max 32767 (INT16)
- Chunk size = 128

**Per chunk accumulation:**
- Worst case per position: 255 × 32767 = 8,355,585 ≈ 2^23
- Sum over 128 positions: 128 × 2^23 = 2^30
- INT32 max: 2^31 - 1

**Margin:** 2^31 / 2^30 = 2× safety factor. ✓

**For INT64 total accumulation:**
- Max positions: 32K tokens
- Per-position at full precision: 2^30 (LUT) × 2^15 (V) = 2^45
- Sum over 32K: 32K × 2^45 = 2^60
- INT64 max: 2^63

**Margin:** 2^63 / 2^60 = 8× safety factor. ✓

## Summary

**Deferred normalization IS compatible with VNNI**, with these constraints:

1. **Scale weights more aggressively** (8-10 bits instead of 15)
2. **Chunk accumulation** with INT32 → INT64 dumps every 128-512 positions
3. **Single finalization** at the end (one INT64 division per element)

This eliminates the O(N × head_dim) FP divisions while staying VNNI-safe.
