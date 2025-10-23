# Fused IQ4_NL GEMM Kernel Design

## Current Architecture (2-Stage)

```cpp
// Stage 1: Dequantize entire block (32 elements) to temporary buffer
alignas(64) float B_block[32];
tensor_->decode_block_at(j, kb, B_block);  // IQ4_NL → FP32

// Stage 2: Dot product with dequantized buffer
float result = dot_product_simd(A_row, B_block, 32);
```

**Memory Traffic:**
- Read: 18 bytes quantized (16 bytes qs + 2 bytes scale)
- Write: 128 bytes FP32 buffer (32 × 4 bytes)
- Read: 128 bytes FP32 buffer (for dot product)
- **Total: 274 bytes** (18 read + 128 write + 128 read)

**Cache Pressure:**
- Temporary buffer thrashes L1 cache
- Single-use data written then immediately read

---

## Fused Architecture (1-Stage)

```cpp
/**
 * @brief Fused dot product: A (FP32) · B (IQ4_NL block, no intermediate buffer)
 * 
 * Combines dequantization + dot product in single kernel.
 * Eliminates 128-byte temporary buffer (saves L1 bandwidth).
 * 
 * @param A_fp32 Input vector (FP32, length 32)
 * @param B_block Quantized weight block (IQ4_NL format)
 * @return Dot product result
 */
static inline float dot_product_fused_iq4nl(
    const float* A_fp32,
    const IQ4_NLBlock& B_block
) {
#if defined(__AVX512F__)
    // Extract scale once (BF16→FP32 conversion)
    const float scale = simd::fp16_to_fp32(B_block.d);
    const __m512 scale_vec = _mm512_set1_ps(scale);
    
    // Accumulators for two 16-element chunks
    __m512 sum = _mm512_setzero_ps();
    
    // Process first 16 values (low nibbles)
    {
        // Step 1: Extract and lookup low nibbles (16 values)
        alignas(64) int8_t kvals_low[16];
        for (int i = 0; i < 16; ++i) {
            kvals_low[i] = kvalues_iq4nl[B_block.qs[i] & 0x0F];
        }
        
        // Step 2: Convert int8 → FP32 (vectorized)
        __m128i i8_vals = _mm_loadu_si128((__m128i*)kvals_low);
        __m512i i32_vals = _mm512_cvtepi8_epi32(i8_vals);
        __m512 B_fp32 = _mm512_cvtepi32_ps(i32_vals);
        __m512 B_scaled = _mm512_mul_ps(B_fp32, scale_vec);
        
        // Step 3: FMA with A (fused multiply-add)
        __m512 A_vec = _mm512_loadu_ps(A_fp32);
        sum = _mm512_fmadd_ps(A_vec, B_scaled, sum);
    }
    
    // Process second 16 values (high nibbles)
    {
        alignas(64) int8_t kvals_high[16];
        for (int i = 0; i < 16; ++i) {
            kvals_high[i] = kvalues_iq4nl[B_block.qs[i] >> 4];
        }
        
        __m128i i8_vals = _mm_loadu_si128((__m128i*)kvals_high);
        __m512i i32_vals = _mm512_cvtepi8_epi32(i8_vals);
        __m512 B_fp32 = _mm512_cvtepi32_ps(i32_vals);
        __m512 B_scaled = _mm512_mul_ps(B_fp32, scale_vec);
        
        __m512 A_vec = _mm512_loadu_ps(A_fp32 + 16);
        sum = _mm512_fmadd_ps(A_vec, B_scaled, sum);
    }
    
    // Horizontal sum
    return _mm512_reduce_add_ps(sum);
    
#else
    // Scalar fallback
    const float scale = simd::fp16_to_fp32(B_block.d);
    float result = 0.0f;
    
    for (int i = 0; i < 16; ++i) {
        const uint8_t qbyte = B_block.qs[i];
        
        // Low nibble
        int8_t kval_low = kvalues_iq4nl[qbyte & 0x0F];
        result += A_fp32[i] * (scale * static_cast<float>(kval_low));
        
        // High nibble
        int8_t kval_high = kvalues_iq4nl[qbyte >> 4];
        result += A_fp32[i + 16] * (scale * static_cast<float>(kval_high));
    }
    
    return result;
#endif
}
```

**Memory Traffic:**
- Read: 18 bytes quantized (16 bytes qs + 2 bytes scale)
- Read: 128 bytes A_fp32
- **Total: 146 bytes** (saves 128 bytes vs 2-stage!)

---

## Performance Analysis

### Memory Bandwidth Savings

**Current 2-stage:**
```
Per 32-element block:
  Quantized read:     18 bytes
  Buffer write:      128 bytes  ← ELIMINATED
  Buffer read:       128 bytes  ← ELIMINATED
  A read:            128 bytes
  Total:             274 bytes
```

**Fused kernel:**
```
Per 32-element block:
  Quantized read:     18 bytes
  A read:            128 bytes
  Total:             146 bytes  (47% reduction!)
```

### Cache Efficiency

**Current:**
- Temporary buffer (32 floats = 128 bytes) evicts other hot data from L1
- Each block decode: 128-byte allocation
- For Q-projection 4096 tokens: 4096 × (896/32) = 114,688 temp buffers!
- Total temp memory: 14.5 MB thrashing through 32 KB L1 cache

**Fused:**
- No temporary buffers
- Only working registers (zmm0-zmm7)
- Quantized data stays compressed in L2/L3 until needed
- L1 cache available for A matrix and accumulators

### Expected Performance Gain

**Theoretical:**
- Memory bandwidth: **47% reduction** → up to 1.88× speedup (if memory-bound)
- L1 cache pressure: **~450× less** temporary allocation

**Realistic Estimate:**
- Current bottleneck: 18-20% dot_product_simd + 30% OpenMP overhead
- Fused kernel eliminates buffer write/read latency
- Expected gain: **+15-25%** overall throughput

**Target:**
- Q-projection 4096: 394 GFLOPS → **453-492 GFLOPS**
- FFN 512: 525 GFLOPS → **603-656 GFLOPS**

---

## Implementation Strategy

### Step 1: Add Fused Kernel Function

Location: `src/tensors/IQ4_NLTensor.h`

```cpp
private:
    // Existing helper
    static inline float dot_product_simd(const float* a, const float* b, size_t count);
    
    // NEW: Fused kernel
    static inline float dot_product_fused_iq4nl(const float* A_fp32, const IQ4_NLBlock& B_block);
```

### Step 2: Modify multiply() to Use Fused Path

**Current code (lines ~768-780):**
```cpp
for (int kb = 0; kb < num_k_blocks; ++kb) {
    size_t k_start = kb * 32;
    size_t k_count = std::min(32, k - static_cast<int>(k_start));
    
    alignas(64) float B_block[32];  // ← Temporary buffer
    tensor_->decode_block_at(row_offset + j, kb, B_block);
    
    acc[i] += dot_product_simd(A_row + k_start, B_block, k_count);
}
```

**Fused code:**
```cpp
for (int kb = 0; kb < num_k_blocks; ++kb) {
    const IQ4_NLBlock& B_block_ref = tensor_->get_block_at(row_offset + j, kb);
    
    // Fused: dequant + dot product in single kernel
    acc[i] += dot_product_fused_iq4nl(A_row + kb * 32, B_block_ref);
}
```

### Step 3: Apply to All Paths

Need to update:
1. ✅ Cache-blocked path (m=2-16): Lines 758-785
2. ✅ Row-wise path (m>16): Lines 889-945
3. ✅ BF16 cache-blocked: Lines 1115-1145
4. ✅ BF16 row-wise: Lines 1150-1210

---

## Advanced Optimizations (Phase 2)

### Option A: VNNI-Style Fused Kernel

For systems with AVX512-VNNI, we could keep weights in int8:

```cpp
static inline float dot_product_fused_iq4nl_vnni(
    const float* A_fp32,
    const IQ4_NLBlock& B_block
) {
    // Quantize A to int8 (once per block)
    alignas(64) int8_t A_i8[32];
    float A_scale = quantize_fp32_to_int8(A_fp32, A_i8, 32);
    
    // Extract B scale
    float B_scale = simd::fp16_to_fp32(B_block.d);
    
    // VNNI: int8×int8 dot product (4× throughput vs FP32)
    int32_t dot_i32 = vnni_dot_product_iq4nl(A_i8, B_block);
    
    // Rescale
    return (A_scale * B_scale) * static_cast<float>(dot_i32);
}
```

**Pros:** 4× computational throughput (VNNI instruction)
**Cons:** A quantization overhead (~20 cycles per block)

### Option B: Amortized Block Caching

Cache decoded blocks if they're reused across multiple A rows:

```cpp
// Thread-local cache (per column)
struct BlockCache {
    size_t col_idx;
    size_t k_block;
    alignas(64) float decoded[32];
};

// Check cache before decode
if (cache.col_idx == j && cache.k_block == kb) {
    // Cache hit: reuse decoded data
    acc[i] += dot_product_simd(A_row, cache.decoded, 32);
} else {
    // Cache miss: fused kernel
    const IQ4_NLBlock& B = tensor_->get_block_at(j, kb);
    acc[i] += dot_product_fused_iq4nl(A_row, B);
    // Optionally populate cache for next row
}
```

**Pros:** Amortizes decode cost across multiple rows
**Cons:** Only helps when m > 1 (batch decoding)

---

## Testing Plan

### Correctness Verification

```cpp
// Unit test: Fused vs non-fused parity
TEST(IQ4NLFusedKernel, ParityWithSeparateDecodeDot) {
    // Generate test block
    IQ4_NLBlock block;
    block.d = simd::fp32_to_fp16(0.5f);
    for (int i = 0; i < 16; ++i) {
        block.qs[i] = (i << 4) | (15 - i);  // Arbitrary pattern
    }
    
    // Test vector A
    alignas(64) float A[32];
    for (int i = 0; i < 32; ++i) A[i] = static_cast<float>(i) * 0.1f;
    
    // Method 1: Separate decode + dot
    alignas(64) float B_decoded[32];
    IQ4_NLTensor::decodeBlockAVX512(block, B_decoded);
    float result_separate = IQ4_NLTensor::dot_product_simd(A, B_decoded, 32);
    
    // Method 2: Fused
    float result_fused = IQ4_NLTensor::dot_product_fused_iq4nl(A, block);
    
    // Should be bit-exact (same FP operations)
    EXPECT_FLOAT_EQ(result_fused, result_separate);
}
```

### Performance Benchmarking

```bash
# Baseline (current)
./run_benchmark.sh benchmark_iq4nl_gemm > baseline_unfused.txt

# Apply fused kernel
# ... implement changes ...

# Benchmark fused
cmake --build build_release --target benchmark_iq4nl_gemm --parallel
./run_benchmark.sh benchmark_iq4nl_gemm > optimized_fused.txt

# Compare
python3 scripts/compare_benchmarks.py baseline_unfused.txt optimized_fused.txt
```

---

## Risk Assessment

### Low Risk
✅ **Correctness:** Fused kernel performs identical FP operations (bit-exact results)
✅ **Maintainability:** Isolated change, doesn't affect other code paths
✅ **Rollback:** Keep both implementations, use feature flag to switch

### Medium Risk
⚠️ **Register pressure:** Fused kernel uses more registers (may spill)
  - Mitigation: Profile with `perf stat -e cycles,instructions,stalls`
  
⚠️ **Instruction cache:** Larger code footprint
  - Mitigation: Inline carefully, use `__attribute__((always_inline))`

### High Risk
❌ **Compiler optimization interference:** Fused code harder to optimize
  - Mitigation: Hand-tune with intrinsics, verify assembly with `objdump`
  
❌ **Non-AVX512 platforms:** Scalar fallback must be efficient
  - Mitigation: Implement AVX2 variant, test on multiple architectures

---

## Expected Timeline

- **Day 1:** Implement `dot_product_fused_iq4nl()` with AVX512 path
- **Day 2:** Unit tests (correctness parity verification)
- **Day 3:** Integrate into multiply() cache-blocked path
- **Day 4:** Integrate into row-wise and BF16 paths
- **Day 5:** Performance benchmarking and profiling
- **Day 6:** Optimization tuning based on profiling results

**Total:** ~1 week for complete fused kernel implementation and validation

---

## Next Steps

1. **Implement prototype:** Start with cache-blocked path only
2. **Verify parity:** Unit test against existing decode+dot
3. **Benchmark:** Measure Q-projection 4096 performance
4. **Profile:** Check if we eliminated buffer overhead
5. **Expand:** Apply to all paths if successful
6. **Document:** Update changelog with results

**Success Criteria:**
- ✅ Bit-exact results vs current implementation
- ✅ +15% throughput on Q-projection 4096 (394 → 453+ GFLOPS)
- ✅ Reduced L1 cache misses (verify with `perf stat -e L1-dcache-load-misses`)
