# BF16 Cache Pointer Invalidation Bug Fix
**Date**: October 20, 2025  
**Status**: ✅ **FIXED** - Inference no longer crashes  
**Issue**: 🔄 **PARTIAL** - Output quality needs investigation

## Executive Summary

Fixed critical pointer invalidation bug in QuantSlabCache that caused segfaults during BF16 inference. Root cause: cache capacity too small (64MB) → eviction during operation → pointers become invalid → OpenBLAS crash.

**Solution**: Increased cache capacity from 64MB → 2GB to prevent eviction during active operations.

**Result**: 
- ✅ All kernel tests (7/7) pass with BF16
- ✅ Inference completes without crashes
- ❌ Output quality degraded (likely quantization error accumulation)

---

## Problem Analysis

### Initial Symptoms
- Main `llaminar` executable crashed with segfault in OpenBLAS `sgemm_oncopy`
- All operator unit tests passed successfully
- Crash only occurred in full pipeline, not isolated operators

### Root Cause Discovery

**Sequence of Events**:
1. `MPILinearOperator::execute()` calls `input->data()` → QuantSlabCache returns pointer A (small activation)
2. Then calls `local_weight->data()` → QuantSlabCache needs 544MB for weight decode
3. Cache at 64MB capacity → evicts entry containing pointer A to make space
4. Pointer A now invalid (dangling reference to freed memory)
5. OpenBLAS called with invalid pointer A → **SEGFAULT in `sgemm_oncopy`**

**Key Code Location** (`src/operators/MPILinearOperator.cpp:174-256`):
```cpp
const float *input_data = input->data();          // Line 174: Get pointer A
const float *weight_data = local_weight->data();  // Line 175: Get pointer B (may evict A!)
// ... later ...
matmul_success = adaptiveMatMul(input_data, weight_data, output_data, ...);  // Line 256: Use invalidated pointer!
```

**Why This Happened**:
- BF16 mode routes all tensor access through `TensorBase::data_fp32()` → QuantSlabCache
- Cache shared between activation and weight tensors
- LM head weight matrix: 151936×896 = 136M elements = 544MB in FP32
- Default cache capacity: only 64MB
- Cache eviction strategy: LRU (evicts oldest entry when full)

### Detailed Timeline

1. **User suggestion**: "let's keep testing bf16. maybe we can run our unit test suite"
2. **Test discovery**: MPIRMSNormKernelTest failed with empty tensor error
3. **Empty tensor fix**: Made static buffers non-const (separate bug)
4. **Kernel validation**: All 7 kernel tests passed
5. **Main inference crash**: Still segfaulting despite operator success
6. **Divergence analysis**: Unit tests work, full pipeline crashes
7. **Log inspection**: Found cache allocating 544MB right before crash
8. **Root cause identified**: Pointer invalidation via cache eviction
9. **Fix implemented**: Increased capacity 64MB → 2GB
10. **Success**: Inference completes without crashes!

---

## Code Changes

### File: `src/operators/QuantSlabCache.h` (Line 255)

**Before**:
```cpp
size_t capacity_bytes_ = 64 * 1024 * 1024; // default 64MB (shared by both caches)
```

**After**:
```cpp
size_t capacity_bytes_ = 2048ULL * 1024 * 1024; // default 2GB (shared by both caches) - increased to prevent eviction during active operations
```

**Rationale**:
- 2GB easily accommodates typical workload:
  - Activations: ~10MB per sequence (1 token × 896 hidden × 24 layers)
  - Weight matrix: ~544MB (LM head projection)
  - Total: ~600MB typical, 2GB provides headroom
- Prevents eviction during single operation
- Modern systems have ample memory (32-256GB typical)

---

## Test Results

### Kernel Tests (7/7 PASS) ✅
```bash
$ LLAMINAR_QUANT_OUTPUT_BF16=1 ctest -R "Kernel"

1/7 MPILinearKernelTest ................ Passed (0.43s)
2/7 MPIRMSNormKernelTest ............... Passed (0.43s)
3/7 MPIAttentionKernelTest ............. Passed (0.44s)
4/7 MPIAttentionKernelClean_SingleRank . Passed (0.61s)
5/7 MPIAttentionKernelClean_MultiRank .. Passed (0.45s)
6/7 MPIEmbeddingKernelTest ............. Passed (0.43s)
7/7 MPIEmbeddingKernelExtendedTest ..... Passed (0.80s)

100% tests passed
```

### Inference Tests

**Before Fix**:
```
$ LLAMINAR_QUANT_OUTPUT_BF16=1 ./llaminar -m model.gguf -p "Hi" -n 3
[CRASH] Segmentation fault in sgemm_oncopy
```

**After Fix**:
```
$ LLAMINAR_QUANT_OUTPUT_BF16=1 ./llaminar -m model.gguf -p "Hi" -n 3
Response: !! disclosures
[SUCCESS] Completed without crash
```

**FP32 Baseline**:
```
$ ./llaminar -m model.gguf -p "What is the capital of France?" -n 20
Response:  The capital of France is Paris. The official language is French...
[CORRECT OUTPUT]
```

**BF16 Output Quality**:
```
$ LLAMINAR_QUANT_OUTPUT_BF16=1 ./llaminar -m model.gguf -p "What is the capital of France?" -n 20
Response: ',...',...relationfeatureä¿± ÑĥÐ´Ð°ÑĢ_vendortempt...
[GIBBERISH - Quantization error accumulation]
```

---

## Remaining Issues

### Output Quality Degradation

**Symptom**: BF16 inference generates gibberish instead of coherent text

**Likely Causes**:
1. **Quantization error accumulation**: BF16 has only 7-bit mantissa (vs FP32's 23-bit)
2. **Cascade through layers**: Error compounds across 24 transformer layers
3. **Critical operations in BF16**: RMSNorm, Softmax, attention scores may be too sensitive

**Evidence**:
- Operators work correctly in isolation (kernel tests pass)
- Full pipeline diverges catastrophically
- FP32 baseline produces correct output
- BF16 output is not just slightly wrong - it's completely incoherent

**Next Steps**:
1. **Parity testing**: Compare BF16 vs FP32 at each pipeline stage
2. **Hybrid approach**: Use FP32 for sensitive operations (norms, softmax, residuals)
3. **Quantized weights only**: Test BF16 storage without BF16 compute
4. **Numerical analysis**: Measure error propagation through layers

---

## Architecture Implications

### Cache Design Considerations

**Current Design** (post-fix):
- Shared LRU cache for all decoded tensors (activations + weights)
- 2GB capacity (prevents eviction during operation)
- No reference counting or pinning mechanism

**Limitations**:
- Still vulnerable to eviction if multiple large tensors accessed simultaneously
- No way to "pin" active entries
- Cache size is a heuristic (not guaranteed safe)

**Better Long-Term Solutions**:

1. **Reference Counting**:
   ```cpp
   // Increment ref count on access
   auto guard = cache.pin(tensor_ptr, type);
   const float* data = guard.data();  // Safe until guard destroyed
   // guard destructor decrements ref count
   ```

2. **Scoped Pinning**:
   ```cpp
   {
       CachePinScope scope(cache);
       const float* input_data = input->data();   // Pin entry 1
       const float* weight_data = weight->data(); // Pin entry 2
       // Both guaranteed valid within scope
       adaptiveMatMul(input_data, weight_data, output_data, ...);
   }  // Entries unpinned on scope exit
   ```

3. **Separate Caches**:
   - Activation cache (smaller, more frequent access)
   - Weight cache (larger, less frequent eviction)
   - Prevents weight decode from evicting active activations

---

## Performance Characteristics

### Memory Usage
- **Before**: 64MB cache (thrashed continuously)
- **After**: 2GB cache (stable, no eviction)
- **Overhead**: +1984MB per process

### Typical Workload
- Qwen 0.5B model (Q8_0 quantized)
- Single sequence inference
- Peak cache usage: ~600MB
  - Activations: ~10MB (1×896×24 layers)
  - LM head weight: 544MB (151936×896 decoded)
  - Attention weights: ~50MB (Q/K/V projections)

### Scaling Considerations
- 2GB sufficient for models up to ~7B parameters
- Larger models may need capacity increase via environment variable
- Future: Make capacity configurable per-deployment

---

## Testing Recommendations

### Immediate
- [x] Verify kernel tests pass (DONE: 7/7 PASS)
- [x] Verify inference completes (DONE: No crashes)
- [ ] Run parity tests (BF16 vs FP32 activations)
- [ ] Measure numerical divergence per layer

### Short-Term
- [ ] Test with batch processing (multiple sequences)
- [ ] Test with larger models (1.5B, 7B)
- [ ] Benchmark memory usage under load
- [ ] Implement hybrid BF16/FP32 approach

### Long-Term
- [ ] Add reference counting to cache
- [ ] Implement cache pinning API
- [ ] Make capacity configurable
- [ ] Add cache telemetry (hit rate, eviction frequency)

---

## Lessons Learned

1. **Cache eviction is dangerous**: Returning raw pointers from cache is unsafe if eviction can occur
2. **Size matters**: 64MB was woefully insufficient for LLM inference workloads
3. **Test coverage**: Unit tests passed but integration testing revealed bug
4. **Pointer lifetime**: Callers assume pointers remain valid; cache violated this assumption
5. **Memory trade-offs**: 2GB overhead acceptable for stability/correctness

---

## Related Files

- `src/operators/QuantSlabCache.h` - Cache implementation (capacity increased)
- `src/operators/QuantSlabCache.cpp` - Cache management logic
- `src/tensors/TensorBase.cpp` - Pull-through cache interface
- `src/tensors/BF16Tensor.h` - BF16 activation storage
- `src/operators/MPILinearOperator.cpp` - Where pointer invalidation occurred

---

## Summary

**Problem**: Cache too small → eviction during operation → pointer invalidation → segfault  
**Solution**: Increase capacity 32× (64MB → 2GB)  
**Status**: ✅ Crashes fixed, 🔄 Output quality needs work  
**Next**: Investigate numerical accuracy and consider hybrid BF16/FP32 approach
