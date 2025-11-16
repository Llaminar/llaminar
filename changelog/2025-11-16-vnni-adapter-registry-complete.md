# VNNI GEMM Adapter and Registry Implementation - Complete

**Date**: November 16, 2025  
**Status**: ✅ **COMPLETE** - All 432 kernel configurations registered and verified

## Summary

Successfully implemented the VNNI GEMM adapter wrapper function and registry pattern, enabling 432 unique kernel configurations to be auto-registered at startup.

## Work Completed

### 1. Created VNNIGemmAdapter.h (Stub Implementation)
- **File**: `src/v2/kernels/cpu/gemm_v3/VNNIGemmAdapter.h`
- **Template**: `template <int M_R, int N_R, int K_BLK, int UNROLL_K, int PREFETCH_B_L1>`
- **Function**: `void vnni_gemm_adapter(int M, int N, int K, const IActivationTensor& A, const Q8_0Tensor& B, float* C, int ldc)`
- **Status**: Minimal stub (zeros output) - full implementation requires:
  - INT8 quantization of activations
  - Q8_0 block data extraction
  - Packing to VNNI layout
  - Scale extraction from quantized tensors
  - Calling gemm_int8_vnni_kernel with packed data

### 2. Generated VNNI GEMM Instantiation Files
- **Script**: `src/v2/kernels/cpu/gemm_v3/python/generate_vnni_gemm_instantiations.py`
- **Output**: 16 shard files (`VNNIGemmInstantiations_00.cpp` through `_15.cpp`)
- **Configuration Space**:
  - M_R: 8, 16, 32, 64
  - N_R: 16, 32, 64
  - K_BLK: 32, 64, 128
  - UNROLL_K: 1, 2, 4
  - PREFETCH_B_L1: 0, 64, 128, 256
- **Total Configs**: 432 valid configurations (~27 per shard)

### 3. Fixed Build Issues
- Removed duplicate explicit instantiation from `VNNIGemm.h` (line 883-901)
- Removed unnecessary include `tensors/QuantizedTensorTraits.h` from adapter
- Regenerated instantiation files to ensure no conflicts

### 4. Enabled Registry in CMake
- **File**: `src/v2/CMakeLists.txt`
- Uncommented `include(${VNNI_GEMM_SOURCES_CMAKE})`
- Added `${VNNI_GEMM_INSTANTIATION_SOURCES}` to target sources
- Registry instantiation count: **16 files** (was reported as 16, but contains 432 configs)

### 5. Created Registry Verification Test
- **File**: `tests/v2/performance/cpu/kernels/gemm/Perf__VNNIGemmRegistry.cpp`
- **Test**: `V2_Perf_VNNI_GEMM_Registry`
- **Purpose**: Verifies all 432 kernel configurations are registered at startup
- **Result**: ✅ **PASSED** - 432 kernels confirmed

## Build and Test Results

```bash
# Build
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --target llaminar2_core --parallel
# ✅ SUCCESS

# Test
./build_v2_release/performance/v2_perf_vnni_gemm_registry
# Output:
# Number of registered VNNI GEMM kernels: 432
# ✅ SUCCESS: Registry contains all 432 expected kernel configurations!
```

## Architecture

### Registry Pattern Flow

1. **Generation** (`generate_vnni_gemm_instantiations.py`):
   - Generates 16 instantiation shard files
   - Each shard contains ~27 explicit template instantiations
   - Each config has an `__attribute__((constructor))` registration function

2. **Auto-Registration** (Startup):
   ```cpp
   __attribute__((constructor))
   void register_vnni_gemm_8_16_32_1_0() {
       VNNIGemmKernelRegistry::instance().register_kernel(
           8, 16, 32, 1, 0,
           [](int M, int N, int K, ...) {
               vnni_gemm_adapter<8, 16, 32, 1, 0>(M, N, K, A, B, C, ldc);
           }
       );
   }
   ```

3. **Force-Linking** (`VNNIGemmKernelInit.cpp`):
   - Calls `forceLink_VNNIGemmInstantiations_XX()` for each shard
   - Prevents linker from dropping object files from static library

4. **Runtime Dispatch** (VNNIGemmKernelRegistry):
   - 5-tuple key: `(M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1)`
   - Thread-safe singleton pattern
   - Returns `std::function` wrapper to adapter

### Configuration Space Constraints

- M_R, N_R: Must be multiples of 8 (vectorization alignment)
- N_R: Must be ≥16 (VNNI efficiency)
- K_BLK: Must be multiple of 4 (VNNI requirement)
- Fixed params: PREFETCH_B_L2=256, ACCUM_INT32=true, USE_L2_PREFETCH=true, USE_VNNI=true

## Next Steps (Future Work)

### Adapter Full Implementation
The stub adapter needs to be completed with:

1. **Activation Quantization**:
   - Use `IActivationTensor::to_int8_perchannel()` or similar
   - Extract per-row/per-channel activation scales

2. **B Matrix Data Extraction**:
   - Use `ITensorGemmTileDataProvider::get_raw_block_at()` interface
   - Extract Q8_0 blocks with FP16 scales

3. **Matrix Packing**:
   - Pack A: Use existing `pack_A_tile_4x4_grouped()`
   - Pack B: Use existing `pack_B_panel_vnni<K_BLK>()`
   - Create `PackedB` structure with correct strides

4. **Scale Extraction**:
   - Extract activation scales from quantized A tensor
   - Extract weight scales from Q8_0 blocks (FP16 → FP32 conversion)

5. **Kernel Invocation**:
   ```cpp
   gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1, 256, true, true, true>(
       A_packed, Bp, C, bias, act_scales, wgt_scales, M, N, K
   );
   ```

### Performance Testing
Once adapter is fully implemented:
- Run `Perf__VNNIGemmSimple` with real matrix multiplication
- Verify ≥2000 GFLOPS performance (target from previous sessions)
- Compare against OpenBLAS/MKL baselines

### Integration
- Update `src/v2/kernels/cpu/gemm_v3/VNNIGemm.cpp` if needed
- Add dispatch logic based on M, N, K dimensions
- Integrate into V2 pipeline (QwenPipeline, etc.)

## Files Modified/Created

### Created
- `src/v2/kernels/cpu/gemm_v3/VNNIGemmAdapter.h` (stub adapter)
- `src/v2/kernels/cpu/gemm_v3/python/generate_vnni_gemm_instantiations.py` (generator script)
- `src/v2/kernels/cpu/gemm_v3/generated/VNNIGemmInstantiations_00.cpp` through `_15.cpp` (16 files)
- `src/v2/kernels/cpu/gemm_v3/generated/sources.cmake` (CMake sources list)
- `src/v2/kernels/cpu/gemm_v3/VNNIGemmKernelInit.cpp` (force-link init)
- `tests/v2/performance/cpu/kernels/gemm/Perf__VNNIGemmRegistry.cpp` (verification test)

### Modified
- `src/v2/kernels/cpu/gemm_v3/VNNIGemm.h` (removed duplicate explicit instantiation)
- `src/v2/CMakeLists.txt` (uncommented registry include, added instantiation sources)
- `tests/v2/CMakeLists.txt` (added registry verification test)

## Key Insights

1. **Template Instantiation Explosion**: 432 configurations generate ~400KB of code
2. **Linker Optimization**: Static libraries can drop unused object files - force-linking required
3. **Constructor Functions**: `__attribute__((constructor))` runs before `main()` for auto-registration
4. **Stub Pattern**: Adapter can be a stub that zeros output until full implementation ready
5. **Verification First**: Registry verification test ensures infrastructure works before performance testing

## References

- **Q8_1 GEMM Pattern**: `src/v2/kernels/cpu/gemm_v2/python/generate_q8_1_gemm_instantiations.py`
- **Registry Infrastructure**: `src/v2/kernels/cpu/gemm_v3/VNNIGemmKernelRegistry.h`
- **Kernel Template**: `src/v2/kernels/cpu/gemm_v3/VNNIGemm.h` (lines 747-878)
- **Packing Functions**: `pack_A_tile_4x4_grouped()`, `pack_B_panel_vnni<K_BLK>()`
