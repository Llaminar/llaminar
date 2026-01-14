# ROCm INT8 GEMM Implementation

**Date:** January 14, 2026  
**Author:** Agent handover  
**Branch:** `feature/cuda-kernels`  
**Status:** ✅ **COMPLETE** - DenseScale path working correctly with fused scaling

---

## Executive Summary

This document describes the INT8×INT8→FP32 GEMM implementation for AMD ROCm GPUs using AMD's [ComposableKernel (CK)](https://github.com/ROCm/composable_kernel) library.

**Key Decision:** We use a **DenseScale** approach that pre-computes the per-element scale matrix and fuses scaling into the GEMM. This is the only supported path after removing dead code from alternative approaches that had unfixable CK API limitations.

**Architecture:**
```
FP32 Activations → Quantize → INT8 Activations ─┐
                                                 ├─→ CK GEMM with Fused Scaling → FP32 Output
INT8 Weights (pre-packed) ──────────────────────┘
                                                 ↑
                    Pre-computed combined_scale[M×N] = scale_A[m] × scale_B[n]
```

**Why Fused Scaling Matters:** The scaling is applied inside CK's GEMM kernel via the `ScaleSingle` element-wise operation. Each thread computes `E[m,n] = dot(A_row, B_col) * combined_scale[m,n]` in registers before writing to memory—no separate memory pass over the output.

---

## Bugs Found and Fixed

### Bug 1: Type Mismatch in Element Op (FIXED)

**Root Cause:** In CK's gridwise kernel at [external/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_dl_multiple_d.hpp](../external/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_dl_multiple_d.hpp):

1. **`c_thread_buf`** was declared as `FloatAcc` (int32_t) to hold GEMM accumulator values
2. The element op `ScaleSingle: e = static_cast<E>(c) * d` expects to write `float` output
3. The original code created `dst_data_refs` pointing to `c_thread_buf` for **both** output (E) and input (C)
4. Writing `float` to `int32_t` storage caused undefined behavior → deterministic zeros at specific positions

**The Fix (3 changes to gridwise_gemm_dl_multiple_d.hpp):**

**1. Added separate output buffer** (line 421-425):
```cpp
// FIX: Separate buffer for element op output with proper type (FloatC)
auto e_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatC>(
    c_thread_desc_m10_m11_n10_n11.GetElementSpaceSize());
```

**2. Changed element op references** (line 639-642):
```cpp
// FIX: Use tie() to create references to different buffers
auto dst_data_refs = tie(
    e_thread_buf(Number<c_offset>{}),  // E& e (output to FloatC buffer)
    c_thread_buf(Number<c_offset>{})   // C& c (input from FloatAcc buffer)
);
```

**3. Changed final transfer source** (line 670-697):
```cpp
ThreadwiseTensorSliceTransfer_v1r3<
    FloatC,  // FIX: Source is now FloatC (e_thread_buf)
    FloatC,
    ...
>.Run(..., e_thread_buf, ...);  // FIX: Use e_thread_buf
```

### Bug 2: Broadcast D-Tensor Coordinate Calculation (RESOLVED - USE DenseScale)

**Status:** Root cause identified. Resolution: Use DenseScale path instead (broadcast approach abandoned).

**Symptom:** After Bug 1 fix, `DenseScaleGemm_Deterministic128` passes perfectly (cos=1.0, 0 zeros), but `FusedGemmScaling_Deterministic128` still showed cos=0.984 with ~10% zeros.

**Root Cause:** In `device_gemm_multiple_d_dl.hpp` line 392, the D-tensor descriptor is **always created with dimensions M × N**, regardless of actual tensor size. With stride=0 for broadcasting, the offset calculation `m*0 + n*1 = n` is wrong for scale_A[M]—it should be `m`. When `n >= M`, this causes out-of-bounds reads.

**Why It Cannot Be Patched:** Fixing this would require:
1. API changes to accept actual D-tensor dimensions (ABI-breaking)
2. Descriptor system changes (6D tile transforms assume M×N backing)
3. Per-D-tensor loop logic based on actual shape
4. Essentially a complete kernel rewrite

**Resolution:** Use `rocmQuantGemm_executeDenseScale()` which pre-computes the combined[M×N] scale matrix. The pre-compute overhead (~0.1ms) is negligible, and **scaling is still fused** into the GEMM kernel via CK's element-wise op—just with a full matrix instead of two broadcast vectors.

### Results After Fixes

| Test | Cosine | Zeros | Status |
|------|--------|-------|--------|
| DenseScaleGemm_Deterministic128 | 1.0 | 0 | ✅ PASS |
| BaseGemm_NoScale_Deterministic128 | 1.0 | 0 | ✅ PASS |

**Note:** FusedGemmScaling tests have been removed as they use the broken broadcast path.

---

## File Locations

### Kernel Implementation

| File | Description |
|------|-------------|
| [src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip](../src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip) | **Main kernel file** - CK INT8 GEMM implementation. Uses `DeviceGemmMultipleD_Dl` with fused `ScaleSingle` element-wise op. Entry point: `rocmQuantGemm_executeDenseScale()`. |
| [src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp](../src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp) | C++ wrapper class - weight packing, scale extraction, CPU-side orchestration |
| [src/v2/kernels/rocm/ROCmQuantisedGemmKernel.h](../src/v2/kernels/rocm/ROCmQuantisedGemmKernel.h) | Header with `ROCmQuantisedGemmKernel` class and `ROCmPackedWeights` struct |

### Test Files

| File | Build Dir | Description |
|------|-----------|-------------|
| [tests/v2/unit/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp](../tests/v2/unit/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp) | `build_v2` | **Unit tests** (CPU-only) - weight packing validation, no GPU required |
| [tests/v2/integration/Test__ROCmQuantisedGemmKernel.cpp](../tests/v2/integration/Test__ROCmQuantisedGemmKernel.cpp) | `build_v2_integration` | **Integration tests** (GPU required) - full GEMM pipeline, `DenseScaleGemm_Deterministic128` now PASSES |
| [tests/v2/unit/kernels/rocm/Test__ROCmFloatingPointGemmKernel.cpp](../tests/v2/unit/kernels/rocm/Test__ROCmFloatingPointGemmKernel.cpp) | `build_v2` | FP32 GEMM tests (separate kernel, works correctly) |

### CK Patch Location

| File | Description |
|------|-------------|
| [external/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_dl_multiple_d.hpp](../external/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_dl_multiple_d.hpp) | **Patched CK gridwise kernel** - fixes type mismatch in element op handling |

### CMake Integration

| File | Description |
|------|-------------|
| [src/v2/cmake/FindComposableKernel.cmake](../src/v2/cmake/FindComposableKernel.cmake) | CMake find module - locates CK headers, **prioritizes local CK over system ROCm** |

---

## Build Instructions

### Prerequisites
- ROCm 7.1.1 installed at `/opt/rocm-7.1.1`
- Local CK checked out at `external/composable_kernel` (rocm-7.1.1 tag)

### Build Commands

```bash
# Unit tests (Debug, CPU-only tests pass)
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug -DHAVE_ROCM=ON
cmake --build build_v2 --parallel

# Integration tests (GPU tests, includes failing tests)
cmake -B build_v2_integration -S src/v2 -DCMAKE_BUILD_TYPE=Integration -DHAVE_ROCM=ON
cmake --build build_v2_integration --parallel

# Release/Performance tests
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release -DHAVE_ROCM=ON
cmake --build build_v2_release --parallel
```

### Running Tests

```bash
# Run unit tests (will pass - CPU only)
ctest --test-dir build_v2 -R "^V2_Unit_.*ROCm" --output-on-failure

# Run integration tests (FusedGemmScaling_Deterministic128 will FAIL)
ctest --test-dir build_v2_integration -R "^V2_Integration_.*ROCm" --output-on-failure -V

# Run the specific failing test with verbose output
cd build_v2_integration
./tests/v2/v2_integration_rocm_quantised_gemm --gtest_filter="*FusedGemmScaling_Deterministic128*"
```

---

## Local ComposableKernel Fork

### Location
```
/workspaces/llaminar/external/composable_kernel/
```

### Version
```bash
$ cd external/composable_kernel && git describe --tags
rocm-7.1.1
```

This is a shallow clone of the official [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) repository at the `rocm-7.1.1` tag to match the system ROCm version.

### Why Local Fork?
1. **Version matching** - System ROCm is 7.1.1, CK headers must match
2. **Patch capability** - Allows modifying CK internals for debugging (e.g., D-tensor load patterns)
3. **Isolation** - Prevents system updates from breaking the build

### CMake Integration

The [FindComposableKernel.cmake](../src/v2/cmake/FindComposableKernel.cmake) module:

1. **Searches local first**: Uses `NO_DEFAULT_PATH` to prioritize `external/composable_kernel/include`
2. **Falls back to system**: If local not found, searches `/opt/rocm/include`
3. **Creates imported target**: `ComposableKernel::ComposableKernel` for modern CMake usage

```cmake
# Key section from FindComposableKernel.cmake:
set(_LOCAL_CK_PATH "${_WORKSPACE_ROOT}/external/composable_kernel")

find_path(ComposableKernel_INCLUDE_DIR
    NAMES ck/ck.hpp
    HINTS
        ${_LOCAL_CK_PATH}/include          # Local patched CK (highest priority)
        ${_ROCM_PATH}/include
    NO_DEFAULT_PATH
)
```

### Updating the Local Fork

```bash
cd /workspaces/llaminar/external
rm -rf composable_kernel
git clone --depth 1 --branch rocm-X.Y.Z https://github.com/ROCm/composable_kernel.git
```

---

## Kernel Configuration (Working)

### CK Minimum Dimension Requirements

**IMPORTANT:** CK's `DeviceGemmMultipleD_Dl` has strict minimum dimension requirements based on tile size:

| Parameter | Minimum Value | Reason |
|-----------|---------------|--------|
| M | 64 | MPerBlock = 64, one block minimum |
| N | 64 | NPerBlock = 64, one block minimum |
| K | 32 | K0PerBlock × K1 = 8 × 4 = 32 |

Dimensions smaller than these will fail `IsSupportedArgument()`. The kernel does NOT automatically pad.

### D-tensor (Scaling) Configuration

The kernel uses CK's `DeviceGemmMultipleD_Dl` with:

```cpp
// Layouts: Row, Row, Row (A=RowMajor, B=RowMajor, E=RowMajor)
// This matches CK's "mk_kn_mn" naming convention

using DeviceGemmInt8_DenseScale = DeviceGemmMultipleD_Dl<
    Row, Row, Tuple<Row>, Row,           // A, B, D, E layouts
    int8_t, int8_t, Tuple<float>, float, // A, B, D, E data types
    int32_t,                              // Accumulator type
    PassThrough, PassThrough, ScaleSingle,// Element-wise ops
    // Tile configuration from CK's mk_kn_mn instance:
    64, 64, 64,                           // BlockSize, MPerBlock, NPerBlock
    8, 4,                                 // K0PerBlock, K1
    4, 4, 1,                              // M1PerThread, N1PerThread, KPerThread
    ...                                   // Block transfer parameters
>;
```

**Dense Scale Configuration:**
- `DsLayout = Tuple<Row>` - single [M×N] scale matrix
- `StrideDs = {N}` - standard row-major stride
- `ScaleSingle` element-wise op: `e = static_cast<float>(c) * d`

**Note:** A broadcast configuration with two D-tensors (scale_A[M] × scale_B[N]) was attempted but abandoned due to CK API limitations (see Bug 2 above). The dense approach is used instead.

---

## API Reference

### Main Entry Point

```cpp
bool rocmQuantGemm_executeDenseScale(
    const int8_t *d_A_int8,      // [M×K] INT8 activations (device)
    const int8_t *d_weights_int8, // [K×N] INT8 weights (device, transposed)
    float *d_C_fp32,              // [M×N] FP32 output (device)
    const float *d_scales_A,      // [M] per-row activation scales (device)
    const float *d_scales_B,      // [N] per-column weight scales (device)
    int M, int N, int K,          // Dimensions
    int rocm_device_id,           // GPU device index
    float *d_work_buffer,         // Optional [M×N] pre-allocated workspace
    size_t work_buffer_size       // Size of workspace in bytes
);
```

**What it does:**
1. Launches a tiny kernel to compute `combined_scale[m,n] = scale_A[m] × scale_B[n]`
2. Calls CK's `DeviceGemmMultipleD_Dl` with the combined scale as a D-tensor
3. CK fuses the scaling: `E[m,n] = (A × B)[m,n] * combined_scale[m,n]`

### Supporting Functions

| Function | Description |
|----------|-------------|
| `rocmQuantGemm_quantizeActivations()` | FP32 → INT8 with per-row symmetric quantization |
| `rocmQuantGemm_uploadWeights()` | Host → Device weight transfer |
| `rocmQuantGemm_areDimensionsSupported()` | Check if M,N,K meet CK minimums |
| `rocmQuantGemm_getMinM/N/K()` | Query minimum dimensions (64, 64, 32) |
| `rocmQuantGemm_executeNoScale()` | Debug: Raw INT8→INT32 GEMM without scaling |

---

## Key Files to Understand

### CK Grid-Level Kernel (D-Tensor Handling)

```
external/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_dl_multiple_d.hpp
```

This file contains the patched gridwise kernel. Key locations:
- Line 418-425: `c_thread_buf` and `e_thread_buf` buffer declarations
- Line 618-645: Element op loop with `dst_data_refs` and `src_data_refs`
- Line 668-697: Final output transfer using `ThreadwiseTensorSliceTransfer_v1r3`

### CK Official Instance Files

```
external/composable_kernel/library/src/tensor_operation_instance/gpu/gemm_universal/
├── device_gemm_dl_i8_i8_i8_mk_kn_mn_instance.cpp  ← Our layout (Row,Row,Row)
├── device_gemm_dl_i8_i8_i8_km_kn_mn_instance.cpp  ← Wrong layout
└── device_gemm_dl_i8_i8_i8_mk_nk_mn_instance.cpp  ← Wrong layout
```

---

## Environment Info

- **System:** Ubuntu 24.04.2 LTS (Dev Container)
- **ROCm:** 7.1.1 at `/opt/rocm-7.1.1`
- **GPU:** gfx906 (MI50/MI60) - uses DL kernels, not XDL (MFMA)
- **ComposableKernel:** rocm-7.1.1 tag at `external/composable_kernel/`

---

## Quick Reference Commands

```bash
# Rebuild integration tests
cmake --build build_v2_integration --parallel

# Run the DenseScale test (should PASS)
cd build_v2_integration && ./tests/v2/v2_integration_rocm_quantised_gemm_kernel \
    --gtest_filter="*DenseScaleGemm_Deterministic128*"

# Run all ROCm GEMM integration tests
ctest --test-dir build_v2_integration -R "V2_Integration_ROCmQuantisedGemmKernel" -V

# Check CK include path is correct
grep "Found ComposableKernel" build_v2_integration/CMakeCache.txt

# View the CK patch diff
git diff external/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_dl_multiple_d.hpp
```

---

## Known Constraints

### 1. Minimum Dimension Requirements

CK's `DeviceGemmMultipleD_Dl` has strict tile-based minimum dimensions:

| Dimension | Minimum | Reason |
|-----------|---------|--------|
| M | 64 | MPerBlock = 64 |
| N | 64 | NPerBlock = 64 |
| K | 32 | K0PerBlock × K1 = 8 × 4 = 32 |

Dimensions below these will fail `IsSupportedArgument()`. The kernel includes **early-exit guards** that return false with a clear error message for undersized dimensions.

**Impact:** Decode phase (M=1) cannot use this kernel. Options:
- Pad M to 64 (wasteful but works)
- Use CPU fallback for decode
- Implement a different small-tile CK config (future work)

### 2. gfx906-Specific

This implementation is tested only on gfx906 (MI50/MI60). The DL (Deep Learning) kernel uses `v_dot4_i32_i8` instructions. Newer GPUs (MI100+) should use XDL/MFMA kernels for better performance.

---

## Integration Test Status

### Passing Tests

| Test | Description |
|------|-------------|
| `DenseScaleGemm_Deterministic128` | Dense [M×N] scale matrix (production path) |
| `BaseGemm_NoScale_Deterministic128` | Raw INT8→INT32 GEMM without scaling |
| `QuantizeActivations_SmallMatrix` | FP32→INT8 quantization |
| `QuantizeActivations_LargeMatrix` | Large matrix quantization |
| `QuantizeActivations_ZeroRow` | Edge case: all-zero rows |
| `QuantizeActivations_ReconstructionAccuracy` | Quantization quality verification |

### Expected Failures (Dimension Constraints)

| Test | Reason |
|------|--------|
| `CKGemm_SingleRow` | M=1 < 64 minimum |
| `CKGemm_SmallBatches` | Dimensions below minimum |
| `CKGemm_7x13x17` | All dimensions below minimum |
| `CKGemm_NonAligned` | N not divisible by NPerBlock |

**Note:** FusedGemmScaling tests have been removed (used broken broadcast D-tensor path).

---

## Changelog

**January 14, 2026:**
- Identified and fixed CK type mismatch bug (e_thread_buf patch)
- Identified unfixable CK API limitation for broadcast D-tensors
- Removed FusedBroadcast, TwoKernel paths (dead code cleanup)
- DenseScale is now the only supported path with fused scaling
- Added dimension guards with clear error messages
- Added exported dimension query functions
---

## Contact / Context

This work is part of the Llaminar V2 inference engine's ROCm backend. The goal is INT8 quantized GEMM for efficient inference on AMD GPUs.

For questions about the broader architecture, see:
- [.github/copilot-instructions.md](../.github/copilot-instructions.md)
- [.github/instructions/llaminar-architecture-v2.instructions.md](../.github/instructions/llaminar-architecture-v2.instructions.md)