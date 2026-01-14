# ROCmQuantisedGemmKernel Implementation Plan

**Project**: INT8 Quantized GEMM for AMD ROCm (gfx906+)  
**Target**: Parity with CUDAQuantisedGemmKernel using ComposableKernel library  
**Created**: January 13, 2026  
**Status**: Planning

---

## Executive Summary

Implement `ROCmQuantisedGemmKernel` to provide INT8×INT8→INT32 GEMM capability on AMD GPUs, mirroring the existing `CUDAQuantisedGemmKernel`. The implementation will use AMD's **ComposableKernel (CK)** library with `DeviceGemmDl` for gfx906 (MI50) support, with automatic upgrade path to `DeviceGemmXdl` for MI100/MI200/MI300.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ITensorGemm Interface                            │
├──────────────────────────────────────────────────────────────────────────┤
│  multiply_tensor(A, C, ...)  │  multiply_fused_tensor(...)              │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────┐   ┌───────────────────────────────────────┐
│   CUDAQuantisedGemmKernel     │   │      ROCmQuantisedGemmKernel          │
│   (CUTLASS INT8 Tensor Core)  │   │      (ComposableKernel INT8)          │
├───────────────────────────────┤   ├───────────────────────────────────────┤
│ • CUDAQuantisedGemmKernel.h   │   │ • ROCmQuantisedGemmKernel.h           │
│ • CUDAQuantisedGemmKernel.cpp │   │ • ROCmQuantisedGemmKernel.cpp         │
│ • ..._CUTLASS.cu (nvcc)       │   │ • ..._CK.hip (hipcc)                  │
└───────────────────────────────┘   └───────────────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌───────────────────────────────────────┐
                                    │     ComposableKernel Library          │
                                    ├───────────────────────────────────────┤
                                    │ DeviceGemmDl (gfx906, RDNA)           │
                                    │ DeviceGemmXdl (gfx908+, CDNA)         │
                                    └───────────────────────────────────────┘
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        WEIGHT CONVERSION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Quantized Weights (IQ4_NL, Q8_0, Q4_0, etc.)                          │
│       │                                                                 │
│       ▼  [dequantize to FP32]                                          │
│  FP32 Weights [N × K]                                                  │
│       │                                                                 │
│       ▼  [per-column symmetric quantization]                           │
│  INT8 Weights [K × N] ColumnMajor + FP32 scales [N]                    │
│       │                                                                 │
│       ▼  [hipMemcpy to device]                                         │
│  Device INT8 Weights + Device Scales                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        ACTIVATION QUANTIZATION                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FP32 Activations [M × K] RowMajor (on device)                         │
│       │                                                                 │
│       ▼  [per-row symmetric quantization kernel]                       │
│  INT8 Activations [M × K] RowMajor + FP32 scales [M]                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           GEMM + SCALING                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INT8 A [M×K] × INT8 B [K×N] → INT32 C [M×N]   (CK DeviceGemmDl)       │
│       │                                                                 │
│       ▼  [scaling kernel or fused epilogue]                            │
│  FP32 Output [M×N] = INT32 C × scale_A[row] × scale_B[col]             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Breakdown

### Phase 0: Infrastructure Setup
**Goal**: Set up CMake integration for ComposableKernel and create file structure  
**Estimated Time**: 2-3 hours  
**Dependencies**: None

#### Tasks
- [ ] 0.1: Add ComposableKernel find module to CMake (`cmake/FindComposableKernel.cmake`)
- [ ] 0.2: Update `src/v2/CMakeLists.txt` to find and link CK when `HAVE_ROCM=ON`
- [ ] 0.3: Create empty file stubs:
  - `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.h`
  - `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp`
  - `src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip`
- [ ] 0.4: Create test file stub:
  - `tests/v2/unit/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp`
- [ ] 0.5: Verify CK headers are accessible with a simple compile test

#### Deliverables
- CMake successfully finds ComposableKernel
- Empty kernel files compile without errors
- Test executable links (even if tests are empty)

#### Verification
```bash
cmake -B build_v2_rocm -S src/v2 -DHAVE_ROCM=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build_v2_rocm --target ROCmQuantisedGemmKernel_CK.hip.o
```

---

### Phase 1: ROCmPackedWeights and Weight Conversion
**Goal**: Implement host-side weight packing (dequant → requant to INT8 + scales)  
**Estimated Time**: 3-4 hours  
**Dependencies**: Phase 0

#### Tasks
- [ ] 1.1: Define `ROCmPackedWeights` struct in header (mirrors `CUDAPackedWeights`)
  ```cpp
  struct ROCmPackedWeights {
      std::vector<int8_t> int8_data;  // [K × N] ColumnMajor
      std::vector<float> scales;       // [N] per-column scales
      int K = 0, N = 0;
      int8_t* d_int8_data = nullptr;
      float* d_scales = nullptr;
      int rocm_device_id = -1;
      bool uploaded = false;
      ~ROCmPackedWeights();
  };
  ```
- [ ] 1.2: Implement `packWeightsToROCm(const TensorBase* tensor, ROCmPackedWeights& out)`
  - Call `tensor->data()` to get dequantized FP32
  - Per-column max_abs finding
  - Symmetric quantization: `scale = max_abs / 127`, `int8 = round(fp32 / scale)`
- [ ] 1.3: Implement device upload function in HIP file:
  ```cpp
  bool rocmQuantGemm_uploadWeights(const int8_t* h_weights, const float* h_scales,
                                    int8_t** d_weights, float** d_scales,
                                    int K, int N, int device_id);
  ```
- [ ] 1.4: Implement `ROCmPackedWeights` destructor (free device memory)

#### Unit Tests
| Test Name | Description |
|-----------|-------------|
| `PackWeights_BasicQ8_0` | Pack Q8_0 tensor, verify INT8 values and scales |
| `PackWeights_IQ4_NL` | Pack IQ4_NL tensor, verify symmetric quantization |
| `PackWeights_Dimensions` | Verify K, N dimensions match original tensor |
| `PackWeights_ScaleRange` | Verify scales are positive and reasonable |
| `UploadWeights_DeviceMemory` | Upload to device, verify pointers non-null |

#### Deliverables
- `packWeightsToROCm()` produces correct INT8 + scales
- Device upload works without HIP errors

---

### Phase 2: Activation Quantization Kernel
**Goal**: Implement HIP kernel for per-row FP32→INT8 quantization  
**Estimated Time**: 2-3 hours  
**Dependencies**: Phase 0

#### Tasks
- [ ] 2.1: Implement `quantize_activations_kernel` in HIP:
  ```cpp
  __global__ void quantize_activations_kernel(
      const float* A_fp32,  // [M×K]
      int8_t* A_int8,       // [M×K] output
      float* scales_A,      // [M] output
      int M, int K);
  ```
  - One block per row
  - Shared memory reduction for max_abs
  - Quantize: `int8 = round(fp32 * 127 / max_abs)`
- [ ] 2.2: Implement wrapper function:
  ```cpp
  bool rocmQuantGemm_quantizeActivations(
      const float* d_A_fp32, int8_t* d_A_int8, float* d_scales_A,
      int M, int K, int device_id);
  ```
- [ ] 2.3: Add work buffer allocation function:
  ```cpp
  bool rocmQuantGemm_ensureWorkBuffers(
      int8_t** d_A_int8, float** d_scales_A, int32_t** d_C_int32,
      int* work_buffer_M, int M, int K, int N, int device_id);
  ```

#### Unit Tests
| Test Name | Description |
|-----------|-------------|
| `QuantizeActivations_Small` | 4×8 matrix, verify per-row scales |
| `QuantizeActivations_Large` | 512×1024 matrix, verify correctness |
| `QuantizeActivations_ZeroRow` | Row of zeros produces scale=1.0 |
| `QuantizeActivations_Range` | Verify INT8 values in [-127, 127] |
| `WorkBuffers_Allocation` | Verify buffer growth behavior |

#### Deliverables
- Activation quantization kernel produces correct INT8 + scales
- Work buffers allocate and resize correctly

---

### Phase 3: ComposableKernel GEMM Integration
**Goal**: Integrate CK `DeviceGemmDl` for INT8×INT8→INT32 GEMM  
**Estimated Time**: 4-5 hours  
**Dependencies**: Phase 0

#### Tasks
- [ ] 3.1: Create CK instance typedef in HIP file:
  ```cpp
  using DeviceGemmInt8 = ck::tensor_operation::device::DeviceGemmDl<
      ck::tensor_layout::gemm::RowMajor,     // ALayout
      ck::tensor_layout::gemm::ColumnMajor,  // BLayout  
      ck::tensor_layout::gemm::RowMajor,     // CLayout
      int8_t, int8_t, int32_t, int32_t,      // A, B, C, Acc types
      PassThrough, PassThrough, PassThrough,
      GemmDefault,
      /* tunable params - use CK defaults initially */
  >;
  ```
- [ ] 3.2: Implement GEMM execution wrapper:
  ```cpp
  bool rocmQuantGemm_execute(
      const int8_t* d_A_int8,       // [M×K] RowMajor
      const int8_t* d_weights_int8, // [K×N] ColumnMajor
      int32_t* d_C_int32,           // [M×N] RowMajor
      int M, int N, int K, int device_id);
  ```
- [ ] 3.3: Handle `IsSupportedArgument()` check and error reporting
- [ ] 3.4: Add architecture detection for gfx906 vs gfx908+ (future XDL support)

#### Unit Tests
| Test Name | Description |
|-----------|-------------|
| `CKGemm_SmallMatrix` | 16×16×16 GEMM, verify against CPU reference |
| `CKGemm_NonSquare` | 64×128×256 GEMM, verify correctness |
| `CKGemm_LargeMatrix` | 1024×2048×512 GEMM, verify correctness |
| `CKGemm_PaddedDimensions` | Non-aligned dims (e.g., 100×200×300) |
| `CKGemm_DeviceCheck` | Verify `IsSupportedArgument()` works |

#### Deliverables
- CK GEMM executes without errors
- Output matches CPU INT8×INT8→INT32 reference

---

### Phase 4: Output Scaling Kernel
**Goal**: Implement HIP kernel for INT32→FP32 scaling with per-row/col scales  
**Estimated Time**: 2-3 hours  
**Dependencies**: Phase 2, Phase 3

#### Tasks
- [ ] 4.1: Implement scaling kernel:
  ```cpp
  __global__ void apply_scaling_kernel(
      const int32_t* C_int32,  // [M×N]
      float* C_fp32,           // [M×N] output
      const float* scales_A,   // [M] row scales
      const float* scales_B,   // [N] column scales
      int M, int N,
      float alpha, float beta,
      const float* C_existing, // for beta != 0
      const float* bias);      // optional [N] bias
  ```
- [ ] 4.2: Implement wrapper function:
  ```cpp
  bool rocmQuantGemm_applyScaling(
      const int32_t* d_C_int32, float* d_C_fp32,
      const float* d_scales_A, const float* d_scales_B,
      int M, int N, float alpha, float beta,
      const float* d_C_existing, const float* d_bias,
      int device_id);
  ```
- [ ] 4.3: Optimize with 2D thread blocking (16×16)

#### Unit Tests
| Test Name | Description |
|-----------|-------------|
| `Scaling_BasicAlpha` | alpha=1.0, beta=0.0, verify output |
| `Scaling_WithBeta` | alpha=1.0, beta=0.5, accumulate |
| `Scaling_WithBias` | Add bias vector, verify broadcast |
| `Scaling_Precision` | Verify FP32 precision preserved |

#### Deliverables
- Scaling kernel produces correct FP32 output
- Alpha/beta/bias parameters work correctly

---

### Phase 5: ROCmQuantisedGemmKernel Class Implementation
**Goal**: Implement full ITensorGemm interface with type dispatch  
**Estimated Time**: 4-5 hours  
**Dependencies**: Phases 1-4

#### Tasks
- [ ] 5.1: Implement class constructors:
  - `ROCmQuantisedGemmKernel(const TensorBase* weights, int device_id)`
  - `ROCmQuantisedGemmKernel(ROCmPackedWeights* packed, int device_id)`
- [ ] 5.2: Implement `ensureWeightsConverted()` (lazy conversion + upload)
- [ ] 5.3: Implement type dispatch in `multiply_tensor()`:
  | A type | C type | Path |
  |--------|--------|------|
  | Q8_1 | FP32 | Extract INT8+scales → GEMM → scale |
  | FP32 | FP32 | Quantize A → GEMM → scale |
  | FP32 | Q8_1 | Quantize A → GEMM → requant |
- [ ] 5.4: Implement `multiply()` raw pointer interface
- [ ] 5.5: Implement `multiply_fused_tensor()` for multi-projection
- [ ] 5.6: Implement `multiply_activations()` (return false - not supported)
- [ ] 5.7: Implement `supports_device()` and `getKernelSnapshotInfo()`

#### Unit Tests
| Test Name | Description |
|-----------|-------------|
| `Kernel_ConstructFromTensor` | Create from Q8_0 tensor |
| `Kernel_ConstructFromPacked` | Create from pre-packed weights |
| `Kernel_MultiplyTensor_FP32` | FP32 input → FP32 output |
| `Kernel_MultiplyTensor_Q8_1` | Q8_1 input → FP32 output |
| `Kernel_MultiplyFused` | Multiple projections in one call |
| `Kernel_DeviceCheck` | `supports_device()` returns correct |

#### Deliverables
- Full ITensorGemm interface implemented
- All type dispatch paths work correctly

---

### Phase 6: KernelFactory Integration
**Goal**: Register ROCmQuantisedGemmKernel in KernelFactory for automatic selection  
**Estimated Time**: 2-3 hours  
**Dependencies**: Phase 5

#### Tasks
- [ ] 6.1: Add `createROCmQuantisedGemm()` to KernelFactory
- [ ] 6.2: Update `getOrCreateGemm()` to select ROCm kernel when:
  - Device is ROCm GPU
  - Weight tensor is quantized type
- [ ] 6.3: Implement ROCmPackedWeights caching in tensor `cache_` field
- [ ] 6.4: Update device type detection in factory

#### Unit Tests
| Test Name | Description |
|-----------|-------------|
| `Factory_CreatesROCmKernel` | Factory returns ROCmQuantisedGemmKernel on ROCm |
| `Factory_CachesPackedWeights` | Same tensor returns cached kernel |
| `Factory_DeviceSelection` | Correct kernel for CPU vs CUDA vs ROCm |

#### Deliverables
- KernelFactory automatically creates ROCmQuantisedGemmKernel
- Weight caching works correctly

---

### Phase 7: Integration Testing
**Goal**: Verify kernel works in full inference pipeline  
**Estimated Time**: 3-4 hours  
**Dependencies**: Phase 6

#### Tasks
- [ ] 7.1: Create integration test: single-layer GEMM with real model weights
- [ ] 7.2: Create integration test: multi-layer inference comparison (CPU vs ROCm)
- [ ] 7.3: Create parity test: ROCm vs CUDA output comparison (if both available)
- [ ] 7.4: Add benchmark test: throughput measurement on various sizes
- [ ] 7.5: Test with FusedQKVGEMMStage and FusedGateUpGEMMStage

#### Integration Tests
| Test Name | Description |
|-----------|-------------|
| `Integration_SingleLayerGEMM` | Load model weights, run one GEMM |
| `Integration_MultiLayerParity` | Compare CPU vs ROCm across layers |
| `Integration_FusedQKV` | Test with FusedQKVGEMMStage |
| `Integration_FusedGateUp` | Test with FusedGateUpGEMMStage |
| `Integration_EndToEnd` | Full Qwen2 inference comparison |

#### Deliverables
- ROCm kernel produces numerically correct results
- Integration with pipeline stages works

---

### Phase 8: Performance Optimization (Optional)
**Goal**: Tune CK parameters for gfx906 performance  
**Estimated Time**: 4-6 hours  
**Dependencies**: Phase 7

#### Tasks
- [ ] 8.1: Profile GEMM execution with `rocprof`
- [ ] 8.2: Test different CK tile configurations
- [ ] 8.3: Implement architecture-specific tuning (gfx906 vs gfx908+)
- [ ] 8.4: Add XDL kernel path for MI100/MI200/MI300
- [ ] 8.5: Fuse scaling into CK epilogue (custom `CDEElementOp`)

#### Performance Tests
| Test Name | Description |
|-----------|-------------|
| `Perf_SmallBatch` | M=1-8, throughput measurement |
| `Perf_LargeBatch` | M=128-512, throughput measurement |
| `Perf_MemoryBandwidth` | Measure effective bandwidth |
| `Perf_CompareToHipBLAS` | Compare against hipBLAS SGEMM |

#### Deliverables
- Performance within 80% of peak theoretical
- Architecture-specific optimizations applied

---

## Test Matrix

### Unit Tests (build_v2)

| Phase | Test File | Tests |
|-------|-----------|-------|
| 1 | `Test__ROCmQuantisedGemmKernel.cpp` | PackWeights_* (5 tests) |
| 2 | `Test__ROCmQuantisedGemmKernel.cpp` | QuantizeActivations_* (5 tests) |
| 3 | `Test__ROCmQuantisedGemmKernel.cpp` | CKGemm_* (5 tests) |
| 4 | `Test__ROCmQuantisedGemmKernel.cpp` | Scaling_* (4 tests) |
| 5 | `Test__ROCmQuantisedGemmKernel.cpp` | Kernel_* (6 tests) |
| 6 | `Test__KernelFactory.cpp` | Factory_ROCm* (3 tests) |

### Integration Tests (build_v2_integration)

| Phase | Test File | Tests |
|-------|-----------|-------|
| 7 | `Test__ROCmQuantisedGemmIntegration.cpp` | Integration_* (5 tests) |
| 8 | `Test__ROCmQuantisedGemmPerformance.cpp` | Perf_* (4 tests) |

---

## File Structure

```
src/v2/
├── kernels/
│   └── rocm/
│       ├── ROCmQuantisedGemmKernel.h          # Phase 5
│       ├── ROCmQuantisedGemmKernel.cpp        # Phase 5
│       └── ROCmQuantisedGemmKernel_CK.hip     # Phases 2-4
├── cmake/
│   └── FindComposableKernel.cmake             # Phase 0

tests/v2/
├── unit/
│   └── kernels/
│       └── rocm/
│           └── Test__ROCmQuantisedGemmKernel.cpp   # Phases 1-6
└── integration/
    └── kernels/
        └── rocm/
            ├── Test__ROCmQuantisedGemmIntegration.cpp  # Phase 7
            └── Test__ROCmQuantisedGemmPerformance.cpp  # Phase 8
```

---

## Dependencies

### External
- **ComposableKernel**: AMD's kernel library (included with ROCm 5.0+)
- **HIP**: AMD's GPU runtime (included with ROCm)
- **hipBLAS**: For reference comparisons (already integrated)

### Internal
- `ITensorGemm` interface (`tensors/TensorKernels.h`)
- `TensorBase` and quantized tensor types (`tensors/Tensors.h`)
- `KernelFactory` (`kernels/KernelFactory.h`)
- `ROCmBackend` (`backends/rocm/ROCmBackend.h`)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CK not installed on target system | Medium | High | Fallback to dequant→hipBLAS path |
| gfx906 INT8 performance poor | Medium | Medium | Acceptable for functional correctness |
| CK API changes | Low | Medium | Pin to specific CK version |
| Memory alignment issues | Low | High | Add alignment checks, pad buffers |
| Numerical precision loss | Low | Medium | Parity tests against CPU reference |

---

## Success Criteria

1. **Functional**: All unit and integration tests pass
2. **Parity**: Output matches CPU reference within tolerance (cosine sim > 0.999)
3. **Performance**: Throughput > 50% of hipBLAS FP32 GEMM on same hardware
4. **Integration**: Works with existing pipeline stages (FusedQKVGEMMStage, etc.)
5. **Maintainable**: Code structure mirrors CUDAQuantisedGemmKernel for easy comparison

---

## Appendix A: CK Instance Reference

### DeviceGemmDl (gfx906 compatible)

```cpp
// From: ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t K1,
          /* ... many more tunable params ... */>
struct DeviceGemmDl;
```

### Recommended Starting Configuration

```cpp
using DeviceGemmInt8Instance = ck::tensor_operation::device::DeviceGemmDl<
    int8_t, int8_t, int32_t, int32_t,
    ck::tensor_layout::gemm::RowMajor,
    ck::tensor_layout::gemm::ColumnMajor,
    ck::tensor_layout::gemm::RowMajor,
    PassThrough, PassThrough, PassThrough,
    ck::tensor_operation::device::GemmSpecialization::Default,
    256,   // BlockSize
    64,    // MPerBlock
    128,   // NPerBlock
    16,    // K0PerBlock
    4,     // K1
    /* use defaults for remaining params */
>;
```

---

## Appendix B: Memory Layout Reference

| Matrix | Layout | Dimension | Stride |
|--------|--------|-----------|--------|
| A (activations) | RowMajor | [M × K] | K |
| B (weights) | ColumnMajor | [K × N] | K |
| C (output) | RowMajor | [M × N] | N |
| scales_A | - | [M] | 1 |
| scales_B | - | [N] | 1 |

**Note**: CK's ColumnMajor for B means the K dimension is contiguous in memory, which matches our weight packing where we store `[N rows of K elements]` but interpret as `[K × N] ColumnMajor`.
