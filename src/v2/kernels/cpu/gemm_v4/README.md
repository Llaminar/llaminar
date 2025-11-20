# OneDNN GEMM Kernel (gemm_v4)

**Location**: `src/v2/kernels/cpu/gemm_v4/`  
**Status**: Production-ready multi-precision GEMM with INT8, FP32, BF16, and FP16 support  
**Performance**: ~480 GOPS on Qwen 0.5B FFN (8192×896×4864), 21% of theoretical peak

---

## Overview

The OneDNN GEMM kernel (`gemm_v4`) provides high-performance matrix multiplication using Intel's [oneDNN library](https://github.com/oneapi-src/oneDNN). It supports **multiple precision formats** (INT8, FP32, BF16, FP16) with **per-channel quantization** for INT8 paths, achieving near-FP32 accuracy while exploiting specialized compute throughput.

### Key Features

- ✅ **Multi-Precision Support**: INT8 (`s8×s8→s32`), FP32 (`f32×f32→f32`), BF16 (`bf16×bf16→f32`), FP16 (`f16×f16→f32`)
- ✅ **Typed GEMM Interface**: Template-based `multiply_activations_typed()` with compile-time type safety
- ✅ **Mixed Precision Operations**: FP32 activations × BF16/FP16 weights (attention context projection)
- ✅ **Strided Memory Access**: Zero-copy multi-head attention via `multiply_activations_strided_typed()`
- ✅ **Per-Channel Quantization**: Independent scales per activation row and weight column (INT8 path)
- ✅ **Fused GEMM + Softmax**:
    - INT8 path: `run_onednn_int8_matmul_with_softmax()` (INT8 matmul + numerically stable softmax)
    - FP32 path: `run_onednn_fp32_matmul_softmax()` with softmax post-op, falling back to host softmax
    - Typed path: `multiply_with_softmax_typed()` for native-precision attention scores
- ✅ **Zero-Copy Integration**: Direct tensor API without intermediate buffers (via `ActivationView`)
- ✅ **FP32 Reference Validation**: Automated correctness checks in performance tests
- ✅ **Adapter Pattern**: Seamless integration with V2 pipeline infrastructure (`OneDNNGemmKernel` + `onednn_gemm_adapter`)

---

## Architecture

### Component Hierarchy

```
OneDNNGemmKernel.h
├── onednn_engine()                          # Singleton oneDNN CPU engine
├── onednn_stream()                          # Singleton execution stream
│
├── Low-Level Primitives:
│   ├── run_onednn_int8_matmul()             # INT8: s8×s8→s32
│   ├── run_onednn_fp32_matmul()             # FP32: f32×f32→f32
│   ├── run_onednn_bf16_matmul()             # BF16: bf16×bf16→f32 (AVX512_BF16)
│   ├── run_onednn_fp16_matmul()             # FP16: f16×f16→f32 (AVX512_FP16)
│   ├── run_onednn_mixed_bf16_matmul()       # Mixed: f32×bf16→f32 (scores@V)
│   └── run_onednn_mixed_fp16_matmul()       # Mixed: f32×fp16→f32 (scores@V)
│
├── Fused Operations:
│   ├── run_onednn_fp32_matmul_softmax()     # FP32 matmul + softmax post-op
│   ├── run_onednn_int8_matmul_with_softmax()# INT8 matmul + host softmax
│   └── apply_softmax_inplace()              # Vectorized softmax (libmvec)
│
├── Helper Classes:
│   └── ActivationView                       # Lightweight IActivationTensor view
│
└── OneDNNGemmKernel (ITensorGemm):
    ├── multiply()                            # Weight GEMM (INT8 path)
    ├── multiply_activations()                # Activation GEMM (FP32)
    ├── multiply_activations_strided()        # Strided GEMM (FP32)
    ├── multiply_activations_typed_impl()     # Type-erased typed GEMM
    ├── multiply_activations_strided_typed_impl()  # Type-erased strided typed
    ├── multiply_with_softmax()               # Weight GEMM + softmax
    ├── multiply_with_softmax_typed_impl()    # Type-erased softmax GEMM
    └── multiply_with_softmax_strided_typed_impl() # Type-erased strided softmax

OneDNNGemmAdapter.h
├── pack_weights_to_int8()        # Weight tensor → WeightPack (col-major INT8)
├── onednn_gemm_from_packed()     # Core INT8 GEMM + scale application
└── onednn_gemm_adapter()         # High-level pipeline interface (INT8 GEMM)
```

### Data Flow

```
IActivationTensor::createGemm()
    ↓
onednn_gemm_adapter(M, N, K, activation, weight, output)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. Quantize Activations (FP32/BF16/FP16 → INT8)            │
│    activation.to_int8_activation_pack(M, K)                 │
│    → ActivationPack { int8[M×K], row_scales[M] }            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Quantize Weights (Q8_0/Q6_K/IQ4_NL → INT8)              │
│    weight.to_int8_perchannel(K, N)                          │
│    → WeightPack { int8[K×N], col_scales[N] }               │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. OneDNN INT8 GEMM                                         │
│    dnnl::matmul(s8[M×K], s8[K×N]) → s32[M×N]               │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Dequantize Output (INT32 → Format-Preserving)           │
│    output.from_int32_with_scales(accum, row_scales,        │
│                                  col_scales, bias)          │
│    ✅ Preserves output tensor format (BF16/FP16/FP32/INT32) │
└─────────────────────────────────────────────────────────────┘

For **fused softmax** paths, the data flow extends with an additional step:

```text
// INT8 attention scores: QK^T with softmax
run_onednn_int8_matmul_with_softmax(A_int8, B_int8, scores, params)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3b. INT8 GEMM + Softmax                                      │
│     - INT8 matmul → INT32 scores                            │
│     - Row/column-wise scale + bias application              │
│     - Numerically stable softmax (libmvec-accelerated)      │
└─────────────────────────────────────────────────────────────┘

// FP32 attention scores (fallback or non-INT8 path)
run_onednn_fp32_matmul_softmax(A_fp32, B_fp32, scores, M, N, K, axis)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3c. FP32 GEMM + oneDNN softmax post-op                      │
│     - If unsupported axis, falls back to FP32 matmul +      │
│       `apply_softmax_inplace()` on the host.                │
└─────────────────────────────────────────────────────────────┘
```
```

---

## API Reference

### High-Level Pipeline Interface

```cpp
#include "kernels/cpu/gemm_v4/OneDNNGemmAdapter.h"
using namespace llaminar2::gemm_v4;

// Primary adapter used by pipelines
bool onednn_gemm_adapter(
    int M,                          // Number of output rows
    int N,                          // Number of output columns  
    int K,                          // Shared dimension (activation cols / weight rows)
    const IActivationTensor &A,     // Activation tensor (FP32/BF16/FP16/Q8_1)
    const TensorBase &B,            // Weight tensor (Q8_0/Q6_K/IQ4_NL/etc.)
    IActivationTensor &output,      // Destination tensor (format preserved: FP32/BF16/FP16/INT32)
    const float *bias = nullptr     // Optional bias vector [N]
);
```

**Returns**: `true` on success, `false` on failure  
**Thread Safety**: Thread-safe (uses thread-local accumulator buffer)  
**Memory**: No dynamic allocation per call (reuses static `accum_buffer`)  
**Format Preservation**: Output tensor format is preserved (BF16→BF16, FP16→FP16, FP32→FP32, INT32→INT32)

### ITensorGemm Interface (OneDNNGemmKernel)

`OneDNNGemmKernel` implements `ITensorGemm` and routes its methods to the adapter and fused helpers:

```cpp
class OneDNNGemmKernel : public ITensorGemm {
public:
    // Bound to a specific weight tensor (B matrix)
    explicit OneDNNGemmKernel(const TensorBase *weight_tensor);

    // INT8 weight GEMM: C = A @ B^T
    bool multiply(const float *A, float *C,
                  int m, int n, int k,
                  bool transpose_B = true,
                  float alpha = 1.0f,
                  float beta = 0.0f,
                  const MPIContext *mpi_ctx = nullptr,
                  int device_idx = -1) override;

    // FP32 activation GEMM: C = A @ B^T
    bool multiply_activations(const float *A, const float *B, float *C,
                              int m, int n, int k,
                              bool transpose_B = true,
                              float alpha = 1.0f,
                              float beta = 0.0f,
                              const MPIContext *mpi_ctx = nullptr,
                              int device_idx = -1) override;

    // Strided variant (copies into contiguous buffers when needed)
    bool multiply_activations_strided(const float *A, const float *B, float *C,
                                      int m, int n, int k,
                                      int lda, int ldb, int ldc,
                                      bool transpose_B = true,
                                      float alpha = 1.0f,
                                      float beta = 0.0f,
                                      const MPIContext *mpi_ctx = nullptr,
                                      int device_idx = -1) override;

    // Typed activation GEMM (template interface)
    template <typename ActT, typename WeightT>
    bool multiply_activations_typed(const ActT *A, const WeightT *B, float *C,
                                   int m, int n, int k,
                                   bool transpose_B = true,
                                   float alpha = 1.0f, float beta = 0.0f,
                                   const MPIContext *mpi_ctx = nullptr,
                                   int device_idx = -1,
                                   ActivationFormat format = ActivationFormat::FP32);

    // Type-erased implementation for typed GEMM
    bool multiply_activations_typed_impl(const void *A, const void *B, float *C,
                                        int m, int n, int k,
                                        bool transpose_B,
                                        float alpha, float beta,
                                        const MPIContext *mpi_ctx,
                                        int device_idx,
                                        ActivationFormat format_A,
                                        ActivationFormat format_B) override;

    // Typed strided activation GEMM (template interface)
    template <typename ActT, typename WeightT>
    bool multiply_activations_strided_typed(const ActT *A, const WeightT *B, float *C,
                                           int m, int n, int k,
                                           int lda, int ldb, int ldc,
                                           bool transpose_B = true,
                                           float alpha = 1.0f, float beta = 0.0f,
                                           const MPIContext *mpi_ctx = nullptr,
                                           int device_idx = -1,
                                           ActivationFormat format = ActivationFormat::FP32);

    // Type-erased implementation for strided typed GEMM
    bool multiply_activations_strided_typed_impl(const void *A, const void *B, float *C,
                                                int m, int n, int k,
                                                int lda, int ldb, int ldc,
                                                bool transpose_B,
                                                float alpha, float beta,
                                                const MPIContext *mpi_ctx,
                                                int device_idx,
                                                ActivationFormat format_A,
                                                ActivationFormat format_B) override;

    // Fused GEMM + softmax on weights: C = softmax(A @ B^T)
    bool multiply_with_softmax(const float *A, float *C,
                               int m, int n, int k,
                               bool transpose_B = true,
                               int softmax_axis = 1,
                               const MPIContext *mpi_ctx = nullptr,
                               int device_idx = -1) override;

    // Template-based typed fused matmul + softmax
    template <typename ActT, typename WeightT>
    bool multiply_with_softmax_typed(const ActT *A, const WeightT *B, float *C,
                                    int m, int n, int k,
                                    float scale = 1.0f,
                                    bool transpose_B = true,
                                    int softmax_axis = 1,
                                    const float *mask = nullptr,
                                    bool is_causal = false,
                                    const MPIContext *mpi_ctx = nullptr,
                                    int device_idx = -1,
                                    ActivationFormat format = ActivationFormat::FP32);

    // Type-erased implementation for typed softmax GEMM
    bool multiply_with_softmax_typed_impl(const void *A, const void *B, float *C,
                                         int m, int n, int k,
                                         float scale,
                                         bool transpose_B,
                                         int softmax_axis,
                                         const float *mask,
                                         bool is_causal,
                                         const MPIContext *mpi_ctx,
                                         int device_idx,
                                         ActivationFormat format_A,
                                         ActivationFormat format_B) override;

    // Template-based strided typed fused matmul + softmax
    template <typename ActT, typename WeightT>
    bool multiply_with_softmax_strided_typed(const ActT *A, const WeightT *B, float *C,
                                            int m, int n, int k,
                                            int lda, int ldb, int ldc,
                                            float scale = 1.0f,
                                            bool transpose_B = true,
                                            int softmax_axis = 1,
                                            const float *mask = nullptr,
                                            bool is_causal = false,
                                            const MPIContext *mpi_ctx = nullptr,
                                            int device_idx = -1,
                                            ActivationFormat format = ActivationFormat::FP32);

    // Type-erased implementation for strided typed softmax GEMM
    bool multiply_with_softmax_strided_typed_impl(const void *A, const void *B, float *C,
                                                 int m, int n, int k,
                                                 int lda, int ldb, int ldc,
                                                 float scale,
                                                 bool transpose_B,
                                                 int softmax_axis,
                                                 const float *mask,
                                                 bool is_causal,
                                                 const MPIContext *mpi_ctx,
                                                 int device_idx,
                                                 ActivationFormat format_A,
                                                 ActivationFormat format_B) override;
};
```

**Precision Dispatch Logic**:
- **FP32×FP32**: `run_onednn_fp32_matmul()` (universal fallback)
- **BF16×BF16**: `run_onednn_bf16_matmul()` (AVX512_BF16 required, falls back to FP32)
- **FP16×FP16**: `run_onednn_fp16_matmul()` (AVX512_FP16 required, falls back to FP32)
- **FP32×BF16**: `run_onednn_mixed_bf16_matmul()` (attention context: scores@V)
- **FP32×FP16**: `run_onednn_mixed_fp16_matmul()` (attention context: scores@V)
- **INT8×INT8**: `run_onednn_int8_matmul()` (via adapter, quantization path)

**Strided Operations**: `multiply_activations_strided_typed_impl()` detects row-major layouts and falls back to contiguous copy when strides don't match expected dimensions.

### Tensor Conversion Methods

All activation and weight tensors provide standardized conversion interfaces:

#### IActivationTensor Interface

```cpp
class IActivationTensor {
public:
    // Quantize activations to INT8 with per-row scales
    virtual ActivationPack to_int8_activation_pack(int rows, int cols) const = 0;
};

struct ActivationPack {
    std::vector<int8_t> data;        // INT8 tensor [rows×cols] (row-major)
    std::vector<float> row_scales;   // Per-row scales [rows]
    int rows, cols;
};
```

**Implemented by**: `FP32Tensor`, `BF16Tensor`, `FP16Tensor`, `Q8_1Tensor`, `INT32Tensor`

#### IActivationTensor Interface (Output Conversion)

```cpp
class IActivationTensor {
public:
    // Convert INT32 accumulator to output tensor format
    virtual bool from_int32_with_scales(
        const int32_t *src_int32,       // INT32 accumulator [rows×cols]
        const float *row_scales,        // Per-row scales [rows]
        const float *col_scales,        // Per-column scales [cols]
        const float *bias = nullptr     // Optional bias [cols]
    ) = 0;
};
```

**Implemented by**: `FP32Tensor`, `BF16Tensor`, `FP16Tensor`, `INT32Tensor`  
**Format Preservation**: Each tensor type converts INT32→native format (e.g., BF16Tensor outputs BF16)

#### TensorBase Interface (Weights)

```cpp
class TensorBase {
public:
    // Quantize weights to INT8 with per-column scales
    virtual bool to_int8_perchannel(
        int8_t *dst_int8,           // Output INT8 tensor [rows×cols]
        float *dst_col_scales,      // Per-column scales [cols]
        float *dst_row_scales = nullptr  // Optional per-row scales [rows]
    ) const = 0;
};
```

**Implemented by**: All quantized weight tensors (`Q8_0Tensor`, `Q6_KTensor`, `IQ4_NLTensor`, etc.)

---

## Integration with V2 Pipelines

### Pattern 1: Direct Adapter Call

```cpp
// In Qwen2Pipeline::forward()
auto Q_proj = weights_.wq[layer];   // Q8_0Tensor weight
auto hidden = activation_buffer;    // FP32Tensor activation
auto &q_proj_output = workspace.q[layer]; // BF16Tensor scratch [seq_len, d_model]

bool success = onednn_gemm_adapter(
    seq_len,        // M
    d_model,        // N  
    d_model,        // K
    *hidden,        // IActivationTensor
    *Q_proj,        // TensorBase
    q_proj_output   // Destination tensor (BF16 preserved: INT32→BF16 conversion)
);
```

### Pattern 2: ITensorGemm Interface (Future)

```cpp
// Planned: Kernel creation from activation tensor
auto gemm = activation->createGemm();  // Returns OneDNNGemmKernel
gemm->execute(weight, output, {...});
```

**Current Status**: Direct adapter calls are used throughout pipelines. Future refactoring may introduce `createGemm()` factory method for polymorphic kernel selection.

---

## Quantization Strategy

### Per-Channel vs Per-Tensor

The OneDNN kernel uses **per-channel quantization** (separate scales per row/column) rather than per-tensor (single global scale):

**Advantages**:
- ✅ Higher accuracy: Adapts to per-channel magnitude variance
- ✅ Eliminates outlier clipping: Each channel scaled independently
- ✅ Preserves dynamic range: No loss from global normalization

**Trade-offs**:
- ❌ Slightly higher memory: N+M scales vs 1-2 global scales
- ❌ Dequant overhead: Per-element scale multiplication (mitigated by vectorization)

### Scale Computation (Symmetric Quantization)

```cpp
// Activation row scales (max absolute value per row)
for (int i = 0; i < M; ++i) {
    float max_abs = max(|A[i,:]|);
    row_scales[i] = max_abs / 127.0f;  // INT8 range: [-127, 127]
}

// Weight column scales (max absolute value per column)
for (int j = 0; j < N; ++j) {
    float max_abs = max(|B[:,j]|);
    col_scales[j] = max_abs / 127.0f;
}

// Quantization
A_int8[i,k] = round(A[i,k] / row_scales[i]);
B_int8[k,j] = round(B[k,j] / col_scales[j]);

// Dequantization
C[i,j] = accum_int32[i,j] × row_scales[i] × col_scales[j];
```

**Format Conversion**: The `from_int32_with_scales()` method applies the scale multiplication and converts to the output tensor's native format:
- **FP32Tensor**: Direct FP32 computation
- **BF16Tensor**: FP32 intermediate → BF16 rounding
- **FP16Tensor**: FP32 intermediate → FP16 rounding
- **INT32Tensor**: Stores raw INT32 accumulator (scales stored separately)

This enables memory-efficient pipelines (e.g., BF16 activations throughout) while maintaining INT8 compute throughput.

---

## Performance Characteristics

### Benchmark Results (Qwen 0.5B FFN Gate)

**Configuration**: M=8192, N=4864, K=896 (Qwen 2.5 0.5B layer 0 FFN gate projection)  
**Hardware**: 2×28-core Intel Xeon (56 physical cores)  
**Threads**: 28 threads (OMP_NUM_THREADS=28, single socket)

**Throughput**: 480 GOPS (21% of 2240 GOPS theoretical peak)  
**Latency**: 148ms average (50 iterations)  
**Accuracy** (INT8 vs FP32): Max abs diff: `≈1e-5`, Relative L2: `<0.0001%`

### Scaling Characteristics

| Matrix Size (M×N×K) | Throughput | Efficiency | Notes |
|---------------------|------------|------------|-------|
| 8192×4864×896       | 480 GOPS   | 21%        | Large prefill (production) |
| 1024×4864×896       | ~350 GOPS  | 16%        | Medium batch |
| 1×4864×896          | ~80 GOPS   | 4%         | Single token (decode) |

**Key Observations**:
- **Prefill-optimized**: Best efficiency at M≥1024 (batch or long sequences)
- **Cache-friendly**: K=896 fits in L2, minimizes memory bottleneck
- **Thread scaling**: Near-linear up to 28 threads, diminishing returns beyond
 - **Fused softmax**: INT8 + softmax path is competitive with FP32 scores,
     while the FP32 fused primitive saves a separate softmax kernel launch.

---

## Correctness Validation

### FP32 Reference Comparison

All performance tests include automated FP32 reference validation:

```cpp
// tests/v2/performance/cpu/kernels/gemm_v4/Perf__OneDNNGemm_QwenProfile.cpp

// 1. Run INT8 GEMM
onednn_gemm_from_packed(activation_pack, weight_pack, output, ...);

// 2. Reconstruct FP32 baseline from packs
for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
        A_fp32[m,k] = activation_pack.data[m,k] * activation_pack.row_scales[m];
    }
}
// (same for weights)

// 3. Run FP32 reference
run_onednn_fp32_matmul(A_fp32, B_fp32, C_fp32, M, N, K);

// 4. Compare and assert
max_abs_diff = max(|output - C_fp32|);
rel_l2 = ||output - C_fp32||_2 / ||C_fp32||_2;

ASSERT_LT(max_abs_diff, 0.5);       // Absolute error guard
ASSERT_LT(rel_l2, 0.05);            // Relative L2 guard (5%)
```

**Why Reconstruct from Packs?**  
We dequantize the INT8 packs rather than using original FP32 tensors to ensure the baseline represents **exactly what the INT8 path sees**. This eliminates false positives from quantization noise vs computation errors.

### Running Validation Tests

```bash
cd /workspaces/llaminar

# Build and run performance test with validation
./run_benchmark.sh v2_perf_onednn_gemm_qwen_profile

# Expected output:
# Step 1: OneDNN GEMM
#   Average time:  148.19 ms
#   Throughput:    481.83 GOPS
#   Efficiency vs 2240 GOPS peak: 21.51%
#   Adapter verification: PASS (matches packed execution)
# Step 2: FP32 reference comparison
#   Max abs diff:  0.000011
#   Relative L2:   0.0000%
#   FP32 reference check: PASS
```

---

## Build Configuration

### CMake Dependencies

```cmake
# OneDNN library (required)
find_package(DNNL REQUIRED)

# Compiler flags
target_compile_definitions(llaminar2_core PRIVATE HAVE_ONEDNN)
target_link_libraries(llaminar2_core PRIVATE DNNL::dnnl)
```

### Installation (Ubuntu/Debian)

```bash
# Install OneDNN from system packages
sudo apt-get install libdnnl-dev

# Or build from source
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```

### Verification

```bash
# Check OneDNN availability
pkg-config --modversion dnnl

# Build llaminar with OneDNN
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Run performance test (REQUIRED: use CTest for stable environment)
cd /workspaces/llaminar
ctest --test-dir build_v2_release -R "V2_Perf_OneDNNGemm_QwenProfile" --verbose
```

**⚠️ IMPORTANT**: Performance tests **MUST** be run via `ctest` to ensure:
- Proper CPU core binding (`--bind-to socket --map-by socket`)
- Optimal OpenMP thread configuration (`OMP_NUM_THREADS=28`, `OMP_PLACES=sockets`)
- Consistent BLAS library threading (`OPENBLAS_NUM_THREADS=28`)
- MPI environment setup for memory access patterns

Running the executable directly will produce **inconsistent and degraded performance** (e.g., 996-1182 GOPS) compared to CTest's stable environment (1456 GOPS, 65% efficiency).

**Expected Performance** (via CTest):
- **Throughput**: ~1456 GOPS on M=8192, N=4864, K=896 (Qwen 0.5B FFN)
- **Efficiency**: 65% of 2240 GOPS theoretical peak
- **Accuracy**: <0.00001 max absolute error vs FP32 reference

---

## Precision Support Matrix

| Operation | FP32 | BF16 | FP16 | Mixed (FP32×BF16/FP16) | INT8 |
|-----------|------|------|------|------------------------|------|
| `multiply_activations()` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `multiply_activations_typed()` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `multiply_activations_strided_typed()` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `multiply_with_softmax_typed()` | ✅ | ✅ | ✅ | ❌ | ❌ |
| `multiply()` (weight GEMM) | ❌ | ❌ | ❌ | ❌ | ✅ |

**Hardware Requirements**:
- **BF16**: AVX512_BF16 (Sapphire Rapids, Zen 4+)
- **FP16**: AVX512_FP16 (Sapphire Rapids) or emulation fallback
- **INT8**: AVX512-VNNI (Cascade Lake+) or scalar fallback

**Fallback Behavior**:
- Missing BF16/FP16 support → Convert to FP32, run FP32 GEMM
- Missing AVX512-VNNI → Scalar INT8 accumulation (significant slowdown)

---

## Limitations and Future Work

### Current Limitations

1. **CPU-Only**: No GPU backend (use CUDA/ROCm kernels for GPU execution)
2. **No Bias Fusion**: Bias addition happens in dequant step (not fused into OneDNN primitive)
3. **Static Threading**: Thread count set at pipeline init (no dynamic adjustment)
4. **INT8 Range**: Clamps to `[-127, 127]` vs full `[-128, 127]` to avoid asymmetry
5. **Strided Overhead**: Non-row-major layouts require copy to contiguous buffer
6. **Softmax Axis**: OneDNN softmax post-op only supports last dimension (falls back to host for other axes)

### Planned Enhancements

- [ ] **OneDNN Bias Fusion**: Add bias as post-op to matmul descriptor (eliminate dequant-stage fusion)
- [x] **BF16 Path**: Leverage BF16 matmul for models with BF16 weights (✅ implemented)
- [x] **FP16 Path**: Leverage FP16 matmul for FP16 activations (✅ implemented)
- [ ] **Kernel Caching**: Reuse compiled primitives across calls (JIT amortization)
- [ ] **INT4 Support**: Add IQ4_NL direct path (bypass INT8 expansion)
- [ ] **Async Execution**: Non-blocking GEMM for pipeline overlap
- [ ] **In-Place Strided**: Avoid copy overhead by using oneDNN's strided memory descriptors

---

## See Also

- **V2 Architecture**: `.github/instructions/llaminar-v2-architecture.instructions.md`
- **Tensor API**: `src/v2/tensors/Tensors.h` (conversion methods)
- **VNNI Kernel** (gemm_v3): `src/v2/kernels/cpu/gemm_v3/` (AVX512-VNNI alternative)
- **OneDNN Docs**: https://oneapi-src.github.io/oneDNN/

---

## Authors

- **David Sanftenberg** - Initial implementation and validation framework
- **OneDNN Team** - Underlying matrix multiplication primitives

**Last Updated**: November 20, 2025
