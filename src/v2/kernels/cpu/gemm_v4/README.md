# OneDNN GEMM Kernel (gemm_v4)

**Location**: `src/v2/kernels/cpu/gemm_v4/`  
**Status**: Production-ready INT8 GEMM with FP32 reference validation  
**Performance**: ~480 GOPS on Qwen 0.5B FFN (8192×896×4864), 21% of theoretical peak

---

## Overview

The OneDNN GEMM kernel (`gemm_v4`) provides high-performance INT8 matrix multiplication using Intel's [oneDNN library](https://github.com/oneapi-src/oneDNN). It leverages **per-channel quantization** with separate activation (row) and weight (column) scales to achieve near-FP32 accuracy while exploiting INT8 compute throughput.

### Key Features

- ✅ **INT8 Compute**: Uses `dnnl::matmul` with `s8×s8→s32` accumulation
- ✅ **Per-Channel Quantization**: Independent scales per activation row and weight column
- ✅ **Zero-Copy Integration**: Direct tensor API without intermediate buffers
- ✅ **FP32 Reference Validation**: Automated correctness checks in performance tests
- ✅ **Adapter Pattern**: Seamless integration with V2 pipeline infrastructure

---

## Architecture

### Component Hierarchy

```
OneDNNGemmAdapter.h
├── pack_weights_to_int8()        # Weight tensor → WeightPack (col-major INT8)
├── onednn_gemm_from_packed()     # Core INT8 GEMM + scale application
└── onednn_gemm_adapter()         # High-level pipeline interface

OneDNNGemm.h
├── onednn_engine()               # Singleton oneDNN CPU engine
├── onednn_stream()               # Singleton execution stream
├── run_onednn_int8_matmul()      # Low-level s8×s8→s32 primitive
└── run_onednn_fp32_matmul()      # FP32 reference GEMM (validation only)
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
**Accuracy**: Max abs diff vs FP32: `1.1e-5`, Relative L2: `<0.0001%`

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
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2 --parallel

# Run tests
cd build_v2 && ctest -R V2_Perf_OneDNNGemm_QwenProfile -V
```

---

## Limitations and Future Work

### Current Limitations

1. **CPU-Only**: No GPU backend (use CUDA/ROCm kernels for GPU execution)
2. **No Bias Fusion**: Bias addition happens in dequant step (not fused into OneDNN primitive)
3. **Static Threading**: Thread count set at pipeline init (no dynamic adjustment)
4. **INT8 Range**: Clamps to `[-127, 127]` vs full `[-128, 127]` to avoid asymmetry

### Planned Enhancements

- [ ] **OneDNN Bias Fusion**: Add bias as post-op to matmul descriptor (eliminate dequant-stage fusion)
- [ ] **BF16 Path**: Leverage BF16 matmul for models with BF16 weights (bypass INT8 quantization)
- [ ] **Kernel Caching**: Reuse compiled primitives across calls (JIT amortization)
- [ ] **INT4 Support**: Add IQ4_NL direct path (bypass INT8 expansion)
- [ ] **Async Execution**: Non-blocking GEMM for pipeline overlap

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

**Last Updated**: November 17, 2025
