# Precision Architecture in Llaminar V2

**Status**: Implementation in Progress (2025-01-24)  
**Author**: David Sanftenberg

## Overview

Llaminar V2 supports **mixed-precision inference** with native precision paths for FP32, BF16, FP16, and INT8 (quantized) formats. This document describes the architecture, supported precision combinations, and performance recommendations.

## Architectural Principles

### 1. Activation Precision vs Weight Precision

**Activation Precision**: The numerical format of intermediate activation buffers (hidden states, Q/K/V, attention scores, FFN outputs).

**Weight Precision**: The numerical format of model weights (embeddings, Q/K/V projections, FFN weights, LM head).

**Key Design Rule**: Activation and weight precisions are **independently configurable** but subject to hardware and performance constraints.

### 2. Native Precision Paths

Llaminar V2 provides **native-precision GEMM kernels** to avoid unnecessary format conversions:

| Activation Format | Weight Format | GEMM Kernel | OneDNN Primitive | Accumulator |
|-------------------|---------------|-------------|------------------|-------------|
| FP32 | FP32 | `run_onednn_fp32_matmul()` | `f32f32f32` | FP32 |
| BF16 | BF16 | `run_onednn_bf16_matmul()` | `bf16bf16f32` | FP32 |
| FP16 | FP16 | `run_onednn_fp16_matmul()` | `fp16fp16f32` | FP32 |
| FP32/BF16/FP16 | Q8_0/IQ4_NL | `onednn_gemm_adapter()` | `int8int8s32` | INT32 |
| INT8 (Q8_1) | Q8_0/IQ4_NL | `run_onednn_int8_matmul()` | `int8int8s32` | INT32 |

**Performance**: Native precision paths avoid conversion overhead and leverage hardware acceleration (AVX512_BF16, AVX512-VNNI, etc.).

### 3. Template-Based GEMM Interface

`ITensorGemm` now provides **typed GEMM methods** for precision-aware computation:

```cpp
// Template-based typed GEMM (new interface)
template <typename ActT, typename WeightT>
bool multiply_activations_typed(
    const ActT *A, const WeightT *B, float *C,
    int m, int n, int k,
    bool transpose_B = true,
    float alpha = 1.0f, float beta = 0.0f,
    const MPIContext *mpi_ctx = nullptr,
    int device_idx = -1);

// Legacy FP32-only GEMM (maintained for compatibility)
bool multiply_activations(
    const float *A, const float *B, float *C,
    int m, int n, int k,
    bool transpose_B = true,
    float alpha = 1.0f, float beta = 0.0f,
    const MPIContext *mpi_ctx = nullptr,
    int device_idx = -1);
```

**Usage in CpuAttentionKernelT**:
```cpp
// Template instantiation determines precision at compile time
template <typename TensorType>
class CpuAttentionKernelT : public ITensorAttention {
    using ElementType = typename ActivationTraits<TensorType>::ElementType;

    // Q@K^T with typed GEMM (no FP32 conversion!)
    gemm->multiply_activations_typed<ElementType, ElementType>(
        Q_typed, K_typed, scores,  // ElementType inputs, float output
        seq_len, seq_len, head_dim,
        true, scale, 0.0f, nullptr, -1);
};
```

## Supported Precision Combinations

### Tier 1: Native Precision (Optimal Performance)

**Recommended configurations** with zero conversion overhead:

| Configuration | Activation | Weight | GEMM Path | Performance | Use Case |
|---------------|------------|--------|-----------|-------------|----------|
| **FP32 Native** | FP32 | FP32 | FP32×FP32→FP32 | Baseline (100%) | Debugging, high precision |
| **BF16 Native** | BF16 | BF16 | BF16×BF16→FP32 | 1.5-2× faster | AVX512_BF16 CPUs |
| **FP16 Native** | FP16 | FP16 | FP16×FP16→FP32 | 1.5-2× faster | AVX512_FP16 CPUs |
| **INT8 Native** | Q8_1 | Q8_0/IQ4_NL | INT8×INT8→INT32 | 2-4× faster | AVX512-VNNI CPUs |

**Hardware Requirements**:
- BF16: AVX512_BF16 (Intel Cooper Lake+, AMD Zen 4+)
- FP16: AVX512_FP16 (Intel Sapphire Rapids+)
- INT8: AVX512-VNNI (Intel Cascade Lake+, AMD Zen 4+)

### Tier 2: Quantized Weights (Mixed Precision)

**Quantized weights with FP32/BF16/FP16 activations**:

| Configuration | Activation | Weight | GEMM Path | Performance | Conversion Overhead |
|---------------|------------|--------|-----------|-------------|---------------------|
| FP32 Act + Q8_0 Weight | FP32 | Q8_0 | INT8×INT8→INT32 | 1.5-3× faster | Per-row quant (low) |
| BF16 Act + Q8_0 Weight | BF16 | Q8_0 | INT8×INT8→INT32 | 1.8-3.5× faster | BF16→FP32→INT8 (medium) |
| FP16 Act + Q8_0 Weight | FP16 | Q8_0 | INT8×INT8→INT32 | 1.8-3.5× faster | FP16→FP32→INT8 (medium) |
| FP32 Act + IQ4_NL Weight | FP32 | IQ4_NL | INT8×INT8→INT32 | 2-4× faster | Per-row quant + weight decode |

**Rationale**: Quantized weights provide 4-8× memory reduction with minimal accuracy loss. Activations are quantized **on-the-fly** per GEMM call using `to_int8_activation_pack()`.

### Tier 3: Unsupported/Discouraged Combinations

**Avoid these configurations** (high conversion overhead, no performance benefit):

| Configuration | Activation | Weight | Why Discouraged |
|---------------|------------|--------|-----------------|
| ❌ INT8 Act + FP32 Weight | Q8_1 | FP32 | Forced FP32→INT8 weight conversion (expensive) |
| ❌ BF16 Act + FP16 Weight | BF16 | FP16 | Cross-format conversion, no hardware path |
| ❌ FP32 Act + BF16 Weight | FP32 | BF16 | BF16→FP32 weight conversion (negates precision) |

**Recommendation**: If activations are quantized (INT8), use quantized weights (Q8_0/IQ4_NL). If weights are FP32/BF16/FP16, match activation precision.

## Precision Validation and Defaults

### Command-Line Configuration (Future Work)

**Proposed CLI flags** (to be implemented in ArgumentParser):

```bash
# Activation precision (controls intermediate buffers)
--activation-precision fp32|bf16|fp16|int8  (default: fp32)

# Weight precision (determined by GGUF model file)
# --weight-precision is READ-ONLY (no CLI flag, inferred from model)

# Example: BF16 activations with quantized weights
./llaminar2 --model qwen2.5-0.5b-q8_0.gguf --activation-precision bf16

# Example: Native FP32 precision
./llaminar2 --model qwen2.5-0.5b-fp32.gguf --activation-precision fp32

# Example: INT8 activations with INT8 weights
./llaminar2 --model qwen2.5-0.5b-q8_0.gguf --activation-precision int8
```

### Automatic Precision Validation

**Policy enforcement** (to be implemented in pipeline initialization):

```cpp
// Pseudocode: Precision policy validation
struct PrecisionPolicy {
    ActivationFormat activation_format;
    TensorType weight_format;

    bool is_valid() const {
        // Tier 1: Native precision (always valid)
        if (activation_format == FP32 && weight_format == FP32) return true;
        if (activation_format == BF16 && weight_format == BF16) return true;
        if (activation_format == FP16 && weight_format == FP16) return true;
        if (activation_format == INT8 && is_quantized(weight_format)) return true;

        // Tier 2: Quantized weights (always valid)
        if (is_quantized(weight_format) && 
            (activation_format == FP32 || activation_format == BF16 || activation_format == FP16))
            return true;

        // Tier 3: Discouraged (warn but allow)
        LOG_WARN("Unsupported precision combination: activation=" << activation_format 
                 << " weight=" << weight_format);
        LOG_WARN("Performance may be suboptimal. Consider matching precisions.");
        return true; // Allow with warning
    }

    std::string get_recommendation() const {
        if (activation_format == INT8 && !is_quantized(weight_format)) {
            return "Use quantized weights (Q8_0/IQ4_NL) with INT8 activations";
        }
        if (is_quantized(weight_format) && activation_format == INT8) {
            return "Configuration valid: INT8 fast path enabled";
        }
        // ... more recommendations
    }
};
```

## Implementation Status

### Completed (2025-01-24)

✅ **ITensorGemm Interface Extensions**:
- Added `ActivationFormat` enum for precision tracking
- Added `multiply_activations_typed<ActT, WeightT>()` template methods
- Added `multiply_activations_strided_typed<ActT, WeightT>()` template methods

✅ **OneDNN Primitive Support**:
- Implemented `run_onednn_bf16_matmul()` (bf16bf16f32)
- Implemented `run_onednn_fp16_matmul()` (fp16fp16f32)
- Existing `run_onednn_fp32_matmul()` (f32f32f32)
- Existing `run_onednn_int8_matmul()` (int8int8s32)

✅ **OneDNNGemmKernel Extensions**:
- Implemented `multiply_activations_typed<float, float>()` (FP32 path)
- Implemented `multiply_activations_typed<uint16_t, uint16_t>()` (BF16 path)
- Implemented `multiply_activations_strided_typed<ActT, WeightT>()` (strided fallback)

### In Progress

🚧 **CpuAttentionKernelT Integration**:
- Replace `multiply_activations()` calls with `multiply_activations_typed()`
- Eliminate forced FP32 conversions in attention computation
- Add template specializations for BF16/FP16/INT8 attention

### Pending (Future Work)

⏳ **Precision Validation**:
- Add `PrecisionPolicy` validation in pipeline initialization
- Implement CLI flags `--activation-precision`
- Add user-facing warnings for discouraged combinations

⏳ **FP16 GEMM Testing**:
- Validate `run_onednn_fp16_matmul()` on AVX512_FP16 hardware
- Add E2E tests for FP16 attention

⏳ **INT8 Typed Path**:
- Implement `multiply_activations_typed<int8_t, int8_t>()`
- Integrate with existing `onednn_gemm_adapter()` INT8 path

⏳ **Automatic Format Detection**:
- Disambiguate BF16 vs FP16 for `uint16_t` inputs (currently assumes BF16)
- Add tensor format tags or traits for runtime detection

## Performance Benchmarks

**Preliminary results** (Qwen 2.5 0.5B, single-token decode, AVX512_BF16 CPU):

| Configuration | Throughput (tok/s) | Speedup vs FP32 | Memory Usage |
|---------------|-------------------|-----------------|--------------|
| FP32 Native | 1.04 | 1.0× (baseline) | 2.0 GB |
| BF16 Native (expected) | 1.8-2.0 | 1.7-1.9× | 1.0 GB |
| FP32 + Q8_0 Weight | 1.5-2.0 | 1.4-1.9× | 0.5 GB |
| BF16 + Q8_0 Weight (expected) | 2.2-2.8 | 2.1-2.7× | 0.5 GB |

**Note**: BF16 benchmarks pending integration into CpuAttentionKernelT.

## Migration Guide

### For Pipeline Developers

**Before (forced FP32 conversion)**:
```cpp
// Old code in CpuAttentionKernelT::compute_typed()
std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
convert_to_fp32(Q_typed, Q_fp32.data(), ...);  // Forced conversion

gemm->multiply_activations(Q_fp32.data(), K_fp32.data(), scores, ...);
```

**After (native precision)**:
```cpp
// New code with typed GEMM
gemm->multiply_activations_typed<ElementType, ElementType>(
    Q_typed, K_typed, scores,  // No conversion!
    seq_len, seq_len, head_dim,
    true, scale, 0.0f, nullptr, -1);
```

### For Kernel Developers

**Adding precision support to custom kernels**:

1. **Override typed GEMM methods** in kernel implementation:
   ```cpp
   class MyCustomGemmKernel : public ITensorGemm {
       // FP32 path
       template <>
       bool multiply_activations_typed<float, float>(...) {
           // Native FP32 implementation
       }

       // BF16 path
       template <>
       bool multiply_activations_typed<uint16_t, uint16_t>(...) {
           // Native BF16 implementation or fallback to FP32
       }
   };
   ```

2. **Use ActivationTraits** for precision-specific operations:
   ```cpp
   template <typename TensorType>
   class MyKernelT {
       using Traits = primitives::ActivationTraits<TensorType>;
       using ElementType = typename Traits::ElementType;

       void process() {
           auto gemm = Traits::create_activation_gemm();
           gemm->multiply_activations_typed<ElementType, ElementType>(...);
       }
   };
   ```

## References

- **ITensorGemm Interface**: `src/v2/tensors/TensorKernels.h`
- **OneDNN GEMM Primitives**: `src/v2/kernels/cpu/gemm_v4/OneDNNGemmKernel.h`
- **CpuAttentionKernelT**: `src/v2/kernels/cpu/CpuAttentionKernelT.h`
- **ActivationTraits**: `src/v2/kernels/cpu/primitives/ActivationTraits.h`
- **OneDNN Documentation**: https://oneapi-src.github.io/oneDNN/

## Future Enhancements

1. **Automatic Precision Selection**: Based on hardware capabilities (CPUID detection)
2. **Mixed-Precision Pipelines**: Different precisions per layer (e.g., FP16 attention, FP32 FFN)
3. **FP8 Support**: E4M3/E5M2 formats for next-gen accelerators
4. **Quantization-Aware Training Integration**: Load QAT models with calibration scales
5. **Dynamic Precision Switching**: Runtime precision adjustment based on numerical stability

---

**Last Updated**: 2025-01-24  
**Version**: V2 (Operator-Free Architecture)
