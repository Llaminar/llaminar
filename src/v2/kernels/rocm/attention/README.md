# ROCm Flash Attention Kernel for MI50

This directory contains the HIP implementation of Flash Attention 2 (prefill) and Flash Decoding (decode) optimized for AMD MI50 (gfx906 / Vega 20) GPUs.

## Architecture Overview

### Target Hardware: AMD MI50 (gfx906)

| Specification | Value |
|---------------|-------|
| Architecture | Vega 20 (GCN 5.0) |
| Compute Units | 60 |
| Wavefront Size | 64 threads |
| LDS per CU | 64 KB |
| HBM2 Bandwidth | 1024 GB/s |
| FP32 Peak | 13.4 TFLOPS |
| FP16 Peak | 26.8 TFLOPS (packed math) |
| Matrix Cores | **None** (first added in CDNA1/MI100) |

### Key Differences from CUDA Implementation

| Aspect | CUDA (Ampere+) | ROCm (MI50) |
|--------|----------------|-------------|
| GEMM Acceleration | WMMA/Tensor Cores | Vectorized VALU |
| Wavefront/Warp Size | 32 | 64 |
| Shared Memory | Up to 164KB opt-in | 64KB LDS |
| Async Copy | `cp.async` | Explicit loads |
| BF16 Support | Native | Limited (emulated) |

## Algorithms

### Flash Attention 2 (Prefill, seq_len > 1)

The prefill kernel implements tiled attention with online softmax:

```
for each Q tile:
    for each KV tile:
        1. Load K, V tiles to LDS (double-buffered)
        2. Compute S = Q @ K^T (vectorized FP16 dot products)
        3. Apply causal mask and softmax scaling
        4. Online softmax: update (m, l, O) accumulators
        5. Accumulate O += P @ V
    Write final O = O / l
```

**MI50-Specific Optimizations:**
- Tile sizes: Q=64, KV=64 (fits in 64KB LDS with double buffering)
- FP16 storage in LDS, FP32 accumulation for numerical stability
- LDS padding to avoid bank conflicts (32 banks)
- 256 threads = 4 wavefronts per block for occupancy
- Vectorized FP16 loads (2 elements per instruction)

### Flash Decoding (Decode, seq_len = 1)

For single-token decode with large KV caches, we use split-K parallelism:

```
Phase 1 (Split-K):
    Grid: (n_heads, num_splits, batch_size)
    Each block processes kv_len/num_splits positions
    Store partial (m, l, O) per split

Phase 2 (Reduction):
    Grid: (n_heads, batch_size)
    Merge partial results using stable softmax
    O_final = Σ exp(m_i - m_max) * O_i / Σ exp(m_i - m_max) * l_i
```

**Split Configuration:**
| KV Length | Splits | Rationale |
|-----------|--------|-----------|
| ≤ 64 | 1 | No parallelism needed |
| 65-127 | 2 | Minimal overhead |
| 128-255 | 4 | Balanced |
| ≥ 256 | 8 | Maximum parallelism |

## File Structure

```
src/v2/kernels/rocm/attention/
├── ROCmFlashAttentionKernelT.h      # C++ header with ITensorAttention interface
├── ROCmFlashAttentionKernelT.cpp    # C++ implementation (delegates to HIP)
├── ROCmFlashAttentionKernels.hip    # HIP device kernels
└── README.md                        # This file
```

## Interface

The kernel implements the `ITensorAttention` interface:

```cpp
namespace llaminar2::rocm {
    template <ActivationPrecision Precision>
    class ROCmFlashAttentionKernelT : public ITensorAttention {
        // Single-sequence attention
        bool compute(const float* Q, const float* K, const float* V, float* output,
                     int seq_len, int n_heads, int n_kv_heads, int head_dim, ...);

        // Batched attention
        bool compute_batch(const float* Q, const float* K, const float* V, float* output,
                           int batch_size, int seq_len, ...);

        // Decode with separate KV length
        bool compute_decode(const float* Q, const float* K, const float* V, float* output,
                            int seq_len, int kv_len, ...);

        // Tensor-based dispatch (primary entry point for stages)
        bool compute_tensor(const ITensor* Q, const ITensor* K, const ITensor* V,
                            ITensor* output, ...);
    };

    using ROCmFlashAttentionKernelFP32 = ROCmFlashAttentionKernelT<ActivationPrecision::FP32>;
    using ROCmFlashAttentionKernelFP16 = ROCmFlashAttentionKernelT<ActivationPrecision::FP16>;
    using ROCmFlashAttentionKernelBF16 = ROCmFlashAttentionKernelT<ActivationPrecision::BF16>;
}
```

## Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| FP32 Attention | ✅ Full | Primary implementation |
| FP16 Attention | ⚠️ Fallback | Uses FP32 kernel internally |
| BF16 Attention | ⚠️ Fallback | MI50 lacks native BF16 |
| Grouped Query Attention (GQA) | ✅ Full | n_heads % n_kv_heads == 0 |
| Multi-Query Attention (MQA) | ✅ Full | n_kv_heads = 1 |
| Causal Masking | ✅ Full | Lower-triangular mask |
| Sliding Window | ✅ Full | window_size parameter |
| head_dim ≤ 64 | ✅ Full | Qwen2.5 |
| head_dim ≤ 128 | ✅ Full | LLaMA-3 |
| head_dim > 128 | ❌ | Requires kernel extension |

## Performance Characteristics

### Memory Bandwidth Bound

Flash Attention on MI50 is primarily **memory bandwidth bound** due to:
1. No Matrix Cores for accelerated GEMM
2. HBM2 bandwidth (1024 GB/s) is the primary bottleneck
3. ALU throughput (13.4 TFLOPS FP32) exceeds memory delivery rate

**Arithmetic Intensity Analysis:**
- Q @ K^T: 2 * seq_len * kv_len * head_dim FLOPs / (seq_len + kv_len) * head_dim * 4 bytes
- For typical LLM shapes, arithmetic intensity < 100 FLOP/byte
- MI50 requires ~100 FLOP/byte to saturate compute

### Expected Performance

| Workload | seq_len | kv_len | Throughput |
|----------|---------|--------|------------|
| Prefill (short) | 128 | 128 | ~50% peak BW |
| Prefill (long) | 2048 | 2048 | ~70% peak BW |
| Decode | 1 | 2048 | ~40% peak BW |

*Note: Flash Decoding improves decode performance by parallelizing over KV cache.*

## Build Integration

Add to CMakeLists.txt:

```cmake
if(HAVE_ROCM)
    set(ROCM_ATTENTION_SOURCES
        kernels/rocm/attention/ROCmFlashAttentionKernelT.cpp
        kernels/rocm/attention/ROCmFlashAttentionKernels.hip
    )
    
    set_source_files_properties(
        kernels/rocm/attention/ROCmFlashAttentionKernels.hip
        PROPERTIES LANGUAGE HIP
    )
    
    target_sources(llaminar2_core PRIVATE ${ROCM_ATTENTION_SOURCES})
endif()
```

## Future Optimizations

### Phase 2: Native FP16 Path
- Implement FP16 kernels with packed math (`v_pk_*` instructions)
- 2x throughput improvement for FP16 inputs
- FP32 accumulation for numerical stability

### Phase 3: CDNA Support (MI100/MI200/MI300)
- Add Matrix Core (MFMA) kernels for CDNA architectures
- Significant speedup from hardware matrix multiply
- Would require separate kernel specializations

### Phase 4: Composable Kernels Integration
- Leverage AMD's Composable Kernel library
- Pre-tuned kernels for various shapes
- Better occupancy and register allocation

## Testing

Run unit tests:
```bash
ctest --test-dir build_v2 -R "V2_Unit_ROCmFlashAttention" --output-on-failure
```

Run integration tests:
```bash
ctest --test-dir build_v2_integration -R "V2_Integration_ROCm_Attention" --output-on-failure
```

## References

1. [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
2. [Flash Decoding Paper](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
3. [AMD ROCm Documentation](https://rocm.docs.amd.com/)
4. [GCN ISA Reference](https://developer.amd.com/resources/developer-guides-manuals/)
5. [Composable Kernel Library](https://github.com/ROCmSoftwarePlatform/composable_kernel)
