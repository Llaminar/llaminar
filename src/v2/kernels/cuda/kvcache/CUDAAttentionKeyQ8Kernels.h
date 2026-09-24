/**
 * @file CUDAAttentionKeyQ8Kernels.h
 * @brief CUDA launch interface for cubic-companded attention-key storage.
 *
 * The launchers in this file are allocation-free and graph-capturable. Every
 * operation requires the exact non-null stream that owns its dependency edge;
 * callers may not substitute the default stream.
 */

#pragma once

#include <cuda_runtime.h>

namespace llaminar2
{
    /**
     * @brief Quantize contiguous FP32 attention heads to AttentionKeyQ8 blocks.
     *
     * @param input Device FP32 input laid out `[block_count, head_dim]`.
     * @param output Device block storage sized for `block_count` blocks.
     * @param block_count Number of complete attention heads to encode.
     * @param head_dim Supported head width: 64, 128, or 256.
     * @param stream Mandatory explicit CUDA stream.
     * @return true when validation and kernel launch succeed.
     */
    [[nodiscard]] bool cudaAttentionKeyQ8Quantize(
        const float *input,
        void *output,
        int block_count,
        int head_dim,
        cudaStream_t stream);

    /**
     * @brief Dequantize contiguous AttentionKeyQ8 blocks to FP32 heads.
     *
     * @param input Device block storage containing `block_count` blocks.
     * @param output Device FP32 output laid out `[block_count, head_dim]`.
     * @param block_count Number of complete attention heads to decode.
     * @param head_dim Supported head width: 64, 128, or 256.
     * @param stream Mandatory explicit CUDA stream.
     * @return true when validation and kernel launch succeed.
     */
    [[nodiscard]] bool cudaAttentionKeyQ8Dequantize(
        const void *input,
        float *output,
        int block_count,
        int head_dim,
        cudaStream_t stream);
} // namespace llaminar2
