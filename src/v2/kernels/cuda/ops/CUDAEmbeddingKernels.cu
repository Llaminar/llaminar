/**
 * @file CUDAEmbeddingKernels.cu
 * @brief CUDA kernel implementation for embedding lookup
 *
 * Supports:
 * - FP32 embedding tables (simple row lookup, for tensors already on GPU)
 * - Native FP16/BF16 embedding tables converted to FP32 at lookup time
 * - EmbedQ8 quantized embedding tables (universal format, handles all GGUF quant types)
 *
 * The EmbedQ8 path works with ANY quantized embedding format (Q4_0, Q8_0, Q6_K,
 * IQ4_NL, Q4_K, etc.). CPU-side repacking via IINT8Unpackable converts the native
 * format into a uniform EmbedQ8Block = {FP16 scale, FP16 min, int8[32]}.
 * This avoids writing per-format GPU kernels while saving 3.5× VRAM vs FP32 upload.
 *
 * @author David Sanftenberg
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

#include "../../common/EmbedQ8Block.h"

using llaminar2::EmbedQ8Block;

extern "C"
{
    // =========================================================================
    // FP32 Embedding Lookup (for FP32 tensors already on GPU via coherence)
    // =========================================================================

    /**
     * @brief CUDA kernel for FP32 embedding lookup
     *
     * Each thread handles one element: output[token_idx * d_model + dim_idx]
     */
    __global__ void embedding_lookup_kernel(
        const float *__restrict__ embed_data, // [vocab_size, d_model]
        const int *__restrict__ token_ids,    // [num_tokens]
        float *__restrict__ output,           // [num_tokens, d_model]
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = num_tokens * d_model;

        if (idx < total_elements)
        {
            int token_idx = idx / d_model;
            int dim_idx = idx % d_model;
            int token_id = token_ids[token_idx];

            // Vocab-parallel: offset token ID by this shard's base
            int local_id = token_id - vocab_offset;

            if (local_id < 0 || local_id >= vocab_size)
            {
                output[idx] = 0.0f;
                return;
            }

            output[idx] = embed_data[local_id * d_model + dim_idx];
        }
    }

    /**
     * @brief Launch FP32 embedding lookup kernel
     */
    cudaError_t launch_embedding_lookup(
        const float *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset,
        cudaStream_t stream)
    {
        int total_elements = num_tokens * d_model;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        embedding_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model, vocab_size, vocab_offset);

        return cudaGetLastError();
    }

    // =========================================================================
    // Native 16-bit Floating Embedding Tables
    // =========================================================================

    /** @brief Convert native FP16 table elements to FP32 during lookup. */
    __global__ void embedding_lookup_fp16_source_kernel(
        const uint16_t *__restrict__ embed_data,
        const int *__restrict__ token_ids,
        float *__restrict__ output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_elements = num_tokens * d_model;
        if (idx >= total_elements)
            return;

        const int token_idx = idx / d_model;
        const int dim_idx = idx % d_model;
        const int local_id = token_ids[token_idx] - vocab_offset;
        if (local_id < 0 || local_id >= vocab_size)
        {
            output[idx] = 0.0f;
            return;
        }

        const uint16_t source_bits = embed_data[local_id * d_model + dim_idx];
        output[idx] = __half2float(*reinterpret_cast<const __half *>(&source_bits));
    }

    /** @brief Convert native BF16 table elements to FP32 during lookup. */
    __global__ void embedding_lookup_bf16_source_kernel(
        const uint16_t *__restrict__ embed_data,
        const int *__restrict__ token_ids,
        float *__restrict__ output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_elements = num_tokens * d_model;
        if (idx >= total_elements)
            return;

        const int token_idx = idx / d_model;
        const int dim_idx = idx % d_model;
        const int local_id = token_ids[token_idx] - vocab_offset;
        if (local_id < 0 || local_id >= vocab_size)
        {
            output[idx] = 0.0f;
            return;
        }

        const uint16_t source_bits = embed_data[local_id * d_model + dim_idx];
        output[idx] = __uint_as_float(static_cast<uint32_t>(source_bits) << 16);
    }

    /** @brief Launch native FP16-table to FP32 embedding lookup. */
    cudaError_t launch_embedding_lookup_fp16_source(
        const uint16_t *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset,
        cudaStream_t stream)
    {
        const int total_elements = num_tokens * d_model;
        constexpr int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        embedding_lookup_fp16_source_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model, vocab_size, vocab_offset);
        return cudaGetLastError();
    }

    /** @brief Launch native BF16-table to FP32 embedding lookup. */
    cudaError_t launch_embedding_lookup_bf16_source(
        const uint16_t *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset,
        cudaStream_t stream)
    {
        const int total_elements = num_tokens * d_model;
        constexpr int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        embedding_lookup_bf16_source_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model, vocab_size, vocab_offset);
        return cudaGetLastError();
    }

    // =========================================================================
    // EmbedQ8 Universal Quantized Embedding Lookup
    // =========================================================================
    //
    // Instead of dequantizing the entire embedding table to FP32 on CPU (~519MB),
    // repacking to EmbedQ8Block (~146MB) and uploading that. Each thread
    // dequantizes one element at lookup time:
    //
    //   value = int8_val * half_to_float(scale) + half_to_float(min)
    //
    // This single kernel handles ALL quantized formats because the CPU-side
    // IINT8Unpackable repack normalizes everything into {scale, min, int8[32]}.
    //
    // Memory savings:  FP32 519MB → EmbedQ8 146MB (3.56× per device)
    // CPU savings:     No FP32 dequantization of entire vocab table
    // Transfer savings: 146MB upload vs 519MB upload (3.56× less PCIe bandwidth)
    // =========================================================================

    /**
     * @brief CUDA kernel for EmbedQ8 quantized embedding lookup
     *
     * Each thread dequantizes one element from the EmbedQ8Block.
     * Layout: blocks_per_row consecutive EmbedQ8Blocks per vocabulary entry.
     *
     * @param embed_q8       EmbedQ8 blocks on GPU [vocab_size * blocks_per_row]
     * @param token_ids      Token IDs on GPU [num_tokens]
     * @param output         FP32 output on GPU [num_tokens, d_model]
     * @param num_tokens     Number of tokens
     * @param d_model        Embedding dimension
     * @param blocks_per_row Number of EmbedQ8 blocks per row (= ceil(d_model/32))
     */
    __global__ void embedding_lookup_q8_kernel(
        const EmbedQ8Block *__restrict__ embed_q8,
        const int *__restrict__ token_ids,
        float *__restrict__ output,
        int num_tokens,
        int d_model,
        int blocks_per_row,
        int vocab_size,
        int vocab_offset)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = num_tokens * d_model;

        if (idx < total_elements)
        {
            int token_idx = idx / d_model;
            int dim_idx = idx % d_model;
            int token_id = token_ids[token_idx];

            // Vocab-parallel: offset token ID by this shard's base
            int local_id = token_id - vocab_offset;

            if (local_id < 0 || local_id >= vocab_size)
            {
                output[idx] = 0.0f;
                return;
            }

            // Locate the EmbedQ8Block containing this element
            int block_idx = dim_idx / 32;
            int elem_in_block = dim_idx % 32;

            const EmbedQ8Block &block = embed_q8[local_id * blocks_per_row + block_idx];

            // Dequantize: value = qs[i] * scale + min
            float scale = __half2float(*reinterpret_cast<const __half *>(&block.d));
            float min_val = __half2float(*reinterpret_cast<const __half *>(&block.m));
            float q_val = static_cast<float>(block.qs[elem_in_block]);

            output[idx] = q_val * scale + min_val;
        }
    }

    /**
     * @brief Launch EmbedQ8 quantized embedding lookup kernel
     */
    cudaError_t launch_embedding_lookup_q8(
        const void *embed_q8,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int blocks_per_row,
        int vocab_size,
        int vocab_offset,
        cudaStream_t stream)
    {
        int total_elements = num_tokens * d_model;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        embedding_lookup_q8_kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const EmbedQ8Block *>(embed_q8),
            token_ids, output, num_tokens, d_model, blocks_per_row, vocab_size, vocab_offset);

        return cudaGetLastError();
    }

} // extern "C"
