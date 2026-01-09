/**
 * @file CUDAEmbeddingKernels.cu
 * @brief CUDA kernel implementation for embedding lookup
 *
 * Simple embedding lookup: output[token_idx] = embed_table[token_id]
 * Each row of the embedding table is d_model floats.
 *
 * @author David Sanftenberg
 */

#include <cuda_runtime.h>
#include <cstdio>

extern "C"
{
    /**
     * @brief CUDA kernel for embedding lookup
     *
     * Each thread handles one element: output[token_idx * d_model + dim_idx]
     */
    __global__ void embedding_lookup_kernel(
        const float *__restrict__ embed_data, // [vocab_size, d_model]
        const int *__restrict__ token_ids,    // [num_tokens]
        float *__restrict__ output,           // [num_tokens, d_model]
        int num_tokens,
        int d_model)
    {
        // Global thread index
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = num_tokens * d_model;

        if (idx < total_elements)
        {
            int token_idx = idx / d_model;
            int dim_idx = idx % d_model;
            int token_id = token_ids[token_idx];

            // Simple row lookup
            output[idx] = embed_data[token_id * d_model + dim_idx];
        }
    }

    /**
     * @brief Launch embedding lookup kernel
     *
     * @param embed_data Embedding table on GPU [vocab_size, d_model]
     * @param token_ids Token IDs on GPU [num_tokens]
     * @param output Output buffer on GPU [num_tokens, d_model]
     * @param num_tokens Number of tokens to embed
     * @param d_model Embedding dimension
     * @param stream CUDA stream (0 for default)
     * @return cudaError_t
     */
    cudaError_t launch_embedding_lookup(
        const float *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        cudaStream_t stream)
    {
        int total_elements = num_tokens * d_model;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        embedding_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model);

        return cudaGetLastError();
    }

} // extern "C"
