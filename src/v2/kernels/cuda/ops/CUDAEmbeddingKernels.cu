/**
 * @file CUDAEmbeddingKernels.cu
 * @brief Resident floating and original-source quantized embedding lookup.
 *
 * Quantized rows preserve their source codes, scales and minima. A launch-time
 * format specialization decodes only requested values through the common
 * source decoder; IQ tables are borrowed from the backend's existing immutable
 * generation owner. Replay performs no initialization, allocation or transfer.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

#include "../../common/NativeEmbeddingDecode.h"
#include "../../common/NativeEmbeddingLayout.h"
#include <limits>

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


} // extern "C"

namespace {
/**
 * @brief Decode one exact original source value with a unique output writer.
 * @tparam Format Immutable source codebook selected before launch.
 * @param source Original packed table; no EmbedQ8 intermediate is present.
 * @param ids Live device token rows on the exact producer stream.
 * @param output Full-width FP32 activation rows.
 * @param total Number of live output elements.
 * @param width Logical row width, including partial final source blocks.
 * @param vocab Number of locally owned original vocabulary rows.
 * @param offset Global first vocabulary row of this shard.
 * @param row_bytes Exact source row stride from prepared layout.
 * @param tables Borrowed immutable backend IQ grid addresses.
 */
template<llaminar2::TensorType Format>
__global__ void cuda_embedding_native_source_kernel(
    const unsigned char *__restrict__ source, const int *__restrict__ ids,
    float *__restrict__ output, size_t total, int width, int vocab, int offset,
    size_t row_bytes, llaminar2::NativeEmbeddingIQTables tables)
{
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const size_t row = index / size_t(width);
    const size_t column = index % size_t(width);
    // Widen subtraction before range checking to avoid invalid-token overflow.
    const int64_t local = int64_t(ids[row]) - int64_t(offset);
    output[index] = local < 0 || local >= vocab ? 0.0f
        : llaminar2::native_embedding::decode<Format>(
            source + size_t(local) * row_bytes, column, tables);
}

/**
 * @brief Validate the complete source layout and submit its specialized lookup.
 * @tparam Format Original table format; never selected from an output dtype.
 * @param source Immutable prepared bytes.
 * @param ids Ordered device token source.
 * @param output Full-width FP32 output.
 * @param rows Live row count for this captured executable.
 * @param width Logical embedding width.
 * @param vocab Locally owned original source rows.
 * @param offset Global start of the local vocabulary shard.
 * @param layout Sole physical byte/stride authority from preparation.
 * @param tables Setup-bound codebook view.
 * @param stream Exact non-null launch stream.
 * @return Backend launch status; incompatible geometry is an error.
 */
template<llaminar2::TensorType Format>
cudaError_t launchNativeSource(const void *source, const int *ids, float *output,
    int rows, int width, int vocab, int offset, llaminar2::NativeEmbeddingLayout layout,
    llaminar2::NativeEmbeddingIQTables tables, cudaStream_t stream)
{
    using Block = typename llaminar2::native_embedding::SourceBlock<Format>::type;
    const size_t blocks = size_t(width) / Block::BLOCK_SIZE + (size_t(width) % Block::BLOCK_SIZE != 0);
    if (layout.block_elements != Block::BLOCK_SIZE || layout.block_bytes != sizeof(Block) ||
        layout.blocks_per_row != blocks || layout.row_stride_bytes != blocks * sizeof(Block) ||
        size_t(vocab) > std::numeric_limits<size_t>::max() / layout.row_stride_bytes ||
        layout.byte_size != size_t(vocab) * layout.row_stride_bytes)
        return cudaErrorInvalidValue;
    if constexpr (Format == llaminar2::TensorType::IQ1_S || Format == llaminar2::TensorType::IQ1_M)
        { if (!tables.iq1s) return cudaErrorInvalidValue; }
    if constexpr (Format == llaminar2::TensorType::IQ2_XXS)
        { if (!tables.iq2xxs) return cudaErrorInvalidValue; }
    if constexpr (Format == llaminar2::TensorType::IQ2_XS)
        { if (!tables.iq2xs) return cudaErrorInvalidValue; }
    if constexpr (Format == llaminar2::TensorType::IQ2_S)
        { if (!tables.iq2s) return cudaErrorInvalidValue; }
    if constexpr (Format == llaminar2::TensorType::IQ3_XXS)
        { if (!tables.iq3xxs) return cudaErrorInvalidValue; }
    if constexpr (Format == llaminar2::TensorType::IQ3_S)
        { if (!tables.iq3s) return cudaErrorInvalidValue; }
    const size_t total = size_t(rows) * size_t(width);
    constexpr unsigned threads = 256;
    const size_t blocks_in_grid = total / threads + (total % threads != 0);
    if (blocks_in_grid > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidValue;
    cuda_embedding_native_source_kernel<Format><<<unsigned(blocks_in_grid), threads, 0, stream>>>(
        static_cast<const unsigned char *>(source), ids, output, total, width, vocab, offset,
        layout.row_stride_bytes, tables);
    return cudaGetLastError();
}
} // namespace

/**
 * @brief Launch original-source quantized lookup without a replay-time converter.
 * @param source Original immutable table storage.
 * @param ids Device token source.
 * @param output Full-width FP32 destination.
 * @param rows Number of live rows represented by this captured executable.
 * @param width Original embedding width.
 * @param vocab Local original table row count.
 * @param offset Global first row of this vocabulary shard.
 * @param layout Exact source format/block/stride/extent identity.
 * @param tables Borrowed source IQ codebooks prepared before capture.
 * @param stream Exact non-null producer stream.
 * @return Backend status, rejecting null bindings and unsupported formats.
 */
extern "C" cudaError_t launch_embedding_lookup_native(
    const void *source, const int *ids, float *output, int rows, int width,
    int vocab, int offset, llaminar2::NativeEmbeddingLayout layout,
    llaminar2::NativeEmbeddingIQTables tables, cudaStream_t stream)
{
    if (!stream || !source || !ids || !output || rows <= 0 || width <= 0 || vocab <= 0 || offset < 0)
        return cudaErrorInvalidValue;
#define LLAMINAR_EMBED_SOURCE(TYPE) \
    case llaminar2::TensorType::TYPE: return launchNativeSource<llaminar2::TensorType::TYPE>( \
        source, ids, output, rows, width, vocab, offset, layout, tables, stream)
    switch (layout.source_type)
    {
        LLAMINAR_EMBED_SOURCE(Q4_0);
        LLAMINAR_EMBED_SOURCE(Q4_1);
        LLAMINAR_EMBED_SOURCE(Q5_0);
        LLAMINAR_EMBED_SOURCE(Q5_1);
        LLAMINAR_EMBED_SOURCE(Q8_0);
        LLAMINAR_EMBED_SOURCE(Q8_1);
        LLAMINAR_EMBED_SOURCE(Q2_K);
        LLAMINAR_EMBED_SOURCE(Q3_K);
        LLAMINAR_EMBED_SOURCE(Q4_K);
        LLAMINAR_EMBED_SOURCE(Q5_K);
        LLAMINAR_EMBED_SOURCE(Q6_K);
        LLAMINAR_EMBED_SOURCE(Q8_K);
        LLAMINAR_EMBED_SOURCE(IQ1_S);
        LLAMINAR_EMBED_SOURCE(IQ1_M);
        LLAMINAR_EMBED_SOURCE(IQ2_XXS);
        LLAMINAR_EMBED_SOURCE(IQ2_XS);
        LLAMINAR_EMBED_SOURCE(IQ2_S);
        LLAMINAR_EMBED_SOURCE(IQ3_XXS);
        LLAMINAR_EMBED_SOURCE(IQ3_S);
        LLAMINAR_EMBED_SOURCE(IQ4_NL);
        LLAMINAR_EMBED_SOURCE(IQ4_XS);
    default: return cudaErrorInvalidValue;
    }
#undef LLAMINAR_EMBED_SOURCE
}
