/**
 * @file ROCmEmbeddingKernels.hip
 * @brief Resident floating and original-source quantized embedding lookup.
 *
 * Quantized rows preserve their source codes, scales and minima. A launch-time
 * format specialization decodes only requested values through the common
 * source decoder; IQ tables are borrowed from the backend's existing immutable
 * generation owner. Replay performs no initialization, allocation or transfer.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../../common/NativeEmbeddingDecode.h"
#include "../../common/NativeEmbeddingLayout.h"
#include <limits>

// BF16 type support
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_bfloat16.h>
#else
typedef struct { uint16_t x; } hip_bfloat16;
#endif

namespace {

// ============================================================================
// BF16 Conversion Helpers
// ============================================================================

__device__ __forceinline__ uint16_t float_to_bf16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    // Round to nearest even (add 0x7FFF + bit 16)
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

// ============================================================================
// FP16 Conversion Helpers
// ============================================================================

__device__ __forceinline__ uint16_t float_to_fp16(float f)
{
    __half h = __float2half(f);
    return *reinterpret_cast<uint16_t*>(&h);
}

} // anonymous namespace

// Q8_1 block structure (matches CPU definition in BlockStructures.h)
// NOTE: Must match llaminar2::Q8_1Block exactly!
struct Q8_1Block
{
    uint16_t d;     // FP16 scale
    int16_t sum_qs; // sum of quantized values (Σqs[i])
    int8_t qs[32];  // quantized values
};

static_assert(sizeof(Q8_1Block) == 36, "Q8_1Block must be 36 bytes");

// ============================================================================
// FP32 Embedding Lookup Kernel
// ============================================================================

/**
 * @brief FP32 embedding lookup kernel
 *
 * Each thread handles one element: output[token_idx * d_model + dim_idx]
 *
 * @param embed_data Embedding table [vocab_size, d_model] FP32
 * @param token_ids  Token indices [num_tokens]
 * @param output     Output buffer [num_tokens, d_model] FP32
 * @param num_tokens Number of tokens
 * @param d_model    Embedding dimension
 */
__global__ void rocm_embedding_fp32_kernel(
    const float *__restrict__ embed_data,
    const int *__restrict__ token_ids,
    float *__restrict__ output,
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

// ============================================================================
// Native 16-bit Floating Embedding Tables with FP32 Output
// ============================================================================

/** @brief Read a resident FP16 embedding table and publish FP32 activations. */
__global__ void rocm_embedding_fp16_source_fp32_kernel(
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

/** @brief Read a resident BF16 embedding table and publish FP32 activations. */
__global__ void rocm_embedding_bf16_source_fp32_kernel(
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

// ============================================================================
// BF16 Embedding Lookup Kernel
// ============================================================================

/**
 * @brief BF16 embedding lookup kernel
 *
 * Reads FP32 embedding table and converts to BF16 output.
 */
__global__ void rocm_embedding_bf16_kernel(
    const float *__restrict__ embed_data,
    const int *__restrict__ token_ids,
    uint16_t *__restrict__ output,
    int num_tokens,
    int d_model)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * d_model;

    if (idx < total_elements)
    {
        int token_idx = idx / d_model;
        int dim_idx = idx % d_model;
        int token_id = token_ids[token_idx];
        float val = embed_data[token_id * d_model + dim_idx];

        // Convert FP32 to BF16
        output[idx] = float_to_bf16(val);
    }
}

// ============================================================================
// FP16 Embedding Lookup Kernel
// ============================================================================

/**
 * @brief FP16 embedding lookup kernel
 *
 * Reads FP32 embedding table and converts to FP16 output.
 */
__global__ void rocm_embedding_fp16_kernel(
    const float *__restrict__ embed_data,
    const int *__restrict__ token_ids,
    uint16_t *__restrict__ output,
    int num_tokens,
    int d_model)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * d_model;

    if (idx < total_elements)
    {
        int token_idx = idx / d_model;
        int dim_idx = idx % d_model;
        int token_id = token_ids[token_idx];
        float val = embed_data[token_id * d_model + dim_idx];

        // Convert FP32 to FP16
        output[idx] = float_to_fp16(val);
    }
}

// ============================================================================
// Q8_1 Embedding Lookup Kernel
// ============================================================================

/**
 * @brief Q8_1 embedding lookup kernel
 *
 * Each block handles one row (one token), quantizing d_model elements to Q8_1 blocks.
 * Each thread processes one Q8_1 block (32 elements).
 *
 * @param embed_data Embedding table [vocab_size, d_model] FP32
 * @param token_ids  Token indices [num_tokens]
 * @param output     Output Q8_1 blocks [num_tokens * blocks_per_row]
 * @param num_tokens Number of tokens
 * @param d_model    Embedding dimension (must be multiple of 32)
 */
__global__ void rocm_embedding_q8_1_kernel(
    const float *__restrict__ embed_data,
    const int *__restrict__ token_ids,
    Q8_1Block *__restrict__ output,
    int num_tokens,
    int d_model)
{
    int token_idx = blockIdx.x;
    int block_idx = threadIdx.x;
    int blocks_per_row = d_model / 32;

    if (token_idx >= num_tokens || block_idx >= blocks_per_row)
        return;

    int token_id = token_ids[token_idx];
    const float *src = &embed_data[token_id * d_model + block_idx * 32];
    Q8_1Block *dst = &output[token_idx * blocks_per_row + block_idx];

    // Find max absolute value for scaling
    float amax = 0.0f;
    float vals[32];

    #pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        vals[i] = src[i];
        float abs_val = fabsf(vals[i]);
        if (abs_val > amax)
            amax = abs_val;
    }

    // Compute scale (d) and inverse scale
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    // Quantize values and compute sum_qs
    int32_t sum_qs = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        int q = __float2int_rn(vals[i] * id);
        q = max(-128, min(127, q));
        dst->qs[i] = static_cast<int8_t>(q);
        sum_qs += q;
    }

    // Store FP16 scale
    dst->d = float_to_fp16(d);
    // Store sum of quantized values (clamped to int16)
    dst->sum_qs = static_cast<int16_t>(max(-32768, min(32767, sum_qs)));
}

// ============================================================================
// Host Launch Wrappers (extern "C" for linking from C++ code)
// ============================================================================

extern "C"
{
    /**
     * @brief Launch FP32 embedding lookup kernel
     */
    hipError_t hipOps_embedding_fp32(
        const float *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset,
        hipStream_t stream)
    {
        // Clear any stale error from previous operations (e.g., heterogeneous backend tests)
        (void)hipGetLastError();

        int total_elements = num_tokens * d_model;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        rocm_embedding_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model, vocab_size, vocab_offset);

        // No sync needed - coherence system handles sync when data is read
        return hipGetLastError();
    }

    /** @brief Launch native FP16-table to FP32 embedding lookup. */
    hipError_t hipOps_embedding_fp16_source_fp32(
        const uint16_t *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset,
        hipStream_t stream)
    {
        (void)hipGetLastError();
        const int total_elements = num_tokens * d_model;
        constexpr int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        rocm_embedding_fp16_source_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model, vocab_size, vocab_offset);
        return hipGetLastError();
    }

    /** @brief Launch native BF16-table to FP32 embedding lookup. */
    hipError_t hipOps_embedding_bf16_source_fp32(
        const uint16_t *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset,
        hipStream_t stream)
    {
        (void)hipGetLastError();
        const int total_elements = num_tokens * d_model;
        constexpr int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        rocm_embedding_bf16_source_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model, vocab_size, vocab_offset);
        return hipGetLastError();
    }

    /**
     * @brief Launch BF16 embedding lookup kernel
     */
    hipError_t hipOps_embedding_bf16(
        const float *embed_data,
        const int *token_ids,
        uint16_t *output,
        int num_tokens,
        int d_model,
        hipStream_t stream)
    {
        // Clear any stale error from previous operations
        (void)hipGetLastError();

        int total_elements = num_tokens * d_model;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        rocm_embedding_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model);

        // No sync needed - coherence system handles sync when data is read
        return hipGetLastError();
    }

    /**
     * @brief Launch FP16 embedding lookup kernel
     */
    hipError_t hipOps_embedding_fp16(
        const float *embed_data,
        const int *token_ids,
        uint16_t *output,
        int num_tokens,
        int d_model,
        hipStream_t stream)
    {
        // Clear any stale error from previous operations
        (void)hipGetLastError();

        int total_elements = num_tokens * d_model;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        rocm_embedding_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model);

        // No sync needed - coherence system handles sync when data is read
        return hipGetLastError();
    }

    /**
     * @brief Launch Q8_1 embedding lookup kernel
     *
     * @note d_model must be a multiple of 32
     */
    hipError_t hipOps_embedding_q8_1(
        const float *embed_data,
        const int *token_ids,
        void *output,
        int num_tokens,
        int d_model,
        hipStream_t stream)
    {
        if (d_model % 32 != 0)
        {
            fprintf(stderr, "[ROCm] Embedding Q8_1 requires d_model to be multiple of 32, got %d\n", d_model);
            return hipErrorInvalidValue;
        }

        // Clear any stale error from previous operations
        (void)hipGetLastError();

        int blocks_per_row = d_model / 32;
        // One block per token, one thread per Q8_1 block
        dim3 grid(num_tokens);
        dim3 block(blocks_per_row);

        rocm_embedding_q8_1_kernel<<<grid, block, 0, stream>>>(
            embed_data, token_ids, static_cast<Q8_1Block *>(output), num_tokens, d_model);

        // Removed hipDeviceSynchronize() - caller manages coherence via events
        return hipGetLastError();
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
__global__ void rocm_embedding_native_source_kernel(
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
hipError_t launchNativeSource(const void *source, const int *ids, float *output,
    int rows, int width, int vocab, int offset, llaminar2::NativeEmbeddingLayout layout,
    llaminar2::NativeEmbeddingIQTables tables, hipStream_t stream)
{
    using Block = typename llaminar2::native_embedding::SourceBlock<Format>::type;
    const size_t blocks = size_t(width) / Block::BLOCK_SIZE + (size_t(width) % Block::BLOCK_SIZE != 0);
    if (layout.block_elements != Block::BLOCK_SIZE || layout.block_bytes != sizeof(Block) ||
        layout.blocks_per_row != blocks || layout.row_stride_bytes != blocks * sizeof(Block) ||
        size_t(vocab) > std::numeric_limits<size_t>::max() / layout.row_stride_bytes ||
        layout.byte_size != size_t(vocab) * layout.row_stride_bytes)
        return hipErrorInvalidValue;
    if constexpr (Format == llaminar2::TensorType::IQ1_S || Format == llaminar2::TensorType::IQ1_M)
        { if (!tables.iq1s) return hipErrorInvalidValue; }
    if constexpr (Format == llaminar2::TensorType::IQ2_XXS)
        { if (!tables.iq2xxs) return hipErrorInvalidValue; }
    if constexpr (Format == llaminar2::TensorType::IQ2_XS)
        { if (!tables.iq2xs) return hipErrorInvalidValue; }
    if constexpr (Format == llaminar2::TensorType::IQ2_S)
        { if (!tables.iq2s) return hipErrorInvalidValue; }
    if constexpr (Format == llaminar2::TensorType::IQ3_XXS)
        { if (!tables.iq3xxs) return hipErrorInvalidValue; }
    if constexpr (Format == llaminar2::TensorType::IQ3_S)
        { if (!tables.iq3s) return hipErrorInvalidValue; }
    const size_t total = size_t(rows) * size_t(width);
    constexpr unsigned threads = 256;
    const size_t blocks_in_grid = total / threads + (total % threads != 0);
    if (blocks_in_grid > std::numeric_limits<unsigned>::max()) return hipErrorInvalidValue;
    rocm_embedding_native_source_kernel<Format><<<unsigned(blocks_in_grid), threads, 0, stream>>>(
        static_cast<const unsigned char *>(source), ids, output, total, width, vocab, offset,
        layout.row_stride_bytes, tables);
    return hipGetLastError();
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
extern "C" hipError_t hipOps_embedding_native(
    const void *source, const int *ids, float *output, int rows, int width,
    int vocab, int offset, llaminar2::NativeEmbeddingLayout layout,
    llaminar2::NativeEmbeddingIQTables tables, hipStream_t stream)
{
    if (!stream || !source || !ids || !output || rows <= 0 || width <= 0 || vocab <= 0 || offset < 0)
        return hipErrorInvalidValue;
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
    default: return hipErrorInvalidValue;
    }
#undef LLAMINAR_EMBED_SOURCE
}
