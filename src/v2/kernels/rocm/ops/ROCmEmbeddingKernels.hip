/**
 * @file ROCmEmbeddingKernels.hip
 * @brief ROCm/HIP kernels for embedding lookup operations
 *
 * Implements FP32, BF16, FP16, and Q8_1 embedding lookups using HIP.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

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
    int d_model)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * d_model;

    if (idx < total_elements)
    {
        int token_idx = idx / d_model;
        int dim_idx = idx % d_model;
        int token_id = token_ids[token_idx];
        output[idx] = embed_data[token_id * d_model + dim_idx];
    }
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
        hipStream_t stream)
    {
        int total_elements = num_tokens * d_model;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        rocm_embedding_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model);

        hipDeviceSynchronize();
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
        int total_elements = num_tokens * d_model;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        rocm_embedding_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model);

        hipDeviceSynchronize();
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
        int total_elements = num_tokens * d_model;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        rocm_embedding_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
            embed_data, token_ids, output, num_tokens, d_model);

        hipDeviceSynchronize();
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

        int blocks_per_row = d_model / 32;
        // One block per token, one thread per Q8_1 block
        dim3 grid(num_tokens);
        dim3 block(blocks_per_row);

        rocm_embedding_q8_1_kernel<<<grid, block, 0, stream>>>(
            embed_data, token_ids, static_cast<Q8_1Block *>(output), num_tokens, d_model);

        hipDeviceSynchronize();
        return hipGetLastError();
    }

} // extern "C"
