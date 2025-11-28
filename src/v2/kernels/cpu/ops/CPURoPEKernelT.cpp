/**
 * @file CPURoPEKernelT.cpp
 * @brief CPU RoPE kernel implementation (uses vectorized primitives)
 *
 * @author David Sanftenberg
 */

#include "CPURoPEKernelT.h"
#include "../primitives/RoPEPrimitives.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../tensors/Tensors.h"
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <omp.h>
#include <type_traits>
#include <iostream>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    // Thread-local state for decode optimization
    template <typename TensorT>
    thread_local primitives::RoPEPersistentState CPURoPEKernelT<TensorT>::tls_state_;

    // Use primitives library for inverse frequency cache
    template <typename TensorT>
    const std::vector<float> &CPURoPEKernelT<TensorT>::get_inv_freq_cached(int head_dim, float freq_base)
    {
        return primitives::get_inv_freq_cached(head_dim, freq_base);
    }

    template <typename TensorT>
    bool CPURoPEKernelT<TensorT>::apply(
        float *Q, float *K,
        const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float rope_theta,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, FP32Tensor>)
        {
            (void)device_idx; // Device index ignored - always operates on CPU buffers

            if (head_dim % 2 != 0)
            {
                return false; // head_dim must be even
            }

            if (seq_len <= 0)
            {
                return true; // Nothing to do
            }

            // Apply RoPE per-token with individual positions from position_ids array
            // This supports batched inference where sequences have independent positions
            const int q_stride = n_heads * head_dim;
            const int k_stride = n_kv_heads * head_dim;

            // Check for contiguous positions to enable optimized block processing
            bool contiguous = true;
            int start_pos = 0;

            if (position_ids)
            {
                start_pos = position_ids[0];
                // Only check if we have more than 1 token
                if (seq_len > 1)
                {
                    // Check contiguity
                    for (int i = 1; i < seq_len; ++i)
                    {
                        if (position_ids[i] != start_pos + i)
                        {
                            contiguous = false;
                            break;
                        }
                    }
                }
            }
            else
            {
                // If position_ids is null, we assume 0, 1, 2... (standard prefill)
                contiguous = true;
                start_pos = 0;
            }

            // If positions are contiguous and valid (no padding), use optimized block processing
            // This enables OpenMP parallelism in the underlying primitives
            if (contiguous && start_pos >= 0)
            {
                apply_rotation(Q, K, seq_len, head_dim, n_heads, n_kv_heads, start_pos, rope_theta);
                return true;
            }

            for (int tok = 0; tok < seq_len; ++tok)
            {
                int position = position_ids ? position_ids[tok] : tok;

                // Skip RoPE for padding tokens (position_id = -1)
                if (position < 0)
                {
                    continue; // Padding token - leave Q/K unrotated
                }

                float *q_token = Q + tok * q_stride;
                float *k_token = K ? (K + tok * k_stride) : nullptr;

                // Apply RoPE to single token (seq_len=1, n_past=position)
                apply_rotation(q_token, k_token, 1, head_dim, n_heads, n_kv_heads, position, rope_theta);
            }

            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    void CPURoPEKernelT<TensorT>::apply_rotation(
        float *q, float *k,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base)
    {
        // Use vectorized primitives implementation
        primitives::apply_rope_vectorized(
            q, k,
            seq_len, head_dim,
            q_heads, k_heads,
            n_past, freq_base,
            (seq_len == 1) ? &tls_state_ : nullptr);
    }

    template <typename TensorT>
    bool CPURoPEKernelT<TensorT>::apply_bf16(
        uint16_t *Q_bf16, uint16_t *K_bf16,
        const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float rope_theta,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, BF16Tensor>)
        {
            if (device_idx != -1)
            {
                return false; // CPU kernel only
            }

            if (head_dim % 2 != 0)
            {
                return false; // head_dim must be even
            }

            if (seq_len <= 0)
            {
                return true; // Nothing to do
            }

            // Apply RoPE per-token with individual positions from position_ids array
            const int q_stride = n_heads * head_dim;
            const int k_stride = n_kv_heads * head_dim;

            for (int tok = 0; tok < seq_len; ++tok)
            {
                int position = position_ids ? position_ids[tok] : tok;

                // Skip RoPE for padding tokens (position_id = -1)
                if (position < 0)
                {
                    continue; // Padding token - leave Q/K unrotated
                }

                uint16_t *q_token = Q_bf16 + tok * q_stride;
                uint16_t *k_token = K_bf16 ? (K_bf16 + tok * k_stride) : nullptr;

                // Apply RoPE to single token (seq_len=1, n_past=position)
                primitives::apply_rope_bf16(
                    q_token, k_token,
                    1, head_dim,
                    n_heads, n_kv_heads,
                    position, rope_theta,
                    &tls_state_); // Always use persistent state for single-token
            }

            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPURoPEKernelT<TensorT>::apply_fp16(
        uint16_t *Q_fp16, uint16_t *K_fp16,
        const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float rope_theta,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, FP16Tensor>)
        {
            if (device_idx != -1)
            {
                return false; // CPU kernel only
            }

            if (head_dim % 2 != 0)
            {
                return false; // head_dim must be even
            }

            if (seq_len <= 0)
            {
                return true; // Nothing to do
            }

            // Apply RoPE per-token with individual positions from position_ids array
            const int q_stride = n_heads * head_dim;
            const int k_stride = n_kv_heads * head_dim;

            for (int tok = 0; tok < seq_len; ++tok)
            {
                int position = position_ids ? position_ids[tok] : tok;

                // Skip RoPE for padding tokens (position_id = -1)
                if (position < 0)
                {
                    continue; // Padding token - leave Q/K unrotated
                }

                uint16_t *q_token = Q_fp16 + tok * q_stride;
                uint16_t *k_token = K_fp16 ? (K_fp16 + tok * k_stride) : nullptr;

                // Apply RoPE to single token (seq_len=1, n_past=position)
                primitives::apply_rope_fp16(
                    q_token, k_token,
                    1, head_dim,
                    n_heads, n_kv_heads,
                    position, rope_theta,
                    &tls_state_); // Always use persistent state for single-token
            }

            return true;
        }
        else
        {
            return false;
        }
    }

    // Helper for Q8_1 head processing (Fused Dequant+RoPE+Requant)
    // Avoids full head buffer allocation and round-trip overhead
    static void process_q8_1_head_fused(
        Q8_1Block *head_ptr,
        int head_dim,
        const float *cos_cache,
        const float *sin_cache)
    {
        int half_dim = head_dim / 2;
        int blocks_per_half = half_dim / Q8_1Block::BLOCK_SIZE;

#if defined(__AVX2__)
        // AVX2 Implementation
        for (int b = 0; b < blocks_per_half; ++b)
        {
            Q8_1Block &blockA = head_ptr[b];
            Q8_1Block &blockB = head_ptr[b + blocks_per_half];

            // Buffers for rotated values (32 elements each)
            alignas(32) float rotA[32];
            alignas(32) float rotB[32];

            // 1. Dequantize and Rotate
            float scaleA = fp16_to_fp32(blockA.d);
            float scaleB = fp16_to_fp32(blockB.d);
            __m256 vscaleA = _mm256_set1_ps(scaleA);
            __m256 vscaleB = _mm256_set1_ps(scaleB);

            int offset = b * 32; // Offset in head for cos/sin

            for (int i = 0; i < 4; ++i) // 4 * 8 = 32 elements
            {
                // Load 8 int8 from A
                __m128i i8A = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&blockA.qs[i * 8]));
                __m256i i32A = _mm256_cvtepi8_epi32(i8A);
                __m256 fA = _mm256_cvtepi32_ps(i32A);
                fA = _mm256_mul_ps(fA, vscaleA);

                // Load 8 int8 from B
                __m128i i8B = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&blockB.qs[i * 8]));
                __m256i i32B = _mm256_cvtepi8_epi32(i8B);
                __m256 fB = _mm256_cvtepi32_ps(i32B);
                fB = _mm256_mul_ps(fB, vscaleB);

                // Load cos/sin
                __m256 vcos = _mm256_loadu_ps(&cos_cache[offset + i * 8]);
                __m256 vsin = _mm256_loadu_ps(&sin_cache[offset + i * 8]);

                // Rotate: x' = x cos - y sin, y' = x sin + y cos
                __m256 fA_new = _mm256_sub_ps(
                    _mm256_mul_ps(fA, vcos),
                    _mm256_mul_ps(fB, vsin));
                __m256 fB_new = _mm256_add_ps(
                    _mm256_mul_ps(fA, vsin),
                    _mm256_mul_ps(fB, vcos));

                // Store to temp
                _mm256_store_ps(&rotA[i * 8], fA_new);
                _mm256_store_ps(&rotB[i * 8], fB_new);
            }

            // 2. Requantize Block A
            {
                // Find max abs
                __m256 vmax = _mm256_setzero_ps();
                __m256 vabs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
                for (int i = 0; i < 4; ++i)
                {
                    __m256 v = _mm256_load_ps(&rotA[i * 8]);
                    v = _mm256_and_ps(v, vabs_mask);
                    vmax = _mm256_max_ps(vmax, v);
                }
                // Horizontal max
                float max_arr[8];
                _mm256_storeu_ps(max_arr, vmax);
                float max_abs = 0.0f;
                for (int i = 0; i < 8; ++i)
                    max_abs = std::max(max_abs, max_arr[i]);

                // Compute scale
                float d = max_abs / 127.0f;
                blockA.d = fp32_to_fp16(d);
                float inv_d = (d > 1e-10f) ? 1.0f / d : 0.0f;
                __m256 vinv_d = _mm256_set1_ps(inv_d);

                // Quantize and Sum
                int32_t sum = 0;
                __m256 vmin = _mm256_set1_ps(-127.0f);
                __m256 vmax_clamp = _mm256_set1_ps(127.0f);

                for (int i = 0; i < 4; ++i)
                {
                    __m256 v = _mm256_load_ps(&rotA[i * 8]);
                    v = _mm256_mul_ps(v, vinv_d);
                    v = _mm256_max_ps(vmin, _mm256_min_ps(vmax_clamp, v));
                    v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256i vi = _mm256_cvtps_epi32(v);

                    // Pack 32-bit integers to 8-bit
                    // AVX2 packing is a bit involved, scalar store is fine for 32 elements
                    // But let's do scalar store/sum for simplicity and correctness
                    int32_t tmp[8];
                    _mm256_storeu_si256((__m256i *)tmp, vi);
                    for (int j = 0; j < 8; ++j)
                    {
                        int8_t q = static_cast<int8_t>(tmp[j]);
                        blockA.qs[i * 8 + j] = q;
                        sum += q;
                    }
                }
                blockA.sum_qs = static_cast<int16_t>(sum);
            }

            // 3. Requantize Block B (Copy-paste logic, could be helper)
            {
                __m256 vmax = _mm256_setzero_ps();
                __m256 vabs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
                for (int i = 0; i < 4; ++i)
                {
                    __m256 v = _mm256_load_ps(&rotB[i * 8]);
                    v = _mm256_and_ps(v, vabs_mask);
                    vmax = _mm256_max_ps(vmax, v);
                }
                float max_arr[8];
                _mm256_storeu_ps(max_arr, vmax);
                float max_abs = 0.0f;
                for (int i = 0; i < 8; ++i)
                    max_abs = std::max(max_abs, max_arr[i]);

                float d = max_abs / 127.0f;
                blockB.d = fp32_to_fp16(d);
                float inv_d = (d > 1e-10f) ? 1.0f / d : 0.0f;
                __m256 vinv_d = _mm256_set1_ps(inv_d);

                int32_t sum = 0;
                __m256 vmin = _mm256_set1_ps(-127.0f);
                __m256 vmax_clamp = _mm256_set1_ps(127.0f);

                for (int i = 0; i < 4; ++i)
                {
                    __m256 v = _mm256_load_ps(&rotB[i * 8]);
                    v = _mm256_mul_ps(v, vinv_d);
                    v = _mm256_max_ps(vmin, _mm256_min_ps(vmax_clamp, v));
                    v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256i vi = _mm256_cvtps_epi32(v);

                    int32_t tmp[8];
                    _mm256_storeu_si256((__m256i *)tmp, vi);
                    for (int j = 0; j < 8; ++j)
                    {
                        int8_t q = static_cast<int8_t>(tmp[j]);
                        blockB.qs[i * 8 + j] = q;
                        sum += q;
                    }
                }
                blockB.sum_qs = static_cast<int16_t>(sum);
            }
        }
#else
        // Scalar Fallback
        for (int b = 0; b < blocks_per_half; ++b)
        {
            Q8_1Block &blockA = head_ptr[b];
            Q8_1Block &blockB = head_ptr[b + blocks_per_half];

            float scaleA = fp16_to_fp32(blockA.d);
            float scaleB = fp16_to_fp32(blockB.d);

            float rotA[32];
            float rotB[32];
            int offset = b * 32;

            for (int i = 0; i < 32; ++i)
            {
                float fA = scaleA * blockA.qs[i];
                float fB = scaleB * blockB.qs[i];
                float cos_val = cos_cache[offset + i];
                float sin_val = sin_cache[offset + i];

                rotA[i] = fA * cos_val - fB * sin_val;
                rotB[i] = fA * sin_val + fB * cos_val;
            }

            // Requantize A
            {
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                    max_abs = std::max(max_abs, std::abs(rotA[i]));
                float d = max_abs / 127.0f;
                blockA.d = fp32_to_fp16(d);
                float inv_d = (d > 1e-10f) ? 1.0f / d : 0.0f;
                int32_t sum = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int8_t q = (int8_t)std::round(std::max(-127.0f, std::min(127.0f, rotA[i] * inv_d)));
                    blockA.qs[i] = q;
                    sum += q;
                }
                blockA.sum_qs = sum;
            }

            // Requantize B
            {
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                    max_abs = std::max(max_abs, std::abs(rotB[i]));
                float d = max_abs / 127.0f;
                blockB.d = fp32_to_fp16(d);
                float inv_d = (d > 1e-10f) ? 1.0f / d : 0.0f;
                int32_t sum = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int8_t q = (int8_t)std::round(std::max(-127.0f, std::min(127.0f, rotB[i] * inv_d)));
                    blockB.qs[i] = q;
                    sum += q;
                }
                blockB.sum_qs = sum;
            }
        }
#endif
    }

    template <typename TensorT>
    bool CPURoPEKernelT<TensorT>::apply_q8_1(
        void *Q_q8_1, void *K_q8_1,
        const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float rope_theta,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
        {
            if (device_idx != -1)
                return false;
            if (head_dim % Q8_1Block::BLOCK_SIZE != 0)
                return false;

            Q8_1Block *Q_blocks = static_cast<Q8_1Block *>(Q_q8_1);
            Q8_1Block *K_blocks = K_q8_1 ? static_cast<Q8_1Block *>(K_q8_1) : nullptr;

            int blocks_per_head = head_dim / Q8_1Block::BLOCK_SIZE;
            int q_stride_blocks = n_heads * blocks_per_head;
            int k_stride_blocks = n_kv_heads * blocks_per_head;

#pragma omp parallel for if (seq_len > 1)
            for (int tok = 0; tok < seq_len; ++tok)
            {
                int position = position_ids ? position_ids[tok] : tok;
                if (position < 0)
                    continue;

                // Ensure cache is valid for this position
                if (tls_state_.last_pos != position)
                {
                    primitives::update_rope_cache(
                        head_dim, rope_theta, position, tls_state_);
                }

                const float *cos_cache = tls_state_.cos_curr.data();
                const float *sin_cache = tls_state_.sin_curr.data();

                // Debug check
                if (cos_cache == nullptr || sin_cache == nullptr)
                {
                    printf("Error: cos/sin cache is null for pos %d\n", position);
                    continue;
                }

                // Process Q heads
                for (int h = 0; h < n_heads; ++h)
                {
                    Q8_1Block *head_ptr = Q_blocks + tok * q_stride_blocks + h * blocks_per_head;
                    process_q8_1_head_fused(head_ptr, head_dim, cos_cache, sin_cache);
                }

                // Process K heads
                if (K_blocks)
                {
                    for (int h = 0; h < n_kv_heads; ++h)
                    {
                        Q8_1Block *head_ptr = K_blocks + tok * k_stride_blocks + h * blocks_per_head;
                        process_q8_1_head_fused(head_ptr, head_dim, cos_cache, sin_cache);
                    }
                }
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    // Explicit instantiations
    template class CPURoPEKernelT<FP32Tensor>;
    template class CPURoPEKernelT<BF16Tensor>;
    template class CPURoPEKernelT<FP16Tensor>;
    template class CPURoPEKernelT<Q8_1Tensor>;

} // namespace llaminar2