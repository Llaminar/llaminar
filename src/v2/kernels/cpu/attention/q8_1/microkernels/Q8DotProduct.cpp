/**
 * @file Q8DotProduct.cpp
 * @brief Reference and SIMD implementations of Q8_1 dot product
 * @author David Sanftenberg
 */

#include "Q8DotProduct.h"
#include "../../../../../tensors/FP16Utils.h"
#include "../../../../../utils/CPUFeatures.h"

#include <cstdint>
#include <cstring>

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
#include <immintrin.h>
#endif

namespace llaminar::v2::kernels::microkernels
{

    Q8DotProductResult q8_dot_product_ref(const Q8DotProductParams &params)
    {
        float score = 0.0f;

        for (int b = 0; b < params.num_blocks; ++b)
        {
            const Q8_1Block &q_block = params.q_blocks[b];
            const Q8_1Block &k_block = params.k_blocks[b];

            // Get per-block scales
            float d_q = llaminar2::fp16_to_fp32(q_block.d);
            float d_k = llaminar2::fp16_to_fp32(k_block.d);
            float block_scale = d_q * d_k;

            // Compute integer dot product with unsigned conversion
            // vpdpbusd semantics: unsigned × signed accumulation
            // We convert Q to unsigned by adding 128, then correct with sum_qs
            int32_t dot = 0;
            for (int i = 0; i < 32; ++i)
            {
                // Q is signed int8: range [-128, 127]
                // K is signed int8: range [-128, 127]
                // For vpdpbusd: convert Q to unsigned by adding 128
                uint8_t q_unsigned = static_cast<uint8_t>(static_cast<int16_t>(q_block.qs[i]) + 128);
                int8_t k_signed = k_block.qs[i];

                // unsigned × signed product
                dot += static_cast<int32_t>(q_unsigned) * static_cast<int32_t>(k_signed);
            }

            // Correct for the +128 bias: subtract 128 * sum(k)
            // sum(q_unsigned * k) = sum((q + 128) * k) = sum(q*k) + 128*sum(k)
            // So: sum(q*k) = sum(q_unsigned * k) - 128*sum(k)
            dot -= 128 * static_cast<int32_t>(k_block.sum_qs);

            score += static_cast<float>(dot) * block_scale;
        }

        return Q8DotProductResult{score * params.global_scale};
    }

#if defined(__AVX512F__) && defined(__AVX512VNNI__)

    Q8DotProductResult q8_dot_product_avx512(const Q8DotProductParams &params)
    {
        // AVX-512 VNNI implementation using vpdpbusd
        // Process 64 elements at a time (2 blocks)

        __m512i acc = _mm512_setzero_si512();
        float total_score = 0.0f;

        // Constant for unsigned conversion: 128 broadcasted
        const __m512i bias_128 = _mm512_set1_epi8((char)128);

        int b = 0;

        // Process pairs of blocks (64 bytes each)
        for (; b + 1 < params.num_blocks; b += 2)
        {
            const Q8_1Block &q0 = params.q_blocks[b];
            const Q8_1Block &q1 = params.q_blocks[b + 1];
            const Q8_1Block &k0 = params.k_blocks[b];
            const Q8_1Block &k1 = params.k_blocks[b + 1];

            // Load Q blocks (32 bytes each) into lower and upper halves
            __m256i q0_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q0.qs));
            __m256i q1_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q1.qs));
            __m512i q_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q0_256), q1_256, 1);

            // Load K blocks
            __m256i k0_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k0.qs));
            __m256i k1_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k1.qs));
            __m512i k_512 = _mm512_inserti64x4(_mm512_castsi256_si512(k0_256), k1_256, 1);

            // Convert Q to unsigned: q_unsigned = q + 128
            __m512i q_unsigned = _mm512_add_epi8(q_512, bias_128);

            // vpdpbusd: acc += (q_unsigned[i] * k[i]) for 4-element groups, sum to int32
            acc = _mm512_dpbusd_epi32(acc, q_unsigned, k_512);

            // Extract partial sums and apply scales
            // We have 16 int32 accumulators, need to reduce and scale
            int32_t acc_array[16];
            _mm512_storeu_si512(acc_array, acc);

            // Block 0 uses accumulators 0-7, Block 1 uses 8-15
            int32_t dot0 = acc_array[0] + acc_array[1] + acc_array[2] + acc_array[3] +
                           acc_array[4] + acc_array[5] + acc_array[6] + acc_array[7];
            int32_t dot1 = acc_array[8] + acc_array[9] + acc_array[10] + acc_array[11] +
                           acc_array[12] + acc_array[13] + acc_array[14] + acc_array[15];

            // Apply bias correction
            dot0 -= 128 * static_cast<int32_t>(k0.sum_qs);
            dot1 -= 128 * static_cast<int32_t>(k1.sum_qs);

            // Scale and accumulate
            float d_q0 = llaminar2::fp16_to_fp32(q0.d);
            float d_k0 = llaminar2::fp16_to_fp32(k0.d);
            float d_q1 = llaminar2::fp16_to_fp32(q1.d);
            float d_k1 = llaminar2::fp16_to_fp32(k1.d);

            total_score += static_cast<float>(dot0) * d_q0 * d_k0;
            total_score += static_cast<float>(dot1) * d_q1 * d_k1;

            // Reset accumulator for next pair
            acc = _mm512_setzero_si512();
        }

        // Handle remaining block (if odd number)
        for (; b < params.num_blocks; ++b)
        {
            const Q8_1Block &q_block = params.q_blocks[b];
            const Q8_1Block &k_block = params.k_blocks[b];

            // Load 32 bytes
            __m256i q_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q_block.qs));
            __m256i k_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_block.qs));

            // Extend to 512-bit (zero upper half)
            __m512i q_512 = _mm512_castsi256_si512(q_256);
            __m512i k_512 = _mm512_castsi256_si512(k_256);

            // Convert Q to unsigned
            __m512i bias_256 = _mm512_castsi256_si512(_mm256_set1_epi8((char)128));
            __m512i q_unsigned = _mm512_add_epi8(q_512, bias_256);

            // vpdpbusd
            __m512i local_acc = _mm512_setzero_si512();
            local_acc = _mm512_dpbusd_epi32(local_acc, q_unsigned, k_512);

            // Horizontal sum of lower 8 int32s
            int32_t acc_array[16];
            _mm512_storeu_si512(acc_array, local_acc);
            int32_t dot = acc_array[0] + acc_array[1] + acc_array[2] + acc_array[3] +
                          acc_array[4] + acc_array[5] + acc_array[6] + acc_array[7];

            // Bias correction
            dot -= 128 * static_cast<int32_t>(k_block.sum_qs);

            // Scale
            float d_q = llaminar2::fp16_to_fp32(q_block.d);
            float d_k = llaminar2::fp16_to_fp32(k_block.d);
            total_score += static_cast<float>(dot) * d_q * d_k;
        }

        return Q8DotProductResult{total_score * params.global_scale};
    }

#else

    Q8DotProductResult q8_dot_product_avx512(const Q8DotProductParams &params)
    {
        // Fallback to reference if AVX-512 VNNI not available
        return q8_dot_product_ref(params);
    }

#endif

    Q8DotProductResult q8_dot_product(const Q8DotProductParams &params)
    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
        if (llaminar2::cpu_supports_avx512_vnni())
        {
            return q8_dot_product_avx512(params);
        }
#endif
        return q8_dot_product_ref(params);
    }

} // namespace llaminar::v2::kernels::microkernels
