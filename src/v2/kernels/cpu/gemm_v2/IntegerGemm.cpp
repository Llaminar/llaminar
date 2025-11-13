/**
 * @file IntegerGemm.cpp
 * @brief Integer-domain quantized GEMM implementation
 * @author David Sanftenberg
 * @date November 2025
 */

#include "IntegerGemm.h"
#include "tensors/FP16Utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace kernels
    {

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
        /**
         * @brief Horizontally sum all 8 int32 lanes in a __m256i vector
         */
        static inline int32_t horizontal_sum_epi32(__m256i v)
        {
            __m128i low = _mm256_castsi256_si128(v);
            __m128i high = _mm256_extracti128_si256(v, 1);
            __m128i sum = _mm_add_epi32(low, high);
            sum = _mm_hadd_epi32(sum, sum);
            sum = _mm_hadd_epi32(sum, sum);
            return _mm_cvtsi128_si32(sum);
        }

        /**
         * @brief IQ4_NL nibble lookup table replicated from llama.cpp
         */
        alignas(16) static const int8_t IQ4NL_LUT[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113};

        /**
         * @brief Decode one IQ4_NL block (32 nibbles) into a 256-bit int8 vector
         */
        static inline __m256i decode_iq4nl_block_to_vec(const IQ4_NLBlock &block)
        {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            const __m128i mask = _mm_set1_epi8(0x0F);

            const __m128i low = _mm_and_si128(packed, mask);
            const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);

            __m256i indices = _mm256_castsi128_si256(low);
            indices = _mm256_inserti128_si256(indices, high, 1);

            const __m128i table128 = _mm_load_si128(reinterpret_cast<const __m128i *>(IQ4NL_LUT));
            const __m256i table256 = _mm256_broadcastsi128_si256(table128);

            return _mm256_shuffle_epi8(table256, indices);
        }

        /**
         * @brief Sum 32 signed bytes contained in a 256-bit vector
         */
        static inline int32_t sum_int8_vec(__m256i vec)
        {
            const __m128i lo = _mm256_castsi256_si128(vec);
            const __m128i hi = _mm256_extracti128_si256(vec, 1);
            const __m512i lo32 = _mm512_cvtepi8_epi32(lo);
            const __m512i hi32 = _mm512_cvtepi8_epi32(hi);
            return _mm512_reduce_add_epi32(lo32) + _mm512_reduce_add_epi32(hi32);
        }
#endif

        // ========== FP32 → Q8_0 Quantization ==========

        void quantize_fp32_to_q8_0(const float *src, Q8_0Block *dst, size_t n)
        {
            constexpr size_t BLOCK_SIZE = Q8_0Block::BLOCK_SIZE; // 32
            const size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (size_t b = 0; b < num_blocks; ++b)
            {
                const size_t offset = b * BLOCK_SIZE;
                const size_t block_len = std::min(BLOCK_SIZE, n - offset);

                // Find max absolute value in block
                float amax = 0.0f;
                for (size_t i = 0; i < block_len; ++i)
                {
                    amax = std::max(amax, std::fabs(src[offset + i]));
                }

                if (amax == 0.0f)
                {
                    for (size_t i = 0; i < BLOCK_SIZE; ++i)
                    {
                        dst[b].qs[i] = 0;
                    }
                    dst[b].d = fp32_to_fp16(0.0f);
                    continue;
                }

                int8_t q_temp[BLOCK_SIZE];
                float scale = amax / 127.0f;
                float inv_scale = 1.0f / scale;

                // Two-pass refinement: initial quantization followed by optimal scale fit
                for (int iter = 0; iter < 2; ++iter)
                {
                    for (size_t i = 0; i < block_len; ++i)
                    {
                        float val = src[offset + i] * inv_scale;
                        int32_t quant = static_cast<int32_t>(std::round(val));
                        quant = std::max(-127, std::min(127, quant));
                        q_temp[i] = static_cast<int8_t>(quant);
                    }
                    for (size_t i = block_len; i < BLOCK_SIZE; ++i)
                    {
                        q_temp[i] = 0;
                    }

                    if (iter == 0)
                    {
                        double numerator = 0.0;
                        double denominator = 0.0;
                        for (size_t i = 0; i < block_len; ++i)
                        {
                            double x = static_cast<double>(src[offset + i]);
                            double q = static_cast<double>(q_temp[i]);
                            numerator += x * q;
                            denominator += q * q;
                        }
                        if (denominator > 0.0)
                        {
                            scale = static_cast<float>(numerator / denominator);
                            if (scale != 0.0f)
                            {
                                inv_scale = 1.0f / scale;
                            }
                            else
                            {
                                inv_scale = 0.0f;
                            }
                        }
                        else
                        {
                            // All zeros, break early
                            break;
                        }
                    }
                }

                // Commit refined quantization to destination block
                for (size_t i = 0; i < BLOCK_SIZE; ++i)
                {
                    dst[b].qs[i] = q_temp[i];
                }
                dst[b].d = fp32_to_fp16(scale);
            }
        }

        // ========== IQ4_NL × Q8_0 INT8 GEMM (AVX512-VNNI) ==========

#if defined(__AVX512F__) && defined(__AVX512VNNI__)

        bool gemm_int8_iq4nl_vnni(
            const float *A,
            const IQ4_NLTensor *B,
            float *C,
            int M, int N, int K,
            int lda, int ldc)
        {
            // Check VNNI support
            if (!cpu_supports_avx512_vnni())
            {
                return false;
            }

            constexpr size_t BLOCK_SIZE = 32;
            const size_t k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Get B tensor shape and data
            const auto &b_shape = B->shape();
            if (b_shape.size() != 2 || b_shape[1] != static_cast<size_t>(K))
            {
                std::cerr << "[IntegerGemm] Error: B shape mismatch\n";
                return false;
            }

            const size_t b_rows = b_shape[0]; // N (transposed)
            if (b_rows != static_cast<size_t>(N))
            {
                std::cerr << "[IntegerGemm] Error: N dimension mismatch\n";
                return false;
            }

            const uint8_t *b_raw = B->raw_data();
            const IQ4_NLBlock *b_blocks = reinterpret_cast<const IQ4_NLBlock *>(b_raw);

            // Allocate Q8_0 buffer for ALL activations (M × k_blocks)
            // Pre-quantize entire A matrix ONCE (not per-row)
            std::vector<Q8_0Block> a_q8_all(M * k_blocks);

            for (int m = 0; m < M; ++m)
            {
                const float *a_row = A + m * lda;
                Q8_0Block *a_q8_row = &a_q8_all[m * k_blocks];
                quantize_fp32_to_q8_0(a_row, a_q8_row, K);
            }

            // ========== OPTIMIZED GEMM LOOP ==========
            // Both A and B are now pre-quantized to INT8!

            const __m512i bias128 = _mm512_set1_epi8((char)0x80);

            for (int m = 0; m < M; ++m)
            {
                const Q8_0Block *a_q8_row = &a_q8_all[m * k_blocks];

                for (int n = 0; n < N; ++n)
                {
                    double scale_acc = 0.0;
                    const IQ4_NLBlock *b_row_blocks = b_blocks + n * k_blocks;

                    size_t kb = 0;

                    // Process two blocks (64 INT8 elements) at a time to maximize VNNI throughput
                    for (; kb + 1 < k_blocks; kb += 2)
                    {
                        const Q8_0Block &a_block0 = a_q8_row[kb];
                        const Q8_0Block &a_block1 = a_q8_row[kb + 1];
                        const IQ4_NLBlock &b_block0 = b_row_blocks[kb];
                        const IQ4_NLBlock &b_block1 = b_row_blocks[kb + 1];

                        // Load activation blocks into a single 512-bit vector
                        __m256i a_vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_block0.qs));
                        __m256i a_vec_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_block1.qs));
                        __m512i a_vec = _mm512_castsi256_si512(a_vec_lo);
                        a_vec = _mm512_inserti64x4(a_vec, a_vec_hi, 1);

                        // Decode weight blocks into INT8 vectors on-the-fly
                        __m256i b_vec_lo = decode_iq4nl_block_to_vec(b_block0);
                        __m256i b_vec_hi = decode_iq4nl_block_to_vec(b_block1);
                        __m512i b_vec = _mm512_castsi256_si512(b_vec_lo);
                        b_vec = _mm512_inserti64x4(b_vec, b_vec_hi, 1);

                        // Convert activations from signed to unsigned (VNNI requirement)
                        __m512i a_unsigned = _mm512_add_epi8(a_vec, bias128);

                        // Compute correction terms per block (128 * Σ(B_signed))
                        int32_t b_sum0 = sum_int8_vec(b_vec_lo);
                        int32_t b_sum1 = sum_int8_vec(b_vec_hi);

                        __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a_unsigned, b_vec);
                        __m256i acc_lo = _mm512_castsi512_si256(acc);
                        __m256i acc_hi = _mm512_extracti64x4_epi64(acc, 1);

                        int32_t dot0 = horizontal_sum_epi32(acc_lo) - (128 * b_sum0);
                        int32_t dot1 = horizontal_sum_epi32(acc_hi) - (128 * b_sum1);

                        double a_scale0 = static_cast<double>(fp16_to_fp32(a_block0.d));
                        double b_scale0 = static_cast<double>(fp16_to_fp32(b_block0.d));
                        double a_scale1 = static_cast<double>(fp16_to_fp32(a_block1.d));
                        double b_scale1 = static_cast<double>(fp16_to_fp32(b_block1.d));
                        double scale0 = a_scale0 * b_scale0;
                        double scale1 = a_scale1 * b_scale1;

                        scale_acc += static_cast<double>(dot0) * scale0;
                        scale_acc += static_cast<double>(dot1) * scale1;
                    }

                    // Handle tail block if K is odd multiple of BLOCK_SIZE
                    for (; kb < k_blocks; ++kb)
                    {
                        const Q8_0Block &a_block = a_q8_row[kb];
                        const IQ4_NLBlock &b_block = b_row_blocks[kb];

                        __m256i a_vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_block.qs));
                        __m256i a_vec_hi = _mm256_setzero_si256();
                        __m512i a_vec = _mm512_castsi256_si512(a_vec_lo);
                        a_vec = _mm512_inserti64x4(a_vec, a_vec_hi, 1);

                        __m256i b_vec_lo = decode_iq4nl_block_to_vec(b_block);
                        __m256i b_vec_hi = _mm256_setzero_si256();
                        __m512i b_vec = _mm512_castsi256_si512(b_vec_lo);
                        b_vec = _mm512_inserti64x4(b_vec, b_vec_hi, 1);

                        __m512i a_unsigned = _mm512_add_epi8(a_vec, bias128);

                        int32_t b_sum = sum_int8_vec(b_vec_lo);

                        __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a_unsigned, b_vec);
                        __m256i acc_lo = _mm512_castsi512_si256(acc);
                        int32_t dot = horizontal_sum_epi32(acc_lo) - (128 * b_sum);

                        double a_scale = static_cast<double>(fp16_to_fp32(a_block.d));
                        double b_scale = static_cast<double>(fp16_to_fp32(b_block.d));
                        double combined_scale = a_scale * b_scale;
                        scale_acc += static_cast<double>(dot) * combined_scale;
                    }

                    C[m * ldc + n] = static_cast<float>(scale_acc);
                }
            }

            return true;
        }

#else // No AVX512-VNNI support

        bool gemm_int8_iq4nl_vnni(
            const float *,
            const IQ4_NLTensor *,
            float *,
            int, int, int,
            int, int)
        {
            std::cerr << "[IntegerGemm] Error: Compiled without AVX512-VNNI support\n";
            std::cerr << "[IntegerGemm] Rebuild with -mavx512f -mavx512vnni\n";
            return false;
        }

#endif // __AVX512F__ && __AVX512VNNI__

        // ========== Q6_K Integer GEMM (Placeholder) ==========

        bool gemm_int8_q6k_vnni(
            const float *,
            const Q6_KTensor *,
            float *,
            int, int, int,
            int, int)
        {
            std::cerr << "[IntegerGemm] Q6_K variant not yet implemented\n";
            return false;
        }

        // ========== Generic Dispatcher ==========

        bool gemm_int8_dispatch(
            const float *A,
            const TensorBase *B,
            float *C,
            int M, int N, int K)
        {
            if (!B)
            {
                return false;
            }

            const TensorType b_type = B->native_type();

            switch (b_type)
            {
            case TensorType::IQ4_NL:
            {
                const IQ4_NLTensor *b_iq4nl = dynamic_cast<const IQ4_NLTensor *>(B);
                if (!b_iq4nl)
                    return false;
                return gemm_int8_iq4nl_vnni(A, b_iq4nl, C, M, N, K, K, N);
            }

            case TensorType::Q6_K:
            {
                const Q6_KTensor *b_q6k = dynamic_cast<const Q6_KTensor *>(B);
                if (!b_q6k)
                    return false;
                return gemm_int8_q6k_vnni(A, b_q6k, C, M, N, K, K, N);
            }

            default:
                std::cerr << "[IntegerGemm] Unsupported weight format for integer GEMM\n";
                return false;
            }
        }

    } // namespace kernels
} // namespace llaminar2
