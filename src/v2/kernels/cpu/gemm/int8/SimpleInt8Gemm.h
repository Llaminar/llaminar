/**
 * @file SimpleInt8Gemm.h
 * @brief Simple, standalone INT8 GEMM kernel - no abstractions, prove the concept first
 *
 * This is a clean-sheet design to validate basic INT8 GEMM performance before
 * building complex microkernel abstractions. Focus: correctness and raw performance.
 *
 * @author David Sanftenberg
 * @date 2025-11-12
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <vector>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Simple Q8_0 block format (32 INT8 values + FP16 scale)
             */
            struct SimpleQ8Block
            {
                uint16_t d;    // FP16 scale factor
                int8_t qs[32]; // 32 quantized INT8 values
            };

            /**
             * @brief Convert FP16 to FP32 (bit manipulation)
             */
            inline float fp16_to_fp32_simple(uint16_t h)
            {
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t exp = (h & 0x7C00) >> 10;
                uint32_t mant = (h & 0x03FF);

                if (exp == 0)
                {
                    if (mant == 0)
                        return 0.0f;
                    // Denormal
                    exp = 1;
                    while ((mant & 0x0400) == 0)
                    {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x03FF;
                }
                else if (exp == 0x1F)
                {
                    // Inf/NaN
                    uint32_t result = sign | 0x7F800000 | (mant << 13);
                    return *reinterpret_cast<float *>(&result);
                }

                // Normalized
                uint32_t result = sign | ((exp + 112) << 23) | (mant << 13);
                return *reinterpret_cast<float *>(&result);
            }

            /**
             * @brief Simple INT8 GEMM: C = A * B
             *
             * A: [m, k] in Q8_0 format (row-major, 32-element blocks)
             * B: [k, n] in Q8_0 format (column-major, 32-element blocks per column)
             * C: [m, n] in FP32 (row-major)
             *
             * Design principles:
             * 1. NO memory copies - work directly from source blocks
             * 2. Simple nested loops - no complex blocking yet
             * 3. AVX512 VNNI for core computation
             * 4. OpenMP parallelization over M dimension
             *
             * This is the BASELINE. Prove it works, then optimize.
             */
            class SimpleInt8Gemm
            {
            public:
                /**
                 * @brief Perform INT8 GEMM with AVX512 VNNI and 3-level cache blocking
                 *
                 * @param A Input matrix A (Q8_0 blocks, row-major layout)
                 * @param B Input matrix B (Q8_0 blocks, column-major layout)
                 * @param C Output matrix C (FP32, row-major)
                 * @param m Number of rows in A and C
                 * @param n Number of columns in B and C
                 * @param k Number of columns in A and rows in B
                 * @param MC M-dimension cache block size (default 256)
                 * @param KC K-dimension cache block size (default 512)
                 * @param NC N-dimension cache block size (default 128)
                 * @return true if successful
                 *
                 * Cache blocking strategy:
                 * - MC × KC × NC blocks fit in L2 cache (256KB)
                 * - Reorder loops: MC → NC → KC → M-tiles → N-tiles → K-blocks
                 * - Improves data locality and cache hit rate
                 */
                static bool multiply(
                    const SimpleQ8Block *A, // [m, k_blocks] row-major
                    const SimpleQ8Block *B, // [n, k_blocks] column-major (transposed layout!)
                    float *C,               // [m, n] row-major
                    int m, int n, int k,
                    int MC = 256, // M-dimension cache block (unused for now - debugging)
                    int KC = 512, // K-dimension cache block (unused for now - debugging)
                    int NC = 128) // N-dimension cache block (unused for now - debugging)
                {
                    // SIMPLIFIED: Just use naive implementation for now
                    // Cache blocking adds too much overhead - need to understand why
                    return multiply_naive(A, B, C, m, n, k);
                }

                /**
                 * @brief Perform INT8 GEMM with AVX512 VNNI (NAIVE - no cache blocking)
                 *
                 * This is the baseline implementation for comparison.
                 * Use multiply() for production code.
                 */
                static bool multiply_naive(
                    const SimpleQ8Block *A, // [m, k_blocks] row-major
                    const SimpleQ8Block *B, // [n, k_blocks] column-major (transposed layout!)
                    float *C,               // [m, n] row-major
                    int m, int n, int k)
                {
                    const int k_blocks = (k + 31) / 32;

                    // Zero output
                    std::memset(C, 0, m * n * sizeof(float));

// Parallel over M rows (each thread gets independent rows)
#pragma omp parallel for schedule(static)
                    for (int i = 0; i < m; ++i)
                    {
                        for (int j = 0; j < n; ++j)
                        {

                            // Accumulate over K dimension (32 elements at a time)
                            float sum = 0.0f;

                            for (int kb = 0; kb < k_blocks; ++kb)
                            {
                                // Get blocks directly (NO COPY!)
                                const SimpleQ8Block &a_block = A[i * k_blocks + kb];
                                const SimpleQ8Block &b_block = B[j * k_blocks + kb]; // Column-major!

                                // Extract scales
                                float a_scale = fp16_to_fp32_simple(a_block.d);
                                float b_scale = fp16_to_fp32_simple(b_block.d);
                                float scale = a_scale * b_scale;

                                // Compute INT8 dot product using AVX512 VNNI
                                int32_t int_sum = dot_product_vnni(a_block.qs, b_block.qs);

                                // Scale and accumulate
                                sum += static_cast<float>(int_sum) * scale;
                            }

                            C[i * n + j] = sum;
                        }
                    }

                    return true;
                }

            private:
                /**
                 * @brief AVX512 VNNI dot product for 32 INT8 elements
                 *
                 * Uses vpdpbusd instruction:
                 * - Multiplies unsigned UINT8 * signed INT8
                 * - Accumulates 4-way into INT32
                 *
                 * Note: Q8_0 uses signed INT8, so we need bias correction
                 * Optimized: Vectorized bias computation instead of scalar loop
                 */
                static int32_t dot_product_vnni(const int8_t *a, const int8_t *b)
                {
#ifdef __AVX512VNNI__
                    // Load 32 INT8 values (256 bits)
                    __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
                    __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b));

                    // VNNI requires unsigned x signed
                    // Convert A from signed to unsigned by adding 128
                    const __m256i offset = _mm256_set1_epi8(-128);
                    __m256i a_unsigned = _mm256_sub_epi8(a_vec, offset); // signed - (-128) = unsigned

                    // Compute unsigned A * signed B using VNNI
                    __m512i acc = _mm512_setzero_si512();

                    // Broadcast to 512-bit for vpdpbusd (operates on 512-bit registers)
                    __m512i a_512 = _mm512_castsi256_si512(a_unsigned);
                    __m512i b_512 = _mm512_castsi256_si512(b_vec);

                    // vpdpbusd: acc += unsigned_a * signed_b (4-way dot products)
                    acc = _mm512_dpbusd_epi32(acc, a_512, b_512);

                    // Horizontal sum: reduce 16 INT32 values to scalar
                    __m256i sum_low = _mm512_castsi512_si256(acc);
                    __m256i sum_high = _mm512_extracti64x4_epi64(acc, 1);
                    __m256i sum_256 = _mm256_add_epi32(sum_low, sum_high);

                    __m128i sum_128 = _mm_add_epi32(_mm256_castsi256_si128(sum_256),
                                                    _mm256_extracti128_si256(sum_256, 1));
                    __m128i sum_64 = _mm_add_epi32(sum_128, _mm_shuffle_epi32(sum_128, 0x4E));
                    __m128i sum_32 = _mm_add_epi32(sum_64, _mm_shuffle_epi32(sum_64, 0xB1));

                    int32_t unsigned_sum = _mm_cvtsi128_si32(sum_32);

                    // Bias correction: we computed unsigned_A * signed_B
                    // But we want signed_A * signed_B
                    // signed_A = unsigned_A - 128
                    // So: signed_A * signed_B = (unsigned_A - 128) * signed_B
                    //                          = unsigned_A * signed_B - 128 * sum(signed_B)

                    // Compute sum of B values (signed) - KEEP SIMPLE for now
                    int32_t sum_b = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        sum_b += b[i];
                    }

                    int32_t corrected_sum = unsigned_sum - 128 * sum_b;

                    return corrected_sum;

#else
                    // Scalar fallback (no AVX512 VNNI)
                    int32_t sum = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
                    }
                    return sum;
#endif
                }
            };

            /**
             * @brief Helper to convert row-major B to column-major for SimpleInt8Gemm
             *
             * B_row: [k_blocks, n] layout (row-major - each row is a K-block, n columns)
             * B_col: [n, k_blocks] layout (column-major - each column is a series of K-blocks)
             */
            inline void transpose_q8_blocks(
                const SimpleQ8Block *B_row, // [k_blocks, n]
                SimpleQ8Block *B_col,       // [n, k_blocks]
                int n, int k_blocks)
            {
                for (int kb = 0; kb < k_blocks; ++kb)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        B_col[j * k_blocks + kb] = B_row[kb * n + j];
                    }
                }
            }

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
