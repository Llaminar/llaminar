/**
 * @file PackedInt8Gemm.h
 * @brief INT8 GEMM with data packing for maximum performance
 * @author David Sanftenberg
 *
 * Strategy: Pre-pack B matrix into cache-optimized format ONCE,
 * then run highly optimized GEMM on packed data.
 *
 * This trades memory (expanded format) for speed (no repeated decoding).
 */

#pragma once

#include "SimpleInt8Gemm.h" // For SimpleQ8Block definition
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <memory>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Packed INT8 matrix with separated scales
             *
             * Layout: Column-major INT8 data with per-column scales
             * This eliminates repeated FP16→FP32 conversions and allows
             * better vectorization.
             */
            struct PackedInt8Matrix
            {
                std::unique_ptr<int8_t[]> data;      // Column-major INT8 [k, n]
                std::unique_ptr<float[]> col_scales; // Per-column scales [n]
                int rows;                            // K dimension
                int cols;                            // N dimension

                PackedInt8Matrix(int k, int n) : rows(k), cols(n)
                {
                    data = std::make_unique<int8_t[]>(k * n);
                    col_scales = std::make_unique<float[]>(n);
                }
            };

            class PackedInt8Gemm
            {
            public:
                /**
                 * @brief Pack Q8_0 matrix into optimized INT8 format
                 *
                 * Converts from blocked format to plain column-major INT8 with scales.
                 * This should be done ONCE during weight loading, not per-inference.
                 */
                static std::unique_ptr<PackedInt8Matrix> pack_b_matrix(
                    const SimpleQ8Block *B_blocks, // [n, k_blocks] column-major blocks
                    int n, int k)
                {
                    const int k_blocks = (k + 31) / 32;
                    auto packed = std::make_unique<PackedInt8Matrix>(k, n);

// Process each column
#pragma omp parallel for schedule(static)
                    for (int j = 0; j < n; ++j)
                    {
                        // Accumulate scale for this column
                        float col_scale = 1.0f;

                        // Expand all blocks for this column
                        for (int kb = 0; kb < k_blocks; ++kb)
                        {
                            const SimpleQ8Block &block = B_blocks[j * k_blocks + kb];

                            // Convert FP16 scale to FP32
                            uint16_t h = block.d;
                            uint32_t sign = (h & 0x8000) << 16;
                            uint32_t exp = (h & 0x7C00);
                            uint32_t mant = (h & 0x03FF);

                            uint32_t f;
                            if (exp == 0)
                            {
                                if (mant == 0)
                                {
                                    f = sign;
                                }
                                else
                                {
                                    exp = 0x1C000;
                                    while ((mant & 0x0400) == 0)
                                    {
                                        exp -= 0x0400;
                                        mant <<= 1;
                                    }
                                    mant &= 0x03FF;
                                    f = sign | exp | (mant << 13);
                                }
                            }
                            else if (exp == 0x7C00)
                            {
                                f = sign | 0x7F800000 | (mant << 13);
                            }
                            else
                            {
                                f = sign | ((exp + 0x1C000) << 13) | (mant << 13);
                            }

                            float scale;
                            std::memcpy(&scale, &f, sizeof(float));

                            // Multiply scales (we'll apply once at the end)
                            col_scale *= scale;

                            // Copy INT8 values
                            const int k_start = kb * 32;
                            const int k_count = std::min(32, k - k_start);
                            for (int ki = 0; ki < k_count; ++ki)
                            {
                                packed->data[ki + k_start + j * k] = block.qs[ki];
                            }
                        }

                        packed->col_scales[j] = col_scale;
                    }

                    return packed;
                }

                /**
                 * @brief Optimized INT8 GEMM: C = A * B_packed
                 *
                 * Uses AVX512 VNNI on pre-packed data.
                 * Much faster than decoding blocks repeatedly.
                 */
                static bool multiply_packed(
                    const SimpleQ8Block *A_blocks, // [m, k_blocks] row-major
                    const PackedInt8Matrix *B_packed,
                    float *C, // [m, n] row-major
                    int m, int n, int k)
                {
                    const int k_blocks = (k + 31) / 32;

                    // Zero output
                    std::memset(C, 0, m * n * sizeof(float));

// Parallel over M rows
#pragma omp parallel for schedule(static)
                    for (int i = 0; i < m; ++i)
                    {
                        // Process each output column
                        for (int j = 0; j < n; ++j)
                        {
                            float sum = 0.0f;

                            // Accumulate over K in blocks of 32
                            for (int kb = 0; kb < k_blocks; ++kb)
                            {
                                const SimpleQ8Block &a_block = A_blocks[i * k_blocks + kb];

                                // Get A block scale
                                uint16_t h = a_block.d;
                                uint32_t sign = (h & 0x8000) << 16;
                                uint32_t exp = (h & 0x7C00);
                                uint32_t mant = (h & 0x03FF);

                                uint32_t f;
                                if (exp == 0)
                                {
                                    if (mant == 0)
                                    {
                                        f = sign;
                                    }
                                    else
                                    {
                                        exp = 0x1C000;
                                        while ((mant & 0x0400) == 0)
                                        {
                                            exp -= 0x0400;
                                            mant <<= 1;
                                        }
                                        mant &= 0x03FF;
                                        f = sign | exp | (mant << 13);
                                    }
                                }
                                else if (exp == 0x7C00)
                                {
                                    f = sign | 0x7F800000 | (mant << 13);
                                }
                                else
                                {
                                    f = sign | ((exp + 0x1C000) << 13) | (mant << 13);
                                }

                                float a_scale;
                                std::memcpy(&a_scale, &f, sizeof(float));

                                // Get B column data (already unpacked!)
                                const int k_start = kb * 32;
                                const int8_t *b_col = &B_packed->data[k_start + j * k];

                                // Compute INT8 dot product with VNNI
                                __m512i acc = _mm512_setzero_si512();
                                __m512i a_vec = _mm512_loadu_si512((__m512i *)a_block.qs);
                                __m512i b_vec = _mm512_loadu_si512((__m512i *)b_col);

                                // Convert signed INT8 to unsigned for VNNI
                                __m512i offset = _mm512_set1_epi8(-128);
                                __m512i a_unsigned = _mm512_sub_epi8(a_vec, offset);
                                __m512i b_unsigned = _mm512_sub_epi8(b_vec, offset);

                                // VNNI: multiply-add
                                acc = _mm512_dpbusd_epi32(acc, a_unsigned, b_unsigned);

                                // Horizontal sum
                                int32_t int_sum = _mm512_reduce_add_epi32(acc);

                                // Correct for bias (since we converted to unsigned)
                                int_sum -= 128 * 32; // Bias correction

                                // Scale and accumulate
                                sum += static_cast<float>(int_sum) * a_scale * B_packed->col_scales[j];
                            }

                            C[i * n + j] = sum;
                        }
                    }

                    return true;
                }
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
