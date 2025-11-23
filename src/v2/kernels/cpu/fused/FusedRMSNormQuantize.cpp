/**
 * @file FusedRMSNormQuantize.cpp
 * @brief Implementation of fused RMSNorm + INT8 quantization kernel
 * @author David Sanftenberg
 * @date 2025-11-22
 */

#include "FusedRMSNormQuantize.h"
#include "../../../utils/Logger.h"
#include "../../../utils/CPUFeatures.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace llaminar2
{

    bool FusedRMSNormQuantize::execute(
        const float *input,
        const float *weight,
        int8_t *output,
        float *scales,
        int rows,
        int cols,
        float epsilon,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;    // Unused for now
        (void)device_idx; // Must be -1 (CPU)

        // Validate inputs
        if (!input || !weight || !output || !scales)
        {
            LOG_ERROR("[FusedRMSNormQuantize] Null pointer in inputs");
            return false;
        }

        if (rows <= 0 || cols <= 0)
        {
            LOG_ERROR("[FusedRMSNormQuantize] Invalid dimensions: rows="
                      << rows << ", cols=" << cols);
            return false;
        }

        // Process rows in parallel
#pragma omp parallel for
        for (int i = 0; i < rows; ++i)
        {
            const float *input_row = input + i * cols;
            int8_t *output_row = output + i * cols;
            float &scale = scales[i];

            process_row_fused(input_row, weight, output_row, scale, cols, epsilon);
        }

        return true;
    }

    void FusedRMSNormQuantize::process_row_fused(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
        // Dispatch to SIMD implementation at compile time (march=native)
#if defined(__AVX512F__)
        process_row_fused_avx512(input_row, weight, output_row, out_scale, cols, epsilon);
#elif defined(__AVX2__)
        process_row_fused_avx2(input_row, weight, output_row, out_scale, cols, epsilon);
#else
        process_row_fused_scalar(input_row, weight, output_row, out_scale, cols, epsilon);
#endif
    }

#if defined(__AVX512F__)

    void FusedRMSNormQuantize::process_row_fused_avx512(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
        // Step 1: Compute RMS (root mean square)
        __m512 sum_sq = _mm512_setzero_ps();
        const int vec_size = 16;
        int i = 0;

        for (; i + vec_size <= cols; i += vec_size)
        {
            __m512 x = _mm512_loadu_ps(input_row + i);
            sum_sq = _mm512_fmadd_ps(x, x, sum_sq); // sum_sq += x * x
        }

        // Horizontal sum of sum_sq
        float sum_sq_scalar = _mm512_reduce_add_ps(sum_sq);

        // Handle tail elements
        for (; i < cols; ++i)
        {
            float x = input_row[i];
            sum_sq_scalar += x * x;
        }

        // Compute RMS
        float mean_sq = sum_sq_scalar / static_cast<float>(cols);
        float rms = std::sqrt(mean_sq + epsilon);
        float rms_inv = 1.0f / rms;

        // Step 2: Normalize, apply weight, and find max absolute value (for quantization scale)
        __m512 rms_inv_vec = _mm512_set1_ps(rms_inv);
        __m512 max_abs = _mm512_setzero_ps();

        // Temporary buffer for normalized + scaled values (before quantization)
        alignas(64) float normalized[cols];

        i = 0;
        for (; i + vec_size <= cols; i += vec_size)
        {
            __m512 x = _mm512_loadu_ps(input_row + i);
            __m512 g = _mm512_loadu_ps(weight + i);

            // Normalize and scale: x_scaled = (x / rms) * weight
            __m512 x_norm = _mm512_mul_ps(x, rms_inv_vec);
            __m512 x_scaled = _mm512_mul_ps(x_norm, g);

            // Store for quantization pass
            _mm512_storeu_ps(normalized + i, x_scaled);

            // Track max absolute value
            __m512 abs_val = _mm512_abs_ps(x_scaled);
            max_abs = _mm512_max_ps(max_abs, abs_val);
        }

        // Horizontal max
        float max_abs_scalar = _mm512_reduce_max_ps(max_abs);

        // Handle tail
        for (; i < cols; ++i)
        {
            float x = input_row[i];
            float x_norm = x * rms_inv;
            float x_scaled = x_norm * weight[i];
            normalized[i] = x_scaled;
            max_abs_scalar = std::max(max_abs_scalar, std::fabs(x_scaled));
        }

        // Step 3: Compute quantization scale and quantize to INT8
        // Scale to map [-max_abs, max_abs] → [-127, 127]
        float quant_scale = (max_abs_scalar > 0.0f) ? (127.0f / max_abs_scalar) : 1.0f;
        out_scale = 1.0f / quant_scale; // Store dequantization scale

        __m512 quant_scale_vec = _mm512_set1_ps(quant_scale);

        i = 0;
        for (; i + vec_size <= cols; i += vec_size)
        {
            __m512 x_scaled = _mm512_loadu_ps(normalized + i);

            // Quantize: int8 = round(x_scaled * quant_scale)
            __m512 quantized_fp = _mm512_mul_ps(x_scaled, quant_scale_vec);
            __m512i quantized_int32 = _mm512_cvtps_epi32(quantized_fp); // Round + convert to int32

            // Clamp to INT8 range [-127, 127] (avoid -128 for symmetry)
            __m512i clamped = _mm512_max_epi32(quantized_int32, _mm512_set1_epi32(-127));
            clamped = _mm512_min_epi32(clamped, _mm512_set1_epi32(127));

            // Convert int32 → int8 (pack and extract)
            // AVX512 doesn't have direct int32→int8 pack, use scalar for now
            alignas(64) int32_t temp[16];
            _mm512_storeu_si512((__m512i *)temp, clamped);

            for (int j = 0; j < vec_size; ++j)
            {
                output_row[i + j] = static_cast<int8_t>(temp[j]);
            }
        }

        // Handle tail
        for (; i < cols; ++i)
        {
            float x_scaled = normalized[i];
            int32_t quantized = static_cast<int32_t>(std::round(x_scaled * quant_scale));
            quantized = std::max(-127, std::min(127, quantized));
            output_row[i] = static_cast<int8_t>(quantized);
        }
    }

#else
    // Stub for non-AVX512 builds
    void FusedRMSNormQuantize::process_row_fused_avx512(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
        process_row_fused_scalar(input_row, weight, output_row, out_scale, cols, epsilon);
    }
#endif // __AVX512F__

#if defined(__AVX2__)

    void FusedRMSNormQuantize::process_row_fused_avx2(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
        // Step 1: Compute RMS (root mean square)
        __m256 sum_sq = _mm256_setzero_ps();
        const int vec_size = 8;
        int i = 0;

        for (; i + vec_size <= cols; i += vec_size)
        {
            __m256 x = _mm256_loadu_ps(input_row + i);
            __m256 x_sq = _mm256_mul_ps(x, x);
            sum_sq = _mm256_add_ps(sum_sq, x_sq);
        }

        // Horizontal sum of sum_sq
        __m128 sum_high = _mm256_extractf128_ps(sum_sq, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum_sq);
        __m128 sum = _mm_add_ps(sum_low, sum_high);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        float sum_sq_scalar = _mm_cvtss_f32(sum);

        // Handle tail elements
        for (; i < cols; ++i)
        {
            float x = input_row[i];
            sum_sq_scalar += x * x;
        }

        // Compute RMS
        float mean_sq = sum_sq_scalar / static_cast<float>(cols);
        float rms = std::sqrt(mean_sq + epsilon);
        float rms_inv = 1.0f / rms;

        // Step 2: Normalize, apply weight, and find max absolute value
        __m256 rms_inv_vec = _mm256_set1_ps(rms_inv);
        __m256 max_abs = _mm256_setzero_ps();

        alignas(32) float normalized[cols];

        i = 0;
        for (; i + vec_size <= cols; i += vec_size)
        {
            __m256 x = _mm256_loadu_ps(input_row + i);
            __m256 g = _mm256_loadu_ps(weight + i);

            __m256 x_norm = _mm256_mul_ps(x, rms_inv_vec);
            __m256 x_scaled = _mm256_mul_ps(x_norm, g);

            _mm256_storeu_ps(normalized + i, x_scaled);

            // Max absolute value using AND with sign bit mask
            __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            __m256 abs_val = _mm256_and_ps(x_scaled, abs_mask);
            max_abs = _mm256_max_ps(max_abs, abs_val);
        }

        // Horizontal max
        __m128 max_high = _mm256_extractf128_ps(max_abs, 1);
        __m128 max_low = _mm256_castps256_ps128(max_abs);
        __m128 max_combined = _mm_max_ps(max_low, max_high);
        max_combined = _mm_max_ps(max_combined, _mm_movehl_ps(max_combined, max_combined));
        max_combined = _mm_max_ss(max_combined, _mm_shuffle_ps(max_combined, max_combined, 1));
        float max_abs_scalar = _mm_cvtss_f32(max_combined);

        // Handle tail
        for (; i < cols; ++i)
        {
            float x = input_row[i];
            float x_norm = x * rms_inv;
            float x_scaled = x_norm * weight[i];
            normalized[i] = x_scaled;
            max_abs_scalar = std::max(max_abs_scalar, std::fabs(x_scaled));
        }

        // Step 3: Quantize to INT8
        float quant_scale = (max_abs_scalar > 0.0f) ? (127.0f / max_abs_scalar) : 1.0f;
        out_scale = 1.0f / quant_scale;

        __m256 quant_scale_vec = _mm256_set1_ps(quant_scale);

        i = 0;
        for (; i + vec_size <= cols; i += vec_size)
        {
            __m256 x_scaled = _mm256_loadu_ps(normalized + i);
            __m256 quantized_fp = _mm256_mul_ps(x_scaled, quant_scale_vec);

            // Round to nearest integer
            __m256 rounded = _mm256_round_ps(quantized_fp, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256i quantized_int32 = _mm256_cvtps_epi32(rounded);

            // Extract and clamp to INT8 (scalar for simplicity)
            alignas(32) int32_t temp[8];
            _mm256_storeu_si256((__m256i *)temp, quantized_int32);

            for (int j = 0; j < vec_size; ++j)
            {
                int32_t val = std::max(-127, std::min(127, temp[j]));
                output_row[i + j] = static_cast<int8_t>(val);
            }
        }

        // Handle tail
        for (; i < cols; ++i)
        {
            float x_scaled = normalized[i];
            int32_t quantized = static_cast<int32_t>(std::round(x_scaled * quant_scale));
            quantized = std::max(-127, std::min(127, quantized));
            output_row[i] = static_cast<int8_t>(quantized);
        }
    }

#else
    // Stub for non-AVX2 builds
    void FusedRMSNormQuantize::process_row_fused_avx2(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
        process_row_fused_scalar(input_row, weight, output_row, out_scale, cols, epsilon);
    }
#endif // __AVX2__

    // Scalar fallback (portable - no SIMD)

    void FusedRMSNormQuantize::process_row_fused_scalar(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
        // Step 1: Compute RMS
        float sum_sq = 0.0f;
        for (int i = 0; i < cols; ++i)
        {
            float x = input_row[i];
            sum_sq += x * x;
        }

        float mean_sq = sum_sq / static_cast<float>(cols);
        float rms = std::sqrt(mean_sq + epsilon);
        float rms_inv = 1.0f / rms;

        // Step 2: Normalize, apply weight, and find max absolute value
        float max_abs = 0.0f;
        std::vector<float> normalized(cols);

        for (int i = 0; i < cols; ++i)
        {
            float x = input_row[i];
            float x_norm = x * rms_inv;
            float x_scaled = x_norm * weight[i];
            normalized[i] = x_scaled;
            max_abs = std::max(max_abs, std::fabs(x_scaled));
        }

        // Step 3: Quantize to INT8
        float quant_scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;
        out_scale = 1.0f / quant_scale; // Store dequantization scale

        for (int i = 0; i < cols; ++i)
        {
            float x_scaled = normalized[i];
            int32_t quantized = static_cast<int32_t>(std::round(x_scaled * quant_scale));
            quantized = std::max(-127, std::min(127, quantized));
            output_row[i] = static_cast<int8_t>(quantized);
        }
    }

} // namespace llaminar2
