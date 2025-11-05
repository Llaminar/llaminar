/**
 * @file INT8GemmKernel.cpp
 * @brief CPU INT8 GEMM kernel implementation with OneDNN/AVX512-VNNI
 *
 * Optimized paths:
 * 1. OneDNN (preferred): Highly optimized INT8 matmul with runtime dispatch
 * 2. AVX512-VNNI: VPDPBUSD instruction for 4× throughput vs FP32
 * 3. Scalar fallback: Portable INT8 multiply-accumulate
 *
 * @author David Sanftenberg
 */

#include "INT8GemmKernel.h"
#include "../../utils/CPUFeatures.h"
#include "../../utils/Logger.h"

#ifdef HAVE_ONEDNN
#include "oneapi/dnnl/dnnl.hpp"
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    // ========================================================================
    // Quantization: FP32 → INT8 with per-row scales (OneDNN compatible)
    // ========================================================================

    /**
     * @brief Quantize activation matrix with per-row scales
     *
     * OneDNN s8s8s32 matmul supports per-row scales for source (A) and per-column
     * scales for weights (B). This is more standard than per-block quantization.
     *
     * For matrix A [m, k]: Each row gets independent scale
     * Scale mask = 1<<0 (dimension 0, i.e., rows)
     */
    void INT8GemmKernel::quantize_activations_per_row(
        const float *A_fp32,
        int m, int k,
        int8_t *A_int8,
        float *row_scales) const
    {
        for (int i = 0; i < m; ++i)
        {
            // Find max absolute value in this row
            float max_abs = 0.0f;
            for (int j = 0; j < k; ++j)
            {
                max_abs = std::max(max_abs, std::fabs(A_fp32[i * k + j]));
            }

            // Compute scale for this row
            float scale = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;
            row_scales[i] = scale;

            // Quantize row
            const float inv_scale = 1.0f / scale;
            for (int j = 0; j < k; ++j)
            {
                float val = A_fp32[i * k + j] * inv_scale;
                val = std::max(-127.0f, std::min(127.0f, val));
                A_int8[i * k + j] = static_cast<int8_t>(std::round(val));
            }
        }
    }

    // ========================================================================
    // INT8 GEMM: C = A_int8 × B_int8 (with per-row/per-column scales)
    // ========================================================================

#ifdef HAVE_ONEDNN
    /**
     * @brief OneDNN-accelerated INT8 GEMM with per-channel quantization (FP32 output)
     *
     * Legacy wrapper for backward compatibility. Calls flexible implementation.
     */
    void INT8GemmKernel::gemm_int8_perchannel(
        const int8_t *A_int8,
        const float *A_row_scales,
        const int8_t *B_int8,
        const float *B_col_scales,
        float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta) const
    {
        // Call flexible implementation with C_int32=nullptr for FP32 output
        gemm_int8_perchannel_flexible(
            A_int8, A_row_scales,
            B_int8, B_col_scales,
            C, nullptr, // C_fp32=C, C_int32=nullptr
            m, n, k,
            transpose_B,
            alpha, beta);
    }

    /**
     * @brief INT8×INT8 GEMM with flexible INT32 or FP32 output
     *
     * Unified implementation supporting both output formats:
     * - INT32: Raw accumulator for full INT8 pipeline
     * - FP32: Dequantized output for current production pipeline
     */
    void INT8GemmKernel::gemm_int8_perchannel_flexible(
        const int8_t *A_int8,
        const float *A_row_scales,
        const int8_t *B_int8,
        const float *B_col_scales,
        float *C_fp32,
        int32_t *C_int32,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta) const
    {
        using namespace dnnl;

        try
        {
            // Create OneDNN engine and stream (CPU)
            engine eng(engine::kind::cpu, 0);
            stream s(eng);

            // Memory dimensions
            memory::dims a_dims = {m, k};
            memory::dims b_dims = transpose_B ? memory::dims{n, k} : memory::dims{k, n};
            memory::dims c_dims = {m, n};

            // INT8 input descriptors
            auto a_md = memory::desc(a_dims, memory::data_type::s8, memory::format_tag::ab);

            // Handle transpose_B: reorder data to match ab layout
            std::vector<int8_t> B_reordered;
            memory b_mem;

            if (transpose_B)
            {
                // B is [n,k] but stored row-major (ba layout)
                // OneDNN matmul expects B as [k,n] in ab layout
                // We need to transpose: B_reordered[k,n] = B[n,k]^T
                B_reordered.resize(k * n);
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < k; ++j)
                    {
                        B_reordered[j * n + i] = B_int8[i * k + j];
                    }
                }
                auto b_md_transposed = memory::desc({k, n}, memory::data_type::s8, memory::format_tag::ab);
                b_mem = memory(b_md_transposed, eng, B_reordered.data());
            }
            else
            {
                auto b_md_normal = memory::desc(b_dims, memory::data_type::s8, memory::format_tag::ab);
                b_mem = memory(b_md_normal, eng, const_cast<int8_t *>(B_int8));
            }

            // INT32 output (OneDNN matmul accumulates to INT32)
            auto c_i32_md = memory::desc(c_dims, memory::data_type::s32, memory::format_tag::ab);
            std::vector<int32_t> C_i32_temp;
            int32_t *C_i32_ptr = C_int32;

            // If caller wants INT32 output directly, use their buffer
            // Otherwise allocate temporary for dequantization
            if (C_int32 == nullptr)
            {
                C_i32_temp.resize(m * n);
                C_i32_ptr = C_i32_temp.data();
            }

            auto c_i32_mem = memory(c_i32_md, eng, C_i32_ptr);

            // Create matmul primitive
            auto a_mem = memory(a_md, eng, const_cast<int8_t *>(A_int8));
            auto b_md_for_matmul = transpose_B
                                       ? memory::desc({k, n}, memory::data_type::s8, memory::format_tag::ab)
                                       : memory::desc(b_dims, memory::data_type::s8, memory::format_tag::ab);
            auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md_for_matmul, c_i32_md);
            auto matmul_prim = matmul(matmul_pd);

            // Execute INT32 matmul (AVX512-VNNI optimized)
            matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                                    {DNNL_ARG_WEIGHTS, b_mem},
                                    {DNNL_ARG_DST, c_i32_mem}});
            s.wait();

            // If caller wants INT32 output, we're done!
            if (C_int32 != nullptr)
            {
                LOG_DEBUG("[INT8GemmKernel] OneDNN INT8→INT32 GEMM succeeded: "
                          << m << "×" << k << " @ " << k << "×" << n << " → " << m << "×" << n
                          << " (transpose_B=" << transpose_B << ", output=INT32)");
                return;
            }

// Otherwise, dequantize to FP32
// Optimized dequantization: C[i,j] = beta*C[i,j] + alpha * C_i32[i,j] * A_scale[i] * B_scale[j]
#pragma omp parallel for if (m * n > 4096)
            for (int i = 0; i < m; ++i)
            {
                const float scale_A_alpha = A_row_scales[i] * alpha;
                const int32_t *C_i32_row = &C_i32_ptr[i * n];
                float *C_row = &C_fp32[i * n];

#ifdef __AVX512F__
                // AVX512 path: process 16 floats at a time
                const __m512 scale_A_vec = _mm512_set1_ps(scale_A_alpha);
                const __m512 beta_vec = _mm512_set1_ps(beta);

                int j = 0;
                for (; j + 15 < n; j += 16)
                {
                    // Load INT32 values and convert to FP32
                    __m512i c_i32 = _mm512_loadu_si512((__m512i *)&C_i32_row[j]);
                    __m512 c_fp32 = _mm512_cvtepi32_ps(c_i32);

                    // Load B column scales
                    __m512 b_scales = _mm512_loadu_ps(&B_col_scales[j]);

                    // Compute: scale_A_alpha * c_fp32 * b_scales
                    __m512 scaled = _mm512_mul_ps(c_fp32, scale_A_vec);
                    scaled = _mm512_mul_ps(scaled, b_scales);

                    if (beta == 0.0f)
                    {
                        _mm512_storeu_ps(&C_row[j], scaled);
                    }
                    else
                    {
                        __m512 c_prev = _mm512_loadu_ps(&C_row[j]);
                        __m512 result = _mm512_fmadd_ps(beta_vec, c_prev, scaled);
                        _mm512_storeu_ps(&C_row[j], result);
                    }
                }

                // Handle remainder
                for (; j < n; ++j)
                {
                    if (beta == 0.0f)
                    {
                        C_row[j] = static_cast<float>(C_i32_row[j]) * scale_A_alpha * B_col_scales[j];
                    }
                    else
                    {
                        C_row[j] = beta * C_row[j] + static_cast<float>(C_i32_row[j]) * scale_A_alpha * B_col_scales[j];
                    }
                }
#else
                // Scalar fallback
                if (beta == 0.0f)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        C_row[j] = static_cast<float>(C_i32_row[j]) * scale_A_alpha * B_col_scales[j];
                    }
                }
                else
                {
                    for (int j = 0; j < n; ++j)
                    {
                        C_row[j] = beta * C_row[j] + static_cast<float>(C_i32_row[j]) * scale_A_alpha * B_col_scales[j];
                    }
                }
#endif
            }

            LOG_DEBUG("[INT8GemmKernel] OneDNN INT8→FP32 GEMM succeeded: "
                      << m << "×" << k << " @ " << k << "×" << n << " → " << m << "×" << n
                      << " (transpose_B=" << transpose_B << ", output=FP32)");
        }
        catch (const dnnl::error &e)
        {
            LOG_ERROR("[INT8GemmKernel] OneDNN error (flexible): " << e.what() << " (status=" << e.status << ")");
            throw; // Re-throw to caller
        }
    }
#endif

    /**
     * @brief Scalar INT8 GEMM fallback with per-channel scales
     *
     * Computes: C = alpha * (A_int8 * A_row_scales) @ (B_int8 * B_col_scales) + beta * C
     * Uses INT32 accumulation for dot products, then dequantizes to FP32.
     */
    static void gemm_int8_scalar_perchannel(
        const int8_t *A_int8,
        const float *A_row_scales,
        const int8_t *B_int8,
        const float *B_col_scales,
        float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta)
    {
        // Apply beta scaling to existing C
        if (beta != 1.0f)
        {
            for (int i = 0; i < m * n; ++i)
            {
                C[i] *= beta;
            }
        }

        // Compute alpha * A @ B with per-channel scaling
        for (int i = 0; i < m; ++i)
        {
            const float scale_A_row = A_row_scales[i];

            for (int j = 0; j < n; ++j)
            {
                const float scale_B_col = B_col_scales[j];
                int32_t dot_product = 0;

                // INT32 accumulation
                for (int kk = 0; kk < k; ++kk)
                {
                    int8_t a_val = A_int8[i * k + kk];
                    int8_t b_val = transpose_B ? B_int8[j * k + kk] : B_int8[kk * n + j];
                    dot_product += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
                }

                // Dequantize and accumulate
                float result = static_cast<float>(dot_product) * scale_A_row * scale_B_col;
                C[i * n + j] += alpha * result;
            }
        }
    }

    // ========================================================================
    // Public API: multiply() with FP32 activations
    // ========================================================================

    bool INT8GemmKernel::multiply(
        const float *A, float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        if (!weight_tensor_)
        {
            LOG_ERROR("[INT8GemmKernel] Weight tensor is null");
            return false;
        }

        const int8_t *B_int8 = weight_tensor_->int8_data();
        auto shape = weight_tensor_->shape();

        // Validate dimensions
        if (shape.size() != 2)
        {
            LOG_ERROR("[INT8GemmKernel] Weight tensor must be 2D, got " << shape.size() << "D");
            return false;
        }

        int B_rows = shape[0];
        int B_cols = shape[1];

        if (transpose_B)
        {
            if (B_rows != n || B_cols != k)
            {
                LOG_ERROR("[INT8GemmKernel] Dimension mismatch: B [" << B_rows << ", " << B_cols
                                                                     << "], expected [" << n << ", " << k << "] (transposed)");
                return false;
            }
        }
        else
        {
            if (B_rows != k || B_cols != n)
            {
                LOG_ERROR("[INT8GemmKernel] Dimension mismatch: B [" << B_rows << ", " << B_cols
                                                                     << "], expected [" << k << ", " << n << "]");
                return false;
            }
        }

        // Quantize activations with per-row scales
        std::vector<int8_t> A_int8(m * k);
        std::vector<float> A_row_scales(m);
        quantize_activations_per_row(A, m, k, A_int8.data(), A_row_scales.data());

        // Get scales for B (weight tensor)
        // When transpose_B=true, we multiply by ROWS of B, so need per-row scales
        // When transpose_B=false, we multiply by COLUMNS of B, so need per-column scales
        const float *B_col_scales_ptr = nullptr;
        std::vector<float> B_col_scales_fallback;

        if (transpose_B)
        {
            // transpose_B=true: Use per-ROW scales from B tensor
            const auto &row_scales = weight_tensor_->get_row_scales();
            if (!row_scales.empty())
            {
                B_col_scales_ptr = row_scales.data();
                LOG_DEBUG("[INT8GemmKernel] transpose_B=true, using per-row scales ("
                          << row_scales.size() << " rows)");
            }
            else
            {
                LOG_WARN("[INT8GemmKernel] transpose_B=true but no row scales, using global scale");
                B_col_scales_fallback.resize(n, weight_tensor_->scale());
                B_col_scales_ptr = B_col_scales_fallback.data();
            }
        }
        else if (weight_tensor_->has_col_scales())
        {
            // transpose_B=false: Use per-column scales from tensor directly
            B_col_scales_ptr = weight_tensor_->col_scales();
            LOG_DEBUG("[INT8GemmKernel] transpose_B=false, using per-column scales from tensor ("
                      << weight_tensor_->num_col_scales() << " columns)");
        }
        else
        {
            // Fallback: use global scale for all output columns
            LOG_WARN("[INT8GemmKernel] Tensor has no per-channel scales, using global scale (less accurate)");
            B_col_scales_fallback.resize(n, weight_tensor_->scale());
            B_col_scales_ptr = B_col_scales_fallback.data();
        }

        try
        {
#ifdef HAVE_ONEDNN
            // Try OneDNN path first (best performance)
            gemm_int8_perchannel(
                A_int8.data(), A_row_scales.data(),
                B_int8, B_col_scales_ptr,
                C, m, n, k,
                transpose_B, alpha, beta);
            return true;
#else
            // No OneDNN, use scalar fallback
            gemm_int8_scalar_perchannel(
                A_int8.data(), A_row_scales.data(),
                B_int8, B_col_scales_ptr,
                C, m, n, k,
                transpose_B, alpha, beta);
            return true;
#endif
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[INT8GemmKernel] GEMM failed: " << e.what() << ", trying scalar fallback");
            // OneDNN failed, try scalar
            gemm_int8_scalar_perchannel(
                A_int8.data(), A_row_scales.data(),
                B_int8, B_col_scales_ptr,
                C, m, n, k,
                transpose_B, alpha, beta);
            return true;
        }
    }

    // ========================================================================
    // Public API: multiply_activations() (A and B both FP32)
    // ========================================================================

    bool INT8GemmKernel::multiply_activations(
        const float *A, const float *B, float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        // For activation×activation, we'd need to quantize both A and B
        // This is less common for INT8 inference (usually weight×activation)
        LOG_ERROR("[INT8GemmKernel] multiply_activations not yet implemented for INT8");
        return false;
    }

    bool INT8GemmKernel::multiply_activations_strided(
        const float *A, const float *B, float *C,
        int m, int n, int k,
        int lda, int ldb, int ldc,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        LOG_ERROR("[INT8GemmKernel] multiply_activations_strided not yet implemented for INT8");
        return false;
    }

    bool INT8GemmKernel::multiply_int32(
        const float *A, int32_t *C_int32,
        int m, int n, int k,
        bool transpose_B,
        float *A_row_scales_out)
    {
        if (!weight_tensor_)
        {
            LOG_ERROR("[INT8GemmKernel] No weight tensor set");
            return false;
        }

        const auto &shape = weight_tensor_->shape();
        if (shape.size() != 2)
        {
            LOG_ERROR("[INT8GemmKernel] Weight tensor must be 2D, got " << shape.size() << "D");
            return false;
        }

        // Validate dimensions
        const int expected_k = transpose_B ? shape[1] : shape[0];
        const int expected_n = transpose_B ? shape[0] : shape[1];

        if (k != expected_k || n != expected_n)
        {
            LOG_ERROR("[INT8GemmKernel] Dimension mismatch: "
                      << "A=[" << m << "," << k << "], B=[" << expected_k << "," << expected_n << "], "
                      << "transpose_B=" << transpose_B);
            return false;
        }

        // Get INT8 weight data and column scales
        const int8_t *B_int8 = weight_tensor_->int8_data();
        const float *B_col_scales_ptr = weight_tensor_->col_scales();

        if (!B_col_scales_ptr)
        {
            LOG_ERROR("[INT8GemmKernel] Weight tensor missing per-column scales");
            return false;
        }

        try
        {
            // Quantize activations per-row
            std::vector<int8_t> A_int8(m * k);
            std::vector<float> A_row_scales(m);
            quantize_activations_per_row(A, m, k, A_int8.data(), A_row_scales.data());

            // Optional: return A row scales to caller (needed for later dequantization)
            if (A_row_scales_out)
            {
                std::copy(A_row_scales.begin(), A_row_scales.end(), A_row_scales_out);
            }

#ifdef HAVE_ONEDNN
            // Call flexible implementation with C_fp32=nullptr for INT32 output
            gemm_int8_perchannel_flexible(
                A_int8.data(), A_row_scales.data(),
                B_int8, B_col_scales_ptr,
                nullptr, C_int32, // C_fp32=nullptr, C_int32=C_int32
                m, n, k,
                transpose_B,
                1.0f, 0.0f); // alpha, beta not used for INT32 output
            return true;
#else
            LOG_ERROR("[INT8GemmKernel] multiply_int32 requires OneDNN (not available)");
            return false;
#endif
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[INT8GemmKernel] multiply_int32 failed: " << e.what());
            return false;
        }
    }

} // namespace llaminar2
