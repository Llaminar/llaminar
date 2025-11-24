/**
 * @file OneDNNGemmKernel.h
 * @brief ITensorGemm implementation using OneDNN INT8 GEMM
 *
 * Provides a unified ITensorGemm interface using OneDNN's INT8 acceleration.
 * Includes OneDNN primitive wrappers (previously in OneDNNGemm.h) and the
 * ITensorGemm kernel implementation.
 *
 * @author David Sanftenberg
 * @date 2025-01-15
 */

#pragma once

#ifndef HAVE_ONEDNN
#error "OneDNN support is required to use gemm_v4"
#endif

#define DNNL_EXPERIMENTAL_UKERNEL
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>
#include <optional>
#include <map>
#include <mutex>
#include <typeinfo>
#include <immintrin.h>

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../utils/CPUFeatures.h"
#include "../CPUKernelBase.h"
#include "OneDNNGemmAdapter.h"
#include "../../../utils/Logger.h"
#include "../primitives/SoftmaxPrimitives_New.h"

namespace llaminar2
{
    namespace gemm_v4
    {
#undef LLAMINAR_LIBMVEC_WIDTH
        namespace detail
        {
#if defined(__AVX512F__)
#define LLAMINAR_LIBMVEC_WIDTH 16
            constexpr int kLibmvecWidth = 16;
            using LibmVecPack = float __attribute__((vector_size(64)));
            extern "C" LibmVecPack _ZGVeN16v_expf(LibmVecPack) __attribute__((const));
            inline LibmVecPack libmvec_expf(LibmVecPack value)
            {
                return _ZGVeN16v_expf(value);
            }
#elif defined(__AVX__)
#define LLAMINAR_LIBMVEC_WIDTH 8
            constexpr int kLibmvecWidth = 8;
            using LibmVecPack = float __attribute__((vector_size(32)));
            extern "C" LibmVecPack _ZGVcN8v_expf(LibmVecPack) __attribute__((const));
            inline LibmVecPack libmvec_expf(LibmVecPack value)
            {
                return _ZGVcN8v_expf(value);
            }
#elif defined(__SSE2__)
#define LLAMINAR_LIBMVEC_WIDTH 4
            constexpr int kLibmvecWidth = 4;
            using LibmVecPack = float __attribute__((vector_size(16)));
            extern "C" LibmVecPack _ZGVbN4v_expf(LibmVecPack) __attribute__((const));
            inline LibmVecPack libmvec_expf(LibmVecPack value)
            {
                return _ZGVbN4v_expf(value);
            }
#else
#define LLAMINAR_LIBMVEC_WIDTH 0
            constexpr int kLibmvecWidth = 0;
#endif

            inline float apply_libmvec_exp(float *data, int length, float row_max)
            {
                float sum = 0.0f;
#if LLAMINAR_LIBMVEC_WIDTH > 0
                alignas(64) float input[LLAMINAR_LIBMVEC_WIDTH];
                int idx = 0;
                while (idx < length)
                {
                    const int lanes = std::min(LLAMINAR_LIBMVEC_WIDTH, length - idx);
                    for (int lane = 0; lane < lanes; ++lane)
                    {
                        input[lane] = data[idx + lane] - row_max;
                    }
                    for (int lane = lanes; lane < LLAMINAR_LIBMVEC_WIDTH; ++lane)
                    {
                        input[lane] = -std::numeric_limits<float>::infinity();
                    }

                    LibmVecPack vec_in;
                    std::memcpy(&vec_in, input, sizeof(vec_in));
                    LibmVecPack vec_out = libmvec_expf(vec_in);
                    std::memcpy(input, &vec_out, sizeof(vec_out));

                    for (int lane = 0; lane < lanes; ++lane)
                    {
                        const float exp_val = input[lane];
                        data[idx + lane] = exp_val;
                        sum += exp_val;
                    }

                    idx += lanes;
                }
#else
                for (int i = 0; i < length; ++i)
                {
                    const float exp_val = std::expf(data[i] - row_max);
                    data[i] = exp_val;
                    sum += exp_val;
                }
#endif
                return sum;
            }

            // SIMD Accumulation Helpers
            inline void accumulate_simd_1(
                int N_chunk,
                const int32_t *C_temp,
                float d_A,
                const uint16_t *B_scales,
                float *C_row,
                float alpha, float beta,
                bool accumulate_beta)
            {
                int i = 0;
#if defined(__AVX512F__)
                __m512 v_dA = _mm512_set1_ps(d_A);
                __m512 v_alpha = _mm512_set1_ps(alpha);
                __m512 v_beta = _mm512_set1_ps(beta);

                for (; i <= N_chunk - 16; i += 16)
                {
                    __m512 v_C_temp = _mm512_cvtepi32_ps(_mm512_loadu_si512(C_temp + i));
                    __m512 v_dB = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(B_scales + i)));
                    __m512 v_term = _mm512_mul_ps(v_C_temp, _mm512_mul_ps(v_dA, v_dB));

                    __m512 v_C = _mm512_loadu_ps(C_row + i);
                    if (accumulate_beta)
                    {
                        v_C = _mm512_fmadd_ps(v_alpha, v_term, _mm512_mul_ps(v_beta, v_C));
                    }
                    else
                    {
                        v_C = _mm512_fmadd_ps(v_alpha, v_term, v_C);
                    }
                    _mm512_storeu_ps(C_row + i, v_C);
                }
#elif defined(__AVX2__) && defined(__F16C__)
                __m256 v_dA = _mm256_set1_ps(d_A);
                __m256 v_alpha = _mm256_set1_ps(alpha);
                __m256 v_beta = _mm256_set1_ps(beta);

                for (; i <= N_chunk - 8; i += 8)
                {
                    __m256 v_C_temp = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const *)(C_temp + i)));
                    __m256 v_dB = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(B_scales + i)));
                    __m256 v_term = _mm256_mul_ps(v_C_temp, _mm256_mul_ps(v_dA, v_dB));

                    __m256 v_C = _mm256_loadu_ps(C_row + i);
                    if (accumulate_beta)
                    {
                        v_C = _mm256_fmadd_ps(v_alpha, v_term, _mm256_mul_ps(v_beta, v_C));
                    }
                    else
                    {
                        v_C = _mm256_fmadd_ps(v_alpha, v_term, v_C);
                    }
                    _mm256_storeu_ps(C_row + i, v_C);
                }
#endif
                for (; i < N_chunk; ++i)
                {
                    float d_B = fp16_to_fp32(B_scales[i]);
                    float term = (float)C_temp[i] * d_A * d_B;
                    if (accumulate_beta)
                    {
                        C_row[i] = alpha * term + beta * C_row[i];
                    }
                    else
                    {
                        C_row[i] += alpha * term;
                    }
                }
            }

            inline void accumulate_simd_4(
                int N_chunk,
                const int32_t *C_temp0, const int32_t *C_temp1, const int32_t *C_temp2, const int32_t *C_temp3,
                float d_A0, float d_A1, float d_A2, float d_A3,
                const uint16_t *B_scales0, const uint16_t *B_scales1, const uint16_t *B_scales2, const uint16_t *B_scales3,
                float *C_row,
                float alpha, float beta,
                bool accumulate_beta)
            {
                int i = 0;
#if defined(__AVX512F__)
                __m512 v_dA0 = _mm512_set1_ps(d_A0);
                __m512 v_dA1 = _mm512_set1_ps(d_A1);
                __m512 v_dA2 = _mm512_set1_ps(d_A2);
                __m512 v_dA3 = _mm512_set1_ps(d_A3);
                __m512 v_alpha = _mm512_set1_ps(alpha);
                __m512 v_beta = _mm512_set1_ps(beta);

                for (; i <= N_chunk - 16; i += 16)
                {
                    // Term 0
                    __m512 v_C0 = _mm512_cvtepi32_ps(_mm512_loadu_si512(C_temp0 + i));
                    __m512 v_dB0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(B_scales0 + i)));
                    __m512 v_sum = _mm512_mul_ps(v_C0, _mm512_mul_ps(v_dA0, v_dB0));

                    // Term 1
                    __m512 v_C1 = _mm512_cvtepi32_ps(_mm512_loadu_si512(C_temp1 + i));
                    __m512 v_dB1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(B_scales1 + i)));
                    v_sum = _mm512_fmadd_ps(v_C1, _mm512_mul_ps(v_dA1, v_dB1), v_sum);

                    // Term 2
                    __m512 v_C2 = _mm512_cvtepi32_ps(_mm512_loadu_si512(C_temp2 + i));
                    __m512 v_dB2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(B_scales2 + i)));
                    v_sum = _mm512_fmadd_ps(v_C2, _mm512_mul_ps(v_dA2, v_dB2), v_sum);

                    // Term 3
                    __m512 v_C3 = _mm512_cvtepi32_ps(_mm512_loadu_si512(C_temp3 + i));
                    __m512 v_dB3 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)(B_scales3 + i)));
                    v_sum = _mm512_fmadd_ps(v_C3, _mm512_mul_ps(v_dA3, v_dB3), v_sum);

                    // Accumulate to C
                    __m512 v_C = _mm512_loadu_ps(C_row + i);
                    if (accumulate_beta)
                    {
                        v_C = _mm512_fmadd_ps(v_alpha, v_sum, _mm512_mul_ps(v_beta, v_C));
                    }
                    else
                    {
                        v_C = _mm512_fmadd_ps(v_alpha, v_sum, v_C);
                    }
                    _mm512_storeu_ps(C_row + i, v_C);
                }
#elif defined(__AVX2__) && defined(__F16C__)
                __m256 v_dA0 = _mm256_set1_ps(d_A0);
                __m256 v_dA1 = _mm256_set1_ps(d_A1);
                __m256 v_dA2 = _mm256_set1_ps(d_A2);
                __m256 v_dA3 = _mm256_set1_ps(d_A3);
                __m256 v_alpha = _mm256_set1_ps(alpha);
                __m256 v_beta = _mm256_set1_ps(beta);

                for (; i <= N_chunk - 8; i += 8)
                {
                    // Term 0
                    __m256 v_C0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const *)(C_temp0 + i)));
                    __m256 v_dB0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(B_scales0 + i)));
                    __m256 v_sum = _mm256_mul_ps(v_C0, _mm256_mul_ps(v_dA0, v_dB0));

                    // Term 1
                    __m256 v_C1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const *)(C_temp1 + i)));
                    __m256 v_dB1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(B_scales1 + i)));
                    v_sum = _mm256_fmadd_ps(v_C1, _mm256_mul_ps(v_dA1, v_dB1), v_sum);

                    // Term 2
                    __m256 v_C2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const *)(C_temp2 + i)));
                    __m256 v_dB2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(B_scales2 + i)));
                    v_sum = _mm256_fmadd_ps(v_C2, _mm256_mul_ps(v_dA2, v_dB2), v_sum);

                    // Term 3
                    __m256 v_C3 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const *)(C_temp3 + i)));
                    __m256 v_dB3 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(B_scales3 + i)));
                    v_sum = _mm256_fmadd_ps(v_C3, _mm256_mul_ps(v_dA3, v_dB3), v_sum);

                    // Accumulate to C
                    __m256 v_C = _mm256_loadu_ps(C_row + i);
                    if (accumulate_beta)
                    {
                        v_C = _mm256_fmadd_ps(v_alpha, v_sum, _mm256_mul_ps(v_beta, v_C));
                    }
                    else
                    {
                        v_C = _mm256_fmadd_ps(v_alpha, v_sum, v_C);
                    }
                    _mm256_storeu_ps(C_row + i, v_C);
                }
#endif
                for (; i < N_chunk; ++i)
                {
                    float term = (float)C_temp0[i] * d_A0 * fp16_to_fp32(B_scales0[i]) +
                                 (float)C_temp1[i] * d_A1 * fp16_to_fp32(B_scales1[i]) +
                                 (float)C_temp2[i] * d_A2 * fp16_to_fp32(B_scales2[i]) +
                                 (float)C_temp3[i] * d_A3 * fp16_to_fp32(B_scales3[i]);

                    if (accumulate_beta)
                    {
                        C_row[i] = alpha * term + beta * C_row[i];
                    }
                    else
                    {
                        C_row[i] += alpha * term;
                    }
                }
            }

        } // namespace detail

        // ========== OneDNN Primitive Wrappers (from OneDNNGemm.h) ==========

        /**
         * @brief Get singleton OneDNN CPU engine
         */
        inline dnnl::engine &onednn_engine()
        {
            static thread_local dnnl::engine engine_instance(dnnl::engine::kind::cpu, 0);
            return engine_instance;
        }

        /**
         * @brief Get singleton OneDNN execution stream
         */
        inline dnnl::stream &onednn_stream()
        {
            thread_local dnnl::stream stream_instance(onednn_engine());
            return stream_instance;
        }

        /**
         * @brief Execute INT8 matrix multiplication using OneDNN
         */
        inline bool run_onednn_int8_matmul(const int8_t *A,
                                           const int8_t *B,
                                           int32_t *C,
                                           int M,
                                           int N,
                                           int K)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::s8, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::s8, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::s32, tag::ab);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<int8_t *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<int8_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                                {{DNNL_ARG_SRC, src_mem},
                                                 {DNNL_ARG_WEIGHTS, weight_mem},
                                                 {DNNL_ARG_DST, dst_mem}});
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN matmul failed: status=" << e.status
                                                          << " message=" << e.what());
                return false;
            }

            return true;
        }

        inline bool run_onednn_fp32_matmul(const float *A,
                                           const float *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           bool transpose_B,
                                           float alpha = 1.0f,
                                           float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f32, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f32, transpose_B ? tag::ba : tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Apply alpha scaling to the result of A*B
                if (alpha != 1.0f)
                {
                    attr.set_scales_mask(DNNL_ARG_DST, 0); // 0 mask = single scale for whole tensor
                }

                // Apply beta * C addition
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }

                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<float *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                // Set alpha scale if needed
                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                if (alpha != 1.0f)
                {
                    dnnl::memory alpha_mem({{1}, dt::f32, tag::x}, onednn_engine(), &alpha);
                    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, alpha_mem});
                }

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN FP32 matmul failed: status=" << e.status
                                                               << " message=" << e.what());
                return false;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("OneDNN FP32 matmul failed with std::exception: " << e.what());
                return false;
            }
            catch (...)
            {
                LOG_ERROR("OneDNN FP32 matmul failed with unknown exception");
                return false;
            }

            return true;
        }

        /**
         * @brief Execute BF16 matrix multiplication using OneDNN (bf16bf16f32)
         *
         * Computes C = A @ B using OneDNN's native BF16 GEMM with FP32 accumulation.
         * Requires AVX512_BF16 CPU support for optimal performance.
         *
         * **Precision**: BF16 inputs, FP32 accumulator, FP32 output
         * **Format**: A and B are uint16_t* (BF16 bit pattern), C is float*
         *
         * @param A Input matrix A [M, K] (BF16, row-major)
         * @param B Input matrix B [K, N] (BF16, row-major)
         * @param C Output matrix C [M, N] (FP32, row-major)
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A and rows in B
         *
         * @return true on success, false on OneDNN error
         *
         * @note Falls back to FP32 GEMM on CPUs without AVX512_BF16
         */
        inline bool run_onednn_bf16_matmul(const uint16_t *A,
                                           const uint16_t *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           float alpha = 1.0f,
                                           float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::bf16, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::bf16, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                if (alpha != 1.0f)
                {
                    attr.set_scales_mask(DNNL_ARG_DST, 0);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<uint16_t *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                if (alpha != 1.0f)
                {
                    dnnl::memory alpha_mem({{1}, dt::f32, tag::x}, onednn_engine(), &alpha);
                    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, alpha_mem});
                }

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN BF16 matmul failed: status=" << e.status
                                                               << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute FP16 matrix multiplication using OneDNN (fp16fp16f32)
         *
         * Computes C = A @ B using OneDNN's native FP16 GEMM with FP32 accumulation.
         * Requires FP16 CPU support (AVX512_FP16 or fallback emulation).
         *
         * **Precision**: FP16 inputs, FP32 accumulator, FP32 output
         * **Format**: A and B are uint16_t* (FP16 bit pattern), C is float*
         *
         * @param A Input matrix A [M, K] (FP16, row-major)
         * @param B Input matrix B [K, N] (FP16, row-major)
         * @param C Output matrix C [M, N] (FP32, row-major)
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A and rows in B
         *
         * @return true on success, false on OneDNN error
         *
         * @note Falls back to FP32 GEMM on CPUs without FP16 support
         */
        inline bool run_onednn_fp16_matmul(const uint16_t *A,
                                           const uint16_t *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           float alpha = 1.0f,
                                           float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f16, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f16, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                if (alpha != 1.0f)
                {
                    attr.set_scales_mask(DNNL_ARG_DST, 0);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<uint16_t *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                if (alpha != 1.0f)
                {
                    dnnl::memory alpha_mem({{1}, dt::f32, tag::x}, onednn_engine(), &alpha);
                    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, alpha_mem});
                }

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_WARN("OneDNN FP16 matmul failed (likely unsupported hardware): status=" << e.status
                                                                                            << " message=" << e.what());
                return false;
            }

            return true;
        }

        inline bool run_onednn_fp32_matmul_softmax(const float *A,
                                                   const float *B,
                                                   float *C,
                                                   int M,
                                                   int N,
                                                   int K,
                                                   int softmax_axis)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            std::cerr << "run_onednn_fp32_matmul_softmax invoked with M=" << M
                      << " N=" << N << " K=" << K << " axis=" << softmax_axis << std::endl;

            const int ndims = 2;
            int axis = softmax_axis;
            if (axis < 0)
            {
                axis += ndims;
            }
            if (axis != ndims - 1)
            {
                // Current OneDNN softmax post-op implementation only supports the last dimension.
                std::cerr << "OneDNN matmul+softmax rejected axis " << softmax_axis
                          << " (normalized " << axis << ") for ndims=" << ndims << std::endl;
                return false;
            }

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f32, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f32, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;
                ops.append_softmax(axis, false);
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<float *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                                {{DNNL_ARG_SRC, src_mem},
                                                 {DNNL_ARG_WEIGHTS, weight_mem},
                                                 {DNNL_ARG_DST, dst_mem}});
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                std::cerr << "OneDNN matmul+softmax failed: status=" << e.status
                          << " message=" << e.what() << std::endl;
                return false;
            }

            std::cerr << "OneDNN matmul+softmax execution succeeded" << std::endl;
            return true;
        }

        inline bool apply_softmax_inplace(float *data,
                                          int rows,
                                          int cols,
                                          int axis)
        {
            if (!data || rows <= 0 || cols <= 0)
            {
                return false;
            }

            int normalized_axis = axis;
            if (normalized_axis < 0)
            {
                normalized_axis += 2; // Only 2D tensors supported here
            }

            auto exp_range = [](float value) -> float
            {
                return std::exp(value);
            };

            if (normalized_axis == 1)
            {
                for (int r = 0; r < rows; ++r)
                {
                    const size_t base = static_cast<size_t>(r) * static_cast<size_t>(cols);
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int c = 0; c < cols; ++c)
                    {
                        max_val = std::max(max_val, data[base + static_cast<size_t>(c)]);
                    }

                    if (max_val == -std::numeric_limits<float>::infinity())
                    {
                        std::fill(data + base, data + base + cols, 0.0f);
                        continue;
                    }

                    float sum = detail::apply_libmvec_exp(data + base, cols, max_val);

                    const float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                    for (int c = 0; c < cols; ++c)
                    {
                        data[base + static_cast<size_t>(c)] *= inv_sum;
                    }
                }
                return true;
            }

            if (normalized_axis == 0)
            {
                std::vector<float> column(static_cast<size_t>(rows));
                for (int c = 0; c < cols; ++c)
                {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int r = 0; r < rows; ++r)
                    {
                        const float val = data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
                        column[static_cast<size_t>(r)] = val;
                        max_val = std::max(max_val, val);
                    }

                    if (max_val == -std::numeric_limits<float>::infinity())
                    {
                        for (int r = 0; r < rows; ++r)
                        {
                            data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] = 0.0f;
                        }
                        continue;
                    }

                    float sum = detail::apply_libmvec_exp(column.data(), rows, max_val);

                    const float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                    for (int r = 0; r < rows; ++r)
                    {
                        data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] =
                            column[static_cast<size_t>(r)] * inv_sum;
                    }
                }
                return true;
            }

            return false;
        }

        inline const float *prepare_rhs_for_matmul(const float *B,
                                                   int n,
                                                   int k,
                                                   bool transpose_B)
        {
            if (!transpose_B)
            {
                return B;
            }

            thread_local std::vector<float> B_transposed;
            B_transposed.resize(static_cast<size_t>(k) * static_cast<size_t>(n));

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < k; ++j)
                {
                    B_transposed[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] =
                        B[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)];
                }
            }

            return B_transposed.data();
        }

        // ========== Lightweight Activation View ==========

        /**
         * @brief Lightweight view wrapper for raw pointer activations
         *
         * Implements IActivationTensor interface without owning memory.
         * Used to avoid temporary allocations in ITensorGemm::multiply().
         */
        class ActivationView : public IActivationTensor
        {
        private:
            const float *data_;
            float *output_data_; // For from_int32_with_scales()
            std::vector<size_t> shape_;

        public:
            ActivationView(const float *data, size_t rows, size_t cols)
                : data_(data), output_data_(nullptr), shape_{rows, cols} {}

            ActivationView(float *data, size_t rows, size_t cols)
                : data_(data), output_data_(data), shape_{rows, cols} {}

            // IActivationTensor interface - only implement what we need
            ActivationPack to_int8_activation_pack(int rows, int cols) const override
            {
                ActivationPack pack;
                pack.rows = rows;
                pack.cols = cols;
                pack.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
                pack.row_scales.resize(static_cast<size_t>(rows));

                for (int r = 0; r < rows; ++r)
                {
                    const float *row_ptr = data_ + static_cast<size_t>(r) * static_cast<size_t>(cols);
                    float max_abs = 0.0f;
                    for (int c = 0; c < cols; ++c)
                    {
                        float val = std::abs(row_ptr[c]);
                        if (val > max_abs)
                            max_abs = val;
                    }

                    float scale = max_abs / 127.0f;
                    pack.row_scales[r] = scale;

                    if (scale > 0.0f)
                    {
                        float inv_scale = 127.0f / max_abs;
                        for (int c = 0; c < cols; ++c)
                        {
                            float val = row_ptr[c] * inv_scale;
                            pack.data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] =
                                static_cast<int8_t>(std::round(std::min(127.0f, std::max(-127.0f, val))));
                        }
                    }
                    else
                    {
                        std::fill_n(pack.data.data() + static_cast<size_t>(r) * static_cast<size_t>(cols),
                                    cols, int8_t(0));
                    }
                }

                return pack;
            }

            bool from_int32_with_scales(const int32_t *accum, int rows, int cols,
                                        const float *row_scales, const float *col_scales,
                                        const float *bias = nullptr) override
            {
                if (!output_data_)
                {
                    return false;
                }

                for (int r = 0; r < rows; ++r)
                {
                    const float rscale = row_scales ? row_scales[r] : 1.0f;
                    for (int c = 0; c < cols; ++c)
                    {
                        const float cscale = col_scales ? col_scales[c] : 1.0f;
                        float val = static_cast<float>(accum[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)]) *
                                    rscale * cscale;
                        if (bias)
                        {
                            val += bias[c];
                        }
                        output_data_[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] = val;
                    }
                }

                return true;
            }

            // Unused methods (throw if called)
            std::unique_ptr<ITensorRoPE> createRoPE() override
            {
                throw std::runtime_error("ActivationView::createRoPE() not supported");
            }
            std::unique_ptr<ITensorSwiGLU> createSwiGLU() override
            {
                throw std::runtime_error("ActivationView::createSwiGLU() not supported");
            }
            std::unique_ptr<ITensorSoftmax> createSoftmax() override
            {
                throw std::runtime_error("ActivationView::createSoftmax() not supported");
            }
            std::unique_ptr<ITensorRMSNorm> createRMSNorm() override
            {
                throw std::runtime_error("ActivationView::createRMSNorm() not supported");
            }
            std::unique_ptr<ITensorAttention> createAttention() override
            {
                throw std::runtime_error("ActivationView::createAttention() not supported");
            }
            bool applyRoPE(float *, const int *, int, int, int, int, float, bool, const MPIContext *, int) override
            {
                throw std::runtime_error("ActivationView::applyRoPE() not supported");
            }
        };

        struct Int8MatmulSoftmaxParams
        {
            int M;
            int N;
            int K;
            const float *row_scales = nullptr;
            const float *col_scales = nullptr;
            const float *bias = nullptr;
            bool causal = false;
            float softmax_scale = 1.0f;
            bool parallel_softmax = true;
        };

        inline bool run_onednn_int8_matmul_with_softmax(const int8_t *A,
                                                        const int8_t *B,
                                                        float *scores,
                                                        const Int8MatmulSoftmaxParams &params)
        {
            if (!A || !B || !scores)
            {
                return false;
            }

            const int M = params.M;
            const int N = params.N;
            const int K = params.K;
            if (M <= 0 || N <= 0 || K <= 0)
            {
                return false;
            }

            static thread_local std::vector<int32_t> accum_buffer;
            const size_t accum_elems = static_cast<size_t>(M) * static_cast<size_t>(N);
            if (accum_buffer.size() < accum_elems)
            {
                accum_buffer.resize(accum_elems);
            }

            if (!run_onednn_int8_matmul(A, B, accum_buffer.data(), M, N, K))
            {
                return false;
            }

            const int32_t *accum_ptr = accum_buffer.data();
            const float *row_scales = params.row_scales;
            const float *col_scales = params.col_scales;
            const float *bias = params.bias;
            const bool causal = params.causal;
            const float softmax_scale = params.softmax_scale;
            const bool parallel = params.parallel_softmax;

            auto process_row = [&](int row, std::vector<float> &scratch)
            {
                const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(N);
                const float row_scale = row_scales ? row_scales[row] : 1.0f;
                const int valid_cols = causal ? std::min(N, row + 1) : N;

                float row_max = -std::numeric_limits<float>::infinity();
                for (int col = 0; col < valid_cols; ++col)
                {
                    float value = static_cast<float>(accum_ptr[row_offset + static_cast<size_t>(col)]) * row_scale;
                    if (col_scales)
                    {
                        value *= col_scales[col];
                    }
                    if (bias)
                    {
                        value += bias[col];
                    }
                    value *= softmax_scale;

                    scratch[col] = value;
                    row_max = std::max(row_max, value);
                }
                for (int col = valid_cols; col < N; ++col)
                {
                    scratch[col] = -std::numeric_limits<float>::infinity();
                }

                if (!std::isfinite(row_max))
                {
                    std::fill(scores + row_offset, scores + row_offset + static_cast<size_t>(N), 0.0f);
                    return;
                }

                float sum = detail::apply_libmvec_exp(scratch.data(), valid_cols, row_max);

                const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
                float *out_row = scores + row_offset;
                for (int col = 0; col < valid_cols; ++col)
                {
                    out_row[col] = scratch[col] * inv_sum;
                }
                for (int col = valid_cols; col < N; ++col)
                {
                    out_row[col] = 0.0f;
                }
            };

            auto run_serial = [&]()
            {
                std::vector<float> scratch(static_cast<size_t>(N));
                for (int row = 0; row < M; ++row)
                {
                    process_row(row, scratch);
                }
            };

#if defined(_OPENMP)
            if (parallel)
            {
#pragma omp parallel
                {
                    std::vector<float> scratch(static_cast<size_t>(N));
#pragma omp for schedule(static)
                    for (int row = 0; row < M; ++row)
                    {
                        process_row(row, scratch);
                    }
                }
            }
            else
#endif
            {
                run_serial();
            }

            return true;
        }

        /**
         * @brief Execute Mixed Precision BF16 matrix multiplication (f32bf16f32)
         *
         * Computes C = A @ B where A is FP32 and B is BF16.
         * Useful for attention context projection (scores @ V).
         *
         * @param A Input matrix A [M, K] (FP32, row-major)
         * @param B Input matrix B [K, N] (BF16, row-major)
         * @param C Output matrix C [M, N] (FP32, row-major)
         */
        inline bool run_onednn_mixed_bf16_matmul(const float *A,
                                                 const uint16_t *B,
                                                 float *C,
                                                 int M,
                                                 int N,
                                                 int K)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f32, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::bf16, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                                {{DNNL_ARG_SRC, src_mem},
                                                 {DNNL_ARG_WEIGHTS, weight_mem},
                                                 {DNNL_ARG_DST, dst_mem}});
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN Mixed BF16 matmul failed: status=" << e.status
                                                                     << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute Mixed Precision FP16 matrix multiplication (f32fp16f32)
         *
         * Computes C = A @ B where A is FP32 and B is FP16.
         * Useful for attention context projection (scores @ V).
         *
         * @param A Input matrix A [M, K] (FP32, row-major)
         * @param B Input matrix B [K, N] (FP16, row-major)
         * @param C Output matrix C [M, N] (FP32, row-major)
         */
        inline bool run_onednn_mixed_fp16_matmul(const float *A,
                                                 const uint16_t *B,
                                                 float *C,
                                                 int M,
                                                 int N,
                                                 int K)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f32, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f16, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                                {{DNNL_ARG_SRC, src_mem},
                                                 {DNNL_ARG_WEIGHTS, weight_mem},
                                                 {DNNL_ARG_DST, dst_mem}});
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_WARN("OneDNN Mixed FP16 matmul failed (likely unsupported hardware): status=" << e.status
                                                                                                  << " message=" << e.what());
                return false;
            }

            return true;
        }

        class OneDNNGemmKernel : public ITensorGemm, public CPUKernelBase
        {
        public:
            /**
             * @brief Construct kernel bound to a weight tensor
             * @param weight_tensor Pointer to weight tensor (B matrix)
             */
            explicit OneDNNGemmKernel(const TensorBase *weight_tensor = nullptr)
                : weight_tensor_(weight_tensor) {}
            ~OneDNNGemmKernel() override = default;

            /**
             * @brief Check device support (CPU-only for OneDNN)
             */
            bool supports_device(int device_idx) const override
            {
                return device_idx == -1; // CPU only
            }

        private:
            struct ScratchBuffers
            {
                std::vector<float> fp32_buffer_a;
                std::vector<float> fp32_buffer_b;
                std::vector<float> fp32_buffer_c;
                std::vector<uint16_t> fp16_buffer;
                std::vector<uint8_t> byte_buffer_a;
                std::vector<uint8_t> byte_buffer_b;
                std::vector<int32_t> int32_buffer;
            };

            static ScratchBuffers &get_scratch_buffers()
            {
                thread_local ScratchBuffers buffers;
                return buffers;
            }

            static void ensure_fp32_capacity(std::vector<float> &buf, size_t required_size)
            {
                if (buf.size() < required_size)
                {
                    buf.resize(required_size);
                }
            }

            static void ensure_int32_capacity(std::vector<int32_t> &buf, size_t required_size)
            {
                if (buf.size() < required_size)
                {
                    buf.resize(required_size);
                }
            }

            static void ensure_fp16_capacity(std::vector<uint16_t> &buf, size_t required_size)
            {
                if (buf.size() < required_size)
                {
                    buf.resize(required_size);
                }
            }

            static void ensure_byte_capacity(std::vector<uint8_t> &buf, size_t required_size)
            {
                if (buf.size() < required_size)
                {
                    buf.resize(required_size);
                }
            }

            struct BlockWeightPack
            {
                std::vector<int8_t> unpacked_s8; // [K_blocks, N, 32] -> flattened to [K_blocks * 32 * N] or [K, N]?
                // Actually, we want layout compatible with BRGEMM.
                // BRGEMM expects B in [K, N] or [N, K] depending on format.
                // We want [K_blk, N, 32] where 32 is K dimension of block?
                // No, BRGEMM B is usually [K, N] row major or [N, K] col major.
                // OneDNN s8 expects [K, N] (row major) or [N, K] (col major).
                // We used `B_ptr_k = weights.unpacked_s8.data() + (k_blk * K_blk) * N + n`
                // This implies row-major [K, N].

                std::vector<uint16_t> block_scales; // [K_blocks, N] (fp16)
                int N;
                int K;
                int K_blocks;
            };

            static BlockWeightPack pack_weights_generic_blockwise(const TensorBase &tensor, int K, int N)
            {
                BlockWeightPack pack;
                pack.N = N;
                pack.K = K;
                pack.K_blocks = K / 32;

                // Allocate buffers
                pack.unpacked_s8.resize(K * N);
                pack.block_scales.resize(pack.K_blocks * N);

                // Check for IINT8Unpackable interface (NATIVE unpacking path - preferred)
                const auto *int8_unpackable = dynamic_cast<const IINT8Unpackable *>(&tensor);

                // Check for IQ8_0Decodable interface (Q8_0 decode path - fallback)
                const auto *q8_0_decodable = dynamic_cast<const IQ8_0Decodable *>(&tensor);

                // Check for ITensorGemmTileDataProvider (FP32 decode + requantization - slow fallback)
                const auto *tile_provider = dynamic_cast<const ITensorGemmTileDataProvider *>(&tensor);

                if (!int8_unpackable && !q8_0_decodable && !tile_provider)
                {
                    throw std::runtime_error("Tensor does not support block access for generic packing");
                }

#pragma omp parallel for
                for (int n = 0; n < N; ++n)
                {
                    for (int kb = 0; kb < pack.K_blocks; ++kb)
                    {
                        int8_t unpacked_values[32];
                        float scale_fp32;

                        if (int8_unpackable)
                        {
                            // NATIVE unpacking path: Q4_0 → int8 native range (NO requantization)
                            // This preserves original quantization (e.g., Q4_0 range [-8, 7])
                            int8_unpackable->unpack_block_to_int8(n, kb, unpacked_values);
                            scale_fp32 = int8_unpackable->get_block_scale(n, kb);
                        }
                        else if (q8_0_decodable)
                        {
                            // Q8_0 decode path: Direct Q8_0 decode (s8 + scale)
                            Q8_0Block block;
                            q8_0_decodable->decode_to_q8_0(n, kb, &block);

                            std::memcpy(unpacked_values, block.qs, 32);
                            scale_fp32 = fp16_to_fp32(block.d);
                        }
                        else
                        {
                            // Slow fallback: Decode to FP32, then requantize to Q8_0
                            float f32_block[32];
                            tile_provider->decode_block_at(n, kb, f32_block);

                            // Requantize f32_block → Q8_0 range [-127, 127]
                            float max_abs = 0.0f;
                            for (int i = 0; i < 32; ++i)
                                max_abs = std::max(max_abs, std::abs(f32_block[i]));

                            float d = max_abs / 127.0f;
                            if (d < 1e-10f)
                                d = 1.0f;
                            float id = 1.0f / d;

                            scale_fp32 = d;
                            for (int i = 0; i < 32; ++i)
                            {
                                unpacked_values[i] = static_cast<int8_t>(std::round(f32_block[i] * id));
                            }
                        }

                        // Store scale (fp16)
                        pack.block_scales[kb * N + n] = fp32_to_fp16(scale_fp32);

                        // Store s8 values
                        // Layout: [K, N] row-major (as used in microkernel)
                        for (int i = 0; i < 32; ++i)
                        {
                            pack.unpacked_s8[(kb * 32 + i) * N + n] = unpacked_values[i];
                        }
                    }
                }

                return pack;
            }

            bool execute_blockwise_gemm(
                const Q8_1Block *A_blocks,
                const BlockWeightPack &weights,
                float *C,
                int M, int N, int K,
                const float *bias,
                float alpha, float beta,
                bool apply_softmax = false,
                const float *mask = nullptr)
            {
                const bool debug = (M <= 4 && N <= 8); // Enable debug for small matrices

                if (debug)
                {
                    LOG_DEBUG("[execute_blockwise_gemm] M=" << M << " N=" << N << " K=" << K);
                    LOG_DEBUG("[execute_blockwise_gemm] K_blocks=" << (K / 32));
                }

                // 1. Get BRGEMM kernel (once per call)
                // Cache key: N (since M=1, K_blk=32, N_blk=64 are fixed)
                static std::map<int, dnnl::ukernel::brgemm> kernel_cache;
                static std::mutex cache_mutex;

                dnnl::ukernel::brgemm *brg_ptr = nullptr;
                const int N_blk_gemm = 64;
                const int K_blk = 32;

                {
                    std::lock_guard<std::mutex> lock(cache_mutex);
                    auto it = kernel_cache.find(N);
                    if (it == kernel_cache.end())
                    {
                        // LDB must be N (stride of weights)
                        dnnl::ukernel::brgemm brg(1, N_blk_gemm, K_blk, 1, N, N, N,
                                                  dnnl::memory::data_type::s8, dnnl::memory::data_type::s8, dnnl::memory::data_type::s32,
                                                  true);
                        brg.set_add_C(false); // Overwrite C_accum
                        brg.finalize();
                        brg.generate();
                        it = kernel_cache.emplace(N, brg).first;
                    }
                    brg_ptr = &it->second;
                }

                // Loop over M rows
                for (int m = 0; m < M; ++m)
                {
                    const Q8_1Block *A_row = A_blocks + m * (K / 32);
                    float *C_row = C + m * N;
                    const float *mask_row = mask ? mask + m * N : nullptr;

                    // Get scratch buffers (Main thread's buffer)
                    auto &scratch = get_scratch_buffers();
                    ensure_byte_capacity(scratch.byte_buffer_a, K);
                    // We don't need int32_buffer for C_accum anymore as it's stack allocated per thread

                    // 1. No unpacking needed for s8*s8 kernel!
                    // We use A_row[kb].qs directly.
                    int K_blocks = K / 32;

                    if (debug && m == 0)
                    {
                        LOG_DEBUG("[execute_blockwise_gemm] Row 0, block 0 A_q8_1: d=" << fp16_to_fp32(A_row[0].d)
                                                                                       << " sum_qs=" << fp16_to_fp32(A_row[0].sum_qs));
                    }

                    // 2. Execute BRGEMM (Parallelized over N)
                    // Offsets for BRGEMM (always 0,0)
                    static const std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>> offsets = {{0, 0}};

#pragma omp parallel for schedule(static)
                    for (int n = 0; n < N; n += N_blk_gemm)
                    {
                        int N_chunk = std::min(N_blk_gemm, N - n); // Handle partial block

                        // Temporary buffers for unrolled execution
                        // 4 buffers of 64 int32_t
                        alignas(64) int32_t C_temp[4][64];

                        int k_blk = 0;
                        const int8_t *B_base_n = weights.unpacked_s8.data() + n;

                        // Main loop unrolled by 4
                        for (; k_blk <= K_blocks - 4; k_blk += 4)
                        {
                            // Execute 4 GEMMs
                            // Block 0
                            const int8_t *A_ptr0 = A_row[k_blk].qs;
                            const int8_t *B_ptr0 = B_base_n + (static_cast<size_t>(k_blk) * K_blk) * N;
                            brg_ptr->execute(A_ptr0, B_ptr0, offsets, C_temp[0], nullptr);

                            // Block 1
                            const int8_t *A_ptr1 = A_row[k_blk + 1].qs;
                            const int8_t *B_ptr1 = B_base_n + (static_cast<size_t>(k_blk + 1) * K_blk) * N;
                            brg_ptr->execute(A_ptr1, B_ptr1, offsets, C_temp[1], nullptr);

                            // Block 2
                            const int8_t *A_ptr2 = A_row[k_blk + 2].qs;
                            const int8_t *B_ptr2 = B_base_n + (static_cast<size_t>(k_blk + 2) * K_blk) * N;
                            brg_ptr->execute(A_ptr2, B_ptr2, offsets, C_temp[2], nullptr);

                            // Block 3
                            const int8_t *A_ptr3 = A_row[k_blk + 3].qs;
                            const int8_t *B_ptr3 = B_base_n + (static_cast<size_t>(k_blk + 3) * K_blk) * N;
                            brg_ptr->execute(A_ptr3, B_ptr3, offsets, C_temp[3], nullptr);

                            // Scales for A
                            float d_A0 = fp16_to_fp32(A_row[k_blk].d);
                            float d_A1 = fp16_to_fp32(A_row[k_blk + 1].d);
                            float d_A2 = fp16_to_fp32(A_row[k_blk + 2].d);
                            float d_A3 = fp16_to_fp32(A_row[k_blk + 3].d);

                            // Pointers to B scales
                            const uint16_t *B_scales0 = weights.block_scales.data() + static_cast<size_t>(k_blk) * N + n;
                            const uint16_t *B_scales1 = weights.block_scales.data() + static_cast<size_t>(k_blk + 1) * N + n;
                            const uint16_t *B_scales2 = weights.block_scales.data() + static_cast<size_t>(k_blk + 2) * N + n;
                            const uint16_t *B_scales3 = weights.block_scales.data() + static_cast<size_t>(k_blk + 3) * N + n;

                            // Accumulate 4 blocks
                            bool accumulate_beta = (k_blk == 0 && beta != 0.0f);
                            // If k_blk > 0, we always accumulate (beta is effectively 1.0 for the accumulator)
                            // But wait, if k_blk > 0, we just add to C_row.
                            // My helper logic:
                            // if accumulate_beta: C = alpha*term + beta*C
                            // else: C += alpha*term

                            // So if k_blk == 0:
                            //   if beta != 0: accumulate_beta = true
                            //   else: accumulate_beta = false (C = alpha*term, overwrites C)
                            // if k_blk > 0:
                            //   accumulate_beta = false (C += alpha*term)

                            bool use_beta = (k_blk == 0 && beta != 0.0f);
                            // If k_blk == 0 and beta == 0, we want C = alpha*term.
                            // My helper does C += alpha*term if !accumulate_beta.
                            // This is WRONG for k_blk == 0 && beta == 0. We need to overwrite.

                            // Let's fix the helper usage or the helper itself.
                            // Actually, C_row is initialized to 0? No, it's an output buffer.
                            // If k_blk == 0, we must overwrite C_row unless beta != 0.

                            // I'll modify the helper call to handle this.
                            // But wait, I can't easily change the helper behavior without a flag.
                            // Let's assume I fix the helper logic in my head:
                            // Helper:
                            // if (accumulate_beta) C = ... + beta*C
                            // else C += ...

                            // If k_blk == 0 and beta == 0, I want C = ...
                            // So I need to zero C_row first? Or pass a flag "overwrite".

                            // Let's just zero C_row if k_blk == 0 and beta == 0?
                            // That would be an extra pass.

                            // Better: Update helper to take "overwrite" flag.
                            // Or just handle it.

                            // For now, let's assume C_row contains garbage or previous values.

                            // I will use detail::accumulate_simd_4
                            // But I need to handle the overwrite case.

                            // Let's look at the original code:
                            // if (k_blk == 0) {
                            //    if (beta != 0.0f) C = alpha*term + beta*C
                            //    else C = alpha*term
                            // } else {
                            //    C += alpha*term
                            // }

                            // I'll update the helper to support this.
                            // But I already wrote the helper.
                            // I'll update the helper in a separate edit if needed, or just use what I have.
                            // My helper:
                            // if (accumulate_beta) ...
                            // else C += ...

                            // It doesn't support "overwrite".
                            // I should have checked this.

                            // I will update the helper in the previous step? No, I already sent the edit.
                            // I will update the helper NOW.

                            // Wait, I can just zero C_row if k_blk == 0 && beta == 0.
                            // memset(C_row + n, 0, N_chunk * sizeof(float));
                            // Then use accumulate_beta = false (C += ... which is 0 + ...).
                            // This is safe and easy.

                            if (k_blk == 0 && beta == 0.0f)
                            {
                                std::memset(C_row + n, 0, N_chunk * sizeof(float));
                            }

                            detail::accumulate_simd_4(
                                N_chunk,
                                C_temp[0], C_temp[1], C_temp[2], C_temp[3],
                                d_A0, d_A1, d_A2, d_A3,
                                B_scales0, B_scales1, B_scales2, B_scales3,
                                C_row + n,
                                alpha, beta,
                                use_beta);
                        }

                        // Tail loop
                        for (; k_blk < K_blocks; ++k_blk)
                        {
                            const int8_t *A_ptr_k = A_row[k_blk].qs;
                            const int8_t *B_ptr_k = B_base_n + (static_cast<size_t>(k_blk) * K_blk) * N;

                            brg_ptr->execute(A_ptr_k, B_ptr_k, offsets, C_temp[0], nullptr);

                            float d_A = fp16_to_fp32(A_row[k_blk].d);
                            const uint16_t *B_scales = weights.block_scales.data() + static_cast<size_t>(k_blk) * N + n;

                            bool use_beta = (k_blk == 0 && beta != 0.0f);
                            if (k_blk == 0 && beta == 0.0f)
                            {
                                std::memset(C_row + n, 0, N_chunk * sizeof(float));
                            }

                            detail::accumulate_simd_1(
                                N_chunk,
                                C_temp[0],
                                d_A,
                                B_scales,
                                C_row + n,
                                alpha, beta,
                                use_beta);
                        }
                    }

                    // Apply bias if present
                    if (bias)
                    {
#pragma omp parallel for simd
                        for (int n = 0; n < N; ++n)
                        {
                            C_row[n] += bias[n];
                        }
                    }

                    // Apply mask if present (before softmax)
                    if (mask_row)
                    {
#pragma omp parallel for simd
                        for (int n = 0; n < N; ++n)
                        {
                            C_row[n] += mask_row[n];
                        }
                    }

                    // Apply Softmax if requested
                    if (apply_softmax)
                    {
                        // Find max
                        float max_val = -std::numeric_limits<float>::infinity();
#pragma omp parallel for reduction(max : max_val)
                        for (int n = 0; n < N; ++n)
                        {
                            if (C_row[n] > max_val)
                                max_val = C_row[n];
                        }

                        // Compute exp sum
                        float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
                        for (int n = 0; n < N; ++n)
                        {
                            C_row[n] = std::exp(C_row[n] - max_val);
                            sum += C_row[n];
                        }

                        // Normalize
                        float inv_sum = 1.0f / sum;
#pragma omp parallel for simd
                        for (int n = 0; n < N; ++n)
                        {
                            C_row[n] *= inv_sum;
                        }
                    }
                }

                if (debug)
                {
                    LOG_DEBUG("[execute_blockwise_gemm] Final output C[0]=" << C[0] << " C[1]=" << C[1]);
                }

                return true;
            }

        public:
            // 1. FP32 Act x FP32 Wgt
            bool multiply(
                const float *A, float *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)mpi_ctx;
                (void)device_idx;

                if (!weight_tensor_)
                {
                    LOG_ERROR("[OneDNNGemmKernel] multiply requires bound weight tensor");
                    return false;
                }

                // We only support transposed weights (standard for linear layers)
                if (!transpose_B)
                {
                    LOG_WARN("[OneDNNGemmKernel] non-transposed weights not optimized, proceeding anyway");
                }

                if (weight_tensor_->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[OneDNNGemmKernel] multiply requires FP32 weights, got " << static_cast<int>(weight_tensor_->native_type()));
                    return false;
                }

                const float *B = weight_tensor_->data();
                return run_onednn_fp32_matmul(A, B, C, m, n, k, transpose_B, alpha, beta);
            }

            // 2. FP16 Act x FP16 Wgt & 3. BF16 Act x BF16 Wgt
            bool multiply_activations_typed_impl(
                const void *A, const void *B, float *C,
                int m, int n, int k,
                bool transpose_B,
                float alpha, float beta,
                const MPIContext *mpi_ctx,
                int device_idx,
                ActivationFormat format_A, ActivationFormat format_B) override
            {
                (void)mpi_ctx;
                (void)device_idx;

                // Case 2: FP16 x FP16
                if (format_A == ActivationFormat::FP16 && format_B == ActivationFormat::FP16)
                {
                    const uint16_t *A_ptr = static_cast<const uint16_t *>(A);
                    const uint16_t *B_ptr = static_cast<const uint16_t *>(B);

                    if (!B_ptr && weight_tensor_)
                    {
                        if (weight_tensor_->native_type() != TensorType::FP16)
                        {
                            LOG_ERROR("[OneDNNGemmKernel] FP16 activation requires FP16 weights");
                            return false;
                        }
                        B_ptr = reinterpret_cast<const uint16_t *>(weight_tensor_->data());
                    }

                    if (!B_ptr)
                        return false;

                    return run_onednn_fp16_matmul(A_ptr, B_ptr, C, m, n, k, alpha, beta);
                }

                // Case 3: BF16 x BF16
                if (format_A == ActivationFormat::BF16 && format_B == ActivationFormat::BF16)
                {
                    const uint16_t *A_ptr = static_cast<const uint16_t *>(A);
                    const uint16_t *B_ptr = static_cast<const uint16_t *>(B);

                    if (!B_ptr && weight_tensor_)
                    {
                        if (weight_tensor_->native_type() != TensorType::BF16)
                        {
                            LOG_ERROR("[OneDNNGemmKernel] BF16 activation requires BF16 weights");
                            return false;
                        }
                        B_ptr = reinterpret_cast<const uint16_t *>(weight_tensor_->data());
                    }

                    if (!B_ptr)
                        return false;

                    return run_onednn_bf16_matmul(A_ptr, B_ptr, C, m, n, k, alpha, beta);
                }

                LOG_ERROR("[OneDNNGemmKernel] Unsupported typed activation combination");
                return false;
            }

            // 4. Q8_1 Act x IINT8Unpackable Wgt (Blockwise)
            bool multiply_typed_activations(
                const void *A, TensorFormat format_A, const float *A_scales,
                float *C, int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)mpi_ctx;
                (void)device_idx;
                (void)format_A;
                (void)A_scales;
                (void)transpose_B;

                if (!weight_tensor_)
                    return false;

                // Check if weights are IINT8Unpackable
                if (!weight_tensor_->cache_.has_value())
                {
                    try
                    {
                        weight_tensor_->cache_ = pack_weights_generic_blockwise(*weight_tensor_, k, n);
                    }
                    catch (...)
                    {
                        return false;
                    }
                }

                const auto &weights_pack = std::any_cast<const BlockWeightPack &>(weight_tensor_->cache_);
                const Q8_1Block *A_blocks = static_cast<const Q8_1Block *>(A);

                return execute_blockwise_gemm(A_blocks, weights_pack, C, m, n, k, nullptr, alpha, beta);
            }

            // Q8_1 + Softmax
            bool multiply_with_softmax_typed_impl(
                const void *A, const void *B, float *C,
                int m, int n, int k,
                float scale,
                bool transpose_B,
                int softmax_axis,
                const float *mask,
                bool is_causal,
                const MPIContext *mpi_ctx,
                int device_idx,
                ActivationFormat format_A, ActivationFormat format_B) override
            {
                (void)B;
                (void)transpose_B;
                (void)softmax_axis;
                (void)is_causal;
                (void)mpi_ctx;
                (void)device_idx;
                (void)format_A;
                (void)format_B;

                if (!weight_tensor_)
                    return false;

                if (!weight_tensor_->cache_.has_value())
                {
                    try
                    {
                        weight_tensor_->cache_ = pack_weights_generic_blockwise(*weight_tensor_, k, n);
                    }
                    catch (...)
                    {
                        return false;
                    }
                }

                const auto &weights_pack = std::any_cast<const BlockWeightPack &>(weight_tensor_->cache_);
                const Q8_1Block *A_blocks = static_cast<const Q8_1Block *>(A);

                return execute_blockwise_gemm(A_blocks, weights_pack, C, m, n, k, nullptr, scale, 0.0f, true, mask);
            }

            // Deprecated / Unused
            bool multiply_activations(const float *A, const float *B, float *C, int m, int n, int k, bool transpose_B, float alpha, float beta, const MPIContext *mpi_ctx, int device_idx) override
            {
                if (!B && weight_tensor_)
                {
                    return multiply(A, C, m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx);
                }
                return run_onednn_fp32_matmul(A, B, C, m, n, k, transpose_B, alpha, beta);
            }

            bool multiply_with_softmax(const float *A, const float *B, float *C, int m, int n, int k, bool transpose_B, int softmax_axis, const float *mask, const MPIContext *mpi_ctx, int device_idx) override
            {
                LOG_ERROR("[OneDNNGemmKernel] multiply_with_softmax is deprecated. Use typed version.");
                return false;
            }

            bool multiply_activations_strided(const float *A, const float *B, float *C, int m, int n, int k, int lda, int ldb, int ldc, bool transpose_B, float alpha, float beta, const MPIContext *mpi_ctx, int device_idx) override
            {
                LOG_ERROR("[OneDNNGemmKernel] multiply_activations_strided is deprecated.");
                return false;
            }

            bool multiply_activations_strided_typed_impl(const void *A, const void *B, float *C, int m, int n, int k, int lda, int ldb, int ldc, bool transpose_B, float alpha, float beta, const MPIContext *mpi_ctx, int device_idx, ActivationFormat format_A, ActivationFormat format_B) override
            {
                LOG_ERROR("[OneDNNGemmKernel] multiply_activations_strided_typed_impl is deprecated.");
                return false;
            }

            // ========== PUBLIC TEST API (for integration tests) ==========
            std::optional<BlockWeightPack> get_blockwise_weight_pack(int K, int N)
            {
                if (!weight_tensor_)
                    return std::nullopt;
                try
                {
                    return pack_weights_generic_blockwise(*weight_tensor_, K, N);
                }
                catch (...)
                {
                    return std::nullopt;
                }
            }

            bool execute_blockwise_gemm_test(const Q8_1Block *A_q8_1, const BlockWeightPack &weights, float *C, int M, int N, int K, const float *bias = nullptr, float alpha = 1.0f, float beta = 0.0f)
            {
                return execute_blockwise_gemm(A_q8_1, weights, C, M, N, K, bias, alpha, beta);
            }
            // ========== END PUBLIC TEST API ==========

        private:
            const TensorBase *weight_tensor_; ///< Bound weight tensor (B matrix)
        };

    } // namespace gemm_v4
} // namespace llaminar2
