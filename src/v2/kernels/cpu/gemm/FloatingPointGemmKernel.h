/**
 * @file FloatingPointGemmKernel.h
 * @brief ITensorGemm implementation for floating-point GEMM using OneDNN
 *
 * Provides optimized GEMM for homogeneous floating-point type combinations:
 * - FP32 weights × FP32 activations → FP32 output
 * - FP16 weights × FP16 activations → FP32 output
 * - BF16 weights × BF16 activations → FP32 output
 *
 * For quantized weight GEMM (Q4_0, Q8_0, Q8_1, etc.), use CPUQuantisedGemmKernel.
 *
 * @author David Sanftenberg
 * @date 2025-11-26
 */

#pragma once

#ifndef HAVE_ONEDNN
#error "OneDNN support is required for FloatingPointGemmKernel"
#endif

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"
#include "../CPUKernelBase.h"
#include "../primitives/SoftmaxPrimitives_New.h"

namespace llaminar2
{
    namespace gemm
    {
        // ========== OneDNN Engine/Stream Singletons ==========

        /**
         * @brief Get thread-local OneDNN CPU engine
         */
        inline dnnl::engine &onednn_engine()
        {
            static thread_local dnnl::engine engine_instance(dnnl::engine::kind::cpu, 0);
            return engine_instance;
        }

        /**
         * @brief Get thread-local OneDNN execution stream
         */
        inline dnnl::stream &onednn_stream()
        {
            static thread_local dnnl::stream stream_instance(onednn_engine());
            return stream_instance;
        }

        // ========== OneDNN GEMM Primitives ==========

        /**
         * @brief Execute FP32 matrix multiplication using OneDNN with optional fused bias
         *
         * @param A Input matrix A [M, K] (FP32, row-major)
         * @param B Input matrix B [K, N] or [N, K] if transpose_B (FP32)
         * @param C Output matrix C [M, N] (FP32, row-major)
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A and rows in B
         * @param transpose_B Whether B is transposed (stored as [N, K])
         * @param alpha Scale factor for A*B
         * @param beta Scale factor for existing C (for accumulation)
         * @param bias Optional bias vector [N] to fuse into GEMM (nullptr = no bias)
         * @return true on success
         */
        inline bool run_onednn_fp32_matmul(const float *A,
                                           const float *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           bool transpose_B,
                                           float alpha = 1.0f,
                                           float beta = 0.0f,
                                           const float *bias = nullptr)
        {
            const KernelType profile_type = (M == 1) ? KernelType::GEMV_FP32 : KernelType::GEMM_FP32;
            KERNEL_PROFILE_SCOPE(profile_type);

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

                // Optional bias descriptor - shape [1, N] for broadcast across rows
                dnnl::memory::desc bias_md;
                if (bias)
                {
                    dnnl::memory::dims bias_dims = {1, N};
                    bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::ab);
                }

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling: f(x) = alpha * x + 0
                // This is more robust than set_scales_mask for simple scalar multiplication
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                // Create primitive descriptor with or without bias
                dnnl::matmul::primitive_desc matmul_pd = bias
                                                             ? dnnl::matmul::primitive_desc(onednn_engine(), src_md, weight_md, bias_md, dst_md, attr)
                                                             : dnnl::matmul::primitive_desc(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<float *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                // Add bias memory if provided
                if (bias)
                {
                    dnnl::memory bias_mem(bias_md, onednn_engine(), const_cast<float *>(bias));
                    args.insert({DNNL_ARG_BIAS, bias_mem});
                }

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN FP32 matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("OneDNN FP32 matmul failed: " << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute BF16 matrix multiplication using OneDNN (bf16×bf16→f32)
         *
         * @param A Input matrix A [M, K] (BF16 as uint16_t, row-major)
         * @param B Input matrix B [K, N] (BF16 as uint16_t, row-major)
         * @param C Output matrix C [M, N] (FP32, row-major)
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A and rows in B
         * @param transpose_B Whether B is transposed
         * @param alpha Scale factor for A*B
         * @param beta Scale factor for existing C
         * @return true on success
         */
        inline bool run_onednn_bf16_matmul(const uint16_t *A,
                                           const uint16_t *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           bool transpose_B = false,
                                           float alpha = 1.0f,
                                           float beta = 0.0f)
        {
            const KernelType profile_type = (M == 1) ? KernelType::GEMV_FP32 : KernelType::GEMM_FP32;
            KERNEL_PROFILE_SCOPE(profile_type);

            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::bf16, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::bf16, transpose_B ? tag::ba : tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling: f(x) = alpha * x + 0
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
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

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN BF16 matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute FP16 matrix multiplication using OneDNN (fp16×fp16→f32)
         *
         * @param A Input matrix A [M, K] (FP16 as uint16_t, row-major)
         * @param B Input matrix B [K, N] (FP16 as uint16_t, row-major)
         * @param C Output matrix C [M, N] (FP32, row-major)
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A and rows in B
         * @param transpose_B Whether B is transposed
         * @param alpha Scale factor for A*B
         * @param beta Scale factor for existing C
         * @return true on success
         */
        inline bool run_onednn_fp16_matmul(const uint16_t *A,
                                           const uint16_t *B,
                                           float *C,
                                           int M,
                                           int N,
                                           int K,
                                           bool transpose_B = false,
                                           float alpha = 1.0f,
                                           float beta = 0.0f)
        {
            const KernelType profile_type = (M == 1) ? KernelType::GEMV_FP32 : KernelType::GEMM_FP32;
            KERNEL_PROFILE_SCOPE(profile_type);

            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims weight_dims = {K, N};
                dnnl::memory::dims dst_dims = {M, N};

                auto src_md = dnnl::memory::desc(src_dims, dt::f16, tag::ab);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f16, transpose_B ? tag::ba : tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling: f(x) = alpha * x + 0
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
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

        // ========== Strided OneDNN GEMM Primitives ==========

        /**
         * @brief Execute FP32 strided matrix multiplication using OneDNN
         *
         * Supports non-contiguous memory layouts via custom strides.
         * Essential for multi-head attention where Q/K/V heads are interleaved.
         *
         * @param A Input matrix A [M, K] with leading dimension lda
         * @param B Input matrix B [N, K] if transpose_B, else [K, N], with leading dimension ldb
         * @param C Output matrix C [M, N] with leading dimension ldc
         * @param M Number of rows in A and C
         * @param N Number of rows in B (if transpose_B) or columns in B
         * @param K Number of columns in A
         * @param lda Leading dimension of A (stride between rows)
         * @param ldb Leading dimension of B (stride between rows)
         * @param ldc Leading dimension of C (stride between rows)
         * @param transpose_B Whether B is transposed (stored as [N, K])
         * @param alpha Scale factor for A*B
         * @param beta Scale factor for existing C (for accumulation)
         * @return true on success
         */
        inline bool run_onednn_fp32_matmul_strided(const float *A,
                                                   const float *B,
                                                   float *C,
                                                   int M,
                                                   int N,
                                                   int K,
                                                   int lda,
                                                   int ldb,
                                                   int ldc,
                                                   bool transpose_B,
                                                   float alpha = 1.0f,
                                                   float beta = 0.0f)
        {
            const KernelType profile_type = (M == 1) ? KernelType::GEMV_FP32 : KernelType::GEMM_FP32;
            KERNEL_PROFILE_SCOPE(profile_type);

            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                // Memory dimensions (logical shape)
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims dst_dims = {M, N};

                // Strides for row-major layout: {stride_between_rows, stride_between_cols}
                // For row-major: stride_between_cols = 1, stride_between_rows = lda
                dnnl::memory::dims src_strides = {lda, 1};
                dnnl::memory::dims dst_strides = {ldc, 1};

                // Weight dimensions and strides depend on transpose_B
                dnnl::memory::dims weight_dims;
                dnnl::memory::dims weight_strides;

                if (transpose_B)
                {
                    // B is stored as [N, K] (each row is a K-vector)
                    // We want to compute A @ B^T, so logical weight shape is [K, N]
                    weight_dims = {K, N};
                    // B^T: to get element at logical [k, n], we access B[n, k] = B[n * ldb + k]
                    // OneDNN strides for transposed: {1, ldb} means:
                    //   - stride along K (first dim) = 1
                    //   - stride along N (second dim) = ldb
                    weight_strides = {1, ldb};
                }
                else
                {
                    // B is stored as [K, N], logical shape is [K, N]
                    weight_dims = {K, N};
                    weight_strides = {ldb, 1};
                }

                auto src_md = dnnl::memory::desc(src_dims, dt::f32, src_strides);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f32, weight_strides);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_strides);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling: f(x) = alpha * x + 0
                // This is more robust than set_scales_mask for simple scalar multiplication
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }

                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<float *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN FP32 strided matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("OneDNN FP32 strided matmul failed: " << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute BF16 strided matrix multiplication using OneDNN
         */
        inline bool run_onednn_bf16_matmul_strided(const uint16_t *A,
                                                   const uint16_t *B,
                                                   float *C,
                                                   int M,
                                                   int N,
                                                   int K,
                                                   int lda,
                                                   int ldb,
                                                   int ldc,
                                                   bool transpose_B,
                                                   float alpha = 1.0f,
                                                   float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims dst_dims = {M, N};
                dnnl::memory::dims src_strides = {lda, 1};
                dnnl::memory::dims dst_strides = {ldc, 1};

                dnnl::memory::dims weight_dims;
                dnnl::memory::dims weight_strides;

                if (transpose_B)
                {
                    weight_dims = {K, N};
                    weight_strides = {1, ldb};
                }
                else
                {
                    weight_dims = {K, N};
                    weight_strides = {ldb, 1};
                }

                auto src_md = dnnl::memory::desc(src_dims, dt::bf16, src_strides);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::bf16, weight_strides);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_strides);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
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

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN BF16 strided matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute FP16 strided matrix multiplication using OneDNN
         */
        inline bool run_onednn_fp16_matmul_strided(const uint16_t *A,
                                                   const uint16_t *B,
                                                   float *C,
                                                   int M,
                                                   int N,
                                                   int K,
                                                   int lda,
                                                   int ldb,
                                                   int ldc,
                                                   bool transpose_B,
                                                   float alpha = 1.0f,
                                                   float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims dst_dims = {M, N};
                dnnl::memory::dims src_strides = {lda, 1};
                dnnl::memory::dims dst_strides = {ldc, 1};

                dnnl::memory::dims weight_dims;
                dnnl::memory::dims weight_strides;

                if (transpose_B)
                {
                    weight_dims = {K, N};
                    weight_strides = {1, ldb};
                }
                else
                {
                    weight_dims = {K, N};
                    weight_strides = {ldb, 1};
                }

                auto src_md = dnnl::memory::desc(src_dims, dt::f16, src_strides);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f16, weight_strides);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_strides);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
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

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_WARN("OneDNN FP16 strided matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute mixed FP32×BF16 strided matrix multiplication using OneDNN
         *
         * Used for attention: FP32 scores × BF16 V → FP32 output
         */
        inline bool run_onednn_fp32_bf16_matmul_strided(const float *A,
                                                        const uint16_t *B,
                                                        float *C,
                                                        int M,
                                                        int N,
                                                        int K,
                                                        int lda,
                                                        int ldb,
                                                        int ldc,
                                                        bool transpose_B,
                                                        float alpha = 1.0f,
                                                        float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims dst_dims = {M, N};
                dnnl::memory::dims src_strides = {lda, 1};
                dnnl::memory::dims dst_strides = {ldc, 1};

                dnnl::memory::dims weight_dims;
                dnnl::memory::dims weight_strides;

                if (transpose_B)
                {
                    weight_dims = {K, N};
                    weight_strides = {1, ldb};
                }
                else
                {
                    weight_dims = {K, N};
                    weight_strides = {ldb, 1};
                }

                // Mixed precision: FP32 source, BF16 weights, FP32 output
                auto src_md = dnnl::memory::desc(src_dims, dt::f32, src_strides);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::bf16, weight_strides);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_strides);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_WARN("OneDNN FP32×BF16 strided matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute mixed FP32×FP16 strided matrix multiplication using OneDNN
         *
         * Used for attention: FP32 scores × FP16 V → FP32 output
         */
        inline bool run_onednn_fp32_fp16_matmul_strided(const float *A,
                                                        const uint16_t *B,
                                                        float *C,
                                                        int M,
                                                        int N,
                                                        int K,
                                                        int lda,
                                                        int ldb,
                                                        int ldc,
                                                        bool transpose_B,
                                                        float alpha = 1.0f,
                                                        float beta = 0.0f)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                dnnl::memory::dims src_dims = {M, K};
                dnnl::memory::dims dst_dims = {M, N};
                dnnl::memory::dims src_strides = {lda, 1};
                dnnl::memory::dims dst_strides = {ldc, 1};

                dnnl::memory::dims weight_dims;
                dnnl::memory::dims weight_strides;

                if (transpose_B)
                {
                    weight_dims = {K, N};
                    weight_strides = {1, ldb};
                }
                else
                {
                    weight_dims = {K, N};
                    weight_strides = {ldb, 1};
                }

                // Mixed precision: FP32 source, FP16 weights, FP32 output
                auto src_md = dnnl::memory::desc(src_dims, dt::f32, src_strides);
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f16, weight_strides);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_strides);

                dnnl::primitive_attr attr;
                dnnl::post_ops ops;

                // Use eltwise linear for alpha scaling
                if (alpha != 1.0f)
                {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
                }
                if (beta != 0.0f)
                {
                    ops.append_sum(beta);
                }
                attr.set_post_ops(ops);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md, attr);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<uint16_t *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                std::unordered_map<int, dnnl::memory> args;
                args.insert({DNNL_ARG_SRC, src_mem});
                args.insert({DNNL_ARG_WEIGHTS, weight_mem});
                args.insert({DNNL_ARG_DST, dst_mem});

                dnnl::matmul(matmul_pd).execute(onednn_stream(), args);
                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_WARN("OneDNN FP32×FP16 strided matmul failed: status=" << e.status << " message=" << e.what());
                return false;
            }

            return true;
        }

        /**
         * @brief Execute Softmax using OneDNN
         *
         * Computes y = softmax(x) along the last dimension (axis=1).
         * Supports in-place operation (src == dst).
         */
        inline void run_onednn_softmax(float *data, int rows, int cols, int stride)
        {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;

            try
            {
                // Logical dimensions: {rows, cols}
                dnnl::memory::dims dims = {rows, cols};

                // Strides: {stride, 1}
                // OneDNN expects strides to match the memory layout
                dnnl::memory::dims strides = {stride, 1};

                auto md = dnnl::memory::desc(dims, dt::f32, strides);

                // Create memory object wrapping the data
                dnnl::memory mem(md, onednn_engine(), data);

                // Softmax primitive descriptor
                // axis = 1 (columns)
                auto softmax_pd = dnnl::softmax_forward::primitive_desc(
                    onednn_engine(),
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::softmax_accurate,
                    md,
                    md,
                    1);

                // Execute
                dnnl::softmax_forward(softmax_pd).execute(onednn_stream(), {
                                                                               {DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem} // In-place
                                                                           });

                onednn_stream().wait();
            }
            catch (const dnnl::error &e)
            {
                LOG_ERROR("OneDNN softmax failed: status=" << e.status << " message=" << e.what());
            }
        }

        // ========== Floating-Point GEMM Kernel ==========

        /**
         * @brief ITensorGemm implementation for floating-point GEMM
         *
         * Supports only homogeneous floating-point type combinations:
         * - FP32 weights × FP32 activations
         * - FP16 weights × FP16 activations
         * - BF16 weights × BF16 activations
         *
         * For quantized weight GEMM, use CPUQuantisedGemmKernel instead.
         */
        class FloatingPointGemmKernel : public ITensorGemm, public CPUKernelBase
        {
        public:
            /**
             * @brief Construct kernel bound to a floating-point weight tensor
             * @param weight_tensor Pointer to weight tensor (must be FP32, FP16, or BF16)
             */
            explicit FloatingPointGemmKernel(const TensorBase *weight_tensor)
                : weight_tensor_(weight_tensor)
            {
                if (weight_tensor)
                {
                    weight_type_ = weight_tensor->native_type();

                    // Validate weight type
                    if (weight_type_ != TensorType::FP32 &&
                        weight_type_ != TensorType::FP16 &&
                        weight_type_ != TensorType::BF16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Unsupported weight type: " << static_cast<int>(weight_type_));
                        throw std::runtime_error("FloatingPointGemmKernel only supports FP32, FP16, or BF16 weights");
                    }
                }
            }

            ~FloatingPointGemmKernel() override = default;

            /**
             * @brief Check device support (CPU-only for OneDNN)
             */
            bool supports_device(int device_idx) const override
            {
                return device_idx == -1; // CPU only
            }

            // IKernelSnapshotCapable interface
            KernelSnapshotInfo getKernelSnapshotInfo() const override
            {
                // Determine weight dtype from bound tensor
                KernelBufferDtype weight_dtype = KernelBufferDtype::FP32;
                if (weight_type_ == TensorType::FP16)
                    weight_dtype = KernelBufferDtype::FP16;
                else if (weight_type_ == TensorType::BF16)
                    weight_dtype = KernelBufferDtype::BF16;

                return KernelSnapshotInfo::gemm()
                    .withInput("A", "input activations [m, k]", weight_dtype)
                    .withWeight("B", "weight matrix [n, k]", weight_dtype)
                    .withOutput("C", "output matrix [m, n]", weight_dtype)
                    .withScalar("m", "batch dimension", KernelBufferDtype::INT32)
                    .withScalar("n", "output features", KernelBufferDtype::INT32)
                    .withScalar("k", "input features", KernelBufferDtype::INT32)
                    .withScalar("alpha", "output scale factor")
                    .withScalar("beta", "accumulate scale factor");
            }

            // ========== Tensor-Based Interface: multiply_tensor() ==========

            /**
             * @brief Tensor-based GEMM with runtime type checking
             *
             * Validates that activation tensor type matches weight tensor type.
             * Supported combinations:
             * - FP32 activation × FP32 weight
             * - FP16 activation × FP16 weight
             * - BF16 activation × BF16 weight
             *
             * @return true on success, false if type combination not supported
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr,
                int activation_row_offset = 0) override
            {
                (void)mpi_ctx;
                (void)device_idx;
                (void)workspace; // CPU kernel doesn't need external workspace

                if (!weight_tensor_ || !A || !C)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Null tensor pointer");
                    return false;
                }

                const TensorType act_type = A->native_type();
                const auto &a_shape = A->shape();
                const auto &c_shape = C->shape();

                int m = static_cast<int>(a_shape[0]);
                int k = static_cast<int>(a_shape.size() > 1 ? a_shape[1] : 1);
                int n = static_cast<int>(c_shape.size() > 1 ? c_shape[1] : c_shape[0]);

                // Validate type match
                if (act_type != weight_type_)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Activation type (" << static_cast<int>(act_type)
                                                                            << ") must match weight type (" << static_cast<int>(weight_type_) << ")");
                    return false;
                }

                // Extract bias pointer (FP32 fused bias only for now)
                const float *bias_ptr = bias ? bias->data() : nullptr;

                const size_t row_offset = static_cast<size_t>(activation_row_offset) * k;

                switch (weight_type_)
                {
                case TensorType::FP32:
                {
                    const float *A_data = A->data() + row_offset;
                    float *C_data = C->mutable_data();
                    const float *B_data = weight_tensor_->data();
                    // OneDNN FP32 matmul supports fused bias natively
                    return run_onednn_fp32_matmul(A_data, B_data, C_data, m, n, k, transpose_B, alpha, beta, bias_ptr);
                }

                case TensorType::FP16:
                {
                    const auto *A_fp16 = dynamic_cast<const FP16Tensor *>(A);
                    const auto *B_fp16 = dynamic_cast<const FP16Tensor *>(weight_tensor_);
                    if (!A_fp16 || !B_fp16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast to FP16Tensor");
                        return false;
                    }
                    const uint16_t *A_data = A_fp16->typed_data();
                    const uint16_t *B_data = B_fp16->typed_data();
                    float *C_data = C->mutable_data();
                    // TODO: Add fused bias support for FP16 matmul
                    if (bias_ptr)
                    {
                        LOG_WARN("[FloatingPointGemmKernel] Bias not yet supported for FP16 GEMM");
                    }
                    return run_onednn_fp16_matmul(A_data, B_data, C_data, m, n, k, transpose_B, alpha, beta);
                }

                case TensorType::BF16:
                {
                    const auto *A_bf16 = dynamic_cast<const BF16Tensor *>(A);
                    const auto *B_bf16 = dynamic_cast<const BF16Tensor *>(weight_tensor_);
                    if (!A_bf16 || !B_bf16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast to BF16Tensor");
                        return false;
                    }
                    const uint16_t *A_data = A_bf16->typed_data();
                    const uint16_t *B_data = B_bf16->typed_data();
                    float *C_data = C->mutable_data();
                    // TODO: Add fused bias support for BF16 matmul
                    if (bias_ptr)
                    {
                        LOG_WARN("[FloatingPointGemmKernel] Bias not yet supported for BF16 GEMM");
                    }
                    return run_onednn_bf16_matmul(A_data, B_data, C_data, m, n, k, transpose_B, alpha, beta);
                }

                default:
                    LOG_ERROR("[FloatingPointGemmKernel] Unsupported weight type: " << static_cast<int>(weight_type_));
                    return false;
                }
            }

            /**
             * @brief Tensor-based GEMM with explicit dimensions
             *
             * This overload uses caller-provided m, n, k instead of inferring from tensor shapes.
             * Essential for pre-allocated buffers where tensor shape > actual data size.
             *
             * @param A Input activations tensor [>=m, >=k]
             * @param C Output tensor [>=m, >=n]
             * @param m Number of rows to process
             * @param n Number of output columns
             * @param k Number of input columns
             * @param transpose_B Whether B is transposed (typical: true for weights)
             * @param alpha Scale factor
             * @param beta Accumulate factor
             * @param bias Optional bias tensor [n] to add after GEMM (nullptr = no bias)
             * @param mpi_ctx MPI context
             * @param device_idx Device index
             *
             * @return true on success
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr,
                int activation_row_offset = 0) override
            {
                (void)mpi_ctx;
                (void)device_idx;
                (void)workspace; // CPU kernel doesn't need external workspace

                if (!weight_tensor_ || !A || !C)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Null tensor pointer");
                    return false;
                }

                const TensorType act_type = A->native_type();

                // Validate type match
                if (act_type != weight_type_)
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Activation type (" << static_cast<int>(act_type)
                                                                            << ") must match weight type (" << static_cast<int>(weight_type_) << ")");
                    return false;
                }

                // Extract bias pointer (FP32 fused bias only for now)
                const float *bias_ptr = bias ? bias->data() : nullptr;

                const size_t row_offset = static_cast<size_t>(activation_row_offset) * k;

                switch (weight_type_)
                {
                case TensorType::FP32:
                {
                    const float *A_data = A->data() + row_offset;
                    float *C_data = C->mutable_data();
                    const float *B_data = weight_tensor_->data();
                    // OneDNN FP32 matmul supports fused bias natively
                    return run_onednn_fp32_matmul(A_data, B_data, C_data, m, n, k, transpose_B, alpha, beta, bias_ptr);
                }

                case TensorType::FP16:
                {
                    const auto *A_fp16 = dynamic_cast<const FP16Tensor *>(A);
                    const auto *B_fp16 = dynamic_cast<const FP16Tensor *>(weight_tensor_);
                    if (!A_fp16 || !B_fp16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast to FP16Tensor");
                        return false;
                    }
                    const uint16_t *A_data = A_fp16->typed_data();
                    const uint16_t *B_data = B_fp16->typed_data();
                    float *C_data = C->mutable_data();
                    // TODO: Add fused bias support for FP16 matmul
                    if (bias_ptr)
                    {
                        LOG_WARN("[FloatingPointGemmKernel] Bias not yet supported for FP16 GEMM");
                    }
                    return run_onednn_fp16_matmul(A_data, B_data, C_data, m, n, k, transpose_B, alpha, beta);
                }

                case TensorType::BF16:
                {
                    const auto *A_bf16 = dynamic_cast<const BF16Tensor *>(A);
                    const auto *B_bf16 = dynamic_cast<const BF16Tensor *>(weight_tensor_);
                    if (!A_bf16 || !B_bf16)
                    {
                        LOG_ERROR("[FloatingPointGemmKernel] Failed to cast to BF16Tensor");
                        return false;
                    }
                    const uint16_t *A_data = A_bf16->typed_data();
                    const uint16_t *B_data = B_bf16->typed_data();
                    float *C_data = C->mutable_data();
                    // TODO: Add fused bias support for BF16 matmul
                    if (bias_ptr)
                    {
                        LOG_WARN("[FloatingPointGemmKernel] Bias not yet supported for BF16 GEMM");
                    }
                    return run_onednn_bf16_matmul(A_data, B_data, C_data, m, n, k, transpose_B, alpha, beta);
                }

                default:
                    LOG_ERROR("[FloatingPointGemmKernel] Unsupported weight type: " << static_cast<int>(weight_type_));
                    return false;
                }
            }

            // ========== Typed Activation Interface ==========

            /**
             * @brief Typed activation GEMM (FP16×FP16 or BF16×BF16)
             */
            bool multiply_activations_typed_impl(
                const void *A, const void *B, float *C,
                int m, int n, int k,
                bool transpose_B,
                float alpha, float beta,
                const MPIContext *mpi_ctx,
                int device_idx,
                ActivationFormat format_A, ActivationFormat format_B)
            {
                (void)mpi_ctx;
                (void)device_idx;

                // FP16 × FP16
                if (format_A == ActivationFormat::FP16 && format_B == ActivationFormat::FP16)
                {
                    const uint16_t *A_ptr = static_cast<const uint16_t *>(A);
                    const uint16_t *B_ptr = static_cast<const uint16_t *>(B);

                    if (!B_ptr && weight_tensor_)
                    {
                        if (weight_type_ != TensorType::FP16)
                        {
                            LOG_ERROR("[FloatingPointGemmKernel] FP16 activation requires FP16 weights");
                            return false;
                        }
                        const auto *fp16_tensor = dynamic_cast<const FP16Tensor *>(weight_tensor_);
                        B_ptr = fp16_tensor ? fp16_tensor->typed_data() : nullptr;
                    }

                    if (!B_ptr)
                        return false;
                    return run_onednn_fp16_matmul(A_ptr, B_ptr, C, m, n, k, transpose_B, alpha, beta);
                }

                // BF16 × BF16
                if (format_A == ActivationFormat::BF16 && format_B == ActivationFormat::BF16)
                {
                    const uint16_t *A_ptr = static_cast<const uint16_t *>(A);
                    const uint16_t *B_ptr = static_cast<const uint16_t *>(B);

                    if (!B_ptr && weight_tensor_)
                    {
                        if (weight_type_ != TensorType::BF16)
                        {
                            LOG_ERROR("[FloatingPointGemmKernel] BF16 activation requires BF16 weights");
                            return false;
                        }
                        const auto *bf16_tensor = dynamic_cast<const BF16Tensor *>(weight_tensor_);
                        B_ptr = bf16_tensor ? bf16_tensor->typed_data() : nullptr;
                    }

                    if (!B_ptr)
                        return false;
                    return run_onednn_bf16_matmul(A_ptr, B_ptr, C, m, n, k, transpose_B, alpha, beta);
                }

                LOG_ERROR("[FloatingPointGemmKernel] Unsupported activation format combination");
                return false;
            }

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
                ActivationFormat format_A, ActivationFormat format_B)
            {
                // 1. GEMM with scale
                if (!multiply_activations_typed_impl(A, B, C, m, n, k, transpose_B, scale, 0.0f, mpi_ctx, device_idx, format_A, format_B))
                {
                    return false;
                }

                // 2. Add mask
                if (mask)
                {
                    auto add_mask_work = [&]()
                    {
#pragma omp for collapse(2) schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            for (int j = 0; j < n; ++j)
                            {
                                C[i * n + j] += mask[i * n + j];
                            }
                        }
                    };
                    OMP_WORKSHARE_REGION(add_mask_work);
                }

                // 3. Causal mask
                if (is_causal)
                {
                    auto causal_mask_work = [&]()
                    {
#pragma omp for schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            for (int j = i + 1; j < n; ++j)
                            {
                                C[i * n + j] = -std::numeric_limits<float>::infinity();
                            }
                        }
                    };
                    OMP_WORKSHARE_REGION(causal_mask_work);
                }

                // 4. Softmax
                // Use our robust primitives that handle -inf correctly
                primitives::softmax_row_major_fp32(C, m, n, false, 1.0f, true);

                return true;
            }

            bool multiply_activations(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                bool transpose_B, float alpha, float beta,
                const MPIContext *mpi_ctx, int device_idx)
            {
                if (!B && weight_tensor_)
                {
                    return run_onednn_fp32_matmul(A, nullptr, C, m, n, k, transpose_B, alpha, beta);
                }
                return run_onednn_fp32_matmul(A, B, C, m, n, k, transpose_B, alpha, beta);
            }

            /**
             * @brief Tensor-based activation×activation GEMM with type-aware dispatch
             *
             * Handles all input tensor type combinations:
             * - FP32 × FP32: Direct FP32 matmul
             * - BF16 × BF16: OneDNN bf16bf16f32
             * - FP16 × FP16: OneDNN fp16fp16f32
             * - Q8_1 × Q8_1: Dequant to FP32, then FP32 matmul (attention scores always FP32)
             *
             * Output is always written to C via mutable_data() (FP32 for attention scores).
             */
            bool multiply_activations_tensor(
                const TensorBase *A, const TensorBase *B, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                const auto &a_shape = A->shape();
                const auto &b_shape = B->shape();
                int m = static_cast<int>(a_shape[0]);
                int k = static_cast<int>(a_shape.size() > 1 ? a_shape[1] : 1);
                int n = transpose_B ? static_cast<int>(b_shape[0]) : static_cast<int>(b_shape.size() > 1 ? b_shape[1] : 1);

                const TensorType a_type = A->native_type();
                const TensorType b_type = B->native_type();

                // FP32 path (most common for attention)
                if (a_type == TensorType::FP32 && b_type == TensorType::FP32)
                {
                    return run_onednn_fp32_matmul(
                        A->data(), B->data(), C->mutable_data(),
                        m, n, k, transpose_B, alpha, beta);
                }

                // BF16 path
                if (a_type == TensorType::BF16 && b_type == TensorType::BF16)
                {
                    const auto *A_bf16 = dynamic_cast<const BF16Tensor *>(A);
                    const auto *B_bf16 = dynamic_cast<const BF16Tensor *>(B);
                    if (A_bf16 && B_bf16)
                    {
                        return run_onednn_bf16_matmul(
                            A_bf16->typed_data(), B_bf16->typed_data(), C->mutable_data(),
                            m, n, k, transpose_B, alpha, beta);
                    }
                }

                // FP16 path
                if (a_type == TensorType::FP16 && b_type == TensorType::FP16)
                {
                    const auto *A_fp16 = dynamic_cast<const FP16Tensor *>(A);
                    const auto *B_fp16 = dynamic_cast<const FP16Tensor *>(B);
                    if (A_fp16 && B_fp16)
                    {
                        return run_onednn_fp16_matmul(
                            A_fp16->typed_data(), B_fp16->typed_data(), C->mutable_data(),
                            m, n, k, transpose_B, alpha, beta);
                    }
                }

                // Q8_1 path: Use fp32_data() for explicit dequantization
                // Attention scores are always FP32 (softmax needs FP32 precision)
                if (a_type == TensorType::Q8_1 && b_type == TensorType::Q8_1)
                {
                    return run_onednn_fp32_matmul(
                        A->fp32_data(), B->fp32_data(), C->mutable_data(),
                        m, n, k, transpose_B, alpha, beta);
                }

                // Fallback: Convert to FP32 via data()
                return run_onednn_fp32_matmul(
                    A->data(), B->data(), C->mutable_data(),
                    m, n, k, transpose_B, alpha, beta);
            }

            bool multiply_with_softmax(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                bool transpose_B, int softmax_axis, const float *mask,
                const MPIContext *mpi_ctx, int device_idx)
            {
                // 1. GEMM: C = A * B
                // Direct weight GEMM via stored weight tensor
                if (!run_onednn_fp32_matmul(A, nullptr, C, m, n, k, transpose_B, 1.0f, 0.0f))
                {
                    return false;
                }

                // 2. Add mask if provided
                if (mask)
                {
                    auto add_mask_work = [&]()
                    {
#pragma omp for collapse(2) schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            for (int j = 0; j < n; ++j)
                            {
                                C[i * n + j] += mask[i * n + j];
                            }
                        }
                    };
                    OMP_WORKSHARE_REGION(add_mask_work);
                }

                // 3. Softmax
                // Assuming contiguous C for now, or stride=n
                // Use our robust primitives that handle -inf correctly
                primitives::softmax_row_major_fp32(C, m, n, false, 1.0f, true);

                return true;
            }

            /**
             * @brief Fused strided GEMM + softmax for attention Q@K^T computation
             *
             * Computes: C = Softmax(scale * A @ B^T + mask)
             * Used by CpuAttentionKernelT for Q@K^T with causal masking
             */
            bool multiply_with_softmax_strided_typed_impl(
                const void *A, const void *B, float *C,
                int m, int n, int k,
                int lda, int ldb, int ldc,
                float scale,
                bool transpose_B,
                int softmax_axis,
                const float *mask,
                bool is_causal,
                const MPIContext *mpi_ctx,
                int device_idx,
                ActivationFormat format_A,
                ActivationFormat format_B) override
            {
                (void)mpi_ctx;
                (void)device_idx;
                (void)softmax_axis; // We always do row-wise softmax

                // Step 1: Strided GEMM with scaling
                bool gemm_ok = false;

                // Handle homogeneous formats
                if (format_A == format_B)
                {
                    switch (format_A)
                    {
                    case ActivationFormat::FP32:
                        gemm_ok = run_onednn_fp32_matmul_strided(
                            static_cast<const float *>(A),
                            static_cast<const float *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, scale, 0.0f);
                        break;

                    case ActivationFormat::BF16:
                        gemm_ok = run_onednn_bf16_matmul_strided(
                            static_cast<const uint16_t *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, scale, 0.0f);
                        break;

                    case ActivationFormat::FP16:
                        gemm_ok = run_onednn_fp16_matmul_strided(
                            static_cast<const uint16_t *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, scale, 0.0f);
                        break;

                    default:
                        LOG_ERROR("[FloatingPointGemmKernel] Unsupported format for fused GEMM+softmax: "
                                  << static_cast<int>(format_A));
                        return false;
                    }
                }
                else
                {
                    LOG_ERROR("[FloatingPointGemmKernel] Fused GEMM+softmax requires matching formats: A="
                              << static_cast<int>(format_A) << " B=" << static_cast<int>(format_B));
                    return false;
                }

                if (!gemm_ok)
                {
                    return false;
                }

                // Step 2: Apply mask if provided (add to scores)
                // The mask has shape [m, n] with stride ldc
                if (mask)
                {
                    for (int row = 0; row < m; ++row)
                    {
                        for (int col = 0; col < n; ++col)
                        {
                            C[row * ldc + col] += mask[row * n + col];
                        }
                    }
                }

                // Step 3: Apply causal mask (set future positions to -inf)
                if (is_causal)
                {
                    for (int row = 0; row < m; ++row)
                    {
                        for (int col = row + 1; col < n; ++col)
                        {
                            C[row * ldc + col] = -std::numeric_limits<float>::infinity();
                        }
                    }
                }

                // Step 4: Apply row-wise softmax
                // Use our robust primitives that handle -inf correctly (unlike OneDNN)
                if (ldc == n)
                {
                    primitives::softmax_row_major_fp32(C, m, n, false, 1.0f, true);
                }
                else
                {
                    auto softmax_row_work = [&]()
                    {
#pragma omp for schedule(static)
                        for (int r = 0; r < m; ++r)
                        {
                            primitives::softmax_row_fp32(C + r * ldc, n, false, 1.0f, -1);
                        }
                    };
                    OMP_WORKSHARE_REGION(softmax_row_work);
                }

                return true;
            }

            bool multiply_activations_strided(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                int lda, int ldb, int ldc,
                bool transpose_B, float alpha, float beta,
                const MPIContext *mpi_ctx, int device_idx) override
            {
                (void)mpi_ctx;
                (void)device_idx;

                return run_onednn_fp32_matmul_strided(A, B, C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);
            }

            bool multiply_activations_strided_typed_impl(
                const void *A, const void *B, float *C,
                int m, int n, int k,
                int lda, int ldb, int ldc,
                bool transpose_B, float alpha, float beta,
                const MPIContext *mpi_ctx, int device_idx,
                ActivationFormat format_A, ActivationFormat format_B) override
            {
                (void)mpi_ctx;
                (void)device_idx;

                // Handle homogeneous formats (same type for A and B)
                if (format_A == format_B)
                {
                    switch (format_A)
                    {
                    case ActivationFormat::FP32:
                        return run_onednn_fp32_matmul_strided(
                            static_cast<const float *>(A),
                            static_cast<const float *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);

                    case ActivationFormat::BF16:
                        return run_onednn_bf16_matmul_strided(
                            static_cast<const uint16_t *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);

                    case ActivationFormat::FP16:
                        return run_onednn_fp16_matmul_strided(
                            static_cast<const uint16_t *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);

                    default:
                        LOG_ERROR("[FloatingPointGemmKernel] Unsupported homogeneous format: "
                                  << static_cast<int>(format_A));
                        return false;
                    }
                }

                // Handle mixed-precision formats (FP32 scores × typed V for attention)
                if (format_A == ActivationFormat::FP32)
                {
                    switch (format_B)
                    {
                    case ActivationFormat::BF16:
                        return run_onednn_fp32_bf16_matmul_strided(
                            static_cast<const float *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);

                    case ActivationFormat::FP16:
                        return run_onednn_fp32_fp16_matmul_strided(
                            static_cast<const float *>(A),
                            static_cast<const uint16_t *>(B),
                            C, m, n, k, lda, ldb, ldc, transpose_B, alpha, beta);

                    default:
                        LOG_ERROR("[FloatingPointGemmKernel] Unsupported mixed format: FP32×"
                                  << static_cast<int>(format_B));
                        return false;
                    }
                }

                LOG_ERROR("[FloatingPointGemmKernel] Unsupported format combination: A="
                          << static_cast<int>(format_A) << " B=" << static_cast<int>(format_B));
                return false;
            }

        private:
            const TensorBase *weight_tensor_; ///< Bound weight tensor
            TensorType weight_type_ = TensorType::FP32;
        };

    } // namespace gemm
} // namespace llaminar2
