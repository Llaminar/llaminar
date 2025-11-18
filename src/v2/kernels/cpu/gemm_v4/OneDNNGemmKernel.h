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

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "OneDNNGemmAdapter.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{
    namespace gemm_v4
    {
        // ========== OneDNN Primitive Wrappers (from OneDNNGemm.h) ==========

        /**
         * @brief Get singleton OneDNN CPU engine
         */
        inline dnnl::engine &onednn_engine()
        {
            static dnnl::engine engine_instance(dnnl::engine::kind::cpu, 0);
            return engine_instance;
        }

        /**
         * @brief Get singleton OneDNN execution stream
         */
        inline dnnl::stream &onednn_stream()
        {
            static dnnl::stream stream_instance(onednn_engine());
            return stream_instance;
        }

        /**
         * @brief Execute INT8 matrix multiplication using OneDNN
         *
         * C = A @ B (INT8 x INT8 -> INT32 accumulation)
         *
         * @param A Input matrix A [M, K] (INT8, row-major)
         * @param B Input matrix B [K, N] (INT8, row-major)
         * @param C Output matrix C [M, N] (INT32, row-major)
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A and rows in B
         * @return true on success, false on error
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
            catch (const dnnl::error &)
            {
                return false;
            }

            return true;
        }

        /**
         * @brief Execute FP32 matrix multiplication using OneDNN (reference/validation)
         *
         * C = A @ B (FP32 x FP32 -> FP32)
         *
         * @param A Input matrix A [M, K] (FP32, row-major)
         * @param B Input matrix B [K, N] (FP32, row-major)
         * @param C Output matrix C [M, N] (FP32, row-major)
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A and rows in B
         * @return true on success, false on error
         */
        inline bool run_onednn_fp32_matmul(const float *A,
                                           const float *B,
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
                auto weight_md = dnnl::memory::desc(weight_dims, dt::f32, tag::ab);
                auto dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);

                dnnl::matmul::primitive_desc matmul_pd(onednn_engine(), src_md, weight_md, dst_md);

                dnnl::memory src_mem(src_md, onednn_engine(), const_cast<float *>(A));
                dnnl::memory weight_mem(weight_md, onednn_engine(), const_cast<float *>(B));
                dnnl::memory dst_mem(dst_md, onednn_engine(), C);

                dnnl::matmul(matmul_pd).execute(onednn_stream(),
                                                {{DNNL_ARG_SRC, src_mem},
                                                 {DNNL_ARG_WEIGHTS, weight_mem},
                                                 {DNNL_ARG_DST, dst_mem}});
                onednn_stream().wait();
            }
            catch (const dnnl::error &)
            {
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

            const int ndims = 2;
            int axis = softmax_axis;
            if (axis < 0)
            {
                axis += ndims;
            }
            if (axis != ndims - 1)
            {
                // Current OneDNN softmax post-op implementation only supports the last dimension.
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
            catch (const dnnl::error &)
            {
                return false;
            }

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

                    float sum = 0.0f;
                    for (int c = 0; c < cols; ++c)
                    {
                        float ex = exp_range(data[base + static_cast<size_t>(c)] - max_val);
                        data[base + static_cast<size_t>(c)] = ex;
                        sum += ex;
                    }

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
                for (int c = 0; c < cols; ++c)
                {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int r = 0; r < rows; ++r)
                    {
                        max_val = std::max(max_val, data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)]);
                    }

                    float sum = 0.0f;
                    for (int r = 0; r < rows; ++r)
                    {
                        float ex = exp_range(data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] - max_val);
                        data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] = ex;
                        sum += ex;
                    }

                    const float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                    for (int r = 0; r < rows; ++r)
                    {
                        data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] *= inv_sum;
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
            bool applyRMSNorm(const float *, int, int, float, const MPIContext *, int) override
            {
                throw std::runtime_error("ActivationView::applyRMSNorm() not supported");
            }
            bool applyRoPE(float *, const int *, int, int, int, int, float, bool, const MPIContext *, int) override
            {
                throw std::runtime_error("ActivationView::applyRoPE() not supported");
            }
        };

        // ========== ITensorGemm Implementation ==========

        /**
         * @brief OneDNN-based ITensorGemm kernel implementation
         *
         * **Usage**:
         * ```cpp
         * // In tensor's createGemm() method:
         * return std::make_unique<OneDNNGemmKernel>(this);
         * ```
         *
         * **Design**:
         * - Binds to a specific weight tensor at construction
         * - Works with weight tensors implementing TensorBase::to_int8_perchannel()
         * - Uses FP32 activations with per-row INT8 quantization
         * - Supports bias addition
         */
        class OneDNNGemmKernel : public ITensorGemm
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

            /**
             * @brief Execute GEMM: C = alpha * A @ B^T + beta * C
             *
             * Uses OneDNN INT8 acceleration with per-row activation and per-column weight scaling.
             *
             * @param A Input activations [m, k] (FP32, row-major)
             * @param C Output matrix [m, n] (FP32, row-major)
             * @param m Number of rows in A and C
             * @param n Number of columns in B and C (before transpose)
             * @param k Number of columns in A and rows in B
             * @param transpose_B Whether B is stored transposed (must be true for weights)
             * @param alpha Scale factor (must be 1.0)
             * @param beta Scale factor (must be 0.0)
             * @param mpi_ctx MPI context (unused, for future distributed support)
             * @param device_idx Device index (must be -1 for CPU)
             *
             * @return true on success, false on error
             *
             * @note Weight tensor B must be bound to this kernel via constructor
             * @note Currently only supports alpha=1.0, beta=0.0, transpose_B=true
             */
            bool multiply(const float *A, float *C,
                          int m, int n, int k,
                          bool transpose_B = true,
                          float alpha = 1.0f,
                          float beta = 0.0f,
                          const MPIContext *mpi_ctx = nullptr,
                          int device_idx = -1) override
            {
                // Validate CPU-only execution
                if (device_idx != -1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Only CPU execution supported (device_idx="
                              << device_idx << ")");
                    return false;
                }

                // Check for unsupported parameters
                if (alpha != 1.0f || beta != 0.0f)
                {
                    LOG_ERROR("[OneDNNGemmKernel] alpha/beta scaling not yet supported");
                    return false;
                }

                if (!transpose_B)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Non-transposed B not yet supported");
                    return false;
                }

                // Validate weight tensor is bound
                if (!weight_tensor_)
                {
                    LOG_ERROR("[OneDNNGemmKernel] No weight tensor bound to kernel");
                    return false;
                }

                // Validate dimensions
                const auto &shape = weight_tensor_->shape();
                if (shape.size() != 2)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Weight tensor must be 2D");
                    return false;
                }

                // Weight tensor is [N, K] for transpose_B=true (most common layout)
                if (static_cast<int>(shape[0]) != n || static_cast<int>(shape[1]) != k)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Weight tensor shape mismatch: expected ["
                              << n << "," << k << "], got ["
                              << shape[0] << "," << shape[1] << "]");
                    return false;
                }

                // Create lightweight views (no allocation or memcpy!)
                ActivationView activation_view(A, static_cast<size_t>(m), static_cast<size_t>(k));
                ActivationView output_view(C, static_cast<size_t>(m), static_cast<size_t>(n));

                // Execute using adapter (activations quantized on-the-fly)
                return onednn_gemm_adapter(m, n, k,
                                           activation_view,
                                           *weight_tensor_,
                                           output_view,
                                           nullptr); // No bias
            }

            bool multiply_with_softmax(const float *A, float *C,
                                       int m, int n, int k,
                                       bool transpose_B = true,
                                       int softmax_axis = 1,
                                       const MPIContext *mpi_ctx = nullptr,
                                       int device_idx = -1) override
            {
                (void)mpi_ctx;

                if (device_idx != -1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Only CPU execution supported (device_idx="
                              << device_idx << ")");
                    return false;
                }

                if (!transpose_B)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Non-transposed B not yet supported");
                    return false;
                }

                if (!weight_tensor_)
                {
                    LOG_ERROR("[OneDNNGemmKernel] No weight tensor bound to kernel");
                    return false;
                }

                if (m <= 0 || n <= 0 || k <= 0)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Invalid dimensions for fused multiply_with_softmax");
                    return false;
                }

                int axis = softmax_axis;
                if (axis < 0)
                {
                    axis += 2;
                }
                if (axis != 0 && axis != 1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Softmax axis must be 0, 1, or -1 (rows/cols)");
                    return false;
                }

                const auto &shape = weight_tensor_->shape();
                if (shape.size() != 2 || static_cast<int>(shape[0]) != n || static_cast<int>(shape[1]) != k)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Weight tensor shape mismatch for fused softmax path");
                    return false;
                }

                ActivationView activation_view(A, static_cast<size_t>(m), static_cast<size_t>(k));
                ActivationView output_view(C, static_cast<size_t>(m), static_cast<size_t>(n));

                if (!onednn_gemm_adapter(m, n, k,
                                         activation_view,
                                         *weight_tensor_,
                                         output_view,
                                         nullptr))
                {
                    LOG_ERROR("[OneDNNGemmKernel] OneDNN adapter matmul failed for fused softmax path");
                    return false;
                }

                if (!apply_softmax_inplace(C, m, n, axis))
                {
                    LOG_ERROR("[OneDNNGemmKernel] Softmax application failed after matmul");
                    return false;
                }

                return true;
            }

            /**
             * @brief Activation-activation GEMM using OneDNN FP32
             *
             * C = alpha * A @ B^T + beta * C (FP32 x FP32 -> FP32)
             *
             * Used for attention computation (Q @ K^T, scores @ V).
             */
            bool multiply_activations(const float *A, const float *B, float *C,
                                      int m, int n, int k,
                                      bool transpose_B = true,
                                      float alpha = 1.0f,
                                      float beta = 0.0f,
                                      const MPIContext *mpi_ctx = nullptr,
                                      int device_idx = -1) override
            {
                // Validate CPU-only execution
                if (device_idx != -1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Only CPU execution supported (device_idx="
                              << device_idx << ")");
                    return false;
                }

                if (!A || !B || !C)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Null activation buffer in multiply_activations");
                    return false;
                }

                // Check for unsupported parameters
                if (alpha != 1.0f || beta != 0.0f)
                {
                    LOG_ERROR("[OneDNNGemmKernel] alpha/beta scaling not yet supported");
                    return false;
                }

                // For activation-activation, use FP32 matmul
                const float *rhs_ptr = prepare_rhs_for_matmul(B, n, k, transpose_B);
                return run_onednn_fp32_matmul(A, rhs_ptr, C, m, n, k);
            }

            /**
             * @brief Strided activation GEMM (not implemented yet)
             */
            bool multiply_activations_strided(const float *A, const float *B, float *C,
                                              int m, int n, int k,
                                              int lda, int ldb, int ldc,
                                              bool transpose_B = true,
                                              float alpha = 1.0f,
                                              float beta = 0.0f,
                                              const MPIContext *mpi_ctx = nullptr,
                                              int device_idx = -1) override
            {
                LOG_ERROR("[OneDNNGemmKernel] multiply_activations_strided not yet implemented");
                return false;
            }

            bool multiply_activations_with_softmax(const float *A, const float *B, float *C,
                                                   int m, int n, int k,
                                                   bool transpose_B = true,
                                                   int softmax_axis = 1,
                                                   const MPIContext *mpi_ctx = nullptr,
                                                   int device_idx = -1) override
            {
                (void)mpi_ctx;

                if (device_idx != -1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Fused matmul+softmax only available on CPU");
                    return false;
                }

                if (!A || !B || !C)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Null activation buffer in fused matmul+softmax");
                    return false;
                }

                if (m <= 0 || n <= 0 || k <= 0)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Invalid dimensions for fused matmul+softmax");
                    return false;
                }

                int axis = softmax_axis;
                if (axis < 0)
                {
                    axis += 2; // 2D tensor
                }
                if (axis != 0 && axis != 1)
                {
                    LOG_ERROR("[OneDNNGemmKernel] Softmax axis must be 0, 1, or -1 (rows/cols)");
                    return false;
                }

                const float *rhs_ptr = prepare_rhs_for_matmul(B, n, k, transpose_B);
                if (!run_onednn_fp32_matmul_softmax(A, rhs_ptr, C, m, n, k, axis))
                {
                    LOG_WARN("[OneDNNGemmKernel] Falling back to unfused matmul + softmax path");
                    if (!run_onednn_fp32_matmul(A, rhs_ptr, C, m, n, k))
                    {
                        return false;
                    }
                    if (!apply_softmax_inplace(C, m, n, axis))
                    {
                        LOG_ERROR("[OneDNNGemmKernel] Softmax fallback failed");
                        return false;
                    }
                }

                return true;
            }

        private:
            const TensorBase *weight_tensor_; ///< Bound weight tensor (B matrix)
        };

    } // namespace gemm_v4
} // namespace llaminar2
