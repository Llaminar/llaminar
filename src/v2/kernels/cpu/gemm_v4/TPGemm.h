/**
 * @file TPGemm.h
 * @brief Tensor-Parallel GEMM operations for MPI-distributed inference
 *
 * Implements proper work distribution across MPI ranks:
 *
 * ## Column-Parallel GEMM (QKV, Gate, Up projections)
 * ```
 * Input: [m, k] (full, replicated on all ranks)
 * Weight: [n, k] (each rank has slice [n/world_size, k])
 * Output: [m, n/world_size] (local slice, no communication needed)
 * ```
 * Each rank computes C[:, start:end] = A @ W[start:end, :]^T
 *
 * ## Row-Parallel GEMM (Wo, Down projections)
 * ```
 * Input: [m, k/world_size] (local slice from previous column-parallel op)
 * Weight: [n, k] (each rank has slice [n, k/world_size])
 * Output: [m, n] (partial, requires allreduce_sum)
 * ```
 * Each rank computes partial C = A_local @ W[:, start:end]^T
 * Then allreduce_sum to get full C
 *
 * @author David Sanftenberg
 * @date 2025-12-03
 */

#pragma once

#include "../../../utils/MPIContext.h"
#include "../../../utils/Logger.h"
#include "../../../tensors/TensorKernels.h"
#include "QuantisedGemmKernel.h"
#include <vector>
#include <memory>

namespace llaminar2
{
    namespace gemm_v4
    {
        /**
         * @brief Tensor-parallel GEMM mode
         */
        enum class TPGemmMode
        {
            None,           ///< No parallelism (single rank or disabled)
            ColumnParallel, ///< Split output columns (QKV, Gate, Up)
            RowParallel     ///< Split input columns (Wo, Down) - requires allreduce
        };

        /**
         * @brief Tensor-Parallel GEMM wrapper
         *
         * Wraps a QuantisedGemmKernel and adds MPI work distribution.
         * The underlying kernel operates on the local slice only.
         */
        class TPGemmKernel
        {
        public:
            /**
             * @brief Create tensor-parallel GEMM kernel
             *
             * For column-parallel: weight should be the local slice [n_local, k]
             * For row-parallel: weight should be the local slice [n, k_local]
             *
             * @param weight Local weight tensor slice
             * @param mode Parallelism mode
             * @param full_n Full output dimension (before slicing)
             * @param full_k Full input dimension (before slicing)
             */
            TPGemmKernel(const TensorBase *weight, TPGemmMode mode,
                         int full_n, int full_k)
                : mode_(mode), full_n_(full_n), full_k_(full_k)
            {
                kernel_ = std::make_unique<QuantisedGemmKernel>(weight);
            }

            /**
             * @brief Execute tensor-parallel GEMM
             *
             * For ColumnParallel:
             *   - Input A: [m, k] (full)
             *   - Output C: [m, n_local] (local slice)
             *   - No communication needed
             *
             * For RowParallel:
             *   - Input A: [m, k_local] (local slice)
             *   - Output C: [m, n] (full, but partial values)
             *   - Requires allreduce_sum after to combine contributions
             *
             * @param A Input activations (full for column-parallel, slice for row-parallel)
             * @param C Output buffer
             * @param m Sequence length / batch size
             * @param mpi_ctx MPI context for allreduce (required for row-parallel)
             * @param bias Optional bias vector
             * @return true on success
             */
            bool execute(const float *A, float *C, int m,
                         const MPIContext *mpi_ctx,
                         const float *bias = nullptr)
            {
                if (!kernel_ || !A || !C)
                {
                    LOG_ERROR("[TPGemm] Null pointer");
                    return false;
                }

                // Get local dimensions from kernel
                int n_local = kernel_->packed_weights().N;
                int k_local = kernel_->packed_weights().K;

                switch (mode_)
                {
                case TPGemmMode::None:
                    // Standard GEMM, no parallelism
                    return kernel_->multiply(A, C, m, n_local, k_local,
                                             false, 1.0f, 0.0f, mpi_ctx, -1);

                case TPGemmMode::ColumnParallel:
                    // Each rank computes its output columns
                    // Input is full [m, k], output is local [m, n_local]
                    return execute_column_parallel(A, C, m, n_local, full_k_, bias);

                case TPGemmMode::RowParallel:
                    // Each rank computes partial result from its input slice
                    // Input is local [m, k_local], output is full [m, n]
                    return execute_row_parallel(A, C, m, full_n_, k_local, mpi_ctx, bias);

                default:
                    LOG_ERROR("[TPGemm] Unknown mode");
                    return false;
                }
            }

            /**
             * @brief Execute with pre-quantized activations (for fused multi-GEMM)
             */
            bool execute_with_precomputed_q8_1(
                const void *q8_1_activations,
                float *C, int m,
                const MPIContext *mpi_ctx,
                const float *bias = nullptr,
                const GemmFusedOps &fused_ops = GemmFusedOps::none())
            {
                if (!kernel_ || !q8_1_activations || !C)
                {
                    LOG_ERROR("[TPGemm] Null pointer");
                    return false;
                }

                int n_local = kernel_->packed_weights().N;
                int k_local = kernel_->packed_weights().K;

                switch (mode_)
                {
                case TPGemmMode::None:
                case TPGemmMode::ColumnParallel:
                    // Standard path - kernel handles everything
                    return kernel_->multiply_with_precomputed_q8_1(
                        q8_1_activations, C, m, n_local, k_local,
                        bias, false, 1.0f, 0.0f, mpi_ctx, -1, fused_ops);

                case TPGemmMode::RowParallel:
                    // Need allreduce after local GEMM
                    return execute_row_parallel_q8_1(
                        q8_1_activations, C, m, full_n_, k_local, mpi_ctx, bias, fused_ops);

                default:
                    LOG_ERROR("[TPGemm] Unknown mode");
                    return false;
                }
            }

            TPGemmMode mode() const { return mode_; }
            int full_n() const { return full_n_; }
            int full_k() const { return full_k_; }
            const QuantisedGemmKernel *kernel() const { return kernel_.get(); }

        private:
            bool execute_column_parallel(const float *A, float *C, int m,
                                         int n_local, int k, const float *bias)
            {
                // Column-parallel: A is [m, k], C is [m, n_local]
                // Kernel weights are [n_local, k]
                // No communication needed - each rank writes to disjoint output columns

                // Quantize activations (full input)
                size_t buffer_size = kernel_->get_quantized_activation_buffer_size(m, k);
                std::vector<uint8_t> q8_1_buffer(buffer_size);

                if (!kernel_->quantize_activations(A, q8_1_buffer.data(), m, k))
                {
                    LOG_ERROR("[TPGemm] Column-parallel: quantization failed");
                    return false;
                }

                // Execute GEMM for local output columns
                return kernel_->multiply_with_precomputed_q8_1(
                    q8_1_buffer.data(), C, m, n_local, k,
                    bias, false, 1.0f, 0.0f, nullptr, -1, GemmFusedOps::none());
            }

            bool execute_row_parallel(const float *A, float *C, int m,
                                      int n, int k_local,
                                      const MPIContext *mpi_ctx,
                                      const float *bias)
            {
                if (!mpi_ctx)
                {
                    LOG_ERROR("[TPGemm] Row-parallel requires MPI context");
                    return false;
                }

                // Row-parallel: A is [m, k_local], C is [m, n]
                // Kernel weights are [n, k_local]
                // Each rank computes partial contribution, then allreduce

                // Quantize local activation slice
                size_t buffer_size = kernel_->get_quantized_activation_buffer_size(m, k_local);
                std::vector<uint8_t> q8_1_buffer(buffer_size);

                if (!kernel_->quantize_activations(A, q8_1_buffer.data(), m, k_local))
                {
                    LOG_ERROR("[TPGemm] Row-parallel: quantization failed");
                    return false;
                }

                // Allocate local output buffer
                std::vector<float> local_C(static_cast<size_t>(m) * n, 0.0f);

                // Execute GEMM for partial result
                if (!kernel_->multiply_with_precomputed_q8_1(
                        q8_1_buffer.data(), local_C.data(), m, n, k_local,
                        nullptr, // Bias added after allreduce
                        false, 1.0f, 0.0f, nullptr, -1, GemmFusedOps::none()))
                {
                    LOG_ERROR("[TPGemm] Row-parallel: GEMM failed");
                    return false;
                }

                // Allreduce to sum contributions from all ranks
                mpi_ctx->allreduce_sum(local_C.data(), C, static_cast<size_t>(m) * n);

                // Add bias after allreduce (only once, not per-rank)
                if (bias)
                {
#pragma omp parallel for
                    for (int i = 0; i < m; ++i)
                    {
                        float *row = C + i * n;
                        for (int j = 0; j < n; ++j)
                        {
                            row[j] += bias[j];
                        }
                    }
                }

                return true;
            }

            bool execute_row_parallel_q8_1(
                const void *q8_1_activations, float *C, int m,
                int n, int k_local,
                const MPIContext *mpi_ctx,
                const float *bias,
                const GemmFusedOps &fused_ops)
            {
                if (!mpi_ctx)
                {
                    LOG_ERROR("[TPGemm] Row-parallel requires MPI context");
                    return false;
                }

                // Allocate local output buffer
                std::vector<float> local_C(static_cast<size_t>(m) * n, 0.0f);

                // Execute GEMM for partial result (no fused ops until after allreduce)
                if (!kernel_->multiply_with_precomputed_q8_1(
                        q8_1_activations, local_C.data(), m, n, k_local,
                        nullptr, // Bias after allreduce
                        false, 1.0f, 0.0f, nullptr, -1, GemmFusedOps::none()))
                {
                    LOG_ERROR("[TPGemm] Row-parallel: GEMM failed");
                    return false;
                }

                // Allreduce to sum contributions
                mpi_ctx->allreduce_sum(local_C.data(), C, static_cast<size_t>(m) * n);

                // Add bias after allreduce
                if (bias)
                {
#pragma omp parallel for
                    for (int i = 0; i < m; ++i)
                    {
                        float *row = C + i * n;
                        for (int j = 0; j < n; ++j)
                        {
                            row[j] += bias[j];
                        }
                    }
                }

                // Apply fused ops after allreduce (if any)
                // SwiGLU should be applied to combined output
                if (fused_ops.is_swiglu() && fused_ops.gate_input)
                {
                    // SwiGLU: output *= sigmoid(gate) * gate (swish(gate))
                    const float *gate = fused_ops.gate_input;
#pragma omp parallel for
                    for (int i = 0; i < m; ++i)
                    {
                        float *out_row = C + i * n;
                        const float *gate_row = gate + i * n;
                        for (int j = 0; j < n; ++j)
                        {
                            float g = gate_row[j];
                            float swish = g / (1.0f + std::exp(-g)); // swish(x) = x * sigmoid(x)
                            out_row[j] *= swish;
                        }
                    }
                }

                return true;
            }

            std::unique_ptr<QuantisedGemmKernel> kernel_;
            TPGemmMode mode_;
            int full_n_; ///< Full output dimension (before distribution)
            int full_k_; ///< Full input dimension (before distribution)
        };

        /**
         * @brief Tensor-Parallel Fused GEMM for multi-projection operations
         *
         * Handles QKV and Gate/Up projections with proper MPI distribution.
         * Column-parallel: each rank computes slice of outputs.
         */
        class TPFusedGEMM
        {
        public:
            /**
             * @brief Create tensor-parallel fused GEMM
             *
             * @param weights Vector of local weight slices (one per projection)
             * @param mode ColumnParallel for QKV/Gate/Up, RowParallel not supported here
             * @param full_output_dims Full output dimensions per projection
             * @param full_k Full input dimension
             */
            TPFusedGEMM(const std::vector<const TensorBase *> &weights,
                        TPGemmMode mode,
                        const std::vector<int> &full_output_dims,
                        int full_k)
                : mode_(mode), full_output_dims_(full_output_dims), full_k_(full_k)
            {
                for (size_t i = 0; i < weights.size(); ++i)
                {
                    int full_n = (i < full_output_dims.size()) ? full_output_dims[i] : 0;
                    kernels_.push_back(std::make_unique<TPGemmKernel>(
                        weights[i], mode, full_n, full_k));
                }
            }

            /**
             * @brief Execute fused multi-GEMM with column parallelism
             *
             * Quantizes activations once, executes all projections.
             * For column-parallel: each output buffer is [m, n_local].
             *
             * @param input Input activations [m, k]
             * @param outputs Output buffers (one per projection)
             * @param biases Optional bias vectors (nullptr entries OK)
             * @param m Sequence length
             * @param k Input dimension
             * @param mpi_ctx MPI context
             * @return true on success
             */
            bool execute(const float *input,
                         const std::vector<float *> &outputs,
                         const std::vector<const float *> &biases,
                         int m, int k,
                         const MPIContext *mpi_ctx)
            {
                if (outputs.size() != kernels_.size())
                {
                    LOG_ERROR("[TPFusedGEMM] Output count mismatch");
                    return false;
                }

                // Quantize activations once
                if (kernels_.empty())
                    return false;

                size_t buffer_size = kernels_[0]->kernel()->get_quantized_activation_buffer_size(m, k);
                std::vector<uint8_t> q8_1_buffer(buffer_size);

                if (!kernels_[0]->kernel()->quantize_activations(input, q8_1_buffer.data(), m, k))
                {
                    LOG_ERROR("[TPFusedGEMM] Quantization failed");
                    return false;
                }

                // Execute all projections with shared quantized activations
                for (size_t i = 0; i < kernels_.size(); ++i)
                {
                    const float *bias = (i < biases.size()) ? biases[i] : nullptr;
                    if (!kernels_[i]->execute_with_precomputed_q8_1(
                            q8_1_buffer.data(), outputs[i], m, mpi_ctx, bias))
                    {
                        LOG_ERROR("[TPFusedGEMM] Projection " << i << " failed");
                        return false;
                    }
                }

                return true;
            }

            size_t num_projections() const { return kernels_.size(); }
            TPGemmMode mode() const { return mode_; }

        private:
            std::vector<std::unique_ptr<TPGemmKernel>> kernels_;
            TPGemmMode mode_;
            std::vector<int> full_output_dims_;
            int full_k_;
        };

    } // namespace gemm_v4
} // namespace llaminar2
