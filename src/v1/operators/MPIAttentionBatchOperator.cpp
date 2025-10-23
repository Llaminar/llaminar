/**
 * @file MPIAttentionBatchOperator.cpp
 * @brief Implementation of batch-aware multi-head attention
 * @author David Sanftenberg
 * @date 2025-10-16
 */

#include "MPIAttentionBatchOperator.h"
#include "MPILinearBatchOperator.h"
#include "Logger.h"
#include "BiasContracts.h"
#include "common/TensorHealthCheck.h"
#include "common/AttentionPrimitives.h"
#include "common/SoftmaxCore.h"
#include "attention/AttentionStageContracts.h"
#include "utils/DebugEnv.h"
#include "AdaptiveMatmul.h"
#include <cblas.h>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace llaminar
{

    MPIAttentionBatchOperator::MPIAttentionBatchOperator(
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_freq_base)
        : MPIOperatorBase(), n_heads_(n_heads), n_kv_heads_(n_kv_heads), head_dim_(head_dim), rope_freq_base_(rope_freq_base)
    {
        // Distribute heads across MPI ranks
        int rank = getRank();
        int size = getSize();

        // Simple distribution: divide heads evenly
        n_heads_local_ = n_heads_ / size;
        int remainder = n_heads_ % size;
        if (rank < remainder)
        {
            n_heads_local_++;
            head_offset_ = rank * n_heads_local_;
        }
        else
        {
            head_offset_ = remainder * (n_heads_local_ + 1) + (rank - remainder) * n_heads_local_;
        }

        // For GQA, distribute KV heads similarly
        n_kv_heads_local_ = n_kv_heads_ / size;
        int kv_remainder = n_kv_heads_ % size;
        if (rank < kv_remainder)
        {
            n_kv_heads_local_++;
        }

        if (rank == 0)
        {
            LOG_DEBUG("[MPIAttentionBatchOperator] Initialized: n_heads=" << n_heads_
                                                                          << " n_kv_heads=" << n_kv_heads_ << " head_dim=" << head_dim_
                                                                          << " (local: " << n_heads_local_ << " heads, offset=" << head_offset_ << ")");
        }
    }

    std::pair<int, int> MPIAttentionBatchOperator::getKVHeadDistribution() const
    {
        return getKVHeadDistribution(getRank());
    }

    std::pair<int, int> MPIAttentionBatchOperator::getKVHeadDistribution(int rank) const
    {
        int heads_per_rank = n_kv_heads_ / getSize();
        int remainder = n_kv_heads_ % getSize();

        int local_heads = heads_per_rank + (rank < remainder ? 1 : 0);
        int head_offset = rank * heads_per_rank + std::min(rank, remainder);

        return {local_heads, head_offset};
    }

    bool MPIAttentionBatchOperator::validate(
        const std::vector<std::shared_ptr<TensorBase>> &inputs,
        const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 10)
        {
            LOG_ERROR("MPIAttentionBatchOperator: Expected 10 inputs, got " << inputs.size());
            return false;
        }

        if (outputs.size() != 1)
        {
            LOG_ERROR("MPIAttentionBatchOperator: Expected 1 output, got " << outputs.size());
            return false;
        }

        // Validate input shape [B, T, D]
        const auto &input_shape = inputs[0]->shape();
        if (input_shape.size() != 3)
        {
            LOG_ERROR("MPIAttentionBatchOperator: Input must be 3D [B, T, D], got "
                      << input_shape.size() << "D");
            return false;
        }

        int batch_size = input_shape[0];
        int seq_len = input_shape[1];
        int d_model = input_shape[2];

        // Validate output is pre-allocated with correct shape
        if (!outputs[0])
        {
            LOG_ERROR("MPIAttentionBatchOperator: Output tensor is null");
            return false;
        }

        const auto &output_shape = outputs[0]->shape();
        if (output_shape.size() != 3 ||
            output_shape[0] != batch_size ||
            output_shape[1] != seq_len ||
            output_shape[2] != d_model)
        {
            LOG_ERROR("MPIAttentionBatchOperator: Output shape mismatch. Expected ["
                      << batch_size << "," << seq_len << "," << d_model << "], got ["
                      << output_shape[0] << "," << output_shape[1] << "," << output_shape[2] << "]");
            return false;
        }

        return true;
    }

    bool MPIAttentionBatchOperator::execute(
        const std::vector<std::shared_ptr<TensorBase>> &inputs,
        std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            return false;
        }

        const auto &input = inputs[0];
        const auto &wq = inputs[1];
        const auto &wk = inputs[2];
        const auto &wv = inputs[3];
        const auto &wo = inputs[4];
        const auto &bq = inputs[5];
        const auto &bk = inputs[6];
        const auto &bv = inputs[7];
        // KV cache inputs[8], inputs[9] - not yet implemented

        auto &output = outputs[0];

        const auto &input_shape = input->shape();
        int B = input_shape[0]; // batch size
        int T = input_shape[1]; // sequence length
        int D = input_shape[2]; // d_model

        int rank = getRank();
        int world_size = getSize();

        // ========================================================================
        // Weight validation - detect whether weights are pre-sliced or replicated
        // ========================================================================

        // Expected dimensions for REPLICATED weights (full matrices)
        const int expected_wq_rows_full = n_heads_ * head_dim_;    // Full Q projection output
        const int expected_wk_rows_full = n_kv_heads_ * head_dim_; // Full K projection output
        const int expected_wv_rows_full = n_kv_heads_ * head_dim_; // Full V projection output
        const int expected_wo_cols_full = n_heads_ * head_dim_;    // Full output projection input

        // Expected dimensions for PRE-SLICED weights (distributed across ranks)
        const int expected_wq_rows_sliced = (n_heads_ / world_size) * head_dim_;    // Q slice: 7 heads per rank = 448
        const int expected_wk_rows_sliced = (n_kv_heads_ / world_size) * head_dim_; // K slice: 1 head per rank = 64
        const int expected_wv_rows_sliced = (n_kv_heads_ / world_size) * head_dim_; // V slice: 1 head per rank = 64
        const int expected_wo_cols_sliced = (n_heads_ / world_size) * head_dim_;    // O slice: 7 heads per rank = 448

        const int wq_rows = static_cast<int>(wq->shape()[0]);
        const int wq_cols = static_cast<int>(wq->shape()[1]);
        const int wk_rows = static_cast<int>(wk->shape()[0]);
        const int wk_cols = static_cast<int>(wk->shape()[1]);
        const int wv_rows = static_cast<int>(wv->shape()[0]);
        const int wv_cols = static_cast<int>(wv->shape()[1]);
        const int wo_rows = static_cast<int>(wo->shape()[0]);
        const int wo_cols = static_cast<int>(wo->shape()[1]);

        // Detect weight distribution mode
        bool wq_is_sliced = (wq_rows == expected_wq_rows_sliced);
        bool wq_is_full = (wq_rows == expected_wq_rows_full);
        bool wk_is_sliced = (wk_rows == expected_wk_rows_sliced);
        bool wk_is_full = (wk_rows == expected_wk_rows_full);
        bool wv_is_sliced = (wv_rows == expected_wv_rows_sliced);
        bool wv_is_full = (wv_rows == expected_wv_rows_full);
        bool wo_is_sliced = (wo_cols == expected_wo_cols_sliced);
        bool wo_is_full = (wo_cols == expected_wo_cols_full);

        // All weights must use same mode (either all sliced or all full)
        bool all_sliced = (wq_is_sliced && wk_is_sliced && wv_is_sliced && wo_is_sliced);
        bool all_full = (wq_is_full && wk_is_full && wv_is_full && wo_is_full);

        if (!all_sliced && !all_full)
        {
            LOG_ERROR("[MPIAttentionBatch] Inconsistent weight distribution!");
            LOG_ERROR("  wq: " << (wq_is_sliced ? "SLICED" : (wq_is_full ? "FULL" : "INVALID")) << " [" << wq_rows << "," << wq_cols << "]");
            LOG_ERROR("  wk: " << (wk_is_sliced ? "SLICED" : (wk_is_full ? "FULL" : "INVALID")) << " [" << wk_rows << "," << wk_cols << "]");
            LOG_ERROR("  wv: " << (wv_is_sliced ? "SLICED" : (wv_is_full ? "FULL" : "INVALID")) << " [" << wv_rows << "," << wv_cols << "]");
            LOG_ERROR("  wo: " << (wo_is_sliced ? "SLICED" : (wo_is_full ? "FULL" : "INVALID")) << " [" << wo_rows << "," << wo_cols << "]");
            return false;
        }

        bool using_pre_sliced_weights = all_sliced;

        if (rank == 0)
        {
            LOG_DEBUG("[MPIAttentionBatch] Weight distribution mode: " << (using_pre_sliced_weights ? "PRE-SLICED" : "REPLICATED"));
        }

        // Validate input/output dimensions (same for both modes)
        if (wq_cols != D || wk_cols != D || wv_cols != D || wo_rows != D)
        {
            LOG_ERROR("[MPIAttentionBatch] Weight column/row dimension mismatch with D=" << D);
            return false;
        }

        // Health check on weights (detect uninitialized/corrupted data)
        if (debugEnv().attention.validate_output && rank == 0)
        {
            TensorHealthCheck health_checks[] = {
                TensorHealthCheck("input"),
                TensorHealthCheck("wq_global"),
                TensorHealthCheck("wk_global"),
                TensorHealthCheck("wv_global"),
                TensorHealthCheck("wo_global")};
            const float *data_ptrs[] = {
                input->data(), wq->data(), wk->data(), wv->data(), wo->data()};
            size_t sizes[] = {
                static_cast<size_t>(input->size()),
                static_cast<size_t>(wq->size()),
                static_cast<size_t>(wk->size()),
                static_cast<size_t>(wv->size()),
                static_cast<size_t>(wo->size())};

            bool all_healthy = true;
            for (int i = 0; i < 5; ++i)
            {
                health_checks[i].check(data_ptrs[i], sizes[i]);
                health_checks[i].log(rank);
                if (!health_checks[i].is_healthy())
                {
                    all_healthy = false;
                }
            }

            if (!all_healthy)
            {
                LOG_ERROR("[MPIAttentionBatch] Input tensors contain NaN/Inf - aborting");
                return false;
            }
        }

        if (rank == 0)
        {
            LOG_DEBUG("[MPIAttentionBatchOperator] Processing: B=" << B << " T=" << T << " D=" << D
                                                                   << " (n_heads_local=" << n_heads_local_ << ")");
            LOG_DEBUG("[MPIAttentionBatch] Weight shapes validated: wq=[" << wq_rows << "," << wq_cols
                                                                          << "] wk=[" << wk_rows << "," << wk_cols << "] wv=[" << wv_rows << "," << wv_cols
                                                                          << "] wo=[" << wo_rows << "," << wo_cols << "]");
        }

        // ========================================================================
        // Step 1: Q, K, V projections using direct adaptiveMatMul
        // ========================================================================
        auto t_qkv_start = std::chrono::high_resolution_clock::now();

        // CRITICAL: Heads are ALREADY distributed across MPI ranks in this operator.
        // We must NOT use MPILinearBatchOperator which would double-distribute.
        // Each rank computes its local head subset WITHOUT further MPI partitioning.

        // Allocate local Q, K, V tensors
        // Q: [B, T, n_heads_local * head_dim]
        auto q_local = std::make_shared<SimpleTensor>(std::vector<int>{B, T, n_heads_local_ * head_dim_});
        // K, V: [B, T, n_kv_heads_local * head_dim]
        auto k_local = std::make_shared<SimpleTensor>(std::vector<int>{B, T, n_kv_heads_local_ * head_dim_});
        auto v_local = std::make_shared<SimpleTensor>(std::vector<int>{B, T, n_kv_heads_local_ * head_dim_});

        // Q projection (local computation, no MPI distribution)
        {
            // Handle pre-sliced or replicated weights
            int local_rows = n_heads_local_ * head_dim_;
            int row_offset = head_offset_ * head_dim_;

            std::shared_ptr<TensorBase> wq_local;

            if (using_pre_sliced_weights)
            {
                // Weights already sliced - use directly
                wq_local = wq;

                if (getRank() == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[BATCH_Q_PROJ] Layer " << current_layer_idx_ << " Rank " << getRank()
                                                      << " Using PRE-SLICED wq: [" << wq->shape()[0] << ", " << wq->shape()[1] << "]");
                }
            }
            else
            {
                // Replicated weights - extract local rows for this rank's heads
                if (getRank() == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[BATCH_Q_PROJ] Layer " << current_layer_idx_ << " Rank " << getRank()
                                                      << " Extracting from REPLICATED wq:");
                    LOG_DEBUG("  n_heads=" << n_heads_ << " n_heads_local=" << n_heads_local_
                                           << " head_dim=" << head_dim_ << " head_offset=" << head_offset_);
                    LOG_DEBUG("  local_rows=" << local_rows << " row_offset=" << row_offset << " D=" << D);
                    LOG_DEBUG("  Global wq shape: [" << wq->shape()[0] << ", " << wq->shape()[1] << "]");
                }

                wq_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows, D});

                // Extract rows [row_offset : row_offset + local_rows]
                size_t offset_elements = static_cast<size_t>(row_offset) * D;
                size_t copy_elements = static_cast<size_t>(local_rows) * D;
                std::memcpy(wq_local->data(), wq->data() + offset_elements, copy_elements * sizeof(float));
            }

            // Extract local bias if present
            std::shared_ptr<TensorBase> bq_local;
            if (bq && bq->size() > 0)
            {
                if (using_pre_sliced_weights)
                {
                    // Bias already sliced - use directly
                    bq_local = bq;
                }
                else
                {
                    // Replicated bias - extract local portion
                    bq_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows});
                    const float *bq_data = bq->data();
                    float *bq_local_data = bq_local->data();
                    std::copy(bq_data + row_offset, bq_data + row_offset + local_rows, bq_local_data);
                }
            }

            if (getRank() == 0 && current_layer_idx_ == 0)
            {
                LOG_DEBUG("[BATCH_Q_PROJ] Local wq_local first 5: [" << wq_local->data()[0] << ", "
                                                                     << wq_local->data()[1] << ", " << wq_local->data()[2] << ", "
                                                                     << wq_local->data()[3] << ", " << wq_local->data()[4] << "]");
                LOG_DEBUG("[BATCH_Q_PROJ] Input first 5: [" << input->data()[0] << ", " << input->data()[1]
                                                            << ", " << input->data()[2] << ", " << input->data()[3] << ", " << input->data()[4] << "]");
            }

            // CONTRACT: Q projection must compute FULL local head dimensions without MPI re-distribution
            // Input: [B, T, D] where D=d_model=896
            // Weight: [local_rows, D] where local_rows = n_heads_local * head_dim
            // Output: [B, T, local_rows]
            // This is a LOCAL-ONLY matmul - no MPI distribution!
            int m = B * T;
            int n = local_rows;
            int k = D;

            if (getRank() == 0)
            {
                LOG_DEBUG("[FIX_Q_PROJ] About to call adaptiveMatMul: m=" << m << " n=" << n << " k=" << k);
            }

            // Flatten input: [B, T, D] -> [B*T, D]
            const float *input_data = input->data();
            const float *weight_data = wq_local->data();
            float *output_data = q_local->data();

            // Direct matmul: output[m, n] = input[m, k] @ weight[n, k]^T
            // Using distributed_partition=false to prevent MPI re-distribution
            bool success = adaptiveMatMul(input_data, weight_data, output_data,
                                          m, n, k,
                                          /*is_prefill=*/false,
                                          /*distributed_partition=*/false, // CRITICAL: No MPI distribution!
                                          /*transpose_A=*/false,
                                          /*transpose_B=*/true, // Weight is [n, k], we need [k, n]^T
                                          /*alpha=*/1.0f,
                                          /*beta=*/0.0f);

            if (!success)
            {
                LOG_ERROR("MPIAttentionBatchOperator: Q projection matmul failed");
                return false;
            }

            // CONTRACT VALIDATION: Ensure no double-distribution bug
            validateBatchAttentionProjection(
                B, T, D, n_heads_local_, head_dim_,
                input, wq_local, q_local, "Q");

            if (getRank() == 0 && current_layer_idx_ == 0)
            {
                LOG_DEBUG("[BATCH_Q_PROJ] Q output (local) first 10: [" << q_local->data()[0] << ", "
                                                                        << q_local->data()[1] << ", " << q_local->data()[2] << ", " << q_local->data()[3] << ", "
                                                                        << q_local->data()[4] << ", " << q_local->data()[5] << ", " << q_local->data()[6] << ", "
                                                                        << q_local->data()[7] << ", " << q_local->data()[8] << ", " << q_local->data()[9] << "]");
                float q_min = *std::min_element(q_local->data(), q_local->data() + q_local->size());
                float q_max = *std::max_element(q_local->data(), q_local->data() + q_local->size());
                LOG_DEBUG("[BATCH_Q_PROJ] Q output stats: min=" << q_min << " max=" << q_max
                                                                << " size=" << q_local->size());
            }

            // Apply bias if present (linear operator doesn't support bias yet)
            if (bq_local && bq_local->size() > 0)
            {
                float *q_data = q_local->data();
                const float *bias_data = bq_local->data();
                int total_positions = B * T;

#pragma omp parallel for
                for (int pos = 0; pos < total_positions; ++pos)
                {
                    for (int i = 0; i < local_rows; ++i)
                    {
                        q_data[pos * local_rows + i] += bias_data[i];
                    }
                }
            }
        }

        // K projection (similar to Q but for KV heads)
        {
            int local_rows = n_kv_heads_local_ * head_dim_;
            int row_offset = (head_offset_ * n_kv_heads_ / n_heads_) * head_dim_; // Scale offset for GQA

            std::shared_ptr<TensorBase> wk_local;

            if (using_pre_sliced_weights)
            {
                // Weights already sliced - use directly
                wk_local = wk;

                if (debugEnv().attention.verbose && rank_ == 0)
                {
                    LOG_DEBUG("[MPIAttentionBatch] Using PRE-SLICED wk: [" << wk->shape()[0] << "," << wk->shape()[1] << "]");
                }
            }
            else
            {
                // Replicated weights - extract local rows
                if (debugEnv().attention.verbose && rank_ == 0)
                {
                    LOG_DEBUG("[MPIAttentionBatch] K weight extraction from REPLICATED: wk->size()=" << wk->size()
                                                                                                     << " wk->shape()=[" << wk->shape()[0] << "," << wk->shape()[1] << "]"
                                                                                                     << " row_offset=" << row_offset << " local_rows=" << local_rows);
                }

                wk_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows, D});

                // Extract rows - simple memcpy of contiguous data
                size_t offset_elements = static_cast<size_t>(row_offset) * D;
                size_t copy_elements = static_cast<size_t>(local_rows) * D;
                std::memcpy(wk_local->data(), wk->data() + offset_elements, copy_elements * sizeof(float));
            }

            std::shared_ptr<TensorBase> bk_local;
            if (bk && bk->size() > 0)
            {
                if (using_pre_sliced_weights)
                {
                    // Bias already sliced - use directly
                    bk_local = bk;
                }
                else
                {
                    // Replicated bias - extract local portion
                    bk_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows});
                    const float *bk_data = bk->data();
                    float *bk_local_data = bk_local->data();
                    std::copy(bk_data + row_offset, bk_data + row_offset + local_rows, bk_local_data);
                }
            }

            // Direct K projection (no MPI distribution)
            int m_k = B * T;
            int n_k = local_rows;
            int k_k = D;

            const float *input_data = input->data();
            const float *weight_data_k = wk_local->data();
            float *output_data_k = k_local->data();

            bool success_k = adaptiveMatMul(input_data, weight_data_k, output_data_k,
                                            m_k, n_k, k_k,
                                            /*is_prefill=*/false,
                                            /*distributed_partition=*/false, // No MPI distribution!
                                            /*transpose_A=*/false,
                                            /*transpose_B=*/true,
                                            /*alpha=*/1.0f,
                                            /*beta=*/0.0f);

            if (!success_k)
            {
                LOG_ERROR("MPIAttentionBatchOperator: K projection matmul failed");
                return false;
            }

            // CONTRACT VALIDATION: Ensure no double-distribution bug
            int local_rows_k = n_kv_heads_local_ * head_dim_;
            validateBatchAttentionProjection(
                B, T, D, n_kv_heads_local_, head_dim_,
                input, wk_local, k_local, "K");

            if (bk_local && bk_local->size() > 0)
            {
                float *k_data = k_local->data();
                const float *bias_data = bk_local->data();
                int total_positions = B * T;

#pragma omp parallel for
                for (int pos = 0; pos < total_positions; ++pos)
                {
                    for (int i = 0; i < local_rows; ++i)
                    {
                        k_data[pos * local_rows + i] += bias_data[i];
                    }
                }
            }
        }

        // V projection
        {
            int local_rows = n_kv_heads_local_ * head_dim_;
            int row_offset = (head_offset_ * n_kv_heads_ / n_heads_) * head_dim_;

            std::shared_ptr<TensorBase> wv_local;

            if (using_pre_sliced_weights)
            {
                // Weights already sliced - use directly
                wv_local = wv;
            }
            else
            {
                // Replicated weights - extract local rows
                wv_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows, D});

                // Extract rows - simple memcpy of contiguous data
                size_t offset_elements = static_cast<size_t>(row_offset) * D;
                size_t copy_elements = static_cast<size_t>(local_rows) * D;
                std::memcpy(wv_local->data(), wv->data() + offset_elements, copy_elements * sizeof(float));
            }

            std::shared_ptr<TensorBase> bv_local;
            if (bv && bv->size() > 0)
            {
                if (using_pre_sliced_weights)
                {
                    // Bias already sliced - use directly
                    bv_local = bv;
                }
                else
                {
                    // Replicated bias - extract local portion
                    bv_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows});
                    const float *bv_data = bv->data();
                    float *bv_local_data = bv_local->data();
                    std::copy(bv_data + row_offset, bv_data + row_offset + local_rows, bv_local_data);
                }
            }

            // Direct V projection (no MPI distribution)
            int m_v = B * T;
            int n_v = local_rows;
            int k_v = D;

            const float *input_data_v = input->data();
            const float *weight_data_v = wv_local->data();
            float *output_data_v = v_local->data();

            bool success_v = adaptiveMatMul(input_data_v, weight_data_v, output_data_v,
                                            m_v, n_v, k_v,
                                            /*is_prefill=*/false,
                                            /*distributed_partition=*/false, // No MPI distribution!
                                            /*transpose_A=*/false,
                                            /*transpose_B=*/true,
                                            /*alpha=*/1.0f,
                                            /*beta=*/0.0f);

            if (!success_v)
            {
                LOG_ERROR("MPIAttentionBatchOperator: V projection matmul failed");
                return false;
            }

            // CONTRACT VALIDATION: Ensure no double-distribution bug
            int local_rows_v = n_kv_heads_local_ * head_dim_;
            validateBatchAttentionProjection(
                B, T, D, n_kv_heads_local_, head_dim_,
                input, wv_local, v_local, "V");

            if (bv_local && bv_local->size() > 0)
            {
                float *v_data = v_local->data();
                const float *bias_data = bv_local->data();
                int total_positions = B * T;

#pragma omp parallel for
                for (int pos = 0; pos < total_positions; ++pos)
                {
                    for (int i = 0; i < local_rows; ++i)
                    {
                        v_data[pos * local_rows + i] += bias_data[i];
                    }
                }
            }
        }

        // DEBUG: Check Q/K/V after projections
        if (rank == 0 && debugEnv().attention.verbose)
        {
            auto check_tensor = [](const float *data, size_t size, const char *name)
            {
                float min_val = data[0], max_val = data[0];
                for (size_t i = 1; i < size; ++i)
                {
                    min_val = std::min(min_val, data[i]);
                    max_val = std::max(max_val, data[i]);
                }
                LOG_DEBUG("[MPIAttentionBatch] After projection " << name << ": min=" << min_val << " max=" << max_val);
            };
            check_tensor(q_local->data(), q_local->size(), "Q");
            check_tensor(k_local->data(), k_local->size(), "K");
            check_tensor(v_local->data(), v_local->size(), "V");
        }

        // Capture Q/K/V projections for parity testing
        // For consistency with sequential pipeline, gather to global tensors
        if (snapshot_callback_)
        {
            int rank = getRank();
            int size = getSize();

            // DEBUG: Log local Q values before gathering
            if (rank == 0 && current_layer_idx_ == 0)
            {
                LOG_DEBUG("[Q_LOCAL_DEBUG] Before gather - Local Q first 10: ["
                          << q_local->data()[0] << ", " << q_local->data()[1] << ", "
                          << q_local->data()[2] << ", " << q_local->data()[3] << ", "
                          << q_local->data()[4] << ", " << q_local->data()[5] << ", "
                          << q_local->data()[6] << ", " << q_local->data()[7] << ", "
                          << q_local->data()[8] << ", " << q_local->data()[9] << "]");
                float q_min = *std::min_element(q_local->data(), q_local->data() + q_local->size());
                float q_max = *std::max_element(q_local->data(), q_local->data() + q_local->size());
                LOG_DEBUG("[Q_LOCAL_DEBUG] Local Q stats: size=" << q_local->size()
                                                                 << " min=" << q_min << " max=" << q_max);
            }

            // Gather Q: local [B, T, n_heads_local * head_dim] -> global [B, T, n_heads * head_dim]
            auto q_snapshot = std::make_shared<SimpleTensor>(std::vector<int>{B, T, n_heads_ * head_dim_});
            int q_local_size = B * T * n_heads_local_ * head_dim_;

            if (size > 1)
            {
                // Use same gather pattern as MPIAttentionOperator for consistency
                // Gather into temporary buffer, then rearrange from rank-major to row-interleaved
                auto temp_q = std::make_shared<SimpleTensor>(std::vector<int>{B * T * n_heads_ * head_dim_});

                // DEBUG: Log gather parameters
                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[BATCH_GATHER_DEBUG] Q gather parameters:");
                    LOG_DEBUG("  Using MPI_Allgather (matching sequential path)");
                    LOG_DEBUG("  sendcount=" << q_local_size << " per rank");
                    LOG_DEBUG("  Local Q before gather [0:5]: [" << q_local->data()[0] << ", "
                                                                 << q_local->data()[1] << ", " << q_local->data()[2] << ", "
                                                                 << q_local->data()[3] << ", " << q_local->data()[4] << "]");
                }

                // Bulk gather: faster than per-token MPI calls
                MPI_Allgather(q_local->data(), q_local_size, MPI_FLOAT,
                              temp_q->data(), q_local_size, MPI_FLOAT,
                              MPI_COMM_WORLD);

                // Rearrange from rank-major to row-interleaved
                // temp_q layout: [rank0: b0_t0,b0_t1,...,b1_t0,b1_t1,... | rank1: b0_t0,b0_t1,...,b1_t0,b1_t1,... | ...]
                // q_snapshot layout: [b0_t0: r0,r1,... | b0_t1: r0,r1,... | b1_t0: r0,r1,... | ...]
                for (int b = 0; b < B; ++b)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        for (int r = 0; r < size; ++r)
                        {
                            const float *src = temp_q->data() + r * q_local_size + (b * T + t) * n_heads_local_ * head_dim_;
                            float *dst = q_snapshot->data() + (b * T + t) * n_heads_ * head_dim_ + r * n_heads_local_ * head_dim_;
                            std::memcpy(dst, src, n_heads_local_ * head_dim_ * sizeof(float));
                        }
                    }
                }

                // DEBUG: Log gathered result
                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[BATCH_GATHER_DEBUG] After gather and rearrange:");
                    LOG_DEBUG("  Total size: " << q_snapshot->size());
                    LOG_DEBUG("  First 10: [" << q_snapshot->data()[0] << ", " << q_snapshot->data()[1] << ", "
                                              << q_snapshot->data()[2] << ", " << q_snapshot->data()[3] << ", "
                                              << q_snapshot->data()[4] << ", " << q_snapshot->data()[5] << ", "
                                              << q_snapshot->data()[6] << ", " << q_snapshot->data()[7] << ", "
                                              << q_snapshot->data()[8] << ", " << q_snapshot->data()[9] << "]");
                    LOG_DEBUG("  At offset 1792 (rank1 start): [" << q_snapshot->data()[1792] << ", "
                                                                  << q_snapshot->data()[1793] << ", " << q_snapshot->data()[1794] << ", "
                                                                  << q_snapshot->data()[1795] << ", " << q_snapshot->data()[1796] << "]");
                }
            }
            else
            {
                // Single rank: just copy
                std::copy(q_local->data(), q_local->data() + q_local_size, q_snapshot->data());
            }
            snapshot_callback_(PipelineStage::Q_PROJECTION, current_layer_idx_, q_snapshot);

            // Gather K: local [B, T, n_kv_heads_local * head_dim] -> global [B, T, n_kv_heads * head_dim]
            auto k_snapshot = std::make_shared<SimpleTensor>(std::vector<int>{B, T, n_kv_heads_ * head_dim_});
            int k_local_size = B * T * n_kv_heads_local_ * head_dim_;

            if (size > 1)
            {
                // Same pattern as Q: gather into temp, then rearrange
                auto temp_k = std::make_shared<SimpleTensor>(std::vector<int>{B * T * n_kv_heads_ * head_dim_});
                MPI_Allgather(k_local->data(), k_local_size, MPI_FLOAT,
                              temp_k->data(), k_local_size, MPI_FLOAT,
                              MPI_COMM_WORLD);

                // Rearrange from rank-major to row-interleaved (accounting for batch dimension)
                for (int b = 0; b < B; ++b)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        for (int r = 0; r < size; ++r)
                        {
                            const float *src = temp_k->data() + r * k_local_size + (b * T + t) * n_kv_heads_local_ * head_dim_;
                            float *dst = k_snapshot->data() + (b * T + t) * n_kv_heads_ * head_dim_ + r * n_kv_heads_local_ * head_dim_;
                            std::memcpy(dst, src, n_kv_heads_local_ * head_dim_ * sizeof(float));
                        }
                    }
                }
            }
            else
            {
                std::copy(k_local->data(), k_local->data() + k_local_size, k_snapshot->data());
            }
            snapshot_callback_(PipelineStage::K_PROJECTION, current_layer_idx_, k_snapshot);

            // Gather V: local [B, T, n_kv_heads_local * head_dim] -> global [B, T, n_kv_heads * head_dim]
            auto v_snapshot = std::make_shared<SimpleTensor>(std::vector<int>{B, T, n_kv_heads_ * head_dim_});
            int v_local_size = B * T * n_kv_heads_local_ * head_dim_;

            if (size > 1)
            {
                // Same pattern as Q and K: gather into temp, then rearrange
                auto temp_v = std::make_shared<SimpleTensor>(std::vector<int>{B * T * n_kv_heads_ * head_dim_});
                MPI_Allgather(v_local->data(), v_local_size, MPI_FLOAT,
                              temp_v->data(), v_local_size, MPI_FLOAT,
                              MPI_COMM_WORLD);

                // Rearrange from rank-major to row-interleaved (accounting for batch dimension)
                for (int b = 0; b < B; ++b)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        for (int r = 0; r < size; ++r)
                        {
                            const float *src = temp_v->data() + r * v_local_size + (b * T + t) * n_kv_heads_local_ * head_dim_;
                            float *dst = v_snapshot->data() + (b * T + t) * n_kv_heads_ * head_dim_ + r * n_kv_heads_local_ * head_dim_;
                            std::memcpy(dst, src, n_kv_heads_local_ * head_dim_ * sizeof(float));
                        }
                    }
                }
            }
            else
            {
                std::copy(v_local->data(), v_local->data() + v_local_size, v_snapshot->data());
            }
            snapshot_callback_(PipelineStage::V_PROJECTION, current_layer_idx_, v_snapshot);
        }

        auto t_qkv_end = std::chrono::high_resolution_clock::now();
        total_qkv_proj_ms_ += std::chrono::duration<double, std::milli>(t_qkv_end - t_qkv_start).count();

        // ========================================================================
        // Step 2: Reshape to [B, n_heads_local, T, head_dim] and apply RoPE
        // ========================================================================
        auto t_rope_start = std::chrono::high_resolution_clock::now();

        // Q: [B, T, n_heads_local * head_dim] -> [B, n_heads_local, T, head_dim]
        // This is a logical reshape, we'll work with the data in-place

        // DEBUG: Log Q/K BEFORE RoPE
        if (rank == 0 && current_layer_idx_ == 0)
        {
            LOG_DEBUG("[ROPE_DEBUG] BEFORE RoPE application:");
            LOG_DEBUG("  Q_local[0:5]: [" << q_local->data()[0] << ", " << q_local->data()[1] << ", "
                                          << q_local->data()[2] << ", " << q_local->data()[3] << ", " << q_local->data()[4] << "]");
            LOG_DEBUG("  K_local[0:5]: [" << k_local->data()[0] << ", " << k_local->data()[1] << ", "
                                          << k_local->data()[2] << ", " << k_local->data()[3] << ", " << k_local->data()[4] << "]");
        }

        applyRoPE(q_local->data(), k_local->data(), B, T);

        // DEBUG: Check Q/K after RoPE for both sequences
        if (rank == 0 && current_layer_idx_ == 0)
        {
            LOG_DEBUG("[ROPE_DEBUG] AFTER RoPE application:");
            LOG_DEBUG("  Q_local seq0_tok0 [0:3]: [" << q_local->data()[0] << ", " << q_local->data()[1] << ", " << q_local->data()[2] << "]");
            // For B=2, T=5, n_heads_local=7, head_dim=64: seq1 starts at offset T*n_heads_local*head_dim = 5*7*64 = 2240
            int seq1_offset = T * n_heads_local_ * head_dim_;
            LOG_DEBUG("  Q_local seq1_tok0 [offset " << seq1_offset << "]: [" << q_local->data()[seq1_offset] << ", "
                                                     << q_local->data()[seq1_offset + 1] << ", " << q_local->data()[seq1_offset + 2] << "]");
            LOG_DEBUG("  K_local[0:5]: [" << k_local->data()[0] << ", " << k_local->data()[1] << ", "
                                          << k_local->data()[2] << ", " << k_local->data()[3] << ", " << k_local->data()[4] << "]");

            auto check_tensor = [](const float *data, size_t size, const char *name)
            {
                float min_val = data[0], max_val = data[0];
                for (size_t i = 1; i < size; ++i)
                {
                    min_val = std::min(min_val, data[i]);
                    max_val = std::max(max_val, data[i]);
                }
                LOG_DEBUG("[MPIAttentionBatch] After RoPE " << name << ": min=" << min_val << " max=" << max_val);
            };
            check_tensor(q_local->data(), q_local->size(), "Q");
            check_tensor(k_local->data(), k_local->size(), "K");
        }

        // Capture post-RoPE Q and K (concatenated as [Q | K])
        // For consistency with sequential pipeline, gather to global tensors
        if (snapshot_callback_)
        {
            int rank = getRank();
            int size = getSize();

            // First gather Q globally using same pattern as Q/K/V projections
            auto q_global = std::make_shared<SimpleTensor>(std::vector<int>{B, T, n_heads_ * head_dim_});
            int q_local_size = B * T * n_heads_local_ * head_dim_;

            if (size > 1)
            {
                // DEBUG: Log Q_local before gather
                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[ROPE_GATHER_DEBUG] Q_local BEFORE gather:");
                    LOG_DEBUG("  Token 0: [" << q_local->data()[0] << ", " << q_local->data()[1] << ", " << q_local->data()[2] << "]");
                    LOG_DEBUG("  Token 1 (offset 448): [" << q_local->data()[448] << ", " << q_local->data()[449] << ", " << q_local->data()[450] << "]");
                }

                // Gather into temp buffer, then rearrange from rank-major to row-interleaved
                auto temp_q = std::make_shared<SimpleTensor>(std::vector<int>{B * T * n_heads_ * head_dim_});
                MPI_Allgather(q_local->data(), q_local_size, MPI_FLOAT,
                              temp_q->data(), q_local_size, MPI_FLOAT,
                              MPI_COMM_WORLD);

                // DEBUG: Log temp_q after gather
                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[ROPE_GATHER_DEBUG] temp_q AFTER gather:");
                    LOG_DEBUG("  Rank0 token 0: [" << temp_q->data()[0] << ", " << temp_q->data()[1] << ", " << temp_q->data()[2] << "]");
                    LOG_DEBUG("  Rank0 token 1 (offset 448): [" << temp_q->data()[448] << ", " << temp_q->data()[449] << ", " << temp_q->data()[450] << "]");
                    LOG_DEBUG("  Rank1 start (offset 1792): [" << temp_q->data()[1792] << ", " << temp_q->data()[1793] << ", " << temp_q->data()[1794] << "]");
                }

                // Rearrange from rank-major to row-interleaved (accounting for batch dimension)
                for (int b = 0; b < B; ++b)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        for (int r = 0; r < size; ++r)
                        {
                            const float *src = temp_q->data() + r * q_local_size + (b * T + t) * n_heads_local_ * head_dim_;
                            float *dst = q_global->data() + (b * T + t) * n_heads_ * head_dim_ + r * n_heads_local_ * head_dim_;
                            std::memcpy(dst, src, n_heads_local_ * head_dim_ * sizeof(float));
                        }
                    }
                }

                // DEBUG: Log q_global after rearrange
                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[ROPE_GATHER_DEBUG] q_global AFTER rearrange:");
                    LOG_DEBUG("  Token 0: [" << q_global->data()[0] << ", " << q_global->data()[1] << ", " << q_global->data()[2] << "]");
                    LOG_DEBUG("  Token 1 (offset 896): [" << q_global->data()[896] << ", " << q_global->data()[897] << ", " << q_global->data()[898] << "]");
                }
            }
            else
            {
                std::copy(q_local->data(), q_local->data() + q_local_size, q_global->data());
            }

            // Then gather K globally using same pattern
            auto k_global = std::make_shared<SimpleTensor>(std::vector<int>{B, T, n_kv_heads_ * head_dim_});
            int k_local_size = B * T * n_kv_heads_local_ * head_dim_;

            if (size > 1)
            {
                // DEBUG: Log K_local before gather
                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[ROPE_K_GATHER_DEBUG] K_local BEFORE gather:");
                    LOG_DEBUG("  k_local_size=" << k_local_size << " n_kv_heads_local_=" << n_kv_heads_local_);
                    LOG_DEBUG("  Token 0: [" << k_local->data()[0] << ", " << k_local->data()[1] << ", " << k_local->data()[2] << "]");
                }

                // Gather into temp buffer, then rearrange from rank-major to row-interleaved
                auto temp_k = std::make_shared<SimpleTensor>(std::vector<int>{B * T * n_kv_heads_ * head_dim_});
                MPI_Allgather(k_local->data(), k_local_size, MPI_FLOAT,
                              temp_k->data(), k_local_size, MPI_FLOAT,
                              MPI_COMM_WORLD);

                // DEBUG: Log temp_k after gather
                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[ROPE_K_GATHER_DEBUG] temp_k AFTER gather:");
                    LOG_DEBUG("  Rank0 token 0: [" << temp_k->data()[0] << ", " << temp_k->data()[1] << ", " << temp_k->data()[2] << "]");
                    LOG_DEBUG("  Rank1 start (offset " << k_local_size << "): [" << temp_k->data()[k_local_size] << ", " << temp_k->data()[k_local_size + 1] << ", " << temp_k->data()[k_local_size + 2] << "]");
                }

                // Rearrange from rank-major to row-interleaved (accounting for batch dimension)
                for (int b = 0; b < B; ++b)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        for (int r = 0; r < size; ++r)
                        {
                            const float *src = temp_k->data() + r * k_local_size + (b * T + t) * n_kv_heads_local_ * head_dim_;
                            float *dst = k_global->data() + (b * T + t) * n_kv_heads_ * head_dim_ + r * n_kv_heads_local_ * head_dim_;
                            std::memcpy(dst, src, n_kv_heads_local_ * head_dim_ * sizeof(float));
                        }
                    }
                }

                // DEBUG: Log k_global after rearrange
                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[ROPE_K_GATHER_DEBUG] k_global AFTER rearrange:");
                    LOG_DEBUG("  Token 0: [" << k_global->data()[0] << ", " << k_global->data()[1] << ", " << k_global->data()[2] << "]");
                    LOG_DEBUG("  Token 0 rank1 offset (offset 64): [" << k_global->data()[64] << ", " << k_global->data()[65] << ", " << k_global->data()[66] << "]");
                    LOG_DEBUG("  Token 3 offset 384 (3*128): [" << k_global->data()[384] << ", " << k_global->data()[385] << ", " << k_global->data()[386] << "]");
                    LOG_DEBUG("  Token 3 rank1 head (offset 448=384+64): [" << k_global->data()[448] << ", " << k_global->data()[449] << ", " << k_global->data()[450] << "]");
                    LOG_DEBUG("  Token 3 pos 96 (offset 480=384+96): " << k_global->data()[480]);
                }
            }
            else
            {
                std::copy(k_local->data(), k_local->data() + k_local_size, k_global->data());
            }

            // Concatenate global Q and K along feature dimension
            int q_features_global = n_heads_ * head_dim_;
            int k_features_global = n_kv_heads_ * head_dim_;
            int total_features = q_features_global + k_features_global;

            auto rope_snapshot = std::make_shared<SimpleTensor>(std::vector<int>{B, T, total_features});
            float *rope_data = rope_snapshot->data();

            // DEBUG: Log concatenation parameters
            if (rank == 0 && current_layer_idx_ == 0)
            {
                LOG_DEBUG("[ROPE_CONCAT_DEBUG] Concatenation parameters:");
                LOG_DEBUG("  B=" << B << " T=" << T);
                LOG_DEBUG("  q_features_global=" << q_features_global << " k_features_global=" << k_features_global);
                LOG_DEBUG("  total_features=" << total_features);
                LOG_DEBUG("  q_global size=" << q_global->size() << " k_global size=" << k_global->size());
            }

            for (int b = 0; b < B; ++b)
            {
                for (int t = 0; t < T; ++t)
                {
                    int q_offset = (b * T + t) * q_features_global;
                    int k_offset = (b * T + t) * k_features_global;
                    int dst_offset = (b * T + t) * total_features;

                    const float *q_src = q_global->data() + q_offset;
                    const float *k_src = k_global->data() + k_offset;
                    float *dst = rope_data + dst_offset;

                    // DEBUG: Log first token concatenation for both sequences
                    if (rank == 0 && current_layer_idx_ == 0 && t == 0)
                    {
                        LOG_DEBUG("[ROPE_CONCAT_DEBUG] Seq b=" << b << " Token t=0:");
                        LOG_DEBUG("  q_offset=" << q_offset << " k_offset=" << k_offset << " dst_offset=" << dst_offset);
                        LOG_DEBUG("  q_src[0:3]: [" << q_src[0] << ", " << q_src[1] << ", " << q_src[2] << "]");
                        LOG_DEBUG("  k_src[0:3]: [" << k_src[0] << ", " << k_src[1] << ", " << k_src[2] << "]");
                    }

                    std::copy(q_src, q_src + q_features_global, dst);
                    std::copy(k_src, k_src + k_features_global, dst + q_features_global);

                    // DEBUG: Verify after copy for both sequences
                    if (rank == 0 && current_layer_idx_ == 0 && t == 0)
                    {
                        LOG_DEBUG("[ROPE_CONCAT_DEBUG] After copy seq b=" << b << ":");
                        LOG_DEBUG("  dst[0:3] (Q): [" << dst[0] << ", " << dst[1] << ", " << dst[2] << "]");
                        LOG_DEBUG("  dst[896:899] (K): [" << dst[896] << ", " << dst[897] << ", " << dst[898] << "]");
                    }
                }
            }

            // DEBUG: Log rope_snapshot before callback
            if (rank == 0 && current_layer_idx_ == 0)
            {
                LOG_DEBUG("[ROPE_SNAPSHOT_DEBUG] rope_snapshot before callback:");
                LOG_DEBUG("  Shape: [" << B << ", " << T << ", " << total_features << "]");
                LOG_DEBUG("  Total size: " << rope_snapshot->size());
                LOG_DEBUG("  First token Q part [0:3]: [" << rope_snapshot->data()[0] << ", " << rope_snapshot->data()[1] << ", " << rope_snapshot->data()[2] << "]");
                LOG_DEBUG("  First token K start (offset 896): [" << rope_snapshot->data()[896] << ", " << rope_snapshot->data()[897] << ", " << rope_snapshot->data()[898] << "]");
                LOG_DEBUG("  Token 1 Q part (offset 1024): [" << rope_snapshot->data()[1024] << ", " << rope_snapshot->data()[1025] << ", " << rope_snapshot->data()[1026] << "]");
            }

            snapshot_callback_(PipelineStage::ROPE_APPLICATION, current_layer_idx_, rope_snapshot);
        }

        auto t_rope_end = std::chrono::high_resolution_clock::now();
        total_rope_ms_ += std::chrono::duration<double, std::milli>(t_rope_end - t_rope_start).count();

        // ========================================================================
        // Step 2.5: Expand K and V for GQA (Grouped Query Attention)
        // ========================================================================
        auto t_gqa_start = std::chrono::high_resolution_clock::now();

        // IMPORTANT: This must be OUTSIDE the snapshot_callback block!
        // If n_kv_heads < n_heads, replicate KV heads to match Q head count
        // E.g., Qwen: 2 KV heads → 14 Q heads, group_size=7
        // Each KV head serves multiple Q heads

        std::shared_ptr<SimpleTensor> k_expanded_local = k_local;
        std::shared_ptr<SimpleTensor> v_expanded_local = v_local;

        {
            int rank = getRank();
            LOG_DEBUG("[GQA_DEBUG] rank=" << rank << " current_layer_idx_=" << current_layer_idx_);
            if (rank == 0 && current_layer_idx_ == 0)
            {
                LOG_DEBUG("[GQA_DEBUG] Checking GQA condition: n_kv_heads_=" << n_kv_heads_ << " n_heads_=" << n_heads_);
            }

            if (n_kv_heads_ < n_heads_)
            {
                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[GQA_DEBUG] GQA expansion needed!");
                }
                // Allocate expanded K and V: [B, T, n_heads_local * head_dim]
                k_expanded_local = std::make_shared<SimpleTensor>(
                    std::vector<int>{B, T, n_heads_local_ * head_dim_});
                v_expanded_local = std::make_shared<SimpleTensor>(
                    std::vector<int>{B, T, n_heads_local_ * head_dim_});

                // Get KV head distribution for this rank
                auto [local_kv_heads, kv_offset] = getKVHeadDistribution();

                // Expand K and V for each batch element
                for (int b = 0; b < B; ++b)
                {
                    const float *k_src = k_local->data() + b * T * n_kv_heads_local_ * head_dim_;
                    const float *v_src = v_local->data() + b * T * n_kv_heads_local_ * head_dim_;
                    float *k_dst = k_expanded_local->data() + b * T * n_heads_local_ * head_dim_;
                    float *v_dst = v_expanded_local->data() + b * T * n_heads_local_ * head_dim_;

                    llaminar::attn::expand_kv_for_gqa(
                        k_src, v_src,
                        k_dst, v_dst,
                        T,                 // seq_len
                        head_dim_,         // head_dim
                        n_heads_local_,    // n_heads (local Q heads)
                        n_kv_heads_local_, // n_kv_heads (local KV heads)
                        head_offset_,      // head_offset
                        n_heads_,          // total_q_heads
                        false,             // gathered_rank_major (data is already token-major)
                        kv_offset);        // kv_head_offset_for_rank
                }

                // DEBUG: Log GQA expansion
                if (rank == 0 && current_layer_idx_ == 0 && debugEnv().attention.verbose)
                {
                    LOG_DEBUG("[MPIAttentionBatch] GQA expansion:");
                    LOG_DEBUG("  n_kv_heads_local=" << n_kv_heads_local_ << " → n_heads_local=" << n_heads_local_);
                    LOG_DEBUG("  head_offset=" << head_offset_ << " kv_offset=" << kv_offset);
                    LOG_DEBUG("  K before: [" << k_local->data()[0] << ", " << k_local->data()[1] << ", ...]");
                    LOG_DEBUG("  K after: [" << k_expanded_local->data()[0] << ", " << k_expanded_local->data()[1] << ", ...]");
                }
            }
        }

        auto t_gqa_end = std::chrono::high_resolution_clock::now();
        total_gqa_expand_ms_ += std::chrono::duration<double, std::milli>(t_gqa_end - t_gqa_start).count();

        // ========================================================================
        // Step 3: Compute attention scores with per-batch causal masking
        // ========================================================================
        auto t_scores_start = std::chrono::high_resolution_clock::now();
        // scores: [B, n_heads_local, T, T]
        int scores_size = B * n_heads_local_ * T * T;
        std::vector<float> scores(scores_size);

        computeAttentionScores(
            q_local->data(),
            k_expanded_local->data(),
            scores.data(),
            B,
            T);

        auto t_scores_end = std::chrono::high_resolution_clock::now();
        total_scores_ms_ += std::chrono::duration<double, std::milli>(t_scores_end - t_scores_start).count();

        // ========================================================================
        // Step 4: Apply causal mask and softmax per-batch
        // ========================================================================
        auto t_softmax_start = std::chrono::high_resolution_clock::now();
        applyCausalMaskAndSoftmax(scores.data(), B, T);

        auto t_softmax_end = std::chrono::high_resolution_clock::now();
        total_softmax_ms_ += std::chrono::duration<double, std::milli>(t_softmax_end - t_softmax_start).count();

        // DEBUG: Check scores after softmax on all ranks at layer 0
        if (current_layer_idx_ == 0)
        {
            float min_val = scores[0], max_val = scores[0];
            int nan_count = 0;
            for (size_t i = 0; i < scores.size(); ++i)
            {
                if (std::isnan(scores[i]))
                    nan_count++;
                min_val = std::min(min_val, scores[i]);
                max_val = std::max(max_val, scores[i]);
            }
            LOG_DEBUG("[SOFTMAX_CHECK] Rank " << getRank() << " After softmax: min=" << min_val << " max=" << max_val << " nan_count=" << nan_count);
        }

        // ========================================================================
        // Step 5: Compute attention output: scores @ V
        // ========================================================================
        auto t_context_start = std::chrono::high_resolution_clock::now();
        // CRITICAL: Primitive outputs [B, T, n_heads_local * head_dim], NOT [B, n_heads_local, T, head_dim]!
        auto attn_output_local = std::make_shared<SimpleTensor>(
            std::vector<int>{B, T, n_heads_local_ * head_dim_});

        computeAttentionOutput(
            scores.data(),
            v_expanded_local->data(),
            attn_output_local->data(),
            B,
            T);

        // DEBUG: Check attention output after scores @ V on all ranks at layer 0
        if (current_layer_idx_ == 0)
        {
            int nan_count = 0;
            for (size_t i = 0; i < attn_output_local->size(); ++i)
            {
                if (std::isnan(attn_output_local->data()[i]))
                    nan_count++;
            }
            LOG_DEBUG("[ATTN_OUTPUT_CHECK] Rank " << getRank() << " After scores@V: nan_count=" << nan_count
                                                  << " first 5: [" << attn_output_local->data()[0] << ", " << attn_output_local->data()[1]
                                                  << ", " << attn_output_local->data()[2] << ", " << attn_output_local->data()[3] << ", " << attn_output_local->data()[4] << "]");
        }

        // Capture attention context (before output projection)
        // For consistency with sequential pipeline, gather to global tensors
        if (snapshot_callback_)
        {
            int rank = getRank();
            int size = getSize();

            // attn_output_local is already in [B, T, n_heads_local * head_dim] format - perfect!
            // No transpose needed, just use it directly as context_local
            const float *context_local_data = attn_output_local->data();

            // Now gather to global tensor [B, T, n_heads * head_dim]
            // CRITICAL: Must use row-by-row gather to match sequential operator's token-major layout!
            auto context_snapshot = std::make_shared<SimpleTensor>(
                std::vector<int>{B, T, n_heads_ * head_dim_});

            if (size > 1)
            {
                // Gather row-by-row (token-by-token) across all batches
                // This creates token-major layout where heads from all ranks are interleaved
                int local_head_dim = n_heads_local_ * head_dim_;
                int global_head_dim = n_heads_ * head_dim_;

                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[ATTN_CONTEXT_GATHER] Gathering ATTENTION_CONTEXT:");
                    LOG_DEBUG("  B=" << B << " T=" << T);
                    LOG_DEBUG("  local_head_dim=" << local_head_dim << " (n_heads_local=" << n_heads_local_ << " * head_dim=" << head_dim_ << ")");
                    LOG_DEBUG("  global_head_dim=" << global_head_dim << " (n_heads=" << n_heads_ << " * head_dim=" << head_dim_ << ")");
                    LOG_DEBUG("  context_local first 5: [" << context_local_data[0] << ", " << context_local_data[1] << ", "
                                                           << context_local_data[2] << ", " << context_local_data[3] << ", " << context_local_data[4] << "]");
                }

                for (int b = 0; b < B; ++b)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        const float *local_row = context_local_data + (b * T + t) * local_head_dim;
                        float *global_row = context_snapshot->data() + (b * T + t) * global_head_dim;

                        MPI_Allgather(local_row, local_head_dim, MPI_FLOAT,
                                      global_row, local_head_dim, MPI_FLOAT,
                                      MPI_COMM_WORLD);
                    }
                }

                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_DEBUG("[ATTN_CONTEXT_GATHER] After gather:");
                    LOG_DEBUG("  context_snapshot first 5: [" << context_snapshot->data()[0] << ", " << context_snapshot->data()[1] << ", "
                                                              << context_snapshot->data()[2] << ", " << context_snapshot->data()[3] << ", " << context_snapshot->data()[4] << "]");
                    LOG_DEBUG("  At offset 448 (rank1 start): [" << context_snapshot->data()[448] << ", " << context_snapshot->data()[449] << ", "
                                                                 << context_snapshot->data()[450] << "]");
                }
            }
            else
            {
                int context_local_size = B * T * n_heads_local_ * head_dim_;
                std::copy(context_local_data, context_local_data + context_local_size, context_snapshot->data());
            }

            snapshot_callback_(PipelineStage::ATTENTION_CONTEXT, current_layer_idx_, context_snapshot);
        }

        auto t_context_end = std::chrono::high_resolution_clock::now();
        total_context_ms_ += std::chrono::duration<double, std::milli>(t_context_end - t_context_start).count();

        // ========================================================================
        // Step 6: Output projection preparation
        // ========================================================================
        auto t_output_prep_start = std::chrono::high_resolution_clock::now();

        // attn_output_local is already in [B, T, n_heads_local * head_dim] format - no reshape needed!
        auto attn_concat_local = attn_output_local; // Just alias, no copy needed

        auto t_output_prep_end = std::chrono::high_resolution_clock::now();
        total_output_prep_ms_ += std::chrono::duration<double, std::milli>(t_output_prep_end - t_output_prep_start).count();

        // ========================================================================
        // Step 7: Output projection with distributed computation
        // ========================================================================
        auto t_outproj_start = std::chrono::high_resolution_clock::now();
        // Each rank computes partial output: [B*T, n_heads_local*head_dim] @ [n_heads_local*head_dim, D]^T
        // Then MPI_Allreduce sums across ranks to get final [B*T, D] output

        // Extract local column slice of wo for this rank's heads
        // wo shape: [D, n_heads * head_dim] (row-major)
        // We need columns [head_offset*head_dim : (head_offset+n_heads_local)*head_dim]
        // Note: wo_cols already declared in validation section above
        int local_out_cols = n_heads_local_ * head_dim_;
        int col_offset = head_offset_ * head_dim_;

        std::shared_ptr<TensorBase> wo_local;

        if (using_pre_sliced_weights)
        {
            // Weights already column-sliced - use directly
            wo_local = wo;
        }
        else
        {
            // Replicated weights - extract column slice
            wo_local = std::make_shared<SimpleTensor>(std::vector<int>{D, local_out_cols});
            const float *wo_data = wo->data();
            float *wo_local_data = wo_local->data();

            // Extract columns - need to copy column-wise from row-major matrix
            for (int row = 0; row < D; ++row)
            {
                std::copy(
                    wo_data + row * wo_cols + col_offset,
                    wo_data + row * wo_cols + col_offset + local_out_cols,
                    wo_local_data + row * local_out_cols);
            }
        }

        // Reshape attn_concat_local to 2D for matrix multiplication
        int M = B * T;
        int K = local_out_cols;
        int N = D;

        if (current_layer_idx_ == 0) // Log for all ranks on layer 0
        {
            // Check for NaN/Inf in input tensors
            int attn_nan_count = 0, attn_inf_count = 0;
            for (size_t i = 0; i < attn_concat_local->size(); ++i)
            {
                float val = attn_concat_local->data()[i];
                if (std::isnan(val))
                    attn_nan_count++;
                if (std::isinf(val))
                    attn_inf_count++;
            }

            int wo_nan_count = 0, wo_inf_count = 0;
            for (size_t i = 0; i < wo_local->size(); ++i)
            {
                float val = wo_local->data()[i];
                if (std::isnan(val))
                    wo_nan_count++;
                if (std::isinf(val))
                    wo_inf_count++;
            }

            LOG_DEBUG("[WO_PROJ_DEBUG] Rank " << getRank() << " Output projection setup:");
            LOG_DEBUG("  wo_local shape: [" << wo_local->shape()[0] << ", " << wo_local->shape()[1] << "]");
            LOG_DEBUG("  attn_concat_local shape: [" << attn_concat_local->shape()[0] << ", "
                                                     << attn_concat_local->shape()[1] << ", " << attn_concat_local->shape()[2] << "]");
            LOG_DEBUG("  Matrix dims: M=" << M << " N=" << N << " K=" << K);
            LOG_DEBUG("  using_pre_sliced_weights=" << using_pre_sliced_weights);
            LOG_DEBUG("  attn_concat NaN/Inf: " << attn_nan_count << "/" << attn_inf_count);
            LOG_DEBUG("  wo_local NaN/Inf: " << wo_nan_count << "/" << wo_inf_count);
            LOG_DEBUG("  attn_concat first 5: [" << attn_concat_local->data()[0] << ", " << attn_concat_local->data()[1]
                                                 << ", " << attn_concat_local->data()[2] << ", " << attn_concat_local->data()[3] << ", " << attn_concat_local->data()[4] << "]");
            LOG_DEBUG("  wo_local first 5: [" << wo_local->data()[0] << ", " << wo_local->data()[1]
                                              << ", " << wo_local->data()[2] << ", " << wo_local->data()[3] << ", " << wo_local->data()[4] << "]");
        }

        // Compute partial output: [M, K] @ [N, K]^T = [M, N]
        // Note: wo_local is [N, K] stored row-major, so we use transpose_B=true
        bool is_prefill = (B * T == T); // Simple heuristic: batch_size * seq_len == seq_len means single sequence
        bool success = adaptiveMatMul(
            attn_concat_local->data(), // A: [M, K] = [B*T, local_out_cols]
            wo_local->data(),          // B: [N, K] = [D, local_out_cols]
            output->data(),            // C: [M, N] = [B*T, D]
            M,                         // m = B*T
            N,                         // n = D
            K,                         // k = local_out_cols
            is_prefill,                // is_prefill hint
            false,                     // distributed_partition (weights already sliced)
            false,                     // transpose_A
            true,                      // transpose_B (wo_local is [N,K] row-major)
            1.0f,                      // alpha
            0.0f                       // beta
        );

        if (!success)
        {
            LOG_ERROR("adaptiveMatMul failed for output projection on rank " << getRank());
            return false;
        }

        if (current_layer_idx_ == 0) // Log for all ranks on layer 0
        {
            LOG_DEBUG("[WO_PROJ_DEBUG] Rank " << getRank() << " After sgemm, before Allreduce:");
            LOG_DEBUG("  output first 5: [" << output->data()[0] << ", " << output->data()[1]
                                            << ", " << output->data()[2] << ", " << output->data()[3] << ", " << output->data()[4] << "]");

            // Check for NaN/Inf
            int nan_count = 0, inf_count = 0;
            for (size_t i = 0; i < output->size(); ++i)
            {
                if (std::isnan(output->data()[i]))
                    nan_count++;
                if (std::isinf(output->data()[i]))
                    inf_count++;
            }
            if (nan_count > 0 || inf_count > 0)
            {
                LOG_ERROR("  PRE-ALLREDUCE: Detected nan_count=" << nan_count << " inf_count=" << inf_count);
            }
        }

        auto t_outproj_end = std::chrono::high_resolution_clock::now();
        total_output_proj_ms_ += std::chrono::duration<double, std::milli>(t_outproj_end - t_outproj_start).count();

        // ========================================================================
        // Step 8: MPI_Allreduce to sum partial outputs across ranks
        // ========================================================================
        auto t_reduce_start = std::chrono::high_resolution_clock::now();
        if (getSize() > 1)
        {
            MPI_Allreduce(MPI_IN_PLACE, output->data(), output->size(),
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }

        auto t_reduce_end = std::chrono::high_resolution_clock::now();
        total_mpi_reduce_ms_ += std::chrono::duration<double, std::milli>(t_reduce_end - t_reduce_start).count();

        if (current_layer_idx_ == 0) // Log for all ranks on layer 0
        {
            LOG_DEBUG("[WO_PROJ_DEBUG] Rank " << getRank() << " After Allreduce:");
            LOG_DEBUG("  output first 5: [" << output->data()[0] << ", " << output->data()[1]
                                            << ", " << output->data()[2] << ", " << output->data()[3] << ", " << output->data()[4] << "]");

            // Compute L2 norm for magnitude comparison
            float sum_sq = 0.0f;
            for (size_t i = 0; i < output->size(); ++i)
            {
                sum_sq += output->data()[i] * output->data()[i];
            }
            float l2_norm = std::sqrt(sum_sq / output->size());
            LOG_DEBUG("[MAGNITUDE_TRACE] Rank " << getRank() << " Layer0 Attention Output: L2_norm=" << l2_norm << " size=" << output->size());
        }

        if (rank == 0)
        {
            LOG_DEBUG("[MPIAttentionBatchOperator] Completed successfully");
        }

        return true;
    }

    void MPIAttentionBatchOperator::applyRoPE(
        float *q,
        float *k,
        int batch_size,
        int seq_len)
    {
        // Apply RoPE to Q and K tensors using shared AttentionPrimitives implementation
        // Q: [B, T, n_heads_local * head_dim]
        // K: [B, T, n_kv_heads_local * head_dim]

        // DEBUG: Log RoPE parameters
        if (getRank() == 0 && current_layer_idx_ == 0)
        {
            LOG_DEBUG("[ROPE_APPLY_DEBUG] Delegating to llaminar::attn::apply_rope_batched:");
            LOG_DEBUG("  batch_size=" << batch_size << " seq_len=" << seq_len);
            LOG_DEBUG("  n_heads_local_=" << n_heads_local_ << " n_kv_heads_local_=" << n_kv_heads_local_);
            LOG_DEBUG("  head_dim_=" << head_dim_);
            LOG_DEBUG("  n_past=" << n_past_ << " rope_freq_base_=" << rope_freq_base_);
        }

        // Use shared RoPE implementation from AttentionPrimitives
        // This ensures consistency with sequential operator and eliminates code duplication
        llaminar::attn::apply_rope_batched(
            q, k,
            batch_size, seq_len, head_dim_,
            n_heads_local_, n_kv_heads_local_,
            n_past_, rope_freq_base_);

        // DEBUG: Log first few values after RoPE
        if (getRank() == 0 && current_layer_idx_ == 0 && batch_size > 0 && seq_len > 0)
        {
            LOG_DEBUG("[ROPE_APPLY_DEBUG] After apply_rope_batched:");
            LOG_DEBUG("  q[batch=0, t=0, head=0, dim=0:3]: ["
                      << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << "]");
            if (seq_len > 1)
            {
                int t1_offset = n_heads_local_ * head_dim_;
                LOG_DEBUG("  q[batch=0, t=1, head=0, dim=0:3]: ["
                          << q[t1_offset] << ", " << q[t1_offset + 1] << ", "
                          << q[t1_offset + 2] << ", " << q[t1_offset + 3] << "]");
            }
        }
    }

    void MPIAttentionBatchOperator::computeAttentionScores(
        const float *q,
        const float *k,
        float *scores,
        int batch_size,
        int seq_len)
    {
        // Compute Q @ K^T using proven AttentionPrimitives
        // Q, K: [B, T, n_heads_local * head_dim]
        // scores: [B, n_heads_local, T, T]

        if (getRank() == 0 && current_layer_idx_ == 0)
        {
            LOG_DEBUG("[ATTN_SCORES_DEBUG] Delegating to llaminar::attn::compute_qk_scores_batched:");
            LOG_DEBUG("  batch_size=" << batch_size << " seq_len=" << seq_len);
            LOG_DEBUG("  n_heads_local_=" << n_heads_local_ << " head_dim_=" << head_dim_);
        }

        // Use proven implementation from AttentionPrimitives
        // This ensures consistency with sequential operator
        llaminar::attn::compute_qk_scores_batched(
            q, k, scores,
            batch_size, seq_len, head_dim_, n_heads_local_,
            true // causal masking
        );
    }

    void MPIAttentionBatchOperator::applyCausalMaskAndSoftmax(
        float *scores,
        int batch_size,
        int seq_len)
    {
        // Softmax is now applied by compute_qk_scores_batched using the vectorized fused kernel
        // scores: [B, n_heads_local, T, T]
        // Note: causal masking AND softmax already applied by compute_qk_scores_batched

        // This function is now a no-op - kept for API compatibility
        // The fused kernel in compute_qk_scores_batched handles:
        //   1. Q @ K^T computation (GEMM)
        //   2. Scaling by 1/sqrt(head_dim)
        //   3. Causal masking
        //   4. Vectorized softmax (AVX2/AVX512)

        if (getRank() == 0 && current_layer_idx_ == 0)
        {
            LOG_DEBUG("[SOFTMAX_DEBUG] Softmax already applied by vectorized fused kernel in compute_qk_scores_batched");
            LOG_DEBUG("  batch_size=" << batch_size << " seq_len=" << seq_len);
            LOG_DEBUG("  n_heads_local_=" << n_heads_local_);
        }

        // No-op: softmax already applied
    }

    void MPIAttentionBatchOperator::computeAttentionOutput(
        const float *scores,
        const float *v,
        float *output,
        int batch_size,
        int seq_len)
    {
        // Compute scores @ V using proven AttentionPrimitives
        // scores: [B, n_heads_local, T, T]
        // V: [B, T, n_kv_heads_local * head_dim]
        // output: [B, T, n_heads_local * head_dim]

        if (getRank() == 0 && current_layer_idx_ == 0)
        {
            LOG_DEBUG("[ATTN_OUTPUT_DEBUG] Delegating to llaminar::attn::apply_scores_to_v_batched:");
            LOG_DEBUG("  batch_size=" << batch_size << " seq_len=" << seq_len);
            LOG_DEBUG("  n_heads_local_=" << n_heads_local_ << " head_dim_=" << head_dim_);
        }

        // Use proven implementation from AttentionPrimitives
        llaminar::attn::apply_scores_to_v_batched(
            scores, v, output,
            batch_size, seq_len, head_dim_, n_heads_local_);
    }

    void MPIAttentionBatchOperator::printPerformanceBreakdown() const
    {
        if (getRank() != 0)
            return; // Only print on rank 0

        double total_ms = total_qkv_proj_ms_ + total_rope_ms_ + total_gqa_expand_ms_ +
                          total_scores_ms_ + total_softmax_ms_ + total_context_ms_ +
                          total_output_prep_ms_ + total_output_proj_ms_ + total_mpi_reduce_ms_;

        if (total_ms < 0.001)
            return; // Skip if no timing data

        std::cout << "\n[ATTN_BREAKDOWN] Attention Operator Performance:\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Q/K/V Proj:     " << std::setw(8) << total_qkv_proj_ms_ << " ms (" << std::setw(5) << (100.0 * total_qkv_proj_ms_ / total_ms) << "%)\n";
        std::cout << "  RoPE:           " << std::setw(8) << total_rope_ms_ << " ms (" << std::setw(5) << (100.0 * total_rope_ms_ / total_ms) << "%)\n";
        std::cout << "  GQA Expand:     " << std::setw(8) << total_gqa_expand_ms_ << " ms (" << std::setw(5) << (100.0 * total_gqa_expand_ms_ / total_ms) << "%)\n";
        std::cout << "  Attn Scores:    " << std::setw(8) << total_scores_ms_ << " ms (" << std::setw(5) << (100.0 * total_scores_ms_ / total_ms) << "%)\n";
        std::cout << "  Softmax:        " << std::setw(8) << total_softmax_ms_ << " ms (" << std::setw(5) << (100.0 * total_softmax_ms_ / total_ms) << "%)\n";
        std::cout << "  Context (S@V):  " << std::setw(8) << total_context_ms_ << " ms (" << std::setw(5) << (100.0 * total_context_ms_ / total_ms) << "%)\n";
        std::cout << "  Output Prep:    " << std::setw(8) << total_output_prep_ms_ << " ms (" << std::setw(5) << (100.0 * total_output_prep_ms_ / total_ms) << "%)\n";
        std::cout << "  Output Proj:    " << std::setw(8) << total_output_proj_ms_ << " ms (" << std::setw(5) << (100.0 * total_output_proj_ms_ / total_ms) << "%)\n";
        std::cout << "  MPI Reduce:     " << std::setw(8) << total_mpi_reduce_ms_ << " ms (" << std::setw(5) << (100.0 * total_mpi_reduce_ms_ / total_ms) << "%)\n";
        std::cout << "  TOTAL:          " << std::setw(8) << total_ms << " ms\n";
        std::cout << std::flush;
    }

    void MPIAttentionBatchOperator::resetPerformanceCounters()
    {
        total_qkv_proj_ms_ = 0.0;
        total_rope_ms_ = 0.0;
        total_gqa_expand_ms_ = 0.0;
        total_scores_ms_ = 0.0;
        total_softmax_ms_ = 0.0;
        total_context_ms_ = 0.0;
        total_output_prep_ms_ = 0.0;
        total_output_proj_ms_ = 0.0;
        total_mpi_reduce_ms_ = 0.0;
    }

} // namespace llaminar
