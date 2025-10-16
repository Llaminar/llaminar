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
#include "attention/AttentionStageContracts.h"
#include "utils/DebugEnv.h"
#include "AdaptiveMatmul.h"
#include <cblas.h>
#include <cmath>
#include <algorithm>
#include <mpi.h>

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

        // Precompute RoPE frequencies for maximum sequence length
        const int max_seq_len = 2048;
        rope_freqs_.resize(head_dim_ / 2);
        for (int i = 0; i < head_dim_ / 2; ++i)
        {
            rope_freqs_[i] = 1.0f / std::pow(rope_freq_base_, 2.0f * i / head_dim_);
        }

        if (rank == 0)
        {
            LOG_DEBUG("[MPIAttentionBatchOperator] Initialized: n_heads=" << n_heads_
                                                                          << " n_kv_heads=" << n_kv_heads_ << " head_dim=" << head_dim_
                                                                          << " (local: " << n_heads_local_ << " heads, offset=" << head_offset_ << ")");
        }
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
        // CRITICAL DEBUG: This should appear if execute() is being called!
        std::cerr << "[RANK " << getRank() << "] *** MPIAttentionBatchOperator::execute() ENTRY POINT ***" << std::endl;

        if (!validate(inputs, outputs))
        {
            std::cerr << "[RANK " << getRank() << "] *** MPIAttentionBatchOperator::execute() VALIDATION FAILED ***" << std::endl;
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

        // ========================================================================
        // Weight validation - catch dimension mismatches early
        // ========================================================================
        const int expected_wq_rows = n_heads_ * head_dim_;    // Full Q projection output
        const int expected_wk_rows = n_kv_heads_ * head_dim_; // Full K projection output
        const int expected_wv_rows = n_kv_heads_ * head_dim_; // Full V projection output
        const int expected_wo_cols = n_heads_ * head_dim_;    // Full output projection input

        const int wq_rows = static_cast<int>(wq->shape()[0]);
        const int wq_cols = static_cast<int>(wq->shape()[1]);
        const int wk_rows = static_cast<int>(wk->shape()[0]);
        const int wk_cols = static_cast<int>(wk->shape()[1]);
        const int wv_rows = static_cast<int>(wv->shape()[0]);
        const int wv_cols = static_cast<int>(wv->shape()[1]);
        const int wo_rows = static_cast<int>(wo->shape()[0]);
        const int wo_cols = static_cast<int>(wo->shape()[1]);

        // Validate weight dimensions
        if (wq_rows != expected_wq_rows || wq_cols != D)
        {
            LOG_ERROR("[MPIAttentionBatch] wq dimension mismatch: got [" << wq_rows << "," << wq_cols
                                                                         << "], expected [" << expected_wq_rows << "," << D << "]");
            LOG_ERROR("  n_heads=" << n_heads_ << " head_dim=" << head_dim_ << " D=" << D);
            return false;
        }
        if (wk_rows != expected_wk_rows || wk_cols != D)
        {
            LOG_ERROR("[MPIAttentionBatch] wk dimension mismatch: got [" << wk_rows << "," << wk_cols
                                                                         << "], expected [" << expected_wk_rows << "," << D << "]");
            LOG_ERROR("  n_kv_heads=" << n_kv_heads_ << " head_dim=" << head_dim_ << " D=" << D);
            return false;
        }
        if (wv_rows != expected_wv_rows || wv_cols != D)
        {
            LOG_ERROR("[MPIAttentionBatch] wv dimension mismatch: got [" << wv_rows << "," << wv_cols
                                                                         << "], expected [" << expected_wv_rows << "," << D << "]");
            return false;
        }
        if (wo_rows != D || wo_cols != expected_wo_cols)
        {
            LOG_ERROR("[MPIAttentionBatch] wo dimension mismatch: got [" << wo_rows << "," << wo_cols
                                                                         << "], expected [" << D << "," << expected_wo_cols << "]");
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

        // Step 1: Q, K, V projections using direct adaptiveMatMul
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
            // Extract local rows from wq for this rank's heads
            // Global wq shape: [n_heads * head_dim, d_model] = [out_dim, in_dim]
            // Local wq shape: [n_heads_local * head_dim, d_model] = [out_dim_local, in_dim]
            int local_rows = n_heads_local_ * head_dim_;
            int row_offset = head_offset_ * head_dim_;

            if (getRank() == 0 && current_layer_idx_ == 0)
            {
                LOG_INFO("[BATCH_Q_PROJ] Layer " << current_layer_idx_ << " Rank " << getRank()
                                                 << " Q weight extraction:");
                LOG_INFO("  n_heads=" << n_heads_ << " n_heads_local=" << n_heads_local_
                                      << " head_dim=" << head_dim_ << " head_offset=" << head_offset_);
                LOG_INFO("  local_rows=" << local_rows << " row_offset=" << row_offset
                                         << " D=" << D);
                LOG_INFO("  Global wq shape: [" << wq->shape()[0] << ", " << wq->shape()[1] << "]");
                LOG_INFO("  Will extract rows [" << row_offset << ":" << (row_offset + local_rows) << "]");
                LOG_INFO("  wq global first 5: [" << wq->data()[0] << ", " << wq->data()[1] << ", "
                                                  << wq->data()[2] << ", " << wq->data()[3] << ", " << wq->data()[4] << "]");
                LOG_INFO("  wq at offset " << (row_offset * D) << ": [" << wq->data()[row_offset * D]
                                           << ", " << wq->data()[row_offset * D + 1] << ", " << wq->data()[row_offset * D + 2]
                                           << ", " << wq->data()[row_offset * D + 3] << ", " << wq->data()[row_offset * D + 4] << "]");
            }

            // MPILinearBatchOperator expects weights as [out_dim, in_dim]
            auto wq_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows, D});

            // Extract rows [row_offset : row_offset + local_rows] - simple memcpy of contiguous data
            size_t offset_elements = static_cast<size_t>(row_offset) * D;
            size_t copy_elements = static_cast<size_t>(local_rows) * D;
            std::memcpy(wq_local->data(), wq->data() + offset_elements, copy_elements * sizeof(float));

            // Extract local bias if present
            std::shared_ptr<TensorBase> bq_local;
            if (bq && bq->size() > 0)
            {
                bq_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows});
                const float *bq_data = bq->data();
                float *bq_local_data = bq_local->data();
                std::copy(bq_data + row_offset, bq_data + row_offset + local_rows, bq_local_data);
            }

            if (getRank() == 0 && current_layer_idx_ == 0)
            {
                LOG_INFO("[BATCH_Q_PROJ] Local wq_local first 5: [" << wq_local->data()[0] << ", "
                                                                    << wq_local->data()[1] << ", " << wq_local->data()[2] << ", "
                                                                    << wq_local->data()[3] << ", " << wq_local->data()[4] << "]");
                LOG_INFO("[BATCH_Q_PROJ] Input first 5: [" << input->data()[0] << ", " << input->data()[1]
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
                LOG_ERROR("[FIX_Q_PROJ] About to call adaptiveMatMul: m=" << m << " n=" << n << " k=" << k);
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
                LOG_INFO("[BATCH_Q_PROJ] Q output (local) first 10: [" << q_local->data()[0] << ", "
                                                                       << q_local->data()[1] << ", " << q_local->data()[2] << ", " << q_local->data()[3] << ", "
                                                                       << q_local->data()[4] << ", " << q_local->data()[5] << ", " << q_local->data()[6] << ", "
                                                                       << q_local->data()[7] << ", " << q_local->data()[8] << ", " << q_local->data()[9] << "]");
                float q_min = *std::min_element(q_local->data(), q_local->data() + q_local->size());
                float q_max = *std::max_element(q_local->data(), q_local->data() + q_local->size());
                LOG_INFO("[BATCH_Q_PROJ] Q output stats: min=" << q_min << " max=" << q_max
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

            if (debugEnv().attention.verbose && rank_ == 0)
            {
                LOG_DEBUG("[MPIAttentionBatch] K weight extraction: wk->size()=" << wk->size()
                                                                                 << " wk->shape()=[" << wk->shape()[0] << "," << wk->shape()[1] << "]"
                                                                                 << " row_offset=" << row_offset << " local_rows=" << local_rows
                                                                                 << " head_offset_=" << head_offset_ << " n_kv_heads_=" << n_kv_heads_ << " n_heads_=" << n_heads_);
                // Check first few values of global wk
                const float *wk_data = wk->data();
                LOG_DEBUG("[MPIAttentionBatch] Global wk first 5 values: "
                          << wk_data[0] << ", " << wk_data[1] << ", " << wk_data[2] << ", " << wk_data[3] << ", " << wk_data[4]);
            }

            // MPILinearBatchOperator expects weights as [out_dim, in_dim]
            auto wk_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows, D});

            // Extract rows - simple memcpy of contiguous data
            size_t offset_elements = static_cast<size_t>(row_offset) * D;
            size_t copy_elements = static_cast<size_t>(local_rows) * D;
            std::memcpy(wk_local->data(), wk->data() + offset_elements, copy_elements * sizeof(float));

            std::shared_ptr<TensorBase> bk_local;
            if (bk && bk->size() > 0)
            {
                bk_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows});
                const float *bk_data = bk->data();
                float *bk_local_data = bk_local->data();
                std::copy(bk_data + row_offset, bk_data + row_offset + local_rows, bk_local_data);
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

            // MPILinearBatchOperator expects weights as [out_dim, in_dim]
            auto wv_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows, D});

            // Extract rows - simple memcpy of contiguous data
            size_t offset_elements = static_cast<size_t>(row_offset) * D;
            size_t copy_elements = static_cast<size_t>(local_rows) * D;
            std::memcpy(wv_local->data(), wv->data() + offset_elements, copy_elements * sizeof(float));

            std::shared_ptr<TensorBase> bv_local;
            if (bv && bv->size() > 0)
            {
                bv_local = std::make_shared<SimpleTensor>(std::vector<int>{local_rows});
                const float *bv_data = bv->data();
                float *bv_local_data = bv_local->data();
                std::copy(bv_data + row_offset, bv_data + row_offset + local_rows, bv_local_data);
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
                LOG_ERROR("[Q_LOCAL_DEBUG] Before gather - Local Q first 10: ["
                          << q_local->data()[0] << ", " << q_local->data()[1] << ", "
                          << q_local->data()[2] << ", " << q_local->data()[3] << ", "
                          << q_local->data()[4] << ", " << q_local->data()[5] << ", "
                          << q_local->data()[6] << ", " << q_local->data()[7] << ", "
                          << q_local->data()[8] << ", " << q_local->data()[9] << "]");
                float q_min = *std::min_element(q_local->data(), q_local->data() + q_local->size());
                float q_max = *std::max_element(q_local->data(), q_local->data() + q_local->size());
                LOG_ERROR("[Q_LOCAL_DEBUG] Local Q stats: size=" << q_local->size()
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
                    LOG_ERROR("[BATCH_GATHER_DEBUG] Q gather parameters:");
                    LOG_ERROR("  Using MPI_Allgather (matching sequential path)");
                    LOG_ERROR("  sendcount=" << q_local_size << " per rank");
                    LOG_ERROR("  Local Q before gather [0:5]: [" << q_local->data()[0] << ", "
                                                                 << q_local->data()[1] << ", " << q_local->data()[2] << ", "
                                                                 << q_local->data()[3] << ", " << q_local->data()[4] << "]");
                }

                // Bulk gather: faster than per-token MPI calls
                MPI_Allgather(q_local->data(), q_local_size, MPI_FLOAT,
                              temp_q->data(), q_local_size, MPI_FLOAT,
                              MPI_COMM_WORLD);

                // Rearrange from rank-major to row-interleaved
                // temp_q layout: [rank0: t0,t1,t2,... | rank1: t0,t1,t2,... | ...]
                // q_snapshot layout: [t0: r0,r1,... | t1: r0,r1,... | ...]
                for (int t = 0; t < T; ++t)
                {
                    for (int r = 0; r < size; ++r)
                    {
                        const float *src = temp_q->data() + r * q_local_size + t * n_heads_local_ * head_dim_;
                        float *dst = q_snapshot->data() + t * n_heads_ * head_dim_ + r * n_heads_local_ * head_dim_;
                        std::memcpy(dst, src, n_heads_local_ * head_dim_ * sizeof(float));
                    }
                }

                // DEBUG: Log gathered result
                if (rank == 0 && current_layer_idx_ == 0)
                {
                    LOG_ERROR("[BATCH_GATHER_DEBUG] After gather and rearrange:");
                    LOG_ERROR("  Total size: " << q_snapshot->size());
                    LOG_ERROR("  First 10: [" << q_snapshot->data()[0] << ", " << q_snapshot->data()[1] << ", "
                                              << q_snapshot->data()[2] << ", " << q_snapshot->data()[3] << ", "
                                              << q_snapshot->data()[4] << ", " << q_snapshot->data()[5] << ", "
                                              << q_snapshot->data()[6] << ", " << q_snapshot->data()[7] << ", "
                                              << q_snapshot->data()[8] << ", " << q_snapshot->data()[9] << "]");
                    LOG_ERROR("  At offset 1792 (rank1 start): [" << q_snapshot->data()[1792] << ", "
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

                // Rearrange from rank-major to row-interleaved
                for (int t = 0; t < T; ++t)
                {
                    for (int r = 0; r < size; ++r)
                    {
                        const float *src = temp_k->data() + r * k_local_size + t * n_kv_heads_local_ * head_dim_;
                        float *dst = k_snapshot->data() + t * n_kv_heads_ * head_dim_ + r * n_kv_heads_local_ * head_dim_;
                        std::memcpy(dst, src, n_kv_heads_local_ * head_dim_ * sizeof(float));
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

                // Rearrange from rank-major to row-interleaved
                for (int t = 0; t < T; ++t)
                {
                    for (int r = 0; r < size; ++r)
                    {
                        const float *src = temp_v->data() + r * v_local_size + t * n_kv_heads_local_ * head_dim_;
                        float *dst = v_snapshot->data() + t * n_kv_heads_ * head_dim_ + r * n_kv_heads_local_ * head_dim_;
                        std::memcpy(dst, src, n_kv_heads_local_ * head_dim_ * sizeof(float));
                    }
                }
            }
            else
            {
                std::copy(v_local->data(), v_local->data() + v_local_size, v_snapshot->data());
            }
            snapshot_callback_(PipelineStage::V_PROJECTION, current_layer_idx_, v_snapshot);
        }

        // Step 2: Reshape to [B, n_heads_local, T, head_dim] and apply RoPE
        // Q: [B, T, n_heads_local * head_dim] -> [B, n_heads_local, T, head_dim]
        // This is a logical reshape, we'll work with the data in-place

        applyRoPE(q_local->data(), k_local->data(), B, T);

        // DEBUG: Check Q/K after RoPE
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

            // First gather Q globally
            auto q_global = std::make_shared<SimpleTensor>(std::vector<int>{B, T, n_heads_ * head_dim_});
            int q_local_size = B * T * n_heads_local_ * head_dim_;

            if (size > 1)
            {
                std::vector<int> recvcounts(size);
                std::vector<int> displs(size);
                int offset = 0;
                for (int r = 0; r < size; ++r)
                {
                    int heads_on_rank = n_heads_ / size + (r < (n_heads_ % size) ? 1 : 0);
                    recvcounts[r] = B * T * heads_on_rank * head_dim_;
                    displs[r] = offset;
                    offset += recvcounts[r];
                }
                MPI_Allgatherv(q_local->data(), q_local_size, MPI_FLOAT,
                               q_global->data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                               MPI_COMM_WORLD);
            }
            else
            {
                std::copy(q_local->data(), q_local->data() + q_local_size, q_global->data());
            }

            // Then gather K globally
            auto k_global = std::make_shared<SimpleTensor>(std::vector<int>{B, T, n_kv_heads_ * head_dim_});
            int k_local_size = B * T * n_kv_heads_local_ * head_dim_;

            if (size > 1)
            {
                std::vector<int> recvcounts(size);
                std::vector<int> displs(size);
                int offset = 0;
                for (int r = 0; r < size; ++r)
                {
                    int kv_heads_on_rank = n_kv_heads_ / size + (r < (n_kv_heads_ % size) ? 1 : 0);
                    recvcounts[r] = B * T * kv_heads_on_rank * head_dim_;
                    displs[r] = offset;
                    offset += recvcounts[r];
                }
                MPI_Allgatherv(k_local->data(), k_local_size, MPI_FLOAT,
                               k_global->data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                               MPI_COMM_WORLD);
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

            for (int b = 0; b < B; ++b)
            {
                for (int t = 0; t < T; ++t)
                {
                    const float *q_src = q_global->data() + (b * T + t) * q_features_global;
                    const float *k_src = k_global->data() + (b * T + t) * k_features_global;
                    float *dst = rope_data + (b * T + t) * total_features;

                    std::copy(q_src, q_src + q_features_global, dst);
                    std::copy(k_src, k_src + k_features_global, dst + k_features_global);
                }
            }

            snapshot_callback_(PipelineStage::ROPE_APPLICATION, current_layer_idx_, rope_snapshot);
        }

        // Step 3: Compute attention scores with per-batch causal masking
        // scores: [B, n_heads_local, T, T]
        int scores_size = B * n_heads_local_ * T * T;
        std::vector<float> scores(scores_size);

        computeAttentionScores(
            q_local->data(),
            k_local->data(),
            scores.data(),
            B,
            T);

        // Step 4: Apply causal mask and softmax per-batch
        applyCausalMaskAndSoftmax(scores.data(), B, T);

        // DEBUG: Check scores after softmax
        if (rank == 0 && debugEnv().attention.verbose)
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
            LOG_DEBUG("[MPIAttentionBatch] After softmax scores: min=" << min_val << " max=" << max_val << " nan_count=" << nan_count);
        }

        // Step 5: Compute attention output: scores @ V
        // attn_output: [B, n_heads_local, T, head_dim]
        auto attn_output_local = std::make_shared<SimpleTensor>(
            std::vector<int>{B, n_heads_local_ * T, head_dim_});

        computeAttentionOutput(
            scores.data(),
            v_local->data(),
            attn_output_local->data(),
            B,
            T);

        // Capture attention context (before output projection)
        // For consistency with sequential pipeline, gather to global tensors
        if (snapshot_callback_)
        {
            int rank = getRank();
            int size = getSize();

            // First transpose local data from [B, n_heads_local, T, head_dim] to [B, T, n_heads_local * head_dim]
            auto context_local = std::make_shared<SimpleTensor>(
                std::vector<int>{B, T, n_heads_local_ * head_dim_});

            const float *attn_data = attn_output_local->data();
            float *context_local_data = context_local->data();

            for (int b = 0; b < B; ++b)
            {
                for (int h = 0; h < n_heads_local_; ++h)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        const float *src = attn_data + (b * n_heads_local_ * T + h * T + t) * head_dim_;
                        float *dst = context_local_data + (b * T + t) * (n_heads_local_ * head_dim_) + h * head_dim_;
                        std::copy(src, src + head_dim_, dst);
                    }
                }
            }

            // Now gather to global tensor [B, T, n_heads * head_dim]
            auto context_snapshot = std::make_shared<SimpleTensor>(
                std::vector<int>{B, T, n_heads_ * head_dim_});
            int context_local_size = B * T * n_heads_local_ * head_dim_;

            if (size > 1)
            {
                std::vector<int> recvcounts(size);
                std::vector<int> displs(size);
                int offset = 0;
                for (int r = 0; r < size; ++r)
                {
                    int heads_on_rank = n_heads_ / size + (r < (n_heads_ % size) ? 1 : 0);
                    recvcounts[r] = B * T * heads_on_rank * head_dim_;
                    displs[r] = offset;
                    offset += recvcounts[r];
                }
                MPI_Allgatherv(context_local_data, context_local_size, MPI_FLOAT,
                               context_snapshot->data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                               MPI_COMM_WORLD);
            }
            else
            {
                std::copy(context_local_data, context_local_data + context_local_size, context_snapshot->data());
            }

            snapshot_callback_(PipelineStage::ATTENTION_CONTEXT, current_layer_idx_, context_snapshot);
        }

        // Step 6: Reshape and concatenate heads
        // attn_output_local: [B, n_heads_local, T, head_dim] -> [B, T, n_heads_local * head_dim]
        auto attn_concat_local = std::make_shared<SimpleTensor>(
            std::vector<int>{B, T, n_heads_local_ * head_dim_});

        {
            const float *attn_data = attn_output_local->data();
            float *concat_data = attn_concat_local->data();

            // Transpose from [B, n_heads_local, T, head_dim] to [B, T, n_heads_local, head_dim]
            for (int b = 0; b < B; ++b)
            {
                for (int h = 0; h < n_heads_local_; ++h)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        const float *src = attn_data + (b * n_heads_local_ * T + h * T + t) * head_dim_;
                        float *dst = concat_data + (b * T + t) * (n_heads_local_ * head_dim_) + h * head_dim_;
                        std::copy(src, src + head_dim_, dst);
                    }
                }
            }
        }

        // Step 7: Output projection with distributed computation
        // Each rank computes partial output: [B*T, n_heads_local*head_dim] @ [n_heads_local*head_dim, D]^T
        // Then MPI_Allreduce sums across ranks to get final [B*T, D] output

        // Extract local column slice of wo for this rank's heads
        // wo shape: [D, n_heads * head_dim] (row-major)
        // We need columns [head_offset*head_dim : (head_offset+n_heads_local)*head_dim]
        // Note: wo_cols already declared in validation section above
        int local_out_cols = n_heads_local_ * head_dim_;
        int col_offset = head_offset_ * head_dim_;

        auto wo_local = std::make_shared<SimpleTensor>(std::vector<int>{D, local_out_cols});
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

        // Reshape attn_concat_local to 2D for matrix multiplication
        int M = B * T;
        int K = local_out_cols;
        int N = D;

        // Compute partial output: [M, K] @ [N, K]^T = [M, N]
        // Note: wo_local is [N, K] stored row-major, so we use CblasTrans
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K,
                    1.0f,
                    attn_concat_local->data(), K,
                    wo_local->data(), K,
                    0.0f,
                    output->data(), N);

        // Step 8: MPI_Allreduce to sum partial outputs across ranks
        if (getSize() > 1)
        {
            MPI_Allreduce(MPI_IN_PLACE, output->data(), output->size(),
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
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
        // Apply RoPE to Q and K tensors
        // Q: [B, T, n_heads_local * head_dim] viewed as [B, n_heads_local, T, head_dim]
        // K: [B, T, n_kv_heads_local * head_dim] viewed as [B, n_kv_heads_local, T, head_dim]

        const int half_dim = head_dim_ / 2;

// Apply to Q
#pragma omp parallel for collapse(3)
        for (int b = 0; b < batch_size; ++b)
        {
            for (int h = 0; h < n_heads_local_; ++h)
            {
                for (int t = 0; t < seq_len; ++t)
                {
                    float *q_pos = q + (b * seq_len + t) * (n_heads_local_ * head_dim_) + h * head_dim_;

                    for (int i = 0; i < half_dim; ++i)
                    {
                        float freq = rope_freqs_[i];
                        float theta = t * freq;
                        float cos_theta = std::cos(theta);
                        float sin_theta = std::sin(theta);

                        float x0 = q_pos[2 * i];
                        float x1 = q_pos[2 * i + 1];

                        q_pos[2 * i] = x0 * cos_theta - x1 * sin_theta;
                        q_pos[2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
                    }
                }
            }
        }

// Apply to K
#pragma omp parallel for collapse(3)
        for (int b = 0; b < batch_size; ++b)
        {
            for (int h = 0; h < n_kv_heads_local_; ++h)
            {
                for (int t = 0; t < seq_len; ++t)
                {
                    float *k_pos = k + (b * seq_len + t) * (n_kv_heads_local_ * head_dim_) + h * head_dim_;

                    for (int i = 0; i < half_dim; ++i)
                    {
                        float freq = rope_freqs_[i];
                        float theta = t * freq;
                        float cos_theta = std::cos(theta);
                        float sin_theta = std::sin(theta);

                        float x0 = k_pos[2 * i];
                        float x1 = k_pos[2 * i + 1];

                        k_pos[2 * i] = x0 * cos_theta - x1 * sin_theta;
                        k_pos[2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
                    }
                }
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
        // Compute Q @ K^T for each batch and head independently
        // Q: [B, T, n_heads_local * head_dim] viewed as [B, n_heads_local, T, head_dim]
        // K: [B, T, n_kv_heads_local * head_dim] viewed as [B, n_kv_heads_local, T, head_dim]
        // scores: [B, n_heads_local, T, T]

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

        // For GQA, we need to handle head groups
        int heads_per_kv = n_heads_ / n_kv_heads_;

        for (int b = 0; b < batch_size; ++b)
        {
            for (int h = 0; h < n_heads_local_; ++h)
            {
                // Determine which KV head this query head uses (for GQA)
                // For tensor parallelism, typically each rank has n_kv_heads_local == 1
                // and all local query heads share that single KV head
                int kv_h = 0; // Simplified: use first local KV head
                if (n_kv_heads_local_ > 1)
                {
                    // If we have multiple KV heads per rank, distribute query heads among them
                    int heads_per_kv_local = (n_heads_local_ + n_kv_heads_local_ - 1) / n_kv_heads_local_;
                    kv_h = h / heads_per_kv_local;
                    if (kv_h >= n_kv_heads_local_)
                        kv_h = n_kv_heads_local_ - 1;
                }

                // Get pointers for this batch and head
                const float *q_head = q + (b * seq_len) * (n_heads_local_ * head_dim_) + h * head_dim_;
                const float *k_head = k + (b * seq_len) * (n_kv_heads_local_ * head_dim_) + kv_h * head_dim_;
                float *scores_head = scores + (b * n_heads_local_ + h) * seq_len * seq_len;

                // Compute Q @ K^T: [T, head_dim] @ [head_dim, T] = [T, T]
                // Q layout: q_head[t * (n_heads_local * head_dim) + 0..head_dim-1]
                // K layout: k_head[t * (n_kv_heads_local * head_dim) + 0..head_dim-1]

                int q_stride = n_heads_local_ * head_dim_;
                int k_stride = n_kv_heads_local_ * head_dim_;

                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len,     // M: rows of Q
                    seq_len,     // N: cols of K^T (rows of K)
                    head_dim_,   // K: cols of Q, cols of K^T
                    scale,       // alpha: scale by 1/sqrt(head_dim)
                    q_head,      // A: Q
                    q_stride,    // lda: stride includes all heads
                    k_head,      // B: K
                    k_stride,    // ldb: stride includes all KV heads
                    0.0f,        // beta
                    scores_head, // C: scores
                    seq_len      // ldc
                );
            }
        }
    }

    void MPIAttentionBatchOperator::applyCausalMaskAndSoftmax(
        float *scores,
        int batch_size,
        int seq_len)
    {
        // Apply causal mask and softmax independently per batch and head
        // scores: [B, n_heads_local, T, T]

        const float NEG_INF = -1e9f;

#pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b)
        {
            for (int h = 0; h < n_heads_local_; ++h)
            {
                float *scores_head = scores + (b * n_heads_local_ + h) * seq_len * seq_len;

                // Apply causal mask: position i can only attend to positions <= i
                for (int i = 0; i < seq_len; ++i)
                {
                    // Apply mask to future positions
                    for (int j = i + 1; j < seq_len; ++j)
                    {
                        scores_head[i * seq_len + j] = NEG_INF;
                    }

                    // Compute softmax for this row
                    float *row = scores_head + i * seq_len;

                    // Find max (only over valid positions 0..i)
                    float max_val = row[0];
                    for (int j = 1; j <= i; ++j)
                    {
                        max_val = std::max(max_val, row[j]);
                    }

                    // Compute exp and sum
                    float sum = 0.0f;
                    for (int j = 0; j <= i; ++j)
                    {
                        row[j] = std::exp(row[j] - max_val);
                        sum += row[j];
                    }

                    // Normalize (and zero out future positions)
                    for (int j = 0; j <= i; ++j)
                    {
                        row[j] /= sum;
                    }
                    for (int j = i + 1; j < seq_len; ++j)
                    {
                        row[j] = 0.0f;
                    }
                }
            }
        }
    }

    void MPIAttentionBatchOperator::computeAttentionOutput(
        const float *scores,
        const float *v,
        float *output,
        int batch_size,
        int seq_len)
    {
        // Compute scores @ V for each batch and head
        // scores: [B, n_heads_local, T, T]
        // V: [B, T, n_kv_heads_local * head_dim] viewed as [B, n_kv_heads_local, T, head_dim]
        // output: [B, n_heads_local, T, head_dim]

        int heads_per_kv = n_heads_ / n_kv_heads_;

        for (int b = 0; b < batch_size; ++b)
        {
            for (int h = 0; h < n_heads_local_; ++h)
            {
                // Map local query head to local KV head (simplified for tensor parallelism)
                int kv_h = 0;
                if (n_kv_heads_local_ > 1)
                {
                    int heads_per_kv_local = (n_heads_local_ + n_kv_heads_local_ - 1) / n_kv_heads_local_;
                    kv_h = h / heads_per_kv_local;
                    if (kv_h >= n_kv_heads_local_)
                        kv_h = n_kv_heads_local_ - 1;
                }

                const float *scores_head = scores + (b * n_heads_local_ + h) * seq_len * seq_len;
                const float *v_head = v + (b * seq_len) * (n_kv_heads_local_ * head_dim_) + kv_h * head_dim_;
                float *out_head = output + (b * n_heads_local_ + h) * seq_len * head_dim_;

                int v_stride = n_kv_heads_local_ * head_dim_;

                // scores @ V: [T, T] @ [T, head_dim] = [T, head_dim]
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len,     // M: rows of scores
                    head_dim_,   // N: cols of V
                    seq_len,     // K: cols of scores, rows of V
                    1.0f,        // alpha
                    scores_head, // A: scores
                    seq_len,     // lda
                    v_head,      // B: V
                    v_stride,    // ldb
                    0.0f,        // beta
                    out_head,    // C: output
                    head_dim_    // ldc
                );
            }
        }
    }

} // namespace llaminar
