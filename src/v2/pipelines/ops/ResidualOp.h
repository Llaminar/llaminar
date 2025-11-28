/**
 * @file ResidualOp.h
 * @brief Self-validating residual connection operation
 *
 * Residual connections are used throughout transformers:
 * - After attention: hidden = residual + attn_output
 * - After FFN: hidden = residual + ffn_output
 *
 * Handles batch padding by zeroing out padding positions to prevent NaN propagation.
 *
 * Usage:
 * @code
 * ResidualOp residual;
 *
 * // Simple residual (no padding handling)
 * TRY_OP(residual(saved_residual, projection_output, current_hidden,
 *                 seq_len, d_model, "layer0_ATTN_RESIDUAL"));
 *
 * // Batched residual with padding mask
 * TRY_OP(residual.batched(saved_residual, projection_output, current_hidden,
 *                         batch_size, padded_seq_len, d_model, sequence_lengths,
 *                         "layer0_FFN_RESIDUAL"));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include <omp.h>

namespace llaminar2
{

    /**
     * @brief Self-validating residual connection operation
     *
     * Replaces the verbose pattern:
     * @code
     * const size_t elements = seq_len * d_model;
     * #pragma omp parallel for
     * for (size_t i = 0; i < elements; ++i) {
     *     output->mutable_data()[i] = residual->data()[i] + projection->data()[i];
     * }
     * CAPTURE_SNAPSHOT("layer0_ATTN_RESIDUAL", output);
     * @endcode
     *
     * With:
     * @code
     * TRY_OP(residual(residual_buf, projection, output, seq_len, d_model, "ATTN_RESIDUAL"));
     * @endcode
     */
    class ResidualOp : public OpBase
    {
    public:
        const char *name() const override { return "ResidualOp"; }

        /**
         * @brief Execute simple residual connection: output = residual + input
         *
         * @param residual Saved residual tensor [rows, cols]
         * @param input Projection output to add [rows, cols]
         * @param output Destination tensor [rows, cols] (can be same as input)
         * @param rows Number of rows
         * @param cols Number of columns
         * @param snapshot_key Snapshot identifier (nullptr to skip)
         *
         * @return true on success, false on validation failure
         */
        bool operator()(
            const TensorBase *residual,
            const TensorBase *input,
            TensorBase *output,
            int rows,
            int cols,
            const char *snapshot_key = nullptr)
        {
            // 1. Validate inputs
            if (!validateTensor(residual, "residual"))
                return false;
            if (!validateTensor(input, "input"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "residual"))
                return false;

            // 2. Execute residual addition
            const size_t elements = static_cast<size_t>(rows) * cols;
            const float *res_data = residual->data();
            const float *in_data = input->data();
            float *out_data = output->mutable_data();

#pragma omp parallel for
            for (size_t i = 0; i < elements; ++i)
            {
                out_data[i] = res_data[i] + in_data[i];
            }

            // Note: Snapshot capture is handled by the calling pipeline
            (void)snapshot_key;

            return true;
        }

        /**
         * @brief Execute batched residual with padding mask
         *
         * Zeros out padding positions to prevent NaN propagation through layers.
         * Critical for batched inference with variable-length sequences.
         *
         * @param residual Saved residual tensor [batch_size * padded_seq_len, cols]
         * @param input Projection output [batch_size * padded_seq_len, cols]
         * @param output Destination tensor [batch_size * padded_seq_len, cols]
         * @param batch_size Number of sequences in batch
         * @param padded_seq_len Maximum sequence length (padded)
         * @param cols Feature dimension (d_model)
         * @param sequence_lengths Actual lengths per sequence [batch_size]
         * @param snapshot_key Snapshot identifier (nullptr to skip)
         *
         * @return true on success, false on validation failure
         */
        bool batched(
            const TensorBase *residual,
            const TensorBase *input,
            TensorBase *output,
            int batch_size,
            int padded_seq_len,
            int cols,
            const std::vector<int> &sequence_lengths,
            const char *snapshot_key = nullptr)
        {
            // 1. Validate inputs
            if (!validateTensor(residual, "residual"))
                return false;
            if (!validateTensor(input, "input"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (batch_size <= 0 || padded_seq_len <= 0 || cols <= 0)
            {
                LOG_ERROR(name() << ": invalid dimensions (batch_size=" << batch_size
                                 << ", padded_seq_len=" << padded_seq_len << ", cols=" << cols << ")");
                return false;
            }
            if (static_cast<int>(sequence_lengths.size()) != batch_size)
            {
                LOG_ERROR(name() << ": sequence_lengths size mismatch (expected " << batch_size
                                 << ", got " << sequence_lengths.size() << ")");
                return false;
            }

            // 2. Execute batched residual with padding mask
            const size_t total_elements = static_cast<size_t>(batch_size) * padded_seq_len * cols;
            const float *res_data = residual->data();
            const float *in_data = input->data();
            float *out_data = output->mutable_data();

#pragma omp parallel for
            for (size_t i = 0; i < total_elements; ++i)
            {
                size_t token_idx = i / cols;
                size_t batch_idx = token_idx / padded_seq_len;
                size_t seq_idx = token_idx % padded_seq_len;

                // Zero out padding positions to prevent NaN propagation
                if (static_cast<int>(seq_idx) >= sequence_lengths[batch_idx])
                {
                    out_data[i] = 0.0f;
                }
                else
                {
                    out_data[i] = res_data[i] + in_data[i];
                }
            }

            // Note: Snapshot capture is handled by the calling pipeline
            (void)snapshot_key;

            return true;
        }
    };

} // namespace llaminar2
