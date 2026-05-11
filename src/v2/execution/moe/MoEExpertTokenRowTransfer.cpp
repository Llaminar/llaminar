/**
 * @file MoEExpertTokenRowTransfer.cpp
 * @brief Model-light sparse token-row transfer helpers for MoE expert overlays.
 */

#include "execution/moe/MoEExpertTokenRowTransfer.h"

#include "tensors/Tensors.h"
#include "utils/Logger.h"

#include <algorithm>
#include <limits>
#include <string>

namespace llaminar2
{
namespace
{
    bool validateFP32Tensor(const TensorBase *tensor, const char *name)
    {
        if (!tensor)
        {
            LOG_ERROR("[MoEExpertTokenRowTransfer] Null " << name << " tensor");
            return false;
        }
        if (tensor->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[MoEExpertTokenRowTransfer] " << name << " tensor must be FP32");
            return false;
        }
        return true;
    }

    bool validateMatrixTensor(
        const TensorBase *tensor,
        const char *name,
        size_t rows,
        size_t cols)
    {
        if (!validateFP32Tensor(tensor, name))
            return false;

        const auto &shape = tensor->shape();
        if (shape.size() != 2 || shape[0] < rows || shape[1] != cols)
        {
            LOG_ERROR("[MoEExpertTokenRowTransfer] " << name << " shape mismatch: expected ["
                      << rows << " or more, " << cols << "]");
            return false;
        }
        return true;
    }

    bool logRowErrors(const std::vector<std::string> &errors)
    {
        for (const auto &error : errors)
            LOG_ERROR("[MoEExpertTokenRowTransfer] " << error);
        return errors.empty();
    }

} // namespace

    const char *toString(MoEExpertTransferMode mode)
    {
        switch (mode)
        {
        case MoEExpertTransferMode::Auto:
            return "Auto";
        case MoEExpertTransferMode::None:
            return "None";
        case MoEExpertTransferMode::DenseFullSequence:
            return "DenseFullSequence";
        case MoEExpertTransferMode::SparseTokenRows:
            return "SparseTokenRows";
        case MoEExpertTransferMode::DecodeOneToken:
            return "DecodeOneToken";
        }
        return "Unknown";
    }

    MoEExpertTransferVolume MoEExpertTokenRowTransfer::estimateVolume(
        int seq_len,
        int top_k,
        int d_model,
        size_t selected_rows,
        MoEExpertTransferMode mode)
    {
        MoEExpertTransferVolume volume;
        volume.mode = mode;
        volume.dense_rows = seq_len > 0 ? static_cast<size_t>(seq_len) : 0;
        volume.selected_rows = selected_rows;
        volume.hidden_row_bytes = d_model > 0 ? static_cast<size_t>(d_model) * sizeof(float) : 0;
        volume.routing_row_bytes = top_k > 0 ? static_cast<size_t>(top_k) * 2u * sizeof(float) : 0;

        volume.dense_outbound_bytes = volume.dense_rows * (volume.hidden_row_bytes + volume.routing_row_bytes);
        volume.dense_return_bytes = volume.dense_rows * volume.hidden_row_bytes;
        volume.sparse_outbound_bytes = volume.selected_rows * (volume.hidden_row_bytes + volume.routing_row_bytes);
        volume.sparse_return_bytes = volume.selected_rows * volume.hidden_row_bytes;

        switch (mode)
        {
        case MoEExpertTransferMode::DenseFullSequence:
            volume.outbound_bytes = volume.dense_outbound_bytes;
            volume.return_bytes = volume.dense_return_bytes;
            break;
        case MoEExpertTransferMode::SparseTokenRows:
        case MoEExpertTransferMode::DecodeOneToken:
            volume.outbound_bytes = volume.sparse_outbound_bytes;
            volume.return_bytes = volume.sparse_return_bytes;
            break;
        case MoEExpertTransferMode::Auto:
        case MoEExpertTransferMode::None:
            volume.outbound_bytes = 0;
            volume.return_bytes = 0;
            break;
        }

        return volume;
    }

    std::vector<std::string> MoEExpertTokenRowTransfer::validateTokenRows(
        const std::vector<int> &token_rows,
        int seq_len)
    {
        std::vector<std::string> errors;
        if (seq_len <= 0)
        {
            errors.push_back("seq_len must be positive");
            return errors;
        }

        std::vector<unsigned char> seen(static_cast<size_t>(seq_len), 0);
        for (int token_row : token_rows)
        {
            if (token_row < 0 || token_row >= seq_len)
            {
                errors.push_back("token row " + std::to_string(token_row) +
                                 " is outside [0, " + std::to_string(seq_len) + ")");
                continue;
            }

            auto &was_seen = seen[static_cast<size_t>(token_row)];
            if (was_seen)
                errors.push_back("duplicate token row " + std::to_string(token_row));
            was_seen = 1;
        }

        return errors;
    }

    MoEExpertSparseTransferBuffers MoEExpertTokenRowTransfer::allocateBuffers(
        size_t row_count,
        int top_k,
        int d_model)
    {
        MoEExpertSparseTransferBuffers buffers;
        buffers.hidden = std::make_shared<FP32Tensor>(
            std::vector<size_t>{row_count, static_cast<size_t>(d_model)});
        buffers.routing_indices = std::make_shared<FP32Tensor>(
            std::vector<size_t>{row_count, static_cast<size_t>(top_k)});
        buffers.routing_weights = std::make_shared<FP32Tensor>(
            std::vector<size_t>{row_count, static_cast<size_t>(top_k)});
        buffers.output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{row_count, static_cast<size_t>(d_model)});
        return buffers;
    }

    bool MoEExpertTokenRowTransfer::ensureBuffers(
        MoEExpertSparseTransferBuffers *buffers,
        size_t row_capacity,
        int top_k,
        int d_model)
    {
        if (!buffers)
        {
            LOG_ERROR("[MoEExpertTokenRowTransfer] Null output buffer bundle");
            return false;
        }
        if (top_k <= 0 || d_model <= 0)
        {
            LOG_ERROR("[MoEExpertTokenRowTransfer] Invalid dimensions top_k=" << top_k
                      << " d_model=" << d_model);
            return false;
        }

        auto has_capacity = [&](const std::shared_ptr<FP32Tensor> &tensor,
                                size_t rows,
                                size_t cols) {
            if (!tensor)
                return false;
            const auto &shape = tensor->shape();
            return shape.size() == 2 && shape[0] >= rows && shape[1] == cols;
        };

        const size_t top_k_cols = static_cast<size_t>(top_k);
        const size_t hidden_cols = static_cast<size_t>(d_model);
        if (has_capacity(buffers->hidden, row_capacity, hidden_cols) &&
            has_capacity(buffers->routing_indices, row_capacity, top_k_cols) &&
            has_capacity(buffers->routing_weights, row_capacity, top_k_cols) &&
            has_capacity(buffers->output, row_capacity, hidden_cols))
        {
            return true;
        }

        *buffers = allocateBuffers(row_capacity, top_k, d_model);
        return true;
    }

    bool MoEExpertTokenRowTransfer::gatherRows(
        const TensorBase *hidden,
        const TensorBase *routing_indices,
        const TensorBase *routing_weights,
        const std::vector<int> &token_rows,
        int seq_len,
        int top_k,
        int d_model,
        MoEExpertSparseTransferBuffers *buffers,
        MoEExpertTransferVolume *volume,
        MoEExpertTransferMode mode)
    {
        if (!buffers)
        {
            LOG_ERROR("[MoEExpertTokenRowTransfer] Null output buffer bundle");
            return false;
        }
        if (top_k <= 0 || d_model <= 0)
        {
            LOG_ERROR("[MoEExpertTokenRowTransfer] Invalid dimensions top_k=" << top_k
                      << " d_model=" << d_model);
            return false;
        }
        if (!logRowErrors(validateTokenRows(token_rows, seq_len)))
            return false;
        if (token_rows.empty())
        {
            LOG_ERROR("[MoEExpertTokenRowTransfer] Cannot gather an empty sparse row set");
            return false;
        }

        if (!validateMatrixTensor(hidden, "hidden", static_cast<size_t>(seq_len), static_cast<size_t>(d_model)) ||
            !validateMatrixTensor(routing_indices, "routing_indices", static_cast<size_t>(seq_len), static_cast<size_t>(top_k)) ||
            !validateMatrixTensor(routing_weights, "routing_weights", static_cast<size_t>(seq_len), static_cast<size_t>(top_k)))
        {
            return false;
        }

        if (!ensureBuffers(buffers, token_rows.size(), top_k, d_model))
            return false;
        float *compact_hidden = buffers->hidden->mutable_data();
        float *compact_indices = buffers->routing_indices->mutable_data();
        float *compact_weights = buffers->routing_weights->mutable_data();
        const float *dense_hidden = hidden->data();
        const float *dense_indices = routing_indices->data();
        const float *dense_weights = routing_weights->data();

        for (size_t sparse_row = 0; sparse_row < token_rows.size(); ++sparse_row)
        {
            const size_t dense_row = static_cast<size_t>(token_rows[sparse_row]);
            std::copy_n(dense_hidden + dense_row * static_cast<size_t>(d_model),
                        static_cast<size_t>(d_model),
                        compact_hidden + sparse_row * static_cast<size_t>(d_model));
            std::copy_n(dense_indices + dense_row * static_cast<size_t>(top_k),
                        static_cast<size_t>(top_k),
                        compact_indices + sparse_row * static_cast<size_t>(top_k));
            std::copy_n(dense_weights + dense_row * static_cast<size_t>(top_k),
                        static_cast<size_t>(top_k),
                        compact_weights + sparse_row * static_cast<size_t>(top_k));
        }

        if (volume)
        {
            *volume = estimateVolume(seq_len, top_k, d_model, token_rows.size(), mode);
        }
        return true;
    }

    bool MoEExpertTokenRowTransfer::scatterAddRows(
        const TensorBase *sparse_output,
        const std::vector<int> &token_rows,
        TensorBase *full_output,
        int full_seq_len,
        int d_model)
    {
        if (d_model <= 0)
        {
            LOG_ERROR("[MoEExpertTokenRowTransfer] Invalid d_model=" << d_model);
            return false;
        }
        if (!logRowErrors(validateTokenRows(token_rows, full_seq_len)))
            return false;
        if (token_rows.empty())
            return true;

        if (!validateMatrixTensor(sparse_output,
                                  "sparse_output",
                                  token_rows.size(),
                                  static_cast<size_t>(d_model)) ||
            !validateMatrixTensor(full_output,
                                  "full_output",
                                  static_cast<size_t>(full_seq_len),
                                  static_cast<size_t>(d_model)))
        {
            return false;
        }

        const float *sparse = sparse_output->data();
        float *full = full_output->mutable_data();
        for (size_t sparse_row = 0; sparse_row < token_rows.size(); ++sparse_row)
        {
            float *dst = full + static_cast<size_t>(token_rows[sparse_row]) * static_cast<size_t>(d_model);
            const float *src = sparse + sparse_row * static_cast<size_t>(d_model);
            for (int dim = 0; dim < d_model; ++dim)
                dst[dim] += src[dim];
        }

        return true;
    }

} // namespace llaminar2