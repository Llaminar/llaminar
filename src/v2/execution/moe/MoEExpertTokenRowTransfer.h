/**
 * @file MoEExpertTokenRowTransfer.h
 * @brief Model-light sparse token-row transfer helpers for MoE expert overlays.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class FP32Tensor;
    class TensorBase;

    enum class MoEExpertTransferMode
    {
        Auto,
        None,
        DenseFullSequence,
        SparseTokenRows,
        DecodeOneToken,
    };

    const char *toString(MoEExpertTransferMode mode);

    struct MoEExpertTransferVolume
    {
        MoEExpertTransferMode mode = MoEExpertTransferMode::None;
        size_t dense_rows = 0;
        size_t selected_rows = 0;
        size_t hidden_row_bytes = 0;
        size_t routing_row_bytes = 0;
        size_t dense_outbound_bytes = 0;
        size_t dense_return_bytes = 0;
        size_t sparse_outbound_bytes = 0;
        size_t sparse_return_bytes = 0;
        size_t outbound_bytes = 0;
        size_t return_bytes = 0;

        size_t totalBytes() const { return outbound_bytes + return_bytes; }
        size_t denseTotalBytes() const { return dense_outbound_bytes + dense_return_bytes; }
        size_t sparseTotalBytes() const { return sparse_outbound_bytes + sparse_return_bytes; }
    };

    struct MoEExpertSparseTransferBuffers
    {
        std::shared_ptr<FP32Tensor> hidden;
        std::shared_ptr<FP32Tensor> routing_indices;
        std::shared_ptr<FP32Tensor> routing_weights;
        std::shared_ptr<FP32Tensor> output;
    };

    class MoEExpertTokenRowTransfer
    {
    public:
        static MoEExpertTransferVolume estimateVolume(
            int seq_len,
            int top_k,
            int d_model,
            size_t selected_rows,
            MoEExpertTransferMode mode);

        static std::vector<std::string> validateTokenRows(
            const std::vector<int> &token_rows,
            int seq_len);

        static MoEExpertSparseTransferBuffers allocateBuffers(
            size_t row_count,
            int top_k,
            int d_model);

        static bool ensureBuffers(
            MoEExpertSparseTransferBuffers *buffers,
            size_t row_capacity,
            int top_k,
            int d_model);

        static bool gatherRows(
            const TensorBase *hidden,
            const TensorBase *routing_indices,
            const TensorBase *routing_weights,
            const std::vector<int> &token_rows,
            int seq_len,
            int top_k,
            int d_model,
            MoEExpertSparseTransferBuffers *buffers,
            MoEExpertTransferVolume *volume = nullptr,
            MoEExpertTransferMode mode = MoEExpertTransferMode::SparseTokenRows);

        static bool scatterAddRows(
            const TensorBase *sparse_output,
            const std::vector<int> &token_rows,
            TensorBase *full_output,
            int full_seq_len,
            int d_model);
    };

} // namespace llaminar2