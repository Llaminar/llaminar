/**
 * @file PipelineResidualWidth.h
 * @brief Authenticate PP activation width from the producing graph's tensor contract.
 *
 * Hidden-state transport carries the graph's residual columns, which can differ
 * from token-embedding or GEMM width. The source tensor is the authority; this
 * setup-only helper never guesses a width from model names or backend policy.
 */
#pragma once

#include "../../../tensors/TensorType.h"
#include "../../../collective/PPActivationContract.h"
#include "../engine/PrefillBucketUtils.h"
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

namespace llaminar2
{
/**
 * @brief Resolve an admitted FP32 residual width for one pipeline transfer contract.
 * @param shape Actual producer hidden tensor shape [resident rows,residual columns].
 * @param type Exact activation scalar format, currently production FP32.
 * @return Physical residual columns transferred for each actual consumer row.
 * @throws std::invalid_argument For absent/nonmatrix, empty or unsupported operands.
 */
[[nodiscard]] inline std::size_t pipelineResidualWidth(
    std::span<const std::size_t> shape, TensorType type)
{
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0 ||
        shape[1] > static_cast<std::size_t>(std::numeric_limits<int>::max()) || type != TensorType::FP32)
        throw std::invalid_argument("Pipeline activation requires an admitted FP32 residual matrix");
    return shape[1];
}

/**
 * @brief Resolve the live payload of a padded prefill chunk from its typed schedule.
 * @param contract Producer-authenticated residual columns and admitted row capacity.
 * @param chunk Actual rows and retained launch bucket, kept distinct by the scheduler.
 * @return Live FP32 bytes; an empty participant sends no activation payload.
 * @throws std::invalid_argument For an incoherent schedule or capacity contract.
 */
[[nodiscard]] inline std::size_t pipelinePrefillLiveBytes(
    const PPStageTransferContract &contract, const PrefillChunkPlan &chunk)
{
    if (contract.dtype != ActivationDType::FP32 || contract.embedding_dim == 0 ||
        contract.embedding_dim > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        contract.max_seq_len == 0 || chunk.real_count < 0 ||
        chunk.bucket_seq_len < chunk.real_count || chunk.bucket_seq_len <= 0 ||
        static_cast<std::size_t>(chunk.bucket_seq_len) > contract.max_seq_len)
        throw std::invalid_argument("Pipeline prefill requires distinct live rows inside its admitted bucket");
    const auto rows = static_cast<std::size_t>(chunk.real_count);
    if (rows > std::numeric_limits<std::size_t>::max() / contract.embedding_dim / sizeof(float))
        throw std::overflow_error("Pipeline live residual payload exceeds addressable bytes");
    return contract.activeBytes(chunk.real_count);
}
} // namespace llaminar2
