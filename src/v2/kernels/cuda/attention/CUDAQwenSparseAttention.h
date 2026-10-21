/**
 * @file CUDAQwenSparseAttention.h
 * @brief Explicit-stream CUDA launch boundary for retained Qwen sparse graphs.
 *
 * Bindings refer to persistent arena/cache storage. Launchers perform no
 * allocation, transfer, synchronization, or host inspection of replay counts.
 */
#pragma once
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include <cuda_runtime.h>
namespace llaminar2::qwen4exp
{
/** @brief Enqueue complete sparse-indexer preparation and deterministic selection.
 * @param params Persistent device bindings and immutable admitted geometry.
 * @param stream Exact non-default CUDA producer/consumer stream.
 * @throws std::invalid_argument For invalid bindings/stream; std::runtime_error on native launch failure.
 */
void cudaQwenSparseIndexer(const QwenSparseIndexer &params, cudaStream_t stream);
/**
 * @brief Select immutable stored FP32 scores with an explicit private candidate.
 * @param params Original device replay, admitted score/output banks and selection policy.
 * @param stream Exact non-default producer/consumer stream, retained by the caller.
 * @throws std::invalid_argument For invalid policy, bounds, aliases or stream.
 * @throws std::runtime_error For an immediate native launch failure.
 * @details No host value access, allocation, score arithmetic or synchronization.
 */
void cudaQwenSparseSelectPools(const QwenSparseIndexer &params, cudaStream_t stream);
/** @brief Enqueue native selected-KV attention on its exact producer/consumer stream.
 * @param params Native device banks, canonical mapping, and exclusive fixed-partition scratch.
 * @param stream Exact non-default CUDA producer/consumer stream.
 * @throws std::invalid_argument For invalid bindings/stream; std::runtime_error on native launch failure.
 */
void cudaQwenSparseAttention(const QwenSparseAttention &params, cudaStream_t stream);
/** @brief Enqueue accepted raw-tail publication without changing the canonical frontier.
 * @param params Device-owned acceptance decision and persistent partial-key publication.
 * @param stream Exact non-default CUDA producer/consumer stream.
 * @throws std::invalid_argument For invalid bindings/stream; std::runtime_error on native launch failure.
 */
void cudaQwenSparseCommitPartial(const QwenSparsePartialCommit &params, cudaStream_t stream);
}
