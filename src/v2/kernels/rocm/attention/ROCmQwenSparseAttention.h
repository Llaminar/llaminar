/**
 * @file ROCmQwenSparseAttention.h
 * @brief Explicit-stream HIP launch boundary for retained Qwen sparse graphs.
 *
 * Bindings refer to persistent arena/cache storage. Launchers perform no
 * allocation, transfer, synchronization, or host inspection of replay counts.
 */
#pragma once
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include <hip/hip_runtime.h>
namespace llaminar2::qwen4exp
{
/** @brief Enqueue complete sparse-indexer preparation and deterministic selection.
 * @param params Persistent device bindings and immutable admitted geometry.
 * @param stream Exact non-default HIP producer/consumer stream.
 * @throws std::invalid_argument For invalid bindings/stream; std::runtime_error on native launch failure.
 */
void rocmQwenSparseIndexer(const QwenSparseIndexer &params, hipStream_t stream);
/**
 * @brief Select immutable stored FP32 scores with an explicit private candidate.
 * @param params Original device replay, admitted score/output banks and selection policy.
 * @param stream Exact non-default producer/consumer stream, retained by the caller.
 * @throws std::invalid_argument For invalid policy, bounds, aliases or stream.
 * @throws std::runtime_error For an immediate native launch failure.
 * @details No host value access, allocation, score arithmetic or synchronization.
 */
void rocmQwenSparseSelectPools(const QwenSparseIndexer &params, hipStream_t stream);
/** @brief Enqueue native selected-KV attention on its exact producer/consumer stream.
 * @param params Native device banks, canonical mapping, and exclusive fixed-partition scratch.
 * @param stream Exact non-default HIP producer/consumer stream.
 * @throws std::invalid_argument For invalid bindings/stream; std::runtime_error on native launch failure.
 */
void rocmQwenSparseAttention(const QwenSparseAttention &params, hipStream_t stream);
/**
 * @brief Explicit experimental FP16 tiled attention with FP32 online totals.
 * @param params Native FP16 banks and the sealed eight- or33-partition arena binding.
 * @param stream Exact non-default HIP stream ordering all borrowed storage.
 * @pre Canonical graph admission or the owning native fixture proves naturally
 * aligned storage for row_capacity*query_heads*program_parts partial records.
 * This raw native boundary receives a pointer, not its byte extent; it checks
 * scratch presence while its caller owns the checked program-specific capacity.
 * @throws std::invalid_argument For non-FP16 banks, invalid vector alignment,
 * unsupported rotary width, missing scratch or invalid stream/bindings.
 * @throws std::runtime_error For an immediate native launch failure.
 * @details This entry never substitutes another policy. It admits post-RoPE
 * FP16 keys or the declared64-coordinate pre-RoPE policy, and shares selected
 * KV across four (eight parts) or six (33 parts) global query heads.
 * Each program preserves its arithmetic across serial/grouped rows; graph setup
 * selects the phase program and admits its exact scratch. Default serving is unchanged.
 */
void rocmQwenSparseFP16Attention(const QwenSparseAttention &params, hipStream_t stream);
/** @brief Enqueue accepted raw-tail publication without changing the canonical frontier.
 * @param params Device-owned acceptance decision and persistent partial-key publication.
 * @param stream Exact non-default HIP producer/consumer stream.
 * @throws std::invalid_argument For invalid bindings/stream; std::runtime_error on native launch failure.
 */
void rocmQwenSparseCommitPartial(const QwenSparsePartialCommit &params, hipStream_t stream);
}
