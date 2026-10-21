/**
 * @file QwenSparseReplay.h
 * @brief Pre-append execution-owner metadata publication for captured sparse attention.
 *
 * A view names the canonical cache cursor/count before any candidate append.
 * The explicit cursor encoding distinguishes CPU oldest-row metadata from GPU
 * next-write metadata. Absolute positions and live rows come from the request
 * authority, independently of cache length. No post-append horizon is guessed.
 */
#pragma once
#include "QwenSparseAttention.h"

namespace llaminar2::qwen4exp
{
/** @brief Exact meaning of the canonical physical ring cursor. */
enum class QwenSparseRingCursor : std::uint8_t { OldestRow, NextWriteRow };
/** @brief Immutable borrowed pre-transaction ring authority on the execution device. */
struct QwenSparsePreAppendSequenceView
{
    const std::int32_t *cursor = nullptr;
    const std::int32_t *cached_tokens = nullptr;
    int capacity = 0;
    QwenSparseRingCursor encoding = QwenSparseRingCursor::NextWriteRow;
};
/** @brief Complete device-/host-owned transaction publication with fixed admitted storage. */
struct QwenSparsePrepareReplay
{
    QwenSparsePreAppendSequenceView sequence;
    const std::int32_t *live_rows = nullptr; ///< Request-authority live count; null explicitly selects all row_capacity rows.
    const std::int32_t *position_ids = nullptr; ///< Absolute positions [row_capacity], or null for explicit fixed contiguous policy.
    QwenSparseReplay *replay = nullptr; ///< Shared immutable transaction snapshot, published before append.
    attention::AttentionDeviceParams *row_parameters = nullptr; ///< Required GPU causal horizon/ring records; CPU may omit unused GPU metadata.
    int row_capacity = 0;
    int first_position = 0; ///< Explicit immutable first position when position_ids is null.
};
/** @brief Validate pointer, cursor-policy and capture geometry without reading live device state. */
void validateQwenSparsePrepareReplay(const QwenSparsePrepareReplay &params);
/** @brief Publish CPU transaction state from authoritative host cache/request metadata. */
void cpuQwenSparsePrepareReplay(const QwenSparsePrepareReplay &params);
/** @brief Publish CUDA transaction state on the exact non-default stream, without synchronization. */
void cudaQwenSparsePrepareReplay(const QwenSparsePrepareReplay &params,void *stream);
/** @brief Publish ROCm transaction state on the exact non-default stream, without synchronization. */
void rocmQwenSparsePrepareReplay(const QwenSparsePrepareReplay &params,void *stream);
} // namespace llaminar2::qwen4exp
