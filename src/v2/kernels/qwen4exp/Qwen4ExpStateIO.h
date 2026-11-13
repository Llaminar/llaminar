/**
 * @file Qwen4ExpStateIO.h
 * @brief Bounded native endpoint-state copies with execution-owned live extents.
 *
 * Checkpoints reserve a maximum payload, but kernels copy only live raw-tail
 * rows and publish their count. Append-only pooled banks are absent. Fixed PLE
 * convolution history includes mathematical zero-padding and is all live state.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include "kernels/common/NativeCounter64.h"
namespace llaminar2::qwen4exp
{
/** @brief Sole row-cardinality authority for a native state transfer. */
enum class QwenStateRowSource : std::uint8_t { Fixed, CanonicalModuloFour, CheckpointHeader, ExecutionCount };
/** @brief Exact native state copy; all addresses belong to one execution device. */
struct QwenStateTransfer
{
    const void *source;
    void *destination;
    std::size_t rowBytes;
    int capacityRows;
    QwenStateRowSource rowSource;
    const std::int32_t *count;
    int fixedRows;
    std::int32_t *publishedRows;
    std::uint64_t *copiedBytes = nullptr; ///< Optional exact live byte observation, including zero-row copies.
    NativeCounter64 aggregateCopiedBytes; ///< Optional original terminal counter pair, atomically accumulated on GPU.
};
/** @brief Native reset/promotion writes preserve integer and float payload bits. */
struct QwenStateFill
{
    void *destination;
    std::size_t words;
    std::uint32_t bits;
};
/** @brief Validate immutable native copy geometry without touching device state. */
void validateQwenStateTransfer(const QwenStateTransfer &params);
/** @brief Validate immutable fill geometry without touching device state. */
void validateQwenStateFill(const QwenStateFill &params);
/** @brief Copy only authoritative CPU live rows, optionally publishing checkpoint row metadata. */
void cpuQwenStateTransfer(const QwenStateTransfer &params);
/** @brief Reset CPU state with exact native bits. */
void cpuQwenStateFill(const QwenStateFill &params);
/** @brief Copy CUDA state on the exact non-default stream using the live device count. */
void cudaQwenStateTransfer(const QwenStateTransfer &params,void *stream);
/** @brief Reset CUDA state on the exact non-default stream. */
void cudaQwenStateFill(const QwenStateFill &params,void *stream);
/** @brief Copy HIP state on the exact non-default stream using the live device count. */
void rocmQwenStateTransfer(const QwenStateTransfer &params,void *stream);
/** @brief Reset HIP state on the exact non-default stream. */
void rocmQwenStateFill(const QwenStateFill &params,void *stream);
}
