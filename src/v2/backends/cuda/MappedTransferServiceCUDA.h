/**
 * @file MappedTransferServiceCUDA.h
 * @brief Backend launch boundary for the typed captured CUDA transfer worker.
 *
 * The implementation is compiled beside the packed-weight unit arithmetic and
 * its codebook tables. This keeps one decoder/table authority without device
 * function pointers, separate device linking or a second conversion kernel.
 */
#pragma once

#include "transfer/MappedTransferProgressABI.h"
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    /** Resolve the complete copy/conversion symbol during cold pool setup. */
    [[nodiscard]] bool prepareMappedTransferServiceCUDA() noexcept;

    /**
     * @brief Enqueue one typed service lifetime on an exact CUDA stream.
     * @param commands Immutable mapped command inbox.
     * @param completions GPU-owned release receipts.
     * @param cursors Device-only claims and resumable byte positions.
     * @param capacity Number of physically admitted commands.
     * @param maximum_bytes Per-command extent admitted by TransferEngine.
     * @param wake Captured interval's terminal word; null only for a finite pass.
     * @param run Explicit finite or graph-bounded service lifetime.
     * @param stream Exact, non-null stream already selected by the backend.
     * @return Whether CUDA accepted the launch without an asynchronous error.
     */
    [[nodiscard]] bool launchMappedTransferServiceCUDA(
        const MappedTransferProgressCommand *commands,
        MappedTransferProgressCompletion *completions,
        MappedTransferServiceCursor *cursors, std::size_t capacity,
        std::size_t maximum_bytes, const std::uint64_t *wake,
        MappedTransferServiceRun run, void *stream) noexcept;
} // namespace llaminar2
