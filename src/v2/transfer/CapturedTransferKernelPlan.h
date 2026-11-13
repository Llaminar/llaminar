/**
 * @file CapturedTransferKernelPlan.h
 * @brief Capture-time lowering geometry for one complete retained byte message.
 *
 * This is a kernel dispatch policy, not another transfer or memory authority.
 * TransferEngine retains the physical owners and authenticates the binding.
 * Small messages use one cooperative block; large messages preserve a parallel
 * copy grid. Selection depends only on the immutable maximum message extent,
 * never a host read of the device-owned live count.
 */
#pragma once

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Two complete native implementations of the same byte protocol. */
    enum class CapturedTransferKernelPlan { SingleBlock, ParallelCopy };

    /**
     * @brief Bound one-CTA work to 32 bytes per thread in a 256-thread block.
     *
     * Larger admitted messages retain a multi-block copy even when one replay
     * happens to be empty. This keeps launch identity immutable across epochs.
     */
    inline constexpr std::uint64_t kCapturedTransferSingleBlockBytes = 8 * 1024;

    /**
     * @brief Select the complete lowering from capture-frozen message geometry.
     * @param maximum_bytes Positive exact size or counted-message maximum.
     * @return Single cooperative block or acquire followed by parallel copy.
     * @throws std::invalid_argument when no positive admitted extent is supplied.
     */
    [[nodiscard]] constexpr CapturedTransferKernelPlan capturedTransferKernelPlan(
        std::uint64_t maximum_bytes)
    {
        if (maximum_bytes == 0)
            throw std::invalid_argument("Captured transfer lowering requires a positive admitted extent");
        return maximum_bytes <= kCapturedTransferSingleBlockBytes
            ? CapturedTransferKernelPlan::SingleBlock : CapturedTransferKernelPlan::ParallelCopy;
    }

    /**
     * @brief Bound each batched endpoint to 64 bytes per thread in its own CTA.
     *
     * An independent endpoint has its own acquire/barrier/publication, even
     * when several endpoints share a native argument packet. Individual launch
     * geometry retains its separate 32-byte-per-thread bound. Both selectors
     * consume only capture-frozen maximum extents, never a live host count.
     */
    inline constexpr std::uint64_t kCapturedTransferBatchSingleBlockBytes = 16 * 1024;

    /**
     * @brief Choose the complete implementation for one endpoint of a batch.
     * @param maximum_bytes Positive fixed bytes or immutable counted capacity.
     * @return Fused one-CTA acquire/copy/publish, or parallel acquire then copy.
     * @throws std::invalid_argument For zero admitted capacity.
     *
     * Every entry of a packet must select SingleBlock to fuse that packet.
     * One larger capacity selects the parallel implementation for the whole
     * packet, including empty replays; native failure never changes selection.
     */
    [[nodiscard]] constexpr CapturedTransferKernelPlan capturedTransferBatchKernelPlan(
        std::uint64_t maximum_bytes)
    {
        if (maximum_bytes == 0)
            throw std::invalid_argument("Captured transfer batch requires a positive admitted extent");
        return maximum_bytes <= kCapturedTransferBatchSingleBlockBytes
            ? CapturedTransferKernelPlan::SingleBlock : CapturedTransferKernelPlan::ParallelCopy;
    }
}
