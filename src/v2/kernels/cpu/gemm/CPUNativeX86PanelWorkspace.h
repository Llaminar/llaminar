/**
 * @file CPUNativeX86PanelWorkspace.h
 * @brief Exact retained slab quote for decode-once x86 NativeQuantised prefill.
 *
 * Each worker borrows one short increasing-K panel from the participant's
 * existing projection bank. Only the original coded weight publication is
 * persistent. Sharing the slab with an invocation's ordered-partial bank is
 * safe because the two projection routes are mutually exclusive.
 *
 * One invocation's demand (invocationFloats) is not monotone in team width:
 * the panel needs one N strip per worker, so a team wider than the strip count
 * cannot select it while a narrower team can. The admitted quote
 * (workspaceFloats) is therefore the maximum demand over every width from one
 * to the admitted worker count; planning and the runtime borrow both call it.
 */
#pragma once

#include "CPUNativeQuantisedPreparedFootprint.h"

#include <algorithm>
#include <concepts>
#include <limits>
#include <stdexcept>

namespace llaminar2::cpu::native_quantised::x86_panel
{
    enum class Format
    {
        None = -1,
        IQ4XS = 4,
        Q4K = 5,
        Q50 = 6,
        Q5K = 7,
        IQ3S = 11
    };
    /// Start sharing a panel across at least eight row pairs. Storage quotes apply
    /// it to the physical capacity; dispatch additionally requires as many live rows.
    inline constexpr int min_rows = 16;
    inline constexpr int depth = 8;     ///< At most eight source blocks in one worker's L1 tile.

    /** @return Supported panel format selected from the admitted physical encoding and source codebook. */
    constexpr Format formatFor(CPUNativeQuantisedEncoding encoding, int codebook) noexcept
    {
        if (encoding == CPUNativeQuantisedEncoding::Q4KSuperblockCoded && codebook == 5)
            return Format::Q4K;
        if (encoding == CPUNativeQuantisedEncoding::CompactMultiScale)
        {
            if (codebook == 7)
                return Format::Q5K;
            if (codebook == 11)
                return Format::IQ3S;
        }
        if (encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale && codebook == 4)
            return Format::IQ4XS;
        if (encoding == CPUNativeQuantisedEncoding::Q5NativeSingleScale && codebook == 6)
            return Format::Q50;
        return Format::None;
    }

    /** @return Whether this panel carries a minimum term as well as a scale. */
    constexpr bool affine(Format format) noexcept
    {
        return format == Format::Q4K || format == Format::Q5K;
    }

    /** @return Slab elements for eight dot groups and the exact factor planes of one K block. */
    constexpr int blockFloats(Format format, int lanes) noexcept
    {
        return (9 + affine(format)) * lanes;
    }

    /** @return Whether a noninitial arithmetic partial can outlive its first physical K panel. */
    constexpr bool needsCarry(int blocks, int partitions) noexcept
    {
        return partitions > 1 && (blocks - 1) / partitions + 1 > depth;
    }

    /**
     * @brief Quote one cache-line-separated worker slot, in elements of the FP32 slab.
     * @param format Admitted panel format.
     * @param lanes Physical SIMD columns per worker.
     * @param rows Physical invocation rows (the carry is addressed by original row).
     * @param blocks Original logical K-block count.
     * @param partitions Exact serial arithmetic partition count, or at most one for contiguous K.
     * @return Space for the panel and, only when needed, an unfinished partial per row.
     */
    inline std::size_t workerFloats(Format format, int lanes, int rows, int blocks, int partitions)
    {
        constexpr auto max = std::numeric_limits<std::size_t>::max() / sizeof(float);
        const auto panel = static_cast<std::size_t>(std::min(blocks, depth) * blockFloats(format, lanes));
        if (needsCarry(blocks, partitions) && static_cast<std::size_t>(rows) > (max - panel - 15) / lanes)
            throw std::overflow_error("Native x86 prefill carry exceeds addressable storage");
        const auto carry = needsCarry(blocks, partitions) ? static_cast<std::size_t>(rows) * lanes : 0;
        return (panel + carry + 15) / 16 * 16;
    }

    /** @return Number of output strips including one strip for any nonempty tail. */
    constexpr int columnStrips(int columns, int lanes) noexcept
    {
        return (columns - 1) / lanes + 1; // Admitted positive N and SIMD width.
    }

    /** @return Whether a physical bucket can keep every admitted worker on a distinct N strip. */
    constexpr bool supportsGeometry(Format format, int lanes, int rows, int columns, int workers) noexcept
    {
        return format != Format::None && rows >= min_rows && (lanes == 8 || lanes == 16) &&
            columns > 0 && workers > 0 && columnStrips(columns, lanes) >= workers;
    }

    /**
     * @brief Exact slab one invocation borrows at its own team width.
     * @param format Admitted panel format.
     * @param lanes Physical SIMD columns per worker.
     * @param rows Physical invocation rows.
     * @param columns Physical output columns.
     * @param blocks Original logical K-block count.
     * @param partitions Serial K partition count selected at this team width.
     * @param workers Exact team width of the invocation.
     * @return One slot per worker; zero if Auto cannot select the panel at this width.
     * @throws std::invalid_argument For non-positive geometry.
     * @throws std::overflow_error For unaddressable storage.
     */
    inline std::size_t invocationFloats(Format format, int lanes, int rows, int columns, int blocks,
        int partitions, int workers)
    {
        if ((lanes != 8 && lanes != 16) || columns <= 0 || blocks <= 0 || partitions < 0 || workers <= 0)
            throw std::invalid_argument("Native x86 prefill panel requires positive admitted geometry");
        if (!supportsGeometry(format, lanes, rows, columns, workers))
            return 0;
        const auto slot = workerFloats(format, lanes, rows, blocks, partitions);
        if (static_cast<std::size_t>(workers) > std::numeric_limits<std::size_t>::max() / sizeof(float) / slot)
            throw std::overflow_error("Native x86 prefill panel exceeds admitted worker storage");
        return slot * static_cast<std::size_t>(workers);
    }

    /**
     * @brief Admitted slab quote covering every team width that can execute the panel.
     * @tparam PartitionsAt Callable mapping a team width to its serial K partition count.
     * @param format Admitted panel format.
     * @param lanes Physical SIMD columns per worker.
     * @param rows Physical invocation rows.
     * @param columns Physical output columns.
     * @param blocks Original logical K-block count.
     * @param workers Admitted worker count of the participant.
     * @param partitions_at The production serial tile policy evaluated at one width.
     * @return max over w in [1, workers] of invocationFloats(..., partitions_at(w), w).
     * @throws std::invalid_argument For non-positive geometry.
     * @throws std::overflow_error For unaddressable storage.
     *
     * Non-decreasing in @p workers, so a team narrower than the admitted width
     * (a serialized retained team, or a runtime team below the participant's
     * admission) never borrows more than was admitted. Only widths with one N
     * strip per worker can select the panel; the carry, and so the slot,
     * depends on the width through the serial partition count. Two partitions
     * give the largest slot of any count, which bounds the remaining widths.
     */
    template <class PartitionsAt>
        requires std::invocable<PartitionsAt &, int>
    std::size_t workspaceFloats(Format format, int lanes, int rows, int columns, int blocks, int workers,
        PartitionsAt &&partitions_at)
    {
        if ((lanes != 8 && lanes != 16) || columns <= 0 || blocks <= 0 || workers <= 0)
            throw std::invalid_argument("Native x86 prefill panel requires positive admitted geometry");
        if (!supportsGeometry(format, lanes, rows, columns, 1))
            return 0;
        const auto largest_slot = workerFloats(format, lanes, rows, blocks, 2);
        std::size_t quote = 0;
        for (int width = std::min(workers, columnStrips(columns, lanes)); width >= 1; --width)
        {
            if (quote / static_cast<std::size_t>(width) >= largest_slot)
                break; // No narrower team can exceed the quote.
            quote = std::max(quote, invocationFloats(format, lanes, rows, columns, blocks,
                static_cast<int>(partitions_at(width)), width));
        }
        return quote;
    }
} // namespace llaminar2::cpu::native_quantised::x86_panel
