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
 *
 * A panel strip is the column span one worker decodes and consumes: eight
 * AVX2 columns, or on AVX-512 a whole 64-column chunk whenever every worker of
 * the team can own one, else sixteen columns. Wide strips let each activation
 * broadcast and each row's per-block scalar work (FP16 scale widening, zero-
 * point correction) serve four vectors instead of one, which the consume needs
 * to match the expanded INT8 kernels' instruction count per dot product;
 * narrower strips keep the panel available to teams wider than N / 64.
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

    /// Columns of a wide AVX-512 strip: one complete prepared 64-column chunk.
    inline constexpr int wide_strip = 64;

    /** @return Number of output strips of width @p strip, including one strip for any nonempty tail. */
    constexpr int columnStrips(int columns, int strip) noexcept
    {
        return (columns - 1) / strip + 1; // Admitted positive N and strip width.
    }

    /**
     * @brief Columns one worker owns per panel strip.
     * @param lanes Physical SIMD width of the invocation (8 AVX2, 16 AVX-512).
     * @param columns Physical output columns.
     * @param workers Exact team width.
     * @return 64 on AVX-512 when every worker can own one 64-column strip, else @p lanes.
     */
    constexpr int stripColumns(int lanes, int columns, int workers) noexcept
    {
        return lanes == 16 && columnStrips(columns, wide_strip) >= workers ? wide_strip : lanes;
    }

    /** @return Row-tile height that keeps a strip's integer and FP32 accumulators in registers. */
    constexpr int rowTile(int strip) noexcept
    {
        return strip == 16 ? 4 : 2;
    }

    /** @return Slab elements for eight dot groups and the exact factor planes of one K block of @p strip columns. */
    constexpr int blockFloats(Format format, int strip) noexcept
    {
        return (9 + affine(format)) * strip;
    }

    /** @return Whether a noninitial arithmetic partial can outlive its first physical K panel. */
    constexpr bool needsCarry(int blocks, int partitions) noexcept
    {
        return partitions > 1 && (blocks - 1) / partitions + 1 > depth;
    }

    /**
     * @brief Quote one cache-line-separated worker slot, in elements of the FP32 slab.
     * @param format Admitted panel format.
     * @param strip Columns of the worker's strip (stripColumns()).
     * @param rows Physical invocation rows (the carry is addressed by original row).
     * @param blocks Original logical K-block count.
     * @param partitions Exact serial arithmetic partition count, or at most one for contiguous K.
     * @return Space for the panel and, only when needed, an unfinished partial per row.
     */
    inline std::size_t workerFloats(Format format, int strip, int rows, int blocks, int partitions)
    {
        constexpr auto max = std::numeric_limits<std::size_t>::max() / sizeof(float);
        const auto panel = static_cast<std::size_t>(std::min(blocks, depth) * blockFloats(format, strip));
        if (needsCarry(blocks, partitions) && static_cast<std::size_t>(rows) > (max - panel - 15) / strip)
            throw std::overflow_error("Native x86 prefill carry exceeds addressable storage");
        const auto carry = needsCarry(blocks, partitions) ? static_cast<std::size_t>(rows) * strip : 0;
        return (panel + carry + 15) / 16 * 16;
    }

    /** @return Whether a physical bucket can keep every admitted worker on a distinct N strip of at least @p lanes columns. */
    constexpr bool supportsGeometry(Format format, int lanes, int rows, int columns, int workers) noexcept
    {
        return format != Format::None && rows >= min_rows && (lanes == 8 || lanes == 16) &&
            columns > 0 && workers > 0 && columnStrips(columns, lanes) >= workers;
    }

    /**
     * @brief Exact slab one invocation borrows at its own team width.
     * @param format Admitted panel format.
     * @param lanes Physical SIMD width (8 AVX2, 16 AVX-512); the strip width follows stripColumns().
     * @param rows Physical invocation rows.
     * @param columns Physical output columns.
     * @param blocks Original logical K-block count.
     * @param partitions Serial K partition count selected at this team width.
     * @param workers Exact team width of the invocation.
     * @return One slot of stripColumns() columns per worker; zero if Auto cannot select the panel at this width.
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
        const auto slot = workerFloats(format, stripColumns(lanes, columns, workers), rows, blocks, partitions);
        if (static_cast<std::size_t>(workers) > std::numeric_limits<std::size_t>::max() / sizeof(float) / slot)
            throw std::overflow_error("Native x86 prefill panel exceeds admitted worker storage");
        return slot * static_cast<std::size_t>(workers);
    }

    /**
     * @brief Admitted slab quote covering every team width that can execute the panel.
     * @tparam PartitionsAt Callable mapping a team width to its serial K partition count.
     * @param format Admitted panel format.
     * @param lanes Physical SIMD width (8 AVX2, 16 AVX-512); the strip width follows stripColumns().
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
     * depends on the width through the serial partition count and the strip
     * width. Two partitions of the widest strip give the largest slot of any
     * width, which bounds the remaining widths.
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
        const auto largest_slot = workerFloats(format, lanes == 16 ? wide_strip : lanes, rows, blocks, 2);
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
