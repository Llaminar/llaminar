/**
 * @file CPUNativeX86PanelWorkspace.h
 * @brief Exact retained slab quote for decode-once x86 NativeQuantised prefill.
 *
 * Each worker borrows one short increasing-K panel from the participant's
 * existing projection bank. Only the original coded weight publication is
 * persistent. Sharing the slab with an invocation's ordered-partial bank is
 * safe because the two projection routes are mutually exclusive.
 */
#pragma once

#include "CPUNativeQuantisedPreparedFootprint.h"

#include <algorithm>
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
    inline constexpr int min_rows = 16; ///< Start sharing a panel across at least eight row pairs.
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

    /** @return Checked shared slab extent for every worker; no panel for noneligible formats or M. */
    inline std::size_t workspaceFloats(Format format, int lanes, int rows, int blocks, int partitions, int workers)
    {
        if (format == Format::None || rows < min_rows)
            return 0;
        if ((lanes != 8 && lanes != 16) || blocks <= 0 || partitions < 0 || workers <= 0)
            throw std::invalid_argument("Native x86 prefill panel requires positive admitted geometry");
        const auto slot = workerFloats(format, lanes, rows, blocks, partitions);
        if (static_cast<std::size_t>(workers) > std::numeric_limits<std::size_t>::max() / sizeof(float) / slot)
            throw std::overflow_error("Native x86 prefill panel exceeds admitted worker storage");
        return slot * static_cast<std::size_t>(workers);
    }
} // namespace llaminar2::cpu::native_quantised::x86_panel
