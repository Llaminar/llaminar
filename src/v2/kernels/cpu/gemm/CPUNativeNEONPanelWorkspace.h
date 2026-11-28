/**
 * @file CPUNativeNEONPanelWorkspace.h
 * @brief Wire-independent scratch geometry for transient native ARM weight panels.
 *
 * The participant's existing CPU projection partial bank owns the storage. One
 * worker borrows one slot at a time. No decoded weights survive a projection.
 * The metadata BOM and execution calculate the same extent from immutable
 * encoding, numerical policy, physical rows and the admitted worker count.
 */
#pragma once

#include "CPUNativeQuantisedPreparedFootprint.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2::cpu::native_quantised::neon_panel
{
    inline constexpr int min_rows = 16; ///< Smallest measured dense reuse domain.
    /**
     * Smallest rows of one ordinary-prefill member of a pooled multi-projection
     * launch (the batched MoE expert route) that streams panels. Many members
     * share one workshare there, so a member needs no column parallelism of
     * its own, and decoding a strip once beats re-decoding it per row tile
     * from two rows up (GB10, IQ3_S and Q5_K expert down projections, both
     * policies). A single row keeps its register tile.
     */
    inline constexpr int min_pooled_rows = 2;
    inline constexpr int columns = 4; ///< Four-column INT8 dot strip.
    inline constexpr int depth = 16; ///< 512 K values; bounds the worker's reusable weight tile.
    inline constexpr int block_bytes = 160; ///< 128 INT8 bytes and up to eight FP32 factor words.
    inline constexpr int panel_floats = depth * block_bytes / sizeof(float);
    inline constexpr int line_floats = 64 / sizeof(float);

    /** @return Whether the resident payload has a measured single-scale packer. */
    [[nodiscard]] constexpr bool supports(CPUNativeQuantisedEncoding encoding, std::uint8_t codebook, bool asymmetric) noexcept
    {
        return (encoding == CPUNativeQuantisedEncoding::CompactMultiScale && (codebook == 7 || codebook == 11)) ||
            (encoding == CPUNativeQuantisedEncoding::Q5NativeSingleScale && !asymmetric);
    }

    /** @return Panel slots the ordinary runtime may consume, including a serialized retained team. */
    [[nodiscard]] constexpr int projectionWorkers(CPUNativeQuantisedEncoding encoding, int workers) noexcept
    { return encoding == CPUNativeQuantisedEncoding::Q5NativeSingleScale ? 1 : workers; }

    /** @return True if a canonical ordered partition outlives one physical K panel. */
    [[nodiscard]] constexpr bool needsCarry(int k, CPUProjectionNumericalPolicy policy) noexcept
    {
        return policy == CPUProjectionNumericalPolicy::GPUAlignedExpert &&
            k > depth * 32 * MoEProjectionNumericalContract::ordered_k_partitions;
    }

    /**
     * @brief Size one worker slot in FP32 units of the existing aligned slab.
     * @param rows Positive physical M.
     * @param k Positive physical K.
     * @param policy Immutable projection arithmetic identity.
     * @return Weight panel, plus an ordered carry only when needed. Padding
     *         keeps the next worker's writes out of this worker's cache line.
     */
    [[nodiscard]] constexpr std::size_t workerFloats(int rows, int k, CPUProjectionNumericalPolicy policy) noexcept
    {
        const std::size_t carry = needsCarry(k, policy) ? std::size_t(rows) * columns : 0;
        return panel_floats + (carry + line_floats - 1) / line_floats * line_floats;
    }

    /**
     * @brief Worker slab extent (FP32 units) of a panel domain starting at @p minimum_rows.
     * @param encoding Prepared execution encoding.
     * @param codebook Source codebook identity.
     * @param asymmetric Whether the source stores an additive minimum.
     * @param rows Positive physical rows (an invocation's rows or a declared envelope).
     * @param k Positive physical K.
     * @param workers Admitted slot count.
     * @param policy Immutable projection arithmetic identity.
     * @param minimum_rows First row count of the panel domain being priced.
     * @return Zero outside the domain; otherwise workers times one worker slot.
     * @throws std::overflow_error For an unaddressable slab.
     */
    [[nodiscard]] inline std::size_t slabFloats(CPUNativeQuantisedEncoding encoding, std::uint8_t codebook,
        bool asymmetric, int rows, int k, int workers, CPUProjectionNumericalPolicy policy, int minimum_rows)
    {
        if (rows < minimum_rows || !supports(encoding, codebook, asymmetric)) return 0;
        const auto slots = static_cast<std::size_t>(workers);
        const auto slot = workerFloats(rows, k, policy);
        if (slots > std::numeric_limits<std::size_t>::max() / sizeof(float) / slot)
            throw std::overflow_error("Native NEON packed panel workspace exceeds addressable storage");
        return slots * slot;
    }

    /** @return Exact slab one dense ordinary-prefill invocation borrows (zero below min_rows). */
    [[nodiscard]] inline std::size_t invocationFloats(CPUNativeQuantisedEncoding encoding, std::uint8_t codebook,
        bool asymmetric, int rows, int k, int workers, CPUProjectionNumericalPolicy policy)
    { return slabFloats(encoding, codebook, asymmetric, rows, k, workers, policy, min_rows); }

    /**
     * @brief Admitted slab envelope a stage declares for a projection of up to @p rows rows.
     * @details The declaration is purpose-agnostic: a stage whose invocations
     * reach min_pooled_rows rows may carry pooled ordinary-prefill members, so
     * the envelope starts there, although a dense invocation first borrows the
     * slab at min_rows. Register-tile routes (decode, verifier) borrow none.
     */
    [[nodiscard]] inline std::size_t workspaceFloats(CPUNativeQuantisedEncoding encoding, std::uint8_t codebook,
        bool asymmetric, int rows, int k, int workers, CPUProjectionNumericalPolicy policy)
    {
        static_assert(min_pooled_rows <= min_rows, "The admitted envelope must cover both panel domains");
        return slabFloats(encoding, codebook, asymmetric, rows, k, workers, policy, min_pooled_rows);
    }
}
