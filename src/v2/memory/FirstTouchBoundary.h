#pragma once

/**
 * @file FirstTouchBoundary.h
 * @brief Execute first-touch stores under one temporary thread placement.
 *
 * The allocator owns the page range, revocation and certification. This
 * interface owns only the calling thread's affinity and memory policy from
 * the first attempted bind through the last restoration.
 */

#include <cstddef>
#include <limits>
#include <sched.h>

namespace llaminar2
{
    /** @brief Kernel possible-node extent for the complete policy mask. */
    struct FirstTouchGeometry
    {
        int max_node = -1; ///< Inclusive maximum possible node ID, not a target selector.
    };

    /**
     * @brief Native operations for one synchronous calling-thread transaction.
     * Policy callbacks take the raw Linux maxnode operand, including its extra bit.
     */
    struct FirstTouchBoundaryOps
    {
        void *const context = nullptr; ///< Borrowed context used only by this call.
        int (*const getThreadAffinity)(void *, cpu_set_t *) = nullptr;
        int (*const setThreadAffinity)(void *, const cpu_set_t *) = nullptr;
        int (*const getMemoryPolicy)(void *, int *, unsigned long *, unsigned long, void *, int) = nullptr;
        int (*const setMemoryPolicy)(void *, int, const unsigned long *, unsigned long) = nullptr;
        int (*const nodeCpuSet)(void *, int, cpu_set_t *) = nullptr;
        void (*const touchRange)(void *, volatile void *, std::size_t, std::size_t) = nullptr;
    };

    /** @brief Operation owning a recoverable result or the primary fatal outcome. */
    enum class FirstTouchPhase
    {
        Precondition,
        CaptureAffinity,
        CapturePolicy,
        ResolveNodeCpus,
        BindAffinity,
        BindPolicy,
        Touch,
        Completed
    };

    /** @return Stable name of the operation for diagnostics. */
    [[nodiscard]] const char *firstTouchPhaseName(FirstTouchPhase phase) noexcept;

    /** @brief Recoverable outcome, returned only on an unmutated or restored thread. */
    struct FirstTouchPopulateResult
    {
        /** @brief Bind the outcome to its actual phase and capture errno by value. */
        constexpr FirstTouchPopulateResult(FirstTouchPhase observed_phase, int observed_errno) noexcept
            : completed(observed_phase == FirstTouchPhase::Completed),
              phase(observed_phase), failure_errno(observed_errno) {}

        const bool completed;
        const FirstTouchPhase phase;
        const int failure_errno;
    };

    inline constexpr auto kFirstTouchMaskWordBits = std::numeric_limits<unsigned long>::digits;

    /** @brief Size a native mask for every bit through the inclusive maximum. */
    [[nodiscard]] constexpr std::size_t firstTouchMaskWordCount(int max_node) noexcept
    {
        return max_node < 0 ? 0 : static_cast<std::size_t>(max_node) / kFirstTouchMaskWordBits + 1;
    }

    /** @brief Set an in-range node bit; leave the mask unchanged on rejection. */
    [[nodiscard]] constexpr bool firstTouchMaskSetNode(
        unsigned long *mask, std::size_t words, int node) noexcept
    {
        const auto extent = firstTouchMaskWordCount(node);
        if (!mask || extent == 0 || extent > words)
            return false;
        mask[extent - 1] |= 1ul << (node % kFirstTouchMaskWordBits);
        return true;
    }

    /** @brief Query a node bit without reading beyond the supplied native mask. */
    [[nodiscard]] constexpr bool firstTouchMaskHasNode(
        const unsigned long *mask, std::size_t words, int node) noexcept
    {
        const auto extent = firstTouchMaskWordCount(node);
        return mask && extent != 0 && extent <= words &&
            (mask[extent - 1] & (1ul << (node % kFirstTouchMaskWordBits))) != 0;
    }

    /** @brief Return true only when all supplied words contain no node bits. */
    [[nodiscard]] constexpr bool firstTouchMaskEmpty(const unsigned long *mask, std::size_t words) noexcept
    {
        if (!mask)
            return words == 0;
        for (std::size_t word = 0; word < words; ++word)
            if (mask[word] != 0)
                return false;
        return true;
    }

    /** @brief Compare complete native masks, including bits above the first word. */
    [[nodiscard]] constexpr bool firstTouchMasksEqual(
        const unsigned long *left, const unsigned long *right, std::size_t words) noexcept
    {
        if (!left || !right)
            return words == 0;
        for (std::size_t word = 0; word < words; ++word)
            if (left[word] != right[word])
                return false;
        return true;
    }

    /** @return Immutable Linux adapter used by the allocator and native call-through tests. */
    [[nodiscard]] const FirstTouchBoundaryOps &firstTouchDefaultOps() noexcept;

    /**
     * @brief Bind the calling thread, touch the supplied extent and restore the original settings.
     * @param base Start of the caller's live writable extent.
     * @param bytes Exact extent, a nonzero multiple of page_bytes.
     * @param page_bytes Store stride supplied by the caller's page geometry.
     * @param numa_node Exact target selected by the allocator.
     * @param geometry Kernel possible-node extent used only to size masks.
     * @param ops Complete immutable operation binding. Native status operations return 0 or a failure with errno.
     * @return The completed phase, or a recoverable rejection with its saved errno.
     * @throws Allocation/callback exceptions; after the first bind attempt they escape only after verified restoration.
     * @warning Either failed restore, failed readback, or a readback mismatch terminates the process.
     */
    [[nodiscard]] FirstTouchPopulateResult firstTouchPopulateRange(
        volatile void *base, std::size_t bytes, std::size_t page_bytes,
        int numa_node, FirstTouchGeometry geometry, const FirstTouchBoundaryOps &ops);
}
