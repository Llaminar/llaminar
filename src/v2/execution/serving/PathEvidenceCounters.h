/**
 * @file PathEvidenceCounters.h
 * @brief Path-proving observability counters (AGENTS maxim 26; design §13).
 *
 * PerfStats is the production truth: every intended production path must be
 * PROVABLE as having run by named counters (full graph capture, MTP depth,
 * routing mode, prefix-cache tier, collectives, transfers, segmentation;
 * no GPU D2H except the terminal result). This typed counter set is the
 * substrate: counters are monotonic, named and never coalesced; an
 * integration gate can require a path to have evidence (assertPathRan
 * refuses a zero count with a precise diagnostic).
 */

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /** @brief Named production-path evidence kinds (dual-encoder of the intended path). */
    enum class PathEvidenceKind : std::uint8_t
    {
        FullGraphCapture = 0,
        MTPDynamicDepth,
        RoutedExpertCompute,
        PrefixCacheTierHit,
        NativeCollective,
        DeviceTransfer,
        SegmentationBoundary,
        TerminalHostResultOnly, ///< Only terminal D2H: no per-token host transfer.
        Count,
    };

    /** @brief Monotonic named counters with a path-required assertion gate. */
    struct PathEvidenceCounters
    {
        std::array<std::uint64_t, static_cast<std::size_t>(PathEvidenceKind::Count)> values_{};

        void record(PathEvidenceKind kind, std::uint64_t n = 1)
        {
            if (kind == PathEvidenceKind::Count)
                throw std::logic_error("Count is a sentinel, not a real counter");
            if (n == 0)
                throw std::logic_error("Path evidence increment must be positive");
            auto &v = values_[static_cast<std::size_t>(kind)];
            if (n > std::numeric_limits<std::uint64_t>::max() - v)
                throw std::overflow_error("Path evidence counter would wrap");
            v += n;
        }

        [[nodiscard]] std::uint64_t get(PathEvidenceKind kind) const
        {
            if (kind == PathEvidenceKind::Count)
                throw std::logic_error("Count is a sentinel, not a real counter");
            return values_[static_cast<std::size_t>(kind)];
        }

        /**
         * @brief Require a path to have run (integration gate).
         * @throws std::logic_error When the counter is zero.
         */
        void assertPathRan(PathEvidenceKind kind) const
        {
            if (get(kind) == 0)
                throw std::logic_error("Required production path has no evidence (counter is zero)");
        }
    };
} // namespace llaminar2
