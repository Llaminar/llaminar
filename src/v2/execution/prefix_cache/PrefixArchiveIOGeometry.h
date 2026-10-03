/**
 * @file PrefixArchiveIOGeometry.h
 * @brief Canonical bounded-memory contract for durable prefix archive I/O.
 *
 * Disk verification and background compaction use disjoint halves of one
 * reusable host buffer. Both the memory planner and implementation consume this
 * definition so admission cannot price one scratch geometry while runtime
 * silently allocates another.
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    /** @brief Exact persistent host scratch retained by one shared archive. */
    struct PrefixArchiveIOGeometry final
    {
        /**
         * @brief Return the bounded transfer/checksum window in bytes.
         *
         * Eight MiB keeps sequential disk I/O economical while remaining
         * independent of model size, context length, and serialized block
         * size.  Large records are folded through this window incrementally.
         */
        [[nodiscard]] static constexpr std::size_t scratchBytes() noexcept
        {
            return 8u * 1024u * 1024u;
        }

        /** @return Foreground checksum window; never shared with maintenance. */
        [[nodiscard]] static constexpr std::size_t verificationBytes() noexcept
        {
            return scratchBytes() / 2u;
        }

        /** @return Background copy window in the same admitted allocation. */
        [[nodiscard]] static constexpr std::size_t compactionBytes() noexcept
        {
            return scratchBytes() - verificationBytes();
        }
    };
} // namespace llaminar2
