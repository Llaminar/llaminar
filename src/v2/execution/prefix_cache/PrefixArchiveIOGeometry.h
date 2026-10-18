/**
 * @file PrefixArchiveIOGeometry.h
 * @brief Canonical bounded-memory contract for durable prefix archive I/O.
 *
 * Background metadata compaction uses one reusable host buffer. Payload files
 * are reclaimed by unlink after durable eviction and are never recopied.
 * Foreground hydration
 * reads directly into admitted final owners. The planner and runtime consume this
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
         * @brief Return the bounded background transfer window in bytes.
         *
         * Four MiB keeps sequential disk I/O economical while remaining
         * independent of model size, context length, and serialized block
         * size.  Large records are folded through this window incrementally.
         */
        [[nodiscard]] static constexpr std::size_t scratchBytes() noexcept
        {
            return 4u * 1024u * 1024u;
        }

        /** @return Background copy window in the same admitted allocation. */
        [[nodiscard]] static constexpr std::size_t compactionBytes() noexcept
        {
            return scratchBytes();
        }
    };
} // namespace llaminar2
