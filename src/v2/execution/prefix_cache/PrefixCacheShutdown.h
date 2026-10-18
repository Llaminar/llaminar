/**
 * @file PrefixCacheShutdown.h
 * @brief Terminal, metadata-planned persistence of participant RAM prefixes.
 *
 * All participants select one bounded set for each physical archive before
 * writing any shutdown payload. RAM recency outranks existing disk residency;
 * recency positions are interleaved across participants whose local LRU ticks
 * are not comparable. Only selected owners serialize their existing immutable
 * RAM leases. MPI exchanges metadata with the planner, never KV payloads.
 */
#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include <compare>
#include <cstdint>
#include <memory>
#include <vector>

namespace llaminar2
{
    class DiskPrefixStorageBackend;
    class PrefixStateCache;
    class IMPIContext;

    /** @brief Opaque shared-archive identity stored in its stable lock inode. */
    struct PrefixArchiveIdentity
    {
        uint64_t high = 0;
        uint64_t low = 0;
        /**
         * @brief Order opaque identities for deterministic shared-archive grouping.
         * @param other Another archive's native identity.
         * @return Lexicographic ordering of the two immutable identity components.
         */
        auto operator<=>(const PrefixArchiveIdentity &other) const = default;
        /** @return Whether native archive initialization published an identity. */
        bool valid() const noexcept { return high != 0 || low != 0; }
    };

    /** @brief Exact selected incarnation; a stale same-key disk version is cold. */
    struct PrefixShutdownBlock
    {
        PrefixCacheKey key;
        PrefixPayloadIdentity payload_identity;
        uint64_t bytes = 0;
    };

    /** @brief Archive-bound selection shared by every writer targeting it. */
    struct PrefixShutdownArchivePlan
    {
        PrefixArchiveIdentity archive;
        uint64_t budget_bytes = 0;
        std::vector<PrefixShutdownBlock> retained;
    };

    /** @brief Original RAM leases in most-recent-first order after cache sealing. */
    struct PrefixCacheShutdownSource
    {
        std::shared_ptr<DiskPrefixStorageBackend> archive;
        std::vector<PrefixBlockHandle> most_recent_first;
    };

    /** @brief Completed terminal persistence evidence, never a capacity ledger. */
    struct PrefixCacheShutdownResult
    {
        uint64_t candidates = 0;
        uint64_t selected = 0;
        uint64_t written = 0;
        uint64_t reused = 0;
        uint64_t payload_bytes_written = 0;
    };

    /**
     * @brief Own one terminal flush across a rank's complete runner tree.
     *
     * The runner lifecycle closes request admission before this object seals
     * caches. Selection precedes payload I/O on every rank. The disk authority
     * alone performs capacity eviction and durable publication, protecting all
     * selected incarnations even when another participant writes later.
     */
    class PrefixCacheShutdown final
    {
    public:
        /**
         * @brief Retain one cache; repeated tree references name only one owner.
         * @param cache Participant-local cache, or null for an uninitialized tier.
         * @throws std::logic_error If collection follows terminal persistence.
         */
        void add(std::shared_ptr<PrefixStateCache> cache);

        /**
         * @brief Seal, select and durably persist the highest-priority RAM entries.
         * @param context Optional complete inference communicator; every rank enters.
         * @return Actual local source and native-write evidence.
         * @throws std::exception For stale lifecycle, inconsistent metadata or failed I/O.
         *
         * Empty ranks exchange only framing. The coordinator receives candidate
         * metadata and returns only each recipient's relevant archive selections.
         * There is no post-write collective or full-payload communication.
         */
        PrefixCacheShutdownResult flush(const IMPIContext *context = nullptr);

    private:
        std::vector<std::shared_ptr<PrefixStateCache>> caches_;
        enum class Phase { Collecting, Flushing, Complete };
        Phase phase_ = Phase::Collecting;
    };
}
