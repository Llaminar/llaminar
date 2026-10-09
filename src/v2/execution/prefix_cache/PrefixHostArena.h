/**
 * @file PrefixHostArena.h
 * @brief Event-aware byte-range ownership inside an admitted prefix allocation.
 *
 * The physical-memory authority owns admission and the backing allocation's
 * lifetime claim. This arena only places archive ranges inside that allocation;
 * it never allocates additional backing or changes its physical capacity. Cache
 * eviction, outstanding handle aliases, and unfinished DMA are distinct states.
 */
#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace llaminar2
{
    /** @brief Shared, bounded placement authority for one persistent host slab. */
    class PrefixHostArena final : public std::enable_shared_from_this<PrefixHostArena>
    {
    public:
        /**
         * @brief Adopt a fully materialized allocation and its canonical claim.
         * @param backing Shared owner whose address is the first backing byte.
         * @param bytes Exact backing capacity, never a new allocation request.
         * @param memory_claim PMA claim released after the backing owner.
         * @return Arena retained by every subsequently acquired range.
         * @throws std::invalid_argument for an absent or empty backing span.
         */
        static std::shared_ptr<PrefixHostArena> create(
            std::shared_ptr<void> backing, size_t bytes,
            std::shared_ptr<void> memory_claim);

        /** @brief Join remaining retired transfers only at final arena teardown. */
        ~PrefixHostArena();

        /**
         * @brief Lease a contiguous range without allocating physical backing.
         * @param bytes Positive payload extent.
         * @param readiness Exact producer completion retained after the last alias.
         * @return Aliasing owner of the first payload byte, or empty when busy.
         * @throws std::runtime_error when a native completion query fails.
         */
        std::shared_ptr<void> acquire(
            size_t bytes, std::shared_ptr<PrefixPayloadReadiness> readiness = {});

        /**
         * @brief Reap completed transfers and report the largest reusable range.
         * @return Contiguous byte headroom; leased and pending ranges are busy.
         * @throws std::runtime_error when a native completion query fails.
         */
        size_t availableBytes();

        /** @return Sum of reusable ranges; section admission also proves their actual geometry. */
        size_t availableStorageBytes();
        /**
         * @brief Prove the exact section BOM fits the current event-safe free ranges.
         * @param sections Ordered independent ownership extents, including zero-byte sections.
         * @return Whether the same placement algorithm used by acquisition can admit all sections.
         */
        bool canAcquireSections(std::span<const size_t> sections);
        /**
         * @brief Atomically lease independently owned sections in the existing backing.
         * @param sections Required serialized extents in allocation order.
         * @param readiness Common archive producer whose completion protects retired destinations.
         * @return One owner per section, or an empty vector without partial placement on contention.
         * @throws std::runtime_error for a failed exact-event query; metadata allocation errors propagate.
         */
        std::vector<std::shared_ptr<void>> acquireSections(std::span<const size_t> sections,
            std::shared_ptr<PrefixPayloadReadiness> readiness = {});

    private:
        /** @brief Placement lifecycle, independent of logical cache membership. */
        enum class State { Free, Leased, Retired };
        /** @brief An existing map node also stores pending retirement, without allocation. */
        struct Extent
        {
            size_t bytes;
            State state = State::Free;
            std::shared_ptr<PrefixPayloadReadiness> readiness;
        };
        /** @brief Last shared range owner returns placement, never physical memory. */
        struct Lease
        {
            std::shared_ptr<PrefixHostArena> arena;
            size_t offset = 0u;
            /** @brief Preserve the exact readiness edge in the retired map node. */
            ~Lease();
        };

        /** @brief Initialize placement after backing and admission already exist. */
        PrefixHostArena(std::shared_ptr<void> backing, size_t bytes,
                        std::shared_ptr<void> memory_claim);
        /** @brief Nonblocking completion queries and adjacent free-range coalescing. */
        void collectRetired();
        /** @brief Plan independent extents against this CPU-owned placement map while its lock is held. */
        std::optional<std::vector<size_t>> planSections(std::span<const size_t> sections) const;
        /** @brief Transition the last range alias to pending under the placement lock. */
        void retire(size_t offset) noexcept;
        // Reverse destruction frees physical storage before its PMA claim.
        std::shared_ptr<void> memory_claim_;
        std::shared_ptr<void> backing_;
        std::mutex mutex_;
        std::map<size_t, Extent> extents_;
    };
} // namespace llaminar2
