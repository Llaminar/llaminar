/**
 * @file PooledImmutableLocalHotReadyRoot.h
 * @brief One fixed mapped LocalReady word with its actual CPU staging lease.
 *
 * The same TransferEngine mapping implementation owns registration, aliases,
 * native teardown and PMA claim lifetime. This wrapper fixes the role and live
 * payload at eight bytes; it cannot select another owner or allocate a second
 * source. The word is device-owned after cold initialization. Observations
 * report only actual retained claim bytes, never its live value or readiness.
 */
#pragma once

#include "loaders/ImmutableLookupNativeOwnership.h"
#include <cstdint>
#include <memory>

namespace llaminar2
{
    class MappedHostTransferRegion;
    class TransferEngine;
    namespace qwen4exp
    {
        /** @brief Exact per-consumer model-lifetime ready publication backing. */
        class PooledImmutableLocalHotReadyRoot final
        {
        public:
            /**
             * @brief Materialize one admitted fixed ready word before capture.
             * @param consumer Original rank/GPU reader, not a storage-owner inference.
             * @param authority Same final PMA as its prepared original GPU context.
             * @return One native mapped owner retaining its actual CPU claim.
             * @throws std::exception Before foreign/unprepared/capture-time materialization,
             *         or on native setup failure; partial native ownership retires first.
             */
            static std::shared_ptr<PooledImmutableLocalHotReadyRoot> create(
                PhysicalMemoryAllocatorIdentity consumer,
                std::shared_ptr<PhysicalMemoryAuthority> authority);
            /**
             * @brief Materialize through the caller's canonical public transfer authority.
             * @param consumer Original rank/GPU reader.
             * @param authority Same final PMA as the original prepared source.
             * @param transfers Public movement authority; recorded backends remain device-free in Unit.
             * @return Exact mapped ready owner, with identical source and claim checks.
             */
            static std::shared_ptr<PooledImmutableLocalHotReadyRoot> create(
                PhysicalMemoryAllocatorIdentity consumer,
                std::shared_ptr<PhysicalMemoryAuthority> authority, const TransferEngine& transfers);
            /** @return Actual immutable mapped-region authority; no live word is observed. */
            const std::shared_ptr<MappedHostTransferRegion>& mappedRegion() const noexcept { return region_; }
            /** @return Original GPU alias consumer, distinct from the physical CPU allocator. */
            PhysicalMemoryAllocatorIdentity consumer() const noexcept { return consumer_; }
            /** @return Actual retained CPU ImmutableLookupStaging claim with zero table bytes. */
            ImmutableLookupNativeOwnerObservation retainedOwnerObservation() const;
            PooledImmutableLocalHotReadyRoot(const PooledImmutableLocalHotReadyRoot&) = delete;
            PooledImmutableLocalHotReadyRoot& operator=(const PooledImmutableLocalHotReadyRoot&) = delete;
        private:
            friend class ::llaminar2::TransferEngine;
            /** @brief Only create() may publish an original successfully mapped owner. */
            PooledImmutableLocalHotReadyRoot(PhysicalMemoryAllocatorIdentity consumer,
                std::shared_ptr<PhysicalMemoryAuthority> authority)
                : consumer_(consumer), authority_(std::move(authority)) {}
            PhysicalMemoryAllocatorIdentity consumer_;
            std::shared_ptr<PhysicalMemoryAuthority> authority_;
            std::shared_ptr<MappedHostTransferRegion> region_;
        };
    }
}
