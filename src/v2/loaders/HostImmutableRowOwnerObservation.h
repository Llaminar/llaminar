/**
 * @file HostImmutableRowOwnerObservation.h
 * @brief Read-only point-in-time observation of a native host owner's live PMA leases.
 *
 * Only actual source/cache/service owners issue this value after ready/source validation.
 * Bytes come from their retained leases, never a planned BOM, geometry or global
 * claimed-byte delta. This is neither an admission permit nor a backing-storage
 * pin: consumers must retain the provider/owner while reusing its storage. The
 * const authority handle preserves identity without exposing allocation methods.
 */
#pragma once
#include "ImmutableRowTableHostPlacement.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <cstddef>
#include <memory>
#include <utility>

namespace llaminar2
{
class ImmutableRowTable;
class CpuAdaptiveImmutableRowCache;
namespace qwen4exp { class ImmutableLookupService; }

/** @brief One validated owner snapshot; subsequent retirement may release these bytes. */
class HostImmutableRowOwnerObservation final
{
public:
    /** @return The exact existing canonical PMA, retained as a read-only identity. */
    const std::shared_ptr<const PhysicalMemoryAuthority> &authority() const noexcept { return authority_; }
    /** @return Exact host allocator; NUMA locality is a separate policy. */
    PhysicalMemoryAllocatorIdentity allocator() const noexcept { return allocator_; }
    /** @return Actual source/cache host locality; SSD page-cache placement remains Linux-owned. */
    ImmutableRowTableHostPlacement hostPlacement() const noexcept { return host_placement_; }
    /** @return Live ImmutableTableStorage leases, including private SSD service and optional native RAM cache. */
    std::size_t tableStorageBytes() const noexcept { return table_storage_bytes_; }
    /** @return Live ImmutableLookupStaging leases owned by this provider, excluding caller responses; a source-service observation also includes its retained channel lease. */
    std::size_t lookupStagingBytes() const noexcept { return lookup_staging_bytes_; }

private:
    friend class ImmutableRowTable;
    friend class CpuAdaptiveImmutableRowCache;
    friend class qwen4exp::ImmutableLookupService;
    /**
     * @brief Only validated live owners construct observations from actual leases.
     * @param authority Existing source/cache PMA; this value grants no mutable authority.
     * @param allocator Exact allocator that issued those leases.
     * @param placement Actual immutable host locality.
     * @param storage Checked sum of actual ImmutableTableStorage lease bytes.
     * @param staging Actual ImmutableLookupStaging lease bytes, zero for whole tables.
     */
    HostImmutableRowOwnerObservation(std::shared_ptr<const PhysicalMemoryAuthority> authority,
        PhysicalMemoryAllocatorIdentity allocator, ImmutableRowTableHostPlacement placement,
        std::size_t storage, std::size_t staging)
        : authority_(std::move(authority)), allocator_(allocator), host_placement_(placement),
          table_storage_bytes_(storage), lookup_staging_bytes_(staging) {}
    std::shared_ptr<const PhysicalMemoryAuthority> authority_;
    PhysicalMemoryAllocatorIdentity allocator_;
    ImmutableRowTableHostPlacement host_placement_;
    std::size_t table_storage_bytes_, lookup_staging_bytes_;
};
} // namespace llaminar2
