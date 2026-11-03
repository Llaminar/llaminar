/**
 * @file ImmutableRowTableLayoutGeometry.cpp
 * @brief Validate compact range/pool declarations without creating a row directory or live ledger.
 *
 * Work scales with declared endpoints/ranges, never with source row count.
 * No free-memory observation, allocation choice, native byte transformation,
 * pointer capture or runtime fallback is present. Existing whole-tier resolved
 * placements use this same implementation through wholeTier().
 */
#include "ImmutableRowTableLayoutGeometry.h"
#include "collective/LocalTPCollectiveInventory.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <tuple>
namespace llaminar2 {
namespace {
/** @return A concrete tier bit; unknown enum values have no permissive interpretation. */
std::uint8_t bit(ImmutableRowStorage tier) noexcept {
    switch(tier) {
    case ImmutableRowStorage::RAM:return 1u;
    case ImmutableRowStorage::MemoryMapped:return 2u;
    case ImmutableRowStorage::VRAM:return 4u;
    }
    return 0u;
}
/** @brief Require an exact rank/device, including a nonnegative physical CPU ordinal. */
void endpoint(PhysicalMemoryAllocatorIdentity id) {
    if(id.world_rank<0 || !id.valid()) throw std::invalid_argument("Layout requires an exact rank/device endpoint");
}
/** @return Deterministic physical endpoint order, independent of mutable capacity observations. */
auto key(PhysicalMemoryAllocatorIdentity id) {return std::tuple{id.world_rank,id.device.type,id.device.ordinal};}
/** @return Deterministic concrete storage endpoint order. */
auto key(ImmutableRowTableOwnerKey id) {return std::tuple{key(id.allocator),id.tier};}
/** @return Checked sum; required even when eventual native byte capacity cannot fit. */
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>std::numeric_limits<std::uint64_t>::max()-a) throw std::overflow_error("Layout slot/record sum overflows");
    return a+b;
}
/**
 * @brief Validate only the bidirectional edges used by the coordinator/owner star.
 * @param inventory Original immutable discovery matrix, with backend-filtered indices.
 * @param devices Coordinator first, followed by its distinct same-vendor storage owners.
 * @return Uniform None or Complete coverage of every required bidirectional pair.
 * @throws std::invalid_argument For missing matrices or a Partial pair or mixed-route storage star.
 *
 * Owner-to-owner edges are not used by these participant-local graphs. Their
 * peer capabilities cannot change the selected coordinator transport.
 */
PeerAccessCoverage directedStarCoverage(const RankInventory& inventory, const std::vector<DeviceId>& devices)
{
    std::optional<PeerAccessCoverage> result;
    for (std::size_t i = 1; i < devices.size(); ++i) {
        const auto edge = localTPPeerAccessCoverage(inventory, {devices.front(), devices[i]});
        if ((edge != PeerAccessCoverage::None && edge != PeerAccessCoverage::Complete) ||
            (result && *result != edge))
            throw std::invalid_argument("Pooled owner star requires explicit uniform bidirectional peer or NoPeer coverage");
        result = edge;
    }
    if (!result) throw std::invalid_argument("Pooled owner star has no required remote edge");
    return *result;
}
/** @return Original-format native payload floor, without pricing metadata or alignment twice. */
std::size_t payload(std::uint64_t rows,const ImmutableRowTableExtent &extent) {
    if(rows>std::numeric_limits<std::size_t>::max()/extent.rowBytes())
        throw std::overflow_error("Layout native payload bytes overflow");
    return static_cast<std::size_t>(rows)*extent.rowBytes();
}
}
ImmutableRowTableTiers ImmutableRowTableTiers::of(std::span<const ImmutableRowStorage> tiers) {
    std::uint8_t bits=0;
    for(auto tier:tiers) {
        const auto value=bit(tier);
        if(!value || (bits&value)) throw std::invalid_argument("Layout tier set is unknown or duplicated");
        bits|=value;
    }
    if(!bits) throw std::invalid_argument("Layout tier set cannot be empty");
    return ImmutableRowTableTiers(bits);
}
bool ImmutableRowTableTiers::contains(ImmutableRowStorage tier) const noexcept {return (bits_&bit(tier))!=0;}
std::optional<PeerAccessCoverage> ImmutableRowTableDirectedTransportPolicy::observe(
    const AdaptiveGPUCachePlacementRequest& placement, const RankInventory& inventory)
{
    placement.validate();
    if (placement.mode != AdaptiveGPUCachePlacementMode::DeclaredRankLocal) return std::nullopt;
    if (placement.coordinator->world_rank != inventory.rank)
        throw std::invalid_argument("Pooled transport inventory belongs to another original rank");
    std::vector<DeviceId> devices{placement.coordinator->device()};
    for (const auto& owner : placement.owners)
        if (owner.participant != *placement.coordinator && owner.participant.backend == placement.coordinator->backend)
            devices.push_back(owner.participant.device());
    if (devices.size() == 1) return std::nullopt;
    const auto coverage = directedStarCoverage(inventory, devices);
    return coverage;
}
std::vector<ImmutableRowTableReaderPeerAccess> ImmutableRowTableDirectedTransportPolicy::observeReaders(
    PhysicalMemoryAllocatorIdentity coordinator, std::span<const PhysicalMemoryAllocatorIdentity> readers,
    const RankInventory& inventory)
{
    if (!coordinator.valid() || !coordinator.device.is_gpu() || coordinator.world_rank != inventory.rank ||
        std::count(readers.begin(), readers.end(), coordinator) != 1)
        throw std::invalid_argument("Shared reader transport lacks its original coordinator/rank population");
    std::vector<ImmutableRowTableReaderPeerAccess> result;
    std::vector<PhysicalMemoryAllocatorIdentity> seen;
    for (const auto reader : readers) {
        if (!reader.valid() || reader.world_rank != coordinator.world_rank || reader.device.type != coordinator.device.type ||
            std::find(seen.begin(), seen.end(), reader) != seen.end())
            throw std::invalid_argument("Shared reader transport changed its original homogeneous reader population");
        seen.push_back(reader);
        if (reader == coordinator) continue;
        const auto coverage = localTPPeerAccessCoverage(inventory, {coordinator.device, reader.device});
        if (coverage != PeerAccessCoverage::None && coverage != PeerAccessCoverage::Complete)
            throw std::invalid_argument("Shared reader transport requires complete bidirectional peer or explicit NoPeer coverage");
        result.push_back({coordinator, reader, coverage});
    }
    return result;
}
void ImmutableRowTableDirectedTransportPolicy::requireOriginal(PhysicalMemoryAllocatorIdentity coordinator,
    std::span<const PhysicalMemoryAllocatorIdentity> owners,
    const std::shared_ptr<const ClusterInventory>& inventory, std::optional<PeerAccessCoverage> observation)
{
    if (owners.empty()) throw std::invalid_argument("Directed transport requires its complete original owner population");
    std::vector<DeviceId> devices{coordinator.device};
    std::vector<PhysicalMemoryAllocatorIdentity> unique;
    for (const auto owner : owners) {
        require(coordinator, owner, observation);
        if (std::find(unique.begin(), unique.end(), owner) != unique.end())
            throw std::invalid_argument("Directed transport repeats an original physical owner");
        unique.push_back(owner);
        if (owner != coordinator && owner.device.type == coordinator.device.type) {
            devices.push_back(owner.device);
        }
    }
    if (devices.size() == 1) {
        if (observation) throw std::invalid_argument("Cross-vendor transport cannot invent a same-vendor peer observation");
        return;
    }
    if (!inventory || inventory->getRank(coordinator.world_rank).rank != coordinator.world_rank)
        throw std::invalid_argument("Directed transport lacks its retained original rank inventory");
    const auto actual = directedStarCoverage(inventory->getRank(coordinator.world_rank), devices);
    if (actual != observation)
        throw std::invalid_argument("Directed transport changed its exact original observed peer coverage");
}
bool ImmutableRowTableDirectedTransportPolicy::supports(PhysicalMemoryAllocatorIdentity coordinator,
    PhysicalMemoryAllocatorIdentity owner, std::optional<PeerAccessCoverage> observation) noexcept
{
    return coordinator.valid() && owner.valid() && coordinator.device.is_gpu() && owner.device.is_gpu() &&
        coordinator.world_rank == owner.world_rank &&
        (!observation || *observation == PeerAccessCoverage::None || *observation == PeerAccessCoverage::Complete) &&
        (owner == coordinator || owner.device.type != coordinator.device.type || observation == PeerAccessCoverage::None || observation == PeerAccessCoverage::Complete);
}
bool ImmutableRowTableDirectedTransportPolicy::usesPeerPayload(PhysicalMemoryAllocatorIdentity coordinator,
    PhysicalMemoryAllocatorIdentity owner, std::optional<PeerAccessCoverage> observation)
{
    require(coordinator, owner, observation);
    return coordinator != owner && coordinator.device.type == owner.device.type &&
        observation == PeerAccessCoverage::Complete;
}
void ImmutableRowTableDirectedTransportPolicy::require(PhysicalMemoryAllocatorIdentity coordinator,
    PhysicalMemoryAllocatorIdentity owner, std::optional<PeerAccessCoverage> observation)
{
    if (!supports(coordinator, owner, observation))
        throw std::invalid_argument("Pooled transport requires rank-local endpoints and explicit original None/Complete coverage for same-vendor remotes");
}
ImmutableRowTableLayoutGeometry ImmutableRowTableLayoutGeometry::checked(
    ImmutableRowTableExtent extent,ImmutableRowTableTiers tiers,ImmutableRowTableResidency residency,
    std::vector<ImmutableRowTableLayoutOwner> owners,std::vector<PhysicalMemoryAllocatorIdentity> consumers,
    std::vector<ImmutableRowTableFixedRange> ranges,std::vector<ImmutableRowTableSourceService> services,
    std::vector<ImmutableRowTableReplicaGroup> replicas,
    std::optional<PhysicalMemoryAllocatorIdentity> coordinator,std::uint64_t inflight,
    ImmutableRowTableAdaptiveGPUDomain gpu_domain, std::optional<PeerAccessCoverage> peer_access,
    std::vector<ImmutableRowTableReaderPeerAccess> reader_peer_access) {
    if (gpu_domain != ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator &&
        gpu_domain != ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal)
        throw std::invalid_argument("Unknown adaptive GPU domain policy");
    if(residency!=ImmutableRowTableResidency::Fixed && residency!=ImmutableRowTableResidency::Adaptive)
        throw std::invalid_argument("Unknown layout residency policy");
    if(consumers.empty()) throw std::invalid_argument("Lookup domain requires actual consumers");
    std::sort(consumers.begin(),consumers.end(),[](auto a,auto b){return key(a)<key(b);});
    for(auto consumer:consumers) endpoint(consumer);
    if(std::adjacent_find(consumers.begin(),consumers.end())!=consumers.end())
        throw std::invalid_argument("Lookup domain has duplicate consumers");
    std::sort(owners.begin(),owners.end(),[](const auto &a,const auto &b){return key(a.key)<key(b.key);});
    std::uint64_t gpu_slots=0;
    for(std::size_t i=0;i<owners.size();++i) {
        const auto &o=owners[i];endpoint(o.key.allocator);
        if(!tiers.contains(o.key.tier) || o.key.tier==ImmutableRowStorage::MemoryMapped ||
           (o.key.tier==ImmutableRowStorage::VRAM ? !o.key.allocator.device.is_gpu() : !o.key.allocator.device.is_cpu()))
            throw std::invalid_argument("Native owner has disabled tier or incompatible physical resource");
        if(o.key.tier==ImmutableRowStorage::VRAM && !o.host_placement.usesTaskPolicy())
            throw std::invalid_argument("GPU storage owner must not claim host NUMA placement");
        if(i && owners[i-1].key==o.key) throw std::invalid_argument("Duplicate native layout owner");
        if(residency==ImmutableRowTableResidency::Fixed ? o.cache_rows!=0 : o.fixed_rows!=0)
            throw std::invalid_argument("Fixed ranges and adaptive cache pools cannot share location authority");
        const auto rows=add(o.fixed_rows,o.cache_rows);
        if(rows>extent.rows()) throw std::invalid_argument("Owner has redundant source-sized native capacity");
        const auto bytes=payload(rows,extent);
        if(o.physical_byte_limit && bytes>*o.physical_byte_limit)
            throw std::invalid_argument("Native payload alone exceeds explicit owner physical byte limit");
        if(o.key.tier==ImmutableRowStorage::VRAM) gpu_slots=add(gpu_slots,o.cache_rows);
    }
    std::sort(services.begin(),services.end(),[](const auto &a,const auto &b){return key(a.allocator)<key(b.allocator);});
    for(std::size_t i=0;i<services.size();++i) {
        endpoint(services[i].allocator);
        const auto backing=services[i].windows.storage();
        const bool fullRAM=backing==ImmutableRowStorage::RAM && residency==ImmutableRowTableResidency::Adaptive;
        if(!tiers.contains(backing) || !services[i].allocator.device.is_cpu() ||
            (backing!=ImmutableRowStorage::MemoryMapped && !fullRAM) ||
            (i && services[i-1].allocator==services[i].allocator))
            throw std::invalid_argument("Runtime SSD service is disabled, duplicated or has incompatible resource/policy");
    }
    if(tiers.contains(ImmutableRowStorage::MemoryMapped)!=std::any_of(services.begin(),services.end(),[](const auto& service){return service.windows.storage()==ImmutableRowStorage::MemoryMapped;}))
        throw std::invalid_argument("Enabled SSD tier needs an explicit original-source service");
    const auto ownerIndex=[&](ImmutableRowTableOwnerKey id) {
        const auto it=std::find_if(owners.begin(),owners.end(),[&](const auto &o){return o.key==id;});
        if(it==owners.end()) throw std::invalid_argument("Range/replica references an undeclared native owner");
        return static_cast<std::size_t>(it-owners.begin());
    };
    if (peer_access && *peer_access != PeerAccessCoverage::None && *peer_access != PeerAccessCoverage::Complete)
        throw std::invalid_argument("Pooled transport requires explicit complete bidirectional peer or NoPeer coverage");
    if (peer_access && (residency != ImmutableRowTableResidency::Adaptive ||
        gpu_domain != ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator))
        throw std::invalid_argument("Peer observation belongs only to its declared shared GPU-owner cohort");
    if(peer_access){
        const auto original=consumers.front();
        for(const auto consumer:consumers)
            if(!consumer.device.is_gpu() || consumer.world_rank!=original.world_rank || consumer.device.type!=original.device.type)
                throw std::invalid_argument("Directed readers require original homogeneous rank-local GPU membership");
    }
    if (peer_access && std::none_of(owners.begin(), owners.end(), [&](const auto& owner) {
            return owner.key.tier == ImmutableRowStorage::VRAM &&
                std::find(consumers.begin(),consumers.end(),owner.key.allocator)==consumers.end() &&
                owner.key.allocator.device.type == consumers.front().device.type;
        }))
        throw std::invalid_argument("Directed geometry has no declared same-vendor remote storage owner");
    // Reader transport is a separate required star. Do not let physical owner
    // observations authorize a compute-reader edge or silently reorder endpoints.
    std::sort(reader_peer_access.begin(), reader_peer_access.end(),
        [](const auto &a, const auto &b) { return key(a.reader) < key(b.reader); });
    for (std::size_t i = 0; i < reader_peer_access.size(); ++i) {
        const auto &edge = reader_peer_access[i];
        if (gpu_domain != ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator ||
            residency != ImmutableRowTableResidency::Adaptive || edge.reader == edge.coordinator ||
            !edge.coordinator.valid() || !edge.coordinator.device.is_gpu() ||
            (coordinator && *coordinator != edge.coordinator) ||
            std::find(consumers.begin(), consumers.end(), edge.coordinator) == consumers.end() ||
            edge.reader.world_rank != edge.coordinator.world_rank ||
            edge.reader.device.type != edge.coordinator.device.type ||
            (i && reader_peer_access.front().coordinator != edge.coordinator) ||
            std::find(consumers.begin(), consumers.end(), edge.reader) == consumers.end() ||
            (edge.coverage != PeerAccessCoverage::None && edge.coverage != PeerAccessCoverage::Complete) ||
            (i && reader_peer_access[i - 1].reader == edge.reader))
            throw std::invalid_argument("Shared reader peer metadata has an invalid, duplicate or undeclared endpoint");
    }
    if (!reader_peer_access.empty() && reader_peer_access.size() + 1 != consumers.size())
        throw std::invalid_argument("Shared reader peer metadata omits a declared compute reader");
    std::uint64_t records=0;
    if(residency==ImmutableRowTableResidency::Adaptive) {
        if(services.empty())throw std::invalid_argument("Adaptive cache requires one explicit original host backing");
        const bool fullRAM=std::any_of(services.begin(),services.end(),[](const auto& service){return service.windows.storage()==ImmutableRowStorage::RAM;});
        if(fullRAM && (services.size()!=1 || tiers.contains(ImmutableRowStorage::MemoryMapped) ||
            std::any_of(owners.begin(),owners.end(),[](const auto& owner){return owner.key.tier==ImmutableRowStorage::RAM && owner.cache_rows;})))
            throw std::invalid_argument("Full-RAM adaptive source must be unique and cannot duplicate bounded RAM cache payload");
        if(!ranges.empty()) throw std::invalid_argument("Adaptive pools must not have fixed source-home ranges");
        if (gpu_domain == ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal) {
            if (coordinator || !replicas.empty() || !inflight)
                throw std::invalid_argument("Consumer-local caches require independent directories and exact request geometry");
            for (const auto consumer : consumers) {
                if (!consumer.device.is_gpu())
                    throw std::invalid_argument("Consumer-local cache requester must be a GPU");
                const auto owner = std::find_if(owners.begin(), owners.end(), [&](const auto& value) {
                    return value.key.tier == ImmutableRowStorage::VRAM && value.key.allocator == consumer;
                });
                if (owner == owners.end())
                    throw std::invalid_argument("Consumer-local cache omitted an original physical owner");
                if (owner->cache_rows) records = add(records, add(owner->cache_rows, inflight));
            }
            for (const auto& owner : owners)
                if (owner.key.tier == ImmutableRowStorage::VRAM &&
                    std::find(consumers.begin(), consumers.end(), owner.key.allocator) == consumers.end())
                    throw std::invalid_argument("Consumer-local cache contains a storage-only owner");
        } else if(gpu_slots) {
            if(!coordinator || !inflight) throw std::invalid_argument("Adaptive GPU pools require exact coordinator and in-flight geometry");
            endpoint(*coordinator);
            const bool member=std::find(consumers.begin(),consumers.end(),*coordinator)!=consumers.end() ||
                std::any_of(owners.begin(),owners.end(),[&](const auto &o){return o.key.allocator==*coordinator;});
            if(!coordinator->device.is_gpu() || !member) throw std::invalid_argument("GPU coordinator is foreign or not a GPU");
            records=add(gpu_slots,inflight);
        } else if(coordinator) throw std::invalid_argument("Host-only pools must not claim a GPU directory coordinator");
        // Hosted requests still need a bounded live-row envelope, even with
        // zero cache slots. That envelope prices host execution scratch; only
        // an actual GPU residency coordinator creates GPU directory records.
        std::vector<ImmutableRowTableOwnerKey> grouped;
        for(auto &group:replicas) {
            if(group.maximum_copies_per_row<2 || group.maximum_copies_per_row>group.owners.size())
                throw std::invalid_argument("Replica group must explicitly bound at least two eligible copies");
            std::sort(group.owners.begin(),group.owners.end(),[](auto a,auto b){return key(a)<key(b);});
            for(auto owner:group.owners) {
                const auto index=ownerIndex(owner);
                if(!owners[index].cache_rows)
                    throw std::invalid_argument("Replica eligibility requires actual native capacity at every selected endpoint");
                if(std::find(grouped.begin(),grouped.end(),owner)!=grouped.end())
                    throw std::invalid_argument("Replica eligibility is duplicated or overlaps another group");
                grouped.push_back(owner);
            }
        }
    } else {
        if(gpu_domain != ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator || coordinator || inflight || !replicas.empty())
            throw std::invalid_argument("Fixed layout cannot publish adaptive directory or replica-pool policy");
        std::vector<std::uint64_t> occupied(owners.size(),0);
        std::vector<ImmutableRowTableFixedRange> primary;
        for(const auto &range:ranges) {
            if(!range.rows || range.first_row>=extent.rows() || range.rows>extent.rows()-range.first_row)
                throw std::invalid_argument("Fixed native source range is empty or out of bounds");
            if(range.role!=ImmutableRowTableCopyRole::Primary && range.role!=ImmutableRowTableCopyRole::Replica)
                throw std::invalid_argument("Unknown fixed copy role");
            const auto index=ownerIndex(range.owner);occupied[index]=add(occupied[index],range.rows);
            if(range.role==ImmutableRowTableCopyRole::Primary) primary.push_back(range);
        }
        // Source coverage is interval arithmetic, never a table-wide mutable bitmap.
        std::sort(primary.begin(),primary.end(),[](const auto &a,const auto &b){return a.first_row<b.first_row;});
        std::uint64_t frontier=0;
        for(const auto &r:primary) {
            if(r.first_row<frontier) throw std::invalid_argument("Primary source ownership overlaps; extra copies must be explicit replicas");
            if(!tiers.contains(ImmutableRowStorage::MemoryMapped) && r.first_row!=frontier)
                throw std::invalid_argument("No-SSD layout has a native primary coverage gap");
            frontier=r.first_row+r.rows;
        }
        if(!tiers.contains(ImmutableRowStorage::MemoryMapped) && frontier!=extent.rows())
            throw std::invalid_argument("No-SSD layout must cover the complete original source");
        std::sort(ranges.begin(),ranges.end(),[](const auto &a,const auto &b){
            return std::tuple{key(a.owner),a.first_row,a.role}<std::tuple{key(b.owner),b.first_row,b.role};});
        for(std::size_t i=0;i<ranges.size();++i) {
            const auto &r=ranges[i];
            if(i && r.owner==ranges[i-1].owner && r.first_row<ranges[i-1].first_row+ranges[i-1].rows)
                throw std::invalid_argument("One physical owner has overlapping copies of the same source rows");
        }
        for(std::size_t i=0;i<owners.size();++i)
            if(occupied[i]!=owners[i].fixed_rows)
                throw std::invalid_argument("Fixed native capacity must equal exactly declared primary/replica copies");
    }
    return {extent,tiers,residency,std::move(owners),std::move(consumers),std::move(ranges),std::move(services),
            std::move(replicas),coordinator,inflight,records,gpu_domain,peer_access,std::move(reader_peer_access)};
}
ImmutableRowTableLayoutGeometry ImmutableRowTableLayoutGeometry::wholeTier(
    const ImmutableRowTableStoragePolicy &storage,ImmutableRowTableExtent extent,
    std::span<const PhysicalMemoryAllocatorIdentity> owners,std::span<const PhysicalMemoryAllocatorIdentity> consumers,
    ImmutableRowTableHostPlacement host_placement) {
    if(owners.empty()) throw std::invalid_argument("Whole-tier geometry requires exact physical owners");
    const auto tier=storage.storage();
    std::vector<ImmutableRowTableLayoutOwner> payloads;
    std::vector<ImmutableRowTableFixedRange> ranges;
    std::vector<ImmutableRowTableSourceService> services;
    if(tier==ImmutableRowStorage::MemoryMapped) {
        for(auto owner:owners) services.push_back({owner,storage});
    } else {
        for(std::size_t i=0;i<owners.size();++i) {
            const ImmutableRowTableOwnerKey owner{owners[i],tier};
            payloads.push_back({owner,extent.rows(),0,std::nullopt,
                tier==ImmutableRowStorage::RAM?host_placement:ImmutableRowTableHostPlacement::taskPolicy()});
            ranges.push_back({0,extent.rows(),owner,i==0?ImmutableRowTableCopyRole::Primary:ImmutableRowTableCopyRole::Replica});
        }
    }
    const std::array tiers{tier};
    return checked(extent,ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Fixed,std::move(payloads),
        {consumers.begin(),consumers.end()},std::move(ranges),std::move(services));
}
std::size_t ImmutableRowTableLayoutGeometry::nativePayloadBytes(ImmutableRowTableOwnerKey owner) const {
    for(const auto &entry:owners_) if(entry.key==owner) return payload(add(entry.fixed_rows,entry.cache_rows),extent_);
    throw std::invalid_argument("Native payload query names a foreign layout owner");
}
std::optional<PhysicalMemoryAllocatorIdentity> ImmutableRowTableLayoutGeometry::gpuCoordinatorFor(
    PhysicalMemoryAllocatorIdentity requester) const {
    if (std::find(consumers_.begin(), consumers_.end(), requester) == consumers_.end())
        throw std::invalid_argument("GPU directory projection requires an original consumer");
    if (gpu_domain_ == ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator) return gpu_coordinator_;
    const auto owner = std::find_if(owners_.begin(), owners_.end(), [&](const auto& value) {
        return value.key.tier == ImmutableRowStorage::VRAM && value.key.allocator == requester;
    });
    return owner != owners_.end() && owner->cache_rows ? std::optional{requester} : std::nullopt;
}
} // namespace llaminar2
