/**
 * @file ResolvedImmutableRowTablePlacement.cpp
 * @brief Validate whole-tier/adaptive serving and explicit fixed-source component ownership.
 * No capacity choice, source payload, allocation or runtime fallback is involved.
 */
#include "ResolvedImmutableRowTablePlacement.h"
#include "CpuAdaptiveImmutableRowCache.h"
#include "AdaptiveImmutableLookupGeometry.h"
#include <algorithm>
#include <array>
#include <stdexcept>
namespace llaminar2 {
ResolvedImmutableRowTablePlacement::ResolvedImmutableRowTablePlacement(
    ImmutableRowTableStoragePolicy storage, ImmutableRowTableExtent extent,
    std::vector<PhysicalMemoryAllocatorIdentity> allocators,
    std::vector<PhysicalMemoryAllocatorIdentity> consumers, ImmutableRowTableHostPlacement locality)
    : value_(WholeTier{storage,extent,std::move(allocators),std::move(consumers),locality})
{
    auto &w=std::get<WholeTier>(value_);
    auto &storage_=w.storage;auto &extent_=w.extent;auto &allocators_=w.allocators;
    auto &consumers_=w.consumers;auto &host_placement_=w.locality;
    if (allocators_.empty())
        throw std::invalid_argument("Resolved immutable row placement requires an exact physical owner");
    std::sort(allocators_.begin(), allocators_.end(), [](const auto &a, const auto &b) {
        return a.world_rank < b.world_rank || (a.world_rank == b.world_rank && a.device < b.device);
    });
    const int rank = allocators_.front().world_rank;
    if (rank < 0 || std::adjacent_find(allocators_.begin(), allocators_.end()) != allocators_.end())
        throw std::invalid_argument("Resolved immutable row placement has duplicate or unbound owners");
    for (const auto &owner : allocators_)
        if (!owner.valid() || owner.world_rank != rank ||
            (storage_.storage() == ImmutableRowStorage::VRAM ? !owner.device.is_gpu() : !owner.device.is_cpu()))
            throw std::invalid_argument("Resolved immutable row placement has foreign rank or incompatible concrete owner");
    std::sort(consumers_.begin(), consumers_.end(), [](const auto &a, const auto &b) {
        return a.world_rank < b.world_rank || (a.world_rank == b.world_rank && a.device < b.device);
    });
    if (consumers_.empty() || std::adjacent_find(consumers_.begin(), consumers_.end()) != consumers_.end())
        throw std::invalid_argument("Resolved immutable row placement requires unique actual PLE consumers");
    for (const auto &consumer : consumers_)
        if (!consumer.valid() || consumer.world_rank != rank ||
            (storage_.storage() == ImmutableRowStorage::VRAM && !consumer.device.is_gpu()))
            throw std::invalid_argument("Resolved immutable row placement has foreign or incompatible PLE consumer");
    if (storage_.storage() == ImmutableRowStorage::VRAM && consumers_ != allocators_)
        throw std::invalid_argument("Resident immutable row storage requires one exact copy per PLE consumer");
    if (storage_.storage() != ImmutableRowStorage::VRAM && allocators_.size() != 1)
        throw std::invalid_argument("Host immutable row storage must have exactly one rank-local owner");
    // Construction validates the common source/layout contract, retaining only
    // the existing authoritative whole-tier fields. There is no second stored
    // geometry, capacity ledger or independently mutable topology mirror.
    (void)ImmutableRowTableLayoutGeometry::wholeTier(storage_, extent_, allocators_, consumers_, host_placement_);
}
ResolvedImmutableRowTablePlacement::ResolvedImmutableRowTablePlacement(
    ImmutableRowTableLayoutGeometry layout,HostAdaptiveRAMSSDRequest requested)
    : value_(HostAdaptive{std::move(layout),requested})
{
    const auto &g=std::get<HostAdaptive>(value_).layout;
    if(g.owners().size()!=1 || g.sourceServices().size()!=1 || g.owners().front().key.tier!=ImmutableRowStorage::RAM ||
       g.owners().front().key.allocator!=g.sourceServices().front().allocator || !g.maximumInflightRows())
        throw std::invalid_argument("Resolved host adaptive placement needs one exact local RAM/SSD owner and maximum live ID envelope");
    const auto &windows=g.sourceServices().front().windows;
    ImmutableRowTablePolicy policy{ImmutableRowStoragePreference::HostAdaptiveRAMSSD,windows.windowBytes(),windows.windowCount(),requested};
    policy.validate();
    const auto geometry=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(g,g.sourceServices().front().allocator,
        static_cast<std::size_t>(g.maximumInflightRows()),requested.policy);
    if(geometry.slots()>requested.maximum_cache_rows || geometry.storageBytes()>requested.cache_physical_byte_limit)
        throw std::invalid_argument("Resolved adaptive storage exceeds its original requested row/physical-byte ceiling");
}
ResolvedImmutableRowTablePlacement::ResolvedImmutableRowTablePlacement(ImmutableRowTableLayoutGeometry layout,
    AdaptiveVRAMRAMSSDRequest requested,std::uint64_t row_elements)
    :value_(GPUAdaptive{std::move(layout),requested,row_elements})
{
    const auto &g=gpuAdaptiveLayout();
    if(g.owners().size()!=2 || g.sourceServices().size()!=1 || g.consumers().size()!=1 ||
        !g.consumers().front().device.is_gpu() || !g.maximumInflightRows() || g.maximumInflightRows()%16)
        throw std::invalid_argument("GPU adaptive placement requires one exact GPU, subordinate RAM/SSD and bounded original IDs");
    const auto &windows=g.sourceServices().front().windows;
    ImmutableRowTablePolicy policy;
    policy.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
    policy.window_bytes=windows.windowBytes();policy.window_count=windows.windowCount();policy.gpu_adaptive=requested;
    policy.validate();
    const auto &e=g.extent();
    const ImmutableRowTableSource source{.rows=e.rows(),.row_elements=row_elements,.format=e.format()};
    const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(g,source,requested);
    if(geometry.physical().owners().front().slots>requested.maximum_gpu_cache_rows ||
        geometry.gpuSlotDependentBytes()>requested.gpu_cache_physical_byte_limit ||
        geometry.host().slots()>requested.host.maximum_cache_rows ||
        geometry.host().storageBytes()>requested.host.cache_physical_byte_limit)
        throw std::invalid_argument("GPU adaptive placement exceeds its original GPU/host slot or physical-byte ceiling");
}
ResolvedImmutableRowTablePlacement::ResolvedImmutableRowTablePlacement(ImmutableRowTableLayoutGeometry layout)
    :value_(FixedHeadShards{std::move(layout)})
{
    const auto &g=std::get<FixedHeadShards>(value_).layout;
    const std::array tiers{ImmutableRowStorage::VRAM};
    if(g.residency()!=ImmutableRowTableResidency::Fixed || g.tiers()!=ImmutableRowTableTiers::of(tiers) ||
        g.owners().empty() || g.owners().size()!=g.consumers().size() || g.fixedRanges().size()!=g.owners().size() ||
        !g.sourceServices().empty() || !g.replicaGroups().empty() || g.gpuCoordinator() || g.maximumInflightRows())
        throw std::invalid_argument("Fixed head shards require exact VRAM primary owners and no other storage/service mode");
    for(const auto &o:g.owners())
    {
        if(o.key.tier!=ImmutableRowStorage::VRAM || !o.fixed_rows || o.cache_rows ||
            std::find(g.consumers().begin(),g.consumers().end(),o.key.allocator)==g.consumers().end() ||
            std::count_if(g.fixedRanges().begin(),g.fixedRanges().end(),[&](const auto &r) {
                return r.owner==o.key && r.role==ImmutableRowTableCopyRole::Primary && r.rows==o.fixed_rows;})!=1)
            throw std::invalid_argument("A fixed head owner must retain exactly one complete declared native primary range");
    }
}
bool ResolvedImmutableRowTablePlacement::isFixedHeadShards() const noexcept {return std::holds_alternative<FixedHeadShards>(value_);}
bool ResolvedImmutableRowTablePlacement::isHostAdaptive() const noexcept {return std::holds_alternative<HostAdaptive>(value_);}
bool ResolvedImmutableRowTablePlacement::isGPUAdaptive() const noexcept {return std::holds_alternative<GPUAdaptive>(value_);}
const ImmutableRowTableStoragePolicy &ResolvedImmutableRowTablePlacement::storagePolicy() const
{
    if(const auto *w=std::get_if<WholeTier>(&value_))return w->storage;
    throw std::logic_error("Non-whole native placement has no whole-table concrete tier");
}
ImmutableRowStorage ResolvedImmutableRowTablePlacement::storage() const {return storagePolicy().storage();}
const ImmutableRowTableHostPlacement &ResolvedImmutableRowTablePlacement::hostPlacement() const
{
    if(isFixedHeadShards())throw std::logic_error("Fixed GPU source ranges have no hosted placement");
    if(const auto *w=std::get_if<WholeTier>(&value_))return w->locality;
    const auto &layout=isGPUAdaptive()?gpuAdaptiveLayout():hostAdaptiveLayout();
    const auto host=layout.sourceServices().front().allocator;
    const auto owner=std::find_if(layout.owners().begin(),layout.owners().end(),[&](const auto &o) {
        return o.key.allocator==host && o.key.tier==ImmutableRowStorage::RAM; });
    if(owner==layout.owners().end())throw std::logic_error("Adaptive placement lost its exact RAM locality");
    return owner->host_placement;
}
const ImmutableRowTableExtent &ResolvedImmutableRowTablePlacement::extent() const noexcept
{
    if(const auto *w=std::get_if<WholeTier>(&value_))return w->extent;
    if(const auto *f=std::get_if<FixedHeadShards>(&value_))return f->layout.extent();
    if(const auto *a=std::get_if<GPUAdaptive>(&value_))return a->layout.extent();
    return std::get<HostAdaptive>(value_).layout.extent();
}
ImmutableRowAllocatorInventory ResolvedImmutableRowTablePlacement::allocators() const noexcept
{
    if(const auto *w=std::get_if<WholeTier>(&value_))return {w->allocators};
    // This component admits owner==consumer. The layout's existing consumer
    // inventory is the immutable identity view, not another stored owner map.
    if(const auto *f=std::get_if<FixedHeadShards>(&value_))return {f->layout.consumers()};
    if(const auto *a=std::get_if<GPUAdaptive>(&value_))
        return {a->layout.consumers(),{&a->layout.sourceServices().front().allocator,1}};
    return {{&std::get<HostAdaptive>(value_).layout.sourceServices().front().allocator,1}};
}
std::span<const PhysicalMemoryAllocatorIdentity> ResolvedImmutableRowTablePlacement::consumers() const noexcept
{
    if(const auto *w=std::get_if<WholeTier>(&value_))return w->consumers;
    if(const auto *f=std::get_if<FixedHeadShards>(&value_))return f->layout.consumers();
    if(const auto *a=std::get_if<GPUAdaptive>(&value_))return a->layout.consumers();
    return std::get<HostAdaptive>(value_).layout.consumers();
}
void ResolvedImmutableRowTablePlacement::requireConsumer(PhysicalMemoryAllocatorIdentity identity) const
{
    const auto members=consumers();
    if(std::find(members.begin(),members.end(),identity)==members.end())
        throw std::invalid_argument("Immutable row placement queried by a foreign PLE consumer");
}
ImmutableRowStorage ResolvedImmutableRowTablePlacement::storageForConsumer(PhysicalMemoryAllocatorIdentity identity) const
{requireConsumer(identity);return storage();}
ImmutableRowLookupExecution ResolvedImmutableRowTablePlacement::lookupForConsumer(PhysicalMemoryAllocatorIdentity identity) const
{
    requireConsumer(identity);
    if(isFixedHeadShards())return ImmutableRowLookupExecution::FixedHeadShardGPU;
    if(isGPUAdaptive())return ImmutableRowLookupExecution::AdaptiveGPUCache;
    return isHostAdaptive()?ImmutableRowLookupExecution::HostProvider:immutableRowLookupExecution(storage());
}
bool ResolvedImmutableRowTablePlacement::containsAllocator(PhysicalMemoryAllocatorIdentity identity) const noexcept
{const auto owners=allocators();return std::find(owners.begin(),owners.end(),identity)!=owners.end();}
const ImmutableRowTableLayoutGeometry &ResolvedImmutableRowTablePlacement::hostAdaptiveLayout() const
{
    if(const auto *a=std::get_if<HostAdaptive>(&value_))return a->layout;
    throw std::logic_error("Whole-tier ngrams have no adaptive layout");
}
const CpuImmutableRowCachePolicy &ResolvedImmutableRowTablePlacement::hostAdaptivePolicy() const
{
    if(const auto *a=std::get_if<HostAdaptive>(&value_))return a->request.policy;
    throw std::logic_error("Whole-tier ngrams have no adaptive replacement policy");
}
const HostAdaptiveRAMSSDRequest &ResolvedImmutableRowTablePlacement::hostAdaptiveRequest() const
{
    if(const auto *a=std::get_if<HostAdaptive>(&value_))return a->request;
    throw std::logic_error("Whole-tier ngrams have no adaptive requested ceilings");
}
const ImmutableRowTableLayoutGeometry &ResolvedImmutableRowTablePlacement::gpuAdaptiveLayout() const
{
    if(const auto *a=std::get_if<GPUAdaptive>(&value_))return a->layout;
    throw std::logic_error("Other ngram placements have no GPU adaptive layout");
}
const AdaptiveVRAMRAMSSDRequest &ResolvedImmutableRowTablePlacement::gpuAdaptiveRequest() const
{
    if(const auto *a=std::get_if<GPUAdaptive>(&value_))return a->request;
    throw std::logic_error("Other ngram placements have no GPU adaptive request");
}
std::uint64_t ResolvedImmutableRowTablePlacement::gpuAdaptiveRowElements() const
{
    if(const auto *a=std::get_if<GPUAdaptive>(&value_))return a->row_elements;
    throw std::logic_error("Other ngram placements have no GPU adaptive native row element declaration");
}
std::size_t ResolvedImmutableRowTablePlacement::gpuAdaptiveMaximumLiveRows() const
{
    return static_cast<std::size_t>(gpuAdaptiveLayout().maximumInflightRows());
}
void ResolvedImmutableRowTablePlacement::requireRequestedPolicy(const ImmutableRowTablePolicy &requested) const
{
    requested.validate();
    if(isGPUAdaptive())
    {
        const auto &windows=gpuAdaptiveLayout().sourceServices().front().windows;
        if(!requested.isGPUAdaptive() || requested.gpu_adaptive!=gpuAdaptiveRequest() ||
            requested.window_bytes!=windows.windowBytes() || requested.window_count!=windows.windowCount())
            throw std::invalid_argument("GPU adaptive reuse changed its explicit policy or original requested ceilings");
        return;
    }
    if(requested.isGPUAdaptive())throw std::invalid_argument("GPU adaptive request cannot reuse another storage arm");
    if(isFixedHeadShards())
    {
        if(!requested.isFixedHeadSharded())
            throw std::invalid_argument("Retained fixed head shards require the original explicit distribution request");
        return;
    }
    if(requested.isFixedHeadSharded())
        throw std::invalid_argument("A replicated/hosted table cannot satisfy fixed head distribution");
    if(isHostAdaptive())
    {
        const auto &windows=hostAdaptiveLayout().sourceServices().front().windows;
        if(!requested.isHostAdaptive() || requested.host_adaptive!=hostAdaptiveRequest() ||
            requested.window_bytes!=windows.windowBytes() || requested.window_count!=windows.windowCount())
            throw std::invalid_argument("Retained adaptive provider request changed, even if admitted slots could coincide");
        return;
    }
    if(requested.isHostAdaptive() || requested.window_bytes!=storagePolicy().windowBytes() ||
        requested.window_count!=storagePolicy().windowCount() ||
        (requested.storage!=ImmutableRowStoragePreference::Automatic &&
         !(requested.storage==ImmutableRowStoragePreference::RAM && storage()==ImmutableRowStorage::RAM) &&
         !(requested.storage==ImmutableRowStoragePreference::VRAM && storage()==ImmutableRowStorage::VRAM) &&
         !(requested.storage==ImmutableRowStoragePreference::MemoryMapped && storage()==ImmutableRowStorage::MemoryMapped)))
        throw std::invalid_argument("Retained whole-tier provider does not honor the exact requested mode/windows");
}
std::size_t ResolvedImmutableRowTablePlacement::hostAdaptiveMaximumLiveRows() const
{return static_cast<std::size_t>(hostAdaptiveLayout().maximumInflightRows());}
const ImmutableRowTableLayoutGeometry &ResolvedImmutableRowTablePlacement::fixedHeadShardLayout() const
{
    if(const auto *f=std::get_if<FixedHeadShards>(&value_))return f->layout;
    throw std::logic_error("This placement does not own fixed head shards");
}
ImmutableRowTableSourceRange ResolvedImmutableRowTablePlacement::fixedHeadSourceRangeFor(PhysicalMemoryAllocatorIdentity owner) const
{
    const auto &g=fixedHeadShardLayout();
    const auto r=std::find_if(g.fixedRanges().begin(),g.fixedRanges().end(),[&](const auto &r){return r.owner.allocator==owner;});
    if(r==g.fixedRanges().end())throw std::invalid_argument("Fixed source range queried by a foreign owner");
    return ImmutableRowTableSourceRange::checked(g.extent(),r->first_row,r->rows);
}
PhysicalMemoryPlan ResolvedImmutableRowTablePlacement::fixedHeadShardStorageBOM(std::span<const PhysicalMemoryResource> resources) const
{
    const auto &g=fixedHeadShardLayout();
    if(resources.size()!=g.owners().size())throw std::invalid_argument("Fixed native source BOM needs every actual owner observation");
    std::vector<PhysicalMemoryAllocatorIdentity> seen;
    PhysicalMemoryPlanBuilder plan;
    for(const auto &r:resources)
    {
        const PhysicalMemoryAllocatorIdentity id{r.world_rank,r.device};
        if(!r.valid() || !containsAllocator(id) || std::find(seen.begin(),seen.end(),id)!=seen.end())
            throw std::invalid_argument("Fixed native source BOM contains a duplicate, missing or foreign owner");
        seen.push_back(id);
        plan.add(r,PhysicalMemoryOwner::ImmutableTableStorage,fixedHeadSourceRangeFor(id).nativeBytes());
    }
    return plan.build();
}
} // namespace llaminar2
