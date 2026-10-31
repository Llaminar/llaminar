/**
 * @file ImmutableRowTablePlacementPlanner.cpp
 * @brief Select a concrete native-table candidate using only canonical PMA fit decisions.
 *
 * Complete candidates extend a frozen higher-priority plan. A failed candidate
 * cannot change model/expert placement or graph/KV geometry. Explicit preferences
 * evaluate one candidate only; Automatic selection ends before materialization.
 */
#include "ImmutableRowTablePlacementPlanner.h"
#include "loaders/ImmutableRowTable.h"
#include "models/qwen4exp/Qwen4ExpFixedHeadShards.h"
#include "loaders/CpuAdaptiveImmutableRowCache.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include <limits>
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
namespace llaminar2
{
namespace
{
/** @return Physical identity, independent of mutable capacity observations. */
PhysicalMemoryAllocatorIdentity identity(const PhysicalMemoryResource &r) { return {r.world_rank, r.device}; }
/** @return Whether this owner is an exact mode-dependent ngram contribution. */
bool modeOwner(PhysicalMemoryOwner owner)
{
    return owner == PhysicalMemoryOwner::ActivationArena || owner == PhysicalMemoryOwner::ImmutableLookupStaging ||
        owner == PhysicalMemoryOwner::NativeGraphExecutable || owner == PhysicalMemoryOwner::WeightLoadStaging;
}
/** @brief Reject mixed higher-priority charges and unrelated/foreign candidate resources. */
void validateContribution(const PhysicalMemoryPlan &plan, ImmutableRowStorage storage,
    const ImmutableRowTablePlacementInput &input)
{
    for (const auto &bom : plan.resources())
    {
        const auto id = identity(bom.resource());
        const bool host = id == identity(input.host());
        const bool consumer = std::any_of(input.consumers().begin(), input.consumers().end(),
            [&](const auto &r) { return identity(r) == id; });
        if (!host && !consumer)
            throw std::invalid_argument("Ngram mode contribution names a foreign physical resource");
        for (std::size_t i = 0; i < PhysicalMemoryBOM::ownerCount(); ++i)
        {
            const auto owner = PhysicalMemoryBOM::ownerAt(i);
            if (bom.alreadyResidentBytes(owner) || (bom.bytes(owner) && !modeOwner(owner)))
                throw std::invalid_argument("Ngram mode contribution contains higher-priority owners or residency credits");
            if(bom.bytes(owner) && ((owner==PhysicalMemoryOwner::ImmutableLookupStaging && !host) ||
                (owner==PhysicalMemoryOwner::ActivationArena && !consumer) ||
                (owner==PhysicalMemoryOwner::NativeGraphExecutable && (!consumer || !id.device.is_gpu()))))
                throw std::invalid_argument("Ngram mode owner is charged to an incompatible physical resource");
            if (storage == ImmutableRowStorage::VRAM && bom.bytes(owner) &&
                owner != PhysicalMemoryOwner::WeightLoadStaging)
                throw std::invalid_argument("Resident ngram candidate must not retain hosted lookup arena/channel/executable costs");
            if (storage != ImmutableRowStorage::VRAM && bom.bytes(owner) &&
                owner == PhysicalMemoryOwner::WeightLoadStaging)
                throw std::invalid_argument("Hosted ngram candidate must not retain resident upload staging");
        }
    }
}
}
ImmutableRowTablePlacementInput ImmutableRowTablePlacementInput::forConsumers(
    ImmutableRowTableExtent extent, PhysicalMemoryResource host, std::span<const PhysicalMemoryResource> consumers,
    ImmutableRowTableHostPlacement host_placement)
{
    if (!host.valid() || host.world_rank < 0 || !host.device.is_cpu() || consumers.empty())
        throw std::invalid_argument("Ngram admission requires an exact rank-local host resource and actual PLE consumers");
    std::vector<PhysicalMemoryResource> ordered(consumers.begin(), consumers.end());
    for (const auto &r : ordered)
        if (!r.valid() || r.world_rank != host.world_rank)
            throw std::invalid_argument("Ngram PLE consumer has invalid or foreign resource authority");
    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) { return a.device < b.device; });
    if (std::adjacent_find(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
            return identity(a) == identity(b); }) != ordered.end())
        throw std::invalid_argument("Ngram PLE consumer inventory contains a duplicate physical resource");
    // The canonical builder authenticates identical observations for a CPU PLE
    // consumer and its shared host owner instead of accepting two sampled budgets.
    PhysicalMemoryPlanBuilder observations;
    observations.add(host, PhysicalMemoryOwner::ImmutableTableStorage, 0u);
    for (const auto &r : ordered)
        observations.add(r, PhysicalMemoryOwner::ImmutableTableStorage, 0u);
    (void)observations.build();
    return ImmutableRowTablePlacementInput(extent, host, std::move(ordered), host_placement);
}
bool ImmutableRowTablePlacementInput::permitsVRAM() const noexcept
{
    return std::all_of(consumers_.begin(), consumers_.end(), [](const auto &r) { return r.device.is_gpu(); });
}
ImmutableRowTableLayoutGeometry ImmutableRowTablePlacementInput::hostAdaptiveLayout(
    const ImmutableRowTablePolicy &requested,std::size_t slots,std::size_t maximum_live_ids) const
{
    requested.validate();
    if(!requested.isHostAdaptive() || slots>requested.host_adaptive.maximum_cache_rows || slots>extent_.rows() || !maximum_live_ids)
        throw std::invalid_argument("Host adaptive layout exceeds its explicit source/request/live-ID bounds");
    const std::array tiers{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
    std::vector<PhysicalMemoryAllocatorIdentity> consumers;
    for(const auto &c:consumers_)consumers.push_back(identity(c));
    return ImmutableRowTableLayoutGeometry::checked(extent_,ImmutableRowTableTiers::of(tiers),
        ImmutableRowTableResidency::Adaptive,{{{identity(host_),ImmutableRowStorage::RAM},0,slots,std::nullopt,host_placement_}},
        std::move(consumers),{},{{identity(host_),ImmutableRowTableStoragePolicy::forStorage(
            ImmutableRowStorage::MemoryMapped,requested.window_bytes,requested.window_count)}},{},std::nullopt,maximum_live_ids);
}
ImmutableRowTableLayoutGeometry ImmutableRowTablePlacementInput::gpuAdaptiveLayout(
    const ImmutableRowTablePolicy &requested,std::size_t gpu_slots,std::size_t host_slots,std::size_t maximum_live_ids) const
{
    requested.validate();
    if(!requested.isGPUAdaptive() || consumers_.size()!=1 || !consumers_.front().device.is_gpu() ||
        gpu_slots>requested.gpu_adaptive.maximum_gpu_cache_rows || host_slots>requested.gpu_adaptive.host.maximum_cache_rows ||
        gpu_slots>extent_.rows() || host_slots>extent_.rows() || !maximum_live_ids || maximum_live_ids%16 ||
        maximum_live_ids>0x7fffffffULL)
        throw std::invalid_argument("GPU adaptive layout requires one exact local GPU and bounded original source/ID capacities");
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
    const auto requester=identity(consumers_.front());
    return ImmutableRowTableLayoutGeometry::checked(extent_,ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
        {{{requester,ImmutableRowStorage::VRAM},0,gpu_slots,std::nullopt},
         {{identity(host_),ImmutableRowStorage::RAM},0,host_slots,std::nullopt,host_placement_}},
        {requester},{},{{identity(host_),ImmutableRowTableStoragePolicy::forStorage(
            ImmutableRowStorage::MemoryMapped,requested.window_bytes,requested.window_count)}},{},
        gpu_slots?std::optional{requester}:std::nullopt,maximum_live_ids);
}
std::size_t immutableRowTableMaximumLiveIds(int ordinary_rows,int batch,std::size_t heads,
    int verifier_rows,int retained_rows)
{
    if(ordinary_rows<=0 || batch<=0 || !heads || verifier_rows<0 || retained_rows<0)
        throw std::invalid_argument("Ngram live-ID projection requires exact positive ordinary/batch/head geometry");
    const auto rows=static_cast<std::size_t>(std::max({ordinary_rows,verifier_rows,retained_rows}));
    const auto requests=static_cast<std::size_t>(batch);
    if(requests>std::numeric_limits<std::size_t>::max()/rows ||
       heads>std::numeric_limits<std::size_t>::max()/(requests*rows))
        throw std::overflow_error("Ngram maximum live-ID envelope overflow");
    return requests*rows*heads;
}
std::size_t ImmutableRowTableExecutionProvision::maximumLiveIds() const
{
    std::size_t maximum=0;
    for(const auto &c:consumers_)maximum=std::max(maximum,immutableRowTableMaximumLiveIds(
        c.rows,c.batch,source_.ple_heads,c.mtp_rows,c.retained_mtp_rows));
    return maximum;
}
const PhysicalMemoryPlan &ImmutableRowTableModeContributions::forStorage(ImmutableRowStorage storage) const
{
    switch (storage)
    {
    case ImmutableRowStorage::VRAM:
        if (!vram) throw std::invalid_argument("Resident ngram candidate has no authenticated usable original upload slot");
        return *vram;
    case ImmutableRowStorage::RAM: return ram;
    case ImmutableRowStorage::MemoryMapped: return mmap;
    }
    throw std::invalid_argument("Unknown concrete ngram storage contribution");
}
std::shared_ptr<const ResolvedImmutableRowTablePlacement> ImmutableRowTableExecutionProvision::fixedHeadPlacement(
    const ImmutableRowTablePlacementInput &input) const
{
    requested_.validate();
    const auto source_extent=ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(source_.ple_table_gguf_type),
        source_.ple_table_rows,static_cast<std::size_t>(source_.ple_table_row_bytes));
    if(source_extent!=input.extent() || source_.ple_table_native_bytes!=source_extent.nativeBytes())
        throw std::invalid_argument("Fixed head placement has a foreign original source extent");
    const auto p=input.consumers().size();
    if(!requested_.isFixedHeadSharded() || (p!=2 && p!=4 && p!=8) || source_.ple_heads!=16 ||
       source_.ple_head_offsets.size()!=16 || source_.ple_head_vocab_sizes.size()!=16)
        throw std::invalid_argument("Fixed head admission requires original16 head metadata and actual TP degree2/4/8");
    std::vector<PhysicalMemoryAllocatorIdentity> order(p,{-1,DeviceId::invalid()});
    for(const auto &c:consumers_)
    {
        const PhysicalMemoryAllocatorIdentity id{c.resource.world_rank,c.resource.device};
        if(c.tp_topology!=ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal ||
           c.tp_coordinate<0 || static_cast<std::size_t>(c.tp_coordinate)>=p || c.shards!=static_cast<int>(p) ||
           c.first_layer!=0 || c.last_layer!=47 || c.batch!=1 || !c.resource.device.is_gpu() ||
           std::none_of(input.consumers().begin(),input.consumers().end(),[&](const auto &r){return r.sameAllocator(c.resource);}))
            throw std::invalid_argument("Fixed head provision is not the declared full-main rank-local TP cohort");
        auto &entry=order[static_cast<std::size_t>(c.tp_coordinate)];
        if(entry.device.is_valid() && entry!=id)
            throw std::invalid_argument("Fixed head provision reassigns one communicator coordinate");
        entry=id;
    }
    if(std::any_of(order.begin(),order.end(),[](const auto &id){return !id.device.is_valid();}))
        throw std::invalid_argument("Fixed head provision omits an actual TP coordinate");
    return Qwen4ExpFixedHeadShards::declare(input.extent(),source_.ple_head_offsets,source_.ple_head_vocab_sizes,order);
}
void ImmutableRowTableExecutionProvision::require(const ImmutableRowTablePolicy &requested,
    const ImmutableRowTablePlacementInput &input,const PhysicalMemoryPlan &base) const
{
    const auto extent=ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(source_.ple_table_gguf_type),
        source_.ple_table_rows,static_cast<std::size_t>(source_.ple_table_row_bytes));
    if(requested_!=requested || extent!=input.extent() || source_.ple_table_native_bytes!=extent.nativeBytes() ||
        host_placement_!=input.hostPlacement() || consumers_.empty())
        throw std::invalid_argument("Final ngram execution provision has foreign policy/source/locality identity");
    for(const auto &consumer:consumers_)
        if(std::none_of(input.consumers().begin(),input.consumers().end(),[&](const auto &r) {
            return r.sameAllocator(consumer.resource); }))
            throw std::invalid_argument("Final ngram execution provision has a foreign consumer");
    for(const auto &consumer:input.consumers())
        if(std::none_of(consumers_.begin(),consumers_.end(),[&](const auto &r) {
            return r.resource.sameAllocator(consumer); }))
            throw std::invalid_argument("Final ngram execution provision omitted an actual consumer");
    if(requested.isGPUAdaptive())
    {
        if(input.consumers().size()!=1 || !input.consumers().front().device.is_gpu() || source_.ple_heads!=16)
            throw std::invalid_argument("GPU adaptive admission requires one exact local GPU PLE domain");
        for(const auto &c:consumers_)
            if(c.shards!=1 || c.batch!=1 || c.first_layer!=0 || c.last_layer!=47 ||
                !c.resource.sameAllocator(input.consumers().front()))
                throw std::invalid_argument("GPU adaptive provision supports only full-main unsharded single-GPU execution");
    }
    if(requested.isFixedHeadSharded())(void)fixedHeadPlacement(input);
    else for(const auto &consumer:consumers_)
        if(consumer.tp_coordinate!=-1 || consumer.tp_topology!=ImmutableRowTableTPTopology::None)
            throw std::invalid_argument("Replicated provision cannot carry fixed head topology");
    if(const auto mismatch=base.requiredFootprintMismatch(required_base_))
        throw std::invalid_argument("Final ngram base omits mandatory execution provision: "+*mismatch);
}
ImmutableRowTablePlacementSelection ImmutableRowTablePlacementPlanner::select(
    const ImmutableRowTableAdmissionRequest &request,const PhysicalMemoryPlan &higher_priority)
{
    // Explicit direct-selector metadata inputs retain their incremental mode
    // contract. Production MemoryPlanner requests always carry a sealed provision.
    if(!request.execution_provision)
        return select(request.requested,request.input,higher_priority,request.contributions);
    request.execution_provision->require(request.requested,request.input,higher_priority);
    if(request.requested.isFixedHeadSharded())
    {
        if(!higher_priority.fits() || !request.contributions.vram || request.contributions.ram.totalBytes() ||
           request.contributions.mmap.totalBytes())
            throw std::invalid_argument("Fixed head final admission requires fitting frozen base, original upload slot and prepriced execution");
        for(const auto &bom:higher_priority.resources())
            if(bom.resource().world_rank==request.input.host().world_rank &&
               (bom.bytes(PhysicalMemoryOwner::ImmutableTableStorage) || bom.bytes(PhysicalMemoryOwner::ImmutableLookupStaging)))
                throw std::invalid_argument("Fixed head base already contains unresolved native table/hosted staging owners");
        validateContribution(*request.contributions.vram,ImmutableRowStorage::VRAM,request.input);
        auto placement=request.execution_provision->fixedHeadPlacement(request.input);
        PhysicalMemoryPlanBuilder candidate(higher_priority);
        candidate.add(request.input.host(),PhysicalMemoryOwner::ImmutableTableStorage,0u);
        for(const auto &r:request.input.consumers())candidate.add(r,PhysicalMemoryOwner::ImmutableTableStorage,0u);
        for(const auto &bom:request.contributions.vram->resources())candidate.add(bom);
        const auto storage=placement->fixedHeadShardStorageBOM(request.input.consumers());
        for(const auto &bom:storage.resources())candidate.add(bom);
        auto complete=candidate.build();
        if (!complete.fits())
        {
            std::string diagnostic = "Explicit fixed head native ranges exceed complete frozen PMA capacity";
            // Report the canonical rejected resource and every owner exactly;
            // a generic table error cannot distinguish payload pressure from
            // retained graph, workspace, or expert ownership defects.
            for (const auto &bom : complete.resources())
                if (!bom.fits())
                    diagnostic += "; " + bom.summary() + " deficit_bytes=" + std::to_string(bom.deficitBytes());
            throw PhysicalMemoryCapacityExhausted(diagnostic);
        }
        return {std::move(placement),std::move(complete)};
    }
    if(request.requested.isGPUAdaptive())
    {
        const auto &input=request.input;const auto &requested=request.requested;
        if(request.contributions.ram.totalBytes() || request.contributions.mmap.totalBytes() ||
            (request.contributions.vram && request.contributions.vram->totalBytes()) || !higher_priority.fits())
            throw std::invalid_argument("GPU adaptive final admission needs fitting frozen execution and no duplicate mode costs");
        const auto &metadata=request.execution_provision->source();
        const ImmutableRowTableSource source{.rows=metadata.ple_table_rows,.row_elements=metadata.ple_row_elements,
            .format=static_cast<GGUFTensorType>(metadata.ple_table_gguf_type)};
        const auto maximum=request.execution_provision->maximumLiveIds();
        const auto candidate=[&](std::size_t gpu_slots,std::size_t host_slots) {
            const auto layout=input.gpuAdaptiveLayout(requested,gpu_slots,host_slots,maximum);
            const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(layout,source,requested.gpu_adaptive);
            PhysicalMemoryPlanBuilder plan(higher_priority);
            plan.add(input.host(),PhysicalMemoryOwner::ImmutableTableStorage,0u);
            for(const auto &consumer:input.consumers())plan.add(consumer,PhysicalMemoryOwner::ImmutableTableStorage,0u);
            const auto optional=geometry.slotDependentBOM(input.consumers().front(),input.host());
            for(const auto &bom:optional.resources())plan.add(bom);
            const bool ceilings=geometry.gpuSlotDependentBytes()<=requested.gpu_adaptive.gpu_cache_physical_byte_limit &&
                geometry.host().storageBytes()<=requested.gpu_adaptive.host.cache_physical_byte_limit;
            return std::pair{ceilings,plan.build()};
        };
        const auto zero=candidate(0,0);
        if(!zero.first || !zero.second.fits())
            throw PhysicalMemoryCapacityExhausted("GPU adaptive mandatory original frame/source/channel does not fit frozen PMA");
        // GPU then RAM priorities are explicit policy; each monotone candidate
        // extends the identical frozen complete plan. No other row, KV, expert
        // quota or higher-priority placement is recomputed when optional fit fails.
        const auto solve=[&](std::uint64_t ceiling,const auto &fits) {
            std::size_t low=0,high=static_cast<std::size_t>(std::min(ceiling,input.extent().rows()));
            while(low<high)
            {
                const auto mid=low+(high-low+1)/2;
                if(fits(mid))low=mid;else high=mid-1;
            }
            return low;
        };
        const auto gpu=solve(requested.gpu_adaptive.maximum_gpu_cache_rows,[&](std::size_t rows) {
            const auto next=candidate(rows,0);return next.first && next.second.fits(); });
        const auto ram=solve(requested.gpu_adaptive.host.maximum_cache_rows,[&](std::size_t rows) {
            const auto next=candidate(gpu,rows);return next.first && next.second.fits(); });
        auto complete=candidate(gpu,ram).second;
        auto placement=std::shared_ptr<const ResolvedImmutableRowTablePlacement>(new ResolvedImmutableRowTablePlacement(
            input.gpuAdaptiveLayout(requested,gpu,ram,maximum),requested.gpu_adaptive,metadata.ple_row_elements));
        return {std::move(placement),std::move(complete)};
    }
    if(request.requested.isHostAdaptive())
    {
        if(request.contributions.ram.totalBytes() || request.contributions.mmap.totalBytes() ||
            (request.contributions.vram && request.contributions.vram->totalBytes()))
            throw std::invalid_argument("Host adaptive execution was already priced before quotas; duplicate incremental arms are forbidden");
        if(!higher_priority.fits())throw std::invalid_argument("Adaptive table requires a fitting frozen higher-priority PMA plan");
        const auto &input=request.input;const auto &requested=request.requested;
        const auto maximum=request.execution_provision->maximumLiveIds();
        const auto candidate=[&](std::size_t slots) {
            const auto layout=input.hostAdaptiveLayout(requested,slots,maximum);
            const auto geometry=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(layout,identity(input.host()),maximum,
                requested.host_adaptive.policy);
            PhysicalMemoryPlanBuilder plan(higher_priority);
            // Zero-byte observations still authenticate the SAME canonical
            // allocator sample; requiredFootprintMismatch intentionally ignores
            // changing free observations and cannot replace this merge check.
            plan.add(input.host(),PhysicalMemoryOwner::ImmutableTableStorage,0u);
            for(const auto &consumer:input.consumers())
                plan.add(consumer,PhysicalMemoryOwner::ImmutableTableStorage,0u);
            const auto cache=geometry.cacheBOM(input.host());
            for(const auto &bom:cache.resources())plan.add(bom);
            return std::pair{geometry.storageBytes()<=requested.host_adaptive.cache_physical_byte_limit,plan.build()};
        };
        auto zero=candidate(0);
        if(!zero.first || !zero.second.fits())
            throw PhysicalMemoryCapacityExhausted("Host adaptive mandatory SSD/request execution cannot be admitted");
        // Exact source-owned bytes are monotone, including page/hash/sketch
        // jumps. Every candidate extends the SAME complete immutable base;
        // no free-byte balance or higher-priority row/quota retry is involved.
        std::size_t low=0,high=static_cast<std::size_t>(std::min<std::uint64_t>(
            requested.host_adaptive.maximum_cache_rows,input.extent().rows()));
        while(low<high)
        {
            const auto mid=low+(high-low+1)/2;const auto next=candidate(mid);
            if(next.first && next.second.fits())low=mid;else high=mid-1;
        }
        auto complete=candidate(low).second;
        auto placement=std::shared_ptr<const ResolvedImmutableRowTablePlacement>(new ResolvedImmutableRowTablePlacement(
            input.hostAdaptiveLayout(requested,low,maximum),requested.host_adaptive));
        return {std::move(placement),std::move(complete)};
    }

    if(request.contributions.ram.totalBytes()!=0 || request.contributions.mmap.totalBytes()!=0)
        throw std::invalid_argument("Final ngram request double charges mandatory lookup execution");
    return selectImpl(request.requested,request.input,higher_priority,request.contributions,true);
}
ImmutableRowTablePlacementSelection ImmutableRowTablePlacementPlanner::select(
    const ImmutableRowTablePolicy &requested, const ImmutableRowTablePlacementInput &input,
    const PhysicalMemoryPlan &higher_priority, const ImmutableRowTableModeContributions &contributions)
{
    return selectImpl(requested,input,higher_priority,contributions,false);
}
ImmutableRowTablePlacementSelection ImmutableRowTablePlacementPlanner::selectImpl(
    const ImmutableRowTablePolicy &requested, const ImmutableRowTablePlacementInput &input,
    const PhysicalMemoryPlan &higher_priority, const ImmutableRowTableModeContributions &contributions,
    bool execution_prepriced)
{
    requested.validate();
    if(requested.isHostAdaptive() || requested.isGPUAdaptive() || requested.isFixedHeadSharded())
        throw std::invalid_argument("Adaptive/fixed head selection requires its complete source-authored execution provision");
    if (!higher_priority.fits())
        throw std::invalid_argument("Higher-priority PMA plan must fit before ngram storage selection");
    for (const auto &bom : higher_priority.resources())
        if (bom.resource().world_rank == input.host().world_rank &&
            (bom.bytes(PhysicalMemoryOwner::ImmutableTableStorage) ||
             (!execution_prepriced && bom.bytes(PhysicalMemoryOwner::ImmutableLookupStaging))))
            throw std::invalid_argument("Higher-priority PMA base already contains unresolved ngram owners");
    for (const auto &r : input.consumers())
        if (!higher_priority.find(identity(r)))
            throw std::invalid_argument("Ngram consumer is absent from the frozen higher-priority PMA plan");
    if(contributions.vram)validateContribution(*contributions.vram,ImmutableRowStorage::VRAM,input);
    validateContribution(contributions.ram,ImmutableRowStorage::RAM,input);
    validateContribution(contributions.mmap,ImmutableRowStorage::MemoryMapped,input);
    const std::array order{ImmutableRowStorage::VRAM, ImmutableRowStorage::RAM, ImmutableRowStorage::MemoryMapped};
    std::vector<PhysicalMemoryAllocatorIdentity> consumers;
    for (const auto &r : input.consumers()) consumers.push_back(identity(r));
    for (const auto storage : order)
    {
        const bool selected = requested.storage == ImmutableRowStoragePreference::Automatic ||
            (requested.storage == ImmutableRowStoragePreference::VRAM && storage == ImmutableRowStorage::VRAM) ||
            (requested.storage == ImmutableRowStoragePreference::RAM && storage == ImmutableRowStorage::RAM) ||
            (requested.storage == ImmutableRowStoragePreference::MemoryMapped && storage == ImmutableRowStorage::MemoryMapped);
        if (!selected) continue;
        if (storage == ImmutableRowStorage::VRAM && !input.permitsVRAM())
        {
            if (requested.storage != ImmutableRowStoragePreference::Automatic)
                throw std::invalid_argument("Explicit VRAM ngrams require every actual PLE consumer to be a GPU");
            continue;
        }
        if (storage == ImmutableRowStorage::VRAM && !contributions.vram)
        {
            if (requested.storage != ImmutableRowStoragePreference::Automatic)
                throw std::invalid_argument("Explicit VRAM ngrams require authenticated original loader geometry with a complete native row fitting its released slot");
            continue;
        }
        const auto concrete = ImmutableRowTableStoragePolicy::forStorage(storage, requested.window_bytes, requested.window_count);
        const auto &mode = contributions.forStorage(storage);
        PhysicalMemoryPlanBuilder candidate(higher_priority);
        // Include every observation before admission, even if this concrete
        // tier needs no storage on it; conflicting sampled budgets fail closed.
        candidate.add(input.host(), PhysicalMemoryOwner::ImmutableTableStorage, 0u);
        for (const auto &r : input.consumers()) candidate.add(r, PhysicalMemoryOwner::ImmutableTableStorage, 0u);
        for (const auto &bom : mode.resources()) candidate.add(bom);
        std::vector<PhysicalMemoryAllocatorIdentity> owners;
        if (storage == ImmutableRowStorage::VRAM)
        {
            for (const auto &r : input.consumers())
            {
                candidate.add(r, PhysicalMemoryOwner::ImmutableTableStorage, input.extent().nativeBytes());
                owners.push_back(identity(r));
            }
        }
        else
        {
            candidate.add(input.host(), PhysicalMemoryOwner::ImmutableTableStorage,
                ImmutableRowTable::physicalBytesForNativeExtent(input.extent().nativeBytes(), concrete));
            owners.push_back(identity(input.host()));
        }
        auto complete = candidate.build();
        if (complete.fits())
        {
            auto placement = std::shared_ptr<const ResolvedImmutableRowTablePlacement>(
                new ResolvedImmutableRowTablePlacement(concrete, input.extent(), std::move(owners), consumers, input.hostPlacement()));
            return {std::move(placement), std::move(complete)};
        }
        if (requested.storage != ImmutableRowStoragePreference::Automatic)
            throw PhysicalMemoryCapacityExhausted("Explicit ngram storage cannot be admitted by the complete PMA plan: " + complete.summary());
    }
    throw PhysicalMemoryCapacityExhausted("No Automatic ngram storage candidate fits the frozen complete PMA plan");
}
}
