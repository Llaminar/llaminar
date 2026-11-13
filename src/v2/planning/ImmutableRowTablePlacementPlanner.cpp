/**
 * @file ImmutableRowTablePlacementPlanner.cpp
 * @brief Select a concrete native-table candidate using only canonical PMA fit decisions.
 *
 * Complete candidates extend a frozen higher-priority plan. A failed candidate
 * cannot change model/expert placement or graph/KV geometry. Explicit preferences
 * evaluate one candidate only; Automatic selection ends before materialization.
 * Shared transport retains its authored source at zero cache capacity without
 * manufacturing a live GPU directory or changing compute participant roles.
 */
#include "execution/qwen4exp/PooledImmutableRetainedParticipantBudget.h"
#include "ImmutableRowTablePlacementPlanner.h"
#include "ImmutableRowTableHostAuthority.h"
#include "loaders/ImmutableRowTable.h"
#include "models/qwen4exp/Qwen4ExpFixedHeadShards.h"
#include "loaders/CpuAdaptiveImmutableRowCache.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include <limits>
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <set>
#include <string>
#include <tuple>
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
        owner == PhysicalMemoryOwner::NativeGraphExecutable || owner == PhysicalMemoryOwner::WeightLoadStaging ||
        owner == PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent;
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
            if(owner==PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent && bom.bytes(owner) &&
               ((!host && (!consumer || !id.device.is_gpu())) ||
                std::none_of(input.consumers().begin(),input.consumers().end(),[](const auto& r){return r.device.is_gpu();})))
                throw std::invalid_argument("Native lookup ordering contribution requires its canonical host or actual GPU consumer");
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
    requireCanonicalImmutableHostObservation(host);
    if(consumers.empty())
        throw std::invalid_argument("Ngram admission requires actual PLE consumers");
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
ImmutableRowTablePlacementInput ImmutableRowTablePlacementInput::withGPUCacheResources(
    const ImmutableRowTablePolicy &requested,std::span<const PhysicalMemoryResource> originals) const
{
    requested.validate();auto result=*this;result.gpu_resources_.clear();result.gpu_placement_.reset();result.gpu_peer_access_.reset();result.reader_peer_access_.clear();
    if(!requested.isGPUAdaptive())throw std::invalid_argument("GPU storage observations require explicit GPU adaptive policy");
    if (requested.gpu_adaptive.local_hot_cache)
    {
        std::vector<AdaptiveGPUCacheParticipant> original_consumers;
        original_consumers.reserve(consumers_.size());
        for (const auto& consumer : consumers_)
            original_consumers.push_back({consumer.world_rank, consumer.device.type, consumer.device.ordinal});
        requested.gpu_adaptive.local_hot_cache->requireConsumers(original_consumers);
    }
    const auto &placement=requested.gpu_adaptive.placement;
    if(placement.mode==AdaptiveGPUCachePlacementMode::ConsumerLocal)
    {
        if(originals.size()!=consumers_.size())throw std::invalid_argument("ConsumerLocal storage resources must equal actual consumer observations");
        PhysicalMemoryPlanBuilder observations;
        for(const auto &consumer:consumers_)observations.add(consumer,PhysicalMemoryOwner::ImmutableTableStorage,0u);
        for(const auto &r:originals)
        {
            if(std::none_of(consumers_.begin(),consumers_.end(),[&](const auto &c){return identity(c)==identity(r);}))
                throw std::invalid_argument("ConsumerLocal storage cannot invent a remote resource");
            observations.add(r,PhysicalMemoryOwner::ImmutableTableStorage,0u);
        }
        (void)observations.build();return result;
    }
    const PhysicalMemoryAllocatorIdentity coordinator{placement.coordinator->world_rank,placement.coordinator->device()};
    if(consumers_.empty() || std::none_of(consumers_.begin(),consumers_.end(),[&](const auto& consumer){return identity(consumer)==coordinator;}))
        throw std::invalid_argument("Declared GPU cache coordinator must be its authored original PLE consumer");
    for(const auto& consumer:consumers_)
        if(!consumer.device.is_gpu() || consumer.world_rank!=coordinator.world_rank || consumer.device.type!=coordinator.device.type)
            throw std::invalid_argument("Shared GPU cache readers require original homogeneous rank-local GPUs");
    PhysicalMemoryPlanBuilder observations;observations.add(host_,PhysicalMemoryOwner::ImmutableTableStorage,0u);
    for(const auto &consumer:consumers_){
        observations.add(consumer,PhysicalMemoryOwner::ImmutableTableStorage,0u);
        const auto original=std::find_if(originals.begin(),originals.end(),[&](const auto& value){return identity(value)==identity(consumer);});
        if(original==originals.end())throw std::invalid_argument("Shared reader lacks its original selected source-worker observation");
        observations.add(*original,PhysicalMemoryOwner::ImmutableTableStorage,0u);
        result.gpu_resources_.push_back(*original);
    }
    for(const auto &participant:placement.declaredParticipants())
    {
        if(participant.world_rank!=host_.world_rank)throw std::invalid_argument("GPU cache participant crossed original rank authority");
        const auto matches=[&](const auto &r){return identity(r)==PhysicalMemoryAllocatorIdentity{participant.world_rank,participant.device()};};
        const auto found=std::find_if(originals.begin(),originals.end(),matches);
        if(found==originals.end() || !found->valid())throw std::invalid_argument("Declared GPU cache participant lacks its original selected source-worker observation");
        for(const auto &r:originals)if(matches(r))observations.add(r,PhysicalMemoryOwner::ImmutableTableStorage,0u);
        observations.add(*found,PhysicalMemoryOwner::ImmutableTableStorage,0u);
        if(std::none_of(result.gpu_resources_.begin(),result.gpu_resources_.end(),[&](const auto& value){return identity(value)==identity(*found);}))
            result.gpu_resources_.push_back(*found);
    }
    (void)observations.build();result.gpu_placement_=placement;return result;
}
bool ImmutableRowTablePlacementInput::permitsVRAM() const noexcept
{
    return std::all_of(consumers_.begin(), consumers_.end(), [](const auto &r) { return r.device.is_gpu(); });
}
ImmutableRowTablePlacementInput ImmutableRowTablePlacementInput::withGPUCachePeerAccess(PeerAccessCoverage coverage) const
{
    if ((coverage != PeerAccessCoverage::None && coverage != PeerAccessCoverage::Complete) || !gpu_placement_ ||
        gpu_placement_->mode != AdaptiveGPUCachePlacementMode::DeclaredRankLocal ||
        !gpu_placement_->coordinator ||
        std::none_of(gpu_placement_->owners.begin(), gpu_placement_->owners.end(), [&](const auto& owner) {
            return owner.participant != *gpu_placement_->coordinator &&
                owner.participant.backend == gpu_placement_->coordinator->backend;
        }))
        throw std::invalid_argument("Pooled transport needs its original declared storage inventory and explicit None/Complete observation");
    auto result = *this;
    result.gpu_peer_access_ = coverage;
    return result;
}
ImmutableRowTablePlacementInput ImmutableRowTablePlacementInput::withReaderPeerAccess(
    std::span<const ImmutableRowTableReaderPeerAccess> edges) const
{
    if (!gpu_placement_ || gpu_placement_->mode != AdaptiveGPUCachePlacementMode::DeclaredRankLocal ||
        !gpu_placement_->coordinator || consumers_.size() < 2 || edges.size() + 1 != consumers_.size())
        throw std::invalid_argument("Shared reader transport requires its complete declared reader population");
    const PhysicalMemoryAllocatorIdentity coordinator{gpu_placement_->coordinator->world_rank,
        gpu_placement_->coordinator->device()};
    auto result = *this;
    result.reader_peer_access_.assign(edges.begin(), edges.end());
    std::sort(result.reader_peer_access_.begin(), result.reader_peer_access_.end(),
        [](const auto &a, const auto &b) {
            return std::tuple{a.reader.world_rank, a.reader.device.type, a.reader.device.ordinal} <
                std::tuple{b.reader.world_rank, b.reader.device.type, b.reader.device.ordinal};
        });
    for (std::size_t i = 0; i < result.reader_peer_access_.size(); ++i) {
        const auto &edge = result.reader_peer_access_[i];
        if (edge.coordinator != coordinator || edge.reader == coordinator ||
            edge.reader.world_rank != coordinator.world_rank || edge.reader.device.type != coordinator.device.type ||
            (edge.coverage != PeerAccessCoverage::None && edge.coverage != PeerAccessCoverage::Complete) ||
            std::none_of(consumers_.begin(), consumers_.end(), [&](const auto &r) { return identity(r) == edge.reader; }) ||
            (i && result.reader_peer_access_[i - 1].reader == edge.reader))
            throw std::invalid_argument("Shared reader transport has a foreign, duplicate or unsupported edge");
    }
    return result;
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
    if(requested.gpu_adaptive.placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal)
    {
        if(gpu_slots)throw std::invalid_argument("Declared GPU owners require explicit per-owner slot counts");
        const std::vector<std::size_t> zero(requested.gpu_adaptive.placement.owners.size(),0);
        return gpuAdaptiveLayout(requested,zero,host_slots,maximum_live_ids);
    }
    const std::vector<std::size_t> slots(consumers_.size(), gpu_slots);
    return gpuAdaptiveLayout(requested,slots,host_slots,maximum_live_ids);
}
ImmutableRowTableLayoutGeometry ImmutableRowTablePlacementInput::gpuAdaptiveLayout(
    const ImmutableRowTablePolicy &requested,std::span<const std::size_t> gpu_slots,std::size_t host_slots,std::size_t maximum_live_ids) const
{
    requested.validate();
    const auto &g=requested.gpu_adaptive;const auto &placement=g.placement;
    const bool declared=placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal;
    if(!requested.isGPUAdaptive() || consumers_.empty() ||
        std::any_of(consumers_.begin(), consumers_.end(), [](const auto& consumer) { return !consumer.device.is_gpu(); }) ||
        gpu_slots.size()!=(declared?placement.owners.size():consumers_.size()) || host_slots>g.host.maximum_cache_rows ||
        host_slots>extent_.rows() || !maximum_live_ids || maximum_live_ids%16 || maximum_live_ids>0x7fffffffULL)
        throw std::invalid_argument("GPU adaptive layout requires exact GPU consumers and bounded original owner/source/ID capacities");
    const auto requester=declared ? PhysicalMemoryAllocatorIdentity{placement.coordinator->world_rank,placement.coordinator->device()} : identity(consumers_.front());
    if(declared && (!gpu_placement_ || *gpu_placement_!=placement || gpu_resources_.empty() || std::none_of(consumers_.begin(),consumers_.end(),[&](const auto& consumer){return identity(consumer)==requester;})))
        throw std::invalid_argument("Declared GPU layout requires authenticated original storage resources and exact consumer coordinator");
    std::vector<ImmutableRowTableLayoutOwner> owners;
    for(std::size_t i=0;i<gpu_slots.size();++i)
    {
        const auto maximum=declared?placement.owners[i].maximum_cache_rows:g.maximum_gpu_cache_rows;
        if(gpu_slots[i]>maximum || gpu_slots[i]>extent_.rows())throw std::invalid_argument("GPU cache owner slots exceed original source/request bounds");
        const auto allocator=declared?PhysicalMemoryAllocatorIdentity{placement.owners[i].participant.world_rank,placement.owners[i].participant.device()}:identity(consumers_[i]);
        owners.push_back({{allocator,ImmutableRowStorage::VRAM},0,gpu_slots[i],std::nullopt});
    }
    owners.push_back({{identity(host_),ImmutableRowStorage::RAM},0,host_slots,std::nullopt,host_placement_});
    const bool active=std::any_of(gpu_slots.begin(),gpu_slots.end(),[](auto n){return n!=0;});
    const std::vector<ImmutableRowStorage> tiers=requested.gpu_adaptive.source_backing==AdaptiveImmutableRowBacking::RAM
        ? std::vector{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM}
        : std::vector{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
    std::vector<PhysicalMemoryAllocatorIdentity> consumers;
    for (const auto& consumer : consumers_) consumers.push_back(identity(consumer));
    const bool independent = !declared && consumers.size() > 1;
    return ImmutableRowTableLayoutGeometry::checked(extent_,ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
        std::move(owners),std::move(consumers),{},{{identity(host_),ImmutableRowTableStoragePolicy::forStorage(
            requested.gpu_adaptive.sourceStorage(),requested.window_bytes,requested.window_count)}},{},
        !independent && active?std::optional{requester}:std::nullopt,maximum_live_ids,
        independent ? ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal : ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator,
        gpu_peer_access_, reader_peer_access_, declared ? std::optional{requester} : std::nullopt);
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
ImmutableRowTablePlacementInput ImmutableRowTablePlacementInput::withFixedHeadPipeline(
    std::shared_ptr<const ImmutableFixedHeadPipelineProjection> projection,
    std::span<const PhysicalMemoryResource> resources) const
{
    if (!projection || fixed_head_pipeline_ || gpu_placement_ || !gpu_resources_.empty() ||
        projection->worldRank() != host_.world_rank || resources.size() != projection->owners().size() ||
        consumers_.size() != projection->consumers().size())
        throw std::invalid_argument("Fixed PLE placement requires exact original pipeline resources");
    ImmutableRowTablePlacementInput result(*this);
    std::set<DeviceId> seen;
    for (const auto& resource : resources) {
        if (!resource.valid() || !seen.insert(resource.device).second)
            throw std::invalid_argument("Fixed PLE placement has invalid or duplicated storage resource");
        (void)projection->owner(identity(resource));
    }
    for (const auto& consumer : consumers_)
        if (std::find(projection->consumers().begin(), projection->consumers().end(), identity(consumer)) ==
                projection->consumers().end() ||
            std::none_of(resources.begin(), resources.end(), [&](const auto& resource) {
                return resource.sameAllocator(consumer);
            }))
            throw std::invalid_argument("Fixed PLE placement changed the original query resource");
    result.gpu_resources_.assign(resources.begin(), resources.end());
    result.fixed_head_pipeline_ = std::move(projection);
    return result;
}

std::shared_ptr<const ResolvedImmutableRowTablePlacement> ImmutableRowTableExecutionProvision::fixedHeadPlacement(
    const ImmutableRowTablePlacementInput &input) const
{
    requested_.validate();
    const auto source_extent=ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(source_.ple_table_gguf_type),
        source_.ple_table_rows,static_cast<std::size_t>(source_.ple_table_row_bytes));
    if(source_extent!=input.extent() || source_.ple_table_native_bytes!=source_extent.nativeBytes())
        throw std::invalid_argument("Fixed head placement has a foreign original source extent");
    if (const auto& pipeline = input.fixedHeadPipeline()) {
        const auto& offsets = source_.ple_head_offsets;
        const auto& sizes = source_.ple_head_vocab_sizes;
        if (!requested_.isFixedHeadSharded() || !source_.ple_heads ||
            offsets.size() != source_.ple_heads || sizes.size() != source_.ple_heads)
            throw std::invalid_argument("Fixed PLE pipeline lost its original source heads");
        std::uint64_t end = 0;
        for (std::size_t head = 0; head < offsets.size(); ++head) {
            if (!sizes[head] || offsets[head] < end || offsets[head] >= source_extent.rows() ||
                sizes[head] > source_extent.rows() - offsets[head])
                throw std::invalid_argument("Fixed PLE pipeline has overlapping or foreign source heads");
            end = offsets[head] + sizes[head];
        }
        for (const auto& reader : consumers_) {
            const auto& owner = pipeline->owner(identity(reader.resource));
            if (owner.stage_index != pipeline->consumerStage() ||
                reader.tp_coordinate != owner.participant_index ||
                reader.tp_topology != ImmutableRowTableTPTopology::HomogeneousPipelineRankLocal ||
                reader.shards != static_cast<int>(pipeline->consumers().size()) ||
                reader.batch != 1 || reader.rows > pipeline->maximumRows() ||
                reader.first_layer > static_cast<int>(source_.ple_decoder_layer) ||
                reader.last_layer < static_cast<int>(source_.ple_decoder_layer))
                throw std::invalid_argument("Fixed PLE reader changed its original stage/source coordinate");
        }
        std::vector<ImmutableRowTableLayoutOwner> owners;
        std::vector<ImmutableRowTableFixedRange> ranges;
        for (const auto& owner : pipeline->owners()) {
            const auto first = static_cast<std::size_t>(owner.first_head);
            const auto after = first + static_cast<std::size_t>(owner.head_count);
            if (first >= offsets.size() || after > offsets.size())
                throw std::invalid_argument("Fixed PLE owner heads exceed the original source");
            const auto begin = first ? offsets[first] : 0;
            const auto finish = after == offsets.size() ? source_extent.rows() : offsets[after];
            const auto range = ImmutableRowTableSourceRange::checked(source_extent, begin, finish - begin);
            const ImmutableRowTableOwnerKey key{owner.allocator, ImmutableRowStorage::VRAM};
            owners.push_back({key, range.rows(), 0, std::nullopt});
            ranges.push_back({range.firstRow(), range.rows(), key, ImmutableRowTableCopyRole::Primary});
        }
        const std::array tiers{ImmutableRowStorage::VRAM};
        const auto readers = pipeline->consumers();
        auto layout = ImmutableRowTableLayoutGeometry::checked(source_extent, ImmutableRowTableTiers::of(tiers),
            ImmutableRowTableResidency::Fixed, std::move(owners), {readers.begin(), readers.end()}, std::move(ranges), {});
        return std::shared_ptr<const ResolvedImmutableRowTablePlacement>(
            new ResolvedImmutableRowTablePlacement(std::move(layout), pipeline));
    }
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
        const bool grouped=input.consumers().size()>1;
        if((input.consumers().size()!=1 && !grouped) || source_.ple_heads!=16 ||
            std::any_of(input.consumers().begin(), input.consumers().end(), [&](const auto& c) {
                return !c.device.is_gpu() || c.device.type != input.consumers().front().device.type;
            }))
            throw std::invalid_argument("GPU adaptive admission requires singleton or exact homogeneous ConsumerLocal compute provenance");
        std::vector<bool> coordinates(input.consumers().size(), false);
        for(const auto &c:consumers_) {
            if(c.shards!=static_cast<int>(grouped?input.consumers().size():1) || c.batch!=1 || c.first_layer!=0 || c.last_layer!=47 ||
                (grouped && (c.tp_topology!=ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal ||
                    c.tp_coordinate<0 || static_cast<std::size_t>(c.tp_coordinate)>=coordinates.size() || coordinates[c.tp_coordinate])))
                throw std::invalid_argument("GPU adaptive provision lost its actual full-main consumer coordinate");
            if (grouped) coordinates[c.tp_coordinate] = true;
        }
        if (grouped && std::find(coordinates.begin(), coordinates.end(), false) != coordinates.end())
            throw std::invalid_argument("GPU adaptive provision omitted an original TP coordinate");
    }
    if(requested.isFixedHeadSharded())(void)fixedHeadPlacement(input);
    else if (!(requested.isGPUAdaptive() && input.consumers().size()>1))
      for(const auto &consumer:consumers_)
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
        const auto storage=placement->fixedHeadShardStorageBOM(
            request.input.fixedHeadPipeline() ? request.input.gpuResources() : request.input.consumers());
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
        const bool grouped = requested.gpu_adaptive.placement.mode == AdaptiveGPUCachePlacementMode::ConsumerLocal && input.consumers().size() > 1;
        const auto local_count = requested.gpu_adaptive.local_hot_cache ? input.consumers().size() : 0u;
        std::vector<std::size_t> local_slots(local_count,0u);
        std::optional<AdaptiveImmutableLookupGeometry> remote_floor;
        if (!grouped) remote_floor.emplace(AdaptiveImmutableLookupGeometry::fromLayoutFloor(
            input.gpuAdaptiveLayout(requested,0,0,maximum),source,requested.gpu_adaptive));
        const auto* originalNativeHost=request.execution_provision->requiredBase().find({input.host().world_rank,DeviceId::cpu()});
        const auto nativeHost=originalNativeHost?originalNativeHost->resource():input.host();
        const auto candidate=[&](std::span<const std::size_t> local_slots,
            std::span<const std::size_t> gpu_slots,std::size_t host_slots) {
            const auto layout=input.gpuAdaptiveLayout(requested,gpu_slots,host_slots,maximum);
            std::optional<AdaptiveImmutableLookupGeometry> geometry;
            if (!grouped)
            {
                const auto local = AdaptiveImmutableLookupGeometry::declareLocalHotCaches(
                    layout,source,requested.gpu_adaptive,local_slots);
                geometry.emplace(AdaptiveImmutableLookupGeometry::fromLayoutWithLocalHot(
                    layout,source,requested.gpu_adaptive,local));
            }
            PhysicalMemoryPlanBuilder plan(higher_priority);
            plan.add(input.host(),PhysicalMemoryOwner::ImmutableTableStorage,0u);
            for(const auto &consumer:input.consumers())plan.add(consumer,PhysicalMemoryOwner::ImmutableTableStorage,0u);
            const auto optional=grouped ? AdaptiveImmutableLookupGeometry::consumerLocalSlotDependentBOM(
                layout,source,requested.gpu_adaptive,input.gpuResources(),input.host(),request.native_ordering_events)
                : geometry->slotDependentBOMWithOrderingProjection(input.gpuResources(),input.host(),request.native_ordering_events);
            for(const auto &bom:optional.resources())plan.add(bom);
            if (!grouped) {
                const auto remote=qwen4exp::PooledImmutableRetainedParticipantBudget::slotDependentBOM(
                    *geometry,*remote_floor,nativeHost,input.gpuResources(),request.native_ordering_events);
                for(const auto& line:remote.resources())plan.add(line);
            }
            const auto hostGeometry = grouped ? AdaptiveImmutableLookupGeometry::sharedHostGeometry(layout,requested.gpu_adaptive) : geometry->host();
            bool ceilings=hostGeometry.storageBytes()<=requested.gpu_adaptive.host.cache_physical_byte_limit;
            const auto &declaration=requested.gpu_adaptive.placement;
            for(const auto &device:input.gpuResources())
            {
                std::size_t limit=requested.gpu_adaptive.gpu_cache_physical_byte_limit;
                if(declaration.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal)
                {
                    const auto owner=std::find_if(declaration.owners.begin(),declaration.owners.end(),[&](const auto& value){
                        return identity(device)==PhysicalMemoryAllocatorIdentity{value.participant.world_rank,value.participant.device()};});
                    // Source directory policy and foreign reader control are
                    // typed original-PMA costs, not undeclared cache replicas.
                    // Only actual storage owners carry per-owner cache limits.
                    if(owner==declaration.owners.end())continue;
                    limit=owner->cache_physical_byte_limit;
                }
                const auto bytes = grouped ? AdaptiveImmutableLookupGeometry::forRequester(layout,source,requested.gpu_adaptive,
                    identity(device)).gpuSlotDependentBytes(identity(device)) : geometry->gpuSlotDependentBytes(identity(device));
                ceilings=ceilings && bytes<=limit;
            }
            if (requested.gpu_adaptive.local_hot_cache)
                for (const auto& local : geometry->localHotCaches())
                {
                    const auto id = local.allocator();
                    const auto limits = requested.gpu_adaptive.local_hot_cache->ceilingsFor(
                        {id.world_rank,id.device.type,id.device.ordinal});
                    ceilings = ceilings && local.slots() <= limits.maximum_cache_rows &&
                        local.physicalBytes() <= limits.cache_physical_byte_limit;
                }
            return std::pair{ceilings,plan.build()};
        };
        const auto &declaration=requested.gpu_adaptive.placement;
        std::vector<std::size_t> gpu(declaration.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal?declaration.owners.size():input.consumers().size(),0);
        const auto zero=candidate(local_slots,gpu,0);
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
        // Use the SAME balanced, monotone grant equation for each optional
        // tier. Local grants are frozen before pool grants, then RAM; no model,
        // expert or KV grant is reopened and no capacity arithmetic is copied.
        const auto admit_balanced = [&](std::vector<std::size_t>& grants,
            const std::vector<std::uint64_t>& ceilings, const auto& fits)
        {
            if (grants.size() != ceilings.size()) throw std::logic_error("Cache grant roster differs from its original ceilings");
            if (grants.empty()) return;
            std::vector<std::size_t> individual(grants.size(),0);
            for (std::size_t owner = 0; owner < grants.size(); ++owner)
                individual[owner] = solve(ceilings[owner],[&](std::size_t rows) {
                    auto proposed = grants; proposed[owner] = rows; return fits(proposed); });
            const auto level = solve(*std::max_element(individual.begin(),individual.end()),[&](std::size_t rows) {
                auto proposed = individual;
                for (auto& slots : proposed) slots = std::min(slots,rows);
                return fits(proposed);
            });
            for (std::size_t owner = 0; owner < grants.size(); ++owner) grants[owner] = std::min(individual[owner],level);
            for (std::size_t owner = 0; owner < grants.size(); ++owner)
                grants[owner] = solve(individual[owner],[&](std::size_t rows) {
                    auto proposed = grants; proposed[owner] = rows; return fits(proposed); });
        };
        if (requested.gpu_adaptive.local_hot_cache)
        {
            std::vector<std::uint64_t> ceilings;
            for (const auto& consumer : input.consumers())
                ceilings.push_back(requested.gpu_adaptive.local_hot_cache->ceilingsFor(
                    {consumer.world_rank,consumer.device.type,consumer.device.ordinal}).maximum_cache_rows);
            admit_balanced(local_slots,ceilings,[&](const auto& proposed) {
                const auto next = candidate(proposed,gpu,0); return next.first && next.second.fits(); });
        }
        std::vector<std::uint64_t> pool_ceilings;
        for (std::size_t owner = 0; owner < gpu.size(); ++owner)
            pool_ceilings.push_back(declaration.mode == AdaptiveGPUCachePlacementMode::DeclaredRankLocal
                ? declaration.owners[owner].maximum_cache_rows : requested.gpu_adaptive.maximum_gpu_cache_rows);
        admit_balanced(gpu,pool_ceilings,[&](const auto& proposed) {
            const auto next = candidate(local_slots,proposed,0); return next.first && next.second.fits(); });
        const auto ram=solve(requested.gpu_adaptive.host.maximum_cache_rows,[&](std::size_t rows) {
            const auto next=candidate(local_slots,gpu,rows);return next.first && next.second.fits(); });
        auto complete=candidate(local_slots,gpu,ram).second;
        auto layout = input.gpuAdaptiveLayout(requested,gpu,ram,maximum);
        auto local = AdaptiveImmutableLookupGeometry::declareLocalHotCaches(layout,source,requested.gpu_adaptive,local_slots);
        auto placement=std::shared_ptr<const ResolvedImmutableRowTablePlacement>(new ResolvedImmutableRowTablePlacement(
            std::move(layout),requested.gpu_adaptive,metadata.ple_row_elements,std::move(local)));
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
