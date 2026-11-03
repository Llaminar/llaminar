/**
 * @file ImmutableLookupSetupObservation.h
 * @brief Passive immutable admitted-capacity evidence from original lookup roots and PMA leases.
 *
 * This historical setup value is not occupancy, a lookup receipt, a reservation,
 * or permission to reuse retired storage. Every row describes disjoint original
 * physical roots. Consumer borrows never duplicate a shared host source row.
 */
#pragma once
#include "ImmutableRowTableLayoutGeometry.h"
#include "HostImmutableRowMode.h"
#include "CpuAdaptiveImmutableRowCache.h"
#include "HostImmutableRowOwnerObservation.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "utils/PerfStatsCollector.h"
#include "execution/qwen4exp/ImmutableLookupService.h"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>
namespace llaminar2 {
/** @brief Immutable admitted lookup capacities, captured only after original placement validation. */
class ImmutableLookupSetupObservation final {
    /** @brief Stable physical ordering used only to reject incomplete setup membership. */
    struct AllocatorLess {bool operator()(const PhysicalMemoryAllocatorIdentity& a,const PhysicalMemoryAllocatorIdentity& b)const {
        return a.world_rank!=b.world_rank?a.world_rank<b.world_rank:a.device<b.device;
    }};
public:
    /** @brief Disjoint physical allocation roles; these say nothing about executable ownership. */
    enum class Role {GPUCache,GPUPolicy,GPUSourceStaging,SharedHostSource};
    /** @brief One original lease-backed capacity row, with no mutable cache state. */
    struct Owner {
        PhysicalMemoryAllocatorIdentity allocator;
        Role role;
        std::size_t admitted_slots=0,payload_bytes=0,metadata_bytes=0;
        std::size_t table_storage_bytes=0,lookup_staging_bytes=0,source_storage_bytes=0;
        bool operator==(const Owner&)const=default;
    };
    /**
     * @brief Capture exact original ready adaptive roots without performing native operations.
     * @param layout Original resolved full layout, never configured ceilings.
     * @param service Original shared provider/channel owner; its binding authenticates source coordinates and generation.
     * @param hostGeometry Original checked host geometry; its existing subranges only partition actual leases.
     * @param sources Exactly one original source per declared CPU channel; shared readers borrow it.
     * @return Immutable historical capacities; rows own no allocation or mutable authority.
     * @throws std::exception For partial, duplicate, foreign, stale or incompatible roots/leases.
     */
    static ImmutableLookupSetupObservation captureAdaptive(const ImmutableRowTableLayoutGeometry& layout,
        const CpuAdaptiveImmutableRowCacheGeometry& hostGeometry,const std::shared_ptr<qwen4exp::ImmutableLookupService>& service,
        std::span<const std::shared_ptr<const qwen4exp::PooledImmutableSourcePreparedStorage>> sources)
    {
        if(!service || sources.empty() || !sources.front())throw std::invalid_argument("Lookup setup has no complete original service/source");
        const auto binding=service->bindSubordinate(sources.front()->geometry().allocator().device);
        const auto& source=binding.source();
        const auto mode=service->mode();
        const auto host=service->retainedSourceOwnerObservation();
        if(!hostImmutableRowModeIsAdaptive(mode) || !host.authority() || sources.empty() ||
            layout.residency()!=ImmutableRowTableResidency::Adaptive || layout.sourceServices().size()!=1 ||
            host.allocator()!=layout.sourceServices().front().allocator ||
            source.rows!=layout.extent().rows() || source.rowBytes()!=layout.extent().rowBytes())
            throw std::invalid_argument("Lookup setup requires its complete original adaptive source/layout/PMA");
        const auto sourceStorage=layout.sourceServices().front().windows.storage();
        if((mode==HostImmutableRowMode::AdaptiveFullRAM)!=(sourceStorage==ImmutableRowStorage::RAM) ||
            (mode==HostImmutableRowMode::AdaptiveFullRAM && hostGeometry.slots()))
            throw std::invalid_argument("Lookup setup backing/cache differs from original source");
        const bool local=layout.adaptiveGPUDomain()==ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal;
        std::set<PhysicalMemoryAllocatorIdentity,AllocatorLess> expectedSources;
        if(local)expectedSources.insert(layout.consumers().begin(),layout.consumers().end());
        else {
            if(!sources.front())throw std::invalid_argument("Lookup setup has a null original source");
            if(sources.size()!=1)throw std::invalid_argument("Shared lookup setup requires one original source");
            // C0 still has its original source allocator even without an optional policy coordinator.
            expectedSources.insert(sources.front()->geometry().allocator());
            if(std::find(layout.consumers().begin(),layout.consumers().end(),*expectedSources.begin())==layout.consumers().end())
                throw std::invalid_argument("Lookup setup source is not an original consumer");
            if(layout.gpuCoordinator() && *layout.gpuCoordinator()!=*expectedSources.begin())
                throw std::invalid_argument("Lookup setup replaced its authored coordinator");
        }
        if(sources.size()!=expectedSources.size())throw std::invalid_argument("Lookup setup omitted an original source channel");
        ImmutableLookupSetupObservation result(mode,source,binding.storageGeneration());
        std::set<PhysicalMemoryAllocatorIdentity,AllocatorLess> sourceAllocators,ownerAllocators;
        for(const auto& original:sources) {
            if(!original)throw std::invalid_argument("Lookup setup has a null original source");
            service->bindSubordinate(original->geometry().allocator().device).requireFor(original->geometry());
            const auto identity=original->captureIdentity();
            const auto allocator=original->geometry().allocator();
            if(!expectedSources.contains(allocator) || !sourceAllocators.insert(allocator).second)
                throw std::invalid_argument("Lookup setup source membership is incomplete or duplicate");
            const auto aggregate=original->retainedOwnerObservation();
            requireAuthority(host,aggregate);
            std::size_t constituentStorage=0,constituentStaging=0;
            for(const auto& owner:original->originalOwners()) {
                if(!owner)throw std::invalid_argument("Lookup setup lost an original physical owner");
                const auto observed=owner->retainedOwnerObservation();requireAuthority(host,observed);
                const auto& p=owner->plan().parameters();
                const auto declared=std::find_if(layout.owners().begin(),layout.owners().end(),[&](const auto& row){
                    return row.key.tier==ImmutableRowStorage::VRAM && row.key.allocator==observed.allocator();});
                if(declared==layout.owners().end() || declared->cache_rows!=p.owner.capacity() ||
                    !ownerAllocators.insert(observed.allocator()).second || p.cache.payload>p.cache.bytes ||
                    observed.tableStorageBytes()!=p.cache.bytes)
                    throw std::invalid_argument("Lookup setup physical grants differ from original admitted roots");
                result.rows_.push_back({observed.allocator(),Role::GPUCache,p.owner.capacity(),
                    p.cache.bytes-p.cache.payload,p.cache.payload,observed.tableStorageBytes(),observed.lookupStagingBytes()});
                if(observed.allocator()==allocator){constituentStorage=add(constituentStorage,observed.tableStorageBytes());constituentStaging=add(constituentStaging,observed.lookupStagingBytes());}
            }
            const auto coordinator=original->coordinator();
            if(!coordinator)throw std::invalid_argument("Lookup setup lost original policy roots");
            const auto observed=coordinator->retainedOwnerObservation();requireAuthority(host,observed);
            if(observed.allocator()!=allocator)throw std::invalid_argument("Lookup setup replaced original policy allocator");
            result.rows_.push_back({allocator,Role::GPUPolicy,0,0,observed.tableStorageBytes(),observed.tableStorageBytes(),observed.lookupStagingBytes()});
            constituentStorage=add(constituentStorage,observed.tableStorageBytes());constituentStaging=add(constituentStaging,observed.lookupStagingBytes());
            if(aggregate.tableStorageBytes()!=constituentStorage || aggregate.lookupStagingBytes()<constituentStaging)
                throw std::logic_error("Lookup setup aggregate does not partition its original physical leases");
            result.rows_.push_back({allocator,Role::GPUSourceStaging,0,0,0,0,aggregate.lookupStagingBytes()-constituentStaging});
            if(original->captureIdentity()!=identity)throw std::logic_error("Lookup setup original source changed during capture");
        }
        const auto declaredOwners=std::count_if(layout.owners().begin(),layout.owners().end(),[](const auto& row){return row.key.tier==ImmutableRowStorage::VRAM;});
        if(ownerAllocators.size()!=static_cast<std::size_t>(declaredOwners))throw std::invalid_argument("Lookup setup omitted an original physical owner");
        // Host storage may contain a full RAM table or bounded cache+SSD service.
        // Preserve its actual aggregate lease; do not reinterpret it as GPU payload.
        result.rows_.push_back(hostOwner(hostGeometry,service));
        return result;
    }
    /**
     * @brief Project the shared host's original named source/cache ranges once.
     * @param geometry Existing checked cache/source declaration, never requested ceilings.
     * @param service Actual retained source/channel owner.
     * @return Disjoint exact host lease row; two/four consumer borrows do not multiply it.
     * @throws std::exception For missing/stale service or incompatible lease/range partitions.
     */
    static Owner hostOwner(const CpuAdaptiveImmutableRowCacheGeometry& geometry,
        const std::shared_ptr<qwen4exp::ImmutableLookupService>& service) {
        if(!service || !hostImmutableRowModeIsAdaptive(service->mode()))throw std::invalid_argument("Lookup setup needs its original adaptive host owner");
        const auto host=service->retainedSourceOwnerObservation();
        if(host.tableStorageBytes()!=add(geometry.sourceBytes(),geometry.storageBytes()) ||
            host.lookupStagingBytes()<geometry.executionBytes() ||
            (service->mode()==HostImmutableRowMode::AdaptiveFullRAM)!=(geometry.sourceStorage()==ImmutableRowStorage::RAM) ||
            (service->mode()==HostImmutableRowMode::AdaptiveFullRAM && geometry.slots()))
            throw std::invalid_argument("Lookup setup host leases do not match their original named source/cache regions");
        return {host.allocator(),Role::SharedHostSource,geometry.slots(),geometry.cachePayloadStorageBytes(),
            geometry.cacheMetadataBytes(),host.tableStorageBytes(),host.lookupStagingBytes(),geometry.sourceBytes()};
    }
    /** @return Disjoint immutable setup rows; subsequent request reset changes none of them. */
    std::span<const Owner> owners()const noexcept{return rows_;}
    /** @return Original provider mode, distinct from configured requested backing. */
    HostImmutableRowMode mode()const noexcept{return mode_;}
    /** @return Original immutable source metadata; not a new file identity or authority. */
    const ImmutableRowTableSource& source()const noexcept{return source_;}
    /** @return Whether a repeated Ready observation retains exactly the same immutable capacities and source. */
    bool sameCapacity(const ImmutableLookupSetupObservation& other)const {
        return mode_==other.mode_ && source_generation_==other.source_generation_ && rows_==other.rows_ && source_.path==other.source_.path &&
            source_.tensor_name==other.source_.tensor_name && source_.file_offset==other.source_.file_offset &&
            source_.rows==other.source_.rows && source_.row_elements==other.source_.row_elements && source_.format==other.source_.format;
    }
    /** @brief Emit only the exact setup-owned family; request work never enters these rows. */
    void publish()const {
        if(!PerfStatsCollector::isDomainEnabled("immutable_lookup_setup"))return;
        for(const auto& row:rows_) {
            const PerfStatsCollector::Tags tags{{"world_rank",std::to_string(row.allocator.world_rank)},
                {"role",roleName(row.role)},{"admitted_slots",std::to_string(row.admitted_slots)},
                {"payload_bytes",std::to_string(row.payload_bytes)},{"metadata_bytes",std::to_string(row.metadata_bytes)},
                {"table_storage_bytes",std::to_string(row.table_storage_bytes)},{"source_storage_bytes",std::to_string(row.source_storage_bytes)},{"lookup_staging_bytes",std::to_string(row.lookup_staging_bytes)},
                {"storage",std::string(hostImmutableRowModeName(mode_))},{"source_generation",std::to_string(source_generation_)},{"source_file_offset",std::to_string(source_.file_offset)},
                {"source_path",source_.path},{"source_tensor",source_.tensor_name},{"source_format",std::to_string(static_cast<unsigned>(source_.format))},{"source_row_bytes",std::to_string(source_.rowBytes())},{"source_rows",std::to_string(source_.rows)},{"source_row_elements",std::to_string(source_.row_elements)}};
            PerfStatsCollector::addCounter("immutable_lookup_setup","admitted_owner",1,"setup",row.allocator.device.toString(),tags);
        }
    }
private:
    /** @brief Retain only historical values; native owners are not extended by evidence consumers. */
    ImmutableLookupSetupObservation(HostImmutableRowMode mode,ImmutableRowTableSource source,std::uint64_t generation)
        :mode_(mode),source_(std::move(source)),source_generation_(generation){}
    /** @brief Validate identical original PMA control block, not just equal planned capacity. */
    static void requireAuthority(const HostImmutableRowOwnerObservation& host,const ImmutableLookupNativeOwnerObservation& gpu) {
        if(host.authority().get()!=gpu.authority().get() || host.authority().owner_before(gpu.authority()) || gpu.authority().owner_before(host.authority()))
            throw std::invalid_argument("Lookup setup cannot combine foreign PMA owners");
    }
    /** @return Checked evidence aggregation only; this never decides admission or capacity. */
    static std::size_t add(std::size_t a,std::size_t b){if(b>std::numeric_limits<std::size_t>::max()-a)throw std::overflow_error("Lookup setup byte overflow");return a+b;}
    /** @return Closed stable role label. */
    static const char* roleName(Role role){switch(role){case Role::GPUCache:return "gpu_cache";case Role::GPUPolicy:return "gpu_policy";case Role::GPUSourceStaging:return "gpu_source_staging";case Role::SharedHostSource:return "shared_host_source";}throw std::invalid_argument("Unknown lookup setup role");}
    HostImmutableRowMode mode_;
    ImmutableRowTableSource source_;
    std::uint64_t source_generation_; ///< Existing service-issued generation, never a new evidence identity.
    std::vector<Owner> rows_;
};
} // namespace llaminar2
