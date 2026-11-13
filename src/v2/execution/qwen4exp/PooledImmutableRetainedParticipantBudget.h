/**
 * @file PooledImmutableRetainedParticipantBudget.h
 * @brief Source-authored remote executable and directed channel obligations in the original candidate BOM.
 *
 * Each independently runnable remote owner needs one existing General device
 * slot and its declared mapped certificate or native-General host purpose. These are immutable estimator
 * contributions to the original PMA, never a live capacity ledger. Optional
 * geometry is evaluated before selection; no creator adds an after-quota cost.
 */
#pragma once
#include <algorithm>
#include <type_traits>
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "PooledImmutableDirectedOwnerProgram.h"
#include "PooledImmutableDirectedTransport.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/NativeOrderingEventBudgetProjection.h"
namespace llaminar2::qwen4exp {
/** @brief Complete per-model remote definitions shared by exclusive source variants. */
class PooledImmutableRetainedParticipantBudget final {
public:
    /** @brief Project every actual nonlocal owner through its explicit mapped-flat or native-General contract. */
    static PhysicalMemoryPlan completeBOM(const AdaptiveImmutableLookupGeometry& geometry,
        PhysicalMemoryResource originalCPU0,std::span<const PhysicalMemoryResource> originalGPUs,
        const NativeOrderingEventBudgetProjection& source=NativeOrderingEventBudgetProjection::installed()){
        PhysicalMemoryPlanBuilder result;
        const auto& physical=geometry.physical();
        // The same owner cohort creates both directed channels before capture.
        // Price their CPU0 mapped banks and endpoint GPU cursors here so every
        // floor, optional growth, resolved plan and wire restore retains them.
        const auto channels=PooledImmutableDirectedTransport::completeBOM(
            physical,geometry.coordinator(),originalCPU0,originalGPUs,
            geometry.readerGeometry() ? &*geometry.readerGeometry() : nullptr,
            geometry.localHotTransport()?PooledNativeReaderMessageProfile::CompactNative:PooledNativeReaderMessageProfile::ReplicatedFP32);
        for(const auto& line:channels.resources())result.add(line);
        bool sourceHolder=false;
        for(std::uint32_t coordinate=0;coordinate<physical.owners().size();++coordinate){
            const auto& owner=physical.owners()[coordinate];
            if(owner.allocator==physical.coordinator())continue;
            if (geometry.readerGeometry() && geometry.readerGeometry()->nativeCollective() &&
                owner.allocator.device.type == physical.coordinator().device.type &&
                std::find(geometry.readerGeometry()->consumers().begin(), geometry.readerGeometry()->consumers().end(), owner.allocator) !=
                    geometry.readerGeometry()->consumers().end())
                continue; // Same actual continuation parent owns these inline stages; no second SDK executable/host class exists.
            if(originalCPU0.device!=DeviceId::cpu() || originalCPU0.world_rank!=owner.allocator.world_rank)
                throw std::invalid_argument("Remote graph planning requires its original rank CPU0 observation");
            const auto gpu=std::find_if(originalGPUs.begin(),originalGPUs.end(),[&](const auto& resource){
                return resource.device==owner.allocator.device && resource.world_rank==owner.allocator.world_rank;});
            if(gpu==originalGPUs.end())throw std::invalid_argument("Remote graph planning lost an original owner observation");
            if(!sourceHolder){
                result.add(originalCPU0,PhysicalMemoryOwner::NativeStorageGraphExecutable,
                    RetainedParticipantGraphSourceSlots::physicalBytes(physical.owners().size()));
                sourceHolder=true;
            }
            const auto program=PooledImmutableDirectedOwnerProgram::declareStorage(physical,geometry.coordinator(),coordinate);
            std::visit([&](const auto& contract){
                using Contract=std::decay_t<decltype(contract)>;
                if constexpr(std::is_same_v<Contract,PooledRemoteGraphProgram>){
                    const auto host=source.remoteHostBOM(contract,originalCPU0,*gpu);
                    for(const auto& line:host.resources())result.add(line);
                    result.add(*gpu,PhysicalMemoryOwner::NativeStorageGraphExecutable,
                        GPUGraphMemoryContract::reservationBytesPerExecutable(gpu->device));
                }else{
                    // Native-library graphs consume the existing full General
                    // contract under their own storage purpose. No flat class,
                    // model reservation or caller-sized allowance can replace it.
                    const auto native=RetainedParticipantGraphExecutable::nativeStorageBOM(contract,originalCPU0,*gpu);
                    for(const auto& line:native.resources())result.add(line);
                }
            },program);
        }
        return result.build();
    }
    /** @brief Add only positive growth above the same declared zero-slot program floor. */
    static PhysicalMemoryPlan slotDependentBOM(const AdaptiveImmutableLookupGeometry& geometry,
        const AdaptiveImmutableLookupGeometry& floor,PhysicalMemoryResource originalCPU0,
        std::span<const PhysicalMemoryResource> originalGPUs,
        const NativeOrderingEventBudgetProjection& source=NativeOrderingEventBudgetProjection::installed()){
        const auto actual=completeBOM(geometry,originalCPU0,originalGPUs,source);
        const auto minimum=completeBOM(floor,originalCPU0,originalGPUs,source);
        PhysicalMemoryPlanBuilder result;
        for(const auto& line:actual.resources()){
            const auto* baseline=minimum.find({line.resource().world_rank,line.resource().device});
            if(!baseline)throw std::logic_error("Remote graph candidate changed its original resource set");
            for(std::size_t index=0;index<PhysicalMemoryBOM::ownerCount();++index){
                const auto owner=static_cast<PhysicalMemoryOwner>(index);
                if(line.bytes(owner)<baseline->bytes(owner))
                    throw std::logic_error("Remote graph class is smaller than its declared mandatory floor");
                result.add(line.resource(),owner,line.bytes(owner)-baseline->bytes(owner));
            }
        }
        return result.build();
    }
};
}
