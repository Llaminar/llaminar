/**
 * @file OriginalMappedLoweringAdmission.h
 * @brief Exact original-rank native admission for the existing distributed lowering fixtures.
 *
 * This fixture owns no alternate transport or live ledger. It binds observed
 * original source/peer domains, prepares only each rank's own native contexts,
 * then merges the actual channel/source/epoch declarations into that SAME PMA.
 * Graph/model borrowers must remain lexically inside this owner.
 */
#pragma once
#include "fixtures/backends/OriginalMappedNativeCohorts.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "execution/moe/MoEOverlayActivationBatchInventory.h"
#include "execution/moe/MoEOverlayActivationBatchResources.h"
#include "execution/moe/MoEOverlayActivationChannelPlan.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "execution/moe/MoEOverlayRankBatchPreparation.h"
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "planning/PlanningPublication.h"
#include "utils/MPIContext.h"
#include <map>
#include <set>

namespace llaminar2::test {
/** @brief One exact original rank-pair setup and terminal lifetime. */
class OriginalMappedLoweringAdmission final {
public:
    /** @brief Observe original cohorts before selected worker birth.
     * @param source Exact source backend/width.
     * @param peer Exact peer backend/width.
     * @throws std::exception For absent original geometry or invalid worker birth. */
    OriginalMappedLoweringAdmission(OriginalMappedNativeDomainRequest source,OriginalMappedNativeDomainRequest peer)
        :mpi_(MPIContextFactory::global()),cohorts_(OriginalMappedNativeCohorts::observe(mpi_->clusterInventory(),source,peer)) {
        checkpoint([&]{if(!mpi_->topology())throw std::logic_error("Original lowering topology is unavailable");
            const auto devices=cohorts_.localDevices(mpi_->rank());
            if(!devices.empty())original_=PreparedNativeLookupFixture::prepare(devices,mpi_->rank());});
    }
    /** @brief Keep the genuine peer alive through source graph/model retirement. */
    ~OriginalMappedLoweringAdmission() noexcept {
        try {checkpoint([&]{});registry_.reset();controller_claim_.reset();original_.reset();}
        catch(...){std::terminate();}
    }
    /** @return Same original full discovery/transport context. */
    const std::shared_ptr<IMPIContext>& mpi()const noexcept{return mpi_;}
    /** @return Authenticated physical source and peer domain binding. */
    const OriginalMappedNativeCohorts& cohorts()const noexcept{return cohorts_;}
    /** @return Whether this original rank owns the actual continuation. */
    bool source()const noexcept{return mpi_->rank()==cohorts_.sourceRank();}
    /** @return Same original admitted source authority, never a fixture-created replacement. */
    const std::shared_ptr<PhysicalMemoryAuthority>& memory()const{return original_->memory();}
    /** @return Original physical observation retained before worker birth. */
    const PhysicalMemoryResource& resource(DeviceId device)const{return original_->resource(device);}
    /** @return Exact selected original worker capability for one local participant. */
    const PreparedGPUContextUsePermit& permit(DeviceId device)const{return original_->permit(device);}
    /** @return The actual pre-rendezvoused channel registry, retaining original GPU grants. */
    const std::shared_ptr<MoEOverlayRankBatchTransportRegistry>& registry()const{return registry_;}
    /** @brief Execute setup/terminal work under original all-rank failure consensus.
     * @param body Work performed once by each actual original rank.
     * @throws std::exception For a local or remote failed phase. */
    void checkpoint(const std::function<void()>& body)const {acceptPlanningCostPreparation(mpi_,body);}
    /** @brief Bind the two existing declared domains to observed physical owners.
     * @param plan Existing source/model placement, preserving all tier and expert policies.
     * @throws std::invalid_argument If the fixture no longer declares exactly its original two domains. */
    void bind(MoERoutedExpertPlacementPlan& plan)const {
        if(plan.domains.size()!=2)throw std::invalid_argument("Distributed lowering requires its two authored original domains");
        for(size_t role=0;role<2;++role){auto& domain=plan.domains[role];const auto& devices=role?cohorts_.peerDevices():cohorts_.sourceDevices();
            domain.owner_rank=role?cohorts_.peerRank():cohorts_.sourceRank();domain.world_ranks.clear();domain.participants.clear();
            for(const auto device:devices)domain.participants.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));}
    }
    /** @brief Admit and construct only each original rank's local endpoint.
     * @param plan Exact bound placement.
     * @param family Actual model/source layer manifest retained by both endpoints.
     * @param sources Exactly constructed continuation Batch sources.
     * @param rows Actual admitted activation-row envelope.
     * @param width Actual hidden width.
     * @param topk Actual route fanout.
     * @param controller_bytes Exact current-rank first-touch pages from original fabric planLayout.
     * @param extra Additional typed source contributions such as an admitted mapped local route exchange.
     * @throws std::exception For invalid declaration, admission or native endpoint construction. */
    void prepare(const MoERoutedExpertPlacementPlan& plan,const MoEOverlayInferenceGraphFamilyIdentity& family,
        std::shared_ptr<const MoEOverlayActivationBatchInventory> sources,int rows,int width,int topk,
        size_t controller_bytes=0,const PhysicalMemoryPlan* extra=nullptr) {
        const auto channels=MoEOverlayActivationChannelPlanner::plan({.placement_plan=&plan,
            .cluster_inventory=cohorts_.inventory().get(),.row_capacity=size_t(rows),.d_model=width,.top_k=topk,
            .graph_family_count=makeMoEOverlayActivationGraphFamilyManifests(family).size(),
            .original_native_slots=MoEOverlayActivationChannelPlanner::originalNativeSlotGeometry(plan,*cohorts_.inventory(),family)});
        if(channels.channels.size()!=1 || !sources)throw std::invalid_argument("Lowering fixture changed its original channel/source roster");
        const auto& channel=channels.channels.front();
        checkpoint([&]{if(original_){
            const auto host=original_->resource(DeviceId::cpu());PhysicalMemoryPlanBuilder bill;
            const auto add=[&](const PhysicalMemoryPlan& contribution){for(const auto& resource:contribution.resources())bill.add(resource);};
            for(const auto& charge:channels.staging_charges)if(charge.world_rank==mpi_->rank())
                bill.add(original_->resource(charge.device),PhysicalMemoryOwner::ActivationTransportStaging,charge.bytes);
            if(source())for(const auto device:cohorts_.sourceDevices())bill.add(original_->resource(device),PhysicalMemoryOwner::ExecutionWorkspace,
                DeviceMoEOverlayEpochArena::allocationBytesForSlots(1u)+sizeof(int32_t));
            std::set<DeviceId> cold;
            if(source()){
                const auto gpu=original_->resource(channel.source_device);
                bool asynchronous=false;
                for(const auto& entry:sources->sources())for(size_t ordinal=0;ordinal<entry.model_layers.size();++ordinal){
                    add(MoEOverlayActivationBatchResources::bom(std::span(&channel,1),entry.physical_rows,uint32_t(ordinal),host,gpu));
                    asynchronous|=entry.physical_rows>1;}
                if(asynchronous){add(MoEOverlayActivationBatchResources::sharedLaneStreamsBOM(std::span(&channel,1),host,gpu));cold.insert(channel.source_device);}
            }
            for(const auto& lane:channel.localLanes(mpi_->rank()))if(lane.device.is_gpu() && cold.insert(lane.device).second)
                add(MoEOverlayActivationBatchResources::coldPublicationBOM(host,original_->resource(lane.device)));
            if(controller_bytes)bill.add(host,PhysicalMemoryOwner::ActivationTransportStaging,controller_bytes);
            if(extra)add(*extra);
            original_->admit(bill.build());
            if(controller_bytes)controller_claim_.emplace(memory()->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ActivationTransportStaging,controller_bytes));
        }});
        const auto owner_map=MoEExpertOwnerMap::build(plan);
        registry_=prepareOriginalMoEOverlayRankBatchTransports({plan,owner_map,*cohorts_.inventory(),family,
            mpi_,memory(),sources,{},width,topk});
        if(!registry_)throw std::logic_error("Original lowering transport registry was not prepared");
    }
private:
    std::shared_ptr<IMPIContext> mpi_;
    OriginalMappedNativeCohorts cohorts_;
    std::unique_ptr<PreparedNativeLookupFixture> original_;
    std::optional<PhysicalMemoryAllocationLease> controller_claim_;
    std::shared_ptr<MoEOverlayRankBatchTransportRegistry> registry_;
};
}
