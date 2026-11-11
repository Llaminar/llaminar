/**
 * @file OriginalMappedComponentResources.h
 * @brief One original native source for a component's complete immutable replay roster.
 *
 * Temporary banks/mappings are serial alternatives. Native graph commitments
 * retain their actual creator count through ProcessRuntime; observer events
 * are created once and reused only after the old helper's real terminal.
 * Original MPI inventory selects the cohort before this owner exists. No
 * ordinal/rank alias, capacity sample, anonymous reserve or source adoption is
 * introduced by these component-only allocation and terminal helpers.
 */
#pragma once
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/OriginalRankLocalNativeReplay.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "backends/NativeAuxiliaryStreamOwnership.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEOverlayNodeLocalRouteExchange.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2::test {
/** @brief Existing geometry/policy for one serial component replay, declared before admission. */
struct OriginalMappedComponentProgram final {
    enum class Kind { Ticket, Sparse, Dense };
    Kind kind;
    std::size_t rows, width, epochs;
    bool retained=false;
    bool operator==(const OriginalMappedComponentProgram&)const=default;
    /** @return Actual logical GPU allocations in creator order, grouped by endpoint role. */
    std::array<std::vector<std::size_t>,2> deviceBanks()const {
        const auto bytes=rows*width*sizeof(float);
        if(kind==Kind::Ticket)return {{{4096,bytes,32*4096},{}}};
        if(kind==Kind::Dense)return {{{bytes},{bytes}}};
        return {{{bytes*8,rows*8*sizeof(std::int32_t),bytes,bytes*epochs,sizeof(std::int32_t),sizeof(MoENodeLocalRoutePeerDeviceBinding)},
                 {bytes*8,rows*8*sizeof(std::int32_t)}}};
    }
};
/** @brief Actual original source/PMA, event roster and program membership; never an allocator ledger. */
class OriginalMappedComponentSource final {
    using Graph=RetainedParticipantGraphExecutableFixtureAccess;
    /** @brief The installed event unit brackets its actual creator/destructor. */
    struct Event final {
        IBackend* backend;DeviceId device;void* handle=nullptr;
        std::unique_ptr<NativeOrderingEventGroup> owner;
        Event(const PreparedGPUContextUsePermit& original):backend(getBackendFor(original.device())),device(original.device()),
            owner(NativeOrderingEventGroup::claim(original.memory(),original.provenance(),NativeOrderingEventProgram::captureSubmissionRetirement())) {
            NativeOrderingEventGroupReplayAccess::begin(*owner);
            try { original.worker()->submitAndWait([&] {handle=backend->createEvent(device.ordinal);});
                if(!handle)throw std::runtime_error("Original component event creation failed"); }
            catch(...) { NativeOrderingEventGroupReplayAccess::complete(*owner);throw; }
        }
        ~Event(){if(handle)backend->destroyEvent(handle,device.ordinal);NativeOrderingEventGroupReplayAccess::complete(*owner);}
    };
public:
    /** @brief Prepare/admit one complete original cohort and all actual serial program roles. */
    OriginalMappedComponentSource(std::vector<DeviceId> devices,std::vector<OriginalMappedComponentProgram> programs)
        :devices_(std::move(devices)),programs_(std::move(programs)),
         original_(PreparedNativeLookupFixture::prepare(devices_,MPIContextFactory::global()->rank())) {
        if(programs_.empty())throw std::invalid_argument("Original component source has no declared programs");
        PhysicalMemoryPlanBuilder total;
        std::vector<PhysicalMemoryPlan> alternatives;
        std::vector<std::size_t> events(devices_.size(),1),graphs(devices_.size(),0);
        bool ticket=false;
        const auto host=original_->resource(DeviceId::cpu());
        for(const auto& program:programs_) {
            if(!program.rows || !program.width || !program.epochs)throw std::invalid_argument("Original component has empty geometry");
            // This finite protocol has one ticket role or two route roles.
            // Validate that type before indexing its fixed role-bank array.
            if(program.kind==OriginalMappedComponentProgram::Kind::Ticket) {
                if(devices_.size()!=1)throw std::invalid_argument("CPU ticket needs exactly its original GPU owner");
            } else if(devices_.size()!=2)
                throw std::invalid_argument("Local route component needs its two distinct original GPUs");
            PhysicalMemoryPlanBuilder temporary;
            const auto banks=program.deviceBanks();
            for(std::size_t role=0;role<devices_.size();++role)for(const auto bytes:banks[role])
                temporary.add(original_->resource(devices_[role]),PhysicalMemoryOwner::ExecutionWorkspace,bytes);
            if(program.kind==OriginalMappedComponentProgram::Kind::Ticket) {
                ticket=true;events[0]=34;++graphs[0];
                temporary.add(host,PhysicalMemoryOwner::ActivationTransportStaging,
                    TransferEngine::mappedHostRegionAllocationBytes(program.rows*program.width*sizeof(float))+
                    TransferEngine::mappedHostRegionAllocationBytes(32*4096));
                const std::array requests{MappedHostTransferArena::AllocationRequest{
                    MoEOverlayCanonicalRouteReturnTicketStorage::metadataLayout(program.rows).bytes,64}};
                temporary.add(host,PhysicalMemoryOwner::ActivationTransportStaging,MappedHostTransferArena::backingBytes(requests));
            } else {
                MoEOverlayNodeLocalRouteExchange::Config topology{devices_,devices_.front(),"original_component_routes"};
                const auto storage=MoEOverlayNodeLocalRouteExchange::declareStorage(topology,program.rows,8,program.width);
                const auto bom=storage.bom(host,original_->resource(devices_.front()));for(const auto& row:bom.resources())temporary.add(row);
                if(program.retained)for(auto& count:graphs)++count;
            }
            alternatives.push_back(temporary.build());
        }
        total.addMutuallyExclusive(alternatives);
        const auto add=[&](const PhysicalMemoryPlan& bom){for(const auto& row:bom.resources())total.add(row);};
        for(std::size_t role=0;role<devices_.size();++role) {
            for(std::size_t event=0;event<events[role];++event)add(NativeOrderingEventGroup::bom(
                NativeOrderingEventProgram::captureSubmissionRetirement(),host,original_->resource(devices_[role])));
            if(graphs[role])add(RetainedParticipantGraphExecutable::generalBOM(host,original_->resource(devices_[role]),graphs[role]));
        }
        if(ticket) {
            add(NativeAuxiliaryStreamOwnership::bom(host,original_->resource(devices_[0])));
            add(TransferEngine::persistentTransferExecutionLanesBOM(32,host,original_->resource(devices_[0])));
        }
        original_->admit(total.build());
        events_.resize(devices_.size());
        for(std::size_t role=0;role<devices_.size();++role)for(std::size_t event=0;event<events[role];++event)
            events_[role].push_back(std::make_unique<Event>(original_->permit(devices_[role])));
    }
    /** @brief All helper banks/graphs must retire before the actual original source reset. */
    ~OriginalMappedComponentSource()noexcept {try{events_.clear();original_->requirePayloadRetired();original_->retire();original_->requireExactRetired();}catch(...){std::terminate();}}
    /** @return Original immutable program, rejecting a helper which escaped its admission roster. */
    const OriginalMappedComponentProgram& requireProgram(const OriginalMappedComponentProgram& requested)const {
        const auto found=std::find(programs_.begin(),programs_.end(),requested);
        if(found==programs_.end())throw std::logic_error("Component helper changed its admitted immutable program");return *found;
    }
    /** @return Actual selected source resource index, never inferred from native ordinal. */
    std::size_t index(DeviceId device)const {auto found=std::find(devices_.begin(),devices_.end(),device);
        if(found==devices_.end())throw std::logic_error("Component borrowed a foreign original GPU");return found-devices_.begin();}
    /** @return Same original source permit used by every creator/submission. */
    const PreparedGPUContextUsePermit& permit(DeviceId device)const {index(device);return original_->permit(device);}
    /** @return Sole admitted original physical allocation issuer. */
    const std::shared_ptr<PhysicalMemoryAuthority>& memory()const{return original_->memory();}
    /** @return Same original publication retained by the actual MPI context. */
    auto inventory()const{return MPIContextFactory::global()->clusterInventory();}
    /** @return Original prepared stream; progress is separate only for the one-GPU ticket proof. */
    void* stream(DeviceId device,bool progress=false)const {
        const auto& original=permit(device);if(!progress)return original.worker()->defaultStream();
        void* result=nullptr;original.worker()->submitAndWait([&]{result=original.worker()->getOrCreateAdmittedAuxiliaryStream(
            "canonical_ticket_progress",original);});return result;
    }
    /** @return One reusable actual event, after the previous helper retired all users. */
    void* event(DeviceId device,std::size_t ordinal)const {return events_.at(index(device)).at(ordinal)->handle;}
private:
    std::vector<DeviceId> devices_;std::vector<OriginalMappedComponentProgram> programs_;
    std::unique_ptr<PreparedNativeLookupFixture> original_;
    std::vector<std::vector<std::unique_ptr<Event>>> events_;
};
}
