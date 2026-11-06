/**
 * @file MoERecurrentOriginalSlotPacketReplay.inc
 * @brief Two-physical-GPU captured packet replay over original submitted slots.
 *
 * One retained R1 executable consumes several immutable original admissions
 * before any host terminal observation. Packet stages perform the production
 * begin and post-payload retirement. The source has one aggregate stream
 * frontier; follower cache events retain the actual original slot authority.
 * Disposal follows all native receipts; this component cannot mint the PGR-only
 * aggregate-clear permit and does not certify next-request rearm or learned R1.
 */
#include "backends/NativeContextPreparation.h"
#include "backends/NativeOrderingEventGroup.h"
#include "execution/local_execution/graph/OriginalSlotCaptureTerminals.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/compute_stages/stages/MoEOverlayEpochBoundaryStage.h"
#include "execution/compute_stages/stages/MoEOverlayActivationPacketStages.h"
#include "execution/compute_stages/stages/ResidualAddStage.h"
#include "fixtures/backends/FollowerPredecessorDiagnosticGraphResources.h"
#include "memory/CPUStorageRoot.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "execution/moe/MoEOverlayActivationChannelPlan.h"

namespace llaminar2::test { namespace {
/** @brief Reject incomplete native construction without advertising a skipped proof. */
void requireRecurrentPacket(bool good,const char *reason)
{ if(!good)throw std::runtime_error(reason); }

/** @return Actual startup allocator observation for this original source. */
PhysicalMemoryResource recurrentPacketResource(DeviceId device)
{
    auto &manager=DeviceManager::instance();manager.initialize(-1,false);
    const auto kind=device.is_cpu()?ComputeBackendType::CPU:device.is_cuda()?ComputeBackendType::GPU_CUDA:ComputeBackendType::GPU_ROCM;
    const auto &inventory=manager.devices();
    const auto found=std::find_if(inventory.begin(),inventory.end(),[&](const auto &entry){
        return entry.type==kind && (device.is_cpu() || entry.device_id==device.ordinal);});
    requireRecurrentPacket(found!=inventory.end(),"Original recurrent packet endpoint is absent");
    return {0,device,found->total_memory_bytes,found->free_memory_bytes};
}

/**
 * @brief Exact same-PMA tensor roots; GPU claims precede TransferEngine storage.
 * @details All tensors are cold inputs/outputs. Diagnostic snapshots never
 * authorize a packet or replace its actual captured arithmetic.
 */
struct RecurrentPacketTensor final {
    CPUStorageRoot<float> root;
    std::optional<PhysicalMemoryAllocationLease> device_claim;
    std::unique_ptr<FP32Tensor> tensor;
    RecurrentPacketTensor(std::size_t count):root(count,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()){}
    /** @brief Declare exact CPU allocator extent and addressed GPU payload. */
    void declare(PhysicalMemoryPlanBuilder &plan,PhysicalMemoryResource host,PhysicalMemoryResource gpu)const {
        plan.add(host,PhysicalMemoryOwner::ActivationArena,root.physicalBytes());
        plan.add(gpu,PhysicalMemoryOwner::ActivationArena,root.capacity()*sizeof(float));
    }
    /**
     * @brief Materialize original roots once before capture.
     * @param initialize Optional cold producer invoked on owned host storage
     * before its FIRST asynchronous upload; the callback is never retained.
     */
    void prepare(const std::shared_ptr<PhysicalMemoryAuthority>& memory,DeviceId device,void *stream,
        std::size_t rows,std::size_t cols,float initial=0.0f,
        void (*initialize)(std::span<float>)=nullptr) {
        tensor=std::make_unique<FP32Tensor>(std::vector<std::size_t>{rows,cols},root.allocate(memory),DeviceId::cpu());
        std::fill_n(tensor->mutable_typed_data(),root.capacity(),initial);
        if(initialize)initialize({tensor->mutable_typed_data(),root.capacity()});
        device_claim.emplace(memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,root.capacity()*sizeof(float)));
        TransferEngine::prepareDeviceInput(tensor.get(),device,stream);
    }
};

/**
 * @brief Cold identity controls never issue admission or materialize changed lanes.
 * @tparam Stage One of the four actual scalar packet stage types.
 */
template<class Stage> void checkRecurrentPacketIdentity(const Stage &original) {
    const auto base=original.graphCaptureVariantSignature();
    const auto check=[&](auto edit) {
        auto params=original.getParams();edit(params.lane);
        Stage changed(std::move(params));
        EXPECT_NE(base,changed.graphCaptureVariantSignature());
    };
    check([](auto &lane){++lane.original_invocations.original_slots;--lane.original_invocations.original_slot_count;});
    check([](auto &lane){--lane.original_invocations.original_slot_count;});
    check([](auto &lane){++lane.retained_stage_count;});
    check([](auto &lane){lane.original_invocations={};});
}

/** @brief Closed replay inputs; every case executes the same production packet path. */
enum class RecurrentPacketProfile : std::uint8_t { Full, PartialVerifier, Empty };

/**
 * @brief Own one source process, two actual GPUs and two finite packet families.
 * @details Context/PMA populations cannot coexist with the legacy unprepared
 * packet fixture. Explicit isolated CTests select these disabled-by-default
 * bodies; no alternate native launcher or authority is introduced.
 */
class RecurrentOriginalSlotPacketReplay final {
    using Cache=DeviceGraphExecutor::GraphSegmentCache;
    static constexpr int width=8,depth=3,target=1;
    static constexpr std::size_t families=2,invocations=4;
    struct Family final {
        explicit Family(int rows):rows(rows),hidden(rows*width),indices(rows),weights(rows),
            canonical(rows*width),zero(rows*width),increment(rows*width),received(rows*width),received_indices(rows),received_weights(rows),local(rows*width){}
        int rows;
        RecurrentPacketTensor hidden,indices,weights,canonical,zero,increment,received,received_indices,received_weights,local;
        std::array<MoEOverlayMappedActivationDeviceLane,2> lanes;
        std::array<std::vector<std::shared_ptr<IComputeStage>>,2> stages;
        std::array<std::vector<std::uint64_t>,2> signatures;
        std::array<bool,2> native_attempted{};
        std::array<std::unique_ptr<IGPUGraphCapture>,2> graphs;
        std::array<std::optional<PhysicalMemoryAllocationLease>,2> graph_claims;
        Cache follower_cache;
        std::array<void*,2> active{};
        std::array<std::optional<PhysicalMemoryAllocationLease>,2> active_claims;
    };
public:
    /** @brief Select two physical devices of one backend; never emulate ranks. */
    explicit RecurrentOriginalSlotPacketReplay(DeviceId source_device,RecurrentPacketProfile profile=RecurrentPacketProfile::Full)
        :devices_{source_device,source_device.is_cuda()?DeviceId::cuda(1):DeviceId::rocm(1)},
         geometry_(MoEOverlayInferenceOriginalSlotGeometry::forRetainedDraftCapacity(depth)),
         family_{std::make_unique<Family>(1),std::make_unique<Family>(4)},evidence_root_((3+4)*width,
             PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),profile_(profile) {}

    /** @brief Native failure remains fatal if original absorbing cancellation cannot drain. */
    ~RecurrentOriginalSlotPacketReplay() {
        try { dispose(); } catch(...) { std::terminate(); }
    }

    /** @brief Admit exact cold graph/event/tensor populations through original PMA. */
    void prepare() {
        const auto host=recurrentPacketResource(DeviceId::cpu());
        std::array gpu{recurrentPacketResource(devices_[0]),recurrentPacketResource(devices_[1])};
        const std::array resources{host,gpu[0],gpu[1]};
        source_=NativeContextPreparation::create(resources,0,devices_);
        std::array<std::optional<PreparedGPUContextUsePermit>,2> permits;
        for(unsigned p=0;p<2;++p)permits[p].emplace(source_->acquire(devices_[p]));
        memory_=source_->memory();source_->freezeForPlanning();
        PhysicalMemoryPlanBuilder plan(source_->mandatoryBOM());
        const auto program=NativeOrderingEventProgram::captureSubmissionRetirement();
        // Exactly two raw executable creators on EACH physical endpoint.
        for(unsigned p=0;p<2;++p) {
            const auto graphs=FollowerPredecessorDiagnosticGraphResources::bom(host,gpu[p]);
            for(const auto &entry:graphs.resources())plan.add(entry);
            const auto singleton=NativeOrderingEventGroup::bom(program,host,gpu[p]);
            for(unsigned cache=0;cache<(p?2u:1u);++cache) {
                for(const auto &entry:singleton.resources())plan.add(entry);
                if(p) { const auto extras=OriginalSlotCaptureTerminals::extraEventBOM(geometry_,host,gpu[p]);
                    for(const auto &entry:extras.resources())plan.add(entry); }
            }
            plan.add(gpu[p],PhysicalMemoryOwner::ExecutionWorkspace,DeviceMoEOverlayEpochArena::allocationBytesForSlots(1u));
            // Two route arrays and immutable epoch scalars, not full expert capacity.
            plan.add(gpu[p],PhysicalMemoryOwner::ActivationArena,2*(2*sizeof(std::int32_t)+sizeof(std::uint32_t)));
            for(auto &f:family_) {
                for(auto *tensor:p==0?std::vector<RecurrentPacketTensor*>{&f->hidden,&f->indices,&f->weights,&f->canonical,&f->zero,&f->increment}:
                    std::vector<RecurrentPacketTensor*>{&f->received,&f->received_indices,&f->received_weights,&f->local})tensor->declare(plan,host,gpu[p]);
                plan.add(gpu[p],PhysicalMemoryOwner::ActivationArena,sizeof(std::int32_t));
            }
        }
        plan.add(host,PhysicalMemoryOwner::ActivationArena,evidence_root_.physicalBytes());
        plan.add(gpu[0],PhysicalMemoryOwner::ActivationArena,evidence_root_.capacity()*sizeof(float));
        declareChannels();
        layout_=channel_plan_.channels.at(0).mapping_layout;
        for(const auto &resource:resources)plan.add(resource,PhysicalMemoryOwner::ActivationTransportStaging,
            channel_plan_.stagingBytesFor(0,resource.device));
        source_->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
        // The canonical staging plan is an admission ceiling, not a second
        // allocation. Materialize only this actual mmap and original grants;
        // packet tensor roots below claim their own actual storage separately.
        channel_claims_.push_back(memory_->claimNewAllocation(DeviceId::cpu(),
            PhysicalMemoryOwner::ActivationTransportStaging,layout_.mapping_bytes));
        for(const auto &lane:channel_plan_.channels.at(0).localLanes(0))
            channel_claims_.push_back(memory_->claimNewAllocation(lane.device,
                PhysicalMemoryOwner::ActivationTransportStaging,channel_plan_.device_grant_bytes_per_lane));
        for(unsigned p=0;p<2;++p) {
            workers_[p]=permits[p]->worker();
            for(auto &slot:graph_backing_[p])slot.emplace(FollowerPredecessorDiagnosticGraphResources::prepare(*permits[p]));
            permits[p].reset();
        }
        slots_.reserve(geometry_.slotCount());identities_.reserve(geometry_.slotCount());terminals_.reserve(geometry_.slotCount());
        makeTransport();
        evidence_=evidence_root_.allocate(memory_);
        evidence_claim_.emplace(memory_->claimNewAllocation(devices_[0],PhysicalMemoryOwner::ActivationArena,evidence_root_.capacity()*sizeof(float)));
        workers_[0]->submitAndWait([&]{
            requireRecurrentPacket(source_cache_.ensureCaptureStream(workers_[0].get(),devices_[0],false),"Original source aggregate stream missing");
            requireRecurrentPacket(source_cache_.prepareCaptureStreamTerminal(workers_[0].get(),memory_),"Original source aggregate terminal missing");
            evidence_device_=getBackendFor(devices_[0])->allocate(evidence_root_.capacity()*sizeof(float),devices_[0].ordinal);
            requireRecurrentPacket(evidence_device_!=nullptr,"Original packet evidence allocation failed");
        });
        for(unsigned p=0;p<2;++p)workers_[p]->submitAndWait([&,p]{prepareEndpoint(p);});
        workers_[0]->submitAndWait([&]{checkCaptureIdentity();});
    }

    /** @brief Queue three SAME R1 parents and a verifier before aggregate observation. */
    void replay() {
        const MoEOverlayInferenceCommandIdentity command{.request_generation=1u,.command_id=1u,.initial_placement_epoch=1u};
        requireRecurrentPacket(protocol_->beginCommand(command),"Original packet command was rejected");
        for(std::size_t i=0;i<invocations;++i) {
            const unsigned family=i<3?0u:1u;
            auto ticket=makeMoEOverlayInferenceExecutionTicket(topology_,command,i+1u,i+1u,1u,
                i<3?MoEOverlayInferenceGraphRole::MTPDraft:MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
                1,family_[family]->rows,family_[family]->rows,depth,i<3?static_cast<int>(i):-1);
            const auto admission=protocol_->accept(ticket);
            requireRecurrentPacket(admission.accepted(),"Original packet slot admission failed");
            requireRecurrentPacket(protocol_->markSubmitted(admission.slot_index,ticket),"Original packet submission ownership failed");
            slots_.push_back(protocol_->originalSubmittedSlot(ticket));
            identities_.push_back(mapped_->publishOriginalNativeInvocation(slots_.back(),target,family,i+1u,
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    (std::chrono::steady_clock::now()+std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs)).time_since_epoch()).count())));
            auto &f=*family_[family];
            workers_[1]->submitAndWait([&,family]{
                if(!terminals_.empty())family_[last_family_]->follower_cache.enqueuePublishedCaptureStreamTerminalWait(
                    terminals_.back(),f.follower_cache.capture_stream,workers_[1].get());
                requireRecurrentPacket(f.graphs[1]->launchOnStream(f.follower_cache.capture_stream),"Original R1 follower captured launch failed");
                terminals_.push_back(f.follower_cache.publishCaptureStreamTerminal(slots_.back()));
            });
            workers_[0]->submitAndWait([&,i]{
                source_enqueued_=true;
                requireRecurrentPacket(f.graphs[0]->launchOnStream(source_cache_.capture_stream),"Original R1 source captured launch failed");
                requireRecurrentPacket(getBackendFor(devices_[0])->deviceCopyAsync(
                    static_cast<float*>(evidence_device_)+(i<3?i:3u)*width,f.canonical.tensor->gpu_data_ptr(),
                    f.rows*width*sizeof(float),devices_[0].ordinal,source_cache_.capture_stream),"Original packet snapshot enqueue failed");
            });
            last_family_=family;
        }
        // The first terminal observation is AFTER every source/follower submit.
        workers_[0]->submitAndWait([&]{
            requireRecurrentPacket(getBackendFor(devices_[0])->deviceToHost(evidence_.data(),evidence_device_,
                evidence_root_.capacity()*sizeof(float),devices_[0].ordinal,source_cache_.capture_stream),"Original aggregate output copy failed");
            source_terminal_=source_cache_.publishCaptureStreamTerminal();
        });
        source_cache_.waitForPublishedCaptureStreamTerminal(source_terminal_,Cache::HostFenceWaitPolicy::ActiveProgress);
        source_terminal_={};
        for(std::size_t i=0;i<terminals_.size();++i) {
            family_[i<3?0u:1u]->follower_cache.waitForPublishedCaptureStreamTerminal(terminals_[i],Cache::HostFenceWaitPolicy::ActiveProgress);
            ++terminal_observed_;
            const auto traffic=mapped_->completedOriginalNativeInvocation(slots_[i],target,identities_[i]);
            const std::uint64_t live=profile_==RecurrentPacketProfile::Empty?0u:
                (i<3?1u:profile_==RecurrentPacketProfile::PartialVerifier?2u:4u);
            EXPECT_EQ(traffic.dispatch_live_rows,live);
            EXPECT_EQ(traffic.return_live_rows,live);
            EXPECT_EQ(traffic.dispatch_live_entries,live);
            EXPECT_EQ(traffic.return_live_entries,live);
            EXPECT_EQ(traffic.dispatch_payload_bytes,moeOverlayDispatchPayloadBytes(live,live,width));
            EXPECT_EQ(traffic.return_payload_bytes,moeOverlayReturnPayloadBytes(live,width));
            EXPECT_EQ(traffic.dispatch_shared_hidden_bytes,i<3?0u:live*width*sizeof(float));
            EXPECT_EQ(traffic.dispatch_stage_count,1u);
            EXPECT_EQ(traffic.return_stage_count,1u);
            for(int r=0;r<(i<3?1:4);++r)for(int col=0;col<width;++col)
                EXPECT_EQ(evidence_[(i<3?i:3u)*width+r*width+col],(profile_==RecurrentPacketProfile::Empty ||
                    (profile_==RecurrentPacketProfile::PartialVerifier && i==3u && r%2!=0))?0.0f:
                    static_cast<float>(i<3?(i+1u)*(i+2u)/2u:1u));
        }
        for(const auto &f:family_)for(unsigned p=0;p<2;++p)for(std::size_t i=0;i<f->stages[p].size();++i)
            EXPECT_EQ(f->stages[p][i]->graphCaptureVariantSignature(),f->signatures[p][i]);
        observed_=true;
    }
    /**
     * @brief Abort an issued successor without launching its source generation.
     *
     * The first invocation may finish or observe the absorbing abort. Neither
     * outcome is relabelled success. Both follower enqueues precede the only
     * source aggregate observation, and no per-draft host fence is introduced.
     */
    void replayCancelledSuccessor() {
        const MoEOverlayInferenceCommandIdentity command{.request_generation=1u,.command_id=1u,.initial_placement_epoch=1u};
        requireRecurrentPacket(protocol_->beginCommand(command),"Original cancellation command was rejected");
        auto &f=*family_[0];
        for(std::size_t i=0;i<2u;++i) {
            const auto ticket=makeMoEOverlayInferenceExecutionTicket(topology_,command,i+1u,i+1u,1u,
                MoEOverlayInferenceGraphRole::MTPDraft,1,1,1,depth,static_cast<int>(i));
            const auto admission=protocol_->accept(ticket);
            requireRecurrentPacket(admission.accepted(),"Original cancellation slot was rejected");
            requireRecurrentPacket(protocol_->markSubmitted(admission.slot_index,ticket),"Original cancellation submission ownership failed");
            slots_.push_back(protocol_->originalSubmittedSlot(ticket));
            identities_.push_back(mapped_->publishOriginalNativeInvocation(slots_.back(),target,0u,i+1u,
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    (std::chrono::steady_clock::now()+std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs)).time_since_epoch()).count())));
            workers_[1]->submitAndWait([&]{
                if(!terminals_.empty())f.follower_cache.enqueuePublishedCaptureStreamTerminalWait(
                    terminals_.back(),f.follower_cache.capture_stream,workers_[1].get());
                requireRecurrentPacket(f.graphs[1]->launchOnStream(f.follower_cache.capture_stream),"Original cancelled follower launch failed");
                terminals_.push_back(f.follower_cache.publishCaptureStreamTerminal(slots_.back()));
            });
            if(i==0u)workers_[0]->submitAndWait([&]{
                source_enqueued_=true;
                requireRecurrentPacket(f.graphs[0]->launchOnStream(source_cache_.capture_stream),"Original pre-cancellation source launch failed");
            });
        }
        // Source graph generation2 is deliberately never submitted. The exact
        // existing cancellation authority must unblock the already queued peer
        // even if generation1 has not begun, without a guessed generation floor.
        mapped_->cancelOriginalNativeInvocation(slots_.back(),target,identities_.back());
        workers_[0]->submitAndWait([&]{source_terminal_=source_cache_.publishCaptureStreamTerminal();});
        // Identify the original wait frontier without adding a probe, fence,
        // or alternate timeout. Detailed mapped metadata is acquired only if
        // the existing exact-terminal watchdog reports its original failure.
        std::cerr<<"recurrent_cancel_wait begin=source_aggregate generation="<<source_terminal_.generation<<std::endl;
        source_cache_.waitForPublishedCaptureStreamTerminal(source_terminal_,Cache::HostFenceWaitPolicy::ActiveProgress,
            [&]{return cancellationDiagnostic("source_aggregate",0u);});
        std::cerr<<"recurrent_cancel_wait complete=source_aggregate"<<std::endl;
        source_terminal_={};
        for(std::size_t index=0;index<terminals_.size();++index) {
            const auto &terminal=terminals_[index];
            std::cerr<<"recurrent_cancel_wait begin=follower invocation="<<index
                <<" original_slot="<<slots_[index].index()<<" generation="<<terminal.generation<<std::endl;
            f.follower_cache.waitForPublishedCaptureStreamTerminal(terminal,Cache::HostFenceWaitPolicy::ActiveProgress,
                [&,index]{return cancellationDiagnostic("follower",index);});
            std::cerr<<"recurrent_cancel_wait complete=follower invocation="<<index<<std::endl;
            ++terminal_observed_;
        }
        EXPECT_LT(std::atomic_ref(f.lanes[0].control_host->admission.last_generation).load(std::memory_order_acquire),2u);
        EXPECT_EQ(std::atomic_ref(f.lanes[0].control_host->admission.state).load(std::memory_order_acquire),
            static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Failed));
        EXPECT_EQ(std::atomic_ref(f.lanes[0].control_host->admission.ready_signal).load(std::memory_order_acquire),
            kMoEOverlayActivationAbortTimeline);
        EXPECT_THROW(mapped_->completedOriginalNativeInvocation(slots_.back(),target,identities_.back()),std::logic_error);
        observed_=true; // Physical observation only; no successful packet claim.
    }
private:
    /**
     * @brief Acquire bounded mapped metadata at the ORIGINAL terminal deadline.
     * @param frontier Exact source aggregate or follower wait that failed.
     * @param invocation Actual submitted fixture ordinal, never an epoch guess.
     * @return Individually acquired passive observations; not a coherent state
     * snapshot, completion receipt, readiness decision, or device readback.
     */
    std::string cancellationDiagnostic(const char *frontier,std::size_t invocation)const;
    /** @brief Derive transport only from original cold two-device declaration. */
    void makeTransport();
    void declareChannels();
    void checkCaptureIdentity();
    /** @brief Materialize exact participant tensors, stages, and two captures. */
    void prepareEndpoint(unsigned endpoint);
    /** @brief Drain actual terminals before native handles, mappings, and source withdrawal. */
    void dispose();
    std::array<DeviceId,2> devices_;
    MoEOverlayInferenceOriginalSlotGeometry geometry_;
    std::array<std::unique_ptr<Family>,2> family_;
    CPUStorageRoot<float> evidence_root_;
    AlignedVector<float> evidence_;
    void *evidence_device_=nullptr;
    std::optional<PhysicalMemoryAllocationLease> evidence_claim_,mapping_claim_;
    std::shared_ptr<NativeContextPreparation> source_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::array<std::shared_ptr<IWorkerGPUContext>,2> workers_;
    std::array<std::array<std::optional<NativeContextRetirementSlot>,2>,2> graph_backing_;
    std::array<std::shared_ptr<DeviceMoEOverlayEpochArena>,2> arenas_;
    std::array<std::unique_ptr<IDeviceContext>,2> contexts_;
    std::array<void*,2> placement_storage_{};
    std::array<std::optional<PhysicalMemoryAllocationLease>,2> placement_claims_;
    MoEOverlayNodeLocalActivationLayout layout_;
    MoEOverlayInferenceTopologyIdentity topology_{.workspace_generation=1u,.topology_fingerprint_low=0x7231u,
        .topology_fingerprint_high=0x7232u,.source_world_rank=0,.target_world_rank=0};
    std::unique_ptr<MoEOverlayInferenceTransactionProtocol> protocol_;
    std::shared_ptr<IMoEOverlayRankBatchTransport> transport_;
    std::shared_ptr<IMoEOverlayMappedActivationTransport> mapped_;
    MoEOverlayActivationChannelPlan channel_plan_;
    std::vector<PhysicalMemoryAllocationLease> channel_claims_;
    Cache source_cache_;
    Cache::CaptureStreamTerminalTicket source_terminal_;
    std::vector<Cache::CaptureStreamTerminalTicket> terminals_;
    std::vector<MoEOverlayInferenceOriginalSlot> slots_;
    std::vector<MoEOverlayActivationEpochIdentity> identities_;
    unsigned last_family_=0u;
    std::size_t terminal_observed_=0u;
    bool observed_=false,disposed_=false,source_enqueued_=false;
    RecurrentPacketProfile profile_=RecurrentPacketProfile::Full;
};

/** @brief Use the original planner for physical grants, mapping, and staging. */
void RecurrentOriginalSlotPacketReplay::declareChannels() {
    MoERoutedExpertPlacementPlan placement;
    placement.enabled=true;placement.topology=RoutedExpertPlacementTopology::TieredOverlay;
    placement.continuation_domain="source";placement.base_model_domain="source";placement.shared_expert_domain="source";
    placement.continuation_domain_spec.domain="source";placement.continuation_domain_spec.logical_root_participant=0;
    for(unsigned p=0;p<2;++p) {
        RoutedExpertDomain domain;domain.name=p?"follower":"source";domain.scope=ExecutionDomainScope::RANK_LOCAL;
        domain.participants={devices_[p].is_cuda()?GlobalDeviceAddress::cuda(devices_[p].ordinal):GlobalDeviceAddress::rocm(devices_[p].ordinal)};
        domain.world_ranks={0};domain.owner_rank=0;domain.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
        placement.domains.push_back(domain);
        placement.routed_tiers.push_back(RoutedExpertTier{.name=domain.name,.domain=domain.name,.priority=p?30:80});
    }
    placement.authority_execution=resolveMoEOverlayAuthorityExecutionKind(placement);
    requireRecurrentPacket(placement.authority_execution==MoEOverlayAuthorityExecutionKind::DeviceResident,"Original all-GPU declaration lost device authority");
    ClusterInventory cluster;cluster.world_size=1;cluster.ranks.resize(1);
    cluster.ranks[0].rank=0;cluster.ranks[0].node_id=0;cluster.ranks[0].hostname="original-packet-source";
    channel_plan_=MoEOverlayActivationChannelPlanner::plan({.placement_plan=&placement,.cluster_inventory=&cluster,
        .row_capacity=4u,.d_model=width,.top_k=1,.graph_family_count=2u,.original_native_slots=geometry_});
    requireRecurrentPacket(channel_plan_.channels.size()==1u,"Original two-GPU declaration did not produce exactly one channel");
    requireRecurrentPacket(channel_plan_.channels[0].targetParticipantIds()==std::vector<int>{target},"Original follower participant identity changed");
}

/** @brief Construct one channel with both actual GPU aliases in the same process. */
void RecurrentOriginalSlotPacketReplay::makeTransport() {
    const auto &channel=channel_plan_.channels.at(0);
    const std::string name="recurrent_original_slot_packet_"+std::to_string(::getpid());
    MoEOverlayRankBatchTransportConfig config;
    config.mpi_ctx=makeContext(0,name,1);config.endpoint_scope=MoEOverlayActivationEndpointScope::RankLocalGPU;
    config.rank_local_source_device=devices_[0];config.source_world_rank=0;config.target_world_rank=0;
    config.workspace=std::make_shared<MoEOverlayRankBatchWireWorkspace>(MoEOverlayRankBatchWireWorkspace::Config{
        .participant_ids={target},.max_total_rows=4u,.max_total_entries=4u,.d_model=width,.top_k=1,
        .return_layout=layout_.geometry.return_layout});
    config.max_rows_per_participant=4;config.max_entries_per_participant=4;config.d_model=width;config.top_k=1;
    config.tier_index=channel.tier_index;config.domain_ordinal=channel.domain_ordinal;config.channel_identity=name;
    config.transaction_topology=topology_;config.source_endpoint={.world_rank=0,.participant_id=0,.tier_priority=80,.domain_ordinal=0};
    config.target_tier_priority=30;config.activation_layout=layout_;config.local_lanes=channel.localLanes(0);
    config.activation_graph_families={
        {.graph_role_mask=1u<<static_cast<unsigned>(MoEOverlayInferenceGraphRole::MTPDraft),.model_layer_indices={0}},
        {.graph_role_mask=1u<<static_cast<unsigned>(MoEOverlayInferenceGraphRole::MTPGroupedVerifier),.model_layer_indices={0}}};
    protocol_=std::make_unique<MoEOverlayInferenceTransactionProtocol>(MoEOverlayInferenceTransactionProtocol::Config{
        .topology=topology_,.slot_count=geometry_.slotCount(),.max_request_count=1,.max_rows_per_request=4,.max_mtp_draft_depth=depth});
    transport_=createMoEOverlayRankLocalGPUActivationTransport(std::move(config));
    mapped_=std::dynamic_pointer_cast<IMoEOverlayMappedActivationTransport>(transport_);
    requireRecurrentPacket(bool(mapped_),"Original local GPU transport lost its mapped contract");
    mapped_->bindOriginalNativeSlots(protocol_->originalSlotDomain());
    for(unsigned f=0;f<families;++f)for(unsigned p=0;p<2;++p)
        family_[f]->lanes[p]=mapped_->activationDeviceLane(target,f,devices_[p]);
}

/** @brief Capture exact existing packet stages, retaining every referenced owner. */
void RecurrentOriginalSlotPacketReplay::prepareEndpoint(unsigned p) {
    const auto device=devices_[p];auto *backend=getBackendFor(device);auto &worker=*workers_[p];
    contexts_[p]=IDeviceContext::create(device);
    arenas_[p]=std::make_shared<DeviceMoEOverlayEpochArena>(DeviceMoEOverlayEpochArena::Config{
        .device_id=device,.memory_authority=memory_});
    const std::size_t placement_bytes=2*(2*sizeof(std::int32_t)+sizeof(std::uint32_t));
    placement_claims_[p].emplace(memory_->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,placement_bytes));
    placement_storage_[p]=backend->allocate(placement_bytes,device.ordinal);
    requireRecurrentPacket(placement_storage_[p]!=nullptr,"Original exact route placement allocation failed");
    // Both immutable bank projections name the actual follower; the device
    // arena ticket alone selects which projection may execute.
    const std::array<std::uint32_t,6> placement{0u,target,1u,0u,target,1u};
    void *setup=p?family_[0]->follower_cache.capture_stream:source_cache_.capture_stream;
    if(p) {
        for(auto &f:family_) {
            requireRecurrentPacket(f->follower_cache.ensureCaptureStream(&worker,device,false),"Original follower capture stream missing");
            requireRecurrentPacket(f->follower_cache.prepareCaptureStreamTerminal(&worker,memory_),"Original follower terminal preparation failed");

        }
        setup=family_[0]->follower_cache.capture_stream;
    }
    requireRecurrentPacket(backend->hostToDevice(placement_storage_[p],placement.data(),placement_bytes,device.ordinal,setup),"Original route projection upload failed");
    // This cold source array remains alive until the cold upload terminal.
    if(p) {
        requireRecurrentPacket(worker.recordEventChecked(family_[0]->follower_cache.terminal_event,setup),"Original cold projection terminal record failed");
        requireRecurrentPacket(worker.synchronizeEventChecked(family_[0]->follower_cache.terminal_event),"Original cold projection upload failed");
    } else {
        const auto setup_ticket=source_cache_.publishCaptureStreamTerminal();
        source_cache_.waitForPublishedCaptureStreamTerminal(setup_ticket,Cache::HostFenceWaitPolicy::ActiveProgress);
    }
    for(unsigned fidx=0;fidx<families;++fidx) {
        auto &f=*family_[fidx];void *stream=p?f.follower_cache.capture_stream:source_cache_.capture_stream;
        auto &lane=f.lanes[p];
        auto *bank=static_cast<std::uint32_t*>(placement_storage_[p]);
        const MoEOverlayRoutePlacementDeviceBinding binding{
            .banks={{reinterpret_cast<const std::int32_t*>(bank),bank+2},{reinterpret_cast<const std::int32_t*>(bank+3),bank+5}},
            .ticket=arenas_[p]->requestTicket(0u),.status=arenas_[p]->requestStatus(0u),.expert_count=2u};
        f.active_claims[p].emplace(memory_->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,sizeof(std::int32_t)));
        f.active[p]=backend->allocate(sizeof(std::int32_t),device.ordinal);
        requireRecurrentPacket(f.active[p]!=nullptr,"Original packet live-row allocation failed");
        const std::int32_t rows=f.rows;
        requireRecurrentPacket(backend->hostToDevice(f.active[p],&rows,sizeof(rows),device.ordinal,stream),"Original packet live-row upload failed");
        if(p==0) {
            f.hidden.prepare(memory_,device,stream,f.rows,width);
            f.increment.prepare(memory_,device,stream,f.rows,width,1.0f);
            // Populate this original host root BEFORE its first H2D. The
            // asynchronous upload must never race a later host rewrite.
            void (*initialize_indices)(std::span<float>)=nullptr;
            if(profile_==RecurrentPacketProfile::PartialVerifier && f.rows>1)
                initialize_indices=[](std::span<float> values) {
                    for(std::size_t row=0;row<values.size();++row)values[row]=row%2==0?1.0f:0.0f;
                };
            f.indices.prepare(memory_,device,stream,f.rows,1,
                profile_==RecurrentPacketProfile::Empty?0.0f:1.0f,initialize_indices);
            f.weights.prepare(memory_,device,stream,f.rows,1,1.0f);
            f.canonical.prepare(memory_,device,stream,f.rows,width);
            f.zero.prepare(memory_,device,stream,f.rows,width);
        } else {
            f.received.prepare(memory_,device,stream,f.rows,width);
            f.received_indices.prepare(memory_,device,stream,f.rows,1);
            f.received_weights.prepare(memory_,device,stream,f.rows,1);
            f.local.prepare(memory_,device,stream,f.rows,width);
        }
        MoEOverlayEpochBoundaryStage::Params acquire{.device_id=device,.arena=arenas_[p],.request_slot=0u,
            .operation=MoEOverlayEpochBoundaryStage::Operation::Acquire};
        if(p)acquire.peer_epoch_source=MoEOverlayEpochBoundaryStage::PeerEpochSource{
            .mapped_region=lane.mapped_region,.binding={.control=lane.control_device,.grant=lane.grant_device,.stage_ordinal=0u}};
        f.stages[p].push_back(std::make_shared<MoEOverlayEpochBoundaryStage>(acquire));
        if(p==0) {
            ResidualAddStage::Params advance{.device_id=device,.input=f.increment.tensor.get(),.residual=f.hidden.tensor.get(),
                .output=f.hidden.tensor.get(),.num_elements=static_cast<std::size_t>(rows*width)};
            f.stages[p].push_back(std::make_shared<ResidualAddStage>(advance));
            ResidualAddStage::Params clear{.device_id=device,.input=f.zero.tensor.get(),.output=f.canonical.tensor.get(),
                .num_elements=static_cast<std::size_t>(rows*width)};
            f.stages[p].push_back(std::make_shared<ResidualAddStage>(clear));
            MoEOverlayActivationDispatchPackStage::Params dispatch{.device_id=device,.lane=lane,.hidden=f.hidden.tensor.get(),
                .routing_indices=f.indices.tensor.get(),.routing_weights=f.weights.tensor.get(),.placement=binding,
                .active_row_count_device=static_cast<const std::int32_t*>(f.active[p]),.physical_rows=rows,.stage_ordinal=0u,.model_layer_index=0};
            f.stages[p].push_back(std::make_shared<MoEOverlayActivationDispatchPackStage>(dispatch));
            MoEOverlayActivationReturnConsumeStage::Params receive{.device_id=device,.lane=lane,
                .canonical_route_contributions=f.canonical.tensor.get(),.physical_rows=rows,.stage_ordinal=0u,.model_layer_index=0};
            f.stages[p].push_back(std::make_shared<MoEOverlayActivationReturnConsumeStage>(receive));
        } else {
            MoEOverlayActivationDispatchConsumeStage::Params receive{.device_id=device,.lane=lane,.placement=binding,
                .hidden=f.received.tensor.get(),.routing_indices=f.received_indices.tensor.get(),.routing_weights=f.received_weights.tensor.get(),
                .active_row_count_device=static_cast<std::int32_t*>(f.active[p]),.physical_rows=rows,.stage_ordinal=0u,.model_layer_index=0};
            f.stages[p].push_back(std::make_shared<MoEOverlayActivationDispatchConsumeStage>(receive));
            ResidualAddStage::Params compute{.device_id=device,.input=f.received.tensor.get(),.residual=f.local.tensor.get(),
                .output=f.local.tensor.get(),.num_elements=static_cast<std::size_t>(rows*width)};
            f.stages[p].push_back(std::make_shared<ResidualAddStage>(compute));
            MoEOverlayActivationReturnPackStage::Params send{.device_id=device,.lane=lane,
                .local_canonical_route_contributions=f.local.tensor.get(),.physical_rows=rows,.stage_ordinal=0u,.model_layer_index=0};
            f.stages[p].push_back(std::make_shared<MoEOverlayActivationReturnPackStage>(send));
        }
        acquire.operation=MoEOverlayEpochBoundaryStage::Operation::Release;acquire.peer_epoch_source.reset();
        f.stages[p].push_back(std::make_shared<MoEOverlayEpochBoundaryStage>(acquire));
        for(auto &stage:f.stages[p])requireRecurrentPacket(stage->prepareGraphLaunch(contexts_[p].get(),stream),"Original packet stage preparation failed");
        // Upload alone does not certify a future capture consumer. Join each
        // actual external producer event on THIS exact recording stream while
        // uncaptured, using the same public transfer authority as the executor.
        for(auto *tensor:p==0?std::vector<RecurrentPacketTensor*>{&f.hidden,&f.indices,&f.weights,&f.canonical,&f.zero,&f.increment}:
            std::vector<RecurrentPacketTensor*>{&f.received,&f.received_indices,&f.received_weights,&f.local})
            TransferEngine::requireDeviceInput(tensor->tensor.get(),device,stream);
        using Ledger=GraphCaptureDependencyLedger;
        std::vector<Ledger::StagePlan> plan;
        for(const auto &stage:f.stages[p])plan.push_back({.stage_identity=stage.get(),.stage_name=stage->name(),.recording_stream=stream});
        if(p==0) {
            // Acquire -> increment -> clear -> dispatch -> return -> release.
            // These are the actual typed stage bindings in the sequence above;
            // no request state, readiness exception, or output is fabricated.
            plan.at(1).external_inputs={f.increment.tensor.get(),f.hidden.tensor.get()};
            plan.at(1).outputs={f.hidden.tensor.get()};
            plan.at(2).external_inputs={f.zero.tensor.get()};
            plan.at(2).outputs={f.canonical.tensor.get()};
            plan.at(3).external_inputs={f.indices.tensor.get(),f.weights.tensor.get()};
            plan.at(3).internal_inputs={{f.hidden.tensor.get(),1u}};
            plan.at(4).internal_inputs={{f.canonical.tensor.get(),2u}};
            plan.at(4).outputs={f.canonical.tensor.get()};
        } else {
            // Acquire -> receive -> real additive work -> return -> release.
            plan.at(1).outputs={f.received.tensor.get(),f.received_indices.tensor.get(),f.received_weights.tensor.get()};
            plan.at(2).internal_inputs={{f.received.tensor.get(),1u}};
            plan.at(2).external_inputs={f.local.tensor.get()};
            plan.at(2).outputs={f.local.tensor.get()};
            plan.at(3).internal_inputs={{f.local.tensor.get(),2u}};
        }
        Ledger dependencies(device,stream,std::move(plan),"original-slot-packet-component");
        // Cold upload storage remains live through this exact native prefix.
        if(p) {
            requireRecurrentPacket(worker.recordEventChecked(f.follower_cache.terminal_event,stream),"Original cold follower prefix record failed");
            requireRecurrentPacket(worker.synchronizeEventChecked(f.follower_cache.terminal_event),"Original cold follower prefix failed");
            f.follower_cache.bindOriginalCaptureTerminalSlots(protocol_->originalSlotDomain(),memory_,geometry_);
        } else {
            const auto prepared=source_cache_.publishCaptureStreamTerminal();
            source_cache_.waitForPublishedCaptureStreamTerminal(prepared,Cache::HostFenceWaitPolicy::ActiveProgress);
        }
        f.graph_claims[p].emplace(memory_->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,nativeGraphCaptureObjectBytes(device)));
        graph_backing_[p][fidx]->armNativeAttempt();f.native_attempted[p]=true;
        f.graphs[p]=worker.createGraphCapture(stream);
        requireRecurrentPacket(bool(f.graphs[p]),"Original packet graph creator failed");
        ScopedBackendGraphCapture capture(worker,*f.graphs[p],"original-slot-packet-component",&dependencies);
        requireRecurrentPacket(capture.begin(),"Original packet capture begin failed");
        for(auto &stage:f.stages[p]) {
            ScopedGraphCaptureStage recording(stage.get(),stream);
            requireRecurrentPacket(stage->execute(contexts_[p].get()),"Original captured packet stage failed");
            recording.complete();
        }
        capture.finish();
        std::string detail;
        requireRecurrentPacket(f.graphs[p]->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());
        requireRecurrentPacket(f.graphs[p]->instantiate(),"Original packet graph instantiate failed");
        for(const auto &stage:f.stages[p])f.signatures[p].push_back(stage->graphCaptureVariantSignature());
    }
}

/** @brief All six packet capture variants bind exact native roster and policy. */
void RecurrentOriginalSlotPacketReplay::checkCaptureIdentity() {
    for(const auto &f:family_)for(const auto &stages:f->stages)for(const auto &stage:stages) {
        if(const auto *p=dynamic_cast<const MoEOverlayActivationDispatchPackStage*>(stage.get()))checkRecurrentPacketIdentity(*p);
        if(const auto *p=dynamic_cast<const MoEOverlayActivationDispatchConsumeStage*>(stage.get()))checkRecurrentPacketIdentity(*p);
        if(const auto *p=dynamic_cast<const MoEOverlayActivationReturnPackStage*>(stage.get()))checkRecurrentPacketIdentity(*p);
        if(const auto *p=dynamic_cast<const MoEOverlayActivationReturnConsumeStage*>(stage.get()))checkRecurrentPacketIdentity(*p);
    }
    auto &f=*family_[0];
    const auto transaction=[&](MoEOverlayMappedActivationDeviceLane lane) {
        return std::make_shared<MoEOverlayActivationLaneBatchState>(devices_[0],
            std::vector<MoEOverlayMappedActivationDeviceLane>{std::move(lane)},1,0u,0);
    };
    const MoEOverlayActivationDispatchPackStage *actual_dispatch=nullptr;
    for(const auto &stage:f.stages[0])if(const auto *dispatch=dynamic_cast<const MoEOverlayActivationDispatchPackStage*>(stage.get()))actual_dispatch=dispatch;
    requireRecurrentPacket(actual_dispatch!=nullptr,"Original scalar dispatch declaration is missing");
    MoEOverlayActivationDispatchPackBatchStage::Params send{.device_id=devices_[0],.transaction=transaction(f.lanes[0]),
        .hidden=f.hidden.tensor.get(),.routing_indices=f.indices.tensor.get(),.routing_weights=f.weights.tensor.get(),
        .placement=actual_dispatch->getParams().placement,.active_row_count_device=actual_dispatch->getParams().active_row_count_device};
    MoEOverlayActivationReturnConsumeBatchStage::Params receive{.device_id=devices_[0],.transaction=send.transaction,
        .canonical_route_contributions=f.canonical.tensor.get()};
    const auto send_key=MoEOverlayActivationDispatchPackBatchStage(send).graphCaptureVariantSignature();
    const auto receive_key=MoEOverlayActivationReturnConsumeBatchStage(receive).graphCaptureVariantSignature();
    const auto check=[&](auto edit) {
        auto lane=f.lanes[0];edit(lane);send.transaction=transaction(lane);receive.transaction=send.transaction;
        EXPECT_NE(send_key,MoEOverlayActivationDispatchPackBatchStage(send).graphCaptureVariantSignature());
        EXPECT_NE(receive_key,MoEOverlayActivationReturnConsumeBatchStage(receive).graphCaptureVariantSignature());
    };
    check([](auto &lane){++lane.original_invocations.original_slots;--lane.original_invocations.original_slot_count;});
    check([](auto &lane){--lane.original_invocations.original_slot_count;});
    // The batch constructor rejects a manifest mismatch before a cache key
    // can exist. This is stronger than allowing an invalid capture variant.
    auto invalid_manifest=f.lanes[0];
    ++invalid_manifest.retained_stage_count;
    EXPECT_THROW((void)transaction(invalid_manifest),std::invalid_argument);
    check([](auto &lane){lane.original_invocations={};});
}

/** @brief Diagnose only already mapped CPU-visible protocol storage. */
std::string RecurrentOriginalSlotPacketReplay::cancellationDiagnostic(const char *frontier,std::size_t invocation)const {
    std::ostringstream out;
    out<<"recurrent_cancel frontier="<<frontier<<" invocation="<<invocation
        <<" native_observations="<<terminal_observed_<<" source_stream="<<source_cache_.capture_stream
        <<" source_ticket="<<(arenas_[0]?arenas_[0]->requestTicket(0u):nullptr)
        <<" follower_ticket="<<(arenas_[1]?arenas_[1]->requestTicket(0u):nullptr);
    if(!family_[0])return out.str()+" family_unavailable";
    const auto &lane=family_[0]->lanes[0];
    out<<" follower_stream="<<family_[0]->follower_cache.capture_stream;
    if(!lane.control_host || !lane.mapped_region)return out.str()+" mapped_control_unavailable";
    auto load=[](auto &value){return std::atomic_ref(value).load(std::memory_order_acquire);};
    auto status=[&](const char *name,MoEOverlayActivationEndpointStatus &value) {
        out<<' '<<name<<"{generation="<<load(value.generation)<<",state="<<load(value.state)
            <<",code="<<load(value.code)<<",operation="<<load(value.operation)
            <<",published="<<load(value.last_published_stage)<<",consumed="<<load(value.last_consumed_stage)
            <<",diagnostic="<<load(value.failure_diagnostic)<<",auxiliary="<<load(value.failure_auxiliary)
            <<",payload_retired="<<load(value.payload_retired_generation)<<'}';
    };
    auto &control=*lane.control_host;
    out<<" channel_nonce="<<control.channel.channel_nonce
        <<" admission{generation="<<load(control.admission.last_generation)
        <<",ready="<<load(control.admission.ready_signal)<<",state="<<load(control.admission.state)
        <<",code="<<load(control.admission.code)<<'}';
    status("continuation",control.continuation_status);status("follower",control.follower_status);
    for(std::size_t bank=0;bank<kMoEOverlayActivationBufferCount;++bank) {
        auto &buffer=control.buffers[bank];
        out<<" bank"<<bank<<"{dispatch="<<load(buffer.dispatch_signal.value)
            <<",return="<<load(buffer.return_signal.value)
            <<",dispatch_rows="<<load(buffer.dispatch_descriptor.live_rows)
            <<",return_rows="<<load(buffer.return_descriptor.live_rows)<<'}';
    }
    // This fixture has ONE actual mapped participant. Its roster starts at the
    // canonical layout range, shared by both retained graph families. Never
    // dereference the GPU alias or create a second host progress ledger.
    const auto &range=layout_.original_native_slots;
    if(layout_.geometry.participant_count!=1u || !range.validFor(lane.mapped_region->sizeBytes()))
        return out.str()+" original_roster_range_invalid";
    auto *records=static_cast<MoEOverlayNativeInvocationSlot*>(lane.mapped_region->mutableHostData(range.offset));
    const auto count=std::min<std::size_t>(slots_.size(),2u);
    for(std::size_t index=0;index<count;++index) {
        const auto original=slots_[index].index();
        if(original>=geometry_.slotCount() || original>=range.bytes/sizeof(MoEOverlayNativeInvocationSlot)) { out<<" original_slot_out_of_range="<<original;continue; }
        auto &record=records[original];
        out<<" original"<<original<<"{published="<<load(record.published_generation)
            <<",cancelled="<<load(record.cancelled_generation)
            <<",continuation_retired="<<load(record.continuation.retired_generation)
            <<",follower_retired="<<load(record.follower.retired_generation)<<'}';
        status("original_continuation",record.continuation.status);
        status("original_follower",record.follower.status);
    }
    return out.str();
}

/** @brief Never dispose mapped slots while a native endpoint can still use them. */
void RecurrentOriginalSlotPacketReplay::dispose() {
    if(disposed_)return;
    if(mapped_ && !observed_)for(std::size_t i=0;i<std::min(slots_.size(),identities_.size());++i)
        mapped_->cancelOriginalNativeInvocation(slots_[i],target,identities_.at(i));
    if(source_cache_.capture_stream && source_cache_.terminal_event && !observed_ && !source_terminal_.valid())workers_[0]->submitAndWait([&]{
        source_terminal_=source_cache_.publishCaptureStreamTerminal();
    });
    if(source_terminal_.valid()) {
        source_cache_.waitForPublishedCaptureStreamTerminal(source_terminal_,Cache::HostFenceWaitPolicy::ActiveProgress);
        source_terminal_={};
    }
    for(std::size_t i=terminal_observed_;i<terminals_.size();++i)if(!observed_)
        family_[i<3?0u:1u]->follower_cache.waitForPublishedCaptureStreamTerminal(terminals_[i],Cache::HostFenceWaitPolicy::ActiveProgress);
    if(!observed_ && workers_[1])workers_[1]->submitAndWait([&]{
        // An SDK error can occur after accepted work but before its immutable
        // receipt is returned. Existing absorbing cancellation precedes this
        // final physical prefix; it grants no semantic packet completion.
        for(auto &family:family_)if(family && family->follower_cache.terminal_event) {
            auto &cache=family->follower_cache;
            requireRecurrentPacket(workers_[1]->recordEventChecked(cache.terminal_event,cache.capture_stream),"Failed packet cleanup prefix record failed");
            requireRecurrentPacket(workers_[1]->synchronizeEventChecked(cache.terminal_event),"Failed packet cleanup prefix did not retire");
        }
    });
    // Submitted is retained until physical disposal; no logical success or
    // aggregate-clear permit is invented by this component fixture.
    for(unsigned p=0;p<2;++p)if(workers_[p])workers_[p]->submitAndWait([&,p]{
        for(unsigned f=0;f<families;++f) {
            auto &family=*family_[f];family.graphs[p].reset();family.graph_claims[p].reset();
            family.stages[p].clear();
            if(family.active[p])getBackendFor(devices_[p])->free(family.active[p],devices_[p].ordinal);
            family.active[p]=nullptr;family.active_claims[p].reset();
            if(family.native_attempted[p])graph_backing_[p][f]->completeNativeHandles();
        }
        if(placement_storage_[p])getBackendFor(devices_[p])->free(placement_storage_[p],devices_[p].ordinal);
        placement_storage_[p]=nullptr;placement_claims_[p].reset();
        if(p==0 && evidence_device_)getBackendFor(devices_[p])->free(evidence_device_,devices_[p].ordinal);
        if(p==0)evidence_device_=nullptr;
    });
    for(auto &family:family_)family.reset();
    source_cache_.reset(Cache::StreamResetPolicy::Destroy);
    mapped_.reset();transport_.reset();channel_claims_.clear();protocol_.reset();
    arenas_={};contexts_={};graph_backing_={};workers_={};evidence_claim_.reset();evidence_={};
    if(source_) { EXPECT_EQ(source_->retireReleasedContexts(),2u);source_.reset(); }
    disposed_=true;
}

TEST(MoERecurrentOriginalSlotPacketReplay,DISABLED_CUDATwoPhysicalEndpointsAggregate) {
    RecurrentOriginalSlotPacketReplay replay(DeviceId::cuda(0));replay.prepare();replay.replay();
}
TEST(MoERecurrentOriginalSlotPacketReplay,DISABLED_ROCmTwoPhysicalEndpointsAggregate) {
    RecurrentOriginalSlotPacketReplay replay(DeviceId::rocm(0));replay.prepare();replay.replay();
}
TEST(MoERecurrentOriginalSlotPacketReplay,DISABLED_CUDATwoPhysicalEndpointsPartialVerifier) {
    RecurrentOriginalSlotPacketReplay replay(DeviceId::cuda(0),RecurrentPacketProfile::PartialVerifier);replay.prepare();replay.replay();
}
TEST(MoERecurrentOriginalSlotPacketReplay,DISABLED_CUDATwoPhysicalEndpointsEmpty) {
    RecurrentOriginalSlotPacketReplay replay(DeviceId::cuda(0),RecurrentPacketProfile::Empty);replay.prepare();replay.replay();
}
TEST(MoERecurrentOriginalSlotPacketReplay,DISABLED_CUDATwoPhysicalEndpointsCancelBeforeSuccessorSource) {
    RecurrentOriginalSlotPacketReplay replay(DeviceId::cuda(0));replay.prepare();replay.replayCancelledSuccessor();
}
TEST(MoERecurrentOriginalSlotPacketReplay,DISABLED_ROCmTwoPhysicalEndpointsPartialVerifier) {
    RecurrentOriginalSlotPacketReplay replay(DeviceId::rocm(0),RecurrentPacketProfile::PartialVerifier);replay.prepare();replay.replay();
}
TEST(MoERecurrentOriginalSlotPacketReplay,DISABLED_ROCmTwoPhysicalEndpointsEmpty) {
    RecurrentOriginalSlotPacketReplay replay(DeviceId::rocm(0),RecurrentPacketProfile::Empty);replay.prepare();replay.replay();
}
TEST(MoERecurrentOriginalSlotPacketReplay,DISABLED_ROCmTwoPhysicalEndpointsCancelBeforeSuccessorSource) {
    RecurrentOriginalSlotPacketReplay replay(DeviceId::rocm(0));replay.prepare();replay.replayCancelledSuccessor();
}
} } // namespace llaminar2::test
