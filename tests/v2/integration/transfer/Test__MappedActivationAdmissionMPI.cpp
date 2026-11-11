/**
 * @file Test__MappedActivationAdmissionMPI.cpp
 * @brief Original-rank admitted GPU metadata parents across the real shared channel.
 *
 * One original native source per process owns all four direction/recipe
 * definitions before either test launches. Discovery retains actual UUID/NUMA
 * ownership; each process creates only its local GPU grant and native graphs.
 * Follower-host admission releases both complete captured metadata parents.
 * The48-stage recipe reuses the two original physical timeline banks without
 * host stepping. This control-only proof neither fabricates payload work nor
 * claims model inference. Every native definition and mapping retires before
 * the original source and MPI session.
 */
#include <gtest/gtest.h>
#include "fixtures/backends/OriginalMappedNativeCohorts.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEOverlayActivationBatchResources.h"
#include "execution/moe/MoEOverlayActivationEpochProtocol.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "planning/PlanningPublication.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <exception>
#include <memory>
#include <numeric>
#include <thread>
#include <unistd.h>

namespace llaminar2::test { namespace {
constexpr int source_participant=2,target_participant=7,width=8,topk=2;
constexpr std::uint32_t retained_stages=48;
constexpr std::uint64_t publication=9;
/** Reject incomplete native evidence; unavailable original hardware is qualified before launch. */
void requireAdmission(bool valid,const char* reason){if(!valid)throw std::runtime_error(reason);}
/** Actual complete parent recipe, with no test-side per-stage submissions. */
enum class AdmissionRecipe {Publication,AlternatingBanks};
/** Four source/recipe definitions retain the same original lexical process source. */
class OriginalAdmissionSuite final {
    using Graph=RetainedParticipantGraphExecutableFixtureAccess;
    struct Frame {
        explicit Frame(OriginalMappedNativeCohorts cohort,AdmissionRecipe kind):cohorts(std::move(cohort)),recipe(kind){}
        OriginalMappedNativeCohorts cohorts;
        AdmissionRecipe recipe;
        MoEOverlayRankBatchTransportConfig config;
        std::shared_ptr<MoEOverlayNodeLocalRankBatchTransport> transport;
        MoEOverlayMappedActivationDeviceLane lane;
        std::shared_ptr<RetainedParticipantGraphExecutable> graph;
        std::unique_ptr<MoEOverlayActivationEpochProtocol> protocol;
        std::optional<MoEOverlayActivationEpochIdentity> identity;
        bool submitted=false,pending=false;
    };
public:
    /** Bind both actual endpoint roles before native birth; unused original ranks stay idle. */
    OriginalAdmissionSuite():mpi_(MPIContextFactory::global()),cohorts_(OriginalMappedNativeCohorts::observe(
        mpi_->clusterInventory(),{DeviceType::CUDA,1},{DeviceType::ROCm,1})),
        frames_{Frame(cohorts_,AdmissionRecipe::Publication),Frame(cohorts_.reversed(),AdmissionRecipe::Publication),
                Frame(cohorts_,AdmissionRecipe::AlternatingBanks),Frame(cohorts_.reversed(),AdmissionRecipe::AlternatingBanks)} {}
    /** Physical drain and off-worker graph retirement precede source retirement. */
    ~OriginalAdmissionSuite() noexcept {try{dispose();}catch(...){std::terminate();}}
    /** Admit the exact four original channel/grant/definition declarations, then capture. */
    void prepare();
    /** Run both original source directions for one retained recipe. */
    void run(AdmissionRecipe recipe);
private:
    /** One original communicator owns all test checkpoints; no WORLD substitute exists. */
    void checkpoint(const std::function<void()>& body){acceptPlanningCostPreparation(mpi_,body);}
    /** Poll only the exact original participant's terminal stream. */
    void await(){if(!worker_)return;const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for(;;){GPUStreamExecutionState state=GPUStreamExecutionState::Pending;
            worker_->submitAndWait([&]{state=worker_->queryStreamExecutionState(stream_,"mapped admission original terminal");});
            if(state==GPUStreamExecutionState::Complete)return;
            requireAdmission(std::chrono::steady_clock::now()<deadline,"Original mapped admission terminal timed out");std::this_thread::yield();}}
    /** Absorbing failure signal releases native metadata waits before owners unwind. */
    void abortPending()noexcept;
    /** Release all actual payload/native owners, then validate original source retirement. */
    void dispose();
    std::shared_ptr<IMPIContext> mpi_;
    OriginalMappedNativeCohorts cohorts_;
    std::array<Frame,4> frames_;
    std::unique_ptr<PreparedNativeLookupFixture> original_;
    std::shared_ptr<IWorkerGPUContext> worker_;
    void* stream_=nullptr;
    DeviceId local_=DeviceId::invalid();
    bool admitted_=false,disposed_=false;
};

void OriginalAdmissionSuite::prepare(){
    // Complete original topology before endpoint-only transport construction.
    checkpoint([&]{requireAdmission(mpi_->topology()!=nullptr,"Original mapped admission topology is unavailable");});
    const auto devices=cohorts_.localDevices(mpi_->rank());
    checkpoint([&]{if(!devices.empty()){
        requireAdmission(devices.size()==1,"Admission recipe requires exactly its declared local endpoint");local_=devices.front();
        original_=PreparedNativeLookupFixture::prepare(devices,mpi_->rank());worker_=original_->permit(local_).worker();stream_=worker_->defaultStream();
    }});
    for(std::size_t index=0;index<frames_.size();++index){auto& frame=frames_[index];const auto& binding=frame.cohorts;
        std::uint64_t generation=mpi_->rank()==binding.sourceRank()?original_->permit(local_).generation():0;
        std::uint32_t process=mpi_->rank()==binding.sourceRank()?std::uint32_t(::getpid()):0;
        std::array<std::int32_t,3> original_identity{std::bit_cast<std::int32_t>(std::uint32_t(generation)),
            std::bit_cast<std::int32_t>(std::uint32_t(generation>>32)),std::bit_cast<std::int32_t>(process)};
        mpi_->broadcast_int32(original_identity.data(),original_identity.size(),binding.sourceRank());
        generation=std::bit_cast<std::uint32_t>(original_identity[0])|(std::uint64_t(std::bit_cast<std::uint32_t>(original_identity[1]))<<32);
        process=std::bit_cast<std::uint32_t>(original_identity[2]);
        auto& config=frame.config;config.mpi_ctx=mpi_;config.source_world_rank=binding.sourceRank();config.target_world_rank=binding.peerRank();
        config.workspace=std::make_shared<MoEOverlayRankBatchWireWorkspace>(MoEOverlayRankBatchWireWorkspace::Config{
            .participant_ids={target_participant},.max_total_rows=1,.max_total_entries=topk,.d_model=width,.top_k=topk});
        config.max_rows_per_participant=1;config.max_entries_per_participant=topk;config.d_model=width;config.top_k=topk;
        config.tier_index=1;config.domain_ordinal=9;config.channel_identity="original_mapped_admission_"+std::to_string(process)+"_"+std::to_string(index);
        config.transaction_slot_count=8;config.transaction_topology={.workspace_generation=generation,
            .topology_fingerprint_low=0x6d6170706564+index,.topology_fingerprint_high=0x61646d697373,
            .source_world_rank=binding.sourceRank(),.target_world_rank=binding.peerRank()};
        config.source_endpoint={.world_rank=binding.sourceRank(),.participant_id=source_participant,.tier_priority=0,.domain_ordinal=4};
        config.target_tier_priority=1;
        std::vector<std::int32_t> layers(frame.recipe==AdmissionRecipe::Publication?1:retained_stages);std::iota(layers.begin(),layers.end(),5);
        config.activation_graph_families={{.graph_role_mask=1u<<unsigned(MoEOverlayInferenceGraphRole::MainPrefill),.model_layer_indices=std::move(layers)}};
        config.activation_layout=planMoEOverlayNodeLocalActivationLayout({.participant_count=1,.max_rows_per_participant=1,
            .max_entries_per_participant=topk,.d_model=width,.activation_graph_family_count=1});
        if(!devices.empty())config.local_lanes={{.participant_id=target_participant,.device=local_}};
    }
    checkpoint([&]{if(original_){
        const auto host=original_->resource(DeviceId::cpu()),gpu=original_->resource(local_);PhysicalMemoryPlanBuilder bill;
        const auto add=[&](const PhysicalMemoryPlan& part){for(const auto& row:part.resources())bill.add(row);};
        add(RetainedParticipantGraphExecutable::generalBOM(host,gpu,frames_.size()));
        add(MoEOverlayActivationBatchResources::coldPublicationBOM(host,gpu));
        for(const auto& frame:frames_){const auto& config=frame.config;const auto& layout=config.activation_layout;
            bill.add(host,PhysicalMemoryOwner::ActivationTransportStaging,
                (mpi_->rank()==config.source_world_rank?layout.sourceOwnedBytes():layout.targetOwnedBytes())+
                MoEOverlayActivationBatchResources::grantUploadDeclaration().physicalBytes());
            bill.add(gpu,PhysicalMemoryOwner::ActivationTransportStaging,MoEOverlayActivationBatchResources::grantGPUBytes());
        }
        original_->admit(bill.build());admitted_=true;
    }});
    // Each actor creates only its original local endpoint; native registration
    // enters the ordinary real rank-pair startup protocol on this communicator.
    checkpoint([&]{if(original_)for(auto& frame:frames_){frame.config.physical_memory_authority=original_->memory();
        frame.transport=std::make_shared<MoEOverlayNodeLocalRankBatchTransport>(frame.config);
        frame.lane=frame.transport->activationDeviceLane(target_participant,0,local_);
        requireAdmission(frame.lane.valid(),"Original mapped admission endpoint has no actual grant");
        frame.protocol=std::make_unique<MoEOverlayActivationEpochProtocol>(frame.transport->activationEpochControl(target_participant,0),frame.transport->activationEpochConfig(target_participant,0));
        frame.graph=Graph::original(original_->permit(local_),stream_);
        worker_->submitAndWait([&]{auto& graph=Graph::capture(frame.graph);ScopedBackendGraphCapture recording(*worker_,graph,"original mapped admission metadata parent",nullptr);
            requireAdmission(recording.begin(),"Original admission native capture failed");auto& transfer=TransferEngine::instance();
            transfer.enqueueMappedTimelineWait64(*frame.lane.mapped_region,frame.lane.admission_signal_offset,kMoEOverlayActivationAdmissionTimeline,local_,stream_);
            const bool source=mpi_->rank()==frame.cohorts.sourceRank();
            if(frame.recipe==AdmissionRecipe::Publication)transfer.enqueueMappedTimelinePublish64(*frame.lane.mapped_region,
                source?frame.lane.dispatch_signal_offsets[0]:frame.lane.return_signal_offsets[0],publication,local_,stream_);
            else for(std::uint32_t stage=0;stage<retained_stages;++stage){const auto bank=moeOverlayActivationBufferIndex(stage);
                const auto value=moeOverlayActivationLeasedTimelineValue(moeOverlayActivationBufferVisit(stage));
                if(source){transfer.enqueueMappedTimelinePublish64(*frame.lane.mapped_region,frame.lane.dispatch_signal_offsets[bank],value,local_,stream_);
                    transfer.enqueueMappedTimelineWait64(*frame.lane.mapped_region,frame.lane.return_signal_offsets[bank],value,local_,stream_);}
                else{transfer.enqueueMappedTimelineWait64(*frame.lane.mapped_region,frame.lane.dispatch_signal_offsets[bank],value,local_,stream_);
                    transfer.enqueueMappedTimelinePublish64(*frame.lane.mapped_region,frame.lane.return_signal_offsets[bank],value,local_,stream_);}}
            recording.finish();requireAdmission(graph.nodeCount()==(frame.recipe==AdmissionRecipe::Publication?2u:1u+2u*retained_stages),"Original admission parent changed its exact native node population");
            requireAdmission(graph.prepareRuntimeContextStorage(original_->memory()) && graph.instantiate(),"Original admission parent instantiation failed");
        });
    }});
}

void OriginalAdmissionSuite::run(AdmissionRecipe recipe){
    for(auto& frame:frames_)if(frame.recipe==recipe){
        checkpoint([&]{if(original_ && mpi_->rank()==frame.cohorts.peerRank()){
            const MoEOverlayInferenceCommandIdentity command{.request_generation=1,.command_id=1,.initial_placement_epoch=41};
            const auto ticket=makeMoEOverlayInferenceExecutionTicket(frame.config.transaction_topology,command,1,1,41,MoEOverlayInferenceGraphRole::MainPrefill,1,1,1);
            std::string error;frame.identity=frame.protocol->arm(ticket,1,30'000'000'000ULL,&error);requireAdmission(bool(frame.identity),error.c_str());
        }});
        checkpoint([&]{if(original_)worker_->submitAndWait([&]{frame.pending=true;frame.submitted=true;requireAdmission(Graph::submit(frame.graph,1),"Original admission parent launch failed");});});
        checkpoint([&]{if(original_){await();requireAdmission(Graph::observeTerminal(frame.graph),"Original admission parent lacks actual terminal evidence");frame.pending=false;
            std::array<std::uint64_t,kMoEOverlayActivationBufferCount> expected{};
            if(recipe==AdmissionRecipe::Publication)expected[0]=publication;
            else for(std::uint32_t stage=0;stage<retained_stages;++stage)expected[moeOverlayActivationBufferIndex(stage)]=moeOverlayActivationLeasedTimelineValue(moeOverlayActivationBufferVisit(stage));
            auto& control=*frame.lane.control_host;
            for(std::size_t bank=0;bank<expected.size();++bank){
                requireAdmission(std::atomic_ref<std::uint64_t>(control.buffers[bank].dispatch_signal.value).load(std::memory_order_acquire)==expected[bank],"Original cross-process dispatch timeline differs");
                requireAdmission(std::atomic_ref<std::uint64_t>(control.buffers[bank].return_signal.value).load(std::memory_order_acquire)==expected[bank],"Original cross-process return timeline differs");}
        }});
    }
}

void OriginalAdmissionSuite::abortPending()noexcept{
    for(auto& frame:frames_)if(frame.pending && frame.lane.control_host){auto& control=*frame.lane.control_host;
        for(auto& bank:control.buffers){std::atomic_ref<std::uint64_t>(bank.dispatch_signal.value).store(kMoEOverlayActivationAbortTimeline,std::memory_order_release);
            std::atomic_ref<std::uint64_t>(bank.return_signal.value).store(kMoEOverlayActivationAbortTimeline,std::memory_order_release);}
        std::atomic_ref<std::uint64_t>(control.admission.ready_signal).store(kMoEOverlayActivationAbortTimeline,std::memory_order_release);}
}
void OriginalAdmissionSuite::dispose(){
    if(disposed_)return;abortPending();await();
    for(auto& frame:frames_)if(frame.graph && frame.submitted)requireAdmission(Graph::observeTerminal(frame.graph),"Original admission final retirement lacks native terminal");
    for(auto& frame:frames_){frame.graph.reset();frame.identity.reset();frame.protocol.reset();frame.lane={};frame.transport.reset();frame.config.physical_memory_authority.reset();}
    worker_.reset();if(original_){if(admitted_)original_->requirePayloadRetired();original_->retire();original_->requireExactRetired();original_.reset();}
    disposed_=true;
}
} // namespace

/** One admitted original source owns both existing test bodies and all four native frames. */
class Test__MappedActivationAdmissionMPI:public ::testing::Test {
protected:
    /** Prepare one complete source family per original process, never an independent per-test adopter. */
    static void SetUpTestSuite(){source=std::make_unique<OriginalAdmissionSuite>();source->prepare();}
    /** Every native/PMA owner leaves scope before the original MPI main retires the session. */
    static void TearDownTestSuite(){source.reset();}
    static inline std::unique_ptr<OriginalAdmissionSuite> source;
};
TEST_F(Test__MappedActivationAdmissionMPI,FollowerHostArmReleasesCUDAAndROCmGraphsInBothRankDirections){source->run(AdmissionRecipe::Publication);}
TEST_F(Test__MappedActivationAdmissionMPI,FortyEightStageRetainedRoundTripCompletesInBothRankDirections){source->run(AdmissionRecipe::AlternatingBanks);}
} // namespace llaminar2::test
