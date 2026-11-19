/**
 * @file PlanningGPUCohortServiceMeasurement.cpp
 * @brief Captured prepared-service observations on complete original GPU cohorts.
 *
 * One private source2 runtime per original member protects serving routing,
 * demand and epochs. All members capture and replay together on their original
 * workers/context. The root clocks complete service; concurrent member compute
 * maxima and input-ready per-frame communication remain different typed units.
 * Every original bank reader, prepared engine, workspace and native borrow
 * outlives its actual captured consumers and terminal observation.
 */
#include "PlanningGPUExpertMeasurement.h"
#include "PlanningExecutionMeasurement.h"
#include "WorkspaceMemoryEstimator.h"
#include "backends/BackendManager.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "collective/OriginalNativeGraphCaptureGroup.h"
#include "execution/compute_stages/stages/TPAllreduceStage.h"
#include "execution/compute_stages/stages/MoEIntermediateShardPipeline.h"
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include <map>
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/MoEExpertShardCohort.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "kernels/KernelFactory.h"
#include "kernels/IMoEKernel.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "utils/DebugEnv.h"
#include <bit>
#include <cstddef>
#include <chrono>
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>

namespace llaminar2 {
namespace {
/** @brief Each cold definition preserves its distinct observation boundary. */
enum class CohortObservationMode {Complete,ConcurrentCompute,LiveCommunication,EmptyCommunication};
/** @brief Fail before an unsupported source or lifecycle can produce a service price. */
void requireCohort(bool value,const char* operation) {
    if(!value)throw std::runtime_error(std::string("Original GPU cohort service: ")+operation);
}
/** @brief Source readers and private roots retained until every original native terminal. */
struct PreparedCohortLane {
    MoEOverlayParticipantBankLease bank;
    std::shared_ptr<const MoEOverlayFragmentFamily> family;
    std::shared_ptr<RetainedParticipantGraphExecutable> graph;
    std::shared_ptr<IWorkerGPUContext> worker;
    IBackend* backend=nullptr;
    DeviceId device;
    void* stream=nullptr;
    PhysicalMemoryAllocationLease tensor_device_claim,runtime_host_claim,runtime_device_claim;
    AlignedVector<float> expected,observed;
    AlignedVector<uint64_t> histogram;
    std::unique_ptr<FP32Tensor> hidden,indices,weights,canonical,output,release;
    std::unique_ptr<INT32Tensor> live_rows;
    std::unique_ptr<DeviceMoERuntimeTable> runtime;
    std::shared_ptr<MoERoutedPipelineKernelOwner> kernel_owner;
    ComputeGraph stages;
    std::vector<GraphCaptureDependencyLedger::StagePlan> dependencies;
    std::string compute_name,fold_name,sum_name;
    std::unique_ptr<IDeviceContext> context;
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    TPAllreduceStage* sum=nullptr;
};
} // namespace

PlanningGPUCohortServiceObservation PlanningGPUExpertMeasurement::measurePreparedCohort(
    const MoEOverlayParticipantResidencyRegistry& registry,uint64_t epoch,
    const std::shared_ptr<const MoEExpertShardCohort>& cohort,int layer,int expert,
    ExpertHistogramSource phase,const NativeLocalTPContextBorrow& original_native,
    const std::shared_ptr<PhysicalMemoryAuthority>& memory,const ModelMemoryProfile& profile,
    const RankInventory& inventory,
    std::span<const std::shared_ptr<RetainedParticipantGraphExecutable>> calibrations)
{
    if(!memory || !cohort || epoch!=registry.initialEpoch() || epoch>UINT32_MAX || layer!=cohort->source().layer_idx ||
       cohort->participants().empty() || calibrations.size()!=cohort->participants().size() ||
       inventory.rank!=memory->worldRank() || debugEnv().skip_allreduce)
        throw std::invalid_argument("GPU cohort service requires its complete original initial source and rank");
    static_assert(PlanningExecutionMeasurement::kTimedInvocations==MoEOverlayCPUServiceMeasurement::kMeasuredSamples);
    const int rows=MoEOverlayCPUServiceMeasurement::rowsForPhase(phase);
    const auto members=cohort->participants();
    const auto source_geometry=cohort->ownership(0).geometry();
    const int h=source_geometry.model_columns,e=source_geometry.experts;
    if(expert<0 || expert>=e || profile.d_model!=h || profile.expert_count!=e ||
       profile.expert_feed_forward_length!=source_geometry.intermediate_columns)
        throw std::invalid_argument("GPU cohort service changed original full expert geometry");
    std::string error;
    const auto source=registry.acquireOriginalPreparedSource(&error);
    if(!source)throw std::invalid_argument("GPU cohort service lost original model source: "+error);
    const auto* placement=registry.initialOwnerMap().placementFor(layer,expert);
    if(!placement || placement->shard_cohort!=cohort)
        throw std::invalid_argument("GPU cohort service selected a different original logical target");
    {
        const auto use=original_native.acquire();const auto& native=use.context();
        if(native.degree()!=int(members.size()) || native.devices().size()!=members.size() ||
           (members.size()>1 && native.backend()!=CollectiveBackendType::NCCL && native.backend()!=CollectiveBackendType::RCCL))
            throw std::invalid_argument("GPU cohort service lost its original complete native context");
        for(size_t member=0;member<members.size();++member)
            if(!members[member].address.isGPU() || members[member].world_rank!=memory->worldRank() ||
               native.devices()[member]!=members[member].address ||
               (members.size()>1 && (members[member].address.toLocalDeviceId().is_cuda()
                    ? native.backend()!=CollectiveBackendType::NCCL : native.backend()!=CollectiveBackendType::RCCL)))
                throw std::invalid_argument("GPU cohort service changed original native member order or rank");
    }
    std::vector<std::unique_ptr<PreparedCohortLane>> lanes;
    lanes.reserve(members.size());
    // Finish every source/family check before any member records a collective.
    for(size_t member=0;member<members.size();++member) {
        auto lane=std::make_unique<PreparedCohortLane>();
        const auto endpoint=registry.endpoint(members[member].participant_id);
        lane->bank=endpoint?endpoint->acquire(epoch):MoEOverlayParticipantBankLease{};
        if(!lane->bank || lane->bank->device!=members[member].address.toLocalDeviceId() ||
           !lane->bank->containsModelLayer(layer))
            throw std::invalid_argument("GPU cohort service omitted an original prepared member bank");
        const auto& bank=lane->bank->layerForModelLayer(layer);
        lane->family=bank.fragment_family;
        if(!bank.valid(e) || !lane->family || lane->family->cohort()!=cohort ||
           lane->family->participantIndex()!=member || lane->family->preparation()!=source->preparation() ||
           !lane->family->accepts(bank.fragments.at(expert),expert) || !bank.resident_mask.at(expert) ||
           (member && lane->family->model()!=lanes.front()->family->model()))
            throw std::invalid_argument("GPU cohort service has foreign model/family/member source provenance");
        lane->graph=calibrations[member];lane->device=members[member].address.toLocalDeviceId();
        if(!lane->graph || lane->graph->purpose_!=RetainedParticipantGraphExecutable::DefinitionPurpose::PreparedServiceCalibration ||
           lane->graph->phase_!=RetainedParticipantGraphExecutable::Phase::Fresh || lane->graph->device_!=lane->device ||
           lane->graph->memory_.get()!=memory.get() || lane->graph->memory_.owner_before(memory) ||
           memory.owner_before(lane->graph->memory_) || lane->graph->worker_->ownsCurrentThread())
            throw std::invalid_argument("GPU cohort service requires each same-source fresh original calibration owner");
        lane->graph->requireLiveOriginal();lane->worker=lane->graph->worker_;lane->stream=lane->graph->stream_;
        lane->backend=getBackendFor(lane->device);
        requireCohort(lane->backend && lane->stream,"missing exact original backend/stream");
        lanes.push_back(std::move(lane));
    }
    ILocalTPContext* active_native=nullptr;
    /** @brief Expose only the context protected by this exact enclosing scoped borrow. */
    struct ActiveNativeScope {
        ILocalTPContext*& current;
        /** @brief Bind the exact context protected by the enclosing revocable use. */
        explicit ActiveNativeScope(ILocalTPContext*& slot,ILocalTPContext& context):current(slot){current=&context;}
        /** @brief Clear the alias before its original scoped borrow releases. */
        ~ActiveNativeScope(){current=nullptr;}
    };
    const auto abort_native=[&]() noexcept {
        try{
            if(active_native) {if(members.size()>1 && !active_native->signalNativeCompletionAbort())std::terminate();}
            else {const auto use=original_native.acquire();if(members.size()>1 && !use.context().signalNativeCompletionAbort())std::terminate();}
        }
        catch(...){std::terminate();}
    };
    const auto join_workers=[&](std::span<std::future<void>> pending) {
        std::exception_ptr first;
        joinOriginalNativeWorkerSubmissions(pending,[&](std::exception_ptr failure){
            if(!first)first=failure;abort_native();},
            std::chrono::steady_clock::now()+std::chrono::seconds(30));
        if(first)std::rethrow_exception(first);
    };
    const auto dispatch_all=[&](auto body) {
        std::vector<std::future<void>> pending;pending.reserve(lanes.size());
        try {for(size_t member=0;member<lanes.size();++member)
            pending.push_back(lanes[member]->worker->submitAsync([&,member]{body(member);}));}
        catch(...) {const auto failure=std::current_exception();abort_native();join_workers(pending);std::rethrow_exception(failure);}
        join_workers(pending);
    };
    const auto await_terminals=[&] {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for(;;) {
            bool complete=true;
            for(auto& lane:lanes)complete=lane->graph->observeTerminal() && complete;
            if(complete)return;
            if(std::chrono::steady_clock::now()>=deadline)
                throw std::runtime_error("Original GPU cohort service terminal exceeded 30s");
            std::this_thread::yield();
        }
    };
    const auto submit_all=[&] {
        std::vector<std::future<void>> pending;pending.reserve(lanes.size());
        try{for(size_t member=0;member<lanes.size();++member)pending.push_back(lanes[member]->worker->submitAsync([&,member]{
            auto& lane=*lanes[member];
            requireCohort(lane.graph->submit(lane.graph->lastSubmission_+1),"original cohort submission rejected");}));}
        catch(...){abort_native();join_workers(pending);throw;}
        join_workers(pending);
    };
    const auto elapsed=[&](PreparedCohortLane& lane) {
        float ms=0;bool accepted=false;
        lane.worker->submitAndWait([&]{accepted=lane.backend->eventElapsedTimeMs(
            lane.graph->calibrationTimingEvents_[0],lane.graph->calibrationTimingEvents_[1],lane.device.ordinal,&ms);});
        if(!accepted || !std::isfinite(ms) || ms<=0 || double(ms)*1.0e6>=double(UINT64_MAX))
            throw std::runtime_error("Original GPU cohort has no completed representable native timing");
        return uint64_t(std::ceil(double(ms)*1.0e6));
    };
    PlanningGPUCohortServiceObservation result{.target=MoEOverlayRouteTarget::cohort(placement->tier_idx),
        .layer=layer,.expert=expert,.phase=phase,.rows_per_frame=uint64_t(rows),
        .measured_frames=PlanningExecutionMeasurement::kTimedInvocations,.participants=members.size()};
    try {
        dispatch_all([&](size_t member) {
            const auto use=original_native.acquire();auto& native=use.context();auto& lane=*lanes[member];
            const auto& bank=lane.bank->layerForModelLayer(layer);
            const auto& ownership=cohort->ownership(member);
            const auto roots=MoEOverlayCPUServiceMeasurement::gpuJointStorageLayout(
                {h,source_geometry.intermediate_columns},e,rows);
            lane.tensor_device_claim=memory->claimNewAllocation(lane.device,
                PhysicalMemoryOwner::ExecutionWorkspace,roots.deviceBytes());
            const auto runtime_bytes=DeviceMoERuntimeTable::preparedFragmentServiceStorageGeometry(lane.device,e,rows);
            lane.runtime_host_claim=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,runtime_bytes.host_bytes);
            lane.runtime_device_claim=memory->claimNewAllocation(lane.device,PhysicalMemoryOwner::ExecutionWorkspace,runtime_bytes.device_bytes);
            const auto tensor=[&](const CPUStorageRoot<float>& root,std::vector<size_t> shape) {
                return std::make_unique<FP32Tensor>(shape,root.allocate(memory),DeviceId::cpu());};
            lane.hidden=tensor(roots.common.input,{size_t(rows),size_t(h)});
            lane.indices=tensor(roots.common.indices,{size_t(rows),1});lane.weights=tensor(roots.common.weights,{size_t(rows),1});
            lane.output=tensor(roots.common.output,{size_t(rows),size_t(h)});
            lane.canonical=tensor(roots.canonical_routes,{size_t(rows),1,size_t(h)});lane.release=tensor(roots.release,{1});
            lane.live_rows=std::make_unique<INT32Tensor>(std::vector<size_t>{1},roots.active_rows.allocate(memory),DeviceId::cpu());
            lane.expected=roots.expected_output.allocate(memory);lane.observed=roots.observed_output.allocate(memory);
            lane.histogram=roots.histogram_evidence.allocate(memory);
            for(int row=0;row<rows;++row) {
                lane.indices->mutable_typed_data()[row]=float(expert);lane.weights->mutable_typed_data()[row]=1.f;
                for(int col=0;col<h;++col)lane.hidden->mutable_typed_data()[size_t(row)*h+col]=
                    0.0001f*float((col%31*13+row*7)%31-15);
            }
            lane.release->mutable_typed_data()[0]=0;lane.live_rows->mutable_typed_data()[0]=rows;
            for(auto* input:{lane.hidden.get(),lane.indices.get(),lane.weights.get(),lane.release.get()})
                TransferEngine::prepareDeviceInput(input,lane.device,lane.stream);
            TransferEngine::prepareDeviceInput(lane.live_rows.get(),lane.device,lane.stream);
            TransferEngine::prepareDeviceOutput(lane.output.get(),lane.device,lane.stream);
            TransferEngine::prepareDeviceOutput(lane.canonical.get(),lane.device,lane.stream);
            lane.runtime=std::make_unique<DeviceMoERuntimeTable>(DeviceMoERuntimeTable::Config{
                .device_id=lane.device,.num_layers=1,.num_experts=e,.top_k=1,.mirror_to_device=true,
                .prefill_token_capacity=rows,.fragment_families={lane.family},.first_model_layer=layer});
            MoEPlacementUpdate update;update.epoch=uint32_t(epoch);update.expert_count=uint32_t(e);
            update.participant_id=uint32_t(member);update.participant_count=uint32_t(members.size());
            update.experts.resize(e);update.local_compute_mask.assign(e,0);update.replica_role.assign(e,0);
            update.resident_participant_mask.assign(e,0);update.overlay_route_participant.resize(e);
            for(int id=0;id<e;++id) {
                const auto* original=registry.initialOwnerMap().placementFor(layer,id);
                requireCohort(original!=nullptr,"missing original placement row");
                update.overlay_route_participant[id]=original->shard_cohort
                    ? MoEOverlayRouteTarget::cohort(original->tier_idx).word():MoEOverlayRouteTarget::whole(original->owner_participant).word();
                auto& descriptor=update.experts[id];descriptor.logical_expert_id=id;
                descriptor.projection_set=DeviceMoEProjectionSet::IntermediateShard;
                if(!bank.resident_mask[id])continue;
                requireCohort(lane.family->accepts(bank.fragments[id],id) &&
                    exportDeviceMoEFragmentWeightDescriptors(bank.fragments[id],descriptor),"foreign original prepared fragment");
                descriptor.local_slot=id;descriptor.flags=toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                    DeviceMoEExpertFlags::Resident | DeviceMoEExpertFlags::LocalCompute);update.local_compute_mask[id]=1;
            }
            requireCohort(lane.runtime->prepareInactiveBank(layer,update) &&
                lane.runtime->flipActiveBank(layer,uint32_t(epoch),lane.stream),"private original runtime publication failed");
            lane.runtime->hostLayerState(layer).topk_expert_ids[0]=expert;
            lane.runtime->hostLayerState(layer).topk_weights[0]=1;
            lane.runtime->sealAndPublishCompleteInitialRuntimeState(lane.stream);
            lane.kernel_owner=std::make_shared<MoERoutedPipelineKernelOwner>();
            lane.context=IDeviceContext::create(lane.device);
            const auto service_phase=phase==ExpertHistogramSource::DecodeToken?MoEOverlayServicePhaseHint::Decode:
                phase==ExpertHistogramSource::PrefillChunk?MoEOverlayServicePhaseHint::Prefill:MoEOverlayServicePhaseHint::GroupedVerifier;
            MoEIntermediateShardPipelineParams local;
            local.family=lane.family;local.runtime=lane.runtime.get();local.kernel_owner=lane.kernel_owner;
            local.rows=rows;local.top_k=1;local.phase=service_phase;local.demand=MoEGroupedPlanDemand::None;
            local.row_mode=phase==ExpertHistogramSource::DecodeToken?MoEIntermediateShardRowMode::Decode:MoEIntermediateShardRowMode::Grouped;
            local.prefix="original_prepared_service/";local.hidden={lane.hidden.get(),BufferId::NORMALIZED};
            local.routing_indices={lane.indices.get(),BufferId::MOE_EXPERT_INDICES};
            local.routing_weights={lane.weights.get(),BufferId::MOE_EXPERT_WEIGHTS};
            local.route_partials={lane.canonical.get(),BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS};
            local.output={lane.output.get(),BufferId::MOE_COMBINED_OUTPUT};
            const MoEIntermediateShardPreparedServiceRoute route(lane.family,lane.runtime.get(),expert,epoch,rows,
                service_phase,lane.indices.get(),lane.weights.get());
            const auto partial=appendPreparedMoEIntermediateShardPipeline(lane.stages,local,bank,route);
            lane.compute_name=local.prefix+"local_fragment_compute";lane.fold_name=partial.producer;
            TPAllreduceStage::Params sum;sum.device_id=lane.device;sum.tp_ctx=&native;sum.tensor=lane.output.get();
            sum.count=size_t(rows)*h;sum.precision="fp32";sum.arithmetic_policy=TPAllreduceArithmeticPolicy::CanonicalRankOrder;
            sum.tensor_buffer_id=BufferId::MOE_COMBINED_OUTPUT;sum.sideband_device_index=int(member);
            sum.stage_name="original_prepared_cohort_sum";
            sum.live_rows=NativeCollectiveRows(DeviceRowRange::deviceCounted(rows,
                static_cast<const int32_t*>(lane.live_rows->gpu_data_ptr())),h);
            auto sum_stage=std::make_unique<TPAllreduceStage>(sum);lane.sum=sum_stage.get();lane.sum_name=sum.stage_name;
            lane.stages.addNode(lane.sum_name,std::move(sum_stage),lane.device);
            lane.stages.addDependency(lane.sum_name,lane.fold_name);
            WorkspaceRequirements requirements;
            for(const auto& name:lane.stages.getExecutionOrder())
                if(auto* consumer=dynamic_cast<IWorkspaceConsumer*>(lane.stages.getNode(name)->stage.get()))
                    requirements.merge(consumer->getWorkspaceRequirements(rows));
            const auto physical=std::find_if(inventory.gpus.begin(),inventory.gpus.end(),[&](const auto& gpu){
                return gpu.type==lane.device.type && gpu.local_device_id==lane.device.ordinal;});
            requireCohort(physical!=inventory.gpus.end() && physical->compute_units>0,"missing actual GPU compute geometry");
            const auto declared=WorkspaceMemoryEstimator::estimatePreparedExpertService(profile,
                {.device=lane.device,.device_compute_units=physical->compute_units,.resident_graph_rows=rows,
                 .first_layer=layer,.last_layer=layer},ownership);
            requireCohort(requirements.total_bytes_with_alignment()<=declared,"cohort invocation exceeded canonical source quote");
            lane.workspace=std::make_unique<DeviceWorkspaceManager>(lane.device,requirements.total_bytes_with_alignment(),memory);
            requireCohort(lane.workspace->allocate(requirements),"cohort invocation admission failed");
            for(const auto& name:lane.stages.getExecutionOrder()) {
                auto* stage=lane.stages.getNode(name)->stage.get();
                if(auto* consumer=dynamic_cast<IWorkspaceConsumer*>(stage))consumer->bindWorkspace(lane.workspace.get());
                requireCohort(stage->prepareGraphLaunch(lane.context.get(),lane.stream),"original stage capture preparation failed");
            }
            const std::map<BufferId,TensorBase*> buffers{{BufferId::NORMALIZED,lane.hidden.get()},
                {BufferId::MOE_EXPERT_INDICES,lane.indices.get()},{BufferId::MOE_EXPERT_WEIGHTS,lane.weights.get()},
                {BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,lane.canonical.get()},{BufferId::MOE_COMBINED_OUTPUT,lane.output.get()}};
            std::map<BufferId,size_t> writers;
            for(const auto& name:lane.stages.getExecutionOrder()) {
                auto* stage=lane.stages.getNode(name)->stage.get();
                GraphCaptureDependencyLedger::StagePlan plan{.stage_identity=stage,.stage_name=name};
                const auto contract=stage->bufferContract();
                for(const auto& input:contract.allArenaReads()) {
                    const auto* tensor=buffers.at(input.id);
                    if(const auto writer=writers.find(input.id);writer!=writers.end())plan.internal_inputs.push_back({tensor,writer->second});
                    else plan.external_inputs.push_back(tensor);
                }
                for(auto* weight:contract.weight_tensors)plan.external_inputs.push_back(dynamic_cast<TensorBase*>(weight));
                for(const auto& output:contract.allWrites()) {
                    plan.outputs.push_back(buffers.at(output.id));writers[output.id]=lane.dependencies.size();
                }
                lane.dependencies.push_back(std::move(plan));
            }
            for(auto* input:{lane.hidden.get(),lane.indices.get(),lane.weights.get(),lane.release.get()})
                TransferEngine::requireDeviceInput(input,lane.device,lane.stream);
            TransferEngine::requireDeviceInput(lane.live_rows.get(),lane.device,lane.stream);
        });
        for(const auto mode:{CohortObservationMode::Complete,CohortObservationMode::ConcurrentCompute,
                            CohortObservationMode::LiveCommunication,CohortObservationMode::EmptyCommunication}) {
            std::vector<OriginalNativeGraphRecording> recording;recording.reserve(lanes.size());
            std::vector<std::unique_ptr<GraphCaptureDependencyLedger>> ledgers;ledgers.reserve(lanes.size());
            for(size_t member=0;member<lanes.size();++member) {
                auto& lane=*lanes[member];
                lane.worker->submitAndWait([&] {
                    const int count=mode==CohortObservationMode::EmptyCommunication?0:rows;
                    lane.live_rows->mutable_typed_data()[0]=count;
                    TransferEngine::prepareDeviceInput(lane.live_rows.get(),lane.device,lane.stream);
                    TransferEngine::requireDeviceInput(lane.live_rows.get(),lane.device,lane.stream);
                    for(auto* input:{lane.hidden.get(),lane.indices.get(),lane.weights.get(),lane.release.get()})
                        TransferEngine::requireDeviceInput(input,lane.device,lane.stream);
                });
                ledgers.push_back(std::make_unique<GraphCaptureDependencyLedger>(lane.device,lane.stream,lane.dependencies,
                    "original_prepared_cohort"));
                recording.push_back({.worker=lane.worker,.device=lane.device,.capture=&lane.graph->capture(),.dependencies=ledgers.back().get(),
                    .record=[&,member,mode] {
                        const auto use=original_native.acquire();auto& native=use.context();auto& member_lane=*lanes[member];
                        const auto marker=[&](int index) {
                            // These clocks are nodes of this original captured
                            // definition, not external coherence publications.
                            // The worker's checked event API admits its exact
                            // capture stream; IBackend deliberately rejects
                            // external event publication during native capture.
                            requireCohort(member_lane.worker->recordCapturedTimingEventChecked(
                                member_lane.graph->calibrationTimingEvents_[index],member_lane.stream),
                                "captured component clock marker failed");};
                        // Establish the same input-ready rendezvous that the
                        // CPU cohort probe places outside its root interval.
                        // Captured clock markers therefore exclude host worker
                        // submission skew without reading a device count.
                        const std::array input_ready{LocalTPCollectiveSidebandBuffer{
                            .kind=LocalTPCollectiveSidebandKind::AllreduceSum,.send_buffer=nullptr,
                            .recv_buffer=member_lane.release->gpu_data_ptr(),.element_count=1,
                            .dtype=CollectiveDataType::FLOAT32,.root_device_index=0,.name="original_prepared_input_ready"}};
                        if(members.size()>1)requireCohort(native.collectiveSidebandSpanOnStream(input_ready,int(member),
                            member_lane.stream,"original_prepared_input_ready"),"original input-ready rendezvous failed");
                        if(mode==CohortObservationMode::Complete)marker(0);
                        // One original native start-release follows ready input.
                        // Singleton execution is its original identity, with no
                        // fabricated peer, HOST staging or native wire bytes.
                        const std::array start{LocalTPCollectiveSidebandBuffer{
                            .kind=LocalTPCollectiveSidebandKind::Broadcast,.send_buffer=member_lane.release->gpu_data_ptr(),
                            .recv_buffer=member_lane.release->gpu_data_ptr(),.element_count=1,
                            .dtype=CollectiveDataType::FLOAT32,.root_device_index=0,.name="original_prepared_start"}};
                        if(members.size()>1)requireCohort(native.collectiveSidebandSpanOnStream(start,int(member),
                            member_lane.stream,"original_prepared_cohort_start"),"original start release failed");
                        if(mode==CohortObservationMode::ConcurrentCompute)marker(0);
                        for(const auto* name:{&member_lane.compute_name,&member_lane.fold_name}) {
                            auto* stage=member_lane.stages.getNode(*name)->stage.get();
                            ScopedGraphCaptureStage scope(stage);
                            requireCohort(stage->execute(member_lane.context.get()),"original prepared compute/fold stage failed");
                            scope.complete();
                        }
                        if(mode==CohortObservationMode::ConcurrentCompute)marker(1);
                        if(mode==CohortObservationMode::LiveCommunication || mode==CohortObservationMode::EmptyCommunication) {
                            // This setup-only input-ready rendezvous is outside
                            // the communication clock. It prevents faster local
                            // compute from being mispriced as native sum time.
                            const std::array ready{LocalTPCollectiveSidebandBuffer{
                                .kind=LocalTPCollectiveSidebandKind::AllreduceSum,.send_buffer=nullptr,
                                .recv_buffer=member_lane.release->gpu_data_ptr(),.element_count=1,
                                .dtype=CollectiveDataType::FLOAT32,.root_device_index=0,.name="original_prepared_sum_ready"}};
                            if(members.size()>1)requireCohort(native.collectiveSidebandSpanOnStream(ready,int(member),
                                member_lane.stream,"original_prepared_sum_input_ready"),"original communication rendezvous failed");
                            marker(0);
                        }
                        {
                            ScopedGraphCaptureStage scope(member_lane.sum);
                            requireCohort(member_lane.sum->execute(member_lane.context.get()),"original complete cohort sum failed");
                            scope.complete();
                        }
                        if(mode==CohortObservationMode::Complete || mode==CohortObservationMode::LiveCommunication ||
                           mode==CohortObservationMode::EmptyCommunication)marker(1);
                    }});
            }
            const auto native_capture=members.size()>1?std::optional{original_native}:std::optional<NativeLocalTPContextBorrow>{};
            OriginalNativeGraphCaptureGroup::record(recording,native_capture,
                "original_prepared_cohort_layer"+std::to_string(layer)+"_expert"+std::to_string(expert)+
                "_phase"+std::to_string(int(phase))+"_mode"+std::to_string(int(mode)));
            dispatch_all([&](size_t member) {
                auto& lane=lanes[member];auto& capture=lane->graph->capture();
                requireCohort(capture.nodeCount()>0 && capture.prepareRuntimeContextStorage(lane->graph->memory_) &&
                    capture.instantiate(),"complete original calibration materialization failed");});
            {
                // Actual GPU invocations retain the SAME original context until
                // every native terminal. The matched capture group's own cold
                // borrow has already ended, avoiding recursive shared locking.
                const auto native_use=original_native.acquire();
                ActiveNativeScope active(active_native,native_use.context());
                submit_all();await_terminals(); // One complete untimed warmup.
                auto& root=*lanes.front();
                for(int sample=0;sample<PlanningExecutionMeasurement::kTimedInvocations;++sample) {
                    submit_all();await_terminals();
                    uint64_t duration=0;
                    if(mode==CohortObservationMode::ConcurrentCompute)
                        for(auto& lane:lanes)duration=std::max(duration,elapsed(*lane));
                    else if(mode==CohortObservationMode::Complete || members.size()>1)duration=elapsed(root);
                    auto* total=mode==CohortObservationMode::Complete?&result.complete_service_nanoseconds:
                        mode==CohortObservationMode::ConcurrentCompute?&result.concurrent_compute_nanoseconds:
                        mode==CohortObservationMode::LiveCommunication?&result.live_frame_communication_nanoseconds:
                        &result.empty_frame_communication_nanoseconds;
                    if(duration>UINT64_MAX-*total)throw std::overflow_error("Original cohort component interval overflow");
                    *total+=duration;
                }
                // Observation is terminal-only. No output/epoch/histogram value
                // is downloaded to drive a captured decision or repair state.
                for(auto& lane:lanes)lane->worker->submitAndWait([&] {
                    const auto copy=[&](void* destination,const void* source_pointer,size_t bytes) {
                        requireCohort(lane->backend->deviceToHost(destination,source_pointer,bytes,
                            lane->device.ordinal,lane->stream),"terminal component evidence download failed");};
                    copy(lane->observed.data(),lane->output->gpu_data_ptr(),size_t(rows)*h*sizeof(float));
                    const auto* state=reinterpret_cast<const std::byte*>(lane->runtime->deviceLayerState(layer));
                    constexpr std::array offsets{offsetof(DeviceMoELayerRuntime,decode_histogram),
                        offsetof(DeviceMoELayerRuntime,decode_local_histogram),offsetof(DeviceMoELayerRuntime,prefill_histogram),
                        offsetof(DeviceMoELayerRuntime,prefill_local_histogram),offsetof(DeviceMoELayerRuntime,grouped_verifier_histogram),
                        offsetof(DeviceMoELayerRuntime,grouped_verifier_local_histogram)};
                    for(size_t plane=0;plane<offsets.size();++plane)
                        copy(lane->histogram.data()+plane*e,state+offsets[plane],size_t(e)*sizeof(uint64_t));
                    requireCohort(lane->backend->recordEvent(lane->graph->calibrationTimingEvents_[1],
                        lane->device.ordinal,lane->stream),"terminal evidence event publication failed");
                });
                // Use the original timing event only as this setup observation's
                // exact terminal; native graph definition retirement follows it.
                for(auto& lane:lanes) {
                    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                    for(;;) {
                        bool queried=false,ready=false;lane->worker->submitAndWait([&]{queried=lane->backend->queryEvent(
                            lane->graph->calibrationTimingEvents_[1],lane->device.ordinal,&ready);});
                        requireCohort(queried,"terminal evidence event query failed");if(ready)break;
                        if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("Cohort evidence exceeded 30s");
                        std::this_thread::yield();
                    }
                    requireCohort(std::all_of(lane->histogram.begin(),lane->histogram.end(),[](auto value){return value==0;}),
                        "prepared-only cohort probe authored demand");
                    requireCohort(std::all_of(lane->observed.begin(),lane->observed.end(),[](float value){return std::isfinite(value);}),
                        "prepared cohort produced a nonfinite output");
                    if(mode==CohortObservationMode::Complete)
                        std::copy(lane->observed.begin(),lane->observed.end(),lane->expected.begin());
                    else if(mode!=CohortObservationMode::EmptyCommunication)
                        requireCohort(std::equal(lane->observed.begin(),lane->observed.end(),lane->expected.begin(),
                            [](float actual,float expected){return std::bit_cast<uint32_t>(actual)==std::bit_cast<uint32_t>(expected);}),
                            "component observation changed captured complete FFN bytes");
                }
                if(mode!=CohortObservationMode::EmptyCommunication)
                    for(size_t member=1;member<lanes.size();++member)
                        requireCohort(std::equal(lanes[member]->observed.begin(),lanes[member]->observed.end(),
                            lanes.front()->observed.begin(),[](float actual,float expected){
                                return std::bit_cast<uint32_t>(actual)==std::bit_cast<uint32_t>(expected);}),
                            "complete cohort participants disagree on canonical output bytes");
                for(auto& lane:lanes)lane->graph->resetPreparedCalibration();
            }
        }
        // Every source-owned native definition is Fresh before its private
        // runtime/workspace/tensors retire on the exact original worker.
        const auto teardown_native=original_native.acquire();
        ActiveNativeScope teardown_active(active_native,teardown_native.context());
        for(auto& lane:lanes)lane->worker->submitAndWait([&] {
            for(const auto& name:lane->stages.getExecutionOrder())
                lane->stages.getNode(name)->stage->invalidateKernelDynamicState();
            lane->stages.clear();lane->sum=nullptr;lane->kernel_owner.reset();
            lane->runtime.reset();lane->workspace.reset();lane->context.reset();
            lane->live_rows.reset();lane->release.reset();lane->output.reset();lane->canonical.reset();
            lane->weights.reset();lane->indices.reset();lane->hidden.reset();
        });
        return result;
    } catch(...) {
        const auto failure=std::current_exception();abort_native();
        try {
            const auto teardown_native=original_native.acquire();
            ActiveNativeScope teardown_active(active_native,teardown_native.context());
            await_terminals();
            for(auto& lane:lanes)if(lane->graph->phase_!=RetainedParticipantGraphExecutable::Phase::Fresh)
                lane->graph->retireDefinition();
            for(auto& lane:lanes)lane->worker->submitAndWait([&] {
                for(const auto& name:lane->stages.getExecutionOrder())
                    lane->stages.getNode(name)->stage->invalidateKernelDynamicState();
                lane->stages.clear();lane->sum=nullptr;lane->kernel_owner.reset();
                lane->runtime.reset();lane->workspace.reset();lane->context.reset();
                lane->live_rows.reset();lane->release.reset();lane->output.reset();lane->canonical.reset();
                lane->weights.reset();lane->indices.reset();lane->hidden.reset();
            });
        } catch(...) {std::terminate();} // Missing native retirement never permits freeing live source bytes.
        std::rethrow_exception(failure);
    }
}
} // namespace llaminar2
