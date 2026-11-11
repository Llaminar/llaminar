/**
 * @file MoECPUExpertShardRouteStage.cpp
 * @brief Exact original TP membership and source-native compact CPU shard publication.
 *
 * Ordinary sparse prefill retains its original replicated CPU cohort packet as
 * live-size transport authority. Decode/verifier and generic records keep their
 * original transport policy; no count exchange or alternate arithmetic is added.
 */
#include "MoECPUExpertShardRouteStage.h"
#include "../../moe/CPUExpertShardSourceSelection.h"
#include "../../moe/MoEOverlaySparseCollective.h"
#include "../../moe/MoEOverlayActivationEpochProtocol.h"
#include "../../moe/CanonicalMoERouteRecord.h"
#include "../../../kernels/cpu/moe/CPUCanonicalShardRouteRecords.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/Logger.h"
#include "../../../collective/GlobalTPContext.h"
#include <limits>
#include <chrono>
#include <array>
#include <stdexcept>
namespace llaminar2 {
namespace {
/**
 * @brief Measure the completed host interval without manufacturing a nonzero sample.
 * @param begin Start of the original operation in the steady-clock domain.
 * @return Elapsed nanoseconds, retaining an actual zero-length observation.
 */
std::uint64_t rootTimingElapsed(std::chrono::steady_clock::time_point begin) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-begin).count());
}
}
MoECPUExpertShardRouteStage::RootTimingObservation::RootTimingObservation(
    MoECPUExpertShardRouteStage& stage,bool enabled) {
    if(!enabled || stage.params_.tp->myIndex()!=stage.params_.root_participant)return;
    if(stage.timing_observation_)throw std::logic_error("CPU root timing scope already borrowed");
    stage_=&stage;stage.timing_observation_=this;
}
MoECPUExpertShardRouteStage::RootTimingObservation::~RootTimingObservation() {
    if(stage_)stage_->timing_observation_=nullptr;
}

MoECPUExpertShardRouteStage::MoECPUExpertShardRouteStage(Params params)
    : IComputeStage(DeviceId::cpu()),params_(std::move(params)) {
    const auto &p=params_;
    if(p.original_sparse_routes && p.standalone_execution_phase)
        throw std::invalid_argument("CPU shard sparse execution already owns its original phase");
    if(!p.original_sparse_routes && (!p.standalone_execution_phase ||
        expertHistogramProductionSourceIndex(*p.standalone_execution_phase)==kExpertHistogramProductionSourceCount))
        throw std::invalid_argument("CPU shard standalone execution requires an explicit production phase");
    if(!p.original_source || !p.tp || p.tp->scope()!=TPScope::NODE_LOCAL ||
       !p.packed_records ||
       (!p.original_sparse_routes && (!p.route_indices || !p.route_weights)) ||
       (p.original_sparse_routes && (p.route_indices || p.route_weights)) ||
       (p.tp->myIndex()==p.root_participant && !p.combined_routes) ||
       p.rows<=0 || p.top_k<=0 || p.columns<=0 || p.root_participant<0 ||
       p.root_participant>=p.tp->degree())
        throw std::invalid_argument("CPU expert shard publication requires original NODE_LOCAL source/context/buffers");
    const auto &source=*p.original_source;
    const auto &cohort=source.cohort();
    if(!cohort || cohort->participants().size()!=static_cast<size_t>(p.tp->degree()) ||
       source.participantIndex()!=static_cast<size_t>(p.tp->myIndex()) ||
       p.tp->worldRanks().size()!=cohort->participants().size())
        throw std::invalid_argument("CPU expert shard publication has foreign original cohort membership");
    for(size_t member=0;member<cohort->participants().size();++member)
        if(p.tp->worldRanks()[member]!=cohort->participants()[member].world_rank)
            throw std::invalid_argument("CPU expert shard publication reordered original rank membership");
    const auto local=p.tp->localDevice();
    const auto &original=cohort->participants()[source.participantIndex()];
    if(!local.isCPU() || local.device_ordinal!=original.address.device_ordinal ||
       local.numa_node!=original.address.numa_node ||
       source.participant().participant_id!=original.participant_id ||
       source.ownership().geometry().model_columns!=p.columns)
        throw std::invalid_argument("CPU expert shard publication changed original rank/NUMA/source geometry");
    const size_t rows=static_cast<size_t>(p.rows),k=static_cast<size_t>(p.top_k),h=static_cast<size_t>(p.columns);
    const auto maximum=std::numeric_limits<size_t>::max(),degree=cohort->participants().size();
    if(rows>maximum/k || rows*k>maximum/h || rows*k*h>maximum/sizeof(float) ||
       rows*k>maximum/degree || rows*k*degree>maximum/sizeof(int32_t) ||
       rows*k>static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("CPU expert shard publication live geometry overflows original records");
    const auto slots=rows*k;
    for(const auto *tensor:{p.packed_records,p.route_indices,p.route_weights,p.combined_routes})
        if(tensor && (tensor->native_type()!=TensorType::FP32 || !tensor->home_device().is_cpu()))
            throw std::invalid_argument("CPU expert shard publication requires original host FP32 route tensors");
    const bool root=p.tp->myIndex()==p.root_participant;
    const auto capacity=canonical_moe_route_record::recordCapacity(p.packed_records->numel(),p.columns);
    if(capacity<(root?slots*degree:slots) ||
       (!p.original_sparse_routes && (p.route_indices->numel()<slots || p.route_weights->numel()<slots)) || (root && p.combined_routes->numel()<slots*h) ||
       (root && (p.source_ranges.size()!=degree || p.participant_route_maps.size()<degree*slots ||
                 p.required_route_weights.size()<slots)) || (!root && !p.source_ranges.empty()))
        throw std::invalid_argument("CPU expert shard publication lacks its original admitted live record/scratch capacity");
    // A receive may mutate its packed bank before the semantic validator runs.
    // Reject all aliases cold, so neither transport headers nor scratch can
    // overwrite original routing state while discovering malformed records.
    if(p.packed_records->numel()>maximum/sizeof(float))
        throw std::invalid_argument("CPU expert shard packed byte extent overflows");
    const auto *sparse=p.original_sparse_routes;
    if(sparse && (sparse->entry_capacity<slots || sparse->d_model!=p.columns || sparse->top_k!=p.top_k ||
       !sparse->original_route_slots_host || !sparse->expert_ids_host || !sparse->route_weights_host))
        throw std::invalid_argument("CPU shard gather requires exact original sparse route roots");
    const bool host_canonical=p.return_publication==OriginalReturnPublication::CanonicalHostRows;
    if(host_canonical && (!sparse || !p.original_return_rows || p.original_canonical_ticket || p.original_preweighted_routes ||
        !p.original_continuation_device.is_cpu() || p.original_return_rows->layout!=MoEOverlayReturnLayout::CanonicalExpertRoutes ||
        p.original_return_rows->d_model!=p.columns || p.original_return_rows->row_capacity<slots ||
        p.original_return_rows->row_capacity>maximum/h/sizeof(float) ||
        p.original_return_rows->row_capacity>maximum/sizeof(int32_t)))
        throw std::invalid_argument("CPU joint host return requires its exact original unweighted canonical packet");
    if(host_canonical) {
        const auto* native=dynamic_cast<GlobalTPContext*>(p.tp);
        if(!native || !p.original_tp_owner || !native->authenticatesOriginalCPUExpertBorrow(p.original_tp_owner))
            throw std::invalid_argument("CPU joint host return requires its original admitted native CPU domain");
    }
    if(sparse && (!p.original_return_rows || (root && !host_canonical && !p.original_canonical_ticket && (!p.original_preweighted_routes ||
       p.original_return_route_capacity<slots))))
        throw std::invalid_argument("CPU shard sparse return requires the exact original endpoint return bank");
    if(p.original_prepared_layer && (!p.original_prepared_layer->sourceSelection() ||
       p.original_prepared_layer->sourceSelection()!=p.original_source ||
       p.original_prepared_layer->participantId()!=source.participant().participant_id ||
       p.original_prepared_layer->layer()!=source.layer()))
        throw std::invalid_argument("CPU shard gather replaced its original prepared source publication");
    if(p.original_tp_owner && p.original_tp_owner.get()!=p.tp)
        throw std::invalid_argument("CPU shard gather has a foreign retained context owner");
    if(p.original_canonical_ticket && (!sparse || !p.original_prepared_layer || !p.original_tp_owner ||
       p.original_preweighted_routes || !p.original_canonical_ticket->hasValidConstructionIdentity() ||
       p.original_canonical_ticket->layerIndex()!=p.original_source->layer() ||
       p.original_canonical_ticket->dModel()!=p.columns || !p.original_continuation_device.is_gpu() ||
       p.original_canonical_ticket->continuationDevice()!=p.original_continuation_device ||
       p.original_canonical_ticket->routeCapacity()<slots))
        throw std::invalid_argument("CPU joint canonical ticket requires its exact source/frame/context/layer/GPU binding");
    const std::array<std::pair<const void*,size_t>,13> regions{{
        {p.packed_records->data(),p.packed_records->numel()*sizeof(float)},
        {p.route_indices?p.route_indices->data():nullptr,p.route_indices?slots*sizeof(float):0},
        {p.route_weights?p.route_weights->data():nullptr,p.route_weights?slots*sizeof(float):0},
        {root?p.combined_routes->data():nullptr,root?slots*h*sizeof(float):0},
        {root?p.source_ranges.data():nullptr,root?degree*sizeof(RootedFloatRecordRange):0},
        {root?p.participant_route_maps.data():nullptr,root?degree*slots*sizeof(int32_t):0},
        {root?p.required_route_weights.data():nullptr,root?slots*sizeof(float):0},
        {sparse?sparse->original_route_slots_host:nullptr,sparse?slots*sizeof(int32_t):0},
        {sparse?sparse->expert_ids_host:nullptr,sparse?slots*sizeof(int32_t):0},
        {sparse?sparse->route_weights_host:nullptr,sparse?slots*sizeof(float):0},
        {root?p.original_preweighted_routes:nullptr,root && sparse && p.original_preweighted_routes?slots*h*sizeof(float):0},
        {root && host_canonical?p.original_return_rows->row_ids_host:nullptr,
            root && host_canonical?p.original_return_rows->row_capacity*sizeof(int32_t):0},
        {root && host_canonical?p.original_return_rows->output_rows_fp32:nullptr,
            root && host_canonical?p.original_return_rows->row_capacity*h*sizeof(float):0}}};
    for(size_t i=0;i<regions.size();++i) {
        const auto [pointer,bytes]=regions[i];
        if(!bytes)continue;
        const auto begin=reinterpret_cast<uintptr_t>(pointer);
        if(!pointer || bytes>std::numeric_limits<uintptr_t>::max()-begin)
            throw std::invalid_argument("CPU expert shard binding has no exact admitted byte extent");
        for(size_t prior=0;prior<i;++prior) {
            const auto [other,otherBytes]=regions[prior];
            const auto start=reinterpret_cast<uintptr_t>(other);
            if(otherBytes && begin<start+otherBytes && start<begin+bytes)
                throw std::invalid_argument("CPU expert shard original receive/router/scratch roots overlap");
        }
    }
}
bool MoECPUExpertShardRouteStage::execute(IDeviceContext* ctx)
{
    if(timing_observation_)timing_observation_->reset();
    const bool ok=executeWithBank(ctx,nullptr);
    if(ok && timing_observation_)timing_observation_->completed_=true;
    return ok;
}
bool MoECPUExpertShardRouteStage::executeWithBank(IDeviceContext *ctx,MoEOverlayParticipantBankLease* completed_bank) {
    auto &p=params_;
    std::optional<MoEOverlayCanonicalRouteReturnTicketStorage::Publication> publication;
    size_t local=0, gathered=0;
    MoEOverlayParticipantBankLease invocation_bank;
    const auto finish=[&](){if(completed_bank)*completed_bank=invocation_bank;return true;};
    const auto reject=[&](const char* reason){
        // Preserve the exact original rejection before requestAbort may terminate
        // the MPI job. These are borrowed host metadata, never runtime probes.
        const auto* input=p.original_sparse_routes;
        LOG_ERROR("[MoECPUExpertShardRouteStage] Rooted publication rejected: reason=" << reason
            << " stage=" << p.stage_name << " layer=" << p.original_source->layer()
            << " participant=" << p.original_source->participant().participant_id
            << " world_rank=" << p.original_source->participant().world_rank
            << " member=" << p.tp->myIndex() << " root=" << p.root_participant
            << " context=" << (ctx?ctx->deviceId().toString():"null")
            << " expected_epoch=" << (p.original_prepared_layer?p.original_prepared_layer->epoch():0u)
            << " packet_epoch=" << (input?input->residency_epoch:0u)
            << " live_rows=" << (input?input->live_row_count:0u)
            << " live_entries=" << (input?input->live_entry_count:0u)
            << " local_records=" << local << " gathered_records=" << gathered
            << " row_capacity=" << p.rows << " top_k=" << p.top_k);
        publication.reset(); // Cancel any armed lease before publishing the original abort.
        if(p.original_canonical_ticket && !p.original_canonical_ticket->publishAbort()) {
            LOG_ERROR("[MoECPUExpertShardRouteStage] Original ticket abort publication failed: stage=" << p.stage_name);
            std::terminate();
        }
        p.tp->requestAbort();return false;
    };
    if(p.original_sparse_routes) {
        if(!p.original_prepared_layer)return reject("original_prepared_layer_missing");
        auto epoch=p.original_prepared_layer->acquireEpoch(p.original_sparse_routes->residency_epoch);
        if(!epoch)return reject("original_sparse_epoch_bank");
        invocation_bank=epoch->bankLease();
    }
    if(p.original_canonical_ticket) {
        const auto* native=dynamic_cast<const GlobalTPContext*>(p.tp);
        const auto* input=p.original_sparse_routes;
        if(!hasOriginalCanonicalTicket())return reject("original_ticket_identity");
        if(!native)return reject("original_native_context_type");
        if(!native->authenticatesOriginalCPUExpertBorrow(p.original_tp_owner))return reject("original_native_context_borrow");
        if(!p.original_prepared_layer)return reject("original_prepared_layer_missing");
        if(!input)return reject("original_sparse_packet_missing");
        if(!invocation_bank || input->residency_epoch!=invocation_bank->epoch)return reject("original_sparse_epoch");
        if(input->key.layer_idx!=p.original_source->layer())return reject("original_sparse_layer");
        if(input->target_participant!=p.original_source->participant().participant_id)return reject("original_sparse_participant");
        publication.emplace(p.original_canonical_ticket->arm(input->residency_epoch));
        if(!*publication)return reject("original_ticket_arm");
    }
    if(!ctx)return reject("execution_context_missing");
    // This host service runs inside the original captured GPU parent. Its
    // scheduling context does not own CPU execution: source/NUMA and the exact
    // retained native CPU borrower were authenticated above. Only that same
    // ticket's continuation GPU may schedule it; ordinary followers stay CPU-only.
    const auto scheduling_device=ctx->deviceId();
    if(!acceptsSchedulingContext(scheduling_device))return reject("foreign_scheduling_context");
    if(p.tp->isAbortRequested())return reject("original_native_abort_requested");
    auto *records=p.packed_records->mutable_data();
    const size_t slots=static_cast<size_t>(p.rows)*static_cast<size_t>(p.top_k);
    local=canonical_moe_route_record::readRecordCount(records,p.packed_records->numel());
    if(local>slots) return reject("packed_count_exceeds_admission");
    const auto *sparse=p.original_sparse_routes;
    if(sparse && (!sparse->key.isValid() || sparse->key.layer_idx!=p.original_source->layer()))
        return reject("original_sparse_key");
    const auto source_phase=sparse?sparse->key.histogram_source:*p.standalone_execution_phase;
    const char *phase=nullptr;
    switch(source_phase) {
        case ExpertHistogramSource::DecodeToken: phase="decode";break;
        case ExpertHistogramSource::PrefillChunk: phase="prefill";break;
        case ExpertHistogramSource::GroupedVerifier: phase="grouped_verifier";break;
        case ExpertHistogramSource::SyntheticTest: return reject("original_execution_phase");
        default: return reject("original_execution_phase");
    }
    if(sparse && (sparse->live_entry_count>slots || sparse->live_entry_count>sparse->entry_capacity))return reject("sparse_count_exceeds_admission");
    if(sparse && local!=sparse->live_entry_count)return reject("packed_sparse_count_mismatch");
    if(sparse && sparse->target_participant!=p.original_source->participant().participant_id)return reject("sparse_participant_mismatch");
    const auto gather_begin=timing_observation_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    if(timing_observation_)timing_observation_->sample_.source=source_phase;
    const size_t receive_capacity=slots*p.original_source->cohort()->participants().size();
    std::optional<CPURootedPublicationTransport> selected_transport;
    bool gathered_ok=false;
    if(sparse && source_phase==ExpertHistogramSource::PrefillChunk)
    {
        auto* native=dynamic_cast<GlobalTPContext*>(p.tp);
        if(!native || !native->authenticatesOriginalCPUExpertBorrow(p.original_tp_owner))
            return reject("original_replicated_prefill_context");
        CPURootedPublicationTransport selected;
        // Prepared bank, packet epoch, exact member and packed count were
        // authenticated above. Only this private native API may derive a common
        // live-size choice from the producer's replicated cohort route set.
        gathered_ok=native->gatherReplicatedCPUExpertPrefillRoutesToRoot(
            *p.original_source,*sparse,p.original_tp_owner,records,records,receive_capacity,
            p.root_participant,gathered,p.stage_name,p.source_ranges,selected);
        selected_transport=selected;
    }
    else
        gathered_ok=p.tp->gatherVariableFloatRecordsToRoot(records,local,records,receive_capacity,
            canonical_moe_route_record::recordWidth(p.columns),p.root_participant,gathered,
            p.stage_name,p.source_ranges);
    if(!gathered_ok)return reject("original_rooted_gather");
    if(timing_observation_)timing_observation_->sample_.gather_ns=rootTimingElapsed(gather_begin);
    const auto publish=[&]() {
        if(!PerfStatsCollector::isDomainEnabled("moe_canonical_publication")) return;
        PerfStatsCollector::Tags tags{{"stage",p.stage_name},{"root",std::to_string(p.root_participant)},
            {"participant",std::to_string(p.tp->myIndex())},{"degree",std::to_string(p.tp->degree())},
            {"local_records",std::to_string(local)},
            {"record_width",std::to_string(canonical_moe_route_record::recordWidth(p.columns))},
            {"expert_compute","tensor_sharded"}};
        if(selected_transport)tags.emplace("transport",CPURootedPublicationTransportPolicy::name(*selected_transport));
        PerfStatsCollector::addCounter("moe_canonical_publication","cpu_packed_route_gather_calls",1.0,
            phase,DeviceId::cpu().to_string(),tags);
        PerfStatsCollector::addCounter("moe_canonical_publication","cpu_packed_route_gather_local_bytes",
            static_cast<double>(local*canonical_moe_route_record::recordWidth(p.columns)*sizeof(float)),
            phase,DeviceId::cpu().to_string(),tags);
    };
    if(p.tp->myIndex()!=p.root_participant) {
        if(publication && !publication->publish(0u))return reject("contributor_empty_ticket_publication");
        if(p.original_canonical_ticket || p.return_publication==OriginalReturnPublication::CanonicalHostRows) {
            p.original_return_rows->live_row_count=0u;
            p.original_return_rows->layout=MoEOverlayReturnLayout::CanonicalExpertRoutes;
        }
        publish();return finish();
    }
    std::optional<cpu::moe::CanonicalShardSparseRouteView> sparse_view;
    if(sparse)sparse_view=cpu::moe::CanonicalShardSparseRouteView{
        {sparse->original_route_slots_host,sparse->live_entry_count},
        {sparse->expert_ids_host,sparse->live_entry_count},
        {sparse->route_weights_host,sparse->live_entry_count}};
    const auto combine_begin=timing_observation_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    const cpu::moe::CanonicalShardRouteRecordArgs combine_args{
        {records,gathered*canonical_moe_route_record::recordWidth(p.columns)},p.source_ranges,
        {p.route_indices?p.route_indices->data():nullptr,p.route_indices?slots:0},
        {p.route_weights?p.route_weights->data():nullptr,p.route_weights?slots:0},
        invocation_bank?std::span<const int>(invocation_bank->layerForModelLayer(p.original_source->layer()).cpu_fragment_expert_ids):p.original_source->expertIds(),
        p.participant_route_maps,p.required_route_weights,
        {p.combined_routes->mutable_data(),slots*static_cast<size_t>(p.columns)},
        p.original_source->ownership().geometry().experts,p.rows,p.top_k,p.columns,sparse_view};
    if(publication) {
        if(!combineAndPublishMoEOverlayCanonicalShardRouteRows(*p.original_canonical_ticket,
            *publication,*sparse,combine_args))return reject("canonical_fused_combine_ticket_publication");
        if(timing_observation_)timing_observation_->sample_.fused_combine_preweight_publication_ns=rootTimingElapsed(combine_begin);
        // The mapped ticket owns these values. This protocol return carries
        // only completion metadata, never a duplicate weighted route payload.
        p.original_return_rows->live_row_count=0u;
        p.original_return_rows->layout=MoEOverlayReturnLayout::CanonicalExpertRoutes;
    } else {
        if(!cpu::moe::combineCanonicalShardRouteRecords(combine_args))return reject("canonical_fixed_order_combine");
        if(timing_observation_)timing_observation_->sample_.combine_ns=rootTimingElapsed(combine_begin);
        if(p.return_publication==OriginalReturnPublication::CanonicalHostRows &&
            !copyMoEOverlayCanonicalRouteRowsToHostReturn(*sparse,
                {p.combined_routes->data(),slots*static_cast<size_t>(p.columns)},*p.original_return_rows))
            return reject("original_unweighted_host_return");
    }
    publish();
    return finish();
}
bool MoECPUExpertShardRouteStage::acceptsSchedulingContext(DeviceId scheduling_device) const noexcept {
    return scheduling_device.is_cpu() ||
        (hasOriginalCanonicalTicket() && scheduling_device==params_.original_continuation_device);
}
bool MoECPUExpertShardRouteStage::hasOriginalCanonicalTicket() const noexcept {
    const auto& ticket=params_.original_canonical_ticket;
    return ticket && ticket->hasValidBoundIdentity() && params_.original_source &&
        ticket->layerIndex()==params_.original_source->layer() && ticket->dModel()==params_.columns &&
        params_.original_continuation_device.is_gpu() && ticket->continuationDevice()==params_.original_continuation_device;
}
bool MoECPUExpertShardRouteStage::publishConcurrentManualFailure() noexcept {
    if(timing_observation_)timing_observation_->reset();
    LOG_ERROR("[MoECPUExpertShardRouteStage] Concurrent manual failure publication requested: stage="
        << params_.stage_name << " layer=" << params_.original_source->layer()
        << " participant=" << params_.original_source->participant().participant_id);
    if(!hasOriginalCanonicalTicket()) {
        LOG_ERROR("[MoECPUExpertShardRouteStage] Concurrent failure rejected: original_ticket_identity");
        return false;
    }
    const bool released=params_.original_canonical_ticket->publishAbort();
    if(!released)LOG_ERROR("[MoECPUExpertShardRouteStage] Concurrent failure rejected: original_ticket_abort_publication");
    if(released)params_.tp->requestAbort();
    return released;
}
std::optional<CPUExpertShardRouteCompletion> MoECPUExpertShardRouteStage::executeOriginalJointPublication(
    IDeviceContext *ctx,MoEOverlayActivationEpochProtocol &protocol,
    const MoEOverlayActivationEpochIdentity &identity,std::uint32_t stage) {
    if(timing_observation_)timing_observation_->reset();
    const auto &source=*params_.original_source;
    const auto expected = source.participantIndex() == static_cast<size_t>(params_.root_participant)
                              ? MoEOverlayActivationReturnRole::JointRoot
                              : MoEOverlayActivationReturnRole::JointContributor;
    const auto reject_original = [&](const char* reason) -> std::optional<CPUExpertShardRouteCompletion> {
        LOG_ERROR("[MoECPUExpertShardRouteStage] Original endpoint publication rejected: reason=" << reason
            << " stage=" << params_.stage_name << " ordinal=" << stage
            << " layer=" << source.layer() << " participant=" << source.participant().participant_id
            << " world_rank=" << source.participant().world_rank
            << " expected_role=" << static_cast<unsigned>(expected)
            << " received_role=" << identity.return_role);
        params_.tp->requestAbort();
        return std::nullopt;
    };
    const auto* native=dynamic_cast<const GlobalTPContext*>(params_.tp);
    const char* invalid = !native ? "original_native_context_type"
        : !params_.original_prepared_layer ? "original_prepared_layer_missing"
        : !params_.original_prepared_layer->sourceSelection() ? "original_prepared_source_missing"
        : params_.original_prepared_layer->sourceSelection()!=params_.original_source ? "original_source_lease"
        : !native->authenticatesOriginalCPUExpertBorrow(params_.original_tp_owner) ? "original_native_context_borrow"
        : !params_.original_tp_owner || params_.original_tp_owner.get()!=params_.tp ? "original_native_context_owner"
        : !protocol.matchesActiveIdentity(identity) ? "original_protocol_identity"
        : protocol.config_.return_role!=expected ? "original_protocol_return_role"
        : identity.return_role!=static_cast<std::uint32_t>(expected) ? "original_identity_return_role"
        : identity.target_participant_id!=source.participant().participant_id ? "original_identity_participant"
        : identity.target_world_rank!=source.participant().world_rank ? "original_identity_rank"
        : protocol.modelLayer(stage)!=source.layer() ? "original_protocol_layer"
        : protocol.endpointStatus(MoEOverlayActivationEndpoint::Follower).last_consumed_stage!=static_cast<int32_t>(stage)
            ? "original_consumed_stage" : nullptr;
    if(invalid)return reject_original(invalid);
    // The consumed packet's mathematical phase must be the same original
    // graph role already authenticated by the live endpoint protocol.
    const auto role=static_cast<MoEOverlayInferenceGraphRole>(identity.graph_role);
    const auto* original=params_.original_sparse_routes;
    const auto expected_phase=role==MoEOverlayInferenceGraphRole::MainPrefill
        ?ExpertHistogramSource::PrefillChunk:role==MoEOverlayInferenceGraphRole::MainDecode
        ?ExpertHistogramSource::DecodeToken:ExpertHistogramSource::GroupedVerifier;
    if(!original || role==MoEOverlayInferenceGraphRole::None ||
        (role!=MoEOverlayInferenceGraphRole::MainPrefill && role!=MoEOverlayInferenceGraphRole::MainDecode &&
         role!=MoEOverlayInferenceGraphRole::MTPDraft && role!=MoEOverlayInferenceGraphRole::MTPGroupedVerifier) ||
        original->key.histogram_source!=expected_phase)return reject_original("original_graph_phase");
    MoEOverlayParticipantBankLease completed_bank;
    if(!executeWithBank(ctx,&completed_bank))return std::nullopt; // Exact epoch retained through final return.
    if(!protocol.matchesActiveIdentity(identity)) {
        LOG_ERROR("[MoECPUExpertShardRouteStage] Original endpoint identity changed after execute: stage=" << params_.stage_name);
        return std::nullopt;
    }
    if(params_.original_sparse_routes) {
        const auto &input=*params_.original_sparse_routes;
        auto &output=*params_.original_return_rows;
        const auto &dispatch=protocol.control_->buffers[moeOverlayActivationBufferIndex(stage)].dispatch_descriptor;
        if(!completed_bank || input.residency_epoch!=completed_bank->epoch ||
           input.residency_epoch!=dispatch.placement_epoch || input.live_row_count!=dispatch.live_rows ||
           input.live_entry_count!=dispatch.live_entries || input.target_participant!=source.participant().participant_id ||
           output.residency_epoch!=input.residency_epoch || output.source_participant!=input.target_participant ||
           output.target_participant!=input.source_participant || input.live_row_count>output.row_capacity ||
           (input.live_row_count && (!input.row_ids_host || !output.row_ids_host))) {
            return reject_original("original_dispatch_return_tuple");
        }
        // One root publishes each combined route once. Contributors retire
        // metadata only; the continuation's original ordered fold still owns
        // top-k accumulation, after the fixed participant-order addition here.
        if (expected == MoEOverlayActivationReturnRole::JointRoot)
        {
            const auto publication_begin=timing_observation_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            const auto width=static_cast<size_t>(params_.columns);
            for(size_t entry=0;entry<input.live_entry_count;++entry) {
                const auto slot=static_cast<size_t>(input.original_route_slots_host[entry]);
                if(slot>=params_.original_return_route_capacity) {
                    return reject_original("original_return_slot_capacity");
                }
            }
            if(!preweightMoEOverlayCanonicalRouteRows(input,
                {params_.combined_routes->data(),params_.combined_routes->numel()},
                MoEOverlayCanonicalRouteSourceLayout::originalSlots(),
                {params_.original_preweighted_routes,params_.original_return_route_capacity*width},
                MoEOverlayCanonicalRouteSourceIndex::OriginalRouteSlot)) {
                return reject_original("original_return_preweight");
            }
            for(size_t row=0;row<input.live_row_count;++row)output.row_ids_host[row]=input.row_ids_host[row];
            output.live_row_count=input.live_row_count;
            if(timing_observation_)timing_observation_->sample_.preweight_publication_ns=rootTimingElapsed(publication_begin);
        }
        else
            output.live_row_count = 0;
    }
    if(timing_observation_)timing_observation_->completed_=true;
    return CPUExpertShardRouteCompletion(protocol.control_,identity,stage,params_.original_source,
        params_.original_tp_owner,params_.root_participant,params_.original_prepared_layer,std::move(completed_bank));
}
StageBufferRequirements MoECPUExpertShardRouteStage::getBufferRequirements() const {
    StageBufferRequirements r;
    r.addInout("packed_route_records",params_.packed_records->shape(),toBufferTensorType(TensorType::FP32));
    if(params_.route_indices)r.addInput("route_indices",params_.route_indices->shape(),toBufferTensorType(TensorType::FP32));
    if(params_.route_weights)r.addInput("route_weights",params_.route_weights->shape(),toBufferTensorType(TensorType::FP32));
    if(params_.combined_routes)
        r.addOutput("combined_routes",params_.combined_routes->shape(),toBufferTensorType(TensorType::FP32));
    return r;
}
StageDumpInfo MoECPUExpertShardRouteStage::buildDumpInfoImpl() const {
    StageDumpInfo info;
    info.addScalarInt("seq_len",params_.rows);info.addScalarInt("top_k",params_.top_k);
    info.addScalarInt("d_model",params_.columns);info.addScalarInt("root_participant",params_.root_participant);
    return info;
}
} // namespace llaminar2
