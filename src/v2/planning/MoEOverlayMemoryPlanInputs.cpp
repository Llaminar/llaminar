/**
 * @file MoEOverlayMemoryPlanInputs.cpp
 * @brief Canonical ExpertOverlay retained-family and rank BOM projection.
 *
 * This is the production runner's assembly, shared with automatic planning.
 * Native executable ownership is not compilation-stage count. Request prefill
 * segments do not truncate retained MTP capacity, and CPU histogram storage
 * follows the same distributed publication contract used by live maintenance.
 * Frozen execution and parent boundaries authenticate one shared GPU family.
 * Restored-prefix MTP condition graphs retain their distinct bridge identity;
 * ordinary speculative/committed conditions cannot borrow that admission.
 */
#include "planning/MoEOverlayMemoryPlanInputs.h"
#include "execution/moe/MoEOverlayActivationBatchInventory.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "models/qwen/LayerGraphIdentity.h"
#include "config/BackendSelector.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "execution/mtp/MTPGraphOwnerPlan.h"
#include "execution/mtp/DeviceGenerationExecutionPolicy.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <set>
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEOverlayNoPeerResidency.h"

namespace llaminar2
{
std::shared_ptr<const MoEOverlayActivationBatchInventory> MoEOverlayActivationBatchInventory::declare(
    const MoEOverlayInferenceGraphFamilyIdentity& family, std::span<const int> prefill_rows,
    int segment_rows, int batch, const MTPRuntimeConfig& mtp,
    MTPVerifierGraphFamily verifier_family, std::size_t query_programs,
    PipelineForwardOnlyGraphOwner forward_only_owner)
{
    if (!family.valid() || segment_rows <= 0 || batch <= 0 ||
        family.max_request_count != batch || query_programs == 0 ||
        query_programs > kSparseQueryExecutionProgramCount || prefill_rows.empty() ||
        !std::is_sorted(prefill_rows.begin(), prefill_rows.end()) ||
        std::adjacent_find(prefill_rows.begin(), prefill_rows.end()) != prefill_rows.end() ||
        prefill_rows.front() <= 0 || prefill_rows.back() > segment_rows)
        throw std::invalid_argument("Batch source inventory requires original retained rows and model family");
    const auto flatten = [batch](int rows) {
        if (rows <= 0 || rows > std::numeric_limits<int>::max() / batch)
            throw std::overflow_error("Batch source flattened row extent overflows int");
        return rows * batch;
    };
    auto result = std::shared_ptr<MoEOverlayActivationBatchInventory>(new MoEOverlayActivationBatchInventory);
    std::vector<int> main_layers;
    for (int i = 0; i < family.main_layer_count; ++i)
        main_layers.push_back(family.first_model_layer + i);
    const auto add = [&](SourceKind kind, std::size_t family_ordinal,
                         SparseQueryExecutionProgram program, int rows, int maximum,
                         const std::vector<int>& layers) {
        result->sources_.push_back({kind, family_ordinal, program, rows, maximum, layers});
    };
    (void)OrdinaryGenerationGraphPlan::pipelineForwardOnlyExecutableCount(query_programs, forward_only_owner);
    const auto retained = retainsMTPGraphCapacity(mtp);
    const auto width_count = resolveMTPRetainedVerifierWidthCount(mtp, verifier_family);
    for (std::size_t program_index = 0; program_index < query_programs; ++program_index)
    {
        const auto program = static_cast<SparseQueryExecutionProgram>(program_index);
        for (const int rows : prefill_rows)
            add(SourceKind::PrefillBucket, 0, program, flatten(rows), flatten(rows), main_layers);
        // These are the same two named replaceable identities reserved by the
        // canonical prefill cache. Their envelope is allocation capacity only.
        static_assert(kExpertOverlayPrefillGraphIdentityReserve == 2);
        add(SourceKind::RequestPrefill, 0, program, 0, flatten(segment_rows), main_layers);
        add(SourceKind::PrefixRehydration, 0, program, 0, flatten(segment_rows), main_layers);
        add(SourceKind::Decode, 0, program, batch, batch, main_layers);
        if (forward_only_owner == PipelineForwardOnlyGraphOwner::Participant)
            add(SourceKind::ForwardOnly, 0, program, 1, 1, main_layers);
        if (!retained) continue;
        add(SourceKind::RestoredPrefixBridge, 0, program, batch, batch, main_layers);
        for (const auto purpose : kMTPConditionForwardPurposes)
            add(purpose == MTPConditionForwardPurpose::CommittedSerialToken
                    ? SourceKind::CommittedCondition : SourceKind::SpeculativeCondition,
                0, program, batch, batch, main_layers);
        const auto depth = resolveMTPRetainedDraftCapacity(mtp);
        for (std::size_t index = 0; index < width_count; ++index)
        {
            const auto rows = flatten(mtpVerifierGraphFamilyWidthAt(index, depth + 1, verifier_family));
            add(SourceKind::VerifierGreedy, 0, program, rows, rows, main_layers);
            add(SourceKind::VerifierStochastic, 0, program, rows, rows, main_layers);
        }
    }
    // The canonical continuation cache owns Full/Chained host/device sources
    // at depth zero. Other learned manifests may own follower programs, but
    // do not imply extra continuation cache owners. KVOnly exits before FFN.
    const MTPSidecarCaptureLayout sidecar_layout(batch);
    if (retained && !family.mtp_source_layers.empty())
        for (const auto role : {MTPSidecarCaptureRole::Full, MTPSidecarCaptureRole::Chained})
            for (const auto tokens : {DraftTokenSource::Host, DraftTokenSource::Device})
            {
                (void)sidecar_layout.conditionTokenSlot(role, batch);
                add(SourceKind::Draft, 1, SparseQueryExecutionProgram::RankingPossible,
                    batch, batch, {family.mtp_source_layers.front()});
                result->sources_.back().sidecar_role = role;
                result->sources_.back().draft_token_source = tokens;
            }
    return declareSources(result->sources_);
}

std::shared_ptr<const MoEOverlayActivationBatchInventory> MoEOverlayActivationBatchInventory::declareSources(
    std::span<const Source> sources)
{
    if(sources.empty())throw std::invalid_argument("Original Batch source roster cannot be empty");
    for(std::size_t index=0;index<sources.size();++index) {
        const auto& source=sources[index];
        const auto kind=static_cast<int>(source.kind);
        const bool variable=source.kind==SourceKind::RequestPrefill || source.kind==SourceKind::PrefixRehydration;
        const bool draft=source.kind==SourceKind::Draft;
        if(kind<static_cast<int>(SourceKind::PrefillBucket) || kind>static_cast<int>(SourceKind::Draft) ||
            sparseQueryExecutionProgramIndex(source.query_program)>=kSparseQueryExecutionProgramCount ||
            source.maximum_rows<=0 || source.physical_rows<0 || source.physical_rows>source.maximum_rows ||
            (variable!=(source.physical_rows==0)) || (!variable && source.maximum_rows!=source.physical_rows) ||
            (draft && source.query_program!=SparseQueryExecutionProgram::RankingPossible) ||
            source.model_layers.empty() || source.model_layers.front()<0 ||
            !std::is_sorted(source.model_layers.begin(),source.model_layers.end()) ||
            std::adjacent_find(source.model_layers.begin(),source.model_layers.end())!=source.model_layers.end() ||
            draft!=(source.family_ordinal!=0) || draft!=source.sidecar_role.has_value() ||
            draft!=source.draft_token_source.has_value() ||
            (draft && (*source.sidecar_role!=MTPSidecarCaptureRole::Full && *source.sidecar_role!=MTPSidecarCaptureRole::Chained)) ||
            (draft && *source.draft_token_source!=DraftTokenSource::Host && *source.draft_token_source!=DraftTokenSource::Device))
            throw std::invalid_argument("Malformed original Batch source, role, rows or layer identity");
        for(std::size_t prior=0;prior<index;++prior) {
            const auto& other=sources[prior];
            if(source.kind==other.kind && source.family_ordinal==other.family_ordinal &&
                source.query_program==other.query_program && source.physical_rows==other.physical_rows &&
                source.sidecar_role==other.sidecar_role && source.draft_token_source==other.draft_token_source)
                throw std::invalid_argument("Original Batch source owner is duplicated");
        }
    }
    auto result=std::shared_ptr<MoEOverlayActivationBatchInventory>(new MoEOverlayActivationBatchInventory);
    result->sources_.assign(sources.begin(),sources.end());
    return result;
}

std::size_t MoEOverlayActivationBatchInventory::mainGraphIdentityCount() const noexcept
{
    return std::count_if(sources_.begin(), sources_.end(),
        [](const Source& source) { return source.family_ordinal == 0 && source.kind != SourceKind::ForwardOnly; });
}

void MoEOverlayActivationBatchInventory::require(const ForwardInput& input,
    const LayerGraphSourceIdentity& layer, int physical_rows, std::uint32_t stage_ordinal) const
{
    if ((input.graph_submission_intent != ForwardGraphSubmissionIntent::WorkspaceDeclaration &&
         !executableForwardSubmissionIntent(input.graph_submission_intent)) ||
        (input.execution_phase != ForwardExecutionPhase::Prefill &&
         input.execution_phase != ForwardExecutionPhase::Decode))
        throw std::invalid_argument("Batch source has an unknown construction intent or execution phase");
    if ((layer.workload() == GraphLayerWorkload::MTPDepth ||
         input.execution_role == ForwardExecutionRole::GroupedMTPVerifier ||
         input.execution_role == ForwardExecutionRole::MTPCondition) &&
        input.execution_phase != ForwardExecutionPhase::Decode)
        throw std::invalid_argument("Batch learned/condition/verifier source requires its original decode phase");
    if (physical_rows <= 0 || input.batch_size <= 0 || input.seq_len <= 0 ||
        input.batch_size > std::numeric_limits<int>::max() / input.seq_len ||
        input.batch_size * input.seq_len != physical_rows)
        throw std::invalid_argument("Batch source changed its original invocation rows");
    const auto admits = [&](const Source& source) {
        if (stage_ordinal >= source.model_layers.size() ||
            source.model_layers[stage_ordinal] != layer.sourceLayer() ||
            physical_rows > source.maximum_rows ||
            (input.graph_submission_intent != ForwardGraphSubmissionIntent::WorkspaceDeclaration &&
             source.physical_rows != 0 && source.physical_rows != physical_rows))
            return false;
        if (layer.workload() == GraphLayerWorkload::MTPDepth)
        {
            const bool device_tokens = input.token_ids_device != nullptr;
            if (device_tokens == (input.token_ids != nullptr)) return false;
            return source.kind == SourceKind::Draft && layer.predictorDepth() &&
                source.family_ordinal == static_cast<std::size_t>(*layer.predictorDepth()) + 1 &&
                source.sidecar_role == input.mtp_sidecar_capture_role &&
                source.draft_token_source == (device_tokens ? DraftTokenSource::Device : DraftTokenSource::Host);
        }
        if (input.mtp_sidecar_capture_role) return false;
        if (source.family_ordinal != 0 || source.query_program != input.sparse_query_execution_program)
            return false;
        if (input.execution_role == ForwardExecutionRole::GroupedMTPVerifier)
            return input.state_transaction == ForwardStateTransaction::Ordinary &&
                (source.kind == SourceKind::VerifierGreedy || source.kind == SourceKind::VerifierStochastic);
        if (input.execution_role == ForwardExecutionRole::MTPCondition)
        {
            // Native serving materialization and runtime execute the bridge as
            // an MTP condition so shifted depth-zero KV and terminal hidden
            // state publish together. Preserve its separately priced identity.
            if (input.state_transaction == ForwardStateTransaction::RestoredPrefixMTPDecodeBridge)
                return source.kind == SourceKind::RestoredPrefixBridge;
            return input.state_transaction == ForwardStateTransaction::CommittedMTPCondition
                ? source.kind == SourceKind::CommittedCondition
                : input.state_transaction == ForwardStateTransaction::Ordinary &&
                    source.kind == SourceKind::SpeculativeCondition;
        }
        if (input.execution_role != ForwardExecutionRole::MainInference) return false;
        if (input.execution_phase == ForwardExecutionPhase::Prefill)
            return input.state_transaction == ForwardStateTransaction::Ordinary &&
                (input.rehydrate_prefix_runtime_on_device
                    ? source.kind == SourceKind::PrefixRehydration
                    : source.kind == SourceKind::PrefillBucket || source.kind == SourceKind::RequestPrefill);
        if (input.state_transaction == ForwardStateTransaction::RestoredPrefixMTPDecodeBridge)
            return source.kind == SourceKind::RestoredPrefixBridge;
        if (input.state_transaction == ForwardStateTransaction::ForwardOnlyCondition)
            return source.kind == SourceKind::ForwardOnly;
        return input.state_transaction == ForwardStateTransaction::Ordinary && source.kind == SourceKind::Decode;
    };
    if (std::none_of(sources_.begin(), sources_.end(), admits))
        throw std::invalid_argument("Batch graph source, exact rows or original layer ordinal was not admitted");
}


std::vector<PhysicalMemoryResource> originalMoEOverlayNativePreparationResources(
    const MoEExpertOverlayExecutionPlan& execution,const ClusterInventory& inventory,
    const OrchestrationConfig& config,const MoEOverlayCapacityAdmissionPolicy& policy,
    const ImmutableRowTablePolicy& ngram)
{
    const auto* overlay=config.moe_routed_expert_plan.get();
    if(!overlay || !overlay->usesExpertOverlayAuthority() || execution.world_size<=0 ||
       inventory.world_size!=execution.world_size || inventory.ranks.size()!=size_t(execution.world_size) ||
       execution.rank_plans.size()!=size_t(execution.world_size))
        throw std::invalid_argument("Native overlay registry requires original complete execution/inventory/policy");
    const bool host_owner=policy.usesPhysicalResidencyFabric() || MoEExpertOwnerMap::usesCPUJointFragments(*overlay) ||
        (overlay->authority_execution==MoEOverlayAuthorityExecutionKind::HostResident &&
         config.moe_rebalance.mode==MoERebalanceRuntimeMode::Dynamic && !MoEOverlayNoPeerResidency::hasNoPeerTopology(*overlay));
    std::set<std::pair<int,DeviceId>> selected;
    std::set<int> ranks;
    for(const auto& rank:execution.rank_plans){
        if(rank.world_rank<0 || rank.world_rank>=execution.world_size || !ranks.insert(rank.world_rank).second)
            throw std::invalid_argument("Native overlay registry has foreign or duplicate rank ownership");
        if(host_owner || !rank.local_devices.empty())selected.emplace(rank.world_rank,DeviceId::cpu());
        for(const auto device:rank.local_devices){
            if(!device.is_valid() || (!device.is_cpu() && !device.is_gpu()))
                throw std::invalid_argument("Native overlay registry has an invalid declared device");
            selected.emplace(rank.world_rank,device);
        }
    }
    if(ngram.isGPUAdaptive())for(const auto& participant:ngram.gpu_adaptive.placement.declaredParticipants()){
        if(participant.world_rank<0 || participant.world_rank>=execution.world_size || !participant.device().is_gpu())
            throw std::invalid_argument("Native overlay storage registry has a foreign declared participant");
        selected.emplace(participant.world_rank,DeviceId::cpu());selected.emplace(participant.world_rank,participant.device());
    }
    std::vector<PhysicalMemoryResource> result;result.reserve(selected.size());
    for(const auto& [rank,device]:selected){
        const auto& observed=inventory.getRank(rank);
        if(observed.rank!=rank)throw std::invalid_argument("Native overlay registry changed original rank observations");
        std::uint64_t total=0,available=0;
        if(device.is_cpu()){
            total=observed.cpu.memory_bytes?observed.cpu.memory_bytes:observed.cpu_memory_bytes;
            available=observed.cpu.free_memory_bytes;
        }else{
            const auto gpu=std::find_if(observed.gpus.begin(),observed.gpus.end(),[&](const auto& g){return g.type==device.type && g.local_device_id==device.ordinal;});
            if(gpu==observed.gpus.end())throw std::invalid_argument("Native overlay registry device is absent from original rank inventory");
            total=gpu->memory_bytes;available=gpu->free_memory_bytes;
        }
        PhysicalMemoryResource resource{rank,device,total,PhysicalMemoryAuthority::admissionCapacity(available,config.memoryLimitBytes(device.type))};
        if(!resource.valid())throw std::invalid_argument("Native overlay registry has incomplete original capacity");
        result.push_back(resource);
    }
    return result;
}

MoEOverlayMemoryPlanInputs buildMoEOverlayMemoryPlanInputs(
    const MoEOverlayMemoryPlanInputRequest &request, int resident_prefill_rows)
{
    const auto &plan = request.rank_plan;
    const auto &config = request.config;
    const auto &mtp = request.retained_mtp;
    const auto &family = request.graph_family;
    // GGUF block_count includes inline predictors for ordinary exports, while
    // Flash Next declares its main block_count and separate original sidecar.
    // The profile owns that distinction; routed ownership must retain both.
    const int main_layers = request.model.mainLayerCount();
    const int total_layers = request.model.totalLayerCount();
    const int world_size = request.inventory.world_size;
    const auto *overlay = config.moe_routed_expert_plan.get();
    if (!overlay || !overlay->usesExpertOverlayAuthority() || resident_prefill_rows <= 0 ||
        world_size <= 0 || request.inventory.ranks.size() != static_cast<std::size_t>(world_size) ||
        plan.rank < 0 || plan.rank >= world_size ||
        request.inventory.ranks[plan.rank].rank != plan.rank ||
        request.execution.world_size != world_size ||
        request.execution.currentRankPlan().world_rank != plan.rank ||
        request.capacity_policy.overlay_world_size != world_size ||
        !family.valid() || family.first_model_layer != overlay->first_model_layer ||
        family.first_model_layer >= main_layers ||
        family.routedLayerCapacity() > total_layers - family.first_model_layer ||
        request.model.expert_count <= 0 || plan.runtime.batch_size <= 0 ||
        request.model_graph_topology_variant_count == 0 ||
        (request.model_graph_topology_variant_count > 1 && !request.snapshot_capacity.valid()))
        throw std::invalid_argument("ExpertOverlay memory inputs require complete, matching rank/model/capture ownership");

    bool original_cpu_stage=false;
    if(request.original_cpu_stage_program) {
        const auto& original=*request.original_cpu_stage_program;
        const auto& program=original.nativeProgram();
        const auto& members=program.participants();
        const auto index=plan.global_tp_rank_in_domain;
        if(!request.pipeline_stage || !plan.usesGlobalTP() || plan.usesLocalTP() ||
            !plan.local_tp_devices.empty() || !plan.local_tp_weights.empty() ||
            plan.tp_scope!=TPScope::NODE_LOCAL || index<0 || static_cast<size_t>(index)>=members.size() ||
            plan.global_tp_domain_size!=static_cast<int>(members.size()) ||
            members[index].world_rank!=plan.rank || members[index].address!=plan.primary_device ||
            plan.weight_shard.shard_index!=index || plan.weight_shard.total_shards!=plan.global_tp_domain_size ||
            plan.weight_shard.work_fraction!=1.0f/static_cast<float>(plan.global_tp_domain_size) ||
            !plan.weight_shard.validate().empty() || plan.my_domains.size()!=1 || overlay->domains.size()!=1)
            throw std::invalid_argument("CPU pipeline memory requires its original global rank/member/shard program");
        const auto& scope=*request.pipeline_stage;
        const auto& domain=overlay->domains.front();
        const auto& participation=plan.my_domains.front();
        if(original.scope().first_layer!=scope.first_layer || original.scope().last_layer!=scope.last_layer ||
            original.scope().has_embedding!=scope.has_embedding || original.scope().has_lm_head!=scope.has_lm_head ||
            original.scope().generation_domain!=scope.generation_domain ||
            domain.name!=overlay->effectiveBaseModelDomain() || domain.scope!=ExecutionDomainScope::NODE_LOCAL || domain.owner_rank ||
            overlay->continuation_domain_spec.effectiveDensePolicy()!=DenseParallelPolicy::TensorParallel ||
            domain.participants.size()!=members.size() || domain.world_ranks.size()!=members.size() ||
            participation.domain_name!=domain.name || plan.global_tp_domain_id!=std::optional{participation.domain_id} ||
            participation.devices!=domain.participants || participation.weights!=domain.weights ||
            participation.backend!=domain.backend || participation.my_index_in_domain!=index ||
            program.backend()!=BackendSelector::resolve(domain.backend,std::vector<DeviceId>(members.size(),DeviceId::cpu())))
            throw std::invalid_argument("CPU pipeline memory changed its original stage/domain/collective scope");
        for(size_t member=0;member<members.size();++member)
            if(members[member].world_rank!=domain.world_ranks[member] ||
                members[member].address!=domain.participants[member])
                throw std::invalid_argument("CPU pipeline memory changed original ordered rank/NUMA ownership");
        original_cpu_stage=true;
    }
    if (request.pipeline_stage)
    {
        const auto &scope = *request.pipeline_stage;
        if (!scope.isValid() || request.model.mtp_layer_count < 0 || main_layers <= 0 ||
            scope.last_layer > main_layers ||
            scope.has_embedding != (scope.first_layer == 0) ||
            scope.has_lm_head != (scope.last_layer == main_layers) ||
            plan.usesLocalPP() || (plan.usesGlobalTP() && !original_cpu_stage) || plan.usesPipelineParallel() ||
            plan.first_layer != scope.first_layer || plan.last_layer != scope.last_layer - 1 ||
            plan.has_embedding != scope.has_embedding || plan.has_lm_head != scope.has_lm_head ||
            family.first_model_layer != scope.first_layer || family.main_layer_count != scope.layerCount() ||
            (!scope.has_lm_head && !family.mtp_source_layers.empty()))
            throw std::invalid_argument("ExpertOverlay pipeline memory scope differs from its child rank or retained graph family");
    }
    else if (family.first_model_layer != 0 ||
             family.main_layer_count != main_layers)
        throw std::invalid_argument("ExpertOverlay partial memory projection requires its authored pipeline scope");

    if (request.capacity_policy.usesDeviceTransferDirectory() &&
        (!request.capacity_policy.device_rebalance_workspace_capacity ||
         request.capacity_policy.device_rebalance_workspace_capacity->num_layers !=
             static_cast<std::uint32_t>(family.routedLayerCapacity()) ||
         request.capacity_policy.device_rebalance_workspace_capacity->num_experts !=
             static_cast<std::uint32_t>(request.model.expert_count)))
        throw std::invalid_argument("ExpertOverlay device capacity policy differs from its retained routed graph interval");

    const int decode_rows = retainsMTPGraphCapacity(mtp)
        ? std::max(1, resolveMTPRetainedTargetQueryRows(mtp)) : 1;
    if (family.max_decode_rows != decode_rows ||
        family.max_request_count != plan.runtime.batch_size ||
        family.max_mtp_draft_depth != resolveMTPRetainedDraftCapacity(mtp))
        throw std::invalid_argument("ExpertOverlay memory graph family differs from retained MTP/request capacity");
    if (request.prefill.bucket_rows.empty())
        throw std::invalid_argument("ExpertOverlay memory inputs require an explicit prefill graph bucket inventory");
    if (!request.prefill.exact_fixed_pipeline_widths &&
        request.prefill.maximum_cached_buckets <= static_cast<int>(kExpertOverlayPrefillGraphIdentityReserve))
        throw std::invalid_argument("ExpertOverlay prefill graph cache cannot retain its runtime identities and one bucket");

    const int segment_rows = resolvePrefillScheduleRowCapacity(
        resident_prefill_rows, plan.runtime.moe_routed_prefill.overlay_segment_rows);
    const std::vector<int> configured_buckets(
        request.prefill.bucket_rows.begin(), request.prefill.bucket_rows.end());
    const auto retained_buckets = request.prefill.exact_fixed_pipeline_widths
        ? exactPrefillGraphWidths(segment_rows)
        : retainedRawPrefillGraphBucketLadder(configured_buckets, segment_rows, request.prefill.minimum_sequence_rows,
            static_cast<std::size_t>(request.prefill.maximum_cached_buckets) - kExpertOverlayPrefillGraphIdentityReserve);
    if (request.prefill.exact_fixed_pipeline_widths)
        (void)exactPrefillGraphCacheEntries(segment_rows, request.prefill.maximum_cached_buckets_explicit
            ? std::optional{request.prefill.maximum_cached_buckets} : std::nullopt);
    if (retained_buckets.empty())
        throw std::invalid_argument("ExpertOverlay capacity candidate has no cache-resident prefill graph ladder");

    const auto family_count = makeMoEOverlayActivationGraphFamilyManifests(family).size();
    DeviceGenerationDomainDeclaration generation;
    if (request.execution.generationExecutionDomain() == DeviceGenerationExecutionDomain::HostedBoundary)
        generation.requireHostedBoundary();
    if (request.pipeline_stage)
    {
        switch (request.pipeline_stage->generation_domain)
        {
        case DeviceGenerationExecutionDomain::HostedBoundary:
            generation.requireHostedBoundary();
            break;
        case DeviceGenerationExecutionDomain::NativeDomain:
            break;
        default:
            throw std::invalid_argument("ExpertOverlay pipeline memory has an unknown generation domain");
        }
    }
    // One rank/stage inventory is shared by its GPUs. Authenticate that the
    // complete frozen execution declaration gives every GPU the same family;
    // authority residency alone cannot identify mixed or remote boundaries.
    auto verifier_family = resolveMTPVerifierGraphFamily(MTPVerifierExecutionRole::Standalone);
    bool has_gpu_family = false;
    for (const auto &domain : request.execution.domains)
        for (const auto &participant : domain.participants)
        {
            const auto device = participant.address.toLocalDeviceId();
            if (!device.is_gpu()) continue;
            const auto family_for_device = resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole(
                declaredDeviceGenerationExecutionPolicy(device.type, generation.domain())));
            if (has_gpu_family && verifier_family != family_for_device)
                throw std::invalid_argument("ExpertOverlay shared captured graph inventory has nonuniform GPU verifier families");
            verifier_family = family_for_device;
            has_gpu_family = true;
        }
    // CPU-only plans have no native executable consumer; their diagnostic
    // inventory keeps the standalone family without advertising GPU support.
    const auto parent_role = request.pipeline_stage && !request.pipeline_stage->has_lm_head
        ? MTPGenerationParentRole::PipelineFollower : MTPGenerationParentRole::Controller;
    const auto query_programs = request.model.sparseQueryProgramCount();
    const auto serving = resolveCapturedServingGraphMemoryInventory({}, mtp, verifier_family, parent_role, generation.domain(), query_programs);
    // Each admitted bucket and original named runtime identity retains both
    // actual main-query programs. Fixed decode/condition/verifier owners are
    // already multiplied by this same closed model inventory.
    // Rehydration keeps its existing initial-materialization lifecycle; pricing
    // its owner does not advertise it as a cold serving bucket.
    const auto prefill_identities = retained_buckets.size();
    if (query_programs == 0u || query_programs > kSparseQueryExecutionProgramCount ||
        prefill_identities > std::numeric_limits<std::size_t>::max() - kExpertOverlayPrefillGraphIdentityReserve ||
        prefill_identities + kExpertOverlayPrefillGraphIdentityReserve >
            (std::numeric_limits<std::size_t>::max() - serving.fixed_executable_count) / query_programs)
        throw std::overflow_error("Overlay retained prefill program quantity overflows");
    const auto graph_identities = (prefill_identities + kExpertOverlayPrefillGraphIdentityReserve) * query_programs +
        serving.fixed_executable_count;
    const auto batch_inventory = MoEOverlayActivationBatchInventory::declare(
        family, retained_buckets, segment_rows, plan.runtime.batch_size, mtp,
        verifier_family, query_programs,
        request.pipeline_stage ? PipelineForwardOnlyGraphOwner::Participant : PipelineForwardOnlyGraphOwner::None);
    if (batch_inventory->mainGraphIdentityCount() != graph_identities)
        throw std::logic_error("Batch source declaration differs from canonical retained model graph owners");
    auto graph_owners = serving.mtp_graph_owners;
    if (request.native_pipeline_maintenance_budget && graph_owners.ownsControllerExecutables())
        graph_owners = graph_owners.withVerifierPreparationTopology(MTPGraphOwnerPlan::VerifierPreparationTopology::NativePipelineMaintenance);
    const auto captured = resolveMoEOverlayCapturedGraphPlan(
        family.main_layer_count, overlay->authority_execution, graph_identities,
        request.model_graph_topology_variant_count,
        family_count + graph_owners.generalAuxiliaryExecutableSlotCount(),
        graph_owners.boundedHelperExecutableSlotCount());

    std::optional<MoEOverlayHostDemandMemoryPlan> host_demand;
    if (overlay->authority_execution == MoEOverlayAuthorityExecutionKind::HostResident &&
        config.moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic &&
        request.capacity_policy.migration_storage != MoEOverlayMigrationStorageKind::NoPeer)
    {
        // Request-major prefill and flattened verifier rows share one admitted
        // histogram ingress. Charge the maximum; never multiply verifier twice.
        const auto prefill_rows = static_cast<std::uint64_t>(segment_rows) * plan.runtime.batch_size;
        if (prefill_rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("ExpertOverlay demand batch exceeds the routing row ABI");
        host_demand.emplace(MoEOverlayHostDemandGeometry{
            .num_layers = family.routedLayerCapacity(),
            .num_experts = request.model.expert_count,
            .top_k = request.model.expert_used_count,
            .initial_window_rows = config.moe_rebalance.window_size,
            .maximum_window_rows = config.moe_rebalance.max_window_size,
            .maximum_invocation_rows = std::max(static_cast<int>(prefill_rows), decode_rows),
            .publication = world_size > 1 ? MoEOverlayDemandPublicationScope::Distributed
                                         : MoEOverlayDemandPublicationScope::ProcessLocal});
    }

    return {
        .local_capacity = {
            .model_profile = &request.model,
            .rank_plan = &plan,
            .overlay_plan = overlay,
            .rank_inventory = &request.inventory.ranks[plan.rank],
            .cluster_inventory = &request.inventory,
            .rank_execution_kind = request.execution.currentRankPlan().execution_kind,
            .require_host_memory_authority = request.capacity_policy.usesPhysicalResidencyFabric(),
            .native_pipeline_maintenance_budget = request.native_pipeline_maintenance_budget,
            .max_gpu_memory_bytes = config.memoryLimitBytes(DeviceType::CUDA),
            .max_cpu_memory_bytes = config.memoryLimitBytes(DeviceType::CPU),
            .resident_graph_rows = resolveRetainedGraphRowCapacity(resident_prefill_rows, mtp),
            .activation_channel_row_capacity = std::max(decode_rows, segment_rows),
            .activation_graph_family_count = family_count,
            .activation_prefill_graph_rows = retained_buckets,
            .activation_batch_inventory = batch_inventory,
            .original_native_slots = MoEOverlayActivationChannelPlanner::originalNativeSlotGeometry(
                *overlay, request.inventory, family),
            .host_demand_memory = std::move(host_demand),
            .generation_domain = generation.domain(),
            .generation_parent_role = parent_role,
            .captured_graph_plan = captured,
            .graph_snapshot_memory = request.snapshot_capacity,
            .gpu_weight_load = request.gpu_weight_load,
            .graph_family = family, .original_expert_manifest=request.original_expert_manifest,
            .weight_precision=request.weight_precision,.native_source_placement=request.native_source_placement,
            .original_cpu_stage_program=request.original_cpu_stage_program},
        .prefill_segment_rows = segment_rows,
        .model_graph_identity_count = graph_identities};
}
} // namespace llaminar2
