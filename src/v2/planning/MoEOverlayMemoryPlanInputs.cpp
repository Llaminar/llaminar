/**
 * @file MoEOverlayMemoryPlanInputs.cpp
 * @brief Canonical ExpertOverlay retained-family and rank BOM projection.
 *
 * This is the production runner's assembly, shared with automatic planning.
 * Native executable ownership is not compilation-stage count. Request prefill
 * segments do not truncate retained MTP capacity, and CPU histogram storage
 * follows the same distributed publication contract used by live maintenance.
 * Frozen execution and parent boundaries authenticate one shared GPU family.
 */
#include "planning/PlanningObservedResource.h"
#include "planning/MoEOverlayMemoryPlanInputs.h"
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
        const PhysicalMemoryResource resource=withObservedBacking(
            {rank,device,total,PhysicalMemoryAuthority::admissionCapacity(available,config.memoryLimitBytes(device.type))},observed);
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

    if (request.pipeline_stage)
    {
        const auto &scope = *request.pipeline_stage;
        if (!scope.isValid() || request.model.mtp_layer_count < 0 || main_layers <= 0 ||
            scope.last_layer > main_layers ||
            scope.has_embedding != (scope.first_layer == 0) ||
            scope.has_lm_head != (scope.last_layer == main_layers) ||
            plan.usesLocalPP() || plan.usesGlobalTP() || plan.usesPipelineParallel() ||
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
    if (request.prefill.maximum_cached_buckets <= static_cast<int>(kExpertOverlayPrefillGraphIdentityReserve))
        throw std::invalid_argument("ExpertOverlay prefill graph cache cannot retain its runtime identities and one bucket");

    const int segment_rows = resolvePrefillScheduleRowCapacity(
        resident_prefill_rows, plan.runtime.moe_routed_prefill.overlay_segment_rows);
    const std::vector<int> configured_buckets(
        request.prefill.bucket_rows.begin(), request.prefill.bucket_rows.end());
    const auto retained_buckets = retainedRawPrefillGraphBucketLadder(
        configured_buckets, segment_rows, request.prefill.minimum_sequence_rows,
        static_cast<std::size_t>(request.prefill.maximum_cached_buckets) - kExpertOverlayPrefillGraphIdentityReserve);
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
    const auto &graph_owners = serving.mtp_graph_owners;
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
            .max_gpu_memory_bytes = config.memoryLimitBytes(DeviceType::CUDA),
            .max_cpu_memory_bytes = config.memoryLimitBytes(DeviceType::CPU),
            .resident_graph_rows = resolveRetainedGraphRowCapacity(resident_prefill_rows, mtp),
            .activation_channel_row_capacity = std::max(decode_rows, segment_rows),
            .activation_graph_family_count = family_count,
            .activation_prefill_graph_rows = retained_buckets,
            .original_native_slots = MoEOverlayActivationChannelPlanner::originalNativeSlotGeometry(
                *overlay, request.inventory, family),
            .host_demand_memory = std::move(host_demand),
            .generation_domain = generation.domain(),
            .generation_parent_role = parent_role,
            .captured_graph_plan = captured,
            .graph_snapshot_memory = request.snapshot_capacity,
            .gpu_weight_load = request.gpu_weight_load,
            .graph_family = family, .original_expert_manifest=request.original_expert_manifest,
            .weight_precision=request.weight_precision,.native_source_placement=request.native_source_placement},
        .prefill_segment_rows = segment_rows,
        .model_graph_identity_count = graph_identities};
}
} // namespace llaminar2
