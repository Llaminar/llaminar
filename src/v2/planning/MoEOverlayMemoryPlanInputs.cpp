/**
 * @file MoEOverlayMemoryPlanInputs.cpp
 * @brief Canonical ExpertOverlay retained-family and rank BOM projection.
 *
 * This is the production runner's assembly, shared with automatic planning.
 * Native executable ownership is not compilation-stage count. Request prefill
 * segments do not truncate retained MTP capacity, and CPU histogram storage
 * follows the same distributed publication contract used by live maintenance.
 */
#include "planning/MoEOverlayMemoryPlanInputs.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "execution/mtp/MTPGraphOwnerPlan.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
MoEOverlayMemoryPlanInputs buildMoEOverlayMemoryPlanInputs(
    const MoEOverlayMemoryPlanInputRequest &request, int resident_prefill_rows)
{
    const auto &plan = request.rank_plan;
    const auto &config = request.config;
    const auto &mtp = request.retained_mtp;
    const auto &family = request.graph_family;
    const int world_size = request.inventory.world_size;
    const auto *overlay = config.moe_routed_expert_plan.get();
    if (!overlay || !overlay->usesExpertOverlayAuthority() || resident_prefill_rows <= 0 ||
        world_size <= 0 || request.inventory.ranks.size() != static_cast<std::size_t>(world_size) ||
        plan.rank < 0 || plan.rank >= world_size ||
        request.inventory.ranks[plan.rank].rank != plan.rank ||
        request.execution.world_size != world_size ||
        request.execution.currentRankPlan().world_rank != plan.rank ||
        request.capacity_policy.overlay_world_size != world_size ||
        !family.valid() || family.routedLayerCapacity() > request.model.n_layers ||
        request.model.expert_count <= 0 || plan.runtime.batch_size <= 0 ||
        request.model_graph_topology_variant_count == 0 ||
        (request.model_graph_topology_variant_count > 1 && !request.snapshot_capacity.valid()))
        throw std::invalid_argument("ExpertOverlay memory inputs require complete, matching rank/model/capture ownership");

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
    const auto graph_identities = retained_buckets.size() + kExpertOverlayPrefillGraphIdentityReserve +
        resolveMTPRetainedServingForwardModelGraphIdentityCount(mtp);
    const MTPGraphOwnerPlan graph_owners(mtp);
    const auto captured = resolveMoEOverlayCapturedGraphPlan(
        request.model.n_layers, overlay->authority_execution, graph_identities,
        request.model_graph_topology_variant_count,
        family_count + graph_owners.generalAuxiliaryExecutableSlotCount(),
        graph_owners.boundedHelperExecutableSlotCount());

    std::optional<MoEOverlayHostDemandMemoryPlan> host_demand;
    if (overlay->authority_execution == MoEOverlayAuthorityExecutionKind::HostResident &&
        config.moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic)
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
            .host_demand_memory = std::move(host_demand),
            .captured_graph_plan = captured,
            .graph_snapshot_memory = request.snapshot_capacity,
            .gpu_weight_load = request.gpu_weight_load},
        .prefill_segment_rows = segment_rows,
        .model_graph_identity_count = graph_identities};
}
} // namespace llaminar2
