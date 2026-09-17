/**
 * @file MoEOverlayMemoryPlanInputs.h
 * @brief Shared production ExpertOverlay graph/host/transfer BOM input assembly.
 *
 * Candidate search and runner admission must price the same retained graph
 * families. This pure adapter translates existing immutable policies into the
 * local capacity planner's input. It neither selects a topology/shape nor owns
 * memory, free-byte arithmetic, MPI consensus or a second admission ledger.
 */
#pragma once

#include "config/OrchestrationConfig.h"
#include "execution/moe/MoEOverlayInferenceTransaction.h"
#include "execution/moe/MoEOverlayLocalCapacityPlanner.h"
#include <span>

namespace llaminar2
{
/** @brief Explicit capture-cache policy; callers freeze environment intent once. */
struct MoEOverlayPrefillMemoryPolicy
{
    std::span<const int> bucket_rows;
    int minimum_sequence_rows = 0;
    int maximum_cached_buckets = 0;
};

/**
 * @brief Immutable setup inputs for one already compiled rank topology.
 *
 * References outlive the returned local planner input. The retained MTP policy
 * may be wider than the current request, exactly as in production runner reuse.
 * Graph-family identity comes from the model manifest, never a guessed count.
 */
struct MoEOverlayMemoryPlanInputRequest
{
    const ModelMemoryProfile &model;
    const RankExecutionPlan &rank_plan;
    const OrchestrationConfig &config;
    const ClusterInventory &inventory;
    const MoEExpertOverlayExecutionPlan &execution;
    const MoEOverlayCapacityAdmissionPolicy &capacity_policy;
    const MTPRuntimeConfig &retained_mtp;
    const MoEOverlayInferenceGraphFamilyIdentity &graph_family;
    MoEOverlayPrefillMemoryPolicy prefill;
    MoEOverlayGPUWeightLoadCapacityInput gpu_weight_load;
    GraphSnapshotMemoryCapacity snapshot_capacity;
    std::size_t model_graph_topology_variant_count = 1;
};

/** @brief Complete local BOM input and the exact schedule/cache identity it prices. */
struct MoEOverlayMemoryPlanInputs
{
    MoEOverlayLocalCapacityPlannerInput local_capacity;
    int prefill_segment_rows = 0;
    std::size_t model_graph_identity_count = 0;
};

/**
 * @brief Assemble one candidate's complete production BOM without admitting bytes.
 * @param request Model, exact rank topology and retained execution policies.
 * @param resident_prefill_rows Positive candidate chosen by the caller's shape search.
 * @return Inputs consumed unchanged by MoEOverlayLocalCapacityPlanner/PMA.
 * @throws std::invalid_argument for missing/mismatched geometry or incomplete capture policy.
 * @throws std::overflow_error for row or explicit memory-limit overflow.
 *
 * Prefill scheduling, retained MTP rows and channel rows are distinct capacities.
 * Snapshot variants, helper executables and host histogram publication remain
 * named owners. The caller still owns collective readiness and quota selection.
 */
[[nodiscard]] MoEOverlayMemoryPlanInputs buildMoEOverlayMemoryPlanInputs(
    const MoEOverlayMemoryPlanInputRequest &request,
    int resident_prefill_rows);
} // namespace llaminar2
