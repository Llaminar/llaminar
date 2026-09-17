/**
 * @file AutomaticOrchestrationCandidates.h
 * @brief Declarative candidate construction over the observed MPI inventory.
 *
 * Candidate construction does not rank devices, admit memory, load weights or
 * promise an executable graph. Each proposal must subsequently pass the shared
 * model-aware compiler and PhysicalMemoryAuthority. Keeping those steps separate
 * prevents enumeration order, VRAM size or vendor-specific core counts from
 * becoming an accidental performance policy.
 */
#pragma once

#include "config/OrchestrationConfig.h"
#include "planning/ExecutionRankMembership.h"
#include <functional>

namespace llaminar2
{
    class PlanningModelMetadata;

    /**
     * @brief One complete placement proposal in its selected rank namespace.
     *
     * The configuration retains the request's inference policy, but replaces
     * automatic intent with explicit placement. Membership maps its compact
     * rank IDs back to discovery. Neither field is an allocation certificate.
     */
    struct AutomaticOrchestrationCandidate
    {
        OrchestrationStrategy strategy;
        ExecutionRankMembership membership;
        OrchestrationConfig config;
    };

    /**
     * @brief Visit the deterministic, hardware-derived topology search space.
     * @param request Automatic request with no explicit placement.
     * @param model Exact main-layer boundary and model family.
     * @param inventory Canonical rank-indexed hardware observation.
     * @param visit Synchronous consumer; it may compile/price and discard each
     *        proposal so enumeration does not retain a Cartesian product.
     * @throws std::invalid_argument for malformed observation or non-auto intent.
     *
     * Search includes individual endpoints, every rank-local homogeneous GPU
     * subset, observed CPU pools per node and across nodes, two-domain pipelines
     * at every main-layer boundary, and GPU-continuation/CPU-expert overlays.
     * Overlay CPU pools exclude the continuation's physical node when remote
     * peers exist in that pool, permitting actual offload rather than including
     * an idle local CPU rank. GPU-only multi-tier candidates pair disjoint GPU
     * domains in both priority orders. This is an explicit search space, not a
     * claim to enumerate every possible partition of an arbitrary cluster.
     *
     * Hard filters apply before expansion. Hints affect later cost ranking,
     * never construction. No expert quota, hardware speed or byte reserve is
     * invented here. User-selectable execution support remains the compiler's
     * responsibility; rejection must be reported by its consumer.
     */
    void visitAutomaticOrchestrationCandidates(
        const OrchestrationConfig &request,
        const PlanningModelMetadata &model,
        const ClusterInventory &inventory,
        const std::function<void(AutomaticOrchestrationCandidate)> &visit);
}
