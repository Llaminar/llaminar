/**
 * @file MoEOverlayPlanningInputs.h
 * @brief Canonical ExpertOverlay setup-policy inputs for plan/apply and runtime.
 *
 * The hardware-bound domain plan selects the movement implementation. Runtime
 * policy and model geometry determine its transfer/replica/workspace shape.
 * This adapter allocates nothing and keeps no byte ledger; its output is an
 * input to the existing PhysicalMemoryAuthority-backed capacity admission.
 */
#pragma once

#include "execution/moe/MoEOverlayCapacityAdmission.h"

namespace llaminar2
{
    struct OrchestrationConfig;

    /**
     * @brief Resolve the same migration-storage geometry for every setup caller.
     * @param plan Hardware-bound expert domains, priorities and replica policy.
     * @param config Exact serving policy, including user transfer-slot tuning.
     * @param world_size Size of the admitted execution communicator.
     * @param num_layers All routed layers retained by the main/MTP family.
     * @param num_experts Expert count from the authenticated model descriptor.
     * @return Complete typed storage/workspace policy, with no memory reserved.
     * @throws std::invalid_argument for invalid model or communicator geometry.
     * @throws std::logic_error for an inconsistent device-directory topology.
     *
     * Static requires no migration storage. A homogeneous, rank-local GPU tier
     * uses its device directory; arbitrary-tier/cross-rank movement uses the
     * physical residency fabric. Selection belongs to the existing admission
     * policy, never to a frontend's backend-name or tier-name heuristic.
     */
    [[nodiscard]] MoEOverlayCapacityAdmissionPolicy
    resolveMoEOverlayCapacityAdmissionPolicy(
        const MoERoutedExpertPlacementPlan &plan,
        const OrchestrationConfig &config,
        int world_size,
        int num_layers,
        int num_experts);
}
