/**
 * @file MoEExpertOverlayExecutionPlan.h
 * @brief Rank-local role contract for MoE expert overlay orchestration.
 *
 * The runtime plan resolves domain/device descriptors. This execution plan
 * answers the orchestration question for one MPI rank: whether it builds the
 * continuation graph or serves auxiliary expert-overlay domains. Non-root role
 * planning intentionally resolves descriptors without claiming the rank can
 * construct the root DeviceGraphExecutor.
 */

#pragma once

#include "MoEExpertOverlayRuntimePlan.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    enum class OverlayRankRole
    {
        ContinuationRoot,
        LocalAcceleratorParticipant,
        CpuFallbackParticipant,
        RemoteExpertParticipant,
        RelayOnly,
    };

    const char *toString(OverlayRankRole role);

    struct OverlayRankPlan
    {
        int world_rank = -1;
        OverlayRankRole role = OverlayRankRole::RelayOnly;
        std::vector<OverlayRankRole> roles;
        std::vector<std::string> owned_domains;
        std::vector<DeviceId> local_devices;
        bool builds_root_graph = false;
        bool loads_tokenizer = false;
        bool loads_full_model_metadata = false;
        bool loads_root_weights = false;
        bool loads_expert_weights = false;

        bool hasRole(OverlayRankRole role) const;
        bool ownsDomain(const std::string &domain_name) const;
        bool hasLocalDevice(DeviceId device) const;
    };

    struct MoEExpertOverlayExecutionPlan
    {
        std::string continuation_domain;
        std::string shared_expert_domain;
        std::vector<MoEOverlayRuntimeDomain> domains;
        OverlayRankPlan current_rank;

        const OverlayRankPlan &currentRankPlan() const { return current_rank; }
        bool buildsRootGraph() const { return current_rank.builds_root_graph; }
        std::string diagnostics() const;
    };

    MoEExpertOverlayExecutionPlan buildMoEExpertOverlayExecutionPlan(
        const MoEExpertOverlayRuntimePlan &runtime_plan);

    MoEExpertOverlayExecutionPlan resolveMoEExpertOverlayExecutionPlan(
        std::shared_ptr<const MoEExpertParallelPlan> plan,
        int current_world_rank);

} // namespace llaminar2