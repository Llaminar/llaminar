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
#include <optional>
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
        std::vector<std::string> root_weight_domains;
        std::vector<std::string> shared_expert_weight_domains;
        std::vector<std::string> accelerator_routed_expert_domains;
        std::vector<std::string> cpu_fallback_expert_domains;
        std::vector<std::string> worker_fallback_expert_domains;
        std::vector<DeviceId> local_devices;
        bool builds_root_graph = false;
        bool loads_tokenizer = false;
        bool loads_worker_tokenizer_state = false;
        bool loads_full_model_metadata = false;
        bool loads_root_weights = false;
        bool loads_shared_expert_weights = false;
        bool loads_accelerator_routed_experts = false;
        bool loads_cpu_fallback_experts = false;
        bool loads_worker_fallback_experts = false;
        bool loads_expert_weights = false;

        bool hasRole(OverlayRankRole role) const;
        bool ownsDomain(const std::string &domain_name) const;
        bool hasLocalDevice(DeviceId device) const;
    };

    struct MoEExpertOverlayExecutionPlanResolverOptions
    {
        int current_world_rank = 0;

        /// MPI world size. When <= 0, the resolver infers the minimum world
        /// size needed by explicit owner/rank fields and current_world_rank.
        int world_size = 0;
    };

    struct MoEExpertOverlayExecutionPlan
    {
        int world_size = 1;
        std::string continuation_domain;
        std::string base_model_domain;
        std::string shared_expert_domain;
        int continuation_root_rank = -1;
        std::vector<MoEOverlayRuntimeDomain> domains;
        std::vector<OverlayRankPlan> rank_plans;
        OverlayRankPlan current_rank;

        const OverlayRankPlan &currentRankPlan() const { return current_rank; }
        const OverlayRankPlan *rankPlanFor(int world_rank) const;
        bool buildsRootGraph() const { return current_rank.builds_root_graph; }
        std::string diagnostics() const;
    };

    MoEExpertOverlayExecutionPlan buildMoEExpertOverlayExecutionPlan(
        const MoEExpertOverlayRuntimePlan &runtime_plan,
        int world_size = 0);

    MoEExpertOverlayExecutionPlan resolveMoEExpertOverlayExecutionPlan(
        std::shared_ptr<const MoEExpertParallelPlan> plan,
        int current_world_rank);

    MoEExpertOverlayExecutionPlan resolveMoEExpertOverlayExecutionPlan(
        std::shared_ptr<const MoEExpertParallelPlan> plan,
        const MoEExpertOverlayExecutionPlanResolverOptions &options);

    std::optional<std::string> graphNativeMoEOverlayBuildBlocker(
        const MoEExpertOverlayExecutionPlan &execution_plan);

} // namespace llaminar2
