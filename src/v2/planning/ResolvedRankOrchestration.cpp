/**
 * @file ResolvedRankOrchestration.cpp
 * @brief Shared pure composition of rank compilation and expert authority binding.
 *
 * User topology is copied, resolved against observed hardware, and validated
 * before one result is returned. Simple multi-device MoE is normalized through
 * the same single-tier authority as explicit overlays. No frontend-specific
 * placement, fake model geometry, physical-byte arithmetic or MPI protocol lives
 * here. Runtime and automatic candidates therefore use identical role lowering.
 */
#include "planning/ResolvedRankOrchestration.h"
#include "planning/PlanningModelMetadata.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include "config/OrchestrationConfigParser.h"
#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Preserve every validation diagnostic in one fail-closed result.
         * @param errors Complete diagnostics from the existing compiler.
         * @param operation Phase name attached to the rejection.
         * @throws std::invalid_argument when any validation failed.
         */
        void requireValid(const std::vector<std::string> &errors, std::string_view operation)
        {
            if (errors.empty()) return;
            std::string message(operation);
            for (const auto &error : errors) message += "\n  - " + error;
            throw std::invalid_argument(message);
        }

        /**
         * @brief Install one hardware-resolved overlay using shared normalization.
         * @param config Private working copy; the caller's request is never edited.
         * @param inventory Observed rank and device ownership in this namespace.
         * @param origin Explicit declaration or model-driven LocalTP synthesis.
         * @throws std::invalid_argument when binding or normalization fails.
         */
        void bindOverlay(OrchestrationConfig &config, const ClusterInventory &inventory,
                         MoEExpertOverlayPlanInstallOrigin origin)
        {
            if (!config.moe_routed_expert_plan || !config.moe_routed_expert_plan->usesExpertOverlayAuthority())
                return;
            auto bound = bindMoEExpertOverlayPlanToClusterInventory(*config.moe_routed_expert_plan, inventory);
            requireValid(installResolvedMoEExpertOverlayPlan(config, std::move(bound), origin),
                         "Cannot install hardware-resolved ExpertOverlay topology");
        }

        /**
         * @brief Narrow the compiler result to its model-owned overlay role.
         * @param rank_plan Rank/shard facts produced by ExecutionPlanBuilder.
         * @param overlay Bound model topology, never a requested/unresolved map.
         * @param execution Exact continuation/follower ownership for this rank.
         * @throws std::invalid_argument if role and dense-domain geometry disagree.
         */
        void applyOverlayRole(RankExecutionPlan &rank_plan,
                              const MoERoutedExpertPlacementPlan &overlay,
                              const MoEExpertOverlayExecutionPlan &execution)
        {
            // An expert-only endpoint has no ordinary local TP/PP graph. Its
            // device set is owned by the overlay execution plan, not a made-up
            // continuation shard. Preserve the existing CPU control endpoint.
            if (!execution.ownsContinuationGraph())
            {
                rank_plan.local_tp_devices.clear();
                rank_plan.local_tp_weights.clear();
                rank_plan.local_tp_backend = CollectiveBackendType::AUTO;
                rank_plan.local_pp_devices.clear();
                rank_plan.local_pp_layer_boundaries.clear();
                rank_plan.local_pp_stage_tp_info.clear();
                rank_plan.primary_device = GlobalDeviceAddress::cpu();
                rank_plan.weight_shard = {};
                return;
            }
            const auto name = overlay.effectiveBaseModelDomain();
            const auto found = std::find_if(overlay.domains.begin(), overlay.domains.end(),
                [&](const auto &domain) { return domain.name == name; });
            if (found == overlay.domains.end() || found->participants.empty())
                throw std::invalid_argument("ExpertOverlay continuation domain '" + name + "' has no participants");
            const auto &domain = *found;
            if (domain.scope == ExecutionDomainScope::NODE_LOCAL)
            {
                if (overlay.continuation_domain_spec.effectiveDensePolicy() != DenseParallelPolicy::TensorParallel)
                    throw std::invalid_argument("ExpertOverlay NodeTP continuation requires dense_policy=tensor_parallel");
                const bool participates = std::any_of(rank_plan.my_domains.begin(), rank_plan.my_domains.end(),
                    [&](const auto &member) { return member.domain_name == name; });
                if (!participates || !rank_plan.usesGlobalTP() || rank_plan.global_tp_domain_size <= 1)
                    throw std::invalid_argument("ExpertOverlay continuation rank did not retain its cross-rank dense TP domain");
                // NodeTP already owns exact CPU affinity, shard identity and
                // collective membership. Never turn it into another LocalTP.
                rank_plan.local_pp_devices.clear();
                rank_plan.local_pp_layer_boundaries.clear();
                rank_plan.local_pp_stage_tp_info.clear();
                return;
            }
            rank_plan.local_pp_devices.clear();
            rank_plan.local_pp_layer_boundaries.clear();
            rank_plan.local_pp_stage_tp_info.clear();
            rank_plan.primary_device = domain.participants.front();
            rank_plan.local_tp_backend = domain.backend;
            rank_plan.local_tp_weights = domain.weights;
            rank_plan.weight_shard = {};
            if (domain.scope == ExecutionDomainScope::RANK_LOCAL)
            {
                rank_plan.tp_scope = TPScope::RANK_LOCAL;
                rank_plan.local_tp_devices = domain.participants;
            }
            else
            {
                rank_plan.local_tp_devices.clear();
                rank_plan.local_tp_weights.clear();
                rank_plan.tp_scope = TPScope::AUTO;
            }
        }
    }

    ResolvedRankOrchestration::ResolvedRankOrchestration(
        OrchestrationConfig config, RankExecutionPlan rank_plan,
        std::optional<MoEExpertOverlayExecutionPlan> overlay_execution,
        MoEExpertOverlayAuthorityPlanDisposition origin)
        : config_(std::move(config)), rank_plan_(std::move(rank_plan)),
          overlay_execution_(std::move(overlay_execution)), overlay_origin_(origin) {}

    ResolvedRankOrchestration ResolvedRankOrchestration::resolve(
        const OrchestrationConfig &requested, const PlanningModelMetadata &model,
        const ClusterInventory &inventory, IExecutionPlanBuilder &builder, int rank)
    {
        if (inventory.world_size <= 0 || inventory.ranks.size() != static_cast<size_t>(inventory.world_size) ||
            rank < 0 || rank >= inventory.world_size)
            throw std::invalid_argument("Rank orchestration requires exact inventory membership");
        // Compilation may inspect any peer while resolving a domain. Validate
        // the complete rank namespace, not merely the current rank's record.
        for (int peer = 0; peer < inventory.world_size; ++peer)
            if (inventory.ranks[peer].rank != peer)
                throw std::invalid_argument("Rank orchestration requires exact inventory membership");

        auto config = requested;
        const auto geometry = model.executionModelConfig();
        bindOverlay(config, inventory, MoEExpertOverlayPlanInstallOrigin::UserDeclaredDomains);
        requireValid(builder.validateConfig(config, geometry, inventory), "Configuration validation failed");
        auto rank_plan = builder.buildPlanForRank(config, geometry, inventory, rank);
        const auto authority = normalizeMoEExpertOverlayAuthorityPlan({
            .requested_plan = config.moe_routed_expert_plan,
            .model_has_routed_experts = model.memoryProfile().expert_count > 0,
            .world_rank = rank,
            .local_tp_participants = rank_plan.local_tp_devices,
            .local_tp_weights = rank_plan.local_tp_weights,
            .local_tp_backend = rank_plan.local_tp_backend,
            .has_cross_rank_tensor_parallel = rank_plan.usesGlobalTP(),
            .has_pipeline_parallel = rank_plan.usesLocalPP() || rank_plan.usesPipelineParallel(),
            .routed_compute_policy = config.routed_expert_compute_policy,
            .owner_order = config.routed_expert_owner_order,
            .residency_maintenance = config.moe_rebalance.mode,
        });
        config.moe_routed_expert_plan = authority.plan;
        if (authority.synthesized())
        {
            // Synthesis changes the declarative domain inventory. Recompile
            // once before sealing; no consumer may retain the preliminary view.
            bindOverlay(config, inventory, MoEExpertOverlayPlanInstallOrigin::SynthesizedSimpleTP);
            requireValid(builder.validateConfig(config, geometry, inventory), "Implicit ExpertOverlay validation failed");
            rank_plan = builder.buildPlanForRank(config, geometry, inventory, rank);
        }
        config.mtp.depth_defaults_profile = rank_plan.runtime.mtp.depth_defaults_profile;
        std::optional<MoEExpertOverlayExecutionPlan> overlay_execution;
        if (config.moe_routed_expert_plan && config.moe_routed_expert_plan->usesExpertOverlayAuthority())
        {
            overlay_execution.emplace(resolveMoEExpertOverlayExecutionPlan(config.moe_routed_expert_plan,
                MoEExpertOverlayExecutionPlanResolverOptions{.current_world_rank = rank, .world_size = inventory.world_size}));
            applyOverlayRole(rank_plan, *config.moe_routed_expert_plan, *overlay_execution);
        }
        requireValid(rank_plan.validate(), "Plan validation failed");
        return ResolvedRankOrchestration(std::move(config), std::move(rank_plan),
                                         std::move(overlay_execution), authority.disposition);
    }
}
