/**
 * @file ServerExecutionEvidence.h
 * @brief Startup-only observation of the admitted execution topology and policy.
 *
 * Auto planning, saved documents and explicit CLI selection all converge on the
 * runner's frozen plans. Certification observes those plans here; it must not
 * reparse command lines, rediscover hardware, or mistake an expert-only rank's
 * CPU control endpoint for its compute devices. These values never drive
 * execution, allocation, sampling or placement decisions.
 */
#pragma once

#include "config/OrchestrationConfig.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "utils/PerfStatsCollector.h"
#include <map>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Which model work a selected physical participant owns. */
    enum class ServerParticipantRole { ExpertOnly, ModelGraph };

    /**
     * @brief Project selected participants, not every discoverable device.
     * @param plan Admitted rank plan in the execution communicator.
     * @param config Runner-owned resolved configuration, including frozen overlay.
     * @param world_size Size of that exact execution communicator.
     * @return Unique local devices and their attention/model-graph ownership.
     * @throws std::invalid_argument For missing or inconsistent topology.
     *
     * Overlay role resolution uses the same pure resolver as runner admission.
     * It performs no inventory exchange or allocation. For ordinary execution,
     * PP stage TP membership takes precedence over the stage's primary device.
     */
    inline std::map<DeviceId, ServerParticipantRole> serverExecutionParticipants(
        const RankExecutionPlan &plan, const OrchestrationConfig &config, int world_size)
    {
        if (world_size <= 0 || plan.rank < 0 || plan.rank >= world_size)
            throw std::invalid_argument("Server execution evidence requires exact rank membership");
        std::map<DeviceId, ServerParticipantRole> result;
        const auto add = [&](DeviceId device, ServerParticipantRole role)
        {
            if (!device.is_valid() || (!device.is_cpu() && !device.is_gpu()))
                throw std::invalid_argument("Server execution evidence has an invalid device");
            auto [entry, inserted] = result.emplace(device, role);
            // A device may own both routed experts and the main model. Never
            // let visiting its expert role erase its attention obligation.
            if (!inserted && role == ServerParticipantRole::ModelGraph) entry->second = role;
        };
        if (config.moe_routed_expert_plan && config.moe_routed_expert_plan->usesExpertOverlayAuthority())
        {
            const auto overlay = resolveMoEExpertOverlayExecutionPlan(config.moe_routed_expert_plan,
                MoEExpertOverlayExecutionPlanResolverOptions{
                    .current_world_rank = plan.rank, .world_size = world_size});
            const auto &local = overlay.currentRankPlan();
            for (const auto device : local.local_devices) add(device, ServerParticipantRole::ExpertOnly);
            for (const auto &domain : overlay.domains)
            {
                if (domain.name != overlay.base_model_domain) continue;
                for (const auto &participant : domain.participants)
                    if (participant.world_rank_known && participant.world_rank == plan.rank)
                        add(participant.address.toLocalDeviceId(), ServerParticipantRole::ModelGraph);
            }
            // Relay ranks truthfully own no model device. Their membership
            // record still makes their terminal evidence mandatory.
            return result;
        }
        if (!plan.local_pp_devices.empty())
        {
            if (!plan.local_pp_stage_tp_info.empty() &&
                plan.local_pp_stage_tp_info.size() != plan.local_pp_devices.size())
                throw std::invalid_argument("Server execution evidence has incomplete PP stage membership");
            for (size_t stage = 0; stage < plan.local_pp_devices.size(); ++stage)
            {
                if (!plan.local_pp_stage_tp_info.empty() && !plan.local_pp_stage_tp_info[stage].devices.empty())
                    for (const auto &device : plan.local_pp_stage_tp_info[stage].devices)
                        add(device.toLocalDeviceId(), ServerParticipantRole::ModelGraph);
                else
                    add(plan.local_pp_devices[stage].toLocalDeviceId(), ServerParticipantRole::ModelGraph);
            }
        }
        else if (!plan.local_tp_devices.empty())
            for (const auto &device : plan.local_tp_devices)
                add(device.toLocalDeviceId(), ServerParticipantRole::ModelGraph);
        else
            add(plan.primary_device.toLocalDeviceId(), ServerParticipantRole::ModelGraph);
        return result;
    }

    /**
     * @brief Encode bounded selected-device evidence for one server rank.
     * @param participants Canonical participant projection above.
     * @return Versioned tags for one startup counter, including empty relays.
     */
    inline PerfStatsCollector::Tags serverExecutionTopologyTags(
        const std::map<DeviceId, ServerParticipantRole> &participants)
    {
        std::string devices, attention;
        for (const auto &[device, role] : participants)
        {
            if (!devices.empty()) devices += ',';
            devices += device.toString();
            if (role == ServerParticipantRole::ModelGraph)
            {
                if (!attention.empty()) attention += ',';
                attention += device.toString();
            }
        }
        return {{"schema", "1"}, {"source", "resolved_execution_plan"},
                {"devices", devices}, {"attention_devices", attention}};
    }

    /**
     * @brief Observe the authority's admitted service feature policy once.
     * @param plan Runner-owned parsed policy; not mutable device request state.
     * @param config Runner-owned frozen overlay declaration.
     * @return Versioned feature tags; actual execution still needs independent counters.
     */
    inline PerfStatsCollector::Tags serverExecutionPolicyTags(
        const RankExecutionPlan &plan, const OrchestrationConfig &config)
    {
        const auto &runtime = plan.runtime;
        const auto &mtp = runtime.mtp;
        bool llep = false;
        const bool overlay = config.moe_routed_expert_plan &&
            config.moe_routed_expert_plan->usesExpertOverlayAuthority();
        if (overlay)
            for (const auto &domain : config.moe_routed_expert_plan->domains)
                llep |= domain.routed_prefill_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident;
        return {{"schema", "1"}, {"source", "resolved_execution_plan"},
                {"prefix_cache", runtime.prefix_cache.enabled ? "true" : "false"},
                {"mtp", mtp.enabled ? "true" : "false"},
                {"mtp_verify_mode", mtpVerifyModeToString(mtp.verify_mode)},
                {"mtp_depth_policy", mtpDepthPolicyModeToString(mtp.depth_policy.mode)},
                {"mtp_min_depth", std::to_string(mtp.depth_policy.min_depth)},
                {"mtp_max_depth", std::to_string(resolveMTPMaximumExecutionDraftDepth(mtp))},
                {"expert_overlay", overlay ? "true" : "false"},
                {"current_batch_llep", llep ? "true" : "false"},
                {"residency_maintenance", moeRebalanceRuntimeModeToString(runtime.moe_rebalance.mode)}};
    }
}
