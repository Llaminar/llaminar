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
#include "planning/ModelMemoryProfile.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include "utils/PerfStatsCollector.h"
#include <algorithm>
#include <map>
#include <set>
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
     * @param plan Frozen layer and predictor ownership of this rank.
     * @param model Geometry and tensor inventory of the already-loaded GGUF.
     * @return Versioned tags for one startup counter, including empty relays.
     * @throws std::invalid_argument For incomplete model or pipeline geometry.
     *
     * Model execution does not imply full attention: an auto-selected pipeline
     * slice may contain only recurrent layers. Reuse the canonical persistent-
     * state classifier, including tensor-authenticated exceptions to a periodic
     * layer map. This is observation only, not another planner or memory ledger.
     */
    inline PerfStatsCollector::Tags serverExecutionTopologyTags(
        const std::map<DeviceId, ServerParticipantRole> &participants,
        const RankExecutionPlan &plan, const ModelMemoryProfile &model)
    {
        if (model.n_layers <= 0 || model.mtp_layer_count < 0 || model.mtp_layer_count >= model.n_layers)
            throw std::invalid_argument("Server attention evidence requires complete model layer geometry");
        const int main_layers = model.n_layers - model.mtp_layer_count;
        const auto has_attention = [&](int first, int end) {
            if (first < 0 || end < first || end > model.n_layers)
                throw std::invalid_argument("Server attention evidence has an invalid layer interval");
            for (int layer = first; layer < end; ++layer)
                if (PersistentStateMemoryEstimator::isFullAttentionLayer(model, layer)) return true;
            return false;
        };
        const bool predictor_attention = plan.runtime.mtp.enabled && plan.has_lm_head &&
            has_attention(main_layers, model.n_layers);
        const size_t stages = plan.local_pp_devices.size();
        if (stages && (plan.local_pp_layer_boundaries.size() != stages + 1 ||
            (!plan.local_pp_stage_tp_info.empty() && plan.local_pp_stage_tp_info.size() != stages)))
            throw std::invalid_argument("Server attention evidence requires complete pipeline ownership");
        std::string devices, attention;
        for (const auto &[device, role] : participants)
        {
            if (!devices.empty()) devices += ',';
            devices += device.toString();
            if (role == ServerParticipantRole::ModelGraph)
            {
                int first = plan.first_layer;
                int end = plan.last_layer < 0 ? main_layers : plan.last_layer + 1;
                bool owns_predictor = predictor_attention;
                if (stages)
                {
                    // Native TP siblings own the same layer interval. Do not
                    // infer it from the backend, device ordinal or stage name.
                    size_t owner = stages;
                    for (size_t stage = 0; stage < stages; ++stage)
                    {
                        const bool selected = !plan.local_pp_stage_tp_info.empty() &&
                            !plan.local_pp_stage_tp_info[stage].devices.empty()
                            ? std::ranges::any_of(plan.local_pp_stage_tp_info[stage].devices,
                                [&](const auto &address) { return address.toLocalDeviceId() == device; })
                            : plan.local_pp_devices[stage].toLocalDeviceId() == device;
                        if (!selected) continue;
                        if (owner != stages)
                            throw std::invalid_argument("Server attention participant has multiple pipeline owners");
                        owner = stage;
                    }
                    if (owner == stages)
                        throw std::invalid_argument("Server attention participant has no pipeline owner");
                    first = plan.local_pp_layer_boundaries[owner];
                    end = plan.local_pp_layer_boundaries[owner + 1];
                    owns_predictor = predictor_attention && owner + 1 == stages;
                }
                if (end > main_layers || first >= end)
                    throw std::invalid_argument("Server attention evidence has invalid main-model bounds");
                if (!has_attention(first, end) && !owns_predictor) continue;
                if (!attention.empty()) attention += ',';
                attention += device.toString();
            }
        }
        return {{"schema", "1"}, {"source", "resolved_execution_plan"},
                {"devices", devices}, {"attention_devices", attention}};
    }

    /**
     * @brief Observe local pipeline domains without flattening their TP membership.
     * @param plan Frozen rank plan used to construct the actual stage runners.
     * @return One startup record per ordered domain, with exclusive layer bounds.
     * @throws std::invalid_argument For missing layers or ambiguous device ownership.
     *
     * Four selected GPUs alone do not prove TP-over-PP: four singleton stages
     * and two two-way TP stages use the same devices. Keep the existing plan's
     * domain boundaries in the evidence instead of reconstructing them from
     * timing, device order, or CLI strings. This projection never drives work.
     */
    inline std::vector<PerfStatsCollector::Tags> serverPipelineDomainTags(
        const RankExecutionPlan &plan)
    {
        const size_t count = plan.local_pp_devices.size();
        if (count == 0) return {};
        if (count < 2 || plan.local_pp_layer_boundaries.size() != count + 1 ||
            (!plan.local_pp_stage_tp_info.empty() && plan.local_pp_stage_tp_info.size() != count))
            throw std::invalid_argument("Server pipeline evidence requires complete domain/layer ownership");
        std::vector<PerfStatsCollector::Tags> records;
        std::set<DeviceId> owners;
        for (size_t index = 0; index < count; ++index)
        {
            const int first = plan.local_pp_layer_boundaries[index];
            const int end = plan.local_pp_layer_boundaries[index + 1];
            if (first < 0 || end <= first)
                throw std::invalid_argument("Server pipeline evidence requires nonempty contiguous layer domains");
            std::vector<GlobalDeviceAddress> devices{plan.local_pp_devices[index]};
            if (!plan.local_pp_stage_tp_info.empty() && !plan.local_pp_stage_tp_info[index].devices.empty())
                devices = plan.local_pp_stage_tp_info[index].devices;
            if (std::find(devices.begin(), devices.end(), plan.local_pp_devices[index]) == devices.end())
                throw std::invalid_argument("Server pipeline primary device is outside its domain");
            std::string labels;
            const auto backend = devices.front().toLocalDeviceId().type;
            for (const auto &address : devices)
            {
                const auto device = address.toLocalDeviceId();
                if (!device.is_valid() || device.type != backend || !owners.insert(device).second)
                    throw std::invalid_argument("Server pipeline domains require distinct homogeneous participants");
                if (!labels.empty()) labels += ',';
                labels += device.toString();
            }
            records.push_back({{"schema", "1"}, {"source", "resolved_execution_plan"},
                {"scope", "rank_local"}, {"stage", std::to_string(index)},
                {"stages", std::to_string(count)}, {"first_layer", std::to_string(first)},
                {"end_layer", std::to_string(end)}, {"devices", labels}});
        }
        return records;
    }

    /**
     * @brief Bind one executing device to its already-observed physical identity.
     * @param device Selected local participant, never an idle discoverable GPU.
     * @param inventory Exact execution-communicator rank inventory.
     * @param numa_node CPU binding from the admitted rank plan.
     * @return Startup-only tags joining local ordinals to physical node/UUID/NUMA.
     * @throws std::invalid_argument For incomplete or ambiguous observed identity.
     *
     * Distinct ranks may see the same GPU under different ordinals. UUIDs keep
     * that visibility from inflating E2E device counts. CPU identity is the
     * bound NUMA endpoint, not a process count or a vendor-specific tier role.
     */
    inline PerfStatsCollector::Tags serverExecutionParticipantTags(
        DeviceId device, const RankInventory &inventory, int numa_node)
    {
        if (inventory.rank < 0 || inventory.node_id < 0 ||
            !device.is_valid() || (!device.is_cpu() && !device.is_gpu()))
            throw std::invalid_argument("Server participant evidence requires physical rank/device identity");
        std::string physical_id;
        if (device.is_cpu())
        {
            if (numa_node < 0 || (inventory.cpu.numa_node >= 0 && inventory.cpu.numa_node != numa_node))
                throw std::invalid_argument("Server CPU participant has inconsistent NUMA ownership");
            physical_id = "numa:" + std::to_string(numa_node);
        }
        else
        {
            const auto matches = [&](const DeviceInfo &gpu) {
                return gpu.type == device.type && gpu.local_device_id == device.ordinal;
            };
            if (std::count_if(inventory.gpus.begin(), inventory.gpus.end(), matches) != 1)
                throw std::invalid_argument("Server GPU participant lacks unique inventory ownership");
            physical_id = std::find_if(inventory.gpus.begin(), inventory.gpus.end(), matches)->uuid;
            if (physical_id.empty())
                throw std::invalid_argument("Server GPU participant lacks a physical UUID");
        }
        return {{"schema", "1"}, {"source", "resolved_execution_plan"},
                {"identity_source", "communicator_cluster_inventory"},
                {"node_id", std::to_string(inventory.node_id)},
                {"backend", deviceTypeToString(device.type)}, {"physical_id", physical_id}};
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
        // Observe the same execution families as auto search. A single-domain
        // MoE TP candidate still executes through ExpertOverlay; multi-domain
        // expert placement is not a layer pipeline. No decision consumes this
        // display projection.
        const auto strategy = overlay
            ? (config.moe_routed_expert_plan->routed_tiers.size() > 1 ? "expert-overlay" : "tp")
            : (config.pp_stage_definitions.size() > 1 || plan.local_pp_devices.size() > 1 ||
               plan.prev_rank || plan.next_rank) ? "pp"
            : (plan.local_tp_devices.size() > 1 || plan.global_tp_domain_size > 1) ? "tp" : "single";
        return {{"schema", "1"}, {"source", "resolved_execution_plan"},
                {"execution_strategy", strategy},
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
