/**
 * @file MTPDepthPolicyScopeProjection.h
 * @brief Project learned economics from frozen production geometry, never CLI labels.
 *
 * The rank plan supplies actual participation and the original model profile
 * supplies main-layer geometry. Unsupported scalar descriptions retain generic
 * policy explicitly; this changes no execution path or runtime availability.
 */
#pragma once
#include "../config/MTPDepthPolicyScope.h"
#include "../mpi_orchestration/RankExecutionPlan.h"
#include "../mpi_orchestration/DeviceInventory.h"
#include "../../planning/ModelMemoryProfile.h"
#include <algorithm>
#include <stdexcept>
#include <utility>
namespace llaminar2
{
    /** @brief Frozen geometry and an explicit reason when no measured scope applies. */
    struct MTPDepthPolicyScopeProjection
    {
        MTPDepthPolicyScope scope;
        const char *generic_reason = "model_free";
    };

    /**
     * @brief Resolve the complete uniform topology supported by a scalar training scope.
     * @param plan Frozen actual TP/PP participants, not requested degrees.
     * @param model Loaded original main-model profile, including predictor exclusion.
     * @param retained_depth Admitted retained MTP capacity.
     * @param inventory Already-gathered architecture metadata; no device API is queried.
     * @return Complete metadata or generic policy with a precise topology reason.
     * @throws std::invalid_argument For malformed loaded geometry or stage membership.
     *
     * Remote PP does not retain complete stage membership here; nonuniform or
     * composed TP cannot be represented by one width. Neither may borrow a TP4
     * overlay. Expert-only offload followers are not model TP participants.
     */
    [[nodiscard]] inline MTPDepthPolicyScopeProjection projectMTPDepthPolicyScopeGeometry(
        const RankExecutionPlan &plan, const ModelMemoryProfile &model, int retained_depth,
        const ClusterInventory &inventory)
    {
        if (plan.prev_rank || plan.next_rank)
            return {{}, "remote_pp"};
        if (retained_depth == 0)
            return {{}, "no_retained_mtp_capacity"};
        if (plan.global_tp_domain_size != 1)
            return {{}, "global_tp_membership"};
        if (plan.primary_device.device_type == DeviceType::CPU)
            return {{}, "cpu_architecture_unscoped"};
        const int main_layers = model.mainLayerCount();
        if (main_layers <= 0)
            throw std::invalid_argument("MTP learned scope requires positive loaded main-layer geometry");
        // A retained original model profile describes its whole decoder even
        // when a diagnostic runner executes only a subset. Such a plan cannot
        // borrow economics from a complete-model campaign.
        if (plan.first_layer != 0 ||
            (plan.last_layer != -1 && plan.last_layer != main_layers - 1) ||
            !plan.has_embedding || !plan.has_lm_head)
            return {{}, "incomplete_main_model_coverage"};
        int width = plan.totalTPDegree();
        const int stages = std::max(1, static_cast<int>(plan.local_pp_devices.size()));
        if (stages > 1)
        {
            // Parent scope covers the union of all stage intervals, not only
            // its first participant. Boundaries are the frozen execution map.
            const auto &boundaries = plan.local_pp_layer_boundaries;
            if (boundaries.size() != static_cast<size_t>(stages + 1) ||
                boundaries.front() != 0 || boundaries.back() != main_layers ||
                !std::is_sorted(boundaries.begin(), boundaries.end()) ||
                std::adjacent_find(boundaries.begin(), boundaries.end()) != boundaries.end())
                return {{}, "incomplete_main_model_coverage"};
            if (plan.global_tp_domain_size != 1 || !plan.local_tp_devices.empty() || !plan.my_domains.empty())
                return {{}, "composed_tp_pp"};
            const auto &stage_tp = plan.local_pp_stage_tp_info;
            if (!stage_tp.empty() && stage_tp.size() != plan.local_pp_devices.size())
                throw std::invalid_argument("MTP learned scope requires complete frozen local PP membership");
            width = stage_tp.empty() ? 1 : std::max(1, static_cast<int>(stage_tp.front().devices.size()));
            for (const auto &stage : stage_tp)
                if (std::max(1, static_cast<int>(stage.devices.size())) != width)
                    return {{}, "nonuniform_tp_pp"};
        }
        if (plan.rank < 0 || plan.rank >= static_cast<int>(inventory.ranks.size()))
            return {{}, "missing_frozen_device_inventory"};
        const auto &rank = inventory.getRank(plan.rank);
        std::pair<int, int> architecture{0, 0};
        DeviceType architecture_backend = plan.primary_device.device_type;
        const auto observe = [&](const GlobalDeviceAddress &address) {
            if (address.device_type != architecture_backend ||
                (address.hostname != "localhost" && address.hostname != rank.hostname))
                return false;
            const DeviceInfo *found = nullptr;
            for (const auto &gpu : rank.gpus)
                if (gpu.type == address.device_type && gpu.local_device_id == address.device_ordinal)
                {
                    if (found) return false;
                    found = &gpu;
                }
            if (!found || found->compute_capability_major <= 0 || found->compute_capability_minor < 0)
                return false;
            const std::pair<int, int> actual{found->compute_capability_major, found->compute_capability_minor};
            if (architecture.first && architecture != actual) return false;
            architecture = actual;
            return true;
        };
        if (!observe(plan.primary_device))
            return {{}, "missing_or_nonuniform_hardware_architecture"};
        for (const auto &device : plan.local_tp_devices)
            if (!observe(device)) return {{}, "missing_or_nonuniform_hardware_architecture"};
        for (const auto &device : plan.local_pp_devices)
            if (!observe(device)) return {{}, "missing_or_nonuniform_hardware_architecture"};
        for (const auto &stage : plan.local_pp_stage_tp_info)
            for (const auto &device : stage.devices)
                if (!observe(device)) return {{}, "missing_or_nonuniform_hardware_architecture"};
        for (const auto &domain : plan.my_domains)
            for (const auto &device : domain.devices)
                if (!observe(device)) return {{}, "missing_or_nonuniform_hardware_architecture"};
        MTPDepthPolicyScope scope{
            .tp_width = width, .pp_stages = stages, .retained_depth = retained_depth,
            .hidden_dim = model.d_model, .main_layers = main_layers,
            .expert_count = model.expert_count, .experts_per_token = model.expert_used_count,
            .compute_capability_major = architecture.first, .compute_capability_minor = architecture.second};
        if (!scope.valid())
            throw std::invalid_argument("MTP learned scope requires valid loaded model and retained geometry");
        return {scope, "complete_uniform_geometry"};
    }
}
