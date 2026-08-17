/**
 * @file MoEOverlayLocalCapacityPlanner.cpp
 * @brief Rank-local fixed-memory BOM implementation for ExpertOverlay.
 *
 * This implementation mirrors OrchestrationRunner's production memory roles
 * while setting every routed-expert selection to zero. The resulting bytes
 * are therefore the immutable base of the global capacity inequality; exact
 * codebook-specific live and shadow expert bytes are added only by
 * MoEOverlayCapacityResolver.
 */

#include "MoEOverlayLocalCapacityPlanner.h"

#include "planning/ActivationBufferSizing.h"
#include "planning/MemoryPlanner.h"
#include "planning/WeightMemoryEstimator.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        [[nodiscard]] std::size_t checkedAdd(
            std::size_t left,
            std::size_t right,
            const char *what)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay local capacity ") + what +
                    " overflows size_t");
            }
            return left + right;
        }

        [[nodiscard]] std::pair<std::size_t, std::size_t> inventoryMemory(
            const RankInventory &inventory,
            DeviceId device)
        {
            if (device.is_cpu())
            {
                const std::size_t total_bytes =
                    inventory.cpu.memory_bytes > 0
                        ? inventory.cpu.memory_bytes
                        : inventory.cpu_memory_bytes;
                if (total_bytes == 0)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay CPU capacity inventory reports zero memory");
                }
                if (inventory.cpu.free_memory_bytes == 0)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay CPU capacity inventory has no positive NUMA-local available-memory authority; provide a discovered endpoint before admitting CPU experts");
                }
                return {
                    total_bytes,
                    std::min(
                        total_bytes,
                        inventory.cpu.free_memory_bytes),
                };
            }
            if (!device.is_gpu())
            {
                throw std::invalid_argument(
                    "ExpertOverlay local capacity requires CPU, CUDA, or ROCm resources");
            }
            const auto found = std::find_if(
                inventory.gpus.begin(), inventory.gpus.end(),
                [&](const auto &gpu)
                {
                    return gpu.type == device.type &&
                           gpu.local_device_id == device.ordinal;
                });
            if (found == inventory.gpus.end() ||
                found->memory_bytes == 0 ||
                found->free_memory_bytes == 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay rank inventory has no positive memory record for " +
                    device.toString());
            }
            return {found->memory_bytes, found->free_memory_bytes};
        }

        [[nodiscard]] std::size_t usableMemory(
            std::size_t inventory_available,
            DeviceId device,
            const MoEOverlayLocalCapacityPlannerInput &input)
        {
            const auto &limit = device.is_cpu()
                                    ? input.max_cpu_memory_bytes
                                    : input.max_gpu_memory_bytes;
            return limit.has_value()
                       ? std::min(inventory_available, *limit)
                       : inventory_available;
        }

        /** @brief Return the physical SM/CU count for capture-time policy. */
        [[nodiscard]] int inventoryComputeUnits(
            const RankInventory &inventory,
            DeviceId device)
        {
            if (device.is_cpu())
                return inventory.cpu.compute_units;
            const auto found = std::find_if(
                inventory.gpus.begin(), inventory.gpus.end(),
                [&](const auto &gpu)
                {
                    return gpu.type == device.type &&
                           gpu.local_device_id == device.ordinal;
                });
            if (found == inventory.gpus.end())
            {
                throw std::invalid_argument(
                    "ExpertOverlay inventory has no compute geometry for " +
                    device.toString());
            }
            return found->compute_units;
        }

        [[nodiscard]] bool containsDevice(
            const std::vector<DevicePlanConfig> &configs,
            DeviceId device)
        {
            return std::any_of(
                configs.begin(), configs.end(), [&](const auto &config)
                { return config.device == device; });
        }
    } // namespace

    std::string MoEOverlayLocalCapacityPlanner::physicalResourceId(
        int world_rank,
        DeviceId device)
    {
        if (world_rank < 0 || !device.is_valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay physical resource identity requires a bound rank and device");
        }
        return "overlay-rank=" + std::to_string(world_rank) +
               ";device=" + device.toString();
    }

    std::vector<MoEOverlayContinuationShard>
    MoEOverlayLocalCapacityPlanner::continuationShards(
        const RankExecutionPlan &rank_plan,
        bool builds_root_graph)
    {
        std::vector<MoEOverlayContinuationShard> shards;
        if (!builds_root_graph)
            return shards;

        const auto append = [&shards](
            DeviceId device,
            int shard_index,
            int total_shards,
            int first_layer,
            int last_layer)
        {
            shards.push_back({
                .device = device,
                .shard_index = shard_index,
                .total_shards = total_shards,
                .first_layer = first_layer,
                .last_layer = last_layer,
            });
        };

        if (rank_plan.usesLocalPP())
        {
            const auto &devices = rank_plan.local_pp_devices;
            const auto &boundaries =
                rank_plan.local_pp_layer_boundaries;
            const auto &stage_tp = rank_plan.local_pp_stage_tp_info;
            if (boundaries.size() != devices.size() + 1u)
            {
                throw std::invalid_argument(
                    "ExpertOverlay LocalPP continuation has incomplete layer boundaries");
            }
            for (std::size_t stage = 0; stage < devices.size(); ++stage)
            {
                const int first_layer = boundaries[stage];
                const int last_layer = boundaries[stage + 1] - 1;
                if (stage < stage_tp.size() &&
                    stage_tp[stage].devices.size() > 1u)
                {
                    const int degree = static_cast<int>(
                        stage_tp[stage].devices.size());
                    for (int shard = 0; shard < degree; ++shard)
                    {
                        append(
                            stage_tp[stage]
                                .devices[static_cast<std::size_t>(shard)]
                                .toLocalDeviceId(),
                            shard,
                            degree,
                            first_layer,
                            last_layer);
                    }
                }
                else
                {
                    append(
                        devices[stage].toLocalDeviceId(),
                        0,
                        1,
                        first_layer,
                        last_layer);
                }
            }
            return shards;
        }

        if (rank_plan.usesLocalTP())
        {
            const int degree = static_cast<int>(
                rank_plan.local_tp_devices.size());
            for (int shard = 0; shard < degree; ++shard)
            {
                append(
                    rank_plan.local_tp_devices[
                        static_cast<std::size_t>(shard)]
                        .toLocalDeviceId(),
                    shard,
                    degree,
                    rank_plan.first_layer,
                    rank_plan.last_layer);
            }
            return shards;
        }

        append(
            rank_plan.primary_device.toLocalDeviceId(),
            rank_plan.weight_shard.shard_index,
            rank_plan.weight_shard.total_shards,
            rank_plan.first_layer,
            rank_plan.last_layer);
        return shards;
    }

    MoEOverlayLocalCapacityPlannerResult
    MoEOverlayLocalCapacityPlanner::plan(
        const MoEOverlayLocalCapacityPlannerInput &input)
    {
        if (!input.model_profile || !input.rank_plan ||
            !input.overlay_plan || !input.rank_inventory)
        {
            throw std::invalid_argument(
                "ExpertOverlay local capacity planner requires model, rank, topology, and inventory authorities");
        }
        const auto &profile = *input.model_profile;
        const auto &rank_plan = *input.rank_plan;
        const auto &overlay_plan = *input.overlay_plan;
        const auto &inventory = *input.rank_inventory;
        if (profile.n_layers <= 0 || profile.expert_count <= 0 ||
            rank_plan.rank < 0 || inventory.rank != rank_plan.rank)
        {
            throw std::invalid_argument(
                "ExpertOverlay local capacity planner received invalid model/rank geometry");
        }

        const auto bound =
            MoEOverlayCapacityAdmission::boundParticipants(overlay_plan);
        std::set<DeviceId> endpoint_devices;
        std::map<DeviceId, int> endpoint_participant_counts;
        for (const auto &participant : bound)
        {
            if (participant.world_rank == rank_plan.rank)
            {
                endpoint_devices.insert(participant.device);
                auto &count = endpoint_participant_counts[participant.device];
                if (count == std::numeric_limits<int>::max())
                {
                    throw std::overflow_error(
                        "ExpertOverlay local participant count exceeds int");
                }
                ++count;
            }
        }
        std::set<DeviceId> resource_devices = endpoint_devices;
        if (input.require_host_memory_authority)
            resource_devices.insert(DeviceId::cpu());

        const std::vector<int> no_routed_experts(
            static_cast<std::size_t>(profile.n_layers), 0);
        std::vector<DevicePlanConfig> configs;
        std::map<DeviceId, GPUWeightLoadMemoryBOM> gpu_weight_load_boms;
        const auto makeConfig = [&](
            DeviceId device,
            int shard_index,
            int total_shards,
            DeviceExecutionMemoryRole role)
        {
            const auto [total_bytes, available_bytes] =
                inventoryMemory(inventory, device);
            DevicePlanConfig config;
            config.device = device;
            config.device_total_bytes = total_bytes;
            config.device_free_bytes = usableMemory(
                available_bytes, device, input);
            if (device.is_gpu())
            {
                if (!input.gpu_weight_load.has_value() ||
                    input.gpu_weight_load->maximum_source_bytes == 0)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay GPU capacity requires a complete model-specific weight-load contract for " +
                        device.toString());
                }
                /*
                 * The zero-persistent-weight bill isolates the two transient
                 * components. Persistent dense and expert weights are charged
                 * by MemoryPlanner and the global quota resolver respectively;
                 * charging them here would double-count them.
                 */
                const auto load_bom = gpuWeightLoadMemoryBOM(
                    /*planned_weight_bytes=*/0,
                    input.gpu_weight_load->maximum_source_bytes,
                    config.device_free_bytes,
                    total_bytes,
                    input.gpu_weight_load->policy);
                config.headroom_bytes = load_bom.safety_margin_bytes;
                gpu_weight_load_boms[device] = load_bom;
            }
            config.device_compute_units =
                inventoryComputeUnits(inventory, device);
            config.shard_index = shard_index;
            config.total_shards = total_shards;
            config.first_layer = rank_plan.first_layer;
            config.last_layer = rank_plan.last_layer;
            config.batch_size = rank_plan.runtime.batch_size;
            config.max_seq_len = rank_plan.runtime.max_seq_len;
            config.activation_seq_len =
                input.resident_graph_rows > 0 &&
                        (device.is_gpu() ||
                         role == DeviceExecutionMemoryRole::RoutedExpertParticipant)
                    ? std::min(
                          std::max(1, rank_plan.runtime.max_seq_len),
                          input.resident_graph_rows)
                    : resolveActivationBufferSeqLen(
                          config.max_seq_len, device);
            config.mtp_enabled = rank_plan.runtime.mtp.enabled;
            config.mtp_target_query_rows =
                resolveMTPMaxTargetQueryRows(rank_plan.runtime.mtp);
            config.mtp_terminal_logits_layout =
                resolveMTPTerminalLogitsLayout(
                    total_shards > 1,
                    rank_plan.runtime.mtp.terminal_head_policy);
            /*
             * Use the production selector as the single AUTO authority. The
             * estimator accepts activation precision names case-insensitively,
             * including TQ8 for the asymmetric TQ cache's key codec.
             */
            config.kv_precision = activationPrecisionToString(
                resolveKVCacheStoragePrecision(
                    rank_plan.runtime.kv_cache_precision,
                    device.is_cpu()));
            if (total_shards > 1 && profile.n_kv_heads > 0)
            {
                config.local_kv_heads =
                    std::max(1, profile.n_kv_heads / total_shards);
            }
            config.routed_expert_compact_buffer_lifetime =
                RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant;
            const auto participant_count =
                endpoint_participant_counts.find(device);
            config.serial_routed_expert_participant_count =
                participant_count == endpoint_participant_counts.end()
                    ? 0
                    : participant_count->second;
            const int overlay_segment_rows = std::min(
                std::max(1, config.activation_seq_len),
                rank_plan.runtime.moe_routed_prefill
                    .overlay_segment_rows);
            config.serial_routed_expert_compact_rows = std::max(
                overlay_segment_rows,
                config.mtp_enabled
                    ? config.mtp_target_query_rows
                    : 1);
            config.execution_role = role;
            if (role == DeviceExecutionMemoryRole::ContinuationGraph)
            {
                config.additional_weight_sets =
                    resolveAdditionalPersistentWeightSets(
                        overlay_plan.continuation_domain_spec
                            .effectiveDensePolicy(),
                        total_shards);
            }
            config.weight_residency =
                role == DeviceExecutionMemoryRole::ContinuationGraph
                    ? DeviceWeightResidency::
                          continuationWithSelectedRoutedExperts(
                              profile.expert_count, no_routed_experts)
                    : DeviceWeightResidency::selectedRoutedExpertsOnly(
                          profile.expert_count, no_routed_experts);
            return config;
        };

        for (const auto &shard : continuationShards(
                 rank_plan, input.builds_root_graph))
        {
            auto config = makeConfig(
                shard.device,
                shard.shard_index,
                shard.total_shards,
                DeviceExecutionMemoryRole::ContinuationGraph);
            config.first_layer = shard.first_layer;
            config.last_layer = shard.last_layer;
            resource_devices.insert(config.device);
            configs.push_back(std::move(config));
        }

        /* A tier-only device owns its sparse graph workspace but no dense state. */
        for (const DeviceId device : endpoint_devices)
        {
            if (containsDevice(configs, device))
                continue;
            configs.push_back(makeConfig(
                device,
                0,
                1,
                DeviceExecutionMemoryRole::RoutedExpertParticipant));
        }

        MoEOverlayLocalCapacityPlannerResult result;
        result.resident_graph_rows = input.resident_graph_rows;
        result.fixed_memory_plan = MemoryPlanner::plan(profile, configs);

        struct GroupedBytes
        {
            std::size_t fixed = 0;
            std::size_t safety = 0;
        };
        std::map<DeviceId, GroupedBytes> grouped;
        for (const auto &device_plan : result.fixed_memory_plan.devices)
        {
            auto &entry = grouped[device_plan.device];
            entry.fixed = checkedAdd(
                entry.fixed,
                device_plan.total_bytes(),
                "fixed device BOM");
            entry.safety = std::max(
                entry.safety, device_plan.headroom_bytes);
        }

        result.physical_budgets.reserve(resource_devices.size());
        for (const DeviceId device : resource_devices)
        {
            const auto [total_bytes, available_bytes] =
                inventoryMemory(inventory, device);
            (void)total_bytes;
            const auto found = grouped.find(device);
            const GroupedBytes bytes =
                found == grouped.end() ? GroupedBytes{} : found->second;
            const auto gpu_load = gpu_weight_load_boms.find(device);
            const std::size_t gpu_load_staging_bytes =
                gpu_load == gpu_weight_load_boms.end()
                    ? 0
                    : gpu_load->second.staging_bytes;
            result.physical_budgets.push_back({
                .world_rank = rank_plan.rank,
                .device = device,
                .resource_id = physicalResourceId(rank_plan.rank, device),
                .usable_budget_bytes = usableMemory(
                    available_bytes, device, input),
                .fixed_bytes = bytes.fixed,
                .additional_transfer_staging_bytes =
                    gpu_load_staging_bytes,
                .safety_reserve_bytes = bytes.safety,
            });
        }
        return result;
    }
} // namespace llaminar2
