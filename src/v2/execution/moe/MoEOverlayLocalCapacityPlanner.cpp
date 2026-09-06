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
#include "planning/CapturedGraphMemoryEstimator.h"
#include "planning/MemoryPlanner.h"
#include "planning/WeightMemoryEstimator.h"
#include "config/BackendSelector.h"
#include "utils/Logger.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
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

    MoEOverlayCapturedGraphPlan
    resolveMoEOverlayCapturedGraphPlan(
        int model_layer_count,
        MoEOverlayAuthorityExecutionKind authority_execution,
        std::size_t model_graph_identity_count,
        std::size_t model_graph_topology_variant_count,
        std::size_t auxiliary_executable_count,
        std::size_t bounded_helper_executable_count)
    {
        if (model_layer_count <= 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay captured graph inventory requires positive model layers");
        }
        if (authority_execution ==
            MoEOverlayAuthorityExecutionKind::Unresolved)
        {
            throw std::invalid_argument(
                "ExpertOverlay captured graph inventory requires a frozen authority topology");
        }

        std::size_t compilation_units_per_model_graph =
            model_graph_identity_count == 0u ? 0u : 1u;
        if ((model_graph_identity_count == 0u) !=
            (model_graph_topology_variant_count == 0u))
        {
            throw std::invalid_argument(
                "ExpertOverlay captured graph inventory requires topology variants exactly when model graphs are retained");
        }
        if (model_graph_identity_count != 0u &&
            authority_execution ==
                MoEOverlayAuthorityExecutionKind::HostResident)
        {
            const std::size_t layers =
                static_cast<std::size_t>(model_layer_count);
            if (layers == std::numeric_limits<std::size_t>::max())
            {
                throw std::overflow_error(
                    "ExpertOverlay captured graph segment count overflows size_t");
            }
            compilation_units_per_model_graph = layers + 1u;
        }

        return {
            .resident_executables = {
                .model_graph_identity_count = model_graph_identity_count,
                .model_graph_topology_variant_count =
                    model_graph_topology_variant_count,
                .auxiliary_executable_count = auxiliary_executable_count,
                .bounded_helper_executable_count = bounded_helper_executable_count,
            },
            .compilation = {
                .model_graph_identity_count = model_graph_identity_count,
                .model_graph_topology_variant_count =
                    model_graph_topology_variant_count,
                .compilation_units_per_model_graph =
                    compilation_units_per_model_graph,
            },
        };
    }

    void installMoEOverlayRuntimeGPUCapacityObservation(
        RankInventory &inventory,
        DeviceId device,
        std::size_t total_bytes,
        std::size_t free_bytes)
    {
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "ExpertOverlay runtime capacity observation requires a CUDA or ROCm device");
        }
        if (total_bytes == 0 || free_bytes == 0 || free_bytes > total_bytes)
        {
            throw std::invalid_argument(
                "ExpertOverlay runtime capacity observation has invalid memory geometry for " +
                device.toString());
        }

        const auto found = std::find_if(
            inventory.gpus.begin(), inventory.gpus.end(),
            [&](const DeviceInfo &gpu)
            {
                return gpu.type == device.type &&
                       gpu.local_device_id == device.ordinal;
            });
        if (found == inventory.gpus.end())
        {
            throw std::invalid_argument(
                "ExpertOverlay runtime capacity observation names a device absent from rank " +
                std::to_string(inventory.rank) + ": " + device.toString());
        }

        /*
         * Replace both values as one observation. Mixing discovery-time total
         * memory with runtime free memory would make percentage reserves and
         * the final fit equation describe different allocator generations.
         */
        found->memory_bytes = total_bytes;
        found->free_memory_bytes = free_bytes;
    }

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
        OverlayRankExecutionKind execution_kind)
    {
        std::vector<MoEOverlayContinuationShard> shards;
        if (execution_kind !=
                OverlayRankExecutionKind::ContinuationAuthority &&
            execution_kind !=
                OverlayRankExecutionKind::ContinuationPeer)
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
        if (!input.captured_graph_plan.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay local capacity planner requires one consistent captured-graph compile/residency plan");
        }
        if (retainsMTPGraphCapacity(rank_plan.runtime.mtp) &&
            input.resident_graph_rows > 0 &&
            input.resident_graph_rows <
                resolveMTPRetainedTargetQueryRows(rank_plan.runtime.mtp))
        {
            throw std::invalid_argument(
                "ExpertOverlay resident graph rows cannot hold the configured MTP verification ceiling: resident=" +
                std::to_string(input.resident_graph_rows) +
                " required=" + std::to_string(
                    resolveMTPRetainedTargetQueryRows(
                        rank_plan.runtime.mtp)));
        }

        const auto bound =
            MoEOverlayCapacityAdmission::boundParticipants(overlay_plan);
        /* Rank locality, not tier priority, determines whether the retained
         * graph needs a cross-rank activation channel. Keep this predicate in
         * the channel planner so capacity admission and runtime preflight
         * cannot silently classify a same-tier peer differently. */
        const bool has_remote_activation_participant =
            MoEOverlayActivationChannelPlanner::hasRemoteRankParticipants(
                overlay_plan);
        MoEOverlayActivationChannelPlan activation_channel_plan;
        if (has_remote_activation_participant)
        {
            if (!input.cluster_inventory ||
                input.activation_channel_row_capacity <= 0 ||
                input.activation_graph_family_count == 0u)
            {
                throw std::invalid_argument(
                    "Distributed ExpertOverlay capacity requires the complete activation-channel topology and graph geometry");
            }
            activation_channel_plan =
                MoEOverlayActivationChannelPlanner::plan({
                    .placement_plan = &overlay_plan,
                    .cluster_inventory = input.cluster_inventory,
                    .row_capacity = static_cast<std::size_t>(
                        input.activation_channel_row_capacity),
                    .d_model = profile.d_model,
                    .top_k = profile.expert_used_count,
                    .graph_family_count =
                        input.activation_graph_family_count,
                });
        }
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
        for (const auto &charge : activation_channel_plan.staging_charges)
        {
            if (charge.world_rank == rank_plan.rank)
                resource_devices.insert(charge.device);
        }

        const std::vector<int> no_routed_experts(
            static_cast<std::size_t>(profile.n_layers), 0);
        std::vector<DevicePlanConfig> configs;
        const auto makeConfig = [&](
            DeviceId device,
            int shard_index,
            int total_shards,
            DeviceExecutionMemoryRole role)
        {
            const auto [total_bytes, available_bytes] =
                inventoryMemory(inventory, device);
            DevicePlanConfig config;
            config.world_rank = rank_plan.rank;
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
                const auto load_geometry =
                    resolveGPUWeightLoadMemoryGeometry(
                    input.gpu_weight_load->maximum_source_bytes,
                    input.gpu_weight_load->policy);
                config.weight_load_staging = {
                    .device_bytes = load_geometry.staging_bytes,
                    .host_bytes = load_geometry.host_staging_bytes,
                };
            }
            config.device_compute_units =
                inventoryComputeUnits(inventory, device);
            if (device.is_gpu())
            {
                config.graph_snapshot_memory =
                    input.graph_snapshot_memory;
            }
            config.shard_index = shard_index;
            config.total_shards = total_shards;
            config.first_layer = rank_plan.first_layer;
            config.last_layer = rank_plan.last_layer;
            config.owns_embedding =
                role == DeviceExecutionMemoryRole::ContinuationGraph &&
                rank_plan.has_embedding;
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
            /*
             * DevicePlanConfig describes resident setup bytes, not whether the
             * next request executes MTP. A positive disabled capacity therefore
             * reserves the same shifted state and verifier workspace as the
             * enabled runner that will later reuse this physical authority.
             */
            config.mtp_enabled =
                retainsMTPGraphCapacity(rank_plan.runtime.mtp);
            config.mtp_shifted_kv_head_layout =
                resolveMTPShiftedKVHeadLayout(
                    rank_plan.runtime.mtp,
                    /*dense_tensor_parallel=*/total_shards > 1,
                    total_shards);
            config.mtp_target_query_rows =
                std::max(
                    1,
                    resolveMTPRetainedTargetQueryRows(
                        rank_plan.runtime.mtp));
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
            config.prefix_cache = rank_plan.runtime.prefix_cache;
            const bool owns_host_prefix_tier =
                role == DeviceExecutionMemoryRole::ContinuationGraph &&
                config.prefix_cache.enabled &&
                config.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Disabled &&
                config.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Device;
            if (device.is_gpu() &&
                (owns_host_prefix_tier ||
                 config.weight_load_staging.host_bytes != 0u))
            {
                const auto [host_total, host_available] =
                    inventoryMemory(inventory, DeviceId::cpu());
                config.associated_host_memory = PhysicalMemoryResource{
                    .world_rank = rank_plan.rank,
                    .device = DeviceId::cpu(),
                    .total_bytes = host_total,
                    .admission_available_bytes = usableMemory(
                        host_available, DeviceId::cpu(), input),
                };
            }
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
                        total_shards,
                        rank_plan.runtime.mtp);
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

        /**
         * Install the graph/weight authority's exact local TP assignment into
         * auto-capacity accounting. This is deliberately shared through
         * TensorParallelConfig rather than reimplementing remainder or GQA
         * replication rules in the capacity planner.
         */
        const auto bindRankLocalTPAssignment = [&profile](
            DevicePlanConfig &config,
            const std::vector<GlobalDeviceAddress> &participants,
            const std::vector<float> &configured_weights)
        {
            if (participants.size() <= 1u)
                return;

            std::vector<DeviceId> devices;
            devices.reserve(participants.size());
            for (const auto &participant : participants)
                devices.push_back(participant.toLocalDeviceId());
            std::vector<float> weights = configured_weights;
            if (weights.empty())
                weights.assign(devices.size(), 1.0f);

            const auto assignments =
                TensorParallelConfig::proportionalSplit(
                    devices,
                    weights,
                    profile.n_heads,
                    profile.n_kv_heads,
                    profile.d_ff,
                    profile.vocab_size);
            config.bindTensorParallelAssignment(
                assignments.forRank(config.shard_index));
        };

        for (const auto &shard : continuationShards(
                 rank_plan, input.rank_execution_kind))
        {
            auto config = makeConfig(
                shard.device,
                shard.shard_index,
                shard.total_shards,
                DeviceExecutionMemoryRole::ContinuationGraph);
            config.first_layer = shard.first_layer;
            config.last_layer = shard.last_layer;
            config.owns_embedding =
                rank_plan.has_embedding && shard.first_layer == 0;
            if (rank_plan.usesLocalTP())
            {
                std::vector<DeviceId> tp_devices;
                tp_devices.reserve(rank_plan.local_tp_devices.size());
                for (const auto &participant : rank_plan.local_tp_devices)
                    tp_devices.push_back(participant.toLocalDeviceId());
                config.local_tp_backend = BackendSelector::resolve(
                    rank_plan.local_tp_backend, tp_devices);
                bindRankLocalTPAssignment(
                    config,
                    rank_plan.local_tp_devices,
                    rank_plan.local_tp_weights);
            }
            else if (rank_plan.usesLocalPP())
            {
                const auto &boundaries =
                    rank_plan.local_pp_layer_boundaries;
                for (std::size_t stage = 0;
                     stage < rank_plan.local_pp_stage_tp_info.size() &&
                     stage + 1u < boundaries.size();
                     ++stage)
                {
                    const auto &stage_tp =
                        rank_plan.local_pp_stage_tp_info[stage];
                    if (stage_tp.devices.size() <= 1u ||
                        boundaries[stage] != config.first_layer ||
                        boundaries[stage + 1u] - 1 != config.last_layer)
                    {
                        continue;
                    }
                    bindRankLocalTPAssignment(
                        config,
                        stage_tp.devices,
                        stage_tp.tp_weights);
                    std::vector<DeviceId> tp_devices;
                    tp_devices.reserve(stage_tp.devices.size());
                    for (const auto &participant : stage_tp.devices)
                        tp_devices.push_back(
                            participant.toLocalDeviceId());
                    config.local_tp_backend = BackendSelector::resolve(
                        stage_tp.tp_backend, tp_devices);
                    break;
                }
            }
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
        result.activation_channel_plan =
            std::move(activation_channel_plan);
        result.fixed_memory_plan = MemoryPlanner::plan(profile, configs);

        PhysicalMemoryPlanBuilder physical_plan_builder;
        for (const auto &device_plan : result.fixed_memory_plan.devices)
        {
            resource_devices.insert(device_plan.device());
            LOG_DEBUG(
                "[MoEOverlayCapacity] fixed component BOM device="
                << device_plan.device().toString()
                << " weights=" << device_plan.weight_bytes()
                << " additional_weights="
                << device_plan.additional_weight_bytes()
                << " kv_cache=" << device_plan.kv_cache_bytes()
                << " persistent_state="
                << device_plan.persistent_state_bytes()
                << " prefix_staging="
                << device_plan.prefix_cache_staging_bytes()
                << " prefix_device_hot="
                << device_plan.prefix_cache_device_hot_bytes()
                << " collective=" << device_plan.collective_bytes()
                << " activations=" << device_plan.activation_bytes()
                << " workspace=" << device_plan.workspace_bytes()
                << " retained_workspace="
                << device_plan.retained_workspace_bytes()
                << " total=" << device_plan.total_bytes());
            physical_plan_builder.add(device_plan.bom());
        }

        for (const DeviceId device : resource_devices)
        {
            const auto [total_bytes, available_bytes] =
                inventoryMemory(inventory, device);
            const PhysicalMemoryResource resource{
                .world_rank = rank_plan.rank,
                .device = device,
                .total_bytes = total_bytes,
                .admission_available_bytes = usableMemory(
                    available_bytes, device, input),
            };
            const std::size_t activation_staging_bytes =
                result.activation_channel_plan.stagingBytesFor(
                    rank_plan.rank, device);
            const std::size_t captured_graph_bytes =
                device.is_gpu()
                    ? estimateCapturedGraphExecutableBytes(
                          device,
                          input.captured_graph_plan.resident_executables)
                    : 0u;
            PhysicalMemoryBOMBuilder additions(resource);
            additions
                .add(
                    PhysicalMemoryOwner::NativeGraphExecutable,
                    captured_graph_bytes)
                .add(
                    PhysicalMemoryOwner::ActivationTransportStaging,
                    activation_staging_bytes);
            physical_plan_builder.add(additions.build());
        }

        result.physical_memory_admission = std::make_shared<
            const PhysicalMemoryPlanAdmissionCertificate>(
            physical_plan_builder.build());
        result.physical_budgets.reserve(
            result.physical_memory_admission->plan().resources().size());
        for (const auto &bom :
             result.physical_memory_admission->plan().resources())
        {
            const PhysicalMemoryAllocatorIdentity identity{
                .world_rank = bom.resource().world_rank,
                .device = bom.resource().device,
            };
            result.physical_budgets.emplace_back(
                physicalResourceId(identity.world_rank, identity.device),
                result.physical_memory_admission,
                identity);
        }
        return result;
    }
} // namespace llaminar2
