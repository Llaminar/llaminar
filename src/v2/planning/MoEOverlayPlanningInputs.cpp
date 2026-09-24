/**
 * @file MoEOverlayPlanningInputs.cpp
 * @brief Shared production migration-storage policy for planning and runtime.
 *
 * This adapter declares typed storage geometry; PhysicalMemoryAuthority still
 * owns byte admission and live allocation accounting. Keeping transfer slots,
 * replica directories and workspace identities here prevents a frontend from
 * admitting cheaper storage than the runtime will actually materialize.
 */
#include "planning/MoEOverlayPlanningInputs.h"
#include "config/OrchestrationConfig.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <stdexcept>

namespace llaminar2
{
    MoEOverlayCapacityAdmissionPolicy resolveMoEOverlayCapacityAdmissionPolicy(
        const MoERoutedExpertPlacementPlan &plan,
        const OrchestrationConfig &config,
        int world_size,
        int num_layers,
        int num_experts)
    {
        if (world_size <= 0 || num_layers <= 0 || num_experts <= 0)
            throw std::invalid_argument("ExpertOverlay admission policy requires positive model and communicator geometry");
        const auto migration_storage =
            MoEOverlayCapacityAdmission::selectMigrationStorage(
                plan, config.moe_rebalance.mode);
        const bool device_transfer_directory =
            migration_storage ==
            MoEOverlayMigrationStorageKind::DeviceTransferDirectory;
        const bool physical_residency_fabric =
            migration_storage ==
            MoEOverlayMigrationStorageKind::PhysicalResidencyFabric;
        const std::size_t maximum_concurrent_cycles =
            std::max<std::size_t>(
                1,
                config.moe_rebalance
                    .migration_transfer_slots);
        const std::size_t maximum_execution_streams =
            std::max<std::size_t>(
                1,
                config.moe_rebalance
                    .resolvedMigrationExecutionStreams());
        const std::size_t maximum_cycles_per_layer =
            std::max<std::size_t>(
                1,
                config.moe_rebalance.dynamic_max_swaps_per_layer);
        DeviceMoETransferSlotDirectory::BufferedCapacity
            directory_capacity;
        std::optional<DeviceMoERebalanceWorkspaceCapacity>
            device_rebalance_workspace_capacity;
        if (device_transfer_directory)
        {
            if (num_layers <= 0 || num_experts <= 0)
            {
                throw std::invalid_argument(
                    "Device transfer-directory admission requires positive model geometry");
            }
            int hot_replica_cap =
                config.moe_hot_expert_cache.resolveCap(
                    num_experts,
                    /*dynamic_rebalance_enabled=*/true);
            const auto &env = debugEnv();
            if (env.presence.has(
                    "LLAMINAR_MOE_REBALANCE_REPLICAS"))
            {
                hot_replica_cap =
                    std::max(0, env.moe_rebalance.max_replicas);
            }
            hot_replica_cap = std::min(hot_replica_cap, num_experts);
            if (plan.replica_cache_capacity)
                hot_replica_cap = plan.replica_cache_capacity->resolve(hot_replica_cap);

            DeviceMoERebalanceConfig directory_config;
            directory_config.num_layers =
                static_cast<std::uint32_t>(num_layers);
            directory_config.max_hot_replicas_per_participant =
                static_cast<std::uint32_t>(std::min(
                    hot_replica_cap, num_experts));
            directory_capacity =
                DeviceMoETransferSlotDirectory::planRuntimeCapacity(
                    directory_config,
                    /*minimum_active_slots=*/0u,
                    static_cast<std::uint32_t>(std::max(
                        1,
                        env.moe_rebalance
                            .gpu_direct_transfer_wave_experts)),
                    static_cast<std::uint32_t>(std::max(
                        1,
                        env.moe_rebalance
                            .gpu_direct_transfer_buffers)));

            if (plan.routed_tiers.size() != 1u)
            {
                throw std::logic_error(
                    "Device transfer-directory workspace requires one routed tier");
            }
            const std::string &domain_name =
                plan.routed_tiers.front().domain;
            const auto domain = std::find_if(
                plan.domains.begin(),
                plan.domains.end(),
                [&](const RoutedExpertDomain &candidate)
                {
                    return candidate.name == domain_name;
                });
            if (domain == plan.domains.end() ||
                domain->participants.size() < 2u ||
                domain->participants.size() > kDeviceMoEMaxParticipants)
            {
                throw std::logic_error(
                    "Device transfer-directory workspace has no valid homogeneous participant domain");
            }

            std::uint32_t workspace_flags =
                static_cast<std::uint32_t>(
                    DeviceMoERebalanceFlags::ResetHistogramsAfterApply) |
                static_cast<std::uint32_t>(
                    DeviceMoERebalanceFlags::DeferRuntimeApply) |
                static_cast<std::uint32_t>(
                    DeviceMoERebalanceFlags::PlanMissingArrivals);
            if (hot_replica_cap > 0)
            {
                workspace_flags |= static_cast<std::uint32_t>(
                    DeviceMoERebalanceFlags::HotReplicaCache);
            }

            /*
             * The first routed layer produces evidence; the standalone
             * maintenance transaction plans the remaining table window.
             * This mirrors graph construction without importing graph or
             * device pointers into setup-time capacity admission.
             */
            const std::uint32_t layer_window_count =
                static_cast<std::uint32_t>(
                    num_layers > 1 ? num_layers - 1 : 1);
            device_rebalance_workspace_capacity =
                DeviceMoERebalanceWorkspaceCapacity{
                    .num_layers = static_cast<std::uint32_t>(num_layers),
                    .num_experts = static_cast<std::uint32_t>(num_experts),
                    .participant_count = static_cast<std::uint32_t>(
                        domain->participants.size()),
                    .layer_window_count = layer_window_count,
                    .layer_wave_count = static_cast<std::uint32_t>(
                        std::max(
                            0,
                            env.moe_rebalance
                                .device_rebalance_layer_wave_count)),
                    .max_hot_replicas_per_participant =
                        static_cast<std::uint32_t>(std::min(
                            hot_replica_cap, num_experts)),
                    .routed_assignment_policy =
                        domain->routed_decode_assignment_policy ==
                                RoutedExpertAssignmentPolicy::
                                    LeastLoadedResident
                            ? kDeviceMoERebalanceAssignmentLeastLoadedResident
                            : kDeviceMoERebalanceAssignmentStaticOwner,
                    .flags = workspace_flags,
                    .local_transfer_slot_count =
                        directory_capacity.total_slots,
                    .collective_payload_slot_capacity =
                        std::min<std::uint32_t>(
                            directory_capacity.total_slots,
                            static_cast<std::uint32_t>(std::max(
                                1,
                                env.moe_rebalance
                                    .device_rebalance_compact_payload_slots))),
                    .phase = DeviceMoERebalanceStagePhase::PlanCopyApply,
                    .transfer_mode =
                        DeviceMoERebalanceTransferMode::CompactTransferSlots,
                };
        }
        return {
            .migration_storage = migration_storage,
            .device_transfer_directory_capacity =
                directory_capacity,
            .device_rebalance_workspace_capacity =
                device_rebalance_workspace_capacity,
            .shadow_slots_per_endpoint_layer =
                physical_residency_fabric
                    ? MoEOverlayCapacityAdmissionPolicy::
                          requiredShadowSlotsPerEndpointLayer(
                              maximum_concurrent_cycles,
                              maximum_cycles_per_layer)
                    : 0,
            .staging_capacity_bytes =
                physical_residency_fabric
                    ? MoEOverlayCapacityAdmissionPolicy::
                          kProductionStagingBytes
                    : 0,
            .maximum_concurrent_cycles = maximum_concurrent_cycles,
            .maximum_execution_streams = maximum_execution_streams,
            .maximum_cycles_per_layer = maximum_cycles_per_layer,
            .distributed_transport =
                physical_residency_fabric && world_size > 1,
            .overlay_world_size = std::max(1, world_size),
        };
    }
}
