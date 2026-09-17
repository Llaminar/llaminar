/**
 * @file Test__MoEOverlayPlanningInputs.cpp
 * @brief Device-free proofs of shared plan/runtime migration-storage inputs.
 *
 * These tests invoke the production policy adapter, not a command-private
 * estimator. They cover CPU/CUDA/ROCm, one and multiple ranks, replica limits,
 * transfer concurrency, retained MTP capacity, capture/snapshot ownership and
 * invalid geometry without allocating model/device memory. Format-specific byte
 * arithmetic remains covered by the existing
 * all-codebook MoEOverlayCapacityResolver suite downstream of this adapter.
 */
#include "planning/MoEOverlayPlanningInputs.h"
#include "planning/MoEOverlayMemoryPlanInputs.h"
#include "config/OrchestrationConfig.h"
#include <gtest/gtest.h>
#include <limits>

namespace llaminar2
{
    namespace
    {
        /** @return Bound two-participant domain with an explicit authority. */
        MoERoutedExpertPlacementPlan topology(DeviceType backend, bool cross_rank = false)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::SingleDomain;
            plan.authority_execution = backend == DeviceType::CPU
                ? MoEOverlayAuthorityExecutionKind::HostResident
                : MoEOverlayAuthorityExecutionKind::DeviceResident;
            RoutedExpertDomain domain;
            domain.name = "opaque-domain";
            domain.scope = cross_rank ? ExecutionDomainScope::GLOBAL : ExecutionDomainScope::RANK_LOCAL;
            domain.backend = backend == DeviceType::CPU ? CollectiveBackendType::MPI
                : backend == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
            domain.owner_rank = 0;
            if (cross_rank) domain.world_ranks = {0, 1};
            for (int index = 0; index < 2; ++index)
                domain.participants.push_back(GlobalDeviceAddress::fromLocalDeviceId(
                    DeviceId(backend, index), cross_rank && index ? "remote" : "local", index));
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            plan.continuation_domain = domain.name;
            plan.base_model_domain = domain.name;
            plan.shared_expert_domain = domain.name;
            plan.domains = {domain};
            plan.routed_tiers = {{.name = "not-a-temperature", .domain = domain.name, .priority = 37, .fallback = true}};
            return plan;
        }

        /** @brief Model-free, explicit owners for the same production BOM adapter. */
        struct MemoryInputsFixture
        {
            ModelMemoryProfile model;
            RankExecutionPlan rank;
            OrchestrationConfig config;
            ClusterInventory inventory;
            MoEExpertOverlayExecutionPlan execution;
            MoEOverlayCapacityAdmissionPolicy policy;
            MoEOverlayInferenceGraphFamilyIdentity family;
            std::vector<int> buckets{1, 8, 32, 64, 256};

            /** @brief Prepare a declared rank namespace and model/retained MTP geometry. */
            explicit MemoryInputsFixture(DeviceType backend, int world_size = 1)
            {
                model.n_layers = 3;
                model.expert_count = 8;
                model.expert_used_count = 2;
                rank.rank = 0;
                rank.runtime.batch_size = 1;
                rank.runtime.moe_routed_prefill.overlay_segment_rows = 256;
                config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
                config.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(
                    topology(backend, world_size > 1));
                inventory.world_size = world_size;
                inventory.ranks.resize(world_size);
                for (int index = 0; index < world_size; ++index) inventory.ranks[index].rank = index;
                execution = resolveMoEExpertOverlayExecutionPlan(config.moe_routed_expert_plan,
                    {.current_world_rank = 0, .world_size = world_size});
                policy = resolveMoEOverlayCapacityAdmissionPolicy(
                    *config.moe_routed_expert_plan, config, world_size, 3, 8);
                family = {.graph_family_generation = 1, .main_layer_count = 2,
                    .mtp_source_layers = {}, .max_graph_rows = 256, .max_decode_rows = 1,
                    .max_request_count = 1, .max_mtp_draft_depth = 0};
            }

            /** @return Borrowed immutable setup facts, without reading global debug state. */
            MoEOverlayMemoryPlanInputRequest request() const
            {
                return {.model = model, .rank_plan = rank, .config = config,
                    .inventory = inventory, .execution = execution, .capacity_policy = policy,
                    .retained_mtp = rank.runtime.mtp, .graph_family = family,
                    .prefill = {.bucket_rows = buckets, .minimum_sequence_rows = 1, .maximum_cached_buckets = 16},
                    .gpu_weight_load = {.maximum_source_bytes = 1234}};
            }
        };
    }

    TEST(MoEOverlayPlanningInputs, StaticDoesNotPriceAnyMigrationStorageOnAnyBackend)
    {
        OrchestrationConfig config;
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
            for (const int ranks : {1, 2, 3})
            {
                const auto policy = resolveMoEOverlayCapacityAdmissionPolicy(topology(backend), config, ranks, 41, 256);
                EXPECT_FALSE(policy.migrationEnabled());
                EXPECT_EQ(policy.shadow_slots_per_endpoint_layer, 0u);
                EXPECT_EQ(policy.staging_capacity_bytes, 0u);
                EXPECT_EQ(policy.device_transfer_directory_capacity.total_slots, 0u);
                EXPECT_FALSE(policy.device_rebalance_workspace_capacity);
                EXPECT_FALSE(policy.distributed_transport);
                EXPECT_EQ(policy.overlay_world_size, ranks);
            }
    }

    TEST(MoEOverlayPlanningInputs, NativeGpuPolicyKeepsExactModelAndWorkspaceIdentity)
    {
        OrchestrationConfig config;
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            for (const int layers : {1, 40, 41})
            {
                SCOPED_TRACE(::testing::Message() << deviceTypeToString(backend) << " layers=" << layers);
                const auto policy = resolveMoEOverlayCapacityAdmissionPolicy(topology(backend), config, 1, layers, 256);
                ASSERT_TRUE(policy.usesDeviceTransferDirectory());
                ASSERT_TRUE(policy.device_rebalance_workspace_capacity);
                const auto &workspace = *policy.device_rebalance_workspace_capacity;
                EXPECT_EQ(workspace.num_layers, layers);
                EXPECT_EQ(workspace.num_experts, 256u);
                EXPECT_EQ(workspace.participant_count, 2u);
                EXPECT_EQ(workspace.layer_window_count, layers > 1 ? layers - 1 : 1);
                EXPECT_EQ(workspace.local_transfer_slot_count, policy.device_transfer_directory_capacity.total_slots);
                EXPECT_EQ(workspace.phase, DeviceMoERebalanceStagePhase::PlanCopyApply);
                EXPECT_EQ(workspace.transfer_mode, DeviceMoERebalanceTransferMode::CompactTransferSlots);
                EXPECT_GT(policy.device_transfer_directory_capacity.active_slots, 0u);
                EXPECT_GT(policy.device_transfer_directory_capacity.staging_slots, 0u);
                EXPECT_EQ(policy.shadow_slots_per_endpoint_layer, 0u);
                EXPECT_EQ(policy.staging_capacity_bytes, 0u);
            }
    }

    TEST(MoEOverlayPlanningInputs, ArbitraryRankFabricPreservesIndependentConcurrencyInputs)
    {
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
            for (const unsigned slots : {1u, 2u, 8u})
                for (const unsigned cycles_per_layer : {1u, 3u, 16u})
                {
                    OrchestrationConfig config;
                    config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
                    config.moe_rebalance.migration_transfer_slots = slots;
                    config.moe_rebalance.migration_execution_streams = 2u;
                    config.moe_rebalance.dynamic_max_swaps_per_layer = cycles_per_layer;
                    const auto policy = resolveMoEOverlayCapacityAdmissionPolicy(topology(backend, true), config, 3, 41, 256);
                    ASSERT_TRUE(policy.usesPhysicalResidencyFabric());
                    EXPECT_EQ(policy.maximum_concurrent_cycles, slots);
                    EXPECT_EQ(policy.maximum_execution_streams, 2u);
                    EXPECT_EQ(policy.maximum_cycles_per_layer, cycles_per_layer);
                    EXPECT_EQ(policy.shadow_slots_per_endpoint_layer, std::min(slots, cycles_per_layer));
                    EXPECT_EQ(policy.staging_capacity_bytes, MoEOverlayCapacityAdmissionPolicy::kProductionStagingBytes);
                    EXPECT_TRUE(policy.distributed_transport);
                    EXPECT_EQ(policy.overlay_world_size, 3);
                    EXPECT_FALSE(policy.device_rebalance_workspace_capacity);
                    EXPECT_EQ(policy.device_transfer_directory_capacity.total_slots, 0u);
                }
    }

    TEST(MoEOverlayPlanningInputs, MixedGpuTierAndMultiplePrioritiesUseTheExistingFabricPolicy)
    {
        OrchestrationConfig config;
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        auto plan = topology(DeviceType::CUDA);
        plan.domains[0].participants[1] = GlobalDeviceAddress::rocm(0);
        EXPECT_TRUE(resolveMoEOverlayCapacityAdmissionPolicy(plan, config, 1, 41, 256).usesPhysicalResidencyFabric());
        plan = topology(DeviceType::ROCm);
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        auto other = plan.domains[0];
        other.name = "other";
        plan.domains.push_back(other);
        plan.routed_tiers.push_back({.name = "other-priority", .domain = other.name, .priority = 99});
        EXPECT_TRUE(resolveMoEOverlayCapacityAdmissionPolicy(plan, config, 1, 41, 256).usesPhysicalResidencyFabric());
    }

    TEST(MoEOverlayPlanningInputs, InvalidGeometryOrUnresolvedAuthorityCannotReachAdmission)
    {
        OrchestrationConfig config;
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        auto plan = topology(DeviceType::CUDA);
        for (const int invalid : {-1, 0})
        {
            EXPECT_THROW((void)resolveMoEOverlayCapacityAdmissionPolicy(plan, config, invalid, 41, 256), std::invalid_argument);
            EXPECT_THROW((void)resolveMoEOverlayCapacityAdmissionPolicy(plan, config, 1, invalid, 256), std::invalid_argument);
            EXPECT_THROW((void)resolveMoEOverlayCapacityAdmissionPolicy(plan, config, 1, 41, invalid), std::invalid_argument);
        }
        plan.authority_execution = MoEOverlayAuthorityExecutionKind::Unresolved;
        EXPECT_THROW((void)resolveMoEOverlayCapacityAdmissionPolicy(plan, config, 1, 41, 256), std::invalid_argument);
    }

    TEST(MoEOverlayMemoryPlanInputs, BackendSymmetryPreservesExactOwnersAndLimits)
    {
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        {
            MemoryInputsFixture fixture(backend);
            fixture.config.max_gpu_memory_mb = 789;
            fixture.config.max_cpu_memory_mb = 123;
            const auto result = buildMoEOverlayMemoryPlanInputs(fixture.request(), 8);
            const auto &input = result.local_capacity;
            EXPECT_EQ(input.model_profile, &fixture.model);
            EXPECT_EQ(input.rank_plan, &fixture.rank);
            EXPECT_EQ(input.overlay_plan, fixture.config.moe_routed_expert_plan.get());
            EXPECT_EQ(input.rank_inventory, &fixture.inventory.ranks[0]);
            EXPECT_EQ(input.cluster_inventory, &fixture.inventory);
            EXPECT_EQ(input.max_gpu_memory_bytes, 789u * 1024u * 1024u);
            EXPECT_EQ(input.max_cpu_memory_bytes, 123u * 1024u * 1024u);
            ASSERT_TRUE(input.gpu_weight_load);
            EXPECT_EQ(input.gpu_weight_load->maximum_source_bytes, 1234u);
            EXPECT_EQ(result.prefill_segment_rows, 8);
            EXPECT_EQ(input.resident_graph_rows, 8);
            EXPECT_EQ(input.activation_channel_row_capacity, 8);
            EXPECT_EQ(input.activation_graph_family_count, 1u);
            EXPECT_EQ(result.model_graph_identity_count, 4u);
            EXPECT_TRUE(input.captured_graph_plan.valid());
            EXPECT_EQ(input.host_demand_memory.has_value(), backend == DeviceType::CPU);
        }
    }

    TEST(MoEOverlayMemoryPlanInputs, RetainedDepthFifteenCannotBeTruncatedBySmallPrefill)
    {
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Dynamic})
        for (const bool active_mtp : {false, true})
        for (const int requests : {1, 4})
        {
            SCOPED_TRACE(::testing::Message() << deviceTypeToString(backend)
                << " active=" << active_mtp << " mode=" << static_cast<int>(mode)
                << " requests=" << requests);
            MemoryInputsFixture fixture(backend);
            fixture.rank.runtime.batch_size = requests;
            fixture.rank.runtime.mtp.enabled = active_mtp;
            fixture.rank.runtime.mtp.draft_tokens = mode == MTPDepthPolicyMode::Fixed ? 15 : 2;
            fixture.rank.runtime.mtp.graph_capacity_draft_tokens = 15;
            fixture.rank.runtime.mtp.max_request_batch = requests;
            fixture.rank.runtime.mtp.depth_policy.mode = mode;
            fixture.rank.runtime.mtp.depth_policy.max_depth = 15;
            fixture.family.mtp_source_layers = {2};
            fixture.family.max_decode_rows = resolveMTPRetainedTargetQueryRows(fixture.rank.runtime.mtp);
            fixture.family.max_request_count = requests;
            fixture.family.max_mtp_draft_depth = resolveMTPRetainedDraftCapacity(fixture.rank.runtime.mtp);
            const auto result = buildMoEOverlayMemoryPlanInputs(fixture.request(), 1);
            EXPECT_EQ(result.prefill_segment_rows, 1);
            EXPECT_EQ(result.local_capacity.resident_graph_rows, 16 * requests);
            EXPECT_EQ(result.local_capacity.activation_channel_row_capacity, fixture.family.max_decode_rows);
            EXPECT_EQ(result.local_capacity.activation_graph_family_count, 2u);
            if (result.local_capacity.host_demand_memory)
            {
                EXPECT_EQ(result.local_capacity.host_demand_memory->geometry().num_layers, 3);
                EXPECT_EQ(result.local_capacity.host_demand_memory->geometry().maximum_invocation_rows,
                          fixture.family.max_decode_rows);
            }
        }
    }

    TEST(MoEOverlayMemoryPlanInputs, ExplicitRetainedPolicyOwnsCapacityOverTheCurrentRequest)
    {
        MemoryInputsFixture fixture(DeviceType::CPU);
        MTPRuntimeConfig retained = fixture.rank.runtime.mtp;
        retained.graph_capacity_draft_tokens = 15;
        fixture.family.mtp_source_layers = {2};
        fixture.family.max_decode_rows = resolveMTPRetainedTargetQueryRows(retained);
        fixture.family.max_mtp_draft_depth = resolveMTPRetainedDraftCapacity(retained);
        const MoEOverlayMemoryPlanInputRequest request{
            .model = fixture.model, .rank_plan = fixture.rank, .config = fixture.config,
            .inventory = fixture.inventory, .execution = fixture.execution,
            .capacity_policy = fixture.policy, .retained_mtp = retained,
            .graph_family = fixture.family, .prefill = fixture.request().prefill};
        EXPECT_EQ(buildMoEOverlayMemoryPlanInputs(request, 1).local_capacity.resident_graph_rows, 16);
        // The current request is MTP-off; deriving capacity from it loses the
        // retained verifier family that a later request is entitled to reuse.
        EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(fixture.request(), 1), std::invalid_argument);
    }

    TEST(MoEOverlayMemoryPlanInputs, HostDemandUsesFlattenedBatchAndDistributedPublication)
    {
        MemoryInputsFixture fixture(DeviceType::CPU, 3);
        fixture.rank.runtime.batch_size = 4;
        fixture.family.max_request_count = 4;
        const auto result = buildMoEOverlayMemoryPlanInputs(fixture.request(), 32);
        ASSERT_TRUE(result.local_capacity.host_demand_memory);
        const auto &demand = *result.local_capacity.host_demand_memory;
        EXPECT_EQ(demand.geometry().maximum_invocation_rows, 128);
        EXPECT_EQ(demand.geometry().publication, MoEOverlayDemandPublicationScope::Distributed);
        EXPECT_GT(demand.mailboxBytes(), 0u);
        EXPECT_EQ(demand.geometry().num_layers, 2); // No retained sidecar in this request.

        fixture.config.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        EXPECT_FALSE(buildMoEOverlayMemoryPlanInputs(fixture.request(), 32).local_capacity.host_demand_memory);
    }

    TEST(MoEOverlayMemoryPlanInputs, SnapshotVariantsRemainExecutableOwnersNotCompilationUnits)
    {
        MemoryInputsFixture fixture(DeviceType::CPU);
        auto request = fixture.request();
        request.snapshot_capacity = {.per_accelerator_bytes = 4096};
        request.model_graph_topology_variant_count = 2;
        const auto result = buildMoEOverlayMemoryPlanInputs(request, 64);
        EXPECT_EQ(result.local_capacity.graph_snapshot_memory, request.snapshot_capacity);
        EXPECT_EQ(result.local_capacity.captured_graph_plan.resident_executables.model_graph_topology_variant_count, 2u);
        EXPECT_EQ(result.local_capacity.captured_graph_plan.compilation.model_graph_identity_count,
                  result.model_graph_identity_count);
        request.snapshot_capacity = {};
        EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(request, 64), std::invalid_argument);
    }

    TEST(MoEOverlayMemoryPlanInputs, IncompleteRankFamilyCacheOrCapacityFailsBeforeAdmission)
    {
        MemoryInputsFixture fixture(DeviceType::CUDA);
        EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(fixture.request(), 0), std::invalid_argument);
        auto request = fixture.request();
        request.prefill.maximum_cached_buckets = 2;
        EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(request, 8), std::invalid_argument);
        auto empty_ladder = fixture.request();
        empty_ladder.prefill.bucket_rows = {};
        EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(empty_ladder, 8), std::invalid_argument);
        fixture.execution.current_rank.world_rank = 1;
        EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(fixture.request(), 8), std::invalid_argument);
        fixture.execution.current_rank.world_rank = 0;
        fixture.family.max_decode_rows = 16;
        EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(fixture.request(), 8), std::invalid_argument);
        fixture.family.max_decode_rows = 1;
        fixture.config.max_gpu_memory_mb = std::numeric_limits<std::size_t>::max();
        EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(fixture.request(), 8), std::overflow_error);
    }
}
