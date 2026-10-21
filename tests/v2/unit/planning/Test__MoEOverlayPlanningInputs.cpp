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
#include "../models/qwen4exp/Qwen4ExpSidecarFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "loaders/ModelLoader.h"
#include "planning/PlanningModelMetadata.h"
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
                model.mtp_layer_count = 1;
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
                    *config.moe_routed_expert_plan, config, world_size, 2, 8);
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

            /**
             * @brief Project one explicit child while retaining full parent metadata.
             * @param first First owned global main layer.
             * @param last Exclusive owned main end; 64 owns the terminal head.
             * @param retain_mtp Whether this graph family retains the learned predictor.
             * @return Exact boundary passed to the same production memory adapter.
             */
            FactoryPPStageConfig stage(int first, int last, bool retain_mtp)
            {
                model.n_layers = 65;
                rank.first_layer = first;
                rank.last_layer = last - 1;
                rank.has_embedding = first == 0;
                rank.has_lm_head = last == 64;
                rank.runtime.mtp.enabled = retain_mtp;
                config.moe_routed_expert_plan->first_model_layer = first;
                family.first_model_layer = first;
                family.main_layer_count = last - first;
                family.mtp_source_layers = retain_mtp && rank.has_lm_head ? std::vector<int>{64} : std::vector<int>{};
                family.max_decode_rows = std::max(1, resolveMTPRetainedTargetQueryRows(rank.runtime.mtp));
                family.max_mtp_draft_depth = resolveMTPRetainedDraftCapacity(rank.runtime.mtp);
                policy = resolveMoEOverlayCapacityAdmissionPolicy(*config.moe_routed_expert_plan,
                    config, inventory.world_size, family.routedLayerCapacity(), model.expert_count);
                return {.first_layer = first, .last_layer = last,
                    .has_embedding = rank.has_embedding, .has_lm_head = rank.has_lm_head};
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
            fixture.policy = resolveMoEOverlayCapacityAdmissionPolicy(*fixture.config.moe_routed_expert_plan,
                fixture.config, fixture.inventory.world_size, fixture.family.routedLayerCapacity(), fixture.model.expert_count);
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

    /** @test Movable host authorities still require evidence; device owners forbid its mirror. */
    TEST(MoEOverlayMemoryPlanInputs, LocalCapacityDemandValidationPreservesMovableAuthority)
    {
        MemoryInputsFixture host(DeviceType::CPU, 2);
        host.rank.runtime.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        auto host_input = buildMoEOverlayMemoryPlanInputs(host.request(), 32).local_capacity;
        ASSERT_TRUE(host_input.host_demand_memory);
        const auto evidence = host_input.host_demand_memory;
        host_input.host_demand_memory.reset();
        try
        {
            (void)MoEOverlayLocalCapacityPlanner::plan(host_input);
            FAIL() << "Movable host capacity accepted missing routing evidence";
        }
        catch (const std::invalid_argument &error)
        {
            EXPECT_NE(std::string(error.what()).find("requires exactly its routing-evidence BOM"), std::string::npos);
        }

        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        {
            MemoryInputsFixture device(backend);
            device.rank.runtime.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
            auto input = buildMoEOverlayMemoryPlanInputs(device.request(), 32).local_capacity;
            ASSERT_FALSE(input.host_demand_memory);
            input.host_demand_memory = evidence;
            EXPECT_THROW((void)MoEOverlayLocalCapacityPlanner::plan(input), std::invalid_argument);
        }
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

    /** @test Main capture frontiers exclude appended predictor blocks. */
    TEST(MoEOverlayMemoryPlanInputs, PipelineCaptureUsesOwnedMainLayers)
    {
        MemoryInputsFixture fixture(DeviceType::CPU);
        const auto result = buildMoEOverlayMemoryPlanInputs(fixture.request(), 8);
        EXPECT_EQ(result.local_capacity.captured_graph_plan.compilation.compilation_units_per_model_graph,
                  size_t(fixture.family.main_layer_count + 1));
        EXPECT_EQ(result.local_capacity.graph_family.routedLayerCapacity(), 2);
    }

    /** @test All backends retain global origins and compact row counts without truncating metadata. */
    TEST(MoEOverlayMemoryPlanInputs, PipelineScopePreservesOwnedMainAndRoutedSidecars)
    {
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
            for (const int first : {0, 32, 40})
                for (const bool retained : {false, true})
                    for (const auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Dynamic})
                    {
                        SCOPED_TRACE(::testing::Message() << deviceTypeToString(backend) << '/' << first << '/' << retained << '/' << int(mode));
                        MemoryInputsFixture fixture(backend);
                        fixture.rank.runtime.mtp.depth_policy.mode = mode;
                        const auto scope = fixture.stage(first, first == 0 ? 32 : 64, retained);
                        auto request = fixture.request();
                        request.pipeline_stage = scope;
                        const auto result = buildMoEOverlayMemoryPlanInputs(request, 8);
                        const auto &input = result.local_capacity;
                        EXPECT_EQ(input.model_profile, &fixture.model);
                        EXPECT_EQ(input.model_profile->n_layers, 65);
                        EXPECT_EQ(input.graph_family.first_model_layer, first);
                        EXPECT_EQ(input.graph_family.main_layer_count, scope.layerCount());
                        EXPECT_EQ(input.graph_family.routedLayerCapacity(), scope.layerCount() + (retained && scope.has_lm_head ? 1 : 0));
                        EXPECT_EQ(input.graph_family.mtp_source_layers, fixture.family.mtp_source_layers);
                        EXPECT_EQ(input.captured_graph_plan.compilation.compilation_units_per_model_graph,
                            backend == DeviceType::CPU ? size_t(scope.layerCount() + 1) : 1u);
                        if (input.host_demand_memory)
                            EXPECT_EQ(input.host_demand_memory->geometry().num_layers, input.graph_family.routedLayerCapacity());
                    }
    }

    /** @test Foreign origins, missing boundaries and nonterminal predictors fail before admission. */
    TEST(MoEOverlayMemoryPlanInputs, PipelineScopeRejectsForeignOrIncompleteOwnership)
    {
        for (int mutation = 0; mutation < 15; ++mutation)
        {
            SCOPED_TRACE(mutation);
            MemoryInputsFixture fixture(DeviceType::ROCm);
            const auto scope = fixture.stage(32, 64, true);
            auto request = fixture.request();
            request.pipeline_stage = scope;
            switch (mutation)
            {
                case 0: request.pipeline_stage.reset(); break;
                case 1: request.pipeline_stage->first_layer = 31; break;
                case 2: request.pipeline_stage->last_layer = 63; break;
                case 3: request.pipeline_stage->has_embedding = true; break;
                case 4: request.pipeline_stage->has_lm_head = false; break;
                case 5: fixture.rank.first_layer = 0; break;
                case 6: fixture.rank.has_lm_head = false; break;
                case 7: fixture.family.first_model_layer = 0; break;
                case 8: fixture.config.moe_routed_expert_plan->first_model_layer = 0; break;
                case 9: fixture.family.mtp_source_layers = {65}; break;
                case 10: fixture.rank.next_rank = 1; break;
                case 11:
                    request.pipeline_stage = fixture.stage(0, 32, true);
                    fixture.family.mtp_source_layers = {64};
                    break;
                case 12:
                    request.pipeline_stage = fixture.stage(0, 32, false);
                    request.pipeline_stage.reset();
                    break;
                case 13: fixture.policy.device_rebalance_workspace_capacity->num_layers = 65; break;
                case 14: fixture.policy.device_rebalance_workspace_capacity->num_experts = 9; break;
            }
            EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(request, 8), std::invalid_argument);
        }
    }
    /** @test Main-only block_count and inline block_count retain the same exact routed sidecar owners. */
    TEST(MoEOverlayMemoryPlanInputs, SourceCountContractPreservesSeparateAndInlinePredictors)
    {
        const test::Qwen4ExpSidecarFixture source;
        const auto separate = ModelMemoryProfile::fromGGUF(source.main, source.predictor,
            SharedPredictorMaterialization::Active);
        ASSERT_EQ(separate.mainLayerCount(), 48);
        ASSERT_EQ(separate.totalLayerCount(), 49);
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
            for (const bool separate_source : {false, true})
                for (const bool retained : {false, true})
                {
                    SCOPED_TRACE(::testing::Message() << int(backend) << '/' << separate_source << '/' << retained);
                    MemoryInputsFixture fixture(backend);
                    fixture.model.n_layers = separate_source ? 48 : 49;
                    fixture.model.mtp_layer_count = 1;
                    if (separate_source) fixture.model = separate;
                    fixture.rank.runtime.mtp.enabled = retained;
                    fixture.family.main_layer_count = 48;
                    fixture.family.mtp_source_layers = retained ? std::vector<int>{48} : std::vector<int>{};
                    fixture.family.max_decode_rows = std::max(1, resolveMTPRetainedTargetQueryRows(fixture.rank.runtime.mtp));
                    fixture.family.max_mtp_draft_depth = resolveMTPRetainedDraftCapacity(fixture.rank.runtime.mtp);
                    fixture.policy = resolveMoEOverlayCapacityAdmissionPolicy(*fixture.config.moe_routed_expert_plan,
                        fixture.config, 1, fixture.family.routedLayerCapacity(), fixture.model.expert_count);
                    const auto result = buildMoEOverlayMemoryPlanInputs(fixture.request(), 8);
                    EXPECT_EQ(result.local_capacity.model_profile, &fixture.model);
                    EXPECT_EQ(result.local_capacity.model_profile->n_layers, separate_source ? 48 : 49);
                    EXPECT_EQ(result.local_capacity.graph_family.main_layer_count, 48);
                    EXPECT_EQ(result.local_capacity.graph_family.routedLayerCapacity(), retained ? 49 : 48);
                    EXPECT_EQ(result.local_capacity.graph_family.mtp_source_layers, fixture.family.mtp_source_layers);
                    EXPECT_EQ(result.local_capacity.captured_graph_plan.compilation.compilation_units_per_model_graph,
                        backend == DeviceType::CPU ? 49u : 1u);
                }
    }

    /** @test An authored PP terminal scope owns source48; foreign main/source bounds still fail. */
    TEST(MoEOverlayMemoryPlanInputs, SeparatePredictorPipelineUsesCanonicalMainAndTotalBounds)
    {
        const test::Qwen4ExpSidecarFixture source;
        const auto separate = ModelMemoryProfile::fromGGUF(source.main, source.predictor,
            SharedPredictorMaterialization::Active);
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
            for (int mutation = -1; mutation < 7; ++mutation)
            {
                SCOPED_TRACE(::testing::Message() << int(backend) << '/' << mutation);
                MemoryInputsFixture fixture(backend);
                fixture.model = separate;
                fixture.rank.first_layer = 20;
                fixture.rank.last_layer = 47;
                fixture.rank.has_embedding = false;
                fixture.rank.has_lm_head = true;
                fixture.rank.runtime.mtp.enabled = true;
                fixture.config.moe_routed_expert_plan->first_model_layer = 20;
                fixture.family.first_model_layer = 20;
                fixture.family.main_layer_count = 28;
                fixture.family.mtp_source_layers = {48};
                fixture.family.max_decode_rows = resolveMTPRetainedTargetQueryRows(fixture.rank.runtime.mtp);
                fixture.family.max_mtp_draft_depth = resolveMTPRetainedDraftCapacity(fixture.rank.runtime.mtp);
                fixture.policy = resolveMoEOverlayCapacityAdmissionPolicy(*fixture.config.moe_routed_expert_plan,
                    fixture.config, 1, 29, fixture.model.expert_count);
                auto request = fixture.request();
                request.pipeline_stage = FactoryPPStageConfig{.first_layer = 20, .last_layer = 48,
                    .has_embedding = false, .has_lm_head = true};
                switch (mutation)
                {
                case 0: fixture.family.mtp_source_layers = {49}; break;
                case 1: request.pipeline_stage->last_layer = 47; break;
                case 2: fixture.rank.last_layer = 48; break;
                case 3: fixture.family.main_layer_count = 29; break;
                case 4: fixture.family.first_model_layer = 48; break;
                case 5: fixture.model.mtp_layer_count = -1; break;
                case 6: fixture.model.n_layers = std::numeric_limits<int>::max(); break;
                default: break;
                }
                if (mutation == -1)
                {
                    const auto result = buildMoEOverlayMemoryPlanInputs(request, 8);
                    EXPECT_EQ(result.local_capacity.graph_family.first_model_layer, 20);
                    EXPECT_EQ(result.local_capacity.graph_family.main_layer_count, 28);
                    EXPECT_EQ(result.local_capacity.graph_family.routedLayerCapacity(), 29);
                    EXPECT_EQ(result.local_capacity.graph_family.mtp_source_layers, (std::vector<int>{48}));
                }
                else if (mutation == 6)
                    EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(request, 8), std::overflow_error);
                else
                    EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(request, 8), std::invalid_argument);
            }
    }

    /** @test Fixed-owner admission has an explicit global zero-residency entry for source48. */
    TEST(MoEOverlayMemoryPlanInputs, SeparatePredictorFixedOwnersIncludeOriginalRoutedSourceIndex)
    {
        test::Qwen4ExpSidecarFixture source;
        // This bounded terminal fixture executes main47 plus predictor48. Add
        // the real main expert-parent geometry needed by the CPU workspace
        // probe, without expanding any native payload or unrelated layers.
        for (const char *role : {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"})
        {
            const bool down = std::string_view(role) == "ffn_down_exps";
            GGUFTensorInfo tensor{.name = std::string("blk.47.") + role + ".weight",
                .dimensions = down ? std::vector<uint64_t>{640, 2560, 512}
                                   : std::vector<uint64_t>{2560, 640, 512},
                .type = GGUFTensorType::Q8_0};
            test::Qwen4ExpSidecarFixture::sealBytes(tensor);
            source.main.tensors.push_back(std::move(tensor));
        }
        MemoryInputsFixture fixture(DeviceType::CPU);
        fixture.model = ModelMemoryProfile::fromGGUF(source.main, source.predictor,
            SharedPredictorMaterialization::Active);
        fixture.rank.first_layer = fixture.rank.last_layer = 47;
        fixture.rank.has_embedding = false;
        fixture.rank.has_lm_head = true;
        fixture.rank.primary_device = GlobalDeviceAddress::cpu();
        fixture.rank.runtime.max_seq_len = 32;
        fixture.rank.runtime.prefix_cache.enabled = false;
        fixture.rank.runtime.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
        fixture.rank.runtime.mtp.enabled = true;
        auto &overlay = *fixture.config.moe_routed_expert_plan;
        overlay.first_model_layer = 47;
        overlay.domains.front().scope = ExecutionDomainScope::SINGLE;
        overlay.domains.front().participants = {GlobalDeviceAddress::cpu()};
        fixture.execution = resolveMoEExpertOverlayExecutionPlan(fixture.config.moe_routed_expert_plan,
            {.current_world_rank = 0, .world_size = 1});
        fixture.family.first_model_layer = 47;
        fixture.family.main_layer_count = 1;
        fixture.family.mtp_source_layers = {48};
        fixture.family.max_decode_rows = resolveMTPRetainedTargetQueryRows(fixture.rank.runtime.mtp);
        fixture.family.max_mtp_draft_depth = resolveMTPRetainedDraftCapacity(fixture.rank.runtime.mtp);
        fixture.policy = resolveMoEOverlayCapacityAdmissionPolicy(overlay, fixture.config, 1, 2, fixture.model.expert_count);
        auto &observed = fixture.inventory.ranks.front();
        observed.cpu_cores = observed.cpu_worker_threads = 8;
        observed.cpu_execution = test::kSyntheticCPUExecutionGeometry;
        observed.cpu.memory_bytes = observed.cpu.free_memory_bytes = observed.cpu_memory_bytes = 32ull << 30;
        auto request = fixture.request();
        request.pipeline_stage = FactoryPPStageConfig{.first_layer = 47, .last_layer = 48,
            .has_embedding = false, .has_lm_head = true};
        const auto inputs = buildMoEOverlayMemoryPlanInputs(request, 8);
        const auto fixed = MoEOverlayLocalCapacityPlanner::plan(inputs.local_capacity);
        ASSERT_EQ(fixed.device_inputs.size(), 1u);
        const auto &residency = fixed.device_inputs.front().weight_residency;
        // The fixed contribution excludes every routed expert while retaining
        // its source-indexed owner slot. The quota phase alone charges experts.
        EXPECT_EQ(residency.selectedRoutedExpertsForLayer(47), 0);
        EXPECT_EQ(residency.selectedRoutedExpertsForLayer(48), 0);
        EXPECT_THROW((void)residency.selectedRoutedExpertsForLayer(49), std::out_of_range);
        EXPECT_EQ(fixed.device_inputs.front().last_layer, 47);
        EXPECT_TRUE(fixed.device_inputs.front().mtp_enabled);
        EXPECT_FALSE(fixed.host_demand_memory);
    }

    /** @test Complete source-owned fixed BOMs retain production prefix RAM/disk and dense predictor KV. */
    TEST(MoEOverlayMemoryPlanInputs, CompleteQwenSourcePrefixEnabledFixedBOMWholeAndPipeline)
    {
        test::Qwen4ExpPlanningGGUFFixture file;
        ModelLoader loader;
        ASSERT_TRUE(loader.loadModel(file.path()));
        test::Qwen4ExpSidecarFixture source;
        for (const auto &tensor : loader.getModel().tensors)
            if (!source.main.findTensor(tensor.name)) source.main.tensors.push_back(tensor);
        const auto profile = ModelMemoryProfile::fromGGUF(source.main,source.predictor,
            SharedPredictorMaterialization::Active);
        ASSERT_EQ(profile.mainLayerCount(),48);
        ASSERT_EQ(profile.totalLayerCount(),49);
        ASSERT_EQ(profile.flashNextSparseLayerCount(48,48),0u);
        for (const auto backend : {DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm})
            for (const auto interval : {std::pair{0,47},std::pair{0,7},std::pair{8,19},std::pair{20,47},
                                       std::pair{0,0},std::pair{1,1},std::pair{2,47}})
                for (const auto storage : {ImmutableRowStoragePreference::RAM,ImmutableRowStoragePreference::MemoryMapped})
                    for (const auto mode : {MoERebalanceRuntimeMode::Off,MoERebalanceRuntimeMode::Dynamic})
                    {
                        SCOPED_TRACE(::testing::Message()<<int(backend)<<'/'<<interval.first<<'/'<<interval.second<<'/'<<int(storage)<<'/'<<int(mode));
                        MemoryInputsFixture f(backend);
                        f.model=profile;
                        f.rank.first_layer=interval.first;
                        f.rank.last_layer=interval.second;
                        f.rank.has_embedding=interval.first==0;
                        f.rank.has_lm_head=interval.second==47;
                        f.rank.primary_device=GlobalDeviceAddress::fromLocalDeviceId(DeviceId{backend,0});
                        f.buckets={1,8,32,64,256,512};
                        f.rank.runtime.batch_size=1;
                        f.rank.runtime.moe_routed_prefill.overlay_segment_rows=512;
                        f.rank.runtime.max_seq_len=4096;
                        f.rank.runtime.kv_cache_precision=KVCachePrecision::FP16;
                        f.rank.runtime.mtp.enabled=true;
                        f.rank.runtime.mtp.draft_tokens=1;
                        f.rank.runtime.mtp.graph_capacity_draft_tokens=1;
                        f.rank.runtime.mtp.max_request_batch=1;
                        f.rank.runtime.ngram_table_policy.storage=storage;
                        f.rank.runtime.moe_rebalance.mode=mode;
                        auto &prefix=f.rank.runtime.prefix_cache;
                        prefix.enabled=true;
                        prefix.storage_mode=PrefixCacheStorageMode::Tiered;
                        prefix.block_size=64;
                        prefix.ram_budget_bytes=256ull<<20;
                        prefix.device_budget_bytes=512ull<<20;
                        prefix.disk_budget_bytes=1ull<<30;
                        prefix.disk_dir="/tmp/qfn-prefix-bom-metadata-only";
                        auto &overlay=*f.config.moe_routed_expert_plan;
                        overlay.first_model_layer=interval.first;
                        overlay.authority_execution=MoEOverlayAuthorityExecutionKind::HostResident;
                        overlay.domains.front().scope=ExecutionDomainScope::SINGLE;
                        overlay.domains.front().participants={f.rank.primary_device};
                        if(backend!=DeviceType::CPU)
                        {
                            overlay.topology=RoutedExpertPlacementTopology::TieredOverlay;
                            auto host=overlay.domains.front();
                            host.name="host-capacity";
                            host.backend=CollectiveBackendType::MPI;
                            host.participants={GlobalDeviceAddress::cpu()};
                            overlay.domains.push_back(host);
                            overlay.routed_tiers.push_back({.name="host-owners",.domain=host.name,.priority=38,.fallback=true});
                            overlay.routed_tiers.front().fallback=false;
                        }
                        f.config.moe_rebalance.mode=mode;
                        f.execution=resolveMoEExpertOverlayExecutionPlan(f.config.moe_routed_expert_plan,
                            {.current_world_rank=0,.world_size=1});
                        f.family.max_graph_rows=512;
                        f.family.first_model_layer=interval.first;
                        f.family.main_layer_count=interval.second-interval.first+1;
                        f.family.mtp_source_layers=f.rank.has_lm_head?std::vector<int>{48}:std::vector<int>{};
                        f.family.max_decode_rows=resolveMTPRetainedTargetQueryRows(f.rank.runtime.mtp);
                        f.family.max_mtp_draft_depth=resolveMTPRetainedDraftCapacity(f.rank.runtime.mtp);
                        f.policy=resolveMoEOverlayCapacityAdmissionPolicy(overlay,f.config,1,
                            f.family.routedLayerCapacity(),profile.expert_count);
                        auto &observed=f.inventory.ranks.front();
                        observed.cpu_cores=observed.cpu_worker_threads=8;
                        observed.cpu_execution=test::kSyntheticCPUExecutionGeometry;
                        observed.cpu.memory_bytes=observed.cpu.free_memory_bytes=observed.cpu_memory_bytes=256ull<<30;
                        if(backend!=DeviceType::CPU) observed.gpus.push_back({.type=backend,.local_device_id=0,
                            .memory_bytes=256ull<<30,.free_memory_bytes=256ull<<30,.compute_units=60});
                        auto request=f.request();
                        request.gpu_weight_load={.policy=GPUWeightLoadMemoryPolicy{},
                            .maximum_source_bytes=maximumGGUFTensorPayloadBytes(source.main)};
                        if(interval!=std::pair{0,47}) request.pipeline_stage=FactoryPPStageConfig{
                            .first_layer=interval.first,.last_layer=interval.second+1,
                            .has_embedding=f.rank.has_embedding,.has_lm_head=f.rank.has_lm_head};
                        const auto inputs=buildMoEOverlayMemoryPlanInputs(request,512);
                        const auto fixed=MoEOverlayLocalCapacityPlanner::plan(inputs.local_capacity);
                        ASSERT_NE(fixed.physical_memory_admission,nullptr);
                        const auto exact=MemoryPlanner::plan(profile,fixed.device_inputs);
                        size_t host_tier=0,archive=0;
                        for(const auto &resource:fixed.physical_memory_admission->plan().resources())
                        {
                            const auto *bill=exact.physicalPlan().find({resource.resource().world_rank,resource.resource().device});
                            ASSERT_NE(bill,nullptr);
                            for(const auto owner:{PhysicalMemoryOwner::PrefixHostTier,PhysicalMemoryOwner::PrefixDeviceTier,
                                PhysicalMemoryOwner::PrefixArchiveStaging}) EXPECT_EQ(resource.bytes(owner),bill->bytes(owner));
                            host_tier+=resource.bytes(PhysicalMemoryOwner::PrefixHostTier);
                            archive+=resource.bytes(PhysicalMemoryOwner::PrefixArchiveStaging);
                        }
                        EXPECT_GT(host_tier,0u);
                        EXPECT_GT(archive,0u);
                        const auto continuation=std::find_if(fixed.device_inputs.begin(),fixed.device_inputs.end(),
                            [](const auto &input){return input.execution_role==DeviceExecutionMemoryRole::ContinuationGraph;});
                        ASSERT_NE(continuation,fixed.device_inputs.end());
                        EXPECT_TRUE(continuation->prefix_cache.enabled);
                        EXPECT_EQ(continuation->prefix_cache.storage_mode,PrefixCacheStorageMode::Tiered);
                        EXPECT_EQ(continuation->prefix_cache.disk_budget_bytes,1ull<<30);
                        if(f.rank.has_lm_head) EXPECT_EQ(continuation->weight_residency.selectedRoutedExpertsForLayer(48),0);
                    }
    }

}
