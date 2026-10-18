/**
 * @file Test__MoEOverlayPipelineStageBinding.cpp
 * @brief Device-free regressions for the production PP-to-child expert handoff.
 *
 * Real residency authorities and endpoint registries carry tiny model metadata;
 * no device or model payload is loaded. Tests call the same RankOrchestrator
 * projection used by both nested TP and single-device PP construction, including
 * foreign scopes, endpoint substitutions, mode changes and terminal MTP banks.
 */
#include "execution/moe/MoEOverlayPipelineStageBinding.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/moe/MoEOverlayRankBatchTransport.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using Config = RankOrchestrator::Config;
        using Stage = RankOrchestrator::PPStageConfig;
        using Binding = MoEOverlayPipelineStageBinding;

        /** @brief Real, compact authority and empty prepared endpoints for one stage. */
        struct StageFixture
        {
            FactoryPPStageConfig scope;
            MoERoutedExpertModelMetadata metadata;
            std::vector<GlobalDeviceAddress> devices;
            CollectiveBackendType backend;
            Binding::Owners owners;

            /**
             * @brief Construct stage geometry without touching an accelerator.
             * @param first Global origin of two owned main layers.
             * @param terminal Whether the stage owns the model's output boundary.
             * @param auxiliary Retained routed NextN banks, only valid at the terminal.
             * @param type Device family used as immutable topology metadata.
             * @param width Single-device PP or multi-device local TP.
             * @param mode Off, Observe and Dynamic retain distinct explicit policies.
             */
            StageFixture(int first, bool terminal, int auxiliary, DeviceType type,
                         int width, MoERebalanceRuntimeMode mode)
                : scope{.first_layer = first, .last_layer = first + 2,
                        .has_embedding = first == 0, .has_lm_head = terminal},
                  backend(type == DeviceType::CUDA ? CollectiveBackendType::NCCL :
                          type == DeviceType::ROCm ? CollectiveBackendType::RCCL : CollectiveBackendType::HOST)
            {
                for (int ordinal = 0; ordinal < width; ++ordinal)
                    devices.push_back(type == DeviceType::CUDA ? GlobalDeviceAddress::cuda(ordinal, 0) :
                        type == DeviceType::ROCm ? GlobalDeviceAddress::rocm(ordinal, 0) :
                        GlobalDeviceAddress::cpu(ordinal));
                metadata.num_layers = 2 + auxiliary;
                metadata.num_experts = 4;
                metadata.d_model = 8;
                metadata.routed_intermediate_size = 16;
                metadata.first_model_layer = first;
                metadata.main_inference_layer_count = first + 2;
                MoERoutedExpertPlacementPlan plan;
                plan.enabled = true;
                plan.topology = RoutedExpertPlacementTopology::SingleDomain;
                plan.continuation_domain = "stage";
                plan.base_model_domain = "stage";
                plan.shared_expert_domain = "stage";
                plan.first_model_layer = first;
                plan.residency_policy = defaultRoutedExpertResidencyPolicy(mode);
                RoutedExpertDomain domain;
                domain.name = "stage";
                domain.scope = width == 1 ? ExecutionDomainScope::SINGLE : ExecutionDomainScope::RANK_LOCAL;
                domain.backend = backend;
                domain.participants = devices;
                domain.owner_rank = 0;
                domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
                plan.domains.push_back(domain);
                RoutedExpertTier tier;
                tier.name = "all";
                tier.domain = "stage";
                tier.priority = 0;
                tier.fallback = true;
                plan.routed_tiers.push_back(tier);
                plan.continuation_domain_spec.domain = "stage";
                plan.continuation_domain_spec.setDensePolicy(
                    defaultMoEContinuationDensePolicy(domain.scope, devices.size()));
                plan.authority_execution = resolveMoEOverlayAuthorityExecutionKind(plan);
                if (mode != MoERebalanceRuntimeMode::Off)
                {
                    DecodeExpertHistogramConfig histogram;
                    histogram.num_layers = metadata.num_layers;
                    histogram.num_experts = metadata.num_experts;
                    histogram.top_k = 1;
                    histogram.window_size = 4;
                    histogram.token_boundary_layer_idx = scope.last_layer - 1;
                    std::vector<int> expert_owners(4);
                    for (int expert = 0; expert < 4; ++expert) expert_owners[expert] = expert % width;
                    histogram.ownership = MoELayeredExpertOwnership::uniform(
                        metadata.num_layers, width, expert_owners, first);
                    for (const auto &device : devices) histogram.sockets.push_back(device.toLocalDeviceId());
                    owners.histogram = std::make_shared<DecodeExpertHistogram>(std::move(histogram));
                }
                owners.authority = std::make_shared<MoEOverlayResidencyAuthority>(
                    MoEOverlayResidencyAuthority::Config{.initial_plan = plan,
                        .model_metadata = metadata, .maintenance_mode = mode,
                        .histogram = owners.histogram.get()});
                const auto snapshot = owners.authority->snapshot();
                std::vector<int> local_ids(width);
                std::iota(local_ids.begin(), local_ids.end(), 0);
                owners.residency = std::make_shared<MoEOverlayParticipantResidencyRegistry>(
                    MoEOverlayParticipantResidencyRegistry::Config{
                        .owner_map = snapshot->owner_map, .local_participant_ids = local_ids,
                        .num_layers = metadata.num_layers, .num_experts = metadata.num_experts,
                        .initial_epoch = snapshot->epoch, .first_model_layer = first});
            }

            /** @return Production-checked runtime handoff retaining this fixture's owners. */
            std::shared_ptr<const Binding> seal() const
            { return Binding::seal(scope, metadata, devices, owners); }

            /** @return Authored PP stage with an optional completed runtime installation. */
            Stage stage(bool install = true) const
            {
                Stage stage{.first_layer = scope.first_layer, .last_layer = scope.last_layer,
                    .has_embedding = scope.has_embedding, .has_lm_head = scope.has_lm_head,
                    .stage_devices = devices, .tp_backend = backend};
                if (install) stage.moe_runtime = seal();
                return stage;
            }
        };

        /** @return Dense two-stage policy used to detect accidental inheritance. */
        Config densePipeline()
        {
            Config result;
            result.mode = RankOrchestrator::ParallelismMode::PP;
            result.max_seq_len = 262144;
            result.resident_graph_rows = 128;
            result.batch_size = 2;
            result.pp_stages = {
                {.first_layer = 0, .last_layer = 2, .has_embedding = true,
                 .stage_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                 .tp_backend = CollectiveBackendType::RCCL},
                {.first_layer = 2, .last_layer = 4, .has_lm_head = true,
                 .stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                 .tp_backend = CollectiveBackendType::NCCL}};
            return result;
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RetainsExactScopeAcrossDevicesWidthsAndModes)
    {
        for (const auto type : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const int width : {1, 2, 4, 8})
        for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe,
                               MoERebalanceRuntimeMode::Dynamic})
        for (const int first : {0, 32, 40})
        for (const bool terminal : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "type=" << static_cast<int>(type) << " width=" << width
                << " mode=" << static_cast<int>(mode) << " first=" << first << " terminal=" << terminal);
            StageFixture fixture(first, terminal, terminal ? 1 : 0, type, width, mode);
            const auto binding = fixture.seal();
            ASSERT_TRUE(binding);
            EXPECT_EQ(binding->plan()->first_model_layer, first);
            EXPECT_EQ(binding->owners().authority, fixture.owners.authority);
            EXPECT_EQ(binding->owners().residency, fixture.owners.residency);
            EXPECT_EQ(binding->owners().histogram, fixture.owners.histogram);
            EXPECT_NO_THROW(binding->requireDestination(fixture.scope, fixture.devices, {}, fixture.backend, mode));
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RejectsForeignOrMissingRuntimeOwners)
    {
        for (int mutation = 0; mutation < 8; ++mutation)
        {
            SCOPED_TRACE(mutation);
            StageFixture fixture(32, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Dynamic);
            StageFixture other(0, false, 0, DeviceType::ROCm, 2, MoERebalanceRuntimeMode::Dynamic);
            auto owners = fixture.owners;
            switch (mutation)
            {
            case 0: owners.authority.reset(); break;
            case 1: owners.residency.reset(); break;
            case 2: owners.histogram.reset(); break;
            case 3: owners.authority = other.owners.authority; break;
            case 4: owners.residency = other.owners.residency; break;
            case 5: owners.histogram = other.owners.histogram; break;
            case 6: {
                StageFixture foreign_device(32, true, 1, DeviceType::ROCm, 2, MoERebalanceRuntimeMode::Dynamic);
                owners.residency = foreign_device.owners.residency; break;
            }
            case 7: {
                StageFixture wrong_width(32, true, 1, DeviceType::CUDA, 1, MoERebalanceRuntimeMode::Dynamic);
                owners.residency = wrong_width.owners.residency; break;
            }
            }
            EXPECT_ANY_THROW((void)Binding::seal(fixture.scope, fixture.metadata, fixture.devices, owners));
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RejectsForeignGeometryAndNonterminalMTP)
    {
        for (int mutation = 0; mutation < 8; ++mutation)
        {
            SCOPED_TRACE(mutation);
            StageFixture fixture(32, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Off);
            auto metadata = fixture.metadata;
            auto scope = fixture.scope;
            switch (mutation)
            {
            case 0: metadata.first_model_layer = 0; break;
            case 1: metadata.num_layers = 1; break;
            case 2: metadata.num_experts = 5; break;
            case 3: metadata.main_inference_layer_count = 33; break;
            case 4: metadata.num_layers = std::numeric_limits<int>::max(); break;
            case 5: scope.has_lm_head = false; break;
            case 6: scope.has_embedding = true; break;
            case 7: scope.first_layer = -1; break;
            }
            EXPECT_ANY_THROW((void)Binding::seal(scope, metadata, fixture.devices, fixture.owners));
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RejectsDifferentOwnerMapWithIdenticalEndpoints)
    {
        StageFixture fixture(32, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Off);
        auto owners = fixture.owners;
        const auto snapshot = owners.authority->snapshot();
        auto different = snapshot->layered_ownership;
        for (int layer = 32; layer < 35; ++layer)
            for (int expert = 0; expert < 4; ++expert)
                different.assignOwner(layer, expert, 1 - different.owner(layer, expert));
        owners.residency = std::make_shared<MoEOverlayParticipantResidencyRegistry>(
            MoEOverlayParticipantResidencyRegistry::Config{
                .owner_map = MoEExpertOwnerMap::buildExplicit(*snapshot->placement_plan, different),
                .local_participant_ids = {0, 1}, .num_layers = 3, .num_experts = 4,
                .initial_epoch = snapshot->epoch, .first_model_layer = 32});
        EXPECT_THROW((void)Binding::seal(fixture.scope, fixture.metadata, fixture.devices, owners),
                     std::invalid_argument);
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RejectsChangedDestinationAndTPDeclaration)
    {
        StageFixture fixture(32, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Dynamic);
        const auto binding = fixture.seal();
        for (int mutation = 0; mutation < 8; ++mutation)
        {
            SCOPED_TRACE(mutation);
            auto scope = fixture.scope;
            auto devices = fixture.devices;
            auto backend = fixture.backend;
            auto mode = MoERebalanceRuntimeMode::Dynamic;
            std::vector<float> weights;
            switch (mutation)
            {
            case 0: scope.first_layer = 0; break;
            case 1: scope.last_layer = 33; break;
            case 2: scope.has_lm_head = false; break;
            case 3: std::reverse(devices.begin(), devices.end()); break;
            case 4: devices.pop_back(); break;
            case 5: backend = CollectiveBackendType::HOST; break;
            case 6: weights = {0.25f, 0.75f}; break;
            case 7: mode = MoERebalanceRuntimeMode::Off; break;
            }
            EXPECT_THROW(binding->requireDestination(scope, devices, weights, backend, mode), std::invalid_argument);
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, ChildProjectionPreservesDenseRuntimeAndAuthoredTopology)
    {
        auto parent = densePipeline();
        ASSERT_TRUE(parent.validate());
        for (std::size_t index = 0; index < 2; ++index)
        {
            const auto child = parent.forPipelineStage(index);
            EXPECT_TRUE(child.validate());
            EXPECT_TRUE(child.pp_stages.empty());
            EXPECT_EQ(child.devices, parent.pp_stages[index].stage_devices);
            EXPECT_EQ(child.backend, parent.pp_stages[index].tp_backend);
            EXPECT_EQ(child.max_seq_len, 262144u);
            EXPECT_EQ(child.resident_graph_rows, 128);
            EXPECT_EQ(child.batch_size, 2);
            ASSERT_TRUE(child.nested_pp_stage_config);
            EXPECT_EQ(child.nested_pp_stage_config->first_layer, static_cast<int>(index * 2));
            EXPECT_EQ(child.nested_pp_stage_config->has_lm_head, index == 1);
            EXPECT_FALSE(child.moe_routed_expert_plan);
        }
        EXPECT_EQ(parent.pp_stages.size(), 2u);
        EXPECT_THROW((void)parent.forPipelineStage(2), std::out_of_range);
    }

    TEST(Test__MoEOverlayPipelineStageBinding, ChildProjectionUsesIndependentStageAuthorities)
    {
        for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe,
                               MoERebalanceRuntimeMode::Dynamic})
        for (const int width : {1, 2, 4, 8})
        {
            StageFixture early(0, false, 0, DeviceType::ROCm, width, mode);
            StageFixture terminal(2, true, 1, DeviceType::CUDA, width, mode);
            auto parent = densePipeline();
            parent.moe_rebalance.mode = mode;
            parent.pp_stages = {early.stage(), terminal.stage()};
            ASSERT_TRUE(parent.validate());
            auto first = parent.forPipelineStage(0);
            auto last = parent.forPipelineStage(1);
            EXPECT_EQ(first.moe_expert_overlay_residency_authority, early.owners.authority);
            EXPECT_EQ(last.moe_expert_overlay_residency_authority, terminal.owners.authority);
            EXPECT_EQ(first.moe_expert_overlay_participant_residency, early.owners.residency);
            EXPECT_EQ(last.moe_expert_overlay_participant_residency, terminal.owners.residency);
            EXPECT_EQ(first.moe_expert_overlay_decode_histogram, early.owners.histogram);
            EXPECT_EQ(last.moe_expert_overlay_decode_histogram, terminal.owners.histogram);
            EXPECT_EQ(first.moe_routed_expert_plan->first_model_layer, 0);
            EXPECT_EQ(last.moe_routed_expert_plan->first_model_layer, 2);
            EXPECT_EQ(last.routed_expert_compute_policy, RoutedExpertComputePolicy::Apportioned);
            first.moe_routed_expert_plan->first_model_layer = 99;
            EXPECT_EQ(early.owners.authority->snapshot()->placement_plan->first_model_layer, 0);
            EXPECT_EQ(parent.forPipelineStage(0).moe_routed_expert_plan->first_model_layer, 0);
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, ParentRejectsModelWideExpertInheritance)
    {
        StageFixture fixture(0, false, 0, DeviceType::ROCm, 2, MoERebalanceRuntimeMode::Off);
        for (int mutation = 0; mutation < 6; ++mutation)
        {
            SCOPED_TRACE(mutation);
            auto parent = densePipeline();
            switch (mutation)
            {
            case 0: parent.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(
                *fixture.owners.authority->snapshot()->placement_plan); break;
            case 1: parent.moe_expert_overlay_residency_authority = fixture.owners.authority; break;
            case 2: parent.moe_expert_overlay_participant_residency = fixture.owners.residency; break;
            case 3: parent.moe_rank_batch_transport_registry = std::make_shared<MoEOverlayRankBatchTransportRegistry>(); break;
            case 4: parent.moe_node_local_route_transport_policy.ordinary_prefill =
                        MoEOverlayNodeLocalRouteTransport::NativeCollective; break;
            case 5: parent.moe_node_local_route_transport_policy.decode =
                        MoEOverlayNodeLocalRouteTransport::NativeCollective; break;
            }
            EXPECT_FALSE(parent.validate());
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, ParentRejectsPartialOrSwappedStageInstallation)
    {
        StageFixture early(0, false, 0, DeviceType::ROCm, 2, MoERebalanceRuntimeMode::Dynamic);
        StageFixture terminal(2, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Dynamic);
        auto parent = densePipeline();
        parent.pp_stages = {early.stage(), terminal.stage(false)};
        EXPECT_FALSE(parent.validate());
        parent.pp_stages[1] = terminal.stage();
        std::swap(parent.pp_stages[0].moe_runtime, parent.pp_stages[1].moe_runtime);
        EXPECT_FALSE(parent.validate());
        EXPECT_THROW((void)parent.forPipelineStage(0), std::invalid_argument);
    }
}
