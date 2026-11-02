/**
 * @file Test__PreparedForwardGraphTemplate.cpp
 * @brief Actual builder preparation counts, source lifecycle and family isolation.
 *
 * This device-free gate expands real Qwen graphs, checks original owner-map
 * policy/metadata reuse, and rejects malformed cold preparation before source
 * publication. Native capture/replay qualification remains a separate gate.
 */
#include "execution/local_execution/graph/PreparedGraphTemplate.h"
#include "utils/ForwardGraphTemplateFixture.h"
#include <gtest/gtest.h>
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include <array>
#include <stdexcept>
#include <type_traits>

namespace llaminar2::test
{
    TEST(PreparedForwardGraphTemplate, ColdFactoryFailurePublishesNothing)
    {
        PreparedGraphTemplate<int> prepared;
        EXPECT_THROW(prepared.prepare([]() -> int { throw std::runtime_error("incomplete"); }), std::runtime_error);
        EXPECT_EQ(prepared.preparations(), 0u);
        EXPECT_NO_THROW(prepared.requireUnprepared("retry setup"));
        EXPECT_EQ(prepared.prepare([] { return 7; }), 7);
        EXPECT_EQ(prepared.prepare([] { return 9; }), 7);
        EXPECT_EQ(prepared.preparations(), 1u);
        EXPECT_THROW(prepared.requireUnprepared("replace source"), std::logic_error);
        static_assert(!std::is_move_constructible_v<PreparedGraphTemplate<int>>);
    }

    TEST(PreparedForwardGraphTemplate, FailedPrepareAndColdBuildLeaveSourceSetupOpen)
    {
        ForwardTemplateConstructionFixture fixture;
        QwenStandardGraph builder(fixture.source.config, fixture.source.mpi);
        EXPECT_THROW(builder.prepareForwardGraphTemplate(DeviceId::invalid()), std::invalid_argument);
        EXPECT_THROW(builder.prepareForwardGraphTemplate(DeviceId::cuda(0)), std::invalid_argument);
        EXPECT_THROW(builder.prepareForwardGraphTemplate(DeviceId::cpu()), std::invalid_argument);
        ForwardOutput output;
        EXPECT_THROW(builder.buildForwardGraph(fixture.input(1), output), std::runtime_error);
        EXPECT_EQ(builder.forwardGraphTemplateStats().preparations, 0u);
        EXPECT_NO_THROW(builder.setWeights(fixture.weights()));
        EXPECT_THROW(builder.buildForwardGraph(fixture.input(1), output), std::runtime_error);
        EXPECT_EQ(builder.forwardGraphTemplateStats().preparations, 0u);
        builder.setBuffers(fixture.buffers);
        EXPECT_NO_THROW(builder.prepareForwardGraphTemplate(DeviceId::cpu()));
        EXPECT_EQ(builder.forwardGraphTemplateStats().preparations, 1u);
        EXPECT_THROW(builder.setWeights(fixture.weights()), std::logic_error);
        EXPECT_THROW(builder.setWeightBindings({}), std::logic_error);
        EXPECT_THROW(builder.setDecodeReplicatedDenseWeightBindings({}), std::logic_error);
        EXPECT_NO_THROW(builder.setBuffers(fixture.buffers));
        EXPECT_TRUE(builder.setComputeAllPositionLogits(true));
        EXPECT_TRUE(builder.setGroupedMTPVerifier(false));
    }

    TEST(PreparedForwardGraphTemplate, ActualFamilySpecializationsKeepIndependentStages)
    {
        ForwardTemplateConstructionFixture fixture;
        for (const unsigned family : {1u, 2u, 4u, 8u, 16u, 32u, 64u})
        {
            auto builder = fixture.builder();
            builder->prepareForwardGraphTemplate(DeviceId::cpu());
            std::vector<ComputeGraph> graphs;
            graphs.reserve(family);
            for (unsigned member = 0; member < family; ++member)
            {
                // Separate source recipe from row/output topology. Current
                // real scopes remain in charge; no member borrows a stage.
                const int rows = member % 2 ? 4 : 1;
                ASSERT_TRUE(builder->setComputeAllPositionLogits(rows > 1));
                ForwardOutput output;
                graphs.push_back(builder->buildForwardGraph(fixture.input(rows), output));
                ASSERT_GT(graphs.back().size(), 0u);
                if (member)
                {
                    EXPECT_EQ(graphs.back().size(), graphs.front().size());
                    for (const auto &name : graphs.front().getExecutionOrder())
                    {
                        const auto *first = graphs.front().getNode(name);
                        const auto *next = graphs.back().getNode(name);
                        ASSERT_NE(first, nullptr); ASSERT_NE(next, nullptr);
                        EXPECT_EQ(first->dependencies, next->dependencies);
                        EXPECT_EQ(first->stage->type(), next->stage->type());
                        EXPECT_NE(first->stage.get(), next->stage.get());
                    }
                }
            }
            const auto stats = builder->forwardGraphTemplateStats();
            EXPECT_EQ(stats.preparations, 1u);
            EXPECT_EQ(stats.specialization_attempts, family);
            EXPECT_EQ(stats.canonical_owner_map_builds, 0u);
        }
    }

    TEST(PreparedForwardGraphTemplate, ActualRoutedFFNFamilyBuildsCanonicalMapOnce)
    {
        RoutedForwardTemplateFixture fixture;
        for (const unsigned family : {1u, 2u, 4u, 8u, 16u, 32u, 64u})
        {
            auto builder = fixture.builder();
            const auto calls = MoEExpertOwnerMap::initialBuildAttempts();
            builder->prepareForwardGraphTemplate(DeviceId::cpu());
            std::vector<ComputeGraph> graphs; graphs.reserve(family * fixture.config.n_layers);
            for (unsigned member = 0; member < family; ++member)
            for (int layer = 0; layer < fixture.config.n_layers; ++layer)
            {
                graphs.push_back(fixture.specialize(*builder, layer, member % 2 ? 2 : 1));
                const std::string prefix = "layer" + std::to_string(layer) + "_";
                const auto &graph = graphs.back();
                const auto *route = graph.getNode(prefix + "moe_routing");
                const auto *expert = graph.getNode(prefix + "moe_expert_ffn_overlay_fast");
                const auto *combine = graph.getNode(prefix + "moe_combine");
                const auto *residual = graph.getNode(prefix + "ffn_residual");
                ASSERT_NE(route, nullptr); ASSERT_NE(expert, nullptr);
                ASSERT_NE(combine, nullptr);
                if (layer == fixture.config.n_layers - 1) ASSERT_NE(residual, nullptr);
                else EXPECT_EQ(residual, nullptr) << "Intermediate residual is fused into the next layer norm";
                ASSERT_NE(dynamic_cast<const MoEExpertComputeStage *>(expert->stage.get()), nullptr);
                // Independent required routed topology, not a second candidate
                // oracle: actual route -> expert -> combine -> residual.
                EXPECT_NE(std::find(expert->dependencies.begin(), expert->dependencies.end(), prefix + "moe_routing"), expert->dependencies.end());
                EXPECT_NE(std::find(combine->dependencies.begin(), combine->dependencies.end(), prefix + "moe_expert_ffn_overlay_fast"), combine->dependencies.end());
                if (residual)
                    EXPECT_NE(std::find(residual->dependencies.begin(), residual->dependencies.end(), prefix + "moe_combine"), residual->dependencies.end());
                else EXPECT_EQ(graph.terminalNode(), prefix + "moe_combine");
                if (member)
                {
                    const auto &first = graphs.at(static_cast<std::size_t>(layer));
                    EXPECT_NE(first.getNode(prefix + "moe_expert_ffn_overlay_fast")->stage.get(), expert->stage.get());
                }
            }
            EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts() - calls, 1u)
                << "Actual authority must catch any nested constructor, not only the final cached recipe";
            const auto stats = builder->forwardGraphTemplateStats();
            EXPECT_EQ(stats.preparations, 1u);
            EXPECT_EQ(stats.canonical_owner_map_builds, 1u);
            EXPECT_EQ(stats.continuation_resolutions, 1u);
            EXPECT_EQ(stats.routed_layer_specialization_attempts, family * fixture.config.n_layers);
        }
    }

    TEST(PreparedForwardGraphTemplate, ActualGeneralOverlayRetainsOriginalMapAfterBuilderRetires)
    {
        RoutedForwardTemplateFixture fixture;
        auto builder = fixture.serialBuilder();
        std::weak_ptr<const MoEExpertOwnerMap> retired;
        {
            auto graph = fixture.specialize(*builder, 0, 2);
            const MoEExpertDispatchStage *dispatch = nullptr;
            for (const auto &name : graph.getExecutionOrder())
                if (auto *candidate = dynamic_cast<const MoEExpertDispatchStage *>(graph.getNode(name)->stage.get()))
                    dispatch = candidate;
            ASSERT_NE(dispatch, nullptr) << "The actual general producer must own dispatch; fast/dense paths do not prove map lifetime";
            ASSERT_NE(dispatch->params().owner_map, nullptr);
            const auto *original = dispatch->params().owner_map.get();
            retired = dispatch->params().owner_map;
            const auto owners = original->owners();
            builder.reset();
            ASSERT_FALSE(retired.expired());
            EXPECT_EQ(dispatch->params().owner_map.get(), original);
            EXPECT_EQ(dispatch->params().owner_map->owners(), owners);
            EXPECT_NE(dispatch->params().owner_map->ownerFor(0, 0), nullptr);
        }
        EXPECT_TRUE(retired.expired()) << "The final actual graph retires its sole prepared map owner";
    }

    TEST(PreparedForwardGraphTemplate, CanonicalRoutedPreparationRunsOncePerOriginalSource)
    {
        ForwardTemplateConstructionFixture fixture;
        for (const auto order : {RoutedExpertOwnerOrder::Ordinal, RoutedExpertOwnerOrder::Random})
        for (const unsigned family : {1u, 2u, 4u, 8u, 16u, 32u, 64u})
        {
            auto config = fixture.source.config;
            config.moe.routed_expert_plan = forwardTemplatePlan(4, 32, order, 2);
            config.moe.num_experts = 32;
            const auto canonical = MoEExpertOwnerMap::build(*config.moe.routed_expert_plan);
            config.moe.expert_overlay_participant_residency =
                std::make_shared<MoEOverlayParticipantResidencyRegistry>(
                    MoEOverlayParticipantResidencyRegistry::Config{
                        .owner_map = canonical, .local_participant_ids = {},
                        .num_layers = 4, .num_experts = 32, .initial_epoch = 1});
            Qwen35MoEGraph builder(config, fixture.source.mpi);
            builder.setWeights(fixture.weights());
            for (unsigned member = 0; member < family; ++member)
                builder.prepareForwardGraphTemplate(DeviceId::cpu());
            const auto stats = builder.forwardGraphTemplateStats();
            EXPECT_EQ(stats.preparations, 1u);
            EXPECT_EQ(stats.canonical_owner_map_builds, 1u);
            EXPECT_EQ(stats.continuation_resolutions, 1u);
            EXPECT_EQ(stats.original_owner_map_reuses, 1u);
        }
    }

    TEST(PreparedForwardGraphTemplate, DistinctPreferredInitialMapPreservesCanonicalSourceSemantics)
    {
        ForwardTemplateConstructionFixture fixture;
        auto config = fixture.source.config;
        config.moe.routed_expert_plan = forwardTemplatePlan(2, 4, RoutedExpertOwnerOrder::Random, 2);
        const auto canonical = MoEExpertOwnerMap::build(*config.moe.routed_expert_plan);
        std::vector<std::vector<int>> choices(2, std::vector<int>(4));
        for (int layer = 0; layer < 2; ++layer)
        for (int expert = 0; expert < 4; ++expert)
            choices[layer][expert] = 1 - canonical.ownerFor(layer, expert)->owner_participant;
        const auto preferred = MoEExpertOwnerMap::buildExplicit(*config.moe.routed_expert_plan,
            MoELayeredExpertOwnership(2, std::move(choices)));
        ASSERT_NE(canonical.owners(), preferred.owners());
        config.moe.expert_overlay_participant_residency = std::make_shared<MoEOverlayParticipantResidencyRegistry>(
            MoEOverlayParticipantResidencyRegistry::Config{.owner_map = preferred,
                .local_participant_ids = {}, .num_layers = 2, .num_experts = 4, .initial_epoch = 1});
        Qwen35MoEGraph builder(config, fixture.source.mpi);
        builder.setWeights(fixture.weights());
        const auto calls = MoEExpertOwnerMap::initialBuildAttempts();
        EXPECT_NO_THROW(builder.prepareForwardGraphTemplate(DeviceId::cpu()));
        EXPECT_EQ(MoEExpertOwnerMap::initialBuildAttempts() - calls, 1u);
        EXPECT_EQ(builder.forwardGraphTemplateStats().original_owner_map_reuses, 0u)
            << "Preferred initial ownership must not replace the canonical source map used by this graph";
        EXPECT_EQ(config.moe.expert_overlay_participant_residency->initialOwnerMap().owners(), preferred.owners());
    }

    TEST(PreparedForwardGraphTemplate, ActualOwnerMapValidationFailuresAreCounted)
    {
        ForwardTemplateConstructionFixture fixture;
        auto config = fixture.source.config;
        auto bad = std::make_shared<MoERoutedExpertPlacementPlan>(
            *forwardTemplatePlan(4, 16, RoutedExpertOwnerOrder::Ordinal, 2));
        bad->domains.front().routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
        config.moe.routed_expert_plan = bad;
        config.moe.expert_overlay_runtime_plan = resolveMoEExpertOverlayRuntimePlan(
            bad, {.current_world_rank = 0, .validate_mvp_root_reachability = false});
        Qwen35MoEGraph builder(config, fixture.source.mpi);
        builder.setWeights(fixture.weights());
        for (std::uint64_t attempt = 1; attempt <= 3; ++attempt)
        {
            EXPECT_THROW(builder.prepareForwardGraphTemplate(DeviceId::cpu()), std::invalid_argument);
            EXPECT_EQ(builder.forwardGraphTemplateStats().preparations, 0u);
            EXPECT_EQ(builder.forwardGraphTemplateStats().canonical_owner_map_builds, attempt);
            EXPECT_EQ(builder.forwardGraphTemplateStats().continuation_resolutions, 0u);
            EXPECT_NO_THROW(builder.setWeights(fixture.weights()));
        }
    }
}
