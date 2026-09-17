/**
 * @file Test__PlanningPublicationMPI.cpp
 * @brief Root-selected configuration and asymmetric publication failure proofs.
 *
 * Every test uses the real two-rank admission communicator, strict config codec
 * and existing initialization consensus. Failures must leave both ranks at the
 * same terminal; a successful subsequent exchange detects a stranded peer or
 * mismatched collective. No model payload or accelerator is used.
 */
#include "planning/PlanningPublication.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/AutomaticOrchestrationPlanner.h"
#include "config/OrchestrationConfigDocument.h"
#include "utils/MPIContext.h"
#include "../../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace llaminar2;

namespace
{
    /** @return A selected follower with its physical CPU endpoint from canonical observation. */
    OrchestrationConfig followerPlan(const std::shared_ptr<IMPIContext> &context)
    {
        const auto inventory = context->clusterInventory();
        const auto &follower = inventory->ranks.at(1);
        OrchestrationConfig config;
        config.model_path = "/model-directory/shared-weights.gguf";
        config.planning_mode = OrchestrationPlanningMode::Apply;
        config.device_for_this_rank = GlobalDeviceAddress::cpu(follower.cpu.numa_node, follower.hostname);
        config.device_for_this_rank_numa_explicit = true;
        config.execution_rank_selection = ExecutionRankSelection({1});
        config.mpi_procs = 2;
        config.mtp.enabled = true;
        config.mtp.graph_capacity_draft_tokens = 15;
        config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        config.mtp.depth_policy.max_depth = 15;
        config.prefill_max_bucket_size = 64;
        return config;
    }

    /** @brief Prove a complete valid exchange after an adversarial terminal. */
    void requireNextPublication(const std::shared_ptr<IMPIContext> &context)
    {
        bool accepted = false;
        exchangePlanningArtifact(context, PlanningArtifact::ModelMetadata,
            [] { return std::vector<uint8_t>{42}; }, [&](auto bytes) {
                accepted = bytes.size() == 1 && bytes[0] == 42;
            });
        EXPECT_TRUE(accepted);
    }
}

TEST(PlanningPublicationMPI, RootSelectsOnceAndAllRanksReceiveExactApplyConfiguration)
{
    const auto context = MPIContextFactory::global();
    ASSERT_EQ(context->world_size(), 2);
    const auto expected = followerPlan(context);
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        int calls = 0;
        const auto selected = exchangeSelectedOrchestration(context, [&] {
            ++calls;
            if (!context->is_root()) throw std::runtime_error("Follower attempted selection");
            return expected;
        });
        EXPECT_EQ(calls, context->is_root() ? 1 : 0);
        EXPECT_EQ(serializeOrchestrationConfig(selected), serializeOrchestrationConfig(expected));
        // Consume the published namespace with real admission, not a test-only
        // rank map. Discovery root retires; the selected follower owns rank zero.
        const auto admitted = MPIContextFactory::selectRanks(context,
            selected.execution_rank_selection->discoveryRanks());
        EXPECT_EQ(std::holds_alternative<InactiveMPIRank>(admitted), context->is_root());
        if (const auto *active = std::get_if<std::shared_ptr<MPIContext>>(&admitted))
        {
            EXPECT_EQ((*active)->rank(), 0);
            EXPECT_EQ((*active)->world_size(), 1);
            EXPECT_EQ((*active)->clusterInventory()->ranks[0].cpu.numa_node,
                      selected.device_for_this_rank->numa_node);
        }
        requireNextPublication(context);
    }
}

TEST(PlanningPublicationMPI, RootSelectionFailureOrInvalidSelectionFailsEveryRank)
{
    const auto context = MPIContextFactory::global();
    const auto expected = followerPlan(context);
    for (int defect = 0; defect < 5; ++defect)
    {
        int calls = 0;
        EXPECT_THROW((void)exchangeSelectedOrchestration(context, [&] {
            ++calls;
            if (defect == 0) throw std::runtime_error("Root cost evaluator failed");
            auto config = expected;
            if (defect == 1) config.execution_rank_selection.reset();
            if (defect == 2) config.execution_rank_selection = ExecutionRankSelection({2});
            if (defect == 3) config.mpi_procs = 1;
            if (defect == 4) config.planning_mode = OrchestrationPlanningMode::Automatic;
            return config;
        }), std::runtime_error);
        EXPECT_EQ(calls, context->is_root() ? 1 : 0);
        requireNextPublication(context);
    }
}

TEST(PlanningPublicationMPI, RealCandidateAdmissionAndSelectionPublishWithoutFollowerModelReads)
{
    const auto context = MPIContextFactory::global();
    const auto inventory = context->clusterInventory();
    ASSERT_EQ(inventory->world_size, 2);
    int selections = 0;
    const auto selected = exchangeSelectedOrchestration(context, [&] {
        ++selections;
        // Only root owns this tiny metadata-only GGUF. The compiler and PMA
        // admission are real; costs are an explicit decision oracle, not a
        // claim that either physical CPU has been benchmarked here.
        llaminar2::test::PlanningGGUFFixture fixture;
        PlanningModelSource source(fixture.path());
        OrchestrationConfig request;
        request.model_path = source.path();
        request.max_seq_len = 512;
        request.automatic_planning.only_backends = std::vector{DeviceType::CPU};
        request.automatic_planning.only_strategies = std::vector{OrchestrationStrategy::SingleDevice};
        const auto winner = AutomaticOrchestrationPlanner::select(request, source, *inventory,
            {.prefill_bucket_rows = {32, 64}, .minimum_prefill_sequence_rows = 1,
             .maximum_cached_prefill_buckets = 8}, {32, 64},
            [](const auto &candidate, const auto &workload) {
                const bool follower = candidate.membership().discoveryRanks() == std::vector<int>{1};
                return OrchestrationCostEstimate(workload, follower ? 1 : 2, 0.01,
                    "MPI publication regression decision oracle");
            });
        return winner.candidate().config();
    });
    EXPECT_EQ(selections, context->is_root() ? 1 : 0);
    EXPECT_EQ(selected.execution_rank_selection->discoveryRanks(), (std::vector<int>{1}));
    EXPECT_EQ(selected.mpi_procs, 2);
    EXPECT_EQ(selected.device_for_this_rank->numa_node, inventory->ranks[1].cpu.numa_node);
    const auto admitted = MPIContextFactory::selectRanks(context,
        selected.execution_rank_selection->discoveryRanks());
    EXPECT_EQ(std::holds_alternative<InactiveMPIRank>(admitted), context->is_root());
    requireNextPublication(context);
}

TEST(PlanningPublicationMPI, PeerDecoderFailureOrAbsenceCannotPublishSuccess)
{
    const auto context = MPIContextFactory::global();
    for (int failed_rank : {0, 1})
    {
        for (bool missing : {false, true})
        {
            std::function<void(std::span<const uint8_t>)> accept;
            if (!missing || context->rank() != failed_rank)
                accept = [&](auto) {
                    if (context->rank() == failed_rank) throw std::runtime_error("Rank rejected decoded artifact");
                };
            EXPECT_THROW(exchangePlanningArtifact(context, PlanningArtifact::SelectedOrchestration,
                [] { return std::vector<uint8_t>{1, 2, 3}; }, accept), std::runtime_error);
            requireNextPublication(context);
        }
    }
}

TEST(PlanningPublicationMPI, ArtifactMismatchFailsBeforeAnyPayloadIsDecoded)
{
    const auto context = MPIContextFactory::global();
    int accepted = 0;
    for (const auto peer_kind : {PlanningArtifact::SelectedOrchestration, static_cast<PlanningArtifact>(99)})
    {
        EXPECT_THROW(exchangePlanningArtifact(context,
            context->is_root() ? PlanningArtifact::ModelMetadata : peer_kind,
            [] { return std::vector<uint8_t>{7}; }, [&](auto) { ++accepted; }), std::runtime_error);
        EXPECT_EQ(accepted, 0);
        requireNextPublication(context);
    }
}

TEST(PlanningPublicationMPI, MalformedWrapperCannotTakeASingleRankShortcut)
{
    const auto context = MPIContextFactory::global();
    for (int failed_rank : {0, 1})
    {
        auto malformed = std::make_shared<MPIContext>(context->rank(),
            context->rank() == failed_rank ? 1 : context->world_size(), context->communicator());
        EXPECT_THROW((void)exchangePlanningModelMetadata(malformed, [] {
            ModelMemoryProfile profile;
            profile.n_layers = 3;
            profile.d_model = 512;
            profile.n_heads = 8;
            profile.n_kv_heads = 2;
            return PlanningModelMetadata(profile, 3);
        }), std::runtime_error);
        requireNextPublication(context);
    }
}

/** @test Distinct bounded requests/results retain the discovery rank that executed them. */
TEST(PlanningPublicationMPI, SampleScatterGatherHasOneRootAndOneSamplerPerRank)
{
    const auto context = MPIContextFactory::global();
    ASSERT_EQ(context->world_size(), 2);
    for (int iteration = 0; iteration != 20; ++iteration)
    {
        int planned = 0, executed = 0, accepted = 0;
        exchangePlanningSamples(context, [&](int count) {
            ++planned;
            EXPECT_TRUE(context->is_root());
            EXPECT_EQ(count, 2);
            return std::vector<std::vector<uint8_t>>{{17}, {31, 32, 33}};
        }, [&](auto bytes) {
            ++executed;
            const auto expected = context->is_root() ? std::vector<uint8_t>{17} : std::vector<uint8_t>{31, 32, 33};
            EXPECT_EQ(std::vector<uint8_t>(bytes.begin(), bytes.end()), expected);
            return std::vector<uint8_t>(static_cast<size_t>(context->rank() + 2),
                static_cast<uint8_t>(iteration + context->rank()));
        }, [&](auto samples) {
            ++accepted;
            ASSERT_TRUE(context->is_root());
            ASSERT_EQ(samples.size(), 2);
            for (int rank = 0; rank != 2; ++rank)
            {
                EXPECT_EQ(samples[rank].discovery_rank, rank);
                EXPECT_EQ(std::vector<uint8_t>(samples[rank].bytes.begin(), samples[rank].bytes.end()),
                    std::vector<uint8_t>(rank + 2, static_cast<uint8_t>(iteration + rank)));
            }
        });
        EXPECT_EQ(planned, context->is_root() ? 1 : 0);
        EXPECT_EQ(executed, 1);
        EXPECT_EQ(accepted, context->is_root() ? 1 : 0);
    }
    requireNextPublication(context);
}

/** @test A failed remote device/sample cannot disappear from the evidence set. */
TEST(PlanningPublicationMPI, SampleFailureOnEitherRankReachesEveryPeer)
{
    const auto context = MPIContextFactory::global();
    for (int failed_rank : {0, 1})
        for (int defect = 0; defect != 4; ++defect)
        {
            int accepted = 0;
            std::function<std::vector<uint8_t>(std::span<const uint8_t>)> sample = [&](auto) {
                if (context->rank() == failed_rank)
                {
                    if (defect == 1) throw std::runtime_error("native event query failed");
                    if (defect == 2) return std::vector<uint8_t>{};
                }
                return std::vector<uint8_t>{1};
            };
            if (defect == 0 && context->rank() == failed_rank) sample = {};
            const auto wrapper = defect == 3 ? std::make_shared<MPIContext>(context->rank(),
                context->rank() == failed_rank ? 1 : 2, context->communicator()) : context;
            EXPECT_THROW(exchangePlanningSamples(wrapper,
                [](int count) { return std::vector<std::vector<uint8_t>>(count, {1}); },
                sample, [&](auto) { ++accepted; }), std::runtime_error);
            EXPECT_EQ(accepted, 0);
            requireNextPublication(context);
        }
}

/** @test Root preparation/validation and a mixed broadcast/gather cannot strand followers. */
TEST(PlanningPublicationMPI, SampleRootAndProtocolFailuresAreCollective)
{
    const auto context = MPIContextFactory::global();
    for (int defect = 0; defect != 4; ++defect)
    {
        EXPECT_THROW(exchangePlanningSamples(context, [&](int count) {
            if (defect == 0) throw std::runtime_error("cannot describe sample work");
            if (defect == 1) return std::vector<std::vector<uint8_t>>(count - 1, {1});
            if (defect == 2) return std::vector<std::vector<uint8_t>>(count);
            return std::vector<std::vector<uint8_t>>(count, {1});
        }, [](auto) { return std::vector<uint8_t>{2}; },
        [](auto) { throw std::runtime_error("root rejected sample provenance"); }), std::runtime_error);
        requireNextPublication(context);
    }
    if (context->is_root())
        EXPECT_THROW(exchangePlanningSamples(context,
            [](int count) { return std::vector<std::vector<uint8_t>>(count, {1}); },
            [](auto) { return std::vector<uint8_t>{2}; }, [](auto) {}), std::runtime_error);
    else
        EXPECT_THROW(exchangePlanningArtifact(context, PlanningArtifact::ModelMetadata,
            [] { return std::vector<uint8_t>{1}; }, [](auto) {}), std::runtime_error);
    requireNextPublication(context);
}
