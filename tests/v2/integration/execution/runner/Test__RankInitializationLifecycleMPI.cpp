/**
 * @file Test__RankInitializationLifecycleMPI.cpp
 * @brief Multi-rank regressions for exception-safe initialization consensus.
 *
 * These tests use no model or accelerator. They prove that an asymmetric
 * return/exception remains inside the phase protocol and that skipped or
 * reordered phases are rejected by identity instead of being mistaken for a
 * matching collective.
 * Planning metadata exercises that same lifecycle with a root-only producer,
 * real descriptor broadcasts, and fatal read errors before any payload exists.
 * The actual distributed runner must preserve these errors through admission
 * and teardown; the device-free reader Unit cannot certify that MPI boundary.
 * Overlay BOM construction also runs inside a readiness phase: malformed
 * capture policy on one rank must prevent every peer entering budget exchange.
 *
 * @author David Sanftenberg
 * @date August 2026
 */

#include "execution/runner/RankInitializationLifecycle.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/MoEOverlayMemoryPlanInputs.h"
#include "planning/MoEOverlayPlanningInputs.h"
#include "utils/MPIContext.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <stdexcept>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Read and validate the two-rank test communicator. */
        int requireTwoRanks()
        {
            int world_size = 0;
            EXPECT_EQ(
                MPI_Comm_size(MPI_COMM_WORLD, &world_size),
                MPI_SUCCESS);
            if (world_size != 2)
                return -1;

            int rank = -1;
            EXPECT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
            return rank;
        }

        /** @brief Invoke the exact production MPI phase-consensus transport. */
        RankInitializationConsensusResult reachConsensus(
            RankInitializationPhaseIdentity identity,
            RankInitializationLocalOutcome local_outcome)
        {
            return MPIRankInitializationConsensus::reach(
                MPI_COMM_WORLD,
                identity,
                local_outcome);
        }
    } // namespace

    TEST(
        Test__RankInitializationLifecycleMPI,
        AsymmetricExceptionStillReachesOneGlobalFailureTerminal)
    {
        const int rank = requireTwoRanks();
        if (rank < 0)
            GTEST_SKIP() << "requires exactly two MPI ranks";

        const auto result = RankInitializationLifecycle::execute(
            RankInitializationPhaseIdentity{
                .ordinal = 7,
                .name = "buildParticipantGraphs",
            },
            [rank]
            {
                if (rank == 1)
                    throw std::runtime_error("injected rank-local failure");
                return true;
            },
            reachConsensus);

        EXPECT_FALSE(result.succeeded());
        if (rank == 0)
        {
            EXPECT_EQ(
                result.status,
                RankInitializationPhaseStatus::PeerStepFailed);
        }
        else
        {
            EXPECT_EQ(
                result.status,
                RankInitializationPhaseStatus::LocalStepThrewException);
            EXPECT_EQ(result.detail, "injected rank-local failure");
        }
    }

    TEST(
        Test__RankInitializationLifecycleMPI,
        AsymmetricReturnedFailureStillReachesOneGlobalFailureTerminal)
    {
        const int rank = requireTwoRanks();
        if (rank < 0)
            GTEST_SKIP() << "requires exactly two MPI ranks";

        const auto result = RankInitializationLifecycle::execute(
            RankInitializationPhaseIdentity{
                .ordinal = 8,
                .name = "publishRuntimeTables",
            },
            [rank] { return rank == 0; },
            reachConsensus);

        EXPECT_FALSE(result.succeeded());
        EXPECT_EQ(
            result.status,
            rank == 0
                ? RankInitializationPhaseStatus::PeerStepFailed
                : RankInitializationPhaseStatus::LocalStepReturnedFailure);
    }

    TEST(
        Test__RankInitializationLifecycleMPI,
        MismatchedPhaseIdentityIsRejectedOnEveryRank)
    {
        const int rank = requireTwoRanks();
        if (rank < 0)
            GTEST_SKIP() << "requires exactly two MPI ranks";

        const auto result = RankInitializationLifecycle::execute(
            RankInitializationPhaseIdentity{
                .ordinal = rank == 0 ? 9u : 10u,
                .name = rank == 0 ? "captureGraphs" : "startWorkers",
            },
            [] { return true; },
            reachConsensus);

        EXPECT_FALSE(result.succeeded());
        EXPECT_EQ(
            result.status,
            RankInitializationPhaseStatus::PhaseIdentityMismatch);
        EXPECT_NE(result.detail.find("submitted phase ordinal"), std::string::npos);
    }

    TEST(Test__RankInitializationLifecycleMPI, PlanningMetadataReadsOnlyRootAndPreservesEveryDescriptor)
    {
        const int rank = requireTwoRanks();
        ASSERT_GE(rank, 0) << "requires exactly two MPI ranks";
        const auto context = MPIContextFactory::global();
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            int reads = 0;
            const auto metadata = exchangePlanningModelMetadata(context, [&] {
                ++reads;
                if (rank != 0) throw std::runtime_error("Follower must never read GGUF metadata");
                ModelMemoryProfile profile;
                profile.architecture = "qwen2";
                profile.n_layers = 3;
                profile.d_model = 512;
                profile.n_heads = 8;
                profile.n_kv_heads = 2;
                profile.total_native_bytes = static_cast<size_t>(iteration + 1);
                return PlanningModelMetadata(std::move(profile), 2);
            });
            EXPECT_EQ(metadata.mainLayerCount(), 2);
            EXPECT_EQ(metadata.memoryProfile().n_layers, 3);
            EXPECT_EQ(metadata.memoryProfile().total_native_bytes, static_cast<size_t>(iteration + 1));
            EXPECT_EQ(reads, rank == 0 ? 1 : 0);
            int total_reads = 0;
            ASSERT_EQ(MPI_Allreduce(&reads, &total_reads, 1, MPI_INT, MPI_SUM, context->communicator()), MPI_SUCCESS);
            EXPECT_EQ(total_reads, 1);
        }
    }

    TEST(Test__RankInitializationLifecycleMPI, PlanningMetadataRootFailureReachesEveryRankWithoutPayloadWait)
    {
        const int rank = requireTwoRanks();
        ASSERT_GE(rank, 0) << "requires exactly two MPI ranks";
        const auto context = MPIContextFactory::global();
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            int reads = 0;
            int caught = 0;
            try
            {
                (void)exchangePlanningModelMetadata(context, [&]() -> PlanningModelMetadata {
                    ++reads;
                    throw std::runtime_error("injected root metadata failure");
                });
            }
            catch (const std::runtime_error &error)
            {
                caught = 1;
                EXPECT_NE(std::string(error.what()).find("planning-model.read"), std::string::npos);
            }
            EXPECT_EQ(reads, rank == 0 ? 1 : 0);
            int failures = 0;
            ASSERT_EQ(MPI_Allreduce(&caught, &failures, 1, MPI_INT, MPI_SUM, context->communicator()), MPI_SUCCESS);
            EXPECT_EQ(failures, 2);
        }
    }

    TEST(Test__RankInitializationLifecycleMPI, MissingGGUFIsACollectiveReadFailure)
    {
        const int rank = requireTwoRanks();
        ASSERT_GE(rank, 0) << "requires exactly two MPI ranks";
        const auto context = MPIContextFactory::global();
        EXPECT_THROW((void)exchangePlanningModelMetadata(context, [] {
            return readPlanningModelMetadata("/llaminar-metadata-test/nonexistent.gguf");
        }), std::runtime_error);
        // Reaching the next collective on both ranks proves the throwing
        // production reader did not strand a peer in the old size broadcast.
        int completed = 1;
        ASSERT_EQ(MPI_Allreduce(MPI_IN_PLACE, &completed, 1, MPI_INT, MPI_SUM, context->communicator()), MPI_SUCCESS);
        EXPECT_EQ(completed, 2);
    }

    TEST(Test__RankInitializationLifecycleMPI, OverlayMemoryInputFailureReachesConsensusBeforeBudgetExchange)
    {
        const int rank = requireTwoRanks();
        ASSERT_GE(rank, 0) << "requires exactly two MPI ranks";
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        auto overlay = std::make_shared<MoERoutedExpertPlacementPlan>();
        overlay->enabled = true;
        overlay->topology = RoutedExpertPlacementTopology::SingleDomain;
        overlay->authority_execution = MoEOverlayAuthorityExecutionKind::HostResident;
        RoutedExpertDomain domain;
        domain.name = "continuation";
        domain.scope = ExecutionDomainScope::SINGLE;
        domain.owner_rank = 0;
        domain.participants = {GlobalDeviceAddress::cpu()};
        overlay->domains = {domain};
        overlay->continuation_domain = domain.name;
        overlay->base_model_domain = domain.name;
        overlay->shared_expert_domain = domain.name;
        overlay->routed_tiers = {{.name = "priority-37", .domain = domain.name,
            .priority = 37, .fallback = true}};
        config.moe_routed_expert_plan = overlay;
        ModelMemoryProfile model;
        model.n_layers = 2;
        model.expert_count = 8;
        model.expert_used_count = 2;
        RankExecutionPlan rank_plan;
        rank_plan.rank = rank;
        rank_plan.runtime.batch_size = 1;
        ClusterInventory inventory;
        inventory.world_size = 2;
        inventory.ranks.resize(2);
        for (int index = 0; index < 2; ++index) inventory.ranks[index].rank = index;
        const auto execution = resolveMoEExpertOverlayExecutionPlan(overlay,
            {.current_world_rank = rank, .world_size = 2});
        const auto policy = resolveMoEOverlayCapacityAdmissionPolicy(*overlay, config, 2, 2, 8);
        const MoEOverlayInferenceGraphFamilyIdentity family{
            .graph_family_generation = 1, .main_layer_count = 2, .mtp_source_layers = {},
            .max_graph_rows = 8, .max_decode_rows = 1, .max_request_count = 1,
            .max_mtp_draft_depth = 0};
        const std::vector<int> buckets{1, 8};
        MoEOverlayMemoryPlanInputRequest request{
            .model = model, .rank_plan = rank_plan, .config = config, .inventory = inventory,
            .execution = execution, .capacity_policy = policy, .retained_mtp = rank_plan.runtime.mtp,
            .graph_family = family, .prefill = {.bucket_rows = buckets,
                .minimum_sequence_rows = 1, .maximum_cached_buckets = 16}};
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            // Alternate the failing owner: a root-only error must have exactly
            // the same terminal outcome as an error on a remote follower.
            const bool local_failure = rank == iteration % 2;
            request.prefill.maximum_cached_buckets = local_failure ? 2 : 16;
            const auto result = RankInitializationLifecycle::execute(
                {4, "overlay-candidate.fixed-bom"}, [&] {
                    (void)buildMoEOverlayMemoryPlanInputs(request, 8);
                    return true;
                }, reachConsensus);
            EXPECT_FALSE(result.succeeded());
            EXPECT_EQ(result.status, local_failure
                ? RankInitializationPhaseStatus::LocalStepThrewException
                : RankInitializationPhaseStatus::PeerStepFailed);
            if (local_failure)
                EXPECT_NE(result.detail.find("prefill graph cache"), std::string::npos);

            // A subsequent matched phase proves no peer remained in a payload
            // collective. This is protocol recovery in a test, not runtime retry.
            request.prefill.maximum_cached_buckets = 16;
            const auto complete = RankInitializationLifecycle::execute(
                {5, "overlay-candidate.next-shape"}, [&] {
                    return buildMoEOverlayMemoryPlanInputs(request, 8).local_capacity.captured_graph_plan.valid();
                }, reachConsensus);
            EXPECT_TRUE(complete.succeeded()) << complete.detail;
        }
    }

    TEST(Test__RankInitializationLifecycleMPI, RunnerPreservesMetadataFailureAndRetiresOnEveryRank)
    {
        const int rank = requireTwoRanks();
        ASSERT_GE(rank, 0) << "requires exactly two MPI ranks";
        // An empty name used to install invented model dimensions. Both it and
        // an absent path must fail in metadata admission, before graph setup.
        for (const std::string path : {"/llaminar-metadata-test/nonexistent.gguf", ""})
        {
            SCOPED_TRACE(path);
            int failed = 0;
            {
                OrchestrationConfig config = OrchestrationConfig::defaults();
                config.model_path = path;
                config.tp_scope = TPScope::NODE_LOCAL;
                config.tp_degree = 2;
                OrchestrationRunner runner(std::move(config), std::make_unique<ExecutionPlanBuilder>());
                failed = !runner.initialize();
                EXPECT_EQ(failed, 1);
                EXPECT_NE(runner.lastError().find("Failed to publish planning metadata"), std::string::npos)
                    << runner.lastError();
                EXPECT_NE(runner.lastError().find("planning-model.read"), std::string::npos)
                    << runner.lastError();
                if (!path.empty())
                    EXPECT_NE(runner.lastError().find(path), std::string::npos) << runner.lastError();
                else if (rank == 0)
                    EXPECT_NE(runner.lastError().find("requires a GGUF model path"), std::string::npos)
                        << runner.lastError();
                runner.shutdown();
            }
            // Including destruction before this collective verifies that no
            // failed participant left a worker or late collective behind.
            ASSERT_EQ(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD), MPI_SUCCESS);
            EXPECT_EQ(failed, 2);
        }
    }

} // namespace llaminar2::test
