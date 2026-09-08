/**
 * @file Test__RankInitializationLifecycleMPI.cpp
 * @brief Multi-rank regressions for exception-safe initialization consensus.
 *
 * These tests use no model or accelerator. They prove that an asymmetric
 * return/exception remains inside the phase protocol and that skipped or
 * reordered phases are rejected by identity instead of being mistaken for a
 * matching collective.
 *
 * @author David Sanftenberg
 * @date August 2026
 */

#include "execution/runner/RankInitializationLifecycle.h"

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

} // namespace llaminar2::test
