/**
 * @file Test__PrefixCacheCoordinatorMPI.cpp
 * @brief Real-rank prefix coordination regressions without model dependencies.
 *
 * These tests retain both admission endpoints through local and MPI nesting,
 * alongside the existing common-prefix and terminal-payload requirements.
 * They exercise the production collective protocol used by TP/PP serving.
 */
#include "execution/prefix_cache/PrefixCacheCoordinator.h"
#include "execution/prefix_cache/PrefixCacheStats.h"

#include <gtest/gtest.h>
#include <mpi.h>
#include <limits>

using namespace llaminar2;

namespace
{
    /** @brief Build one rank's authenticated prefix lookup observation. */
    PrefixParticipantLookup rankParticipant(
        int rank,
        int tokens,
        bool terminal_logits,
        bool terminal_hidden,
        uint64_t fingerprint,
        uint64_t placement_epoch = 11)
    {
        PrefixParticipantLookup lookup;
        lookup.domain_id = "mpi-prefix-domain";
        lookup.participant_id = rank;
        lookup.device = DeviceId::cpu();
        lookup.placement_epochs = PrefixPlacementEpochSpan::at(placement_epoch);
        lookup.fingerprint_key = fingerprint;
        lookup.supported = true;
        lookup.cache_enabled = true;
        lookup.hit = tokens > 0;
        lookup.matched_tokens = tokens;
        lookup.matched_blocks = tokens / 2;
        lookup.has_terminal_logits = terminal_logits;
        lookup.has_terminal_hidden = terminal_hidden;
        return lookup;
    }

    /** @return Number of ranks initialized by the integration test launcher. */
    int mpiWorldSize()
    {
        int size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }

    /** @return This process's rank in the integration communicator. */
    int mpiWorldRank()
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }
} // namespace

TEST(Test__PrefixCacheCoordinatorMPI, ReducesCommonPrefixAndTerminalStateAcrossRanks)
{
    if (mpiWorldSize() < 2)
        GTEST_SKIP() << "requires at least two MPI ranks";

    const int rank = mpiWorldRank();
    const int tokens = rank == 0 ? 8 : 4;
    const bool terminal_hidden = rank == 0;

    MPIPrefixCollectiveCoordinator coordinator(MPI_COMM_WORLD);
    auto result = coordinatePrefixLookups(
        {rankParticipant(rank, tokens, /*terminal_logits=*/true, terminal_hidden, 0xabcdu)},
        &coordinator);

    EXPECT_TRUE(result.supported);
    EXPECT_TRUE(result.cache_enabled);
    EXPECT_EQ(result.domain_id, "mpi-prefix-domain");
    EXPECT_EQ(result.common_matched_tokens, 4);
    EXPECT_EQ(result.common_matched_blocks, 2);
    EXPECT_TRUE(result.common_terminal_logits);
    EXPECT_FALSE(result.common_terminal_hidden);
    EXPECT_EQ(result.fingerprint_key, 0xabcdu);
    EXPECT_EQ(result.placement_epochs, PrefixPlacementEpochSpan::at(11));
}

TEST(Test__PrefixCacheCoordinatorMPI, ReducesPlacementEpochAcrossRanks)
{
    if (mpiWorldSize() < 2)
        GTEST_SKIP() << "requires at least two MPI ranks";

    const int rank = mpiWorldRank();
    const uint64_t placement_epoch = rank == 0 ? 11u : 19u;

    MPIPrefixCollectiveCoordinator coordinator(MPI_COMM_WORLD);
    auto result = coordinatePrefixLookups(
        {rankParticipant(rank, /*tokens=*/8, /*terminal_logits=*/true,
                         /*terminal_hidden=*/true, 0xabcdu, placement_epoch)},
        &coordinator);

    EXPECT_TRUE(result.supported);
    EXPECT_EQ(result.common_matched_tokens, 8);
    EXPECT_EQ(result.fingerprint_key, 0xabcdu);
    EXPECT_EQ(result.placement_epochs, PrefixPlacementEpochSpan::covering(11, 19));
}

/** @brief Preserve a stale child admission through local and real-MPI nesting. */
TEST(Test__PrefixCacheCoordinatorMPI, NestedAdmissionsRetainMovementAcrossLookups)
{
    ASSERT_GE(mpiWorldSize(), 2);
    const int rank = mpiWorldRank();
    for (int old_rank = 0; old_rank < mpiWorldSize(); ++old_rank)
    {
        // Every rank finishes lookup at epoch eight, but one rank's first
        // child was admitted at seven. The outer reduction must not erase it.
        const auto local = coordinatePrefixLookups({
            rankParticipant(rank * 2, 0, false, false, 0xabcdu,
                            rank == old_rank ? 7 : 8),
            rankParticipant(rank * 2 + 1, 0, false, false, 0xabcdu, 8),
        });
        MPIPrefixCollectiveCoordinator coordinator(MPI_COMM_WORLD);
        const auto global = coordinatePrefixLookups({
            makePrefixParticipantLookup(
                rank, DeviceId::cpu(), makePrefixLookupResult(local, 2)),
        }, &coordinator);
        EXPECT_EQ(global.placement_epochs, PrefixPlacementEpochSpan::covering(7, 8));
        const PrefixCacheRequestSummary cold{
            .admission_placement_epochs = global.placement_epochs,
            .completion_movement_epoch = 8,
        };
        EXPECT_TRUE(cold.crossedMovementEpoch());
        EXPECT_TRUE(cold.movementPrecededAdmissionOf(PrefixCacheRequestSummary{
            .admission_placement_epochs = PrefixPlacementEpochSpan::at(8),
            .completion_movement_epoch = 8,
        }));
    }
}

/** @brief The packed MIN/MAX encoding preserves zero and the full unsigned range. */
TEST(Test__PrefixCacheCoordinatorMPI, PackedAdmissionReductionPreservesUnsignedExtremes)
{
    ASSERT_GE(mpiWorldSize(), 2);
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    MPIPrefixCollectiveCoordinator coordinator(MPI_COMM_WORLD);
    for (int old_rank = 0; old_rank < mpiWorldSize(); ++old_rank)
    {
        PrefixPlacementEpochSpan global;
        ASSERT_TRUE(coordinator.allPlacementEpochs(
            PrefixPlacementEpochSpan::at(mpiWorldRank() == old_rank ? 0 : maximum),
            &global));
        EXPECT_EQ(global, PrefixPlacementEpochSpan::covering(0, maximum));
    }
}

TEST(Test__PrefixCacheCoordinatorMPI, FingerprintMismatchAcrossRanksBypassesCommonHit)
{
    if (mpiWorldSize() < 2)
        GTEST_SKIP() << "requires at least two MPI ranks";

    const int rank = mpiWorldRank();
    const uint64_t fingerprint = rank == 0 ? 0x1000u : 0x2000u;

    MPIPrefixCollectiveCoordinator coordinator(MPI_COMM_WORLD);
    auto result = coordinatePrefixLookups(
        {rankParticipant(rank, /*tokens=*/8, /*terminal_logits=*/true,
                         /*terminal_hidden=*/true, fingerprint)},
        &coordinator);

    EXPECT_FALSE(result.supported);
    EXPECT_EQ(result.common_matched_tokens, 0);
    EXPECT_EQ(result.fingerprint_key, 0u);
    EXPECT_NE(result.clamp_reason.find("fingerprint mismatch"), std::string::npos);
}

TEST(Test__PrefixCacheCoordinatorMPI, RankLocalMissClampsWholeDomainToZero)
{
    if (mpiWorldSize() < 2)
        GTEST_SKIP() << "requires at least two MPI ranks";

    const int rank = mpiWorldRank();
    const int tokens = rank == 0 ? 8 : 0;

    MPIPrefixCollectiveCoordinator coordinator(MPI_COMM_WORLD);
    auto result = coordinatePrefixLookups(
        {rankParticipant(rank, tokens, /*terminal_logits=*/tokens > 0,
                         /*terminal_hidden=*/tokens > 0, 0xabcdu)},
        &coordinator);

    EXPECT_TRUE(result.supported);
    EXPECT_EQ(result.common_matched_tokens, 0);
    EXPECT_FALSE(result.common_terminal_logits);
    EXPECT_FALSE(result.common_terminal_hidden);
}
