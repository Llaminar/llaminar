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

/** @test A non-primary rank's metadata-only MTP/disk section survives one-word reduction. */
TEST(Test__PrefixCacheCoordinatorMPI, RestoredSectionsAndTiersIncludeEveryRank)
{
    ASSERT_GE(mpiWorldSize(), 2);
    MPIPrefixCollectiveCoordinator coordinator(MPI_COMM_WORLD);
    for (int owner = 0; owner < mpiWorldSize(); ++owner)
    {
        std::vector<PrefixBlockHandle> blocks(1);
        blocks[0].layout.includes_mtp_state = mpiWorldRank() == owner;
        blocks[0].has_hybrid_state = mpiWorldRank() != owner;
        blocks[0].tier = mpiWorldRank() == owner ? PrefixStorageTier::Disk : PrefixStorageTier::Ram;
        PrefixRestoreMetadata global;
        ASSERT_TRUE(coordinator.allRestoreMetadata(PrefixRestoreMetadata::fromBlocks(blocks), &global));
        EXPECT_TRUE(global.hasMTPState());
        EXPECT_TRUE(global.hasHybridState());
        EXPECT_STREQ(global.storageTierName(), "mixed");
        // A later all-rank miss must clear the old observation completely.
        ASSERT_TRUE(coordinator.allRestoreMetadata({}, &global));
        EXPECT_EQ(global.wireBits(), 0u);
    }
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

/** @brief A recurrent shard imposes its cold-prefill checkpoint on every rank. */
TEST(Test__PrefixCacheCoordinatorMPI, ReusableCheckpointPolicySurvivesColdNestedCoordination)
{
    ASSERT_GE(mpiWorldSize(), 2);
    for (int recurrent_rank = 0; recurrent_rank < mpiWorldSize(); ++recurrent_rank)
    {
        auto participant = rankParticipant(mpiWorldRank(), 0, false, false, 0xabcdu);
        participant.checkpoint_policy = mpiWorldRank() == recurrent_rank
            ? PrefixCheckpointPolicy::ReusableBoundary
            : PrefixCheckpointPolicy::TerminalOnly;
        const auto local = makePrefixLookupResult(coordinatePrefixLookups({participant}), 64);
        MPIPrefixCollectiveCoordinator coordinator(MPI_COMM_WORLD);
        const auto global = makePrefixLookupResult(coordinatePrefixLookups({
            makePrefixParticipantLookup(mpiWorldRank(), DeviceId::cpu(), local)
        }, &coordinator), 64);
        EXPECT_EQ(global.cached_tokens, 0);
        EXPECT_EQ(global.checkpoint_policy, PrefixCheckpointPolicy::ReusableBoundary);
        EXPECT_EQ(global.reusablePrefillCheckpoint(365, 0), 320);
        EXPECT_EQ(PrefixHarvestSchedule::forPrefill(global, 20416, 0).reusableCheckpoints(),
                  std::vector<int>({4096, 8192, 16384, 20352}));
    }
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

/**
 * @test Original stage ownership stays local while shared terminal admission
 * follows the vocabulary tail, including missing owner state and reset replay.
 * The final minimum models independently observed runtime/window constraints
 * after that shared lookup; it is committed before any stage may restore.
 */
TEST(Test__PrefixCacheCoordinatorMPI, OriginalPipelineRequiredOwnerAdmissionReplay)
{
    ASSERT_EQ(mpiWorldSize(), 2);
    const int rank = mpiWorldRank();
    MPIPrefixCollectiveCoordinator coordinator(MPI_COMM_WORLD);
    for (int replay = 0; replay < 20; ++replay)
    {
        for (int tail_rank = 0; tail_rank < 2; ++tail_rank)
        {
            for (int scenario = 0; scenario < 6; ++scenario)
            {
                const bool tail = rank == tail_rank;
                const int tokens = scenario == 3 && !tail ? 64 : 96;
                auto participant = rankParticipant(
                    rank, tokens,
                    tail && scenario != 2,
                    tail && scenario != 1,
                    0xabcdu + static_cast<uint64_t>(rank),
                    static_cast<uint64_t>(replay));
                participant.requires_terminal_logits = tail;
                participant.requires_terminal_hidden = tail;
                participant.fingerprint_policy =
                    PrefixFingerprintCoordinationPolicy::ValidateParticipantLocally;
                const auto local = makePrefixLookupResult(
                    coordinatePrefixLookups({participant}), 32);
                EXPECT_EQ(local.requires_terminal_hidden, tail);
                EXPECT_EQ(local.has_terminal_hidden, tail && scenario != 1);
                const auto shared = makePrefixLookupResult(coordinatePrefixLookups({
                    makePrefixParticipantLookup(rank, DeviceId::cpu(), local, {},
                        PrefixFingerprintCoordinationPolicy::ValidateParticipantLocally)
                }, &coordinator), 32);
                ASSERT_TRUE(shared.supported);
                EXPECT_EQ(shared.cached_tokens, scenario == 3 ? 64 : 96);
                EXPECT_TRUE(shared.requires_terminal_logits);
                EXPECT_TRUE(shared.requires_terminal_hidden);
                EXPECT_EQ(shared.has_terminal_logits, scenario != 2);
                EXPECT_EQ(shared.has_terminal_hidden, scenario != 1);
                // Shared evidence never changes this participant's physical row.
                EXPECT_EQ(local.has_terminal_hidden, tail && scenario != 1);
                EXPECT_EQ(shared.placement_epochs,
                          PrefixPlacementEpochSpan::at(static_cast<uint64_t>(replay)));

                // Each stage proves its own image; one missing image removes
                // the full-hit stable-window exemption for the whole pipeline.
                const bool local_runtime =
                    !(scenario == 4 && tail) && !(scenario == 5 && !tail);
                bool all_runtime = false;
                ASSERT_TRUE(coordinator.allAndBool(local_runtime, &all_runtime));
                EXPECT_EQ(all_runtime, scenario < 4);
                int candidate = shared.cached_tokens;
                if (scenario == 1 || scenario == 2)
                    candidate -= 32;
                if (!all_runtime)
                    candidate = (candidate / 64) * 64;
                int committed = -1;
                ASSERT_TRUE(coordinator.allMinInt(candidate, &committed));
                bool all_admitted = false;
                ASSERT_TRUE(coordinator.allAndBool(candidate == committed, &all_admitted));
                EXPECT_TRUE(all_admitted);
                EXPECT_EQ(committed, scenario == 0 ? 96 : 64);

                // Incompatible immutable routing windows, or an endpoint that
                // cannot restore the proposal, reject on BOTH ranks before
                // either may import state or submit its first suffix edge.
                for (const bool missing_endpoint : {false, true})
                {
                    const int incompatible_candidate = rank == 0 ? 64 : 96;
                    ASSERT_TRUE(coordinator.allMinInt(
                        missing_endpoint ? 64 : incompatible_candidate, &committed));
                    const bool local_admitted = missing_endpoint
                        ? rank != tail_rank
                        : incompatible_candidate == committed;
                    ASSERT_TRUE(coordinator.allAndBool(local_admitted, &all_admitted));
                    EXPECT_FALSE(all_admitted);
                }
            }
        }
    }
}
