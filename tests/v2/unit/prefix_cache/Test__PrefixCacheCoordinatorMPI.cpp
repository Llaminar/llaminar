#include "execution/prefix_cache/PrefixCacheCoordinator.h"

#include <gtest/gtest.h>
#include <mpi.h>

using namespace llaminar2;

namespace
{
    PrefixParticipantLookup rankParticipant(
        int rank,
        int tokens,
        bool terminal_logits,
        bool terminal_hidden,
        uint64_t fingerprint)
    {
        PrefixParticipantLookup lookup;
        lookup.domain_id = "mpi-prefix-domain";
        lookup.participant_id = rank;
        lookup.device = DeviceId::cpu();
        lookup.placement_epoch = 11;
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

    int mpiWorldSize()
    {
        int size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }

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
