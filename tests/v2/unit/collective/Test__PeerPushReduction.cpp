/**
 * @file Test__PeerPushReduction.cpp
 * @brief Device-free storage and progress partition contracts for small peer reductions.
 *
 * These checks protect arbitrary communicator sizes, exact admission geometry,
 * and disjoint block ownership while native replay proves the compiled device
 * protocol. They do not infer a numerical or performance pass without devices.
 */
#include "collective/PeerPushReduction.h"
#include "collective/RankOrderedPeerPushOptIn.h"
#include "../../utils/PeerPushReplayValues.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>

using llaminar2::PeerPushLayout;

/** @test Membership changes storage linearly without a fixed hardware-count ceiling. */
TEST(PeerPushReduction, StorageCoversEveryOriginalMember)
{
    for (const int ranks : {2,3,4,8,9,32,65,257})
    {
        const auto layout = PeerPushLayout::create(2561,ranks,64);
        EXPECT_EQ(layout.blocks,3u);
        EXPECT_EQ(layout.bank_elements,3072u);
        EXPECT_EQ(layout.inboxBytes(),2u*ranks*3072u*sizeof(float));
        EXPECT_EQ(layout.flagBytes(),std::size_t(ranks)*3*64);
        EXPECT_EQ(layout.endpointBytes(),std::size_t(ranks)*sizeof(llaminar2::PeerPushEndpoint));
    }
}

/** @test Empty/invalid geometry and grids that could deadlock scheduling reject before allocation. */
TEST(PeerPushReduction, InvalidStorageAndUnresidentGridsReject)
{
    EXPECT_THROW(PeerPushLayout::create(0,4,60),std::invalid_argument);
    EXPECT_THROW(PeerPushLayout::create(1,1,60),std::invalid_argument);
    EXPECT_THROW(PeerPushLayout::create(1,4,0),std::invalid_argument);
    EXPECT_THROW(PeerPushLayout::create(1025,4,1),std::invalid_argument);
    EXPECT_THROW(PeerPushLayout::create(SIZE_MAX,4,SIZE_MAX),std::overflow_error);
    EXPECT_NO_THROW(PeerPushLayout::create(1024,4,1));
}

/** @test Changes in live row geometry never assign a byte to another progress lane. */
TEST(PeerPushReduction, ChangingGridsPreserveByteOwnership)
{
    const auto layout=PeerPushLayout::create(65536,8,64);
    for (const std::size_t elements : {1u,33u,1023u,1024u,1025u,2560u,7680u,65536u,33u})
    {
        const auto blocks=PeerPushLayout::blocksFor(elements);
        EXPECT_LE(blocks,layout.blocks);
        EXPECT_EQ(blocks,(elements+1023)/1024);
        for (std::size_t i=0;i<elements;++i)
        {
            const auto owner=i/PeerPushLayout::elements_per_block;
            EXPECT_LT(owner,blocks);
            EXPECT_GE(i,owner*PeerPushLayout::elements_per_block);
            EXPECT_LT(i,(owner+1)*PeerPushLayout::elements_per_block);
        }
    }
    EXPECT_EQ(PeerPushLayout::blocksFor(0),0u);
}

/** @test A stale contribution from any rank changes the replay oracle, including TP2. */
TEST(PeerPushReduction, DirtyReplayRejectsEachStaleParticipant)
{
    for (const int ranks : {2,3,4,8})
        for (int repeat=1;repeat<20;++repeat)
            for (int stale=0;stale<ranks;++stale)
            {
                float expected=0, corrupted=0;
                for (int rank=0;rank<ranks;++rank)
                {
                    expected+=llaminar2::test::peerPushReplayValue(rank,0,1,repeat);
                    corrupted+=llaminar2::test::peerPushReplayValue(rank,0,1,rank==stale?repeat-1:repeat);
                }
                ASSERT_NE(expected,corrupted) << "ranks=" << ranks << " stale=" << stale << " repeat=" << repeat;
            }
}

/** @test Capture and quotation retain one process-lifetime selection. */
TEST(PeerPushReduction, TransportPolicyStaysFrozenAfterEnvironmentChange)
{
    constexpr const char *name = "LLAMINAR_RCCL_RANK_ORDERED_PUSH";
    const char *original = std::getenv(name);
    const auto saved = original ? std::optional{std::string(original)} : std::nullopt;
    const bool selected = llaminar2::rankOrderedPeerPushStartupOptIn();
    const int update = setenv(name, selected ? "0" : "1", 1);
    const bool recorded = llaminar2::rankOrderedPeerPushStartupOptIn();
    const int restore = saved ? setenv(name, saved->c_str(), 1) : unsetenv(name);
    EXPECT_EQ(update, 0);
    EXPECT_EQ(restore, 0);
    EXPECT_EQ(recorded, selected);
}

/** @test Direct-peer opt-in never infers visibility from partial or unqualified inventory. */
TEST(PeerPushReduction, OriginalCliqueAndQualifiedNativeISARequired)
{
    using llaminar2::PeerAccessCoverage;
    using llaminar2::requireRankOrderedPeerPushClique;
    using llaminar2::rankOrderedPeerPushQualifiedISA;

    EXPECT_NO_THROW(requireRankOrderedPeerPushClique(PeerAccessCoverage::Complete));
    EXPECT_THROW(requireRankOrderedPeerPushClique(PeerAccessCoverage::Partial), std::invalid_argument);
    EXPECT_THROW(requireRankOrderedPeerPushClique(PeerAccessCoverage::None), std::invalid_argument);
    EXPECT_THROW(requireRankOrderedPeerPushClique(std::nullopt), std::invalid_argument);
    EXPECT_TRUE(rankOrderedPeerPushQualifiedISA("gfx906"));
    EXPECT_TRUE(rankOrderedPeerPushQualifiedISA("gfx906:sramecc+:xnack-"));
    EXPECT_FALSE(rankOrderedPeerPushQualifiedISA("gfx9060"));
    EXPECT_FALSE(rankOrderedPeerPushQualifiedISA("gfx1100"));
    EXPECT_FALSE(rankOrderedPeerPushQualifiedISA(""));
}
