/**
 * @file Test__MTPPublicationWitness.cpp
 * @brief Authored device-free source-count and v3 publication-bit regressions.
 *
 * These expectations distinguish emitted outputs from published state without
 * a model, a copied controller, GPU work or fabricated historical traces. Full
 * producer and baseline replay coverage remains required independently.
 */
#include "execution/mtp/MTPPublicationWitness.h"
#include "kernels/common/MTPDepthTransactionTrace.h"
#include <gtest/gtest.h>
#include <array>
#include <climits>
#include <cstdint>
using namespace llaminar2;
using namespace llaminar2::sampling_math;

/** @test Actual counts distinguish a fresh rejection, carried full and stopped correction. */
TEST(MTPPublicationWitness, ActualCountConservationIncludesStoppedDebt)
{
    struct Case { int output,leading,published,emitted; bool stopped; };
    const Case cases[]={{5,0,4,5,false},{5,1,5,4,false},
        {2,1,1,1,false},{2,0,1,2,true},{1,0,0,1,true}};
    for (const auto &c : cases)
    {
        MTPPublicationWitness w;
        ASSERT_TRUE(MTPPublicationWitness::fromCounts(c.output,c.leading,c.published,c.stopped,&w));
        EXPECT_TRUE(w.valid());EXPECT_EQ(w.newlyEmittedTokens(),c.emitted);
        EXPECT_EQ(w.publishedStateTokens(),c.published);EXPECT_EQ(w.incomingDebt(),c.leading!=0);
        EXPECT_EQ(w.outgoingDebt(),c.output-c.published!=0);EXPECT_EQ(w.modelStopped(),c.stopped);
        MTPPublicationWitness decoded;
        ASSERT_TRUE(MTPPublicationWitness::fromEmitted(w.newlyEmittedTokens(),w.incomingDebt(),
            w.outgoingDebt(),w.modelStopped(),&decoded));
        EXPECT_EQ(decoded.publishedStateTokens(),w.publishedStateTokens());
    }
}
/** @test Invalid source geometry never narrows to bool or overwrites prior valid output. */
TEST(MTPPublicationWitness, InvalidCountsAndOverflowFailWithoutMutation)
{
    MTPPublicationWitness w;EXPECT_FALSE(w.valid());
    ASSERT_TRUE(MTPPublicationWitness::fromCounts(5,1,4,false,&w));
    const int invalid[][3]={{0,0,0},{2,-1,1},{2,2,1},{1,1,1},{2,0,-1},{2,0,3},{16,0,0}};
    for(const auto &bad : invalid)
    {
        EXPECT_FALSE(MTPPublicationWitness::fromCounts(bad[0],bad[1],bad[2],true,&w));
        EXPECT_EQ(w.newlyEmittedTokens(),4u);EXPECT_EQ(w.publishedStateTokens(),4u);
        EXPECT_FALSE(w.modelStopped());
    }
    EXPECT_FALSE(MTPPublicationWitness::fromCounts(1,0,1,false,nullptr));
    EXPECT_FALSE(MTPPublicationWitness::fromEmitted(UINT32_MAX,false,false,false,&w));
    EXPECT_FALSE(MTPPublicationWitness::fromEmitted(INT_MAX,true,false,false,&w));
    EXPECT_FALSE(MTPPublicationWitness::fromEmitted(0,false,false,false,&w));
    ASSERT_TRUE(MTPPublicationWitness::fromCounts(INT_MAX,0,INT_MAX-1,false,&w));
    EXPECT_EQ(w.newlyEmittedTokens(),static_cast<uint32_t>(INT_MAX));
    ASSERT_TRUE(MTPPublicationWitness::fromCounts(32,1,31,false,&w)); // No trace-depth cap in general publication.
    EXPECT_EQ(w.newlyEmittedTokens(),31u);
}
/** @test All actual debt/stop combinations round trip across supported trace depths and owners. */
TEST(MTPPublicationWitness, TraceV3BitsRoundTripWithoutChangingPhysicalExtent)
{
    EXPECT_EQ(kMTPDepthTransactionTraceVersion,3u);
    EXPECT_EQ(sizeof(MTPDepthTransactionTraceRecord),12u);EXPECT_EQ(sizeof(MTPDepthTransactionTrace),3520u);
    for(int depth=1;depth<=15;++depth) for(bool host:{false,true})
        for(bool incoming:{false,true}) for(bool outgoing:{false,true}) for(bool stop:{false,true})
            for(int y=1;y<=depth+1-static_cast<int>(incoming);++y)
    {
        const MTPDepthTransactionTraceValues v{.pre_depth=depth,.accepted_prefix=0,.post_depth=depth,
            .incoming_debt=incoming,.outgoing_debt=outgoing,.model_stopped=stop,
            .requested_depth=host?depth:0,.effective_depth=host?depth:0,.host_observation=host,
            .recommended_depth=depth,.token_witness=y,.evaluated_windows=0};
        MTPDepthTransactionTraceRecord record{};ASSERT_TRUE(packMTPDepthTransactionTraceRecord(v,&record));
        EXPECT_EQ((record.packed>>29)&1u,static_cast<unsigned>(incoming));
        EXPECT_EQ((record.packed>>30)&1u,static_cast<unsigned>(outgoing));
        EXPECT_EQ((record.packed>>31)&1u,static_cast<unsigned>(stop));
        MTPDepthTransactionTraceValues decoded;ASSERT_TRUE(unpackMTPDepthTransactionTraceRecord(record,&decoded));
        EXPECT_EQ(decoded.incoming_debt,incoming);EXPECT_EQ(decoded.outgoing_debt,outgoing);
        EXPECT_EQ(decoded.model_stopped,stop);EXPECT_EQ(decoded.token_witness,y);
    }
}
