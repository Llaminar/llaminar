/**
 * @file Test__CompletionWitness.cpp
 * @brief Exact-fit safe-order arithmetic from design §5.3, as contract tests.
 *
 * The design's analytical examples use resident-only safe order: two 32K-max
 * requests holding 1K each cannot both be admitted with a 32K pool (either
 * needs 31K but only 30K is free); with 40K and 4K-held each, 32K is free and
 * one needs 28K, so an order exists. These are expansion examples without
 * sharing/transients; the real witness uses the complete resource vector.
 */
#include "planning/CompletionWitness.h"
#include <gtest/gtest.h>
#include <vector>

using llaminar2::planning::CompletionWitness;
using llaminar2::planning::WitnessRequest;
using llaminar2::planning::WitnessStep;
using llaminar2::planning::proveCompletionOrder;

/** @test The design's 32K/40K token-equivalent cases follow phase arithmetic exactly. */
TEST(CompletionWitness, ResidentOnlySafeOrderDesignExamples)
{
    // 32 frames, two requests holding 1 each, peaks 32: either needs 31 but
    // only 30 is free -> no order.
    {
        std::vector<WitnessRequest> requests{
            {.generation=1, .peak_frames=32, .occupied_frames=1, .preserved=false},
            {.generation=2, .peak_frames=32, .occupied_frames=1, .preserved=false},
        };
        CompletionWitness witness = proveCompletionOrder(32, requests);
        EXPECT_FALSE(witness.valid());
        EXPECT_EQ(witness.steps.size(), 0u);
        EXPECT_FALSE(witness.refusal.empty());
    }
    // 40 frames, two requests holding 4 each, peaks 32: 32 free, one needs 28
    // -> order exists, and the second request finishes after the first.
    {
        std::vector<WitnessRequest> requests{
            {.generation=1, .peak_frames=32, .occupied_frames=4, .preserved=false},
            {.generation=2, .peak_frames=32, .occupied_frames=4, .preserved=false},
        };
        CompletionWitness witness = proveCompletionOrder(40, requests);
        ASSERT_TRUE(witness.valid());
        EXPECT_EQ(witness.refusal, "");
        EXPECT_EQ(witness.steps.size(), 4u);
    }
}

/** @test Three-frame exact-fit COW example: simultaneous grants, two-frame refusal. */
TEST(CompletionWitness, ExactFitSharedTailThreeFramePool)
{
    // O=1 (nine sealed tokens), two writers require one private replacement each
    // to reach their tail: three frames admit both; two frames cannot.
    std::vector<WitnessRequest> simultaneous{
        {.generation=1, .peak_frames=2, .occupied_frames=1, .preserved=false},
        {.generation=2, .peak_frames=2, .occupied_frames=1, .preserved=false},
    };
    ASSERT_TRUE(proveCompletionOrder(3, simultaneous).valid());
    EXPECT_FALSE(proveCompletionOrder(2, simultaneous).valid());
}

/** @test Frame-subset witness conservatively refuses the simultaneous two-frame plan. */
TEST(CompletionWitness, TwoFrameSimultaneousPlanRefusesInFrameSubset)
{
    // Each writer needs one destination reaching a two-frame peak while both
    // are live; the two-frame pool cannot hold both. The frame-subset witness
    // refuses (typed), because proving the sequential order additionally
    // requires the complete all-resource witness (design §5.3), not a guessed
    // order in this subset. The three-frame simultaneous plan proves.
    std::vector<WitnessRequest> simultaneous{
        {.generation=1, .peak_frames=2, .occupied_frames=1, .preserved=false},
        {.generation=2, .peak_frames=2, .occupied_frames=1, .preserved=false},
    };
    CompletionWitness witness = proveCompletionOrder(2, simultaneous);
    EXPECT_FALSE(witness.valid());
    EXPECT_FALSE(witness.refusal.empty());
    // Serial admission where only one request is live at a time proves.
    CompletionWitness serial = proveCompletionOrder(2,
        std::vector<WitnessRequest>{{.generation=1, .peak_frames=2, .occupied_frames=1, .preserved=false}});
    ASSERT_TRUE(serial.valid());
    EXPECT_EQ(serial.steps.front().kind, WitnessStep::Kind::Advance);
    EXPECT_EQ(serial.steps.back().kind, WitnessStep::Kind::FinalRetire);
}

/** @test Preservation lets a request continue without reaching a peak in this witness. */
TEST(CompletionWitness, PreservedRequestContinuesAtCurrentExtent)
{
    std::vector<WitnessRequest> requests{
        {.generation=1, .peak_frames=4, .occupied_frames=4, .preserved=true},
        {.generation=2, .peak_frames=2, .occupied_frames=0, .preserved=false},
    };
    CompletionWitness witness = proveCompletionOrder(4, requests);
    ASSERT_TRUE(witness.valid());
    EXPECT_EQ(witness.steps.front().kind, WitnessStep::Kind::Preserve);
}

/** @test Impossible geometry and inconsistent requests are typed refusals, never guessed orders. */
TEST(CompletionWitness, InvalidInputsRefuseWithoutInventingSteps)
{
    EXPECT_FALSE(proveCompletionOrder(0, {}).valid());
    {
        std::vector<WitnessRequest> requests{{.generation=1, .peak_frames=0, .occupied_frames=0}};
        EXPECT_FALSE(proveCompletionOrder(4, requests).valid());
    }
    {
        std::vector<WitnessRequest> requests{{.generation=1, .peak_frames=2, .occupied_frames=3}};
        EXPECT_FALSE(proveCompletionOrder(4, requests).valid());
    }
    {
        std::vector<WitnessRequest> requests{{.generation=1, .peak_frames=8, .occupied_frames=1}};
        EXPECT_FALSE(proveCompletionOrder(4, requests).valid());
    }
    {
        std::vector<WitnessRequest> requests{{.generation=1, .peak_frames=2, .occupied_frames=1},
            {.generation=2, .peak_frames=2, .occupied_frames=1}};
        // Three frames with 1+1 occupied leave one free; the first request
        // advances (needs 1), frees its frame, then the second advances. Valid.
        EXPECT_TRUE(proveCompletionOrder(3, requests).valid());
        // One frame cannot hold even a single occupied request's tail.
        EXPECT_FALSE(proveCompletionOrder(1, requests).valid());
    }
}
