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
#include "planning/PhysicalPagePool.h"
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace llaminar2;

using llaminar2::planning::CompletionWitness;
using llaminar2::planning::WitnessRequest;
using llaminar2::planning::WitnessStep;
using llaminar2::planning::proveCompletionOrder;
using llaminar2::planning::phaseFits;

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

/** @test Preservation lets a request continue at its current extent while another advances. */
TEST(CompletionWitness, PreservedRequestContinuesAtCurrentExtent)
{
    // Preserved request holds 2 of 4 peak frames; the second request advances
    // into the remaining 2 without the first releasing anything.
    std::vector<WitnessRequest> requests{
        {.generation=1, .peak_frames=4, .occupied_frames=2, .preserved=true},
        {.generation=2, .peak_frames=2, .occupied_frames=0, .preserved=false},
    };
    CompletionWitness witness = proveCompletionOrder(4, requests);
    ASSERT_TRUE(witness.valid());
    EXPECT_EQ(witness.steps.front().kind, WitnessStep::Kind::Preserve);
    // Fully occupied preserved request leaves no usable capacity: refusal is
    // correct and must not leave a partial plan.
    std::vector<WitnessRequest> blocked{
        {.generation=1, .peak_frames=4, .occupied_frames=4, .preserved=true},
        {.generation=2, .peak_frames=2, .occupied_frames=0, .preserved=false},
    };
    CompletionWitness refused = proveCompletionOrder(4, blocked);
    EXPECT_FALSE(refused.valid());
    EXPECT_TRUE(refused.steps.empty());
}

/** @test Fail-closed: refusal leaves no partial steps; checked arithmetic never overflows. */
TEST(CompletionWitness, RefusalClearsPartialPlanAndPhaseArithmeticIsChecked)
{
    // Reviewer repro 1: phaseFits(SIZE_MAX, SIZE_MAX, 1) must be false.
    EXPECT_FALSE(phaseFits(std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<std::size_t>::max(), 1));
    EXPECT_FALSE(phaseFits(std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()));
    EXPECT_TRUE(phaseFits(4, 1, 3));
    EXPECT_FALSE(phaseFits(4, 1, 4));

    // Reviewer repro 2: a preserved request's Preserve step must not survive a
    // later refusal as a usable partial plan.
    std::vector<WitnessRequest> requests{
        {.generation=1, .peak_frames=3, .occupied_frames=3, .preserved=true},
        {.generation=2, .peak_frames=4, .occupied_frames=1, .preserved=false},
    };
    CompletionWitness witness = proveCompletionOrder(4, requests);
    EXPECT_FALSE(witness.valid());
    EXPECT_TRUE(witness.steps.empty()); // No lingering Preserve step after refusal.
    EXPECT_FALSE(witness.refusal.empty());
}

/** @test tryAdmitWithWitness uses finite grants plus a proved order, never summed maxima. */
TEST(CompletionWitness, PoolAdmissionUsesFiniteGrantsWithProvedOrder)
{
    const auto device = DeviceId::cpu();
    // 32-frame pool: two requests with peak 32 each holding one frame cannot
    // both be witness-admitted (order proof fails); refusal mutates nothing.
    const PhysicalPagePoolGeometry geometry32{32, 64, 4, 32};
    const auto bytes32 = PhysicalPagePool::physicalBytes(geometry32, device);
    PhysicalMemoryPlanBuilder plan32;
    plan32.add({.world_rank=0, .device=device, .total_bytes=bytes32, .admission_available_bytes=bytes32},
        PhysicalMemoryOwner::KVCache, bytes32);
    auto authority32 = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan32.build()), 0);
    auto pool32 = authority32->createCPUPagePool(device, geometry32, CPUStoragePlacement::local());
    auto first = pool32->tryAdmitWithWitness(32, 1);
    ASSERT_TRUE(first);
    (void)first->assignFrame();
    // Design §5.3: with A holding 1K, B's finite 1K admission enters with 1K
    // occupied, so A would need 31K but only 30K is free -> refused exactly.
    EXPECT_FALSE(pool32->tryAdmitWithWitness(32, 1));
    EXPECT_EQ(first->assignedFrames().size(), 1u);

    // Design §5.3's 40K example: two peak-32 requests holding four each.
    // Finite initial grants (1 each) plus a proved completion order admit both;
    // summed maxima (32+32 > 40) never blocks because no request converts its
    // whole peak before an order exists.
    const PhysicalPagePoolGeometry geometry40{40, 64, 4, 32};
    const auto bytes40 = PhysicalPagePool::physicalBytes(geometry40, device);
    PhysicalMemoryPlanBuilder plan40;
    plan40.add({.world_rank=0, .device=device, .total_bytes=bytes40, .admission_available_bytes=bytes40},
        PhysicalMemoryOwner::KVCache, bytes40);
    auto authority40 = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan40.build()), 0);
    auto pool40 = authority40->createCPUPagePool(device, geometry40, CPUStoragePlacement::local());
    auto a = pool40->tryAdmitWithWitness(32, 1);
    ASSERT_TRUE(a);
    auto b = pool40->tryAdmitWithWitness(32, 1);
    ASSERT_TRUE(b);
    // Extend each to its occupied four without summing maxima: the order proof
    // still holds (each advance fits after the other's current extent).
    EXPECT_TRUE(pool40->extendWithWitness(*a, 3));
    EXPECT_TRUE(pool40->extendWithWitness(*b, 3));
    for (int i = 0; i < 4; ++i) (void)a->assignFrame();
    for (int i = 0; i < 4; ++i) (void)b->assignFrame();
    EXPECT_EQ(a->assignedFrames().size(), 4u);
    EXPECT_EQ(b->assignedFrames().size(), 4u);
}

/** @test Witness extension refuses once the order would break or the peak is exceeded. */
TEST(CompletionWitness, WitnessExtensionRefusesWhenOrderOrPeakBreaks)
{
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry geometry40{40, 64, 4, 32};
    const auto bytes40 = PhysicalPagePool::physicalBytes(geometry40, device);
    PhysicalMemoryPlanBuilder plan40;
    plan40.add({.world_rank=0, .device=device, .total_bytes=bytes40, .admission_available_bytes=bytes40},
        PhysicalMemoryOwner::KVCache, bytes40);
    auto authority40 = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan40.build()), 0);
    auto pool40 = authority40->createCPUPagePool(device, geometry40, CPUStoragePlacement::local());
    auto a = pool40->tryAdmitWithWitness(32, 1);
    ASSERT_TRUE(a);
    EXPECT_THROW(pool40->extendWithWitness(*a, 32), std::logic_error); // Beyond declared peak.
    EXPECT_TRUE(pool40->extendWithWitness(*a, 31));                     // Exactly the peak.
    EXPECT_THROW(pool40->extendWithWitness(*a, 1), std::logic_error);   // Peak already reached.
}

/** @test New admission and witness extension stop before exhaustion; draining permits the next request. */
TEST(CompletionWitness, AdmissionStopsBeforeExhaustionAndDrainingResumesIt)
{
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry geometry{2, 64, 3, 2};
    const auto bytes = PhysicalPagePool::physicalBytes(geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, geometry, CPUStoragePlacement::local());
    auto first = pool->tryAdmitWithWitness(1, 1);
    ASSERT_TRUE(first);
    (void)first->assignFrame();
    // Second witness admission fits (one free frame).
    auto second = pool->tryAdmitWithWitness(1, 1);
    ASSERT_TRUE(second);
    (void)second->assignFrame();
    // Pool exhausted: a new witness admission must refuse before mutation.
    auto third = pool->tryAdmitWithWitness(1, 1);
    EXPECT_FALSE(third);
    // Draining one request re-opens exactly one admission.
    first = PhysicalPagePool::Grant();
    auto fourth = pool->tryAdmitWithWitness(1, 1);
    ASSERT_TRUE(fourth);
    (void)fourth->assignFrame();
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
