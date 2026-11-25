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
using llaminar2::planning::proveCompletionOrderInto;
using llaminar2::planning::phaseFits;
using llaminar2::planning::WitnessBaseOccupancy;
using llaminar2::planning::WitnessWorkspace;
using llaminar2::planning::witnessWorkspaceBytes;

/** @test The design's 32K/40K token-equivalent cases follow phase arithmetic exactly. */
TEST(CompletionWitness, ResidentOnlySafeOrderDesignExamples)
{
    // 32 frames, two requests holding 1 each, peaks 32: either needs 31 but
    // only 30 is free -> no order.
    {
        std::vector<WitnessRequest> requests{
            {.generation=1, .record=1, .peak_frames=32, .occupied_frames=1, .preserved=false},
            {.generation=2, .record=2, .peak_frames=32, .occupied_frames=1, .preserved=false},
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
            {.generation=1, .record=1, .peak_frames=32, .occupied_frames=4, .preserved=false},
            {.generation=2, .record=2, .peak_frames=32, .occupied_frames=4, .preserved=false},
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
        {.generation=1, .record=1, .peak_frames=2, .occupied_frames=1, .preserved=false},
        {.generation=2, .record=2, .peak_frames=2, .occupied_frames=1, .preserved=false},
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
        {.generation=1, .record=1, .peak_frames=2, .occupied_frames=1, .preserved=false},
        {.generation=2, .record=2, .peak_frames=2, .occupied_frames=1, .preserved=false},
    };
    CompletionWitness witness = proveCompletionOrder(2, simultaneous);
    EXPECT_FALSE(witness.valid());
    EXPECT_FALSE(witness.refusal.empty());
    // Serial admission where only one request is live at a time proves.
    CompletionWitness serial = proveCompletionOrder(2,
        std::vector<WitnessRequest>{{.generation=1, .record=1, .peak_frames=2, .occupied_frames=1, .preserved=false}});
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
        {.generation=1, .record=1, .peak_frames=4, .occupied_frames=2, .preserved=true},
        {.generation=2, .record=2, .peak_frames=2, .occupied_frames=0, .preserved=false},
    };
    CompletionWitness witness = proveCompletionOrder(4, requests);
    ASSERT_TRUE(witness.valid());
    EXPECT_EQ(witness.steps.front().kind, WitnessStep::Kind::Preserve);
    // Fully occupied preserved request leaves no usable capacity: refusal is
    // correct and must not leave a partial plan.
    std::vector<WitnessRequest> blocked{
        {.generation=1, .record=1, .peak_frames=4, .occupied_frames=4, .preserved=true},
        {.generation=2, .record=2, .peak_frames=2, .occupied_frames=0, .preserved=false},
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
        {.generation=1, .record=1, .peak_frames=3, .occupied_frames=3, .preserved=true},
        {.generation=2, .record=2, .peak_frames=4, .occupied_frames=1, .preserved=false},
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

/** @test Reviewer seq101: witness counts outstanding grants and proposed extensions before mutation. */
TEST(CompletionWitness, WitnessAccountsOutstandingGrantsAndProposedExtensions)
{
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry geometry{40, 64, 3, 32};
    const auto bytes = PhysicalPagePool::physicalBytes(geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, geometry, CPUStoragePlacement::local());
    auto a = pool->tryAdmitWithWitness(32, 1);
    ASSERT_TRUE(a);
    auto b = pool->tryAdmitWithWitness(32, 1);
    ASSERT_TRUE(b);
    // Reviewer seq101 repro: extend before assignment. A alone to 20 is
    // provable (a completion order exists with B at its initial grant), but B
    // to 20 makes combined committed occupancy 40/40 with peaks 32: neither
    // can reach peak, so the second extension must refuse before mutation.
    EXPECT_TRUE(pool->extendWithWitness(*a, 19));
    EXPECT_FALSE(pool->extendWithWitness(*b, 19));
    EXPECT_EQ(a->remainingCommitment(), 20u);
    EXPECT_EQ(b->remainingCommitment(), 1u); // Nothing converted by refusal.
    // Alternating growth at the bound: A can still extend (its order remains),
    // but B's extension would push combined occupancy beyond a provable order.
    EXPECT_TRUE(pool->extendWithWitness(*a, 10));
    EXPECT_FALSE(pool->extendWithWitness(*b, 10));
    EXPECT_EQ(a->remainingCommitment(), 30u);
    EXPECT_EQ(b->remainingCommitment(), 1u);
}

/** @test Impossible geometry and inconsistent requests are typed refusals, never guessed orders. */
TEST(CompletionWitness, InvalidInputsRefuseWithoutInventingSteps)
{
    EXPECT_FALSE(proveCompletionOrder(0, {}).valid());
    {
        std::vector<WitnessRequest> requests{{.generation=1, .record=1, .peak_frames=0, .occupied_frames=0}};
        EXPECT_FALSE(proveCompletionOrder(4, requests).valid());
    }
    {
        std::vector<WitnessRequest> requests{{.generation=1, .record=1, .peak_frames=2, .occupied_frames=3}};
        EXPECT_FALSE(proveCompletionOrder(4, requests).valid());
    }
    {
        std::vector<WitnessRequest> requests{{.generation=1, .record=1, .peak_frames=8, .occupied_frames=1}};
        EXPECT_FALSE(proveCompletionOrder(4, requests).valid());
    }
    {
        std::vector<WitnessRequest> requests{{.generation=1, .record=1, .peak_frames=2, .occupied_frames=1},
            {.generation=2, .record=2, .peak_frames=2, .occupied_frames=1}};
        // Three frames with 1+1 occupied leave one free; the first request
        // advances (needs 1), frees its frame, then the second advances. Valid.
        EXPECT_TRUE(proveCompletionOrder(3, requests).valid());
        // One frame cannot hold even a single occupied request's tail.
        EXPECT_FALSE(proveCompletionOrder(1, requests).valid());
    }
}

/** @test Reviewer seq102: base occupancy (shared/retiring frames) blocks admission proof. */
TEST(CompletionWitness, PhysicalBaseOccupancyBlocksUnprovableAdmission)
{
    const auto device = DeviceId::cpu();
    const PhysicalPagePoolGeometry geometry{4, 64, 4, 4};
    const auto bytes = PhysicalPagePool::physicalBytes(geometry, device);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0, .device=device, .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::KVCache, bytes);
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
    auto pool = authority->createCPUPagePool(device, geometry, CPUStoragePlacement::local());
    // Three immutable shared frames retained outside any live request: they are
    // canonical physical occupancy and must count in the completion proof.
    std::vector<PhysicalPagePool::SharedPage> shared;
    for (int i = 0; i < 3; ++i)
    {
        auto owner = pool->tryAdmit(1);
        ASSERT_TRUE(owner);
        std::vector<std::byte> payload(8, static_cast<std::byte>(i + 1));
        const auto frame = owner->assignFrame();
        auto write = owner->prepareFreshWrite(frame, payload.size());
        ASSERT_TRUE(write);
        write->copyFrom(payload);
        shared.emplace_back(write->commit().share());
        owner.reset(); // Share is retained; the request record retires.
    }
    auto base = pool->physicalBaseOccupancy();
    EXPECT_EQ(base.shared_frames, 3u);
    // A new peak-4 request with one granted frame cannot be witness-admitted:
    // the proof must see 3 shared base + 1 new + peak 4 requiring 4 free.
    auto candidate = pool->tryAdmitWithWitness(4, 1);
    EXPECT_FALSE(candidate);
    // Peak-2 also refuses: base 3 + grant 1 = 4 occupied, so the request can
    // never obtain its second frame without unrelated owners releasing.
    auto small = pool->tryAdmitWithWitness(2, 1);
    EXPECT_FALSE(small);
    // Releasing the base occupancy re-opens exactly one admission.
    shared.clear();
    auto after = pool->tryAdmitWithWitness(2, 1);
    EXPECT_TRUE(after);
    (void)after->assignFrame();
}

/** @test Reviewer seq103 repros: overflow base, no arbitrary caps, stale cells and OOB guard. */
TEST(CompletionWitness, WitnessStorageAndOverflowContractsHold)
{
    // (1) witnessWorkspaceBytes(SIZE_MAX) must refuse, not accept wrapped sizes.
    EXPECT_THROW(planning::witnessWorkspaceBytes(std::numeric_limits<std::size_t>::max()),
        std::overflow_error);
    EXPECT_THROW(planning::witnessWorkspaceBytes(std::numeric_limits<std::size_t>::max() / 2),
        std::overflow_error);
    // Membership-sized accounting: exact cells for small membership.
    const std::size_t small = planning::witnessWorkspaceBytes(3);
    EXPECT_GT(small, 0u);

    // (2) base occupancy {SIZE_MAX, 1} must refuse (total overflows), not wrap to 0.
    CompletionWitness wrapped = proveCompletionOrder(1,
        WitnessBaseOccupancy{.shared_frames=std::numeric_limits<std::size_t>::max(),
            .retiring_frames=1}, {});
    EXPECT_FALSE(wrapped.valid());
    EXPECT_FALSE(wrapped.refusal.empty());
    // Empty requests with a valid base still prove.
    CompletionWitness empty_ok = proveCompletionOrder(1,
        WitnessBaseOccupancy{.shared_frames=1, .retiring_frames=0}, {});
    EXPECT_TRUE(empty_ok.valid());

    // (3) after success then refusal, the workspace must not retain prior steps.
    WitnessRequest one{.generation=1, .record=1, .peak_frames=1, .occupied_frames=1};
    WitnessWorkspace ws({}, {}, {});
    std::vector<WitnessRequest> cells(4);
    std::vector<WitnessStep> steps(9);
    std::vector<char> refusal(256);
    ws = WitnessWorkspace({cells.data(), cells.size()}, {steps.data(), steps.size()},
        {refusal.data(), refusal.size()});
    // Success: one request at occupied==peak completes.
    EXPECT_TRUE(proveCompletionOrderInto(ws, 1, {}, std::span<const WitnessRequest>(&one, 1)));
    EXPECT_GT(ws.steps().size(), 0u);
    // Refusal: zero frames -> no steps may remain.
    EXPECT_FALSE(proveCompletionOrderInto(ws, 0, {}, std::span<const WitnessRequest>(&one, 1)));
    EXPECT_TRUE(std::all_of(ws.steps().begin(), ws.steps().end(),
        [](const WitnessStep &s) { return s.kind == WitnessStep::Kind::RetireResident &&
            s.frames_required == 0 && s.frames_freed == 0; }));
    EXPECT_FALSE(std::string(ws.refusal()).empty());

    // (4) oversized request span against workspace capacity must refuse, never OOB.
    std::vector<WitnessRequest> one_cell(1);
    std::vector<WitnessStep> three_steps(3);
    std::vector<char> small_refusal(256);
    WitnessWorkspace tiny({one_cell.data(), one_cell.size()}, {three_steps.data(), three_steps.size()},
        {small_refusal.data(), small_refusal.size()});
    std::vector<WitnessRequest> many(4, one);
    EXPECT_FALSE(proveCompletionOrderInto(tiny, 4, {}, std::span<const WitnessRequest>(many)));
    EXPECT_FALSE(std::string(tiny.refusal()).empty());
}
