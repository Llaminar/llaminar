/**
 * @file Test__AllResourceCompletionWitness.cpp
 * @brief Composed all-resource completion witness contracts.
 *
 * These tests prove the composed frame + bank + lease completion order: an
 * admission whose classes all fit is valid; a class over-demand refuses with
 * a precise diagnostic; and the witness is pure (no live counter, no physical
 * accounting — PMA remains the sole authority).
 */
#include "planning/AllResourceCompletionWitness.h"
#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
AllResourceCapacity capacityFor(std::size_t frames, std::size_t banks,
    std::size_t leases, std::vector<AllResourceRequest> requests = {})
{
    AllResourceCapacity c;
    c.frames = frames;
    c.banks = banks;
    c.lease_cells = leases;
    c.requests = std::move(requests);
    return c;
}
AllResourceRequest request(std::size_t frames, std::size_t banks, std::size_t leases,
    bool preserved = false, std::uint64_t generation = 1)
{
    AllResourceRequest r;
    r.generation = generation;
    r.peak_frames = frames;
    r.occupied_frames = frames;
    r.peak_banks = banks;
    r.peak_leases = leases;
    r.preserved = preserved;
    return r;
}
}

/** @test All classes fitting is admitted with a bounded completion order. */
TEST(AllResourceCompletionWitness, AllClassesFitIsValid)
{
    auto c = capacityFor(4, 2, 4, {request(2, 1, 2), request(2, 1, 2)});
    auto result = proveAllResourceCompletion(c);
    EXPECT_TRUE(result.valid) << result.refusal;
    EXPECT_TRUE(result.refusal.empty());
}

/** @test Bank over-demand refuses with the precise class diagnostic. */
TEST(AllResourceCompletionWitness, BankOverDemandRefuses)
{
    auto c = capacityFor(4, 1, 4, {request(2, 1, 2), request(2, 1, 2)});
    auto result = proveAllResourceCompletion(c);
    EXPECT_FALSE(result.valid);
    EXPECT_NE(result.refusal.find("bank"), std::string::npos) << result.refusal;
}

/** @test Lease-cell over-demand refuses independently of frames. */
TEST(AllResourceCompletionWitness, LeaseOverDemandRefuses)
{
    auto c = capacityFor(8, 4, 2, {request(2, 1, 2), request(2, 1, 2)});
    auto result = proveAllResourceCompletion(c);
    EXPECT_FALSE(result.valid);
    EXPECT_NE(result.refusal.find("lease"), std::string::npos) << result.refusal;
}

/** @test Frame over-demand refuses through the composed frame proof. */
TEST(AllResourceCompletionWitness, FrameOverDemandRefuses)
{
    auto c = capacityFor(3, 4, 8, {request(2, 1, 2), request(2, 1, 2)});
    auto result = proveAllResourceCompletion(c);
    EXPECT_FALSE(result.valid);
    EXPECT_FALSE(result.refusal.empty());
}

/** @test Zero/invalid class maxima are rejected typed. */
TEST(AllResourceCompletionWitness, InvalidMaximaRejectedTyped)
{
    auto zero = capacityFor(0, 2, 4, {request(1, 1, 1)});
    EXPECT_FALSE(proveAllResourceCompletion(zero).valid);
    auto no_banks = capacityFor(4, 0, 4, {request(1, 1, 1)});
    EXPECT_FALSE(proveAllResourceCompletion(no_banks).valid);
}

/** @test Overlapping shared base occupancy is counted once, exact-boundary valid. */
TEST(AllResourceCompletionWitness, SharedBaseCountedOnceAtBoundary)
{
    AllResourceCapacity c;
    c.frames = 5;
    c.banks = 1;
    c.lease_cells = 2;
    c.base.shared_frames = 1;   // One shared frame outside requests.
    c.base.retiring_frames = 1; // One retiring frame outside requests.
    c.requests = {request(2, 1, 2), request(1, 0, 0)};
    auto result = proveAllResourceCompletion(c);
    EXPECT_TRUE(result.valid) << result.refusal;
}
