/**
 * @file Test__AdmissionPressureGate.cpp
 * @brief Pressure-admission gate contracts (plan §5.1 / P5 first step).
 *
 * These tests prove the gate admits only when the proposed all-resource
 * witness holds, keeps the request queued (bounded backpressure) when it
 * would violate the proof, refuses typed beyond the admitted Q, and fails
 * closed on an already-broken current witness.
 */
#include "execution/serving/AdmissionPressureGate.h"
#include <gtest/gtest.h>

using namespace llaminar2;
namespace
{
AllResourceWitnessResult validWitness()
{
    return AllResourceWitnessResult{true, {}};
}
AllResourceWitnessResult brokenWitness(std::string reason = "no completion order")
{
    return AllResourceWitnessResult{false, std::move(reason)};
}
}

/** @test A valid proposed witness admits the request under the queue bound. */
TEST(AdmissionPressureGate, ValidProposedWitnessAdmits)
{
    auto result = decidePressureAdmission(validWitness(), validWitness(),
        /*queued_depth=*/0, /*queue_bound_q=*/2);
    EXPECT_TRUE(result.admitted());
    EXPECT_EQ(result.admission, PressureAdmission::Admit);
    EXPECT_TRUE(result.refusal.empty());
}

/** @test A failing proposed witness keeps the request queued (no drop, no admit). */
TEST(AdmissionPressureGate, FailingProposedWitnessStaysQueued)
{
    auto result = decidePressureAdmission(validWitness(), brokenWitness(),
        /*queued_depth=*/1, /*queue_bound_q=*/4);
    EXPECT_EQ(result.admission, PressureAdmission::StayQueued);
    EXPECT_FALSE(result.refusal.empty());
}

/** @test Beyond the admitted Q the request is Refuse (exact-or-reject). */
TEST(AdmissionPressureGate, BeyondQueueBoundRefuses)
{
    auto result = decidePressureAdmission(validWitness(), validWitness(),
        /*queued_depth=*/4, /*queue_bound_q=*/4);
    EXPECT_EQ(result.admission, PressureAdmission::Refuse);
    EXPECT_NE(result.refusal.find("Q"), std::string::npos) << result.refusal;
}

/** @test A broken CURRENT witness fails closed: no new work admitted. */
TEST(AdmissionPressureGate, BrokenCurrentWitnessFailsClosed)
{
    auto result = decidePressureAdmission(brokenWitness(), validWitness(),
        /*queued_depth=*/0, /*queue_bound_q=*/2);
    EXPECT_EQ(result.admission, PressureAdmission::Refuse);
    EXPECT_NE(result.refusal.find("current"), std::string::npos) << result.refusal;
}
