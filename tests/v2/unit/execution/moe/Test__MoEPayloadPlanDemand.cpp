/**
 * @file Test__MoEPayloadPlanDemand.cpp
 * @brief Independent per-source oracle for the production controller scalar demand.
 *
 * This device-free regression exercises the same host/device accumulator used
 * by both native controllers. The oracle deliberately retains the old per-source
 * representation, so validity, inactive owners and scalar reduction are checked
 * without reproducing the optimized arithmetic.
 */
#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace llaminar2::test
{
namespace
{
using moe_rebalance_policy::PayloadPlanDemand;
struct Plan { uint32_t op, source, destination, slot; };
constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();

/** @brief Reference two-stage source aggregation, with independent mask arithmetic. */
PayloadPlanDemand reference(const std::vector<Plan>& plans, uint32_t participants)
{
    std::array<uint32_t, 8> by_source{};
    PayloadPlanDemand result;
    for (const auto& p : plans)
    {
        if ((p.op != 1u && p.op != 3u) || p.source >= 8u ||
            p.destination >= participants || p.destination >= 8u || p.slot == invalid)
            continue;
        by_source[p.source] = std::max(by_source[p.source], p.slot + 1u);
        result.source_mask |= 1u << p.source;
        result.destination_mask |= 1u << p.destination;
        result.edge_mask |= uint64_t{1} << (p.source * 8u + p.destination);
    }
    for (uint32_t source = 0; source < std::min(participants, 8u); ++source)
        result.requested_slots = std::max(result.requested_slots, by_source[source]);
    return result;
}

/** @brief Invoke the production accumulator with precisely the published plan prefix. */
PayloadPlanDemand production(const std::vector<Plan>& plans, uint32_t participants)
{
    PayloadPlanDemand result;
    for (const auto& p : plans)
        moe_rebalance_policy::accumulatePayloadPlanDemand(
            result, p.op, p.source, p.destination, p.slot, participants);
    return result;
}

/** @brief Compare paid slot extent and every ownership mask independently. */
void expectEquivalent(const std::vector<Plan>& plans, uint32_t participants)
{
    const auto expected = reference(plans, participants);
    const auto actual = production(plans, participants);
    EXPECT_EQ(actual.requested_slots, expected.requested_slots);
    EXPECT_EQ(actual.source_mask, expected.source_mask);
    EXPECT_EQ(actual.destination_mask, expected.destination_mask);
    EXPECT_EQ(actual.edge_mask, expected.edge_mask);
}
}

TEST(MoEPayloadPlanDemand, EmptyAndInvalidPlansRemainEmpty)
{
    for (uint32_t participants = 0; participants <= 10u; ++participants)
    {
        expectEquivalent({}, participants);
        expectEquivalent({{0, 0, 0, 99}, {2, 0, 0, 99}, {4, 0, 0, 99},
                          {1, 8, 0, 99}, {3, 0, 8, 99}, {1, 0, 0, invalid}}, participants);
    }
}

TEST(MoEPayloadPlanDemand, InactiveSourcePreservesOwnersWithoutChargingSlots)
{
    const std::vector<Plan> plans{{1, 0, 1, 2}, {3, 7, 0, 1000}};
    const auto actual = production(plans, 2);
    EXPECT_EQ(actual.requested_slots, 3u);
    EXPECT_EQ(actual.source_mask, 0x81u);
    EXPECT_EQ(actual.destination_mask, 3u);
    EXPECT_EQ(actual.edge_mask, (uint64_t{1} << 1u) | (uint64_t{1} << 56u));
    expectEquivalent(plans, 2);
    expectEquivalent({{3, 7, 0, 1000}}, 2);
}

TEST(MoEPayloadPlanDemand, ExhaustiveSinglePlanValidityAndBoundarySlots)
{
    for (uint32_t participants = 0; participants <= 10u; ++participants)
        for (uint32_t op : {0u, 1u, 2u, 3u, 4u, invalid})
            for (uint32_t source = 0; source <= 9u; ++source)
                for (uint32_t destination = 0; destination <= 9u; ++destination)
                    for (uint32_t slot : {0u, 1u, 511u, 65535u, invalid - 1u, invalid})
                        expectEquivalent({{op, source, destination, slot}}, participants);
}

TEST(MoEPayloadPlanDemand, SkewedSourcesUseMaximumRatherThanSumOrLast)
{
    std::vector<Plan> plans{{1, 0, 1, 4}, {3, 1, 0, 10}, {1, 0, 2, 1},
                            {3, 2, 1, 3}, {2, 0, 0, 1000}, {1, 7, 0, 2000}};
    EXPECT_EQ(production(plans, 3).requested_slots, 11u);
    for (unsigned prefix = 0; prefix <= plans.size(); ++prefix)
        expectEquivalent(std::vector<Plan>(plans.begin(), plans.begin() + prefix), 3);
    std::reverse(plans.begin(), plans.end());
    expectEquivalent(plans, 3);
}

TEST(MoEPayloadPlanDemand, MixedPlanStreamsMatchIndependentPerSourceReduction)
{
    uint32_t state = 0x8a7134u;
    const auto next = [&]() { state = state * 1664525u + 1013904223u; return state; };
    for (unsigned trial = 0; trial < 4000; ++trial)
    {
        const uint32_t participants = next() % 11u;
        std::vector<Plan> plans;
        const unsigned length = next() % 64u;
        for (unsigned i = 0; i < length; ++i)
        {
            const auto op = next() % 5u, source = next() % 11u, destination = next() % 11u;
            const auto slot = next();
            plans.push_back({op, source, destination, (slot & 7u) ? slot % 1024u : invalid});
        }
        expectEquivalent(plans, participants);
    }
}
}
