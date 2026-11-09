/**
 * @file Test__RecursiveStageSnapshot.cpp
 * @brief Compile/runtime proof of passive self-containing stage snapshots.
 *
 * A snapshot's stage vector holds complete child snapshots only after the
 * enclosing type is complete. Equality must not eagerly instantiate Entry
 * during declaration on Clang/libc++; no device, model or runtime is involved.
 */
#include "execution/moe/MoEOptimizationStages.h"
#include <gtest/gtest.h>

namespace llaminar2::test
{
namespace
{
/** @brief Minimal passive recursive observation with the real production stage container. */
struct Snapshot
{
    int value = 0;
    MoEOptimizationStages<Snapshot> stages;
    /** @brief Compare the complete recursive observation after type completion.
     * @param other Complete passive observation.
     * @return Whether scalar and every child-stage identity/value match.
     */
    bool operator==(const Snapshot &other) const
    { return value == other.value && stages == other.stages; }
};
}

/** @test Leaf and sealed child equality instantiate only after the snapshot is complete. */
TEST(RecursiveStageSnapshot, CompleteTypeRetainsChildIdentityAndValue)
{
    Snapshot leaf{.value = 7};
    EXPECT_EQ(leaf, leaf);
    const MoEOptimizationStageIdentity identity(0, 0, 2, 2, {GlobalDeviceAddress::cpu(0)}, true);
    Snapshot pipeline{.value = 11, .stages = MoEOptimizationStages<Snapshot>::seal({{identity, leaf}})};
    auto copy = pipeline;
    EXPECT_EQ(copy, pipeline);
    Snapshot changed{.value = 11, .stages = MoEOptimizationStages<Snapshot>::seal({{identity, Snapshot{.value = 8}}})};
    EXPECT_FALSE(changed == pipeline);
    EXPECT_EQ(pipeline.stages.entries().front().value.value, 7);
}
} // namespace llaminar2::test
