#include <gtest/gtest.h>
#include "loaders/gpu_pipeline/LoadOrchestrator.h"

namespace llaminar2
{

static constexpr int kQ4PayloadBytes = 16;

TEST(Test__LoadOrchestrator, AddDevice)
{
    LoadOrchestrator orch;
    EXPECT_EQ(orch.numDevices(), 0u);

    orch.addDevice(0);
    EXPECT_EQ(orch.numDevices(), 1u);

    orch.addDevice(1);
    EXPECT_EQ(orch.numDevices(), 2u);
}

TEST(Test__LoadOrchestrator, PlanWeightForDevice)
{
    LoadOrchestrator orch;
    orch.addDevice(0);

    orch.planWeight(0, "attn_q", 1024, 1024, kQ4PayloadBytes,
                    false, false, 524288);

    auto* pool = orch.getPool(0);
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->numPlannedWeights(), 1u);
    EXPECT_GT(pool->totalPlannedBytes(), 0u);
}

TEST(Test__LoadOrchestrator, GetPoolReturnsCorrect)
{
    LoadOrchestrator orch;
    orch.addDevice(0);

    auto* pool = orch.getPool(0);
    EXPECT_NE(pool, nullptr);
}

TEST(Test__LoadOrchestrator, GetPoolUnknownDevice)
{
    LoadOrchestrator orch;
    orch.addDevice(0);

    auto* pool = orch.getPool(99);
    EXPECT_EQ(pool, nullptr);

    // Const version too
    const auto& const_orch = orch;
    EXPECT_EQ(const_orch.getPool(99), nullptr);
}

TEST(Test__LoadOrchestrator, MultipleDevices)
{
    LoadOrchestrator orch;
    orch.addDevice(0);
    orch.addDevice(1);

    orch.planWeight(0, "w_dev0", 512, 1024, kQ4PayloadBytes,
                    false, false, 262144);
    orch.planWeight(1, "w_dev1", 256, 512, kQ4PayloadBytes,
                    true, false, 131072);

    auto* pool0 = orch.getPool(0);
    auto* pool1 = orch.getPool(1);
    ASSERT_NE(pool0, nullptr);
    ASSERT_NE(pool1, nullptr);

    EXPECT_EQ(pool0->numPlannedWeights(), 1u);
    EXPECT_EQ(pool1->numPlannedWeights(), 1u);

    // Different sizes due to different weight dimensions and asymmetric flag
    EXPECT_NE(pool0->totalPlannedBytes(), pool1->totalPlannedBytes());
}

TEST(Test__LoadOrchestrator, AllocateSucceeds)
{
    LoadOrchestrator orch;
    orch.addDevice(0);
    orch.planWeight(0, "w1", 64, 64, kQ4PayloadBytes, false, false, 1024);

    ASSERT_NO_THROW(orch.allocate(1024, 3));

    auto* pool = orch.getPool(0);
    ASSERT_NE(pool, nullptr);
    EXPECT_TRUE(pool->isAllocated());
}

TEST(Test__LoadOrchestrator, LoadAndFinalizeWithNoJobs)
{
    LoadOrchestrator orch;
    EXPECT_NO_THROW(orch.load());
    EXPECT_NO_THROW(orch.finalize());
}

TEST(Test__LoadOrchestrator, PlanWeightUnknownDeviceThrows)
{
    LoadOrchestrator orch;
    orch.addDevice(0);

    EXPECT_THROW(
        orch.planWeight(99, "w", 64, 64, kQ4PayloadBytes, false, false, 1024),
        std::runtime_error);
}

} // namespace llaminar2
