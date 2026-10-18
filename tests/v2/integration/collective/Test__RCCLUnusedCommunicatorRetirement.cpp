/**
 * @file Test__RCCLUnusedCommunicatorRetirement.cpp
 * @brief Retire real unused RCCL cliques after admission-only initialization.
 *
 * A rejected model may never submit a collective. Direct owners must support
 * repeated initialization/retirement, and pooled owners must retire at process
 * exit without launching a synthetic first collective. CTest observes the
 * process exit as well as GoogleTest's assertions; the latter alone missed the
 * original HIP crash after a successful test-body teardown.
 */
#include "backends/BackendManager.h"
#include "collective/DeviceGroup.h"
#include "collective/backends/RCCLBackend.h"
#include "collective/coordinators/RCCLCoordinator.h"
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

namespace llaminar2::test
{
/** @brief Two/four required participants and an explicit optional eight-device lane. */
class RCCLUnusedCommunicatorRetirement : public ::testing::TestWithParam<int>
{
protected:
    /** @brief Require the exact registered native topology before constructing owners. */
    void SetUp() override
    {
        const auto *backend = getROCmBackend();
        ASSERT_NE(backend, nullptr);
        if (GetParam() == 8 && backend->deviceCount() < 8)
            GTEST_SKIP() << "Requires 8 ROCm devices";
        ASSERT_GE(backend->deviceCount(), GetParam());
        // Each CTest registration owns its process and accelerator lane. Start
        // without a pooled prior owner so every cycle exercises native setup.
        RCCLBackend::drainCoordinatorPool();
    }
};

/** @test Native finalization accepts every unused rank without priming work. */
TEST_P(RCCLUnusedCommunicatorRetirement, DirectOwnerRetiresTwentyUnusedCliques)
{
    std::vector<int> devices(GetParam());
    std::iota(devices.begin(), devices.end(), 0);
    RCCLCoordinator coordinator;
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        SCOPED_TRACE(iteration);
        ASSERT_TRUE(coordinator.initialize(devices)) << coordinator.lastError();
        ASSERT_TRUE(coordinator.isInitialized());
        coordinator.shutdown();
        EXPECT_FALSE(coordinator.isInitialized());
        coordinator.shutdown(); // Retirement remains idempotent.
    }
}

/** @test The native process must exit cleanly with a never-used pooled owner. */
TEST_P(RCCLUnusedCommunicatorRetirement, PooledOwnerRetiresAtProcessExit)
{
    auto builder = DeviceGroupBuilder().setName("unused_rccl_retirement")
        .setScope(CollectiveScope::LOCAL).setLocalRank(0);
    for (int ordinal = 0; ordinal < GetParam(); ++ordinal)
        builder.addDevice(DeviceId::rocm(ordinal));
    RCCLBackend backend;
    ASSERT_TRUE(backend.initialize(builder.build())) << backend.lastError();
    backend.shutdown();
    // Deliberately retain the production pool until process exit. Draining it
    // here would hide the library-destruction ordering that triggered the bug.
}

INSTANTIATE_TEST_SUITE_P(DeviceCounts, RCCLUnusedCommunicatorRetirement,
    ::testing::Values(2, 4, 8), [](const auto &info) { return "Devices" + std::to_string(info.param); });
}
