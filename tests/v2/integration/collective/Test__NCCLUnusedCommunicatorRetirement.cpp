/**
 * @file Test__NCCLUnusedCommunicatorRetirement.cpp
 * @brief Repeated immediate retirement of the production CUDA coordinator.
 *
 * Admission can create a native clique without ever submitting a collective.
 * Each successful initialize must publish a complete running state before the
 * caller can stop it. Twenty reused owner lifetimes complement the deterministic
 * device-free readiness replay and the matching native ROCm retirement test.
 * CTest observes process exit as well as explicit shutdown and its idempotence.
 */
#include "backends/BackendManager.h"
#include "collective/coordinators/NCCLCoordinator.h"
#include <gtest/gtest.h>

namespace llaminar2::test
{
/** @test An unused two-device clique retires immediately without priming device work. */
TEST(NCCLUnusedCommunicatorRetirement, DirectOwnerRetiresTwentyUnusedCliques)
{
    const auto *backend = getCUDABackend();
    ASSERT_NE(backend, nullptr);
    ASSERT_GE(backend->deviceCount(), 2);

    NCCLCoordinator coordinator;
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        SCOPED_TRACE(iteration);
        ASSERT_TRUE(coordinator.initialize({0, 1})) << coordinator.lastError();
        ASSERT_TRUE(coordinator.isInitialized());
        coordinator.shutdown();
        EXPECT_FALSE(coordinator.isInitialized());
        coordinator.shutdown();
    }
}
}
