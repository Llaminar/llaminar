/**
 * @file Test__GPUProjectionContextBorrow.cpp
 * @brief Adversarial replay of exact context retention before submission acquisition.
 *
 * Pool lookup creates only a hardware-free registered worker. Production owner
 * references and production submission locks are used; no CUDA/HIP loader runs.
 */
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUProjectionSubmission.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <future>
using namespace llaminar2;

/** @test Retirement rejects a submitter paused after exact lookup and before lock. */
TEST(GPUProjectionContextBorrow, LookupPauseAndPoolReentry)
{
    auto &pool = GPUDeviceContextPool::instance();
    pool.shutdown();
    pool.registerNvidiaFactory([](int ordinal) {
        return std::make_unique<llaminar2::testing::MockWorkerGPUContext>(ordinal);
    }, 1);
    auto owner = pool.retainContext(DeviceId::cuda(0));
    const auto *identity = owner.get();
    EXPECT_THROW(pool.beginExclusiveGenerationRetirement(DeviceId::cuda(0)), std::logic_error);
    EXPECT_EQ(pool.retainContext(DeviceId::cuda(0)).get(), identity);
    GPUProjectionResources resources;
    std::promise<void> looked_up, proceed;
    auto permission = proceed.get_future();
    std::future<void> capture_waiter;
    {
        auto capture = resources.acquire(reinterpret_cast<void *>(1), GPUProjectionSubmissionKind::Capture);
        auto blocked = std::async(std::launch::async, [&] {
            auto retained = pool.retainContext(DeviceId::cuda(0));
            looked_up.set_value(); permission.wait();
            // This waits for the context submission lock without holding pool mutex.
            auto next = resources.acquire(reinterpret_cast<void *>(1), GPUProjectionSubmissionKind::Capture);
            EXPECT_EQ(retained.get(), identity);
        });
        looked_up.get_future().wait();
        // Only the paused submitter now retains this generation: the rejection
        // would disappear if its lookup regressed to the original raw reference.
        owner.reset();
        EXPECT_THROW(pool.beginExclusiveGenerationRetirement(DeviceId::cuda(0)), std::logic_error);
        proceed.set_value();
        // A waiting submitter must not prevent the capture owner from pool reentry.
        EXPECT_EQ(pool.retainContext(DeviceId::cuda(0)).get(), identity);
        // Release the capture before waiting for the competing future.
        // The future is moved outside this lexical interval below.
        capture_waiter = std::move(blocked);
    }
    capture_waiter.get();
    owner.reset();
    auto retired = pool.beginExclusiveGenerationRetirement(DeviceId::cuda(0));
    EXPECT_TRUE(retired.receipt().retiredLiveContext());
}

/** @test The old unretained lookup does not itself hold a retirement obligation. */
TEST(GPUProjectionContextBorrow, OriginalUnretainedLookupControl)
{
    auto &pool = GPUDeviceContextPool::instance();
    pool.shutdown();
    pool.registerNvidiaFactory([](int ordinal) {
        return std::make_unique<llaminar2::testing::MockWorkerGPUContext>(ordinal);
    }, 1);
    // Never dereference the reference after retirement; the control authenticates
    // why production submission must use the retained lookup instead.
    (void)pool.getNvidiaContext(0);
    auto retired = pool.beginExclusiveGenerationRetirement(DeviceId::cuda(0));
    EXPECT_TRUE(retired.receipt().retiredLiveContext());
}
