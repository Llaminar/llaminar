/**
 * @file Test__GPUExistingContextBorrow.cpp
 * @brief Non-creating original worker lookup and exclusive generation retirement replay.
 *
 * Registered factories construct only hardware-free workers. Factory counters,
 * exact shared borrowers and original pool retirement scopes prove absent and
 * retiring lookups cannot initialize another native context or evade exclusion.
 */
#include "backends/GPUDeviceContextPool.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <atomic>
#include <future>
using namespace llaminar2;
TEST(GPUExistingContextBorrow, AbsentWrongFamilyOrdinalRetiringAndShutdownNeverCreate) {
    auto& pool=GPUDeviceContextPool::instance();pool.shutdown();std::atomic<unsigned> births=0;
    const auto factory=[&](int ordinal){++births;return std::make_unique<llaminar2::testing::MockWorkerGPUContext>(ordinal);};
    pool.registerNvidiaFactory(factory,2);pool.registerAMDFactory(factory,2);
    EXPECT_THROW((void)pool.retainExistingContext(DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW((void)pool.retainExistingContext(DeviceId::cuda(0)),std::logic_error);EXPECT_EQ(births,0u);
    auto created=pool.retainContext(DeviceId::cuda(0));auto borrower=pool.retainExistingContext(DeviceId::cuda(0));
    EXPECT_EQ(created,borrower);EXPECT_EQ(births,1u);
    EXPECT_THROW((void)pool.retainExistingContext(DeviceId::rocm(0)),std::logic_error);
    EXPECT_THROW((void)pool.retainExistingContext(DeviceId::cuda(1)),std::logic_error);EXPECT_EQ(births,1u);
    created.reset();EXPECT_THROW(pool.beginExclusiveGenerationRetirement(DeviceId::cuda(0)),std::logic_error);
    EXPECT_THROW(pool.shutdown(),std::logic_error);borrower.reset();
    {
        auto retirement=pool.beginExclusiveGenerationRetirement(DeviceId::cuda(0));EXPECT_TRUE(retirement.receipt().retiredLiveContext());
        EXPECT_THROW((void)pool.retainExistingContext(DeviceId::cuda(0)),std::logic_error);EXPECT_EQ(births,1u);
        EXPECT_THROW(pool.shutdown(),std::logic_error);
    }
    EXPECT_THROW((void)pool.retainExistingContext(DeviceId::cuda(0)),std::logic_error);EXPECT_EQ(births,1u);
    auto next=pool.retainContext(DeviceId::cuda(0));EXPECT_EQ(births,2u);next.reset();pool.shutdown();
    EXPECT_THROW((void)pool.retainExistingContext(DeviceId::cuda(0)),std::logic_error);EXPECT_EQ(births,2u);
    // Do not leave factories borrowing this test's stack counter.
    pool.registerNvidiaFactory([](int n){return std::make_unique<llaminar2::testing::MockWorkerGPUContext>(n);},8);
    pool.registerAMDFactory([](int n){return std::make_unique<llaminar2::testing::MockWorkerGPUContext>(n);},8);
}
TEST(GPUExistingContextBorrow, ConcurrentOriginalBorrowExcludesRetirementUntilRelease) {
    auto& pool=GPUDeviceContextPool::instance();pool.shutdown();
    pool.registerNvidiaFactory([](int n){return std::make_unique<llaminar2::testing::MockWorkerGPUContext>(n);},8);
    auto original=pool.retainContext(DeviceId::cuda(0));const auto* identity=original.get();original.reset();
    std::promise<void> acquired,release;auto permission=release.get_future();
    auto other=std::async(std::launch::async,[&]{auto held=pool.retainExistingContext(DeviceId::cuda(0));EXPECT_EQ(held.get(),identity);acquired.set_value();permission.wait();});
    acquired.get_future().wait();EXPECT_THROW(pool.beginExclusiveGenerationRetirement(DeviceId::cuda(0)),std::logic_error);
    release.set_value();other.get();
    auto retired=pool.beginExclusiveGenerationRetirement(DeviceId::cuda(0));EXPECT_TRUE(retired.receipt().retiredLiveContext());
    EXPECT_THROW((void)pool.retainExistingContext(DeviceId::cuda(0)),std::logic_error);
}
