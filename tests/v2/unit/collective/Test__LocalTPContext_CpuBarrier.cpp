/**
 * @file Test__LocalTPContext_CpuBarrier.cpp
 * @brief Regression tests for CPU TP barrier-synchronized collectives
 *
 * These tests verify that LocalTPContext correctly handles multi-device CPU TP
 * (HOST backend) with barrier-synchronized allreduce and allgather. They are
 * regression tests for three bugs discovered during CPU TP bringup:
 *
 * Bug 1: allreduce() with HOST backend fell through to GPU paths that called
 *         gpu_data_ptr(), which returns nullptr for CPU tensors → crash.
 *
 * Bug 2: allreduceCpuBarrier() used home_device() to find the device_index for
 *         slot assignment, but all CPU tensors have the same generic "CPU"
 *         home_device → both threads mapped to the same slot → nullptr → crash.
 *         Fix: Use arrival_order for slot assignment (sum is commutative).
 *
 * Bug 3: allgather() used lock_guard on mutex_, which couldn't be unlocked before
 *         entering the barrier → all threads blocked on mutex_ → deadlock.
 *         Fix: Changed to unique_lock with explicit lock.unlock().
 */

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <cmath>

#include "collective/LocalTPContext.h"
#include "collective/ICollectiveBackend.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/Tensors.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LocalTPContext_CpuBarrier : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cpu0_ = GlobalDeviceAddress::cpu(0);
        cpu1_ = GlobalDeviceAddress::cpu(1);
    }

    GlobalDeviceAddress cpu0_;
    GlobalDeviceAddress cpu1_;

    static constexpr size_t kTensorSize = 256;
};

// =============================================================================
// Bug 1 Regression: allreduce must not crash with HOST backend (gpu_data_ptr)
// =============================================================================

/**
 * @test Two CPU devices with HOST backend can allreduce without crashing.
 *
 * Regression for: allreduce() fell through to GPU paths calling gpu_data_ptr()
 * which returns nullptr for CPU tensors, causing a segfault.
 */
TEST_F(Test__LocalTPContext_CpuBarrier, TwoCpuDevicesAllreduceDoesNotCrash)
{
    auto ctx = createLocalTPContext(
        {cpu0_, cpu1_}, {0.5f, 0.5f}, CollectiveBackendType::HOST);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 2);

    auto tensor0 = TestTensorFactory::createFP32({kTensorSize});
    auto tensor1 = TestTensorFactory::createFP32({kTensorSize});

    for (size_t i = 0; i < kTensorSize; ++i)
    {
        tensor0->mutable_data()[i] = 1.0f;
        tensor1->mutable_data()[i] = 2.0f;
    }

    std::atomic<bool> success0{false};
    std::atomic<bool> success1{false};

    std::thread t0([&]()
                   { success0 = ctx->allreduce(tensor0.get()); });

    std::thread t1([&]()
                   { success1 = ctx->allreduce(tensor1.get()); });

    t0.join();
    t1.join();

    EXPECT_TRUE(success0.load()) << "Thread 0 allreduce failed";
    EXPECT_TRUE(success1.load()) << "Thread 1 allreduce failed";
}

// =============================================================================
// Bug 2 Regression: allreduce produces correct sum (arrival_order slots)
// =============================================================================

/**
 * @test Two CPU devices allreduce produces correct element-wise sum.
 *
 * Regression for: allreduceCpuBarrier used home_device() for slot assignment,
 * but all CPU tensors have the same generic "CPU" home device. Both threads
 * mapped to the same slot, leaving the other slot as nullptr → crash or
 * incorrect results.
 * Fix: Use arrival_order for slot assignment (sum is commutative).
 */
TEST_F(Test__LocalTPContext_CpuBarrier, TwoCpuDevicesAllreduceCorrectSum)
{
    auto ctx = createLocalTPContext(
        {cpu0_, cpu1_}, {0.5f, 0.5f}, CollectiveBackendType::HOST);
    ASSERT_NE(ctx, nullptr);

    auto tensor0 = TestTensorFactory::createFP32({kTensorSize});
    auto tensor1 = TestTensorFactory::createFP32({kTensorSize});

    // Fill with distinct values so the sum is verifiable
    for (size_t i = 0; i < kTensorSize; ++i)
    {
        tensor0->mutable_data()[i] = static_cast<float>(i);         // [0, 1, 2, ...]
        tensor1->mutable_data()[i] = static_cast<float>(i) * 10.0f; // [0, 10, 20, ...]
    }

    std::atomic<bool> success0{false};
    std::atomic<bool> success1{false};

    std::thread t0([&]()
                   { success0 = ctx->allreduce(tensor0.get()); });

    std::thread t1([&]()
                   { success1 = ctx->allreduce(tensor1.get()); });

    t0.join();
    t1.join();

    ASSERT_TRUE(success0.load());
    ASSERT_TRUE(success1.load());

    // Both tensors should now contain the element-wise sum: [0, 11, 22, 33, ...]
    for (size_t i = 0; i < kTensorSize; ++i)
    {
        float expected = static_cast<float>(i) + static_cast<float>(i) * 10.0f;
        EXPECT_FLOAT_EQ(tensor0->data()[i], expected)
            << "tensor0 mismatch at index " << i;
        EXPECT_FLOAT_EQ(tensor1->data()[i], expected)
            << "tensor1 mismatch at index " << i;
    }
}

// =============================================================================
// Bug 3 Regression: allgather must not deadlock (unique_lock)
// =============================================================================

/**
 * @test Two CPU devices allgather completes without deadlock.
 *
 * Regression for: allgather() used lock_guard on mutex_, which cannot be
 * unlocked before entering the barrier. All threads blocked waiting to acquire
 * mutex_ that only the first thread held → deadlock.
 * Fix: Changed to unique_lock with explicit lock.unlock() before barrier entry.
 *
 * This test has a 10-second timeout to detect deadlocks.
 */
TEST_F(Test__LocalTPContext_CpuBarrier, TwoCpuDevicesAllgatherDoesNotDeadlock)
{
    auto ctx = createLocalTPContext(
        {cpu0_, cpu1_}, {0.5f, 0.5f}, CollectiveBackendType::HOST);
    ASSERT_NE(ctx, nullptr);

    const size_t shard_size = 128;
    auto shard0 = TestTensorFactory::createFP32({shard_size});
    auto shard1 = TestTensorFactory::createFP32({shard_size});
    auto global0 = TestTensorFactory::createFP32({shard_size * 2});
    auto global1 = TestTensorFactory::createFP32({shard_size * 2});

    for (size_t i = 0; i < shard_size; ++i)
    {
        shard0->mutable_data()[i] = 1.0f;
        shard1->mutable_data()[i] = 2.0f;
    }

    std::atomic<bool> success0{false};
    std::atomic<bool> success1{false};
    std::atomic<bool> finished{false};

    std::thread t0([&]()
                   { success0 = ctx->allgather(shard0.get(), global0.get()); });

    std::thread t1([&]()
                   { success1 = ctx->allgather(shard1.get(), global1.get()); });

    // Deadlock detection: join with timeout
    std::thread watchdog([&]()
                         {
        t0.join();
        t1.join();
        finished.store(true); });

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    watchdog.join();

    ASSERT_TRUE(finished.load()) << "allgather deadlocked (did not complete within 10s)";
    EXPECT_TRUE(success0.load()) << "Thread 0 allgather failed";
    EXPECT_TRUE(success1.load()) << "Thread 1 allgather failed";
}

// =============================================================================
// Stress: Multiple sequential barrier cycles with CPU TP
// =============================================================================

/**
 * @test Multiple sequential allreduces reuse the barrier correctly.
 *
 * Validates that barrier_generation_ and barrier_count_ reset properly
 * between cycles, preventing stale state from affecting subsequent operations.
 */
TEST_F(Test__LocalTPContext_CpuBarrier, MultipleCpuAllreduceCycles)
{
    auto ctx = createLocalTPContext(
        {cpu0_, cpu1_}, {0.5f, 0.5f}, CollectiveBackendType::HOST);
    ASSERT_NE(ctx, nullptr);

    const int num_cycles = 10;

    for (int cycle = 0; cycle < num_cycles; ++cycle)
    {
        auto tensor0 = TestTensorFactory::createFP32({kTensorSize});
        auto tensor1 = TestTensorFactory::createFP32({kTensorSize});

        for (size_t i = 0; i < kTensorSize; ++i)
        {
            tensor0->mutable_data()[i] = static_cast<float>(cycle + 1);
            tensor1->mutable_data()[i] = static_cast<float>(cycle + 1);
        }

        std::atomic<bool> success0{false};
        std::atomic<bool> success1{false};

        std::thread t0([&]()
                       { success0 = ctx->allreduce(tensor0.get()); });

        std::thread t1([&]()
                       { success1 = ctx->allreduce(tensor1.get()); });

        t0.join();
        t1.join();

        ASSERT_TRUE(success0.load()) << "Cycle " << cycle << " thread 0 failed";
        ASSERT_TRUE(success1.load()) << "Cycle " << cycle << " thread 1 failed";

        // Each value was (cycle+1), summed across 2 devices = 2*(cycle+1)
        float expected = static_cast<float>(cycle + 1) * 2.0f;
        EXPECT_FLOAT_EQ(tensor0->data()[0], expected)
            << "Wrong sum at cycle " << cycle;
        EXPECT_FLOAT_EQ(tensor1->data()[0], expected)
            << "Wrong sum at cycle " << cycle;
    }
}

/**
 * @test Allreduce with stage_name parameter works for CPU barrier path.
 *
 * Ensures the stage_name overload correctly dispatches to allreduceCpuBarrier.
 */
TEST_F(Test__LocalTPContext_CpuBarrier, CpuAllreduceWithStageName)
{
    auto ctx = createLocalTPContext(
        {cpu0_, cpu1_}, {0.5f, 0.5f}, CollectiveBackendType::HOST);
    ASSERT_NE(ctx, nullptr);

    auto tensor0 = TestTensorFactory::createFP32({kTensorSize});
    auto tensor1 = TestTensorFactory::createFP32({kTensorSize});

    for (size_t i = 0; i < kTensorSize; ++i)
    {
        tensor0->mutable_data()[i] = 3.0f;
        tensor1->mutable_data()[i] = 7.0f;
    }

    std::atomic<bool> success0{false};
    std::atomic<bool> success1{false};

    std::thread t0([&]()
                   { success0 = ctx->allreduce(tensor0.get(), "layer0_ffn_down_allreduce", 0); });

    std::thread t1([&]()
                   { success1 = ctx->allreduce(tensor1.get(), "layer0_ffn_down_allreduce", 0); });

    t0.join();
    t1.join();

    ASSERT_TRUE(success0.load());
    ASSERT_TRUE(success1.load());

    // 3.0 + 7.0 = 10.0 everywhere
    for (size_t i = 0; i < kTensorSize; ++i)
    {
        EXPECT_FLOAT_EQ(tensor0->data()[i], 10.0f);
        EXPECT_FLOAT_EQ(tensor1->data()[i], 10.0f);
    }
}
