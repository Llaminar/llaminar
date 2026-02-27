/**
 * @file Test__ICollectiveCoordinator.cpp
 * @brief Unit tests for ICollectiveCoordinator base interface
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for the ICollectiveCoordinator abstract base interface:
 * - Template methods compile correctly with a mock implementation
 * - submitAndWait() returns the correct value
 * - submitAsync() returns a valid future
 * - Exception propagation through the template methods
 *
 * These tests do NOT require any GPU - they test the base class template
 * logic using a mock implementation that executes work synchronously.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>

#include "collective/coordinators/ICollectiveCoordinator.h"

using namespace llaminar2;

// =============================================================================
// Mock Implementation for Testing
// =============================================================================

/**
 * @brief Mock coordinator that executes work synchronously (no real threading)
 *
 * This mock allows testing the template methods without GPU dependencies.
 * Work is executed immediately in enqueueWork() rather than on a separate
 * thread, which is sufficient for testing the promise/future machinery.
 */
class MockCollectiveCoordinator : public ICollectiveCoordinator
{
public:
    MockCollectiveCoordinator() = default;
    ~MockCollectiveCoordinator() override = default;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool initialize(const std::vector<int> &device_ordinals) override
    {
        device_ordinals_ = device_ordinals;
        initialized_ = true;
        return true;
    }

    void shutdown() override
    {
        initialized_ = false;
    }

    bool isInitialized() const override
    {
        return initialized_;
    }

    // =========================================================================
    // Synchronization (stub implementations for interface compliance)
    // =========================================================================

    void *getCompletionEvent(int device_idx) const override
    {
        if (device_idx < 0 || static_cast<size_t>(device_idx) >= device_ordinals_.size())
        {
            throw std::out_of_range("Invalid device index");
        }
        // Return a dummy non-null pointer (just for testing)
        return reinterpret_cast<void *>(static_cast<uintptr_t>(device_idx + 1));
    }

    void waitForDeviceEvent(int device_idx, void *worker_event) override
    {
        (void)device_idx;
        (void)worker_event;
        // No-op for mock
    }

    void setComputeStreams(const std::vector<void *> &compute_streams) override
    {
        (void)compute_streams;
        // No-op for mock
    }

    // =========================================================================
    // Test Instrumentation
    // =========================================================================

    int getEnqueueCount() const { return enqueue_count_.load(); }

protected:
    void enqueueWork(std::function<void()> work) override
    {
        enqueue_count_++;
        // Execute work synchronously for testing
        work();
    }

private:
    std::vector<int> device_ordinals_;
    bool initialized_{false};
    std::atomic<int> enqueue_count_{0};
};

/**
 * @brief Mock coordinator that executes work on a separate thread
 *
 * This tests the async behavior more realistically by running work
 * on a different thread from the caller.
 */
class AsyncMockCoordinator : public ICollectiveCoordinator
{
public:
    AsyncMockCoordinator() = default;
    ~AsyncMockCoordinator() override = default;

    bool initialize(const std::vector<int> &device_ordinals) override
    {
        device_ordinals_ = device_ordinals;
        initialized_ = true;
        return true;
    }

    void shutdown() override
    {
        initialized_ = false;
    }

    bool isInitialized() const override
    {
        return initialized_;
    }

    void *getCompletionEvent(int device_idx) const override
    {
        if (device_idx < 0 || static_cast<size_t>(device_idx) >= device_ordinals_.size())
        {
            throw std::out_of_range("Invalid device index");
        }
        return reinterpret_cast<void *>(static_cast<uintptr_t>(device_idx + 1));
    }

    void waitForDeviceEvent(int /*device_idx*/, void * /*worker_event*/) override
    {
        // No-op for mock
    }

    void setComputeStreams(const std::vector<void *> & /*compute_streams*/) override
    {
        // No-op for mock
    }

    std::thread::id getLastExecutionThreadId() const { return last_thread_id_; }

protected:
    void enqueueWork(std::function<void()> work) override
    {
        // Execute work on a separate thread to simulate real async behavior
        std::thread worker([this, work = std::move(work)]()
                           {
            last_thread_id_ = std::this_thread::get_id();
            work(); });
        worker.join(); // For test simplicity, wait for completion
    }

private:
    std::vector<int> device_ordinals_;
    bool initialized_{false};
    mutable std::thread::id last_thread_id_;
};

// =============================================================================
// Test Fixture
// =============================================================================

class Test__ICollectiveCoordinator : public ::testing::Test
{
protected:
    void SetUp() override
    {
        coordinator_ = std::make_unique<MockCollectiveCoordinator>();
        coordinator_->initialize({0, 1});
    }

    void TearDown() override
    {
        if (coordinator_)
        {
            coordinator_->shutdown();
        }
    }

    std::unique_ptr<MockCollectiveCoordinator> coordinator_;
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

/**
 * @test Verify initialize/shutdown lifecycle
 */
TEST_F(Test__ICollectiveCoordinator, Lifecycle_InitializeAndShutdown)
{
    auto coord = std::make_unique<MockCollectiveCoordinator>();

    EXPECT_FALSE(coord->isInitialized());

    EXPECT_TRUE(coord->initialize({0, 1, 2}));
    EXPECT_TRUE(coord->isInitialized());

    coord->shutdown();
    EXPECT_FALSE(coord->isInitialized());
}

/**
 * @test Verify shutdown is idempotent
 */
TEST_F(Test__ICollectiveCoordinator, Lifecycle_ShutdownIdempotent)
{
    auto coord = std::make_unique<MockCollectiveCoordinator>();
    coord->initialize({0});

    coord->shutdown();
    EXPECT_FALSE(coord->isInitialized());

    // Second shutdown should be safe
    coord->shutdown();
    EXPECT_FALSE(coord->isInitialized());
}

// =============================================================================
// submitAndWait() Tests
// =============================================================================

/**
 * @test submitAndWait with int return value
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAndWait_ReturnsIntValue)
{
    int result = coordinator_->submitAndWait([]()
                                             { return 42; });

    EXPECT_EQ(result, 42);
    EXPECT_EQ(coordinator_->getEnqueueCount(), 1);
}

/**
 * @test submitAndWait with bool return value
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAndWait_ReturnsBoolValue)
{
    bool result = coordinator_->submitAndWait([]()
                                              { return true; });

    EXPECT_TRUE(result);
}

/**
 * @test submitAndWait with string return value
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAndWait_ReturnsStringValue)
{
    std::string result = coordinator_->submitAndWait([]()
                                                     { return std::string("hello world"); });

    EXPECT_EQ(result, "hello world");
}

/**
 * @test submitAndWait with void return (no return value)
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAndWait_VoidWork)
{
    int side_effect = 0;

    coordinator_->submitAndWait([&side_effect]()
                                { side_effect = 123; });

    EXPECT_EQ(side_effect, 123);
    EXPECT_EQ(coordinator_->getEnqueueCount(), 1);
}

/**
 * @test submitAndWait captures lambda state correctly
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAndWait_CapturesState)
{
    int a = 10;
    int b = 20;

    int result = coordinator_->submitAndWait([a, b]()
                                             { return a + b; });

    EXPECT_EQ(result, 30);
}

/**
 * @test submitAndWait propagates exceptions
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAndWait_PropagatesException)
{
    EXPECT_THROW(
        coordinator_->submitAndWait([]() -> int
                                    { throw std::runtime_error("test exception"); }),
        std::runtime_error);
}

/**
 * @test submitAndWait propagates exceptions for void work
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAndWait_PropagatesException_VoidWork)
{
    EXPECT_THROW(
        coordinator_->submitAndWait([]()
                                    { throw std::logic_error("void exception"); }),
        std::logic_error);
}

// =============================================================================
// submitAsync() Tests
// =============================================================================

/**
 * @test submitAsync returns valid future with int result
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAsync_ReturnsValidFuture)
{
    auto future = coordinator_->submitAsync([]()
                                            { return 99; });

    EXPECT_TRUE(future.valid());
    EXPECT_EQ(future.get(), 99);
}

/**
 * @test submitAsync returns valid future with bool result
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAsync_ReturnsBoolFuture)
{
    auto future = coordinator_->submitAsync([]()
                                            { return false; });

    EXPECT_TRUE(future.valid());
    EXPECT_FALSE(future.get());
}

/**
 * @test submitAsync returns valid future with string result
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAsync_ReturnsStringFuture)
{
    auto future = coordinator_->submitAsync([]()
                                            { return std::string("async result"); });

    EXPECT_TRUE(future.valid());
    EXPECT_EQ(future.get(), "async result");
}

/**
 * @test submitAsync with void work
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAsync_VoidWork)
{
    int side_effect = 0;

    auto future = coordinator_->submitAsync([&side_effect]()
                                            { side_effect = 456; });

    EXPECT_TRUE(future.valid());
    future.get(); // Wait for completion
    EXPECT_EQ(side_effect, 456);
}

/**
 * @test submitAsync captures state correctly
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAsync_CapturesState)
{
    int multiplier = 5;

    auto future = coordinator_->submitAsync([multiplier]()
                                            { return multiplier * 7; });

    EXPECT_EQ(future.get(), 35);
}

/**
 * @test submitAsync propagates exceptions through future
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAsync_PropagatesException)
{
    auto future = coordinator_->submitAsync([]() -> int
                                            { throw std::runtime_error("async exception"); });

    EXPECT_TRUE(future.valid());
    EXPECT_THROW(future.get(), std::runtime_error);
}

/**
 * @test Multiple async submissions work correctly
 */
TEST_F(Test__ICollectiveCoordinator, SubmitAsync_MultipleSubmissions)
{
    auto future1 = coordinator_->submitAsync([]()
                                             { return 1; });
    auto future2 = coordinator_->submitAsync([]()
                                             { return 2; });
    auto future3 = coordinator_->submitAsync([]()
                                             { return 3; });

    EXPECT_EQ(future1.get(), 1);
    EXPECT_EQ(future2.get(), 2);
    EXPECT_EQ(future3.get(), 3);
    EXPECT_EQ(coordinator_->getEnqueueCount(), 3);
}

// =============================================================================
// Event API Tests
// =============================================================================

/**
 * @test getCompletionEvent returns non-null for valid device index
 */
TEST_F(Test__ICollectiveCoordinator, GetCompletionEvent_ValidIndex)
{
    void *event0 = coordinator_->getCompletionEvent(0);
    void *event1 = coordinator_->getCompletionEvent(1);

    EXPECT_NE(event0, nullptr);
    EXPECT_NE(event1, nullptr);
    EXPECT_NE(event0, event1); // Different devices should have different events
}

/**
 * @test getCompletionEvent throws for invalid device index
 */
TEST_F(Test__ICollectiveCoordinator, GetCompletionEvent_InvalidIndex)
{
    EXPECT_THROW(coordinator_->getCompletionEvent(5), std::out_of_range);
    EXPECT_THROW(coordinator_->getCompletionEvent(-1), std::out_of_range);
}

/**
 * @test waitForDeviceEvent accepts valid parameters
 */
TEST_F(Test__ICollectiveCoordinator, WaitForDeviceEvent_ValidParams)
{
    void *dummy_event = reinterpret_cast<void *>(0x1234);

    // Should not throw for valid device index
    EXPECT_NO_THROW(coordinator_->waitForDeviceEvent(0, dummy_event));
    EXPECT_NO_THROW(coordinator_->waitForDeviceEvent(1, nullptr)); // null event is valid
}

// =============================================================================
// Polymorphism Tests
// =============================================================================

/**
 * @test Interface can be used through base pointer
 */
TEST_F(Test__ICollectiveCoordinator, Polymorphism_BasePointerUsage)
{
    ICollectiveCoordinator *base_ptr = coordinator_.get();

    EXPECT_TRUE(base_ptr->isInitialized());

    int result = base_ptr->submitAndWait([]()
                                         { return 77; });
    EXPECT_EQ(result, 77);

    auto future = base_ptr->submitAsync([]()
                                        { return 88; });
    EXPECT_EQ(future.get(), 88);
}

/**
 * @test Work executes on different thread with AsyncMockCoordinator
 */
TEST(Test__ICollectiveCoordinator_Async, WorkExecutesOnDifferentThread)
{
    AsyncMockCoordinator coord;
    coord.initialize({0});

    std::thread::id caller_thread = std::this_thread::get_id();
    std::thread::id execution_thread;

    coord.submitAndWait([&execution_thread]()
                        { execution_thread = std::this_thread::get_id(); });

    // Verify work executed on a different thread
    EXPECT_NE(execution_thread, caller_thread);
    EXPECT_EQ(execution_thread, coord.getLastExecutionThreadId());

    coord.shutdown();
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

/**
 * @test Coordinator cannot be copied (compile-time test)
 *
 * This test verifies the interface correctly deletes copy operations.
 * The actual check is at compile time via deleted copy constructor/assignment.
 */
TEST_F(Test__ICollectiveCoordinator, NonCopyable)
{
    // These should not compile if uncommented:
    // MockCollectiveCoordinator copy(*coordinator_);
    // MockCollectiveCoordinator copy2;
    // copy2 = *coordinator_;

    // Static assert that the type is not copyable
    EXPECT_FALSE(std::is_copy_constructible_v<MockCollectiveCoordinator>);
    EXPECT_FALSE(std::is_copy_assignable_v<MockCollectiveCoordinator>);
}

/**
 * @test Coordinator cannot be moved (compile-time test)
 */
TEST_F(Test__ICollectiveCoordinator, NonMovable)
{
    // These should not compile if uncommented:
    // MockCollectiveCoordinator moved(std::move(*coordinator_));
    // MockCollectiveCoordinator moved2;
    // moved2 = std::move(*coordinator_);

    // Static assert that the type is not movable
    EXPECT_FALSE(std::is_move_constructible_v<MockCollectiveCoordinator>);
    EXPECT_FALSE(std::is_move_assignable_v<MockCollectiveCoordinator>);
}
