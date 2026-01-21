/**
 * @file Test__ROCmEventSynchronization.cpp
 * @brief Integration test for ROCm event-based synchronization
 *
 * **Purpose**: Validates that ROCm event synchronization correctly waits
 * only for the specific kernel that recorded the event, rather than
 * blocking on all GPU work.
 *
 * **Background**:
 * The ROCmBackend::waitForEvent() function must use hipEventSynchronize()
 * instead of hipStreamSynchronize(0) because:
 * - hipStreamSynchronize(0) waits for ALL work on the stream
 * - hipEventSynchronize(event) waits only for the specific event
 *
 * Using stream sync instead of event sync caused catastrophic performance
 * issues (1.8-10 seconds per sync instead of ~0.05ms) during snapshot
 * capture in parity tests, because each sync waited for all accumulated
 * GPU work instead of just the most recent kernel.
 *
 * **Test Strategy**:
 * 1. Launch multiple kernels with different completion times
 * 2. Record events after each kernel
 * 3. Verify that waiting on an early event completes quickly,
 *    even while later kernels are still running
 * 4. Verify that tensor coherence sync uses event-based waiting
 *
 * @note Requires ROCm device to run. Tests are skipped if no GPU available.
 *
 * @author GitHub Copilot
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <thread>

#ifdef HAVE_ROCM
#include "backends/rocm/ROCmBackend.h"
#include "tensors/TensorClasses.h"
#include "backends/BackendManager.h"
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;

// ============================================================================
// Test Fixture
// ============================================================================

#ifdef HAVE_ROCM

class Test__ROCmEventSynchronization : public ::testing::Test
{
protected:
    void SetUp() override
    {
        backend_ = std::make_unique<ROCmBackend>();
        device_count_ = backend_->deviceCount();

        if (device_count_ == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        device_id_ = 0;
        hipSetDevice(device_id_);
    }

    void TearDown() override
    {
        backend_.reset();
    }

    std::unique_ptr<ROCmBackend> backend_;
    int device_count_ = 0;
    int device_id_ = 0;
};

// ============================================================================
// Test: Event Creation and Destruction
// ============================================================================

/**
 * @brief Verify basic event lifecycle management
 */
TEST_F(Test__ROCmEventSynchronization, EventCreateAndDestroy)
{
    // Create an event
    void *event = backend_->createEvent(device_id_);
    ASSERT_NE(event, nullptr) << "Failed to create HIP event";

    // Destroy the event
    backend_->destroyEvent(event, device_id_);
    // If this doesn't crash, the test passes
}

// ============================================================================
// Test: Event Record and Wait
// ============================================================================

/**
 * @brief Verify event record and synchronization works correctly
 */
TEST_F(Test__ROCmEventSynchronization, EventRecordAndWait)
{
    // Create an event
    void *event = backend_->createEvent(device_id_);
    ASSERT_NE(event, nullptr);

    // Allocate small buffer and do a trivial memset
    const size_t bytes = 1024;
    void *d_ptr = backend_->allocate(bytes, device_id_);
    ASSERT_NE(d_ptr, nullptr);

    // Do some GPU work (trivial but requires kernel launch)
    hipError_t err = hipMemsetAsync(d_ptr, 0, bytes, 0);
    ASSERT_EQ(err, hipSuccess);

    // Record event after the work
    bool recorded = backend_->recordEvent(event, device_id_);
    ASSERT_TRUE(recorded) << "Failed to record event";

    // Wait for the event
    bool waited = backend_->waitForEvent(event, device_id_);
    EXPECT_TRUE(waited) << "Failed to wait for event";

    // Cleanup
    backend_->free(d_ptr, device_id_);
    backend_->destroyEvent(event, device_id_);
}

// ============================================================================
// Test: Event Sync is Fast (Not Blocking All Work)
// ============================================================================

/**
 * @brief Critical regression test: Verify event sync doesn't block on all stream work
 *
 * This test validates the fix where waitForEvent() uses hipEventSynchronize()
 * instead of hipStreamSynchronize(). The latter would cause catastrophic
 * performance issues by waiting for ALL queued work.
 *
 * Strategy:
 * 1. Launch a "quick" operation and record event A
 * 2. Launch a deliberately slow operation (large memset)
 * 3. Verify waiting on event A completes quickly (before slow op finishes)
 *
 * If waitForEvent incorrectly uses hipStreamSynchronize(0), the wait
 * will take as long as the slow operation. With proper hipEventSynchronize,
 * it should return almost immediately.
 */
TEST_F(Test__ROCmEventSynchronization, EventSyncIsEventSpecific_NotStreamWide)
{
    // Create events
    void *event_quick = backend_->createEvent(device_id_);
    void *event_slow = backend_->createEvent(device_id_);
    ASSERT_NE(event_quick, nullptr);
    ASSERT_NE(event_slow, nullptr);

    // Allocate buffers - small for quick, large for slow
    const size_t small_bytes = 1024;              // 1 KB - trivial
    const size_t large_bytes = 256 * 1024 * 1024; // 256 MB - takes time

    void *d_small = backend_->allocate(small_bytes, device_id_);
    void *d_large = backend_->allocate(large_bytes, device_id_);

    if (!d_small || !d_large)
    {
        // Skip if not enough memory
        if (d_small)
            backend_->free(d_small, device_id_);
        if (d_large)
            backend_->free(d_large, device_id_);
        backend_->destroyEvent(event_quick, device_id_);
        backend_->destroyEvent(event_slow, device_id_);
        GTEST_SKIP() << "Not enough GPU memory for large allocation test";
    }

    // Step 1: Launch quick operation and record event
    hipMemsetAsync(d_small, 0, small_bytes, 0);
    backend_->recordEvent(event_quick, device_id_);

    // Step 2: Launch slow operation (will still be running when we wait on quick event)
    // Use multiple iterations to ensure it takes time
    for (int i = 0; i < 10; ++i)
    {
        hipMemsetAsync(d_large, i, large_bytes, 0);
    }
    backend_->recordEvent(event_slow, device_id_);

    // Step 3: Measure time to wait on the QUICK event
    auto start = std::chrono::high_resolution_clock::now();
    bool waited = backend_->waitForEvent(event_quick, device_id_);
    auto end = std::chrono::high_resolution_clock::now();

    EXPECT_TRUE(waited);

    double quick_wait_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // The quick event wait should complete in under 100ms
    // If it's using stream sync incorrectly, it would take much longer
    // (the slow operation takes ~500ms+ on MI50)
    EXPECT_LT(quick_wait_ms, 100.0)
        << "Event wait took " << quick_wait_ms << "ms - suggests stream sync instead of event sync";

    // Now wait for the slow event to ensure cleanup is safe
    backend_->waitForEvent(event_slow, device_id_);

    // Cleanup
    backend_->free(d_small, device_id_);
    backend_->free(d_large, device_id_);
    backend_->destroyEvent(event_quick, device_id_);
    backend_->destroyEvent(event_slow, device_id_);

    std::cout << "[ROCm Event Sync] Quick event wait took: " << quick_wait_ms << " ms" << std::endl;
}

// ============================================================================
// Test: Mapped Tensor Coherence Uses Events
// ============================================================================

/**
 * @brief Verify that mapped tensor ensureOnHost() uses event-based sync
 *
 * When a tensor is marked device-dirty and then accessed via data(),
 * the sync should use the completion event, not a full stream sync.
 */
TEST_F(Test__ROCmEventSynchronization, MappedTensorCoherenceUsesEvents)
{
    // Create a mapped tensor
    DeviceId rocm_device = DeviceId::rocm(device_id_);
    auto tensor = FP32Tensor::createMapped({1024, 1024}, rocm_device); // 4MB

    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported";
    }

    // Ensure tensor is on device
    ASSERT_TRUE(tensor->ensureOnDevice(rocm_device));

    // Simulate a GPU write by marking device dirty
    // In real usage, this would be done after a kernel writes to the tensor
    tensor->mark_device_dirty();

    // Queue some slow work AFTER the tensor was marked dirty
    // If ensureOnHost uses stream sync, it will wait for this slow work
    // If it uses event sync, it will return quickly
    const size_t slow_bytes = 256 * 1024 * 1024;
    void *d_slow = backend_->allocate(slow_bytes, device_id_);
    if (d_slow)
    {
        for (int i = 0; i < 10; ++i)
        {
            hipMemsetAsync(d_slow, i, slow_bytes, 0);
        }
    }

    // Time the ensureOnHost call
    auto start = std::chrono::high_resolution_clock::now();
    tensor->ensureOnHost();
    auto end = std::chrono::high_resolution_clock::now();

    double sync_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Clean up slow work buffer
    if (d_slow)
    {
        hipDeviceSynchronize();
        backend_->free(d_slow, device_id_);
    }

    // The sync should be fast (under 100ms) if using event-based sync
    // With stream sync, it would wait for the slow memsets (~500ms+)
    EXPECT_LT(sync_ms, 100.0)
        << "ensureOnHost took " << sync_ms << "ms - suggests stream sync instead of event sync";

    std::cout << "[Mapped Tensor] ensureOnHost sync took: " << sync_ms << " ms" << std::endl;
}

// ============================================================================
// Test: Multiple Events Independent Sync
// ============================================================================

/**
 * @brief Verify multiple events can be synced independently
 *
 * Creates multiple events in a pipeline and verifies each can be
 * waited on independently without blocking on later events.
 */
TEST_F(Test__ROCmEventSynchronization, MultipleEventsIndependentSync)
{
    constexpr int NUM_EVENTS = 5;
    std::vector<void *> events(NUM_EVENTS);
    std::vector<void *> buffers(NUM_EVENTS);

    const size_t bytes_per_op = 10 * 1024 * 1024; // 10 MB each

    // Create events and launch work
    for (int i = 0; i < NUM_EVENTS; ++i)
    {
        events[i] = backend_->createEvent(device_id_);
        ASSERT_NE(events[i], nullptr);

        buffers[i] = backend_->allocate(bytes_per_op, device_id_);
        if (!buffers[i])
        {
            // Clean up and skip
            for (int j = 0; j < i; ++j)
            {
                backend_->free(buffers[j], device_id_);
                backend_->destroyEvent(events[j], device_id_);
            }
            backend_->destroyEvent(events[i], device_id_);
            GTEST_SKIP() << "Not enough GPU memory for multi-event test";
        }

        // Launch work and record event
        hipMemsetAsync(buffers[i], i, bytes_per_op, 0);
        backend_->recordEvent(events[i], device_id_);
    }

    // Wait on events in ORDER (each should complete quickly after the previous)
    std::vector<double> wait_times(NUM_EVENTS);
    for (int i = 0; i < NUM_EVENTS; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        bool waited = backend_->waitForEvent(events[i], device_id_);
        auto end = std::chrono::high_resolution_clock::now();

        EXPECT_TRUE(waited);
        wait_times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // All wait times should be reasonable (< 100ms each)
    for (int i = 0; i < NUM_EVENTS; ++i)
    {
        EXPECT_LT(wait_times[i], 100.0)
            << "Event " << i << " wait took " << wait_times[i] << "ms";
        std::cout << "[Event " << i << "] wait time: " << wait_times[i] << " ms" << std::endl;
    }

    // Cleanup
    for (int i = 0; i < NUM_EVENTS; ++i)
    {
        backend_->free(buffers[i], device_id_);
        backend_->destroyEvent(events[i], device_id_);
    }
}

#endif // HAVE_ROCM

// ============================================================================
// Fallback Test (No ROCm)
// ============================================================================

#ifndef HAVE_ROCM

TEST(Test__ROCmEventSynchronization, NoROCmAvailable)
{
    GTEST_SKIP() << "No ROCm support compiled (HAVE_ROCM=OFF)";
}

#endif
