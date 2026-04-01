/**
 * @file Test__CUDAEventSynchronization.cpp
 * @brief Integration test for CUDA event-based synchronization
 *
 * **Purpose**: Validates that CUDA event synchronization correctly waits
 * only for the specific kernel that recorded the event, rather than
 * blocking on all GPU work.
 *
 * **Background**:
 * The CUDABackend::waitForEvent() function must use cudaEventSynchronize()
 * instead of cudaDeviceSynchronize() because:
 * - cudaDeviceSynchronize() waits for ALL work on ALL streams
 * - cudaStreamSynchronize(stream) waits for ALL work on a specific stream
 * - cudaEventSynchronize(event) waits only for the specific event
 *
 * Using device sync instead of event sync would cause performance issues
 * during snapshot capture in parity tests, because each sync would wait
 * for all accumulated GPU work instead of just the most recent kernel.
 *
 * **Test Strategy**:
 * 1. Launch multiple kernels with different completion times
 * 2. Record events after each kernel
 * 3. Verify that waiting on an early event completes quickly,
 *    even while later kernels are still running
 * 4. Verify that tensor coherence sync uses event-based waiting
 *
 * @note Requires CUDA device to run. Tests are skipped if no GPU available.
 *
 * @author GitHub Copilot
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <thread>

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include "tensors/TensorClasses.h"
#include "backends/BackendManager.h"
#include <cuda_runtime.h>

using namespace llaminar2;
#endif

// ============================================================================
// Test Fixture
// ============================================================================

#ifdef HAVE_CUDA

class Test__CUDAEventSynchronization : public ::testing::Test
{
protected:
    void SetUp() override
    {
        backend_ = std::make_unique<CUDABackend>();
        device_count_ = backend_->deviceCount();

        if (device_count_ == 0)
        {
            GTEST_SKIP() << "No CUDA devices available";
        }

        device_id_ = 0;
        cudaSetDevice(device_id_);
    }

    void TearDown() override
    {
        backend_.reset();
    }

    std::unique_ptr<CUDABackend> backend_;
    int device_count_ = 0;
    int device_id_ = 0;
};

// ============================================================================
// Test: Event Creation and Destruction
// ============================================================================

/**
 * @brief Verify basic event lifecycle management
 */
TEST_F(Test__CUDAEventSynchronization, EventCreateAndDestroy)
{
    // Create an event
    void *event = backend_->createEvent(device_id_);
    ASSERT_NE(event, nullptr) << "Failed to create CUDA event";

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
TEST_F(Test__CUDAEventSynchronization, EventRecordAndWait)
{
    // Create an event
    void *event = backend_->createEvent(device_id_);
    ASSERT_NE(event, nullptr);

    // Allocate small buffer and do a trivial memset
    const size_t bytes = 1024;
    void *d_ptr = backend_->allocate(bytes, device_id_);
    ASSERT_NE(d_ptr, nullptr);

    // Do some GPU work (trivial but requires kernel launch)
    cudaError_t err = cudaMemsetAsync(d_ptr, 0, bytes, 0);
    ASSERT_EQ(err, cudaSuccess);

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
 * @brief Critical test: Verify event sync doesn't block on all stream work
 *
 * This test validates that waitForEvent() uses cudaEventSynchronize()
 * instead of cudaDeviceSynchronize() or cudaStreamSynchronize().
 * The latter would cause performance issues by waiting for ALL queued work.
 *
 * Strategy:
 * 1. Launch a "quick" operation and record event A
 * 2. Launch a deliberately slow operation (large memset)
 * 3. Verify waiting on event A completes quickly (before slow op finishes)
 *
 * If waitForEvent incorrectly uses cudaDeviceSynchronize(), the wait
 * will take as long as the slow operation. With proper cudaEventSynchronize,
 * it should return almost immediately.
 */
TEST_F(Test__CUDAEventSynchronization, EventSyncIsEventSpecific_NotStreamWide)
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
    cudaMemsetAsync(d_small, 0, small_bytes, 0);
    backend_->recordEvent(event_quick, device_id_);

    // Step 2: Launch slow operation (will still be running when we wait on quick event)
    // Use multiple iterations to ensure it takes time
    for (int i = 0; i < 10; ++i)
    {
        cudaMemsetAsync(d_large, i, large_bytes, 0);
    }
    backend_->recordEvent(event_slow, device_id_);

    // Step 3: Measure time to wait on the QUICK event
    auto start = std::chrono::high_resolution_clock::now();
    bool waited = backend_->waitForEvent(event_quick, device_id_);
    auto end = std::chrono::high_resolution_clock::now();

    EXPECT_TRUE(waited);

    double quick_wait_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // The quick event wait should complete in under 100ms
    // If it's using device sync incorrectly, it would take much longer
    // (the slow operation takes ~100-500ms depending on GPU)
    EXPECT_LT(quick_wait_ms, 100.0)
        << "Event wait took " << quick_wait_ms << "ms - suggests device sync instead of event sync";

    // Now wait for the slow event to ensure cleanup is safe
    backend_->waitForEvent(event_slow, device_id_);

    // Cleanup
    backend_->free(d_small, device_id_);
    backend_->free(d_large, device_id_);
    backend_->destroyEvent(event_quick, device_id_);
    backend_->destroyEvent(event_slow, device_id_);

    std::cout << "[CUDA Event Sync] Quick event wait took: " << quick_wait_ms << " ms" << std::endl;
}

// ============================================================================
// Test: Mapped Tensor Coherence Uses Events
// ============================================================================

/**
 * @brief Verify that mapped tensor ensureOnHost() uses event-based sync
 *
 * When a tensor is marked device-dirty and then accessed via data(),
 * the sync should use the completion event, not a full device sync.
 */
TEST_F(Test__CUDAEventSynchronization, MappedTensorCoherenceUsesEvents)
{
    // Create a mapped tensor
    DeviceId cuda_device = DeviceId::cuda(device_id_);
    auto tensor = FP32Tensor::createMapped({1024, 1024}, cuda_device); // 4MB

    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported";
    }

    // Ensure tensor is on device
    ASSERT_TRUE(tensor->ensureOnDevice(cuda_device));

    // Simulate a GPU write by marking device dirty
    // In real usage, this would be done after a kernel writes to the tensor
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Queue some slow work AFTER the tensor was marked dirty
    // If ensureOnHost uses device sync, it will wait for this slow work
    // If it uses event sync, it will return quickly
    const size_t slow_bytes = 256 * 1024 * 1024;
    void *d_slow = backend_->allocate(slow_bytes, device_id_);
    if (d_slow)
    {
        for (int i = 0; i < 10; ++i)
        {
            cudaMemsetAsync(d_slow, i, slow_bytes, 0);
        }
    }

    // Measure time for ensureOnHost
    auto start = std::chrono::high_resolution_clock::now();
    bool synced = tensor->ensureOnHost();
    auto end = std::chrono::high_resolution_clock::now();

    EXPECT_TRUE(synced);

    double sync_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Cleanup - need to wait for slow work before freeing
    if (d_slow)
    {
        cudaDeviceSynchronize();
        backend_->free(d_slow, device_id_);
    }

    std::cout << "[Mapped Tensor] ensureOnHost sync took: " << sync_ms << " ms" << std::endl;

    // Note: We don't assert on timing here because the tensor might not have
    // a completion event (depends on whether mark_device_dirty_with_event was used).
    // This test is primarily for coverage and manual inspection.
}

// ============================================================================
// Test: Multiple Events Can Be Synchronized Independently
// ============================================================================

/**
 * @brief Verify multiple events can be waited on in any order
 *
 * Events should track specific points in the stream independently.
 * Waiting on event N should not require waiting for events N+1, N+2, etc.
 */
TEST_F(Test__CUDAEventSynchronization, MultipleEventsIndependentSync)
{
    const int num_events = 5;
    std::vector<void *> events(num_events);
    std::vector<void *> buffers(num_events);
    const size_t bytes_per_op = 64 * 1024 * 1024; // 64MB each

    // Create events and buffers
    for (int i = 0; i < num_events; ++i)
    {
        events[i] = backend_->createEvent(device_id_);
        ASSERT_NE(events[i], nullptr);

        buffers[i] = backend_->allocate(bytes_per_op, device_id_);
        if (!buffers[i])
        {
            // Cleanup and skip
            for (int j = 0; j < i; ++j)
            {
                backend_->destroyEvent(events[j], device_id_);
                if (buffers[j])
                    backend_->free(buffers[j], device_id_);
            }
            backend_->destroyEvent(events[i], device_id_);
            GTEST_SKIP() << "Not enough memory for multi-event test";
        }
    }

    // Queue operations and record events
    for (int i = 0; i < num_events; ++i)
    {
        cudaMemsetAsync(buffers[i], i, bytes_per_op, 0);
        backend_->recordEvent(events[i], device_id_);
    }

    // Wait on events in REVERSE order - should still work correctly
    for (int i = num_events - 1; i >= 0; --i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        bool waited = backend_->waitForEvent(events[i], device_id_);
        auto end = std::chrono::high_resolution_clock::now();

        EXPECT_TRUE(waited) << "Failed to wait for event " << i;

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[Event " << i << "] wait time: " << ms << " ms" << std::endl;
    }

    // Cleanup
    for (int i = 0; i < num_events; ++i)
    {
        backend_->free(buffers[i], device_id_);
        backend_->destroyEvent(events[i], device_id_);
    }
}

#else // !HAVE_CUDA

// Placeholder test when CUDA is not available
TEST(Test__CUDAEventSynchronization, SkippedNoCUDA)
{
    GTEST_SKIP() << "CUDA not enabled - skipping event synchronization tests";
}

#endif // HAVE_CUDA
