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
#include "transfer/TransferEngine.h"
#include "../../../utils/ScopedGPUStream.h"
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
        (void)hipSetDevice(device_id_);
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
    llaminar2::test::ScopedGPUStream producer_stream(DeviceId::rocm(device_id_));
    const auto stream = static_cast<hipStream_t>(producer_stream.get());

    // Create an event
    void *event = backend_->createEvent(device_id_);
    ASSERT_NE(event, nullptr);

    // Allocate small buffer and do a trivial memset
    const size_t bytes = 1024;
    void *d_ptr = backend_->allocate(bytes, device_id_);
    ASSERT_NE(d_ptr, nullptr);

    // Do some GPU work (trivial but requires kernel launch)
    hipError_t err = hipMemsetAsync(d_ptr, 0, bytes, stream);
    ASSERT_EQ(err, hipSuccess);

    // Record event after the work
    bool recorded = backend_->recordEvent(event, device_id_, producer_stream.get());
    ASSERT_TRUE(recorded) << "Failed to record event";

    // Wait for the event
    bool waited = backend_->waitForEvent(event, device_id_);
    EXPECT_TRUE(waited) << "Failed to wait for event";

    // Cleanup
    backend_->free(d_ptr, device_id_);
    backend_->destroyEvent(event, device_id_);
}

/**
 * @brief Reject capture-time external publication and prove the post-replay handoff.
 *
 * IBackend events connect completed replay to consumers outside the DAG. The
 * backend rejects attempts to turn that handoff into an internal capture node,
 * then records the event after launch so another stream observes graph writes.
 */
TEST_F(Test__ROCmEventSynchronization,
       CaptureTimePublicationIsRejectedAndPostReplayEventOrdersConsumer)
{
    llaminar2::test::ScopedGPUStream producer_stream(DeviceId::rocm(device_id_));
    llaminar2::test::ScopedGPUStream consumer_stream(DeviceId::rocm(device_id_));
    const auto producer = static_cast<hipStream_t>(producer_stream.get());
    const auto consumer = static_cast<hipStream_t>(consumer_stream.get());

    void *event = backend_->createEvent(device_id_);
    void *device_value = backend_->allocate(sizeof(uint32_t), device_id_);
    ASSERT_NE(event, nullptr);
    ASSERT_NE(device_value, nullptr);

    ASSERT_EQ(hipMemsetAsync(device_value, 0, sizeof(uint32_t), producer),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(producer), hipSuccess);

    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(producer, hipStreamCaptureModeThreadLocal),
              hipSuccess);
    ASSERT_EQ(hipMemsetAsync(device_value, 0x2a, sizeof(uint32_t), producer),
              hipSuccess);
    EXPECT_FALSE(backend_->recordEvent(event, device_id_, producer_stream.get()))
        << "External publication must not silently disappear into a captured DAG";
    ASSERT_EQ(hipStreamEndCapture(producer, &graph), hipSuccess);
    ASSERT_NE(graph, nullptr);

    ASSERT_EQ(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
              hipSuccess);
    ASSERT_EQ(hipGraphLaunch(executable, producer), hipSuccess);
    ASSERT_TRUE(backend_->recordEvent(
        event, device_id_, producer_stream.get()));
    ASSERT_TRUE(backend_->streamWaitEvent(
        consumer_stream.get(), event, device_id_));

    uint32_t observed = 0;
    ASSERT_EQ(hipMemcpyAsync(
                  &observed,
                  device_value,
                  sizeof(observed),
                  hipMemcpyDeviceToHost,
                  consumer),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(consumer), hipSuccess);
    EXPECT_EQ(observed, 0x2a2a2a2au);

    ASSERT_EQ(hipGraphExecDestroy(executable), hipSuccess);
    ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
    backend_->free(device_value, device_id_);
    backend_->destroyEvent(event, device_id_);
}

/**
 * @brief Prove independent reader completion events fan in before row reuse.
 *
 * Device-resident MTP logical state is intentionally single-buffered. A producer
 * first publishes the rows, independent consumers then read them on their own
 * streams, and the next producer may overwrite the rows only after every reader
 * has completed. Each reader publishes an independent preallocated event; the
 * replacement writer waits both without introducing a reader-to-reader edge.
 * Both snapshots must retain the old value while the source is replaced.
 */
TEST_F(Test__ROCmEventSynchronization,
       IndependentReaderEventsFanInBeforeReplacementWriter)
{
    llaminar2::test::ScopedGPUStream producer_stream(DeviceId::rocm(device_id_));
    llaminar2::test::ScopedGPUStream reader_a_stream(DeviceId::rocm(device_id_));
    llaminar2::test::ScopedGPUStream reader_b_stream(DeviceId::rocm(device_id_));
    llaminar2::test::ScopedGPUStream writer_stream(DeviceId::rocm(device_id_));

    const auto producer = static_cast<hipStream_t>(producer_stream.get());
    const auto reader_a = static_cast<hipStream_t>(reader_a_stream.get());
    const auto reader_b = static_cast<hipStream_t>(reader_b_stream.get());
    const auto writer = static_cast<hipStream_t>(writer_stream.get());

    void *publication_ready = backend_->createEvent(device_id_);
    void *reader_a_done = backend_->createEvent(device_id_);
    void *reader_b_done = backend_->createEvent(device_id_);
    void *source = backend_->allocate(sizeof(uint32_t), device_id_);
    void *reader_a_snapshot = backend_->allocate(sizeof(uint32_t), device_id_);
    void *reader_b_snapshot = backend_->allocate(sizeof(uint32_t), device_id_);
    ASSERT_NE(publication_ready, nullptr);
    ASSERT_NE(reader_a_done, nullptr);
    ASSERT_NE(reader_b_done, nullptr);
    ASSERT_NE(source, nullptr);
    ASSERT_NE(reader_a_snapshot, nullptr);
    ASSERT_NE(reader_b_snapshot, nullptr);

    ASSERT_EQ(hipMemsetAsync(source, 0x11, sizeof(uint32_t), producer),
              hipSuccess);
    ASSERT_TRUE(backend_->recordEvent(
        publication_ready, device_id_, producer_stream.get()));

    ASSERT_TRUE(backend_->streamWaitEvent(
        reader_a_stream.get(), publication_ready, device_id_));
    ASSERT_TRUE(backend_->streamWaitEvent(
        reader_b_stream.get(), publication_ready, device_id_));
    ASSERT_EQ(hipMemcpyAsync(
                  reader_a_snapshot,
                  source,
                  sizeof(uint32_t),
                  hipMemcpyDeviceToDevice,
                  reader_a),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(
                  reader_b_snapshot,
                  source,
                  sizeof(uint32_t),
                  hipMemcpyDeviceToDevice,
                  reader_b),
              hipSuccess);

    ASSERT_TRUE(backend_->recordEvent(
        reader_a_done, device_id_, reader_a_stream.get()));
    ASSERT_TRUE(backend_->recordEvent(
        reader_b_done, device_id_, reader_b_stream.get()));

    ASSERT_TRUE(backend_->streamWaitEvent(
        writer_stream.get(), reader_a_done, device_id_));
    ASSERT_TRUE(backend_->streamWaitEvent(
        writer_stream.get(), reader_b_done, device_id_));
    ASSERT_EQ(hipMemsetAsync(source, 0x22, sizeof(uint32_t), writer),
              hipSuccess);

    uint32_t observed_source = 0;
    uint32_t observed_reader_a = 0;
    uint32_t observed_reader_b = 0;
    ASSERT_EQ(hipMemcpyAsync(
                  &observed_source,
                  source,
                  sizeof(uint32_t),
                  hipMemcpyDeviceToHost,
                  writer),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(
                  &observed_reader_a,
                  reader_a_snapshot,
                  sizeof(uint32_t),
                  hipMemcpyDeviceToHost,
                  writer),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(
                  &observed_reader_b,
                  reader_b_snapshot,
                  sizeof(uint32_t),
                  hipMemcpyDeviceToHost,
                  writer),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(writer), hipSuccess);

    EXPECT_EQ(observed_reader_a, 0x11111111u);
    EXPECT_EQ(observed_reader_b, 0x11111111u);
    EXPECT_EQ(observed_source, 0x22222222u);

    backend_->free(reader_b_snapshot, device_id_);
    backend_->free(reader_a_snapshot, device_id_);
    backend_->free(source, device_id_);
    backend_->destroyEvent(reader_b_done, device_id_);
    backend_->destroyEvent(reader_a_done, device_id_);
    backend_->destroyEvent(publication_ready, device_id_);
}

/**
 * @brief Event waits remain valid when another HIP child is ambient.
 *
 * The ROCm backend restores ambient device state after resource operations, so
 * success of the wait and preservation of device 1 together prove it selected
 * device 0 only for the duration of the backend call.
 */
TEST_F(Test__ROCmEventSynchronization,
       StreamWaitEventSelectsDeclaredDeviceOverAmbientDevice)
{
    if (device_count_ < 2)
        GTEST_SKIP() << "Requires at least two ROCm devices";

    constexpr int owner_device = 0;
    constexpr int ambient_device = 1;
    void *stream = backend_->createStream(owner_device);
    void *event = backend_->createEvent(owner_device);
    ASSERT_NE(stream, nullptr);
    ASSERT_NE(event, nullptr);
    ASSERT_TRUE(backend_->recordEvent(event, owner_device, stream));

    ASSERT_EQ(hipSetDevice(ambient_device), hipSuccess);
    ASSERT_TRUE(
        backend_->streamWaitEvent(stream, event, owner_device));

    int restored_device = -1;
    ASSERT_EQ(hipGetDevice(&restored_device), hipSuccess);
    EXPECT_EQ(restored_device, ambient_device)
        << "ROCmBackend must restore the caller's ambient HIP device";
    ASSERT_TRUE(backend_->synchronizeStream(stream, owner_device));

    backend_->destroyEvent(event, owner_device);
    backend_->destroyStream(stream, owner_device);
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
    llaminar2::test::ScopedGPUStream producer_stream(DeviceId::rocm(device_id_));
    const auto stream = static_cast<hipStream_t>(producer_stream.get());

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
    (void)hipMemsetAsync(d_small, 0, small_bytes, stream);
    backend_->recordEvent(event_quick, device_id_, producer_stream.get());

    // Step 2: Launch slow operation (will still be running when we wait on quick event)
    // Use multiple iterations to ensure it takes time
    for (int i = 0; i < 10; ++i)
    {
        (void)hipMemsetAsync(d_large, i, large_bytes, stream);
    }
    backend_->recordEvent(event_slow, device_id_, producer_stream.get());

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
    llaminar2::test::ScopedGPUStream producer_stream(rocm_device);
    auto tensor = FP32Tensor::createMapped({1024, 1024}, rocm_device); // 4MB

    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported";
    }

    // Ensure tensor is on device
    ASSERT_TRUE(tensor->ensureOnDevice(rocm_device, producer_stream.get()));

    // Simulate a GPU write by marking device dirty
    // In real usage, this would be done after a kernel writes to the tensor
    TransferEngine::publishCurrentDeviceWrite(tensor, producer_stream.get());

    // Queue some slow work AFTER the tensor was marked dirty
    // If ensureOnHost uses stream sync, it will wait for this slow work
    // If it uses event sync, it will return quickly
    llaminar2::test::ScopedGPUStream unrelated_stream(rocm_device);
    const auto unrelated_hip_stream =
        static_cast<hipStream_t>(unrelated_stream.get());
    const size_t slow_bytes = 256 * 1024 * 1024;
    void *d_slow = backend_->allocate(slow_bytes, device_id_);
    if (d_slow)
    {
        for (int i = 0; i < 10; ++i)
        {
            (void)hipMemsetAsync(
                d_slow, i, slow_bytes, unrelated_hip_stream);
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
        ASSERT_TRUE(
            backend_->synchronizeStream(unrelated_stream.get(), device_id_));
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
    llaminar2::test::ScopedGPUStream producer_stream(DeviceId::rocm(device_id_));
    const auto stream = static_cast<hipStream_t>(producer_stream.get());

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
        (void)hipMemsetAsync(buffers[i], i, bytes_per_op, stream);
        backend_->recordEvent(
            events[i], device_id_, producer_stream.get());
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
