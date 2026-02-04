/**
 * @file Test__GPUDeviceContextIntegration.cpp
 * @brief Integration tests for GPU Device Context infrastructure
 *
 * **Purpose**: Validates the GPUDeviceContextPool and IWorkerGPUContext
 * implementations for CUDA (NvidiaDeviceContext) and ROCm (AMDDeviceContext).
 *
 * **Tests Cover**:
 * - Device context creation and initialization
 * - Event creation, recording, and synchronization via context API
 * - Concurrent access from multiple threads
 * - Multi-device scenarios (when multiple GPUs available)
 * - Async submission with futures
 *
 * **Thread Safety Model**:
 * The IWorkerGPUContext interface uses a dedicated worker thread per device.
 * All GPU operations must be submitted via submitAndWait() or submitAsync(),
 * which ensures operations execute on the correct thread with proper context.
 *
 * **Note on CUDA/HIP header conflicts**:
 * This test file does NOT include cuda_runtime.h or hip_runtime.h directly
 * to avoid symbol conflicts (dim3, make_*vector functions, etc.) when both
 * HAVE_CUDA and HAVE_ROCM are defined.
 *
 * Instead, memory operations are done via the IBackend interface
 * (CUDABackend/ROCmBackend), and stream/event operations use the
 * platform-agnostic IWorkerGPUContext API (void* handles).
 *
 * @note Requires CUDA and/or ROCm devices to run. Tests skip gracefully
 *       if the required hardware is not available.
 *
 * @author GitHub Copilot
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"

#include <thread>
#include <vector>
#include <numeric>
#include <atomic>
#include <future>
#include <cstring>

using namespace llaminar2;

// ===========================================================================
// CUDA Integration Tests (via IWorkerGPUContext API)
// ===========================================================================

#ifdef HAVE_CUDA

/**
 * @brief Test basic context initialization for NVIDIA device
 *
 * Validates that the context is properly initialized after first access.
 */
TEST(Test__NvidiaDeviceContextIntegration, ContextInitialization)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // Context should be initialized after access
    EXPECT_TRUE(ctx.isInitialized()) << "Context should be initialized after first access";
    EXPECT_EQ(ctx.deviceOrdinal(), 0) << "Device ordinal should be 0";
    EXPECT_FALSE(ctx.deviceName().empty()) << "Device name should not be empty";
}

/**
 * @brief Test memory allocation and deallocation through Backend API
 *
 * Uses CUDABackend::allocate/free via BackendManager to test memory
 * operations in conjunction with device contexts.
 */
TEST(Test__NvidiaDeviceContextIntegration, MemoryAllocationViaBackend)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto *backend = getCUDABackend();
    ASSERT_NE(backend, nullptr) << "CUDABackend not available";

    constexpr size_t SIZE = 1024 * sizeof(float);
    int device_id = 0;

    // Allocate through backend
    void *d_ptr = backend->allocate(SIZE, device_id);
    ASSERT_NE(d_ptr, nullptr) << "Device allocation failed";

    // Set memory to verify it's accessible
    bool memset_ok = backend->memset(d_ptr, 0, SIZE, device_id);
    EXPECT_TRUE(memset_ok) << "Device memset failed";

    // Synchronize to ensure operation completed
    bool sync_ok = backend->synchronize(device_id);
    EXPECT_TRUE(sync_ok) << "Device synchronize failed";

    // Free through backend
    backend->free(d_ptr, device_id);
}

/**
 * @brief Test host-to-device and device-to-host transfers via Backend
 *
 * Validates data round-trip through CUDABackend API.
 */
TEST(Test__NvidiaDeviceContextIntegration, HostDeviceTransferViaBackend)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto *backend = getCUDABackend();
    ASSERT_NE(backend, nullptr) << "CUDABackend not available";

    constexpr size_t N = 1024;
    int device_id = 0;

    std::vector<float> h_src(N);
    std::vector<float> h_dst(N, 0.0f);

    // Initialize source with sequential values
    std::iota(h_src.begin(), h_src.end(), 1.0f);

    // Allocate device memory
    void *d_ptr = backend->allocate(N * sizeof(float), device_id);
    ASSERT_NE(d_ptr, nullptr) << "Device allocation failed";

    // H2D transfer
    bool h2d_ok = backend->hostToDevice(d_ptr, h_src.data(), N * sizeof(float), device_id);
    ASSERT_TRUE(h2d_ok) << "H2D transfer failed";

    // D2H transfer
    bool d2h_ok = backend->deviceToHost(h_dst.data(), d_ptr, N * sizeof(float), device_id);
    ASSERT_TRUE(d2h_ok) << "D2H transfer failed";

    // Synchronize
    backend->synchronize(device_id);

    // Verify data integrity
    EXPECT_EQ(h_src, h_dst) << "Data mismatch after round-trip transfer";

    // Cleanup
    backend->free(d_ptr, device_id);
}

/**
 * @brief Test event creation, recording, and synchronization via context
 *
 * Validates that the IWorkerGPUContext event API works correctly
 * through the worker thread.
 */
TEST(Test__NvidiaDeviceContextIntegration, EventSynchronization)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *event = nullptr;
    bool event_completed = false;

    ctx.submitAndWait([&]()
                      {
        // Create event
        event = ctx.createEvent();
        ASSERT_NE(event, nullptr) << "createEvent returned null";

        // Record event on default stream
        void *stream = ctx.defaultStream();
        EXPECT_NE(stream, nullptr) << "defaultStream returned null";
        ctx.recordEvent(event, stream);

        // Synchronize (wait for event to complete)
        ctx.synchronizeEvent(event);
        event_completed = true;

        // Cleanup
        ctx.destroyEvent(event); });

    EXPECT_TRUE(event_completed) << "Event synchronization did not complete";
}

/**
 * @brief Test concurrent access from multiple threads
 *
 * Validates that multiple threads can safely submit work to the same
 * device context concurrently.
 */
TEST(Test__NvidiaDeviceContextIntegration, ConcurrentAccess)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    constexpr int NUM_THREADS = 4;
    constexpr int OPS_PER_THREAD = 50;
    std::atomic<int> completed{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back([&ctx, &completed]()
                             {
            for (int i = 0; i < OPS_PER_THREAD; ++i)
            {
                ctx.submitAndWait([&ctx]() {
                    // Create and immediately destroy an event as a simple operation
                    void *event = ctx.createEvent();
                    EXPECT_NE(event, nullptr);
                    ctx.recordEvent(event, ctx.defaultStream());
                    ctx.synchronizeEvent(event);
                    ctx.destroyEvent(event);
                });
                completed.fetch_add(1, std::memory_order_relaxed);
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    EXPECT_EQ(completed.load(), NUM_THREADS * OPS_PER_THREAD)
        << "Not all operations completed successfully";
}

/**
 * @brief Test async submission with future-based waiting
 *
 * Validates that submitAsync() returns a valid future that can be
 * waited on, and that work completes correctly.
 */
TEST(Test__NvidiaDeviceContextIntegration, AsyncSubmission)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    std::atomic<bool> work_executed{false};

    // Submit async work
    auto future = ctx.submitAsync([&ctx, &work_executed]()
                                  {
        // Do a simple synchronization to ensure GPU is responsive
        void *event = ctx.createEvent();
        ctx.recordEvent(event, ctx.defaultStream());
        ctx.synchronizeEvent(event);
        ctx.destroyEvent(event);
        work_executed.store(true, std::memory_order_release); });

    // Wait for completion
    future.wait();

    EXPECT_TRUE(work_executed.load(std::memory_order_acquire))
        << "Async work was not executed";
}

/**
 * @brief Test device info methods are accessible without submitting work
 *
 * deviceOrdinal(), deviceName(), and isInitialized() should be thread-safe
 * and callable from any thread without going through submitAndWait().
 */
TEST(Test__NvidiaDeviceContextIntegration, DeviceInfoAccessible)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // These should be callable without submitting work
    int ordinal = ctx.deviceOrdinal();
    std::string name = ctx.deviceName();
    bool initialized = ctx.isInitialized();

    EXPECT_EQ(ordinal, 0) << "Device ordinal should be 0";
    EXPECT_FALSE(name.empty()) << "Device name should not be empty";
    EXPECT_TRUE(initialized) << "Context should be initialized after first access";

    // Verify device name matches backend info
    auto *backend = getCUDABackend();
    if (backend)
    {
        std::string backend_name = backend->deviceName(0);
        EXPECT_EQ(name, backend_name) << "Context device name should match backend";
    }
}

/**
 * @brief Test BLAS handle is available within submitted work
 *
 * Validates that blasHandle() returns a non-null handle when called
 * from within the worker thread context.
 */
TEST(Test__NvidiaDeviceContextIntegration, BlasHandleAvailable)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *blas_handle = nullptr;

    ctx.submitAndWait([&]()
                      { blas_handle = ctx.blasHandle(); });

    EXPECT_NE(blas_handle, nullptr) << "BLAS handle should be non-null";
}

/**
 * @brief Test stream creation and destruction
 *
 * Validates that additional streams can be created and destroyed
 * through the context API.
 */
TEST(Test__NvidiaDeviceContextIntegration, StreamCreationAndDestruction)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    ctx.submitAndWait([&ctx]()
                      {
        // Create a custom stream
        void *custom_stream = ctx.createStream();
        ASSERT_NE(custom_stream, nullptr) << "createStream returned null";

        // Verify it's different from default stream
        void *default_stream = ctx.defaultStream();
        EXPECT_NE(custom_stream, default_stream)
            << "Custom stream should be different from default";

        // Create and sync an event on the custom stream
        void *event = ctx.createEvent();
        ctx.recordEvent(event, custom_stream);
        ctx.synchronizeEvent(event);
        ctx.destroyEvent(event);

        // Destroy the stream
        ctx.destroyStream(custom_stream); });
}

/**
 * @brief Test synchronize() method blocks until GPU work completes
 */
TEST(Test__NvidiaDeviceContextIntegration, ContextSynchronize)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // Submit some work
    ctx.submitAndWait([&ctx]()
                      {
        void *event = ctx.createEvent();
        ctx.recordEvent(event, ctx.defaultStream());
        // Don't synchronize event here
        ctx.destroyEvent(event); });

    // Synchronize from outside the worker thread
    ctx.synchronize();

    // If we get here without hanging, synchronize worked
    SUCCEED();
}

#endif // HAVE_CUDA

// ===========================================================================
// ROCm Integration Tests (via IWorkerGPUContext API)
// ===========================================================================

#ifdef HAVE_ROCM

/**
 * @brief Test basic context initialization for AMD device
 *
 * Validates that the context is properly initialized after first access.
 */
TEST(Test__AMDDeviceContextIntegration, ContextInitialization)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    // Context should be initialized after access
    EXPECT_TRUE(ctx.isInitialized()) << "Context should be initialized after first access";
    EXPECT_EQ(ctx.deviceOrdinal(), 0) << "Device ordinal should be 0";
    EXPECT_FALSE(ctx.deviceName().empty()) << "Device name should not be empty";
}

/**
 * @brief Test memory allocation and deallocation through Backend API
 *
 * Uses ROCmBackend::allocate/free via BackendManager to test memory
 * operations in conjunction with device contexts.
 */
TEST(Test__AMDDeviceContextIntegration, MemoryAllocationViaBackend)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto *backend = getROCmBackend();
    ASSERT_NE(backend, nullptr) << "ROCmBackend not available";

    constexpr size_t SIZE = 1024 * sizeof(float);
    int device_id = 0;

    // Allocate through backend
    void *d_ptr = backend->allocate(SIZE, device_id);
    ASSERT_NE(d_ptr, nullptr) << "Device allocation failed";

    // Set memory to verify it's accessible
    bool memset_ok = backend->memset(d_ptr, 0, SIZE, device_id);
    EXPECT_TRUE(memset_ok) << "Device memset failed";

    // Synchronize to ensure operation completed
    bool sync_ok = backend->synchronize(device_id);
    EXPECT_TRUE(sync_ok) << "Device synchronize failed";

    // Free through backend
    backend->free(d_ptr, device_id);
}

/**
 * @brief Test host-to-device and device-to-host transfers via Backend
 *
 * Validates data round-trip through ROCmBackend API.
 */
TEST(Test__AMDDeviceContextIntegration, HostDeviceTransferViaBackend)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto *backend = getROCmBackend();
    ASSERT_NE(backend, nullptr) << "ROCmBackend not available";

    constexpr size_t N = 1024;
    int device_id = 0;

    std::vector<float> h_src(N);
    std::vector<float> h_dst(N, 0.0f);

    // Initialize source with sequential values
    std::iota(h_src.begin(), h_src.end(), 1.0f);

    // Allocate device memory
    void *d_ptr = backend->allocate(N * sizeof(float), device_id);
    ASSERT_NE(d_ptr, nullptr) << "Device allocation failed";

    // H2D transfer
    bool h2d_ok = backend->hostToDevice(d_ptr, h_src.data(), N * sizeof(float), device_id);
    ASSERT_TRUE(h2d_ok) << "H2D transfer failed";

    // D2H transfer
    bool d2h_ok = backend->deviceToHost(h_dst.data(), d_ptr, N * sizeof(float), device_id);
    ASSERT_TRUE(d2h_ok) << "D2H transfer failed";

    // Synchronize
    backend->synchronize(device_id);

    // Verify data integrity
    EXPECT_EQ(h_src, h_dst) << "Data mismatch after round-trip transfer";

    // Cleanup
    backend->free(d_ptr, device_id);
}

/**
 * @brief Test event creation, recording, and synchronization via context
 *
 * Validates that the IWorkerGPUContext event API works correctly
 * through the worker thread.
 */
TEST(Test__AMDDeviceContextIntegration, EventSynchronization)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *event = nullptr;
    bool event_completed = false;

    ctx.submitAndWait([&]()
                      {
        // Create event
        event = ctx.createEvent();
        ASSERT_NE(event, nullptr) << "createEvent returned null";

        // Record event on default stream
        void *stream = ctx.defaultStream();
        EXPECT_NE(stream, nullptr) << "defaultStream returned null";
        ctx.recordEvent(event, stream);

        // Synchronize (wait for event to complete)
        ctx.synchronizeEvent(event);
        event_completed = true;

        // Cleanup
        ctx.destroyEvent(event); });

    EXPECT_TRUE(event_completed) << "Event synchronization did not complete";
}

/**
 * @brief Test concurrent access from multiple threads
 *
 * Validates that multiple threads can safely submit work to the same
 * device context concurrently.
 */
TEST(Test__AMDDeviceContextIntegration, ConcurrentAccess)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    constexpr int NUM_THREADS = 4;
    constexpr int OPS_PER_THREAD = 50;
    std::atomic<int> completed{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back([&ctx, &completed]()
                             {
            for (int i = 0; i < OPS_PER_THREAD; ++i)
            {
                ctx.submitAndWait([&ctx]() {
                    // Create and immediately destroy an event as a simple operation
                    void *event = ctx.createEvent();
                    EXPECT_NE(event, nullptr);
                    ctx.recordEvent(event, ctx.defaultStream());
                    ctx.synchronizeEvent(event);
                    ctx.destroyEvent(event);
                });
                completed.fetch_add(1, std::memory_order_relaxed);
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    EXPECT_EQ(completed.load(), NUM_THREADS * OPS_PER_THREAD)
        << "Not all operations completed successfully";
}

/**
 * @brief Test async submission with future-based waiting
 *
 * Validates that submitAsync() returns a valid future that can be
 * waited on, and that work completes correctly.
 */
TEST(Test__AMDDeviceContextIntegration, AsyncSubmission)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    std::atomic<bool> work_executed{false};

    // Submit async work
    auto future = ctx.submitAsync([&ctx, &work_executed]()
                                  {
        // Do a simple synchronization to ensure GPU is responsive
        void *event = ctx.createEvent();
        ctx.recordEvent(event, ctx.defaultStream());
        ctx.synchronizeEvent(event);
        ctx.destroyEvent(event);
        work_executed.store(true, std::memory_order_release); });

    // Wait for completion
    future.wait();

    EXPECT_TRUE(work_executed.load(std::memory_order_acquire))
        << "Async work was not executed";
}

/**
 * @brief Test device info methods are accessible without submitting work
 *
 * deviceOrdinal(), deviceName(), and isInitialized() should be thread-safe
 * and callable from any thread without going through submitAndWait().
 */
TEST(Test__AMDDeviceContextIntegration, DeviceInfoAccessible)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    // These should be callable without submitting work
    int ordinal = ctx.deviceOrdinal();
    std::string name = ctx.deviceName();
    bool initialized = ctx.isInitialized();

    EXPECT_EQ(ordinal, 0) << "Device ordinal should be 0";
    EXPECT_FALSE(name.empty()) << "Device name should not be empty";
    EXPECT_TRUE(initialized) << "Context should be initialized after first access";

    // Verify device name matches backend info
    auto *backend = getROCmBackend();
    if (backend)
    {
        std::string backend_name = backend->deviceName(0);
        EXPECT_EQ(name, backend_name) << "Context device name should match backend";
    }
}

/**
 * @brief Test BLAS handle is available within submitted work
 *
 * Validates that blasHandle() returns a non-null handle when called
 * from within the worker thread context.
 */
TEST(Test__AMDDeviceContextIntegration, BlasHandleAvailable)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *blas_handle = nullptr;

    ctx.submitAndWait([&]()
                      { blas_handle = ctx.blasHandle(); });

    EXPECT_NE(blas_handle, nullptr) << "BLAS handle should be non-null";
}

/**
 * @brief Test stream creation and destruction
 *
 * Validates that additional streams can be created and destroyed
 * through the context API.
 */
TEST(Test__AMDDeviceContextIntegration, StreamCreationAndDestruction)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    ctx.submitAndWait([&ctx]()
                      {
        // Create a custom stream
        void *custom_stream = ctx.createStream();
        ASSERT_NE(custom_stream, nullptr) << "createStream returned null";

        // Verify it's different from default stream
        void *default_stream = ctx.defaultStream();
        EXPECT_NE(custom_stream, default_stream)
            << "Custom stream should be different from default";

        // Create and sync an event on the custom stream
        void *event = ctx.createEvent();
        ctx.recordEvent(event, custom_stream);
        ctx.synchronizeEvent(event);
        ctx.destroyEvent(event);

        // Destroy the stream
        ctx.destroyStream(custom_stream); });
}

/**
 * @brief Test synchronize() method blocks until GPU work completes
 */
TEST(Test__AMDDeviceContextIntegration, ContextSynchronize)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    // Submit some work
    ctx.submitAndWait([&ctx]()
                      {
        void *event = ctx.createEvent();
        ctx.recordEvent(event, ctx.defaultStream());
        // Don't synchronize event here
        ctx.destroyEvent(event); });

    // Synchronize from outside the worker thread
    ctx.synchronize();

    // If we get here without hanging, synchronize worked
    SUCCEED();
}

#endif // HAVE_ROCM

// ===========================================================================
// Multi-Device Tests (if multiple GPUs available)
// ===========================================================================

/**
 * @brief Test that multiple NVIDIA devices get distinct contexts
 *
 * Validates that each device has its own context with correct ordinal,
 * and that contexts can execute work independently.
 */
TEST(Test__GPUDeviceContextIntegration, MultipleNvidiaDevices)
{
    auto &pool = GPUDeviceContextPool::instance();
    if (!pool.hasNvidiaSupport() || pool.nvidiaDeviceCount() < 2)
    {
        GTEST_SKIP() << "Need at least 2 CUDA devices";
    }

    auto &ctx0 = pool.getNvidiaContext(0);
    auto &ctx1 = pool.getNvidiaContext(1);

    // Verify distinct contexts
    EXPECT_NE(&ctx0, &ctx1) << "Contexts for different devices should be distinct";
    EXPECT_EQ(ctx0.deviceOrdinal(), 0) << "Context 0 should have ordinal 0";
    EXPECT_EQ(ctx1.deviceOrdinal(), 1) << "Context 1 should have ordinal 1";

    // Verify they can both execute work independently
    std::atomic<int> counter{0};

    auto f0 = ctx0.submitAsync([&]()
                               { counter.fetch_add(1, std::memory_order_acq_rel); });
    auto f1 = ctx1.submitAsync([&]()
                               { counter.fetch_add(1, std::memory_order_acq_rel); });

    f0.wait();
    f1.wait();

    EXPECT_EQ(counter.load(), 2) << "Both contexts should have executed their work";
}

/**
 * @brief Test that multiple AMD devices get distinct contexts
 *
 * Validates that each device has its own context with correct ordinal,
 * and that contexts can execute work independently.
 */
TEST(Test__GPUDeviceContextIntegration, MultipleAMDDevices)
{
    auto &pool = GPUDeviceContextPool::instance();
    if (!pool.hasAMDSupport() || pool.amdDeviceCount() < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices";
    }

    auto &ctx0 = pool.getAMDContext(0);
    auto &ctx1 = pool.getAMDContext(1);

    // Verify distinct contexts
    EXPECT_NE(&ctx0, &ctx1) << "Contexts for different devices should be distinct";
    EXPECT_EQ(ctx0.deviceOrdinal(), 0) << "Context 0 should have ordinal 0";
    EXPECT_EQ(ctx1.deviceOrdinal(), 1) << "Context 1 should have ordinal 1";

    // Verify they can both execute work independently
    std::atomic<int> counter{0};

    auto f0 = ctx0.submitAsync([&]()
                               { counter.fetch_add(1, std::memory_order_acq_rel); });
    auto f1 = ctx1.submitAsync([&]()
                               { counter.fetch_add(1, std::memory_order_acq_rel); });

    f0.wait();
    f1.wait();

    EXPECT_EQ(counter.load(), 2) << "Both contexts should have executed their work";
}

// ===========================================================================
// Pool-Level Tests
// ===========================================================================

/**
 * @brief Test getContext() with device type string (CUDA)
 *
 * Validates that getContext("cuda", N) is equivalent to getNvidiaContext(N).
 */
TEST(Test__GPUDeviceContextIntegration, GetContextByTypeCUDA)
{
    auto &pool = GPUDeviceContextPool::instance();
    if (!pool.hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx_explicit = pool.getNvidiaContext(0);
    auto &ctx_by_type = pool.getContext("cuda", 0);

    EXPECT_EQ(&ctx_explicit, &ctx_by_type)
        << "getContext('cuda', 0) should return same context as getNvidiaContext(0)";

    // Also test case insensitivity
    auto &ctx_upper = pool.getContext("CUDA", 0);
    EXPECT_EQ(&ctx_explicit, &ctx_upper)
        << "getContext should be case-insensitive";
}

/**
 * @brief Test getContext() with device type string (ROCm)
 *
 * Validates that getContext("rocm", N) is equivalent to getAMDContext(N).
 */
TEST(Test__GPUDeviceContextIntegration, GetContextByTypeROCm)
{
    auto &pool = GPUDeviceContextPool::instance();
    if (!pool.hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx_explicit = pool.getAMDContext(0);
    auto &ctx_by_type = pool.getContext("rocm", 0);

    EXPECT_EQ(&ctx_explicit, &ctx_by_type)
        << "getContext('rocm', 0) should return same context as getAMDContext(0)";

    // Test "hip" alias
    auto &ctx_hip = pool.getContext("hip", 0);
    EXPECT_EQ(&ctx_explicit, &ctx_hip)
        << "getContext('hip', 0) should also return AMD context";
}

/**
 * @brief Test device count methods
 *
 * Validates that nvidiaDeviceCount() and amdDeviceCount() return
 * reasonable values consistent with hasNvidiaSupport()/hasAMDSupport().
 */
TEST(Test__GPUDeviceContextIntegration, DeviceCountConsistency)
{
    auto &pool = GPUDeviceContextPool::instance();

    int nvidia_count = pool.nvidiaDeviceCount();
    int amd_count = pool.amdDeviceCount();

    // If support is available, count should be > 0
    if (pool.hasNvidiaSupport())
    {
        EXPECT_GT(nvidia_count, 0) << "hasNvidiaSupport() but count is 0";
    }
    else
    {
        EXPECT_EQ(nvidia_count, 0) << "No NVIDIA support but count > 0";
    }

    if (pool.hasAMDSupport())
    {
        EXPECT_GT(amd_count, 0) << "hasAMDSupport() but count is 0";
    }
    else
    {
        EXPECT_EQ(amd_count, 0) << "No AMD support but count > 0";
    }
}

/**
 * @brief Test context reuse (same context returned for same device)
 *
 * Validates that calling getNvidiaContext(N) multiple times returns
 * the same context instance (lazy initialization, not per-call creation).
 */
TEST(Test__GPUDeviceContextIntegration, ContextReuse)
{
    auto &pool = GPUDeviceContextPool::instance();

    if (pool.hasNvidiaSupport())
    {
        auto &ctx1 = pool.getNvidiaContext(0);
        auto &ctx2 = pool.getNvidiaContext(0);
        EXPECT_EQ(&ctx1, &ctx2) << "Same NVIDIA device should return same context";
    }

    if (pool.hasAMDSupport())
    {
        auto &ctx1 = pool.getAMDContext(0);
        auto &ctx2 = pool.getAMDContext(0);
        EXPECT_EQ(&ctx1, &ctx2) << "Same AMD device should return same context";
    }
}

// ===========================================================================
// Stress Tests
// ===========================================================================

#ifdef HAVE_CUDA

/**
 * @brief Stress test: Many rapid event create/destroy cycles
 *
 * Validates that the context handles rapid event creation/destruction
 * without leaking resources or crashing.
 */
TEST(Test__NvidiaDeviceContextIntegration, StressRapidEventCycles)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    constexpr int NUM_CYCLES = 100;

    ctx.submitAndWait([&ctx]()
                      {
        for (int i = 0; i < NUM_CYCLES; ++i)
        {
            void *event = ctx.createEvent();
            ASSERT_NE(event, nullptr) << "Event creation failed on cycle " << i;
            ctx.recordEvent(event, ctx.defaultStream());
            ctx.synchronizeEvent(event);
            ctx.destroyEvent(event);
        } });
}

/**
 * @brief Stress test: Multiple events in flight
 *
 * Validates that multiple events can be created, recorded, and
 * synchronized without issues.
 */
TEST(Test__NvidiaDeviceContextIntegration, StressMultipleEvents)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    constexpr int NUM_EVENTS = 20;

    ctx.submitAndWait([&ctx]()
                      {
        std::vector<void *> events;
        events.reserve(NUM_EVENTS);

        // Create and record many events
        for (int i = 0; i < NUM_EVENTS; ++i)
        {
            void *event = ctx.createEvent();
            ASSERT_NE(event, nullptr) << "Failed to create event " << i;
            ctx.recordEvent(event, ctx.defaultStream());
            events.push_back(event);
        }

        // Synchronize all events
        for (int i = 0; i < NUM_EVENTS; ++i)
        {
            ctx.synchronizeEvent(events[i]);
        }

        // Destroy all events
        for (void *event : events)
        {
            ctx.destroyEvent(event);
        } });
}

/**
 * @brief Stress test: Rapid memory allocation cycles via Backend
 */
TEST(Test__NvidiaDeviceContextIntegration, StressRapidAllocations)
{
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    auto *backend = getCUDABackend();
    ASSERT_NE(backend, nullptr);

    constexpr int NUM_CYCLES = 100;
    constexpr size_t ALLOC_SIZE = 4096;
    int device_id = 0;

    for (int i = 0; i < NUM_CYCLES; ++i)
    {
        void *ptr = backend->allocate(ALLOC_SIZE, device_id);
        ASSERT_NE(ptr, nullptr) << "Allocation failed on cycle " << i;
        backend->free(ptr, device_id);
    }
}

#endif // HAVE_CUDA

#ifdef HAVE_ROCM

/**
 * @brief Stress test: Many rapid event create/destroy cycles (ROCm)
 *
 * Validates that the context handles rapid event creation/destruction
 * without leaking resources or crashing.
 */
TEST(Test__AMDDeviceContextIntegration, StressRapidEventCycles)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    constexpr int NUM_CYCLES = 100;

    ctx.submitAndWait([&ctx]()
                      {
        for (int i = 0; i < NUM_CYCLES; ++i)
        {
            void *event = ctx.createEvent();
            ASSERT_NE(event, nullptr) << "Event creation failed on cycle " << i;
            ctx.recordEvent(event, ctx.defaultStream());
            ctx.synchronizeEvent(event);
            ctx.destroyEvent(event);
        } });
}

/**
 * @brief Stress test: Multiple events in flight (ROCm)
 *
 * Validates that multiple events can be created, recorded, and
 * synchronized without issues.
 */
TEST(Test__AMDDeviceContextIntegration, StressMultipleEvents)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    constexpr int NUM_EVENTS = 20;

    ctx.submitAndWait([&ctx]()
                      {
        std::vector<void *> events;
        events.reserve(NUM_EVENTS);

        // Create and record many events
        for (int i = 0; i < NUM_EVENTS; ++i)
        {
            void *event = ctx.createEvent();
            ASSERT_NE(event, nullptr) << "Failed to create event " << i;
            ctx.recordEvent(event, ctx.defaultStream());
            events.push_back(event);
        }

        // Synchronize all events
        for (int i = 0; i < NUM_EVENTS; ++i)
        {
            ctx.synchronizeEvent(events[i]);
        }

        // Destroy all events
        for (void *event : events)
        {
            ctx.destroyEvent(event);
        } });
}

/**
 * @brief Stress test: Rapid memory allocation cycles via Backend (ROCm)
 */
TEST(Test__AMDDeviceContextIntegration, StressRapidAllocations)
{
    if (!GPUDeviceContextPool::instance().hasAMDSupport())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    auto *backend = getROCmBackend();
    ASSERT_NE(backend, nullptr);

    constexpr int NUM_CYCLES = 100;
    constexpr size_t ALLOC_SIZE = 4096;
    int device_id = 0;

    for (int i = 0; i < NUM_CYCLES; ++i)
    {
        void *ptr = backend->allocate(ALLOC_SIZE, device_id);
        ASSERT_NE(ptr, nullptr) << "Allocation failed on cycle " << i;
        backend->free(ptr, device_id);
    }
}

#endif // HAVE_ROCM
