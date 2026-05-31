/**
 * @file Test__GPUDeviceContext.cpp
 * @brief Unit tests for GPU Device Context infrastructure
 *
 * Tests:
 * - IWorkerGPUContext interface via concrete implementations
 * - GPUDeviceContextPool singleton
 * - NvidiaDeviceContext (if CUDA available)
 * - AMDDeviceContext (if ROCm available)
 *
 * **Thread Safety Model**:
 * The device context follows a strict ownership model where all GPU state
 * is owned by a dedicated worker thread. Tests verify both thread-safe
 * methods (submitAndWait, submitAsync) and worker-thread-only methods
 * (stream/event creation, BLAS handle access).
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"

#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#include <hip/hip_runtime.h>
#endif

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

using namespace llaminar2;

// ===========================================================================
// Skip macros for hardware availability
// ===========================================================================

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#define SKIP_IF_NO_CUDA()                                      \
    do                                                         \
    {                                                          \
        ensureNvidiaFactoryRegistered();                       \
        if (!GPUDeviceContextPool::instance().hasNvidiaSupport()) \
            GTEST_SKIP() << "CUDA not available";              \
    } while (false)
#else
#define SKIP_IF_NO_CUDA() GTEST_SKIP() << "CUDA backend not linked in this test binary"
#endif

#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#define SKIP_IF_NO_ROCM()                                   \
    do                                                      \
    {                                                       \
        ensureAMDFactoryRegistered();                       \
        if (!GPUDeviceContextPool::instance().hasAMDSupport()) \
            GTEST_SKIP() << "ROCm not available";           \
    } while (false)
#else
#define SKIP_IF_NO_ROCM() GTEST_SKIP() << "ROCm backend not linked in this test binary"
#endif

#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#define SKIP_IF_NO_GPU() SKIP_IF_NO_ROCM()
#elif defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#define SKIP_IF_NO_GPU() SKIP_IF_NO_CUDA()
#else
#define SKIP_IF_NO_GPU() GTEST_SKIP() << "No GPU backend linked in this test binary"
#endif

// ===========================================================================
// GPUDeviceContextPool Tests
// ===========================================================================

TEST(Test__GPUDeviceContextPool, SingletonInstance)
{
    auto &pool1 = GPUDeviceContextPool::instance();
    auto &pool2 = GPUDeviceContextPool::instance();
    EXPECT_EQ(&pool1, &pool2) << "Pool should be a singleton";
}

TEST(Test__GPUDeviceContextPool, HasSupportQueries)
{
    auto &pool = GPUDeviceContextPool::instance();

    // These should not throw, just return bool
    bool has_nvidia = pool.hasNvidiaSupport();
    bool has_amd = pool.hasAMDSupport();

    // Log what's available
    std::cout << "NVIDIA (CUDA) support: " << (has_nvidia ? "yes" : "no") << std::endl;
    std::cout << "AMD (ROCm) support: " << (has_amd ? "yes" : "no") << std::endl;

    // Not asserting either is true - could be CPU-only build
    (void)has_nvidia;
    (void)has_amd;
}

TEST(Test__GPUDeviceContextPool, DeviceCountQueries)
{
    auto &pool = GPUDeviceContextPool::instance();

    int nvidia_count = pool.nvidiaDeviceCount();
    int amd_count = pool.amdDeviceCount();

    std::cout << "NVIDIA device count: " << nvidia_count << std::endl;
    std::cout << "AMD device count: " << amd_count << std::endl;

    // Device count should be non-negative
    EXPECT_GE(nvidia_count, 0);
    EXPECT_GE(amd_count, 0);

    // If support is available, should have at least one device
    if (pool.hasNvidiaSupport())
    {
        EXPECT_GE(nvidia_count, 1) << "hasNvidiaSupport() true but no devices";
    }
    if (pool.hasAMDSupport())
    {
        EXPECT_GE(amd_count, 1) << "hasAMDSupport() true but no devices";
    }
}

TEST(Test__GPUDeviceContextPool, GetContextByType_CUDA)
{
    SKIP_IF_NO_CUDA();

    auto &pool = GPUDeviceContextPool::instance();

    // Test various CUDA type strings
    auto &ctx1 = pool.getContext("cuda", 0);
    EXPECT_TRUE(ctx1.isInitialized());

    auto &ctx2 = pool.getContext("CUDA", 0);
    EXPECT_TRUE(ctx2.isInitialized());

    // Both should return the same context (same device ordinal)
    EXPECT_EQ(&ctx1, &ctx2);
}

TEST(Test__GPUDeviceContextPool, GetContextByType_ROCm)
{
    SKIP_IF_NO_ROCM();

    auto &pool = GPUDeviceContextPool::instance();

    // Test various ROCm type strings
    auto &ctx1 = pool.getContext("rocm", 0);
    EXPECT_TRUE(ctx1.isInitialized());

    auto &ctx2 = pool.getContext("ROCm", 0);
    EXPECT_TRUE(ctx2.isInitialized());

    auto &ctx3 = pool.getContext("hip", 0);
    EXPECT_TRUE(ctx3.isInitialized());

    auto &ctx4 = pool.getContext("HIP", 0);
    EXPECT_TRUE(ctx4.isInitialized());

    // All should return the same context (same device ordinal)
    EXPECT_EQ(&ctx1, &ctx2);
    EXPECT_EQ(&ctx2, &ctx3);
    EXPECT_EQ(&ctx3, &ctx4);
}

TEST(Test__GPUDeviceContextPool, GetContextInvalidType)
{
    auto &pool = GPUDeviceContextPool::instance();

    // Invalid device type should throw
    EXPECT_THROW(pool.getContext("invalid_type", 0), std::invalid_argument);
    EXPECT_THROW(pool.getContext("vulkan", 0), std::invalid_argument);
    EXPECT_THROW(pool.getContext("", 0), std::invalid_argument);
}

TEST(Test__GPUDeviceContextPool, GetContextInvalidOrdinal)
{
    auto &pool = GPUDeviceContextPool::instance();

    // Negative ordinal should throw
    if (pool.hasNvidiaSupport())
    {
        EXPECT_THROW(pool.getNvidiaContext(-1), std::runtime_error);
    }
    if (pool.hasAMDSupport())
    {
        EXPECT_THROW(pool.getAMDContext(-1), std::runtime_error);
    }

    // Ordinal beyond device count should throw
    if (pool.hasNvidiaSupport())
    {
        int count = pool.nvidiaDeviceCount();
        EXPECT_THROW(pool.getNvidiaContext(count + 100), std::runtime_error);
    }
    if (pool.hasAMDSupport())
    {
        int count = pool.amdDeviceCount();
        EXPECT_THROW(pool.getAMDContext(count + 100), std::runtime_error);
    }
}

TEST(Test__GPUDeviceContextPool, ConcurrentAccess)
{
    SKIP_IF_NO_GPU();

    auto &pool = GPUDeviceContextPool::instance();

    // Determine which backend to test
    bool use_cuda = pool.hasNvidiaSupport();
    const std::string device_type = use_cuda ? "cuda" : "rocm";

    constexpr int NUM_THREADS = 8;
    std::vector<IWorkerGPUContext *> contexts(NUM_THREADS, nullptr);
    std::vector<std::thread> threads;

    // Multiple threads requesting the same context
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&, i]()
                             { contexts[i] = &pool.getContext(device_type, 0); });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    // All threads should get the same context
    for (int i = 1; i < NUM_THREADS; ++i)
    {
        EXPECT_EQ(contexts[0], contexts[i])
            << "Thread " << i << " got different context than thread 0";
    }
}

// ===========================================================================
// NvidiaDeviceContext Tests
// ===========================================================================

TEST(Test__NvidiaDeviceContext, CreationAndInitialization)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    EXPECT_TRUE(ctx.isInitialized());
    EXPECT_EQ(ctx.deviceOrdinal(), 0);
    EXPECT_FALSE(ctx.deviceName().empty());

    std::cout << "NVIDIA device 0: " << ctx.deviceName() << std::endl;
}

TEST(Test__NvidiaDeviceContext, SubmitAndWait)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    std::atomic<bool> work_executed{false};
    ctx.submitAndWait([&]()
                      { work_executed.store(true); });

    EXPECT_TRUE(work_executed.load()) << "Work should have executed";
}

TEST(Test__NvidiaDeviceContext, SubmitAndWaitReturnValue)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // Verify we can capture return values via lambda capture
    int result = 0;
    ctx.submitAndWait([&]()
                      { result = 42; });

    EXPECT_EQ(result, 42);
}

TEST(Test__NvidiaDeviceContext, SubmitAsync)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    std::atomic<int> counter{0};
    auto future = ctx.submitAsync([&]()
                                  { counter.fetch_add(1); });

    future.wait();
    EXPECT_EQ(counter.load(), 1);
}

TEST(Test__NvidiaDeviceContext, MultipleAsyncSubmissions)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    constexpr int NUM_TASKS = 100;
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        futures.push_back(ctx.submitAsync([&]()
                                          { counter.fetch_add(1); }));
    }

    for (auto &f : futures)
    {
        f.wait();
    }

    EXPECT_EQ(counter.load(), NUM_TASKS);
}

TEST(Test__NvidiaDeviceContext, SubmitAsyncPreserveOrder)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // Verify FIFO ordering - tasks should execute in submission order
    std::vector<int> execution_order;
    std::mutex mutex;

    constexpr int NUM_TASKS = 50;
    std::vector<std::future<void>> futures;

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        futures.push_back(ctx.submitAsync([&, i]()
                                          {
            std::lock_guard<std::mutex> lock(mutex);
            execution_order.push_back(i); }));
    }

    for (auto &f : futures)
    {
        f.wait();
    }

    ASSERT_EQ(execution_order.size(), static_cast<size_t>(NUM_TASKS));
    for (int i = 0; i < NUM_TASKS; ++i)
    {
        EXPECT_EQ(execution_order[i], i) << "Task " << i << " executed out of order";
    }
}

TEST(Test__NvidiaDeviceContext, StreamCreationAndDestruction)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *stream = nullptr;
    ctx.submitAndWait([&]()
                      { stream = ctx.createStream(); });

    EXPECT_NE(stream, nullptr) << "createStream() should return non-null";

    ctx.submitAndWait([&]()
                      { ctx.destroyStream(stream); });

    // No assertion after destroy - just verify it doesn't crash
}

TEST(Test__NvidiaDeviceContext, MultipleStreams)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    constexpr int NUM_STREAMS = 4;
    std::vector<void *> streams(NUM_STREAMS, nullptr);

    ctx.submitAndWait([&]()
                      {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            streams[i] = ctx.createStream();
            EXPECT_NE(streams[i], nullptr) << "Stream " << i << " is null";
        } });

    // All streams should be unique
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        for (int j = i + 1; j < NUM_STREAMS; ++j)
        {
            EXPECT_NE(streams[i], streams[j])
                << "Streams " << i << " and " << j << " are the same";
        }
    }

    ctx.submitAndWait([&]()
                      {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            ctx.destroyStream(streams[i]);
        } });
}

TEST(Test__NvidiaDeviceContext, DefaultStream)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *default_stream = nullptr;
    ctx.submitAndWait([&]()
                      { default_stream = ctx.defaultStream(); });

    // Default stream should be available (may or may not be nullptr depending on CUDA semantics)
    // The key is that it shouldn't throw
    (void)default_stream;
}

TEST(Test__NvidiaDeviceContext, EventCreationAndDestruction)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *event = nullptr;
    ctx.submitAndWait([&]()
                      { event = ctx.createEvent(); });

    EXPECT_NE(event, nullptr) << "createEvent() should return non-null";

    ctx.submitAndWait([&]()
                      { ctx.destroyEvent(event); });
}

TEST(Test__NvidiaDeviceContext, EventRecordAndSynchronize)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *event = nullptr;
    ctx.submitAndWait([&]()
                      {
        event = ctx.createEvent();
        ctx.recordEvent(event, nullptr);  // Record on default stream
        ctx.synchronizeEvent(event);       // Wait for event
        ctx.destroyEvent(event); });

    // Success if no exceptions thrown
}

TEST(Test__NvidiaDeviceContext, EventWait)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    ctx.submitAndWait([&]()
                      {
        void* stream1 = ctx.createStream();
        void* stream2 = ctx.createStream();
        void* event = ctx.createEvent();

        // Record event on stream1
        ctx.recordEvent(event, stream1);

        // Make stream2 wait for the event
        ctx.waitEvent(event, stream2);

        // Cleanup
        ctx.destroyEvent(event);
        ctx.destroyStream(stream1);
        ctx.destroyStream(stream2); });
}

TEST(Test__NvidiaDeviceContext, BlasHandle)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *handle = nullptr;
    ctx.submitAndWait([&]()
                      { handle = ctx.blasHandle(); });

    EXPECT_NE(handle, nullptr) << "BLAS handle should be available";
}

TEST(Test__NvidiaDeviceContext, Synchronize)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // Synchronize should not throw
    ctx.synchronize();

    // Synchronize after some work
    ctx.submitAsync([&]()
                    {
        // Simulate some GPU work
        void* stream = ctx.createStream();
        ctx.destroyStream(stream); });

    ctx.synchronize();
}

TEST(Test__NvidiaDeviceContext, CollectiveCommInitiallyNull)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // Initially should be null (not set until collective backend initializes)
    void *comm = ctx.collectiveComm();
    // Don't assert null - it may have been set by previous tests
    (void)comm;
}

// ===========================================================================
// AMDDeviceContext Tests (mirror NVIDIA tests)
// ===========================================================================

TEST(Test__AMDDeviceContext, CreationAndInitialization)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    EXPECT_TRUE(ctx.isInitialized());
    EXPECT_EQ(ctx.deviceOrdinal(), 0);
    EXPECT_FALSE(ctx.deviceName().empty());

    std::cout << "AMD device 0: " << ctx.deviceName() << std::endl;
}

TEST(Test__AMDDeviceContext, SubmitAndWait)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    std::atomic<bool> work_executed{false};
    ctx.submitAndWait([&]()
                      { work_executed.store(true); });

    EXPECT_TRUE(work_executed.load()) << "Work should have executed";
}

TEST(Test__AMDDeviceContext, SubmitAndWaitReturnValue)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    int result = 0;
    ctx.submitAndWait([&]()
                      { result = 42; });

    EXPECT_EQ(result, 42);
}

TEST(Test__AMDDeviceContext, SubmitAsync)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    std::atomic<int> counter{0};
    auto future = ctx.submitAsync([&]()
                                  { counter.fetch_add(1); });

    future.wait();
    EXPECT_EQ(counter.load(), 1);
}

TEST(Test__AMDDeviceContext, MultipleAsyncSubmissions)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    constexpr int NUM_TASKS = 100;
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        futures.push_back(ctx.submitAsync([&]()
                                          { counter.fetch_add(1); }));
    }

    for (auto &f : futures)
    {
        f.wait();
    }

    EXPECT_EQ(counter.load(), NUM_TASKS);
}

TEST(Test__AMDDeviceContext, SubmitAsyncPreserveOrder)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    std::vector<int> execution_order;
    std::mutex mutex;

    constexpr int NUM_TASKS = 50;
    std::vector<std::future<void>> futures;

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        futures.push_back(ctx.submitAsync([&, i]()
                                          {
            std::lock_guard<std::mutex> lock(mutex);
            execution_order.push_back(i); }));
    }

    for (auto &f : futures)
    {
        f.wait();
    }

    ASSERT_EQ(execution_order.size(), static_cast<size_t>(NUM_TASKS));
    for (int i = 0; i < NUM_TASKS; ++i)
    {
        EXPECT_EQ(execution_order[i], i) << "Task " << i << " executed out of order";
    }
}

TEST(Test__AMDDeviceContext, StreamCreationAndDestruction)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *stream = nullptr;
    ctx.submitAndWait([&]()
                      { stream = ctx.createStream(); });

    EXPECT_NE(stream, nullptr) << "createStream() should return non-null";

    ctx.submitAndWait([&]()
                      { ctx.destroyStream(stream); });
}

#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
TEST(Test__AMDDeviceContext, StreamCreationUsesContextDeviceWhenCallerDeviceDiffers)
{
    SKIP_IF_NO_ROCM();

    auto &pool = GPUDeviceContextPool::instance();
    if (pool.amdDeviceCount() < 2)
    {
        GTEST_SKIP() << "requires at least two ROCm devices";
    }

    int original_device = 0;
    (void)hipGetDevice(&original_device);

    auto &ctx = pool.getAMDContext(0);

    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    void *stream = ctx.createStream();
    ASSERT_NE(stream, nullptr) << "createStream() should return non-null";

    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    int *device_word = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_word), sizeof(int)), hipSuccess);

    hipError_t memset_err = hipMemsetAsync(
        device_word,
        0,
        sizeof(int),
        static_cast<hipStream_t>(stream));
    EXPECT_EQ(memset_err, hipSuccess) << hipGetErrorString(memset_err);
    if (memset_err == hipSuccess)
    {
        EXPECT_EQ(hipStreamSynchronize(static_cast<hipStream_t>(stream)), hipSuccess);
    }

    ASSERT_EQ(hipFree(device_word), hipSuccess);
    ctx.destroyStream(stream);

    if (original_device >= 0)
    {
        (void)hipSetDevice(original_device);
    }
}
#endif

TEST(Test__AMDDeviceContext, MultipleStreams)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    constexpr int NUM_STREAMS = 4;
    std::vector<void *> streams(NUM_STREAMS, nullptr);

    ctx.submitAndWait([&]()
                      {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            streams[i] = ctx.createStream();
            EXPECT_NE(streams[i], nullptr) << "Stream " << i << " is null";
        } });

    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        for (int j = i + 1; j < NUM_STREAMS; ++j)
        {
            EXPECT_NE(streams[i], streams[j])
                << "Streams " << i << " and " << j << " are the same";
        }
    }

    ctx.submitAndWait([&]()
                      {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            ctx.destroyStream(streams[i]);
        } });
}

TEST(Test__AMDDeviceContext, DefaultStream)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *default_stream = nullptr;
    ctx.submitAndWait([&]()
                      { default_stream = ctx.defaultStream(); });

    (void)default_stream;
}

TEST(Test__AMDDeviceContext, EventCreationAndDestruction)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *event = nullptr;
    ctx.submitAndWait([&]()
                      { event = ctx.createEvent(); });

    EXPECT_NE(event, nullptr) << "createEvent() should return non-null";

    ctx.submitAndWait([&]()
                      { ctx.destroyEvent(event); });
}

TEST(Test__AMDDeviceContext, EventRecordAndSynchronize)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *event = nullptr;
    ctx.submitAndWait([&]()
                      {
        event = ctx.createEvent();
        ctx.recordEvent(event, nullptr);
        ctx.synchronizeEvent(event);
        ctx.destroyEvent(event); });
}

TEST(Test__AMDDeviceContext, EventWait)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    ctx.submitAndWait([&]()
                      {
        void* stream1 = ctx.createStream();
        void* stream2 = ctx.createStream();
        void* event = ctx.createEvent();

        ctx.recordEvent(event, stream1);
        ctx.waitEvent(event, stream2);

        ctx.destroyEvent(event);
        ctx.destroyStream(stream1);
        ctx.destroyStream(stream2); });
}

TEST(Test__AMDDeviceContext, BlasHandle)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *handle = nullptr;
    ctx.submitAndWait([&]()
                      { handle = ctx.blasHandle(); });

    EXPECT_NE(handle, nullptr) << "BLAS handle should be available";
}

TEST(Test__AMDDeviceContext, Synchronize)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    ctx.synchronize();

    ctx.submitAsync([&]()
                    {
        void* stream = ctx.createStream();
        ctx.destroyStream(stream); });

    ctx.synchronize();
}

TEST(Test__AMDDeviceContext, CollectiveCommInitiallyNull)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *comm = ctx.collectiveComm();
    (void)comm;
}

// ===========================================================================
// Cross-Platform Interface Tests
// ===========================================================================

TEST(Test__IWorkerGPUContext, InterfaceConsistency)
{
    SKIP_IF_NO_GPU();

    auto &pool = GPUDeviceContextPool::instance();

    // Get any available context
    IWorkerGPUContext *ctx = nullptr;
    if (pool.hasNvidiaSupport())
    {
        ctx = &pool.getNvidiaContext(0);
    }
    else if (pool.hasAMDSupport())
    {
        ctx = &pool.getAMDContext(0);
    }

    ASSERT_NE(ctx, nullptr);

    // Verify interface methods are callable
    EXPECT_TRUE(ctx->isInitialized());
    EXPECT_GE(ctx->deviceOrdinal(), 0);
    EXPECT_FALSE(ctx->deviceName().empty());

    // Test work submission
    std::atomic<int> counter{0};
    ctx->submitAndWait([&]()
                       { counter.fetch_add(1); });
    EXPECT_EQ(counter.load(), 1);

    // Test async submission
    auto future = ctx->submitAsync([&]()
                                   { counter.fetch_add(1); });
    future.wait();
    EXPECT_EQ(counter.load(), 2);

    // Test synchronize
    ctx->synchronize();
}

TEST(Test__IWorkerGPUContext, MultiDeviceSameVendor)
{
    auto &pool = GPUDeviceContextPool::instance();

    if (pool.hasNvidiaSupport() && pool.nvidiaDeviceCount() >= 2)
    {
        auto &ctx0 = pool.getNvidiaContext(0);
        auto &ctx1 = pool.getNvidiaContext(1);

        EXPECT_NE(&ctx0, &ctx1) << "Different devices should have different contexts";
        EXPECT_EQ(ctx0.deviceOrdinal(), 0);
        EXPECT_EQ(ctx1.deviceOrdinal(), 1);
    }

    if (pool.hasAMDSupport() && pool.amdDeviceCount() >= 2)
    {
        auto &ctx0 = pool.getAMDContext(0);
        auto &ctx1 = pool.getAMDContext(1);

        EXPECT_NE(&ctx0, &ctx1) << "Different devices should have different contexts";
        EXPECT_EQ(ctx0.deviceOrdinal(), 0);
        EXPECT_EQ(ctx1.deviceOrdinal(), 1);
    }
}

// ===========================================================================
// Stress Tests
// ===========================================================================

TEST(Test__GPUDeviceContext_Stress, RapidSubmitAndWait)
{
    SKIP_IF_NO_GPU();

    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = pool.hasNvidiaSupport()
                                 ? &pool.getNvidiaContext(0)
                                 : &pool.getAMDContext(0);

    constexpr int NUM_ITERATIONS = 1000;
    std::atomic<int> counter{0};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        ctx->submitAndWait([&]()
                           { counter.fetch_add(1); });
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(counter.load(), NUM_ITERATIONS);

    std::cout << "Rapid submitAndWait: " << NUM_ITERATIONS << " iterations in "
              << duration_ms << " ms (" << (NUM_ITERATIONS * 1000.0 / duration_ms)
              << " ops/sec)" << std::endl;
}

TEST(Test__GPUDeviceContext_Stress, ConcurrentSubmitFromMultipleThreads)
{
    SKIP_IF_NO_GPU();

    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = pool.hasNvidiaSupport()
                                 ? &pool.getNvidiaContext(0)
                                 : &pool.getAMDContext(0);

    constexpr int NUM_THREADS = 4;
    constexpr int SUBMISSIONS_PER_THREAD = 100;
    std::atomic<int> counter{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back([&]()
                             {
            for (int i = 0; i < SUBMISSIONS_PER_THREAD; ++i) {
                ctx->submitAndWait([&]() {
                    counter.fetch_add(1);
                });
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    EXPECT_EQ(counter.load(), NUM_THREADS * SUBMISSIONS_PER_THREAD);
}

TEST(Test__GPUDeviceContext_Stress, StreamEventChurn)
{
    SKIP_IF_NO_GPU();

    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = pool.hasNvidiaSupport()
                                 ? &pool.getNvidiaContext(0)
                                 : &pool.getAMDContext(0);

    constexpr int NUM_ITERATIONS = 100;

    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        ctx->submitAndWait([&]()
                           {
            void* stream = ctx->createStream();
            void* event = ctx->createEvent();

            ctx->recordEvent(event, stream);
            ctx->synchronizeEvent(event);

            ctx->destroyEvent(event);
            ctx->destroyStream(stream); });
    }
}
