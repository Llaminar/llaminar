/**
 * @file Test__RCCLCoordinator.cpp
 * @brief Integration tests for RCCLCoordinator with AMD GPUs
 * @author GitHub Copilot
 * @date February 2026
 *
 * Tests RCCLCoordinator with actual AMD ROCm GPU hardware:
 * - Single-GPU initialization and operations
 * - Multi-GPU allreduce, allgather, broadcast
 * - Thread safety (concurrent submissions)
 * - Shutdown idempotency
 *
 * NOTE: These tests require AMD GPU(s) with RCCL support.
 * Tests will be skipped if no ROCm devices are available.
 *
 * @see RCCLCoordinator for the implementation
 * @see docs/v2/projects/2026-02/CollectiveCoordinators.md for design documentation
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM

#include "collective/coordinators/RCCLCoordinator.h"
#include "collective/ICollectiveBackend.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__RCCLCoordinator : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Detect available ROCm devices
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess || device_count == 0)
        {
            rocm_device_count_ = 0;
        }
        else
        {
            rocm_device_count_ = device_count;
        }

        if (rocm_device_count_ > 0)
        {
            // Log device info
            for (int i = 0; i < rocm_device_count_; ++i)
            {
                hipDeviceProp_t prop;
                hipGetDeviceProperties(&prop, i);
                std::cout << "  ROCm GPU " << i << ": " << prop.name << std::endl;
            }
        }
    }

    void TearDown() override
    {
        // Synchronize all devices
        for (int i = 0; i < rocm_device_count_; ++i)
        {
            hipSetDevice(i);
            hipDeviceSynchronize();
        }
    }

    /**
     * @brief Allocate device buffer on a specific GPU
     */
    void *allocateDeviceBuffer(int device_id, size_t bytes)
    {
        hipSetDevice(device_id);
        void *ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, bytes);
        if (err != hipSuccess)
        {
            return nullptr;
        }
        return ptr;
    }

    /**
     * @brief Free device buffer
     */
    void freeDeviceBuffer(int device_id, void *ptr)
    {
        hipSetDevice(device_id);
        hipFree(ptr);
    }

    /**
     * @brief Copy host data to device
     */
    void copyHostToDevice(int device_id, void *dst, const void *src, size_t bytes)
    {
        hipSetDevice(device_id);
        hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
        hipDeviceSynchronize();
    }

    /**
     * @brief Copy device data to host
     */
    void copyDeviceToHost(int device_id, void *dst, const void *src, size_t bytes)
    {
        hipSetDevice(device_id);
        hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
    }

    /**
     * @brief Initialize device buffer with a value
     */
    void fillDeviceBuffer(int device_id, float *buffer, size_t count, float value)
    {
        std::vector<float> host_data(count, value);
        copyHostToDevice(device_id, buffer, host_data.data(), count * sizeof(float));
    }

    int rocm_device_count_ = 0;
};

// =============================================================================
// Initialization Tests
// =============================================================================

TEST_F(Test__RCCLCoordinator, InitializeSingleGPU)
{
    if (rocm_device_count_ < 1)
    {
        GTEST_SKIP() << "No ROCm devices available";
    }

    RCCLCoordinator coord;

    // Initially not initialized
    EXPECT_FALSE(coord.isInitialized());

    // Initialize with single device
    ASSERT_TRUE(coord.initialize({0})) << "Failed to initialize with GPU 0: " << coord.lastError();

    // Now should be initialized
    EXPECT_TRUE(coord.isInitialized());
    EXPECT_EQ(coord.numDevices(), 1);
    EXPECT_EQ(coord.deviceOrdinal(0), 0);

    // Clean shutdown
    coord.shutdown();
    EXPECT_FALSE(coord.isInitialized());
}

TEST_F(Test__RCCLCoordinator, InitializeMultiGPU)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices for multi-GPU test";
    }

    RCCLCoordinator coord;

    // Initialize with all available devices (up to 2)
    std::vector<int> devices;
    for (int i = 0; i < std::min(rocm_device_count_, 2); ++i)
    {
        devices.push_back(i);
    }

    ASSERT_TRUE(coord.initialize(devices)) << "Failed to initialize: " << coord.lastError();

    EXPECT_TRUE(coord.isInitialized());
    EXPECT_EQ(coord.numDevices(), static_cast<int>(devices.size()));

    for (size_t i = 0; i < devices.size(); ++i)
    {
        EXPECT_EQ(coord.deviceOrdinal(static_cast<int>(i)), devices[i]);
    }

    coord.shutdown();
}

TEST_F(Test__RCCLCoordinator, InitializeNonContiguousDevices)
{
    if (rocm_device_count_ < 3)
    {
        GTEST_SKIP() << "Need at least 3 ROCm devices for non-contiguous test";
    }

    RCCLCoordinator coord;

    // Initialize with devices 0 and 2 (skip 1)
    std::vector<int> devices = {0, 2};

    ASSERT_TRUE(coord.initialize(devices)) << "Failed to initialize: " << coord.lastError();

    EXPECT_TRUE(coord.isInitialized());
    EXPECT_EQ(coord.numDevices(), 2);
    EXPECT_EQ(coord.deviceOrdinal(0), 0);
    EXPECT_EQ(coord.deviceOrdinal(1), 2);

    coord.shutdown();
}

// =============================================================================
// Allreduce Tests
// =============================================================================

TEST_F(Test__RCCLCoordinator, AllreduceSingleGPU)
{
    if (rocm_device_count_ < 1)
    {
        GTEST_SKIP() << "No ROCm devices available";
    }

    RCCLCoordinator coord;
    ASSERT_TRUE(coord.initialize({0})) << "Failed to initialize: " << coord.lastError();

    constexpr size_t COUNT = 1024;

    // Allocate device buffer
    float *d_buffer = static_cast<float *>(allocateDeviceBuffer(0, COUNT * sizeof(float)));
    ASSERT_NE(d_buffer, nullptr);

    // Fill with known values (1.0)
    fillDeviceBuffer(0, d_buffer, COUNT, 1.0f);

    // Allreduce (single device = no-op for sum, should keep same values)
    std::vector<void *> buffers = {d_buffer};
    ASSERT_TRUE(coord.allreduceMulti(buffers, COUNT, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

    // Synchronize
    ASSERT_TRUE(coord.synchronize());

    // Verify results (should still be 1.0)
    std::vector<float> host_result(COUNT);
    copyDeviceToHost(0, host_result.data(), d_buffer, COUNT * sizeof(float));

    for (size_t i = 0; i < COUNT; ++i)
    {
        EXPECT_FLOAT_EQ(host_result[i], 1.0f) << "Mismatch at index " << i;
    }

    // Cleanup
    freeDeviceBuffer(0, d_buffer);
    coord.shutdown();
}

TEST_F(Test__RCCLCoordinator, AllreduceMultiGPU)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices for multi-GPU allreduce";
    }

    RCCLCoordinator coord;
    ASSERT_TRUE(coord.initialize({0, 1})) << "Failed to initialize: " << coord.lastError();

    constexpr size_t COUNT = 1024;

    // Allocate buffers on each device
    float *d_buffer_0 = static_cast<float *>(allocateDeviceBuffer(0, COUNT * sizeof(float)));
    float *d_buffer_1 = static_cast<float *>(allocateDeviceBuffer(1, COUNT * sizeof(float)));
    ASSERT_NE(d_buffer_0, nullptr);
    ASSERT_NE(d_buffer_1, nullptr);

    // Fill with different values: GPU 0 = 1.0, GPU 1 = 2.0
    fillDeviceBuffer(0, d_buffer_0, COUNT, 1.0f);
    fillDeviceBuffer(1, d_buffer_1, COUNT, 2.0f);

    // Allreduce sum: result should be 1.0 + 2.0 = 3.0 on both devices
    std::vector<void *> buffers = {d_buffer_0, d_buffer_1};
    ASSERT_TRUE(coord.allreduceMulti(buffers, COUNT, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

    // Synchronize
    ASSERT_TRUE(coord.synchronize());

    // Verify results on both devices
    std::vector<float> host_result_0(COUNT);
    std::vector<float> host_result_1(COUNT);
    copyDeviceToHost(0, host_result_0.data(), d_buffer_0, COUNT * sizeof(float));
    copyDeviceToHost(1, host_result_1.data(), d_buffer_1, COUNT * sizeof(float));

    for (size_t i = 0; i < COUNT; ++i)
    {
        EXPECT_FLOAT_EQ(host_result_0[i], 3.0f) << "GPU 0 mismatch at index " << i;
        EXPECT_FLOAT_EQ(host_result_1[i], 3.0f) << "GPU 1 mismatch at index " << i;
    }

    // Cleanup
    freeDeviceBuffer(0, d_buffer_0);
    freeDeviceBuffer(1, d_buffer_1);
    coord.shutdown();
}

// =============================================================================
// Allgather Tests
// =============================================================================

TEST_F(Test__RCCLCoordinator, AllgatherMultiGPU)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices for allgather";
    }

    RCCLCoordinator coord;
    ASSERT_TRUE(coord.initialize({0, 1})) << "Failed to initialize: " << coord.lastError();

    constexpr size_t SEND_COUNT = 512;
    constexpr size_t RECV_COUNT = SEND_COUNT * 2; // 2 devices

    // Allocate send buffers (one per device)
    float *d_send_0 = static_cast<float *>(allocateDeviceBuffer(0, SEND_COUNT * sizeof(float)));
    float *d_send_1 = static_cast<float *>(allocateDeviceBuffer(1, SEND_COUNT * sizeof(float)));
    ASSERT_NE(d_send_0, nullptr);
    ASSERT_NE(d_send_1, nullptr);

    // Allocate recv buffers (must hold all gathered data)
    float *d_recv_0 = static_cast<float *>(allocateDeviceBuffer(0, RECV_COUNT * sizeof(float)));
    float *d_recv_1 = static_cast<float *>(allocateDeviceBuffer(1, RECV_COUNT * sizeof(float)));
    ASSERT_NE(d_recv_0, nullptr);
    ASSERT_NE(d_recv_1, nullptr);

    // Fill send buffers with distinct values: GPU 0 = 1.0, GPU 1 = 2.0
    fillDeviceBuffer(0, d_send_0, SEND_COUNT, 1.0f);
    fillDeviceBuffer(1, d_send_1, SEND_COUNT, 2.0f);

    // Zero recv buffers
    hipSetDevice(0);
    hipMemset(d_recv_0, 0, RECV_COUNT * sizeof(float));
    hipSetDevice(1);
    hipMemset(d_recv_1, 0, RECV_COUNT * sizeof(float));

    // Allgather
    std::vector<const void *> send_buffers = {d_send_0, d_send_1};
    std::vector<void *> recv_buffers = {d_recv_0, d_recv_1};
    ASSERT_TRUE(coord.allgatherMulti(send_buffers, recv_buffers, SEND_COUNT, CollectiveDataType::FLOAT32));

    // Synchronize
    ASSERT_TRUE(coord.synchronize());

    // Verify results - each recv buffer should have [1.0 x SEND_COUNT, 2.0 x SEND_COUNT]
    std::vector<float> host_recv_0(RECV_COUNT);
    std::vector<float> host_recv_1(RECV_COUNT);
    copyDeviceToHost(0, host_recv_0.data(), d_recv_0, RECV_COUNT * sizeof(float));
    copyDeviceToHost(1, host_recv_1.data(), d_recv_1, RECV_COUNT * sizeof(float));

    // First half should be from GPU 0 (1.0), second half from GPU 1 (2.0)
    for (size_t i = 0; i < SEND_COUNT; ++i)
    {
        EXPECT_FLOAT_EQ(host_recv_0[i], 1.0f) << "GPU 0 recv first half mismatch at " << i;
        EXPECT_FLOAT_EQ(host_recv_0[i + SEND_COUNT], 2.0f) << "GPU 0 recv second half mismatch at " << i;
        EXPECT_FLOAT_EQ(host_recv_1[i], 1.0f) << "GPU 1 recv first half mismatch at " << i;
        EXPECT_FLOAT_EQ(host_recv_1[i + SEND_COUNT], 2.0f) << "GPU 1 recv second half mismatch at " << i;
    }

    // Cleanup
    freeDeviceBuffer(0, d_send_0);
    freeDeviceBuffer(1, d_send_1);
    freeDeviceBuffer(0, d_recv_0);
    freeDeviceBuffer(1, d_recv_1);
    coord.shutdown();
}

// =============================================================================
// Broadcast Tests
// =============================================================================

TEST_F(Test__RCCLCoordinator, BroadcastMultiGPU)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices for broadcast";
    }

    RCCLCoordinator coord;
    ASSERT_TRUE(coord.initialize({0, 1})) << "Failed to initialize: " << coord.lastError();

    constexpr size_t COUNT = 1024;

    // Allocate buffers
    float *d_buffer_0 = static_cast<float *>(allocateDeviceBuffer(0, COUNT * sizeof(float)));
    float *d_buffer_1 = static_cast<float *>(allocateDeviceBuffer(1, COUNT * sizeof(float)));
    ASSERT_NE(d_buffer_0, nullptr);
    ASSERT_NE(d_buffer_1, nullptr);

    // Fill root (GPU 0) with 42.0, GPU 1 with zeros
    fillDeviceBuffer(0, d_buffer_0, COUNT, 42.0f);
    hipSetDevice(1);
    hipMemset(d_buffer_1, 0, COUNT * sizeof(float));

    // Broadcast from root=0
    std::vector<void *> buffers = {d_buffer_0, d_buffer_1};
    ASSERT_TRUE(coord.broadcastMulti(buffers, COUNT, CollectiveDataType::FLOAT32, 0));

    // Synchronize
    ASSERT_TRUE(coord.synchronize());

    // Verify results - both devices should have 42.0
    std::vector<float> host_result_0(COUNT);
    std::vector<float> host_result_1(COUNT);
    copyDeviceToHost(0, host_result_0.data(), d_buffer_0, COUNT * sizeof(float));
    copyDeviceToHost(1, host_result_1.data(), d_buffer_1, COUNT * sizeof(float));

    for (size_t i = 0; i < COUNT; ++i)
    {
        EXPECT_FLOAT_EQ(host_result_0[i], 42.0f) << "GPU 0 mismatch at index " << i;
        EXPECT_FLOAT_EQ(host_result_1[i], 42.0f) << "GPU 1 mismatch at index " << i;
    }

    // Cleanup
    freeDeviceBuffer(0, d_buffer_0);
    freeDeviceBuffer(1, d_buffer_1);
    coord.shutdown();
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST_F(Test__RCCLCoordinator, ThreadSafety)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices for thread safety test";
    }

    RCCLCoordinator coord;
    ASSERT_TRUE(coord.initialize({0, 1})) << "Failed to initialize: " << coord.lastError();

    constexpr size_t COUNT = 1024;
    constexpr int NUM_ITERATIONS = 10;
    constexpr int NUM_THREADS = 4;

    // Allocate persistent buffers
    float *d_buffer_0 = static_cast<float *>(allocateDeviceBuffer(0, COUNT * sizeof(float)));
    float *d_buffer_1 = static_cast<float *>(allocateDeviceBuffer(1, COUNT * sizeof(float)));
    ASSERT_NE(d_buffer_0, nullptr);
    ASSERT_NE(d_buffer_1, nullptr);

    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    auto worker = [&](int thread_id)
    {
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            // Each thread does an allreduce
            // Initialize buffers
            fillDeviceBuffer(0, d_buffer_0, COUNT, static_cast<float>(thread_id + 1));
            fillDeviceBuffer(1, d_buffer_1, COUNT, static_cast<float>(thread_id + 1));

            std::vector<void *> buffers = {d_buffer_0, d_buffer_1};
            if (coord.allreduceMulti(buffers, COUNT, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM))
            {
                success_count++;
            }
            else
            {
                failure_count++;
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for (int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back(worker, t);
    }

    // Wait for completion
    for (auto &t : threads)
    {
        t.join();
    }

    // All operations should succeed (coordinator serializes them)
    EXPECT_EQ(failure_count.load(), 0) << "Some allreduce operations failed";
    EXPECT_EQ(success_count.load(), NUM_THREADS * NUM_ITERATIONS);

    // Cleanup
    freeDeviceBuffer(0, d_buffer_0);
    freeDeviceBuffer(1, d_buffer_1);
    coord.shutdown();
}

// =============================================================================
// Shutdown Tests
// =============================================================================

TEST_F(Test__RCCLCoordinator, ShutdownIdempotent)
{
    if (rocm_device_count_ < 1)
    {
        GTEST_SKIP() << "No ROCm devices available";
    }

    RCCLCoordinator coord;
    ASSERT_TRUE(coord.initialize({0})) << "Failed to initialize: " << coord.lastError();
    EXPECT_TRUE(coord.isInitialized());

    // First shutdown
    coord.shutdown();
    EXPECT_FALSE(coord.isInitialized());

    // Second shutdown should be safe (no crash, no-op)
    coord.shutdown();
    EXPECT_FALSE(coord.isInitialized());

    // Third shutdown
    coord.shutdown();
    EXPECT_FALSE(coord.isInitialized());
}

TEST_F(Test__RCCLCoordinator, ShutdownWithoutInitialize)
{
    // Shutdown without initialize should be safe
    RCCLCoordinator coord;
    EXPECT_FALSE(coord.isInitialized());

    // Should not crash
    coord.shutdown();
    EXPECT_FALSE(coord.isInitialized());
}

// =============================================================================
// Event/Synchronization Tests
// =============================================================================

TEST_F(Test__RCCLCoordinator, GetCompletionEvent)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices";
    }

    RCCLCoordinator coord;
    ASSERT_TRUE(coord.initialize({0, 1})) << "Failed to initialize: " << coord.lastError();

    // Get completion events for each device
    void *event_0 = coord.getCompletionEvent(0);
    void *event_1 = coord.getCompletionEvent(1);

    // Events should be non-null
    EXPECT_NE(event_0, nullptr) << "Completion event for device 0 is null";
    EXPECT_NE(event_1, nullptr) << "Completion event for device 1 is null";

    // Events should be different
    EXPECT_NE(event_0, event_1) << "Completion events should be distinct per device";

    coord.shutdown();
}

#endif // HAVE_ROCM
