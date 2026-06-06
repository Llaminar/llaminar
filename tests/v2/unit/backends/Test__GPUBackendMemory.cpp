/**
 * @file Test__GPUBackendMemory.cpp
 * @brief Test GPU backend memory allocation and transfer operations
 *
 * **Purpose**: Validate IBackend memory allocation/deallocation and host↔device transfers.
 *
 * **Tests**:
 * - allocate(): GPU memory allocation
 * - free(): GPU memory deallocation
 * - hostToDevice(): Upload data from host to GPU
 * - deviceToHost(): Download data from GPU to host
 * - synchronize(): Device synchronization
 *
 * **Requirements**: Run with `-DHAVE_CUDA=ON` or `-DHAVE_ROCM=ON`
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <cstdint>

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif

#ifdef HAVE_ROCM
#include "backends/rocm/ROCmBackend.h"
#endif

using namespace llaminar2;

// ============================================================================
// CUDA Backend Memory Tests
// ============================================================================

#ifdef HAVE_CUDA

class Test__GPUBackendMemory_CUDA : public ::testing::Test
{
protected:
    void SetUp() override
    {
        backend = std::make_unique<CUDABackend>();
        device_count = backend->deviceCount();
    }

    void TearDown() override
    {
        backend.reset();
    }

    std::unique_ptr<CUDABackend> backend;
    int device_count = 0;
};

TEST_F(Test__GPUBackendMemory_CUDA, DeviceCount)
{
    // Should have at least 1 CUDA device in CI/dev environment
    EXPECT_GE(device_count, 0) << "CUDA backend should report device count";

    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available, skipping CUDA tests";
    }

    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
}

TEST_F(Test__GPUBackendMemory_CUDA, AllocateAndFree)
{
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int device_id = 0;
    const size_t bytes = 1024 * 1024; // 1 MB

    // Allocate GPU memory
    void *device_ptr = backend->allocate(bytes, device_id);
    ASSERT_NE(device_ptr, nullptr) << "Failed to allocate 1 MB on CUDA device 0";
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(device_ptr) & 0xffu, 0u)
        << "CUDA backend allocations must be 256-byte aligned";

    // Free GPU memory
    backend->free(device_ptr, device_id);
    // If this doesn't crash, free() worked
}

TEST_F(Test__GPUBackendMemory_CUDA, HostToDeviceTransfer)
{
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int device_id = 0;
    const size_t count = 1024;
    const size_t bytes = count * sizeof(float);

    // Create host data
    std::vector<float> host_data(count);
    for (size_t i = 0; i < count; ++i)
    {
        host_data[i] = static_cast<float>(i);
    }

    // Allocate GPU memory
    void *device_ptr = backend->allocate(bytes, device_id);
    ASSERT_NE(device_ptr, nullptr) << "Failed to allocate GPU memory";

    // Transfer host → device
    bool success = backend->hostToDevice(device_ptr, host_data.data(), bytes, device_id);
    EXPECT_TRUE(success) << "hostToDevice failed";

    // Synchronize
    success = backend->synchronize(device_id);
    EXPECT_TRUE(success) << "synchronize failed";

    // Cleanup
    backend->free(device_ptr, device_id);
}

TEST_F(Test__GPUBackendMemory_CUDA, DeviceToHostTransfer)
{
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int device_id = 0;
    const size_t count = 1024;
    const size_t bytes = count * sizeof(float);

    // Create host data (source)
    std::vector<float> host_src(count);
    for (size_t i = 0; i < count; ++i)
    {
        host_src[i] = static_cast<float>(i * 2);
    }

    // Allocate host destination (zero-initialized)
    std::vector<float> host_dst(count, 0.0f);

    // Allocate GPU memory
    void *device_ptr = backend->allocate(bytes, device_id);
    ASSERT_NE(device_ptr, nullptr) << "Failed to allocate GPU memory";

    // Transfer host → device
    bool success = backend->hostToDevice(device_ptr, host_src.data(), bytes, device_id);
    ASSERT_TRUE(success) << "hostToDevice failed";

    // Transfer device → host
    success = backend->deviceToHost(host_dst.data(), device_ptr, bytes, device_id);
    ASSERT_TRUE(success) << "deviceToHost failed";

    // Synchronize
    success = backend->synchronize(device_id);
    ASSERT_TRUE(success) << "synchronize failed";

    // Verify data correctness
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(host_dst[i], host_src[i])
            << "Mismatch at index " << i << ": expected " << host_src[i]
            << ", got " << host_dst[i];
    }

    // Cleanup
    backend->free(device_ptr, device_id);
}

TEST_F(Test__GPUBackendMemory_CUDA, RoundTripTransfer)
{
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int device_id = 0;
    const size_t count = 4096;
    const size_t bytes = count * sizeof(float);

    // Create host data with specific pattern
    std::vector<float> original(count);
    for (size_t i = 0; i < count; ++i)
    {
        original[i] = static_cast<float>(i) * 3.14159f;
    }

    std::vector<float> result(count, 0.0f);

    // Allocate GPU memory
    void *device_ptr = backend->allocate(bytes, device_id);
    ASSERT_NE(device_ptr, nullptr);

    // Round trip: host → device → host
    ASSERT_TRUE(backend->hostToDevice(device_ptr, original.data(), bytes, device_id));
    ASSERT_TRUE(backend->deviceToHost(result.data(), device_ptr, bytes, device_id));
    ASSERT_TRUE(backend->synchronize(device_id));

    // Verify exact match
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i])
            << "Round-trip mismatch at index " << i;
    }

    backend->free(device_ptr, device_id);
}

TEST_F(Test__GPUBackendMemory_CUDA, LargeAllocation)
{
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int device_id = 0;
    const size_t bytes = 100 * 1024 * 1024; // 100 MB

    void *device_ptr = backend->allocate(bytes, device_id);

    // May fail if GPU has insufficient memory
    if (device_ptr != nullptr)
    {
        backend->free(device_ptr, device_id);
    }
    else
    {
        std::cout << "Large allocation (100 MB) failed - insufficient GPU memory" << std::endl;
    }
}

TEST_F(Test__GPUBackendMemory_CUDA, InvalidDeviceID)
{
    const int invalid_device_id = 999;
    const size_t bytes = 1024;

    void *device_ptr = backend->allocate(bytes, invalid_device_id);
    EXPECT_EQ(device_ptr, nullptr) << "Should fail with invalid device ID";
}

TEST_F(Test__GPUBackendMemory_CUDA, FreeNullPointer)
{
    if (device_count == 0)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    const int device_id = 0;

    // Freeing nullptr should be a no-op (no crash)
    backend->free(nullptr, device_id);
}

#endif // HAVE_CUDA

// ============================================================================
// ROCm Backend Memory Tests
// ============================================================================

#ifdef HAVE_ROCM

class Test__GPUBackendMemory_ROCm : public ::testing::Test
{
protected:
    void SetUp() override
    {
        backend = std::make_unique<ROCmBackend>();
        device_count = backend->deviceCount();
    }

    void TearDown() override
    {
        backend.reset();
    }

    std::unique_ptr<ROCmBackend> backend;
    int device_count = 0;
};

TEST_F(Test__GPUBackendMemory_ROCm, DeviceCount)
{
    EXPECT_GE(device_count, 0) << "ROCm backend should report device count";

    if (device_count == 0)
    {
        GTEST_SKIP() << "No ROCm devices available, skipping ROCm tests";
    }

    std::cout << "Found " << device_count << " ROCm device(s)" << std::endl;
}

TEST_F(Test__GPUBackendMemory_ROCm, AllocateAndFree)
{
    if (device_count == 0)
    {
        GTEST_SKIP() << "No ROCm devices available";
    }

    const int device_id = 0;
    const size_t bytes = 1024 * 1024; // 1 MB

    void *device_ptr = backend->allocate(bytes, device_id);
    ASSERT_NE(device_ptr, nullptr) << "Failed to allocate 1 MB on ROCm device 0";
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(device_ptr) & 0xffu, 0u)
        << "ROCm backend allocations must be 256-byte aligned";

    backend->free(device_ptr, device_id);
}

TEST_F(Test__GPUBackendMemory_ROCm, RoundTripTransfer)
{
    if (device_count == 0)
    {
        GTEST_SKIP() << "No ROCm devices available";
    }

    const int device_id = 0;
    const size_t count = 4096;
    const size_t bytes = count * sizeof(float);

    std::vector<float> original(count);
    for (size_t i = 0; i < count; ++i)
    {
        original[i] = static_cast<float>(i) * 2.71828f;
    }

    std::vector<float> result(count, 0.0f);

    void *device_ptr = backend->allocate(bytes, device_id);
    ASSERT_NE(device_ptr, nullptr);

    ASSERT_TRUE(backend->hostToDevice(device_ptr, original.data(), bytes, device_id));
    ASSERT_TRUE(backend->deviceToHost(result.data(), device_ptr, bytes, device_id));
    ASSERT_TRUE(backend->synchronize(device_id));

    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i])
            << "Round-trip mismatch at index " << i;
    }

    backend->free(device_ptr, device_id);
}

#endif // HAVE_ROCM

// ============================================================================
// Fallback Test (No GPU Available)
// ============================================================================

#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)

TEST(Test__GPUBackendMemory, NoGPUAvailable)
{
    GTEST_SKIP() << "No GPU backend compiled (HAVE_CUDA=OFF, HAVE_ROCM=OFF)";
}

#endif
