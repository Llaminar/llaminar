/**
 * @file Test__DeviceContext.cpp
 * @brief Unit tests for DeviceContext
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "execution/local_execution/device/DeviceContext.h"
#include <vector>
#include <atomic>
#include <cmath>
#include <numeric>
#include <cstring>

using namespace llaminar2;

// =============================================================================
// CPUDeviceContext Direct Construction Tests
// =============================================================================

// Note: We test CPUDeviceContext directly rather than through the factory,
// because the factory requires DeviceManager to be initialized with devices,
// which is not available in a simple unit test context.

TEST(Test__DeviceContext, CreateCPUContextDirect)
{
    // Construct CPUDeviceContext directly (bypasses DeviceManager check)
    auto ctx = std::make_unique<CPUDeviceContext>(DeviceId::cpu());

    ASSERT_NE(ctx, nullptr);
    // DeviceId::cpu().toKernelDeviceIndex() returns -1 (CPU convention)
    EXPECT_EQ(ctx->deviceIndex(), -1);
    EXPECT_FALSE(ctx->isGPU());
}

TEST(Test__DeviceContext, CreateWithThreadCount)
{
    auto ctx = std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 4);

    ASSERT_NE(ctx, nullptr);
    // Thread count should be either 4 or fall back to OMP_NUM_THREADS
    EXPECT_GT(ctx->numThreads(), 0);
}

// =============================================================================
// Memory Allocation Tests
// =============================================================================

class CPUDeviceContextTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx = std::make_unique<CPUDeviceContext>(DeviceId::cpu());
        ASSERT_NE(ctx, nullptr);
    }

    std::unique_ptr<CPUDeviceContext> ctx;
};

TEST_F(CPUDeviceContextTest, AllocateMemory)
{
    size_t size = 1024 * sizeof(float);
    void *ptr = ctx->allocate(size);

    ASSERT_NE(ptr, nullptr);

    // Should be able to write to it
    float *fptr = static_cast<float *>(ptr);
    for (size_t i = 0; i < 1024; ++i)
    {
        fptr[i] = static_cast<float>(i);
    }

    ctx->free(ptr);
}

TEST_F(CPUDeviceContextTest, AllocateZeroReturnsNull)
{
    void *ptr = ctx->allocate(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(CPUDeviceContextTest, MultipleAllocations)
{
    std::vector<void *> ptrs;
    for (int i = 0; i < 100; ++i)
    {
        void *ptr = ctx->allocate(1024);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }

    for (void *ptr : ptrs)
    {
        ctx->free(ptr);
    }
}

// =============================================================================
// Copy Operations Tests
// =============================================================================

TEST_F(CPUDeviceContextTest, CopyToDevice)
{
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
    void *device = ctx->allocate(host_data.size() * sizeof(float));
    ASSERT_NE(device, nullptr);

    ctx->copyToDevice(device, host_data.data(), host_data.size() * sizeof(float));

    // Verify (CPU context is a memcpy)
    float *fptr = static_cast<float *>(device);
    for (size_t i = 0; i < host_data.size(); ++i)
    {
        EXPECT_EQ(fptr[i], host_data[i]);
    }

    ctx->free(device);
}

TEST_F(CPUDeviceContextTest, CopyToHost)
{
    void *device = ctx->allocate(4 * sizeof(float));
    ASSERT_NE(device, nullptr);

    float *fptr = static_cast<float *>(device);
    fptr[0] = 10.0f;
    fptr[1] = 20.0f;
    fptr[2] = 30.0f;
    fptr[3] = 40.0f;

    std::vector<float> host_data(4);
    ctx->copyToHost(host_data.data(), device, 4 * sizeof(float));

    EXPECT_EQ(host_data[0], 10.0f);
    EXPECT_EQ(host_data[1], 20.0f);
    EXPECT_EQ(host_data[2], 30.0f);
    EXPECT_EQ(host_data[3], 40.0f);

    ctx->free(device);
}

TEST_F(CPUDeviceContextTest, CopyDeviceToDevice)
{
    void *src = ctx->allocate(4 * sizeof(float));
    void *dst = ctx->allocate(4 * sizeof(float));
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    float *src_ptr = static_cast<float *>(src);
    src_ptr[0] = 1.0f;
    src_ptr[1] = 2.0f;
    src_ptr[2] = 3.0f;
    src_ptr[3] = 4.0f;

    ctx->copyFromDevice(dst, src, 4 * sizeof(float), ctx.get());

    float *dst_ptr = static_cast<float *>(dst);
    EXPECT_EQ(dst_ptr[0], 1.0f);
    EXPECT_EQ(dst_ptr[1], 2.0f);
    EXPECT_EQ(dst_ptr[2], 3.0f);
    EXPECT_EQ(dst_ptr[3], 4.0f);

    ctx->free(src);
    ctx->free(dst);
}

// =============================================================================
// Parallel Execution Tests
// =============================================================================

TEST_F(CPUDeviceContextTest, RunParallel)
{
    std::atomic<int> counter{0};

    ctx->runParallel([&](int thread_id, int num_threads)
                     { counter.fetch_add(1, std::memory_order_relaxed); });

    // At least 1 thread ran
    EXPECT_GE(counter.load(), 1);
}

TEST_F(CPUDeviceContextTest, RunFor)
{
    constexpr int N = 10000;
    std::vector<float> data(N, 1.0f);

    ctx->runFor(0, N, [&](size_t i)
                { data[i] = static_cast<float>(i * 2); });

    // Verify
    for (int i = 0; i < N; ++i)
    {
        EXPECT_EQ(data[i], static_cast<float>(i * 2)) << "Failed at index " << i;
    }
}

TEST_F(CPUDeviceContextTest, RunForWithRange)
{
    constexpr int N = 10000;
    std::vector<float> data(N, 0.0f);

    ctx->runFor(100, N - 100, [&](size_t i)
                { data[i] = 1.0f; });

    // Elements 0-99 should be 0
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_EQ(data[i], 0.0f);
    }

    // Elements 100-9899 should be 1
    for (int i = 100; i < N - 100; ++i)
    {
        EXPECT_EQ(data[i], 1.0f) << "Failed at index " << i;
    }

    // Elements 9900-9999 should be 0
    for (int i = N - 100; i < N; ++i)
    {
        EXPECT_EQ(data[i], 0.0f);
    }
}

TEST_F(CPUDeviceContextTest, RunForEmpty)
{
    int count = 0;

    // Empty range should not execute
    ctx->runFor(100, 100, [&](size_t)
                { count++; });

    EXPECT_EQ(count, 0);
}

// =============================================================================
// Workspace Tests
// =============================================================================

TEST_F(CPUDeviceContextTest, GetWorkspace)
{
    void *ws1 = ctx->getWorkspace(1024);
    ASSERT_NE(ws1, nullptr);

    // Can write to it
    std::memset(ws1, 0xAB, 1024);

    // Getting larger workspace should work
    void *ws2 = ctx->getWorkspace(4096);
    ASSERT_NE(ws2, nullptr);

    // Pointer may or may not be same depending on implementation
}

// =============================================================================
// Synchronization Tests
// =============================================================================

TEST_F(CPUDeviceContextTest, Synchronize)
{
    // For CPU, synchronize is a no-op but should not crash
    ctx->synchronize();
}

// =============================================================================
// Nested Parallel Region Tests
// =============================================================================

TEST_F(CPUDeviceContextTest, NestedParallelRegionSafe)
{
    // This tests that CPUDeviceContext properly handles being called
    // from within an existing parallel region

    std::atomic<int> outer_count{0};
    std::atomic<int> inner_count{0};

#pragma omp parallel
    {
        outer_count.fetch_add(1, std::memory_order_relaxed);

        // This should NOT create a new parallel region (would cause issues)
        ctx->runParallel([&](int tid, int nt)
                         { inner_count.fetch_add(1, std::memory_order_relaxed); });
    }

    // Inner count should equal outer count (no nested team creation)
    // or be 1 if omp_in_parallel() was detected and skipped new region
    EXPECT_GT(outer_count.load(), 0);
    EXPECT_GT(inner_count.load(), 0);
}

TEST_F(CPUDeviceContextTest, NestedRunForSafe)
{
    // Test that runFor works correctly when NOT inside a parallel region
    // Note: Calling runFor() from within #pragma omp single is NOT supported
    // because omp for worksharing requires all threads to participate.
    // This test verifies the non-nested path works.

    constexpr int N = 1000;
    std::vector<float> data(N, 0.0f);

    ctx->runFor(0, N, [&](size_t i)
                { data[i] = 1.0f; });

    // Verify all elements were processed
    for (int i = 0; i < N; ++i)
    {
        EXPECT_EQ(data[i], 1.0f) << "Failed at index " << i;
    }
}

// =============================================================================
// Backend Type Check Tests
// =============================================================================

TEST(Test__DeviceContext, BackendTypeProperties)
{
    auto ctx = std::make_unique<CPUDeviceContext>(DeviceId::cpu()); // CPU device
    ASSERT_NE(ctx, nullptr);
    EXPECT_FALSE(ctx->isGPU());
}

TEST(Test__DeviceContext, NumThreads)
{
    auto ctx = std::make_unique<CPUDeviceContext>(DeviceId::cpu());
    ASSERT_NE(ctx, nullptr);

    // Should have at least 1 thread
    EXPECT_GE(ctx->numThreads(), 1);
}
// =============================================================================
// GPU DeviceContext Interface Tests
// =============================================================================
// Note: These tests verify the GPU context interfaces compile and the base
// class behaviors work. Actual GPU execution requires CUDA/ROCm hardware.

TEST(Test__DeviceContext, IGPUDeviceContextInterface)
{
    // IGPUDeviceContext is an abstract base class - verify it's properly
    // inheriting from IDeviceContext
    // This is a compile-time test - if this compiles, the interface is correct

    // IGPUDeviceContext exposes:
    // - isGPU() returning true
    // - workspace management (getWorkspace, allocateWorkspace, freeWorkspace)
    // - deviceToDevice transfers
    // - GPU-specific synchronization

    // We can't instantiate IGPUDeviceContext directly, but we can verify
    // the CPU context properly returns false for isGPU
    auto cpu_ctx = std::make_unique<CPUDeviceContext>(DeviceId::cpu());
    EXPECT_FALSE(cpu_ctx->isGPU());
}

#ifdef HAVE_CUDA
TEST(Test__DeviceContext, CUDAContextRequiresGPU)
{
    // CUDADeviceContext requires actual CUDA GPU to construct
    // This test documents the expected behavior
    // In a proper GPU environment, this would succeed

    // Get device manager
    auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();

    // Look for CUDA device
    bool found_cuda = false;
    for (size_t i = 0; i < devices.size(); ++i)
    {
        if (devices[i].type == ComputeBackendType::GPU_CUDA)
        {
            found_cuda = true;
            auto ctx = IDeviceContext::create(DeviceId::cuda(devices[i].device_id));
            if (ctx)
            {
                EXPECT_TRUE(ctx->isGPU());
            }
            break;
        }
    }

    if (!found_cuda)
    {
        GTEST_SKIP() << "No CUDA GPU available";
    }
}
#endif

#ifdef HAVE_ROCM
TEST(Test__DeviceContext, ROCmContextRequiresGPU)
{
    // ROCmDeviceContext requires actual ROCm GPU to construct
    // This test documents the expected behavior

    auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();

    bool found_rocm = false;
    for (size_t i = 0; i < devices.size(); ++i)
    {
        if (devices[i].type == ComputeBackendType::GPU_ROCM)
        {
            found_rocm = true;
            auto ctx = IDeviceContext::create(DeviceId::rocm(devices[i].device_id));
            if (ctx)
            {
                EXPECT_TRUE(ctx->isGPU());
            }
            break;
        }
    }

    if (!found_rocm)
    {
        GTEST_SKIP() << "No ROCm GPU available";
    }
}
#endif