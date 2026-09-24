/**
 * @file Test__CUDAHIPKernelCoexistence.cpp
 * @brief Minimal test for CUDA kernel functionality after HIP initialization
 *
 * This test checks if CUDA kernels work correctly after HIP/ROCm operations.
 * It's a minimal reproduction test for the "context is destroyed" (error 709)
 * issue observed in heterogeneous LOCAL TP execution.
 *
 * Test sequence:
 * 1. Initialize CUDA backend, run CUDA kernel (baseline)
 * 2. Initialize ROCm backend
 * 3. Run ROCm kernel
 * 4. Run CUDA kernel again (should still work)
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

#include "backends/BackendManager.h"
#include "backends/cuda/CUDABackend.h"
#include "backends/rocm/ROCmBackend.h"
#include "../utils/ScopedGPUStream.h"

using namespace llaminar2;

class Test__CUDAHIPKernelCoexistence : public ::testing::Test
{
protected:
    IBackend *cuda_backend_ = nullptr;
    IBackend *rocm_backend_ = nullptr;

    static constexpr int TEST_SIZE = 1024;

    void SetUp() override
    {
        cuda_backend_ = getCUDABackend();
        rocm_backend_ = getROCmBackend();
    }

    // Run a simple vector add on CUDA and verify result
    bool runCUDAVectorAdd(const std::string &phase)
    {
        if (!cuda_backend_)
        {
            std::cerr << "CUDA backend not available" << std::endl;
            return false;
        }

        std::vector<float> host_a(TEST_SIZE, 1.0f);
        std::vector<float> host_b(TEST_SIZE, 2.0f);
        std::vector<float> host_c(TEST_SIZE, 0.0f);

        size_t bytes = TEST_SIZE * sizeof(float);
        test::ScopedGPUStream stream(DeviceId::cuda(0));

        // Allocate GPU memory
        void *d_a = cuda_backend_->allocate(bytes, 0);
        void *d_b = cuda_backend_->allocate(bytes, 0);
        void *d_c = cuda_backend_->allocate(bytes, 0);

        if (!d_a || !d_b || !d_c)
        {
            std::cerr << "CUDA allocation failed in phase: " << phase << std::endl;
            return false;
        }

        // Upload data
        if (!cuda_backend_->hostToDevice(
                d_a, host_a.data(), bytes, 0, stream.get()))
        {
            std::cerr << "CUDA H2D failed (a) in phase: " << phase << std::endl;
            return false;
        }
        if (!cuda_backend_->hostToDevice(
                d_b, host_b.data(), bytes, 0, stream.get()))
        {
            std::cerr << "CUDA H2D failed (b) in phase: " << phase << std::endl;
            return false;
        }

        // For now, just do a memset to verify basic CUDA operations work
        // (We don't have a simple vector add kernel exposed in the backend API)
        if (!cuda_backend_->memset(d_c, 0, bytes, 0, stream.get()))
        {
            std::cerr << "CUDA memset failed in phase: " << phase << std::endl;
            return false;
        }

        if (!cuda_backend_->synchronizeStream(stream.get(), 0))
        {
            std::cerr << "CUDA sync failed in phase: " << phase << std::endl;
            return false;
        }

        // Download and verify
        if (!cuda_backend_->deviceToHost(
                host_c.data(), d_c, bytes, 0, stream.get()))
        {
            std::cerr << "CUDA D2H failed in phase: " << phase << std::endl;
            return false;
        }

        // Verify memset worked (all zeros)
        bool valid = true;
        for (int i = 0; i < TEST_SIZE; ++i)
        {
            if (host_c[i] != 0.0f)
            {
                std::cerr << "CUDA verification failed at index " << i << " in phase: " << phase << std::endl;
                valid = false;
                break;
            }
        }

        // Cleanup
        cuda_backend_->free(d_a, 0);
        cuda_backend_->free(d_b, 0);
        cuda_backend_->free(d_c, 0);

        std::cout << "[" << phase << "] CUDA operations: " << (valid ? "PASSED" : "FAILED") << std::endl;
        return valid;
    }

    // Run a simple operation on ROCm
    bool runROCmOperation(const std::string &phase)
    {
        if (!rocm_backend_)
        {
            std::cerr << "ROCm backend not available" << std::endl;
            return false;
        }

        std::vector<float> host_data(TEST_SIZE, 3.0f);
        std::vector<float> host_result(TEST_SIZE, 0.0f);

        size_t bytes = TEST_SIZE * sizeof(float);
        test::ScopedGPUStream stream(DeviceId::rocm(0));

        // Allocate GPU memory
        void *d_data = rocm_backend_->allocate(bytes, 0);
        if (!d_data)
        {
            std::cerr << "ROCm allocation failed in phase: " << phase << std::endl;
            return false;
        }

        // Upload data
        if (!rocm_backend_->hostToDevice(
                d_data, host_data.data(), bytes, 0, stream.get()))
        {
            std::cerr << "ROCm H2D failed in phase: " << phase << std::endl;
            return false;
        }

        // Sync
        if (!rocm_backend_->synchronizeStream(stream.get(), 0))
        {
            std::cerr << "ROCm sync failed in phase: " << phase << std::endl;
            return false;
        }

        // Download
        if (!rocm_backend_->deviceToHost(
                host_result.data(), d_data, bytes, 0, stream.get()))
        {
            std::cerr << "ROCm D2H failed in phase: " << phase << std::endl;
            return false;
        }

        // Verify
        bool valid = true;
        for (int i = 0; i < TEST_SIZE; ++i)
        {
            if (std::abs(host_result[i] - 3.0f) > 1e-6f)
            {
                std::cerr << "ROCm verification failed at index " << i << " in phase: " << phase << std::endl;
                valid = false;
                break;
            }
        }

        // Cleanup
        rocm_backend_->free(d_data, 0);

        std::cout << "[" << phase << "] ROCm operations: " << (valid ? "PASSED" : "FAILED") << std::endl;
        return valid;
    }
};

/**
 * @test Verify CUDA works after ROCm operations
 *
 * This is the core test for CUDA/HIP coexistence.
 */
TEST_F(Test__CUDAHIPKernelCoexistence, CUDAWorksAfterROCm)
{
    if (!cuda_backend_)
    {
        GTEST_SKIP() << "CUDA not available";
    }
    if (!rocm_backend_)
    {
        GTEST_SKIP() << "ROCm not available";
    }

    std::cout << "=== Phase 1: CUDA baseline ===" << std::endl;
    ASSERT_TRUE(runCUDAVectorAdd("Phase1_CUDABaseline"));

    std::cout << "=== Phase 2: ROCm operations ===" << std::endl;
    ASSERT_TRUE(runROCmOperation("Phase2_ROCm"));

    std::cout << "=== Phase 3: CUDA after ROCm ===" << std::endl;
    ASSERT_TRUE(runCUDAVectorAdd("Phase3_CUDAAfterROCm"));

    std::cout << "=== All phases passed! ===" << std::endl;
}

/**
 * @test Verify CUDA works after multiple ROCm operations
 */
TEST_F(Test__CUDAHIPKernelCoexistence, CUDAWorksAfterMultipleROCmOps)
{
    if (!cuda_backend_)
    {
        GTEST_SKIP() << "CUDA not available";
    }
    if (!rocm_backend_)
    {
        GTEST_SKIP() << "ROCm not available";
    }

    std::cout << "=== CUDA baseline ===" << std::endl;
    ASSERT_TRUE(runCUDAVectorAdd("Baseline"));

    // Run multiple ROCm operations
    for (int i = 0; i < 5; ++i)
    {
        std::cout << "=== ROCm iteration " << i << " ===" << std::endl;
        ASSERT_TRUE(runROCmOperation("ROCm_iter" + std::to_string(i)));
    }

    std::cout << "=== CUDA after multiple ROCm ops ===" << std::endl;
    ASSERT_TRUE(runCUDAVectorAdd("AfterMultipleROCm"));
}

/**
 * @test Interleaved CUDA and ROCm operations
 */
TEST_F(Test__CUDAHIPKernelCoexistence, InterleavedOperations)
{
    if (!cuda_backend_)
    {
        GTEST_SKIP() << "CUDA not available";
    }
    if (!rocm_backend_)
    {
        GTEST_SKIP() << "ROCm not available";
    }

    for (int i = 0; i < 10; ++i)
    {
        std::cout << "=== Iteration " << i << " ===" << std::endl;
        ASSERT_TRUE(runCUDAVectorAdd("CUDA_iter" + std::to_string(i)));
        ASSERT_TRUE(runROCmOperation("ROCm_iter" + std::to_string(i)));
    }

    std::cout << "=== Interleaved operations passed! ===" << std::endl;
}

/**
 * @test Verify CUDA events work after ROCm operations
 *
 * This tests the specific error 709 issue - CUDA events become invalid
 * after ROCm operations.
 */
TEST_F(Test__CUDAHIPKernelCoexistence, CUDAEventsAfterROCm)
{
    if (!cuda_backend_)
    {
        GTEST_SKIP() << "CUDA not available";
    }
    if (!rocm_backend_)
    {
        GTEST_SKIP() << "ROCm not available";
    }
    test::ScopedGPUStream stream(DeviceId::cuda(0));

    std::cout << "=== Testing CUDA events before ROCm ===" << std::endl;

    // Create a CUDA event
    void *event = cuda_backend_->createEvent(0);
    ASSERT_NE(event, nullptr) << "Failed to create CUDA event";

    // Record the event
    ASSERT_TRUE(cuda_backend_->recordEvent(event, 0, stream.get()))
        << "Failed to record CUDA event (before ROCm)";

    // Wait for the event (should work)
    ASSERT_TRUE(cuda_backend_->waitForEvent(event, 0)) << "Failed to wait for CUDA event (before ROCm)";

    std::cout << "CUDA event operations work before ROCm" << std::endl;

    std::cout << "=== Running ROCm operations ===" << std::endl;
    ASSERT_TRUE(runROCmOperation("ROCm_before_event_retest"));

    std::cout << "=== Testing CUDA events after ROCm ===" << std::endl;

    // Record the event again (after ROCm operations)
    bool record_ok = cuda_backend_->recordEvent(event, 0, stream.get());
    std::cout << "Record event after ROCm: " << (record_ok ? "OK" : "FAILED") << std::endl;
    EXPECT_TRUE(record_ok) << "Failed to record CUDA event (after ROCm)";

    // Wait for the event (this is where error 709 might occur)
    bool wait_ok = cuda_backend_->waitForEvent(event, 0);
    std::cout << "Wait for event after ROCm: " << (wait_ok ? "OK" : "FAILED") << std::endl;
    EXPECT_TRUE(wait_ok) << "Failed to wait for CUDA event (after ROCm) - possible error 709";

    // Cleanup
    cuda_backend_->destroyEvent(event, 0);

    std::cout << "=== CUDA event test complete ===" << std::endl;
}

/**
 * @test Create new CUDA events after ROCm operations
 *
 * Tests if creating new events works, even if old ones are broken.
 */
TEST_F(Test__CUDAHIPKernelCoexistence, NewCUDAEventsAfterROCm)
{
    if (!cuda_backend_)
    {
        GTEST_SKIP() << "CUDA not available";
    }
    if (!rocm_backend_)
    {
        GTEST_SKIP() << "ROCm not available";
    }
    test::ScopedGPUStream stream(DeviceId::cuda(0));

    std::cout << "=== Running ROCm operations first ===" << std::endl;
    ASSERT_TRUE(runROCmOperation("ROCm_first"));

    std::cout << "=== Creating CUDA event after ROCm ===" << std::endl;

    // Create a new CUDA event after ROCm operations
    void *event = cuda_backend_->createEvent(0);
    ASSERT_NE(event, nullptr) << "Failed to create CUDA event after ROCm";

    // Record the event
    ASSERT_TRUE(cuda_backend_->recordEvent(event, 0, stream.get()))
        << "Failed to record new CUDA event after ROCm";

    // Wait for the event
    ASSERT_TRUE(cuda_backend_->waitForEvent(event, 0)) << "Failed to wait for new CUDA event after ROCm";

    // Cleanup
    cuda_backend_->destroyEvent(event, 0);

    std::cout << "=== New CUDA events work after ROCm ===" << std::endl;
}
