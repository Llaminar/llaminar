/**
 * @file Test__TensorAllocateOnDevice.cpp
 * @brief Unit tests for TensorBase::allocateOnDevice()
 * @author GitHub Copilot
 * @date January 2026
 *
 * Tests the allocateOnDevice() method which allocates GPU memory
 * WITHOUT uploading host data. This is an optimization for OUTPUT
 * tensors where the kernel will overwrite the contents.
 *
 * Test Categories:
 * - Parameter validation (CPU device rejected)
 * - Memory allocation on GPU devices
 * - No H2D transfer on allocation
 * - Reuse of existing allocations
 * - Coherence state management (device_valid_, host_valid_)
 */

#include <gtest/gtest.h>
#include "v2/tensors/Tensors.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/utils/Logger.h"
#include <memory>
#include <vector>

using namespace llaminar2;

namespace llaminar2::test
{

    /**
     * @brief Test fixture for allocateOnDevice() tests
     */
    class Test__TensorAllocateOnDevice : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create a test tensor with known data
            tensor_ = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 64});

            // Fill with test data
            float *data = tensor_->mutable_data();
            for (size_t i = 0; i < 32 * 64; ++i)
            {
                data[i] = static_cast<float>(i);
            }
        }

        void TearDown() override
        {
            tensor_.reset();
        }

        std::unique_ptr<FP32Tensor> tensor_;

        // Helper to check if any GPU backend is available
        bool hasGPU() const
        {
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
            return hasGPUBackend();
#else
            return false;
#endif
        }

        // Helper to get a valid GPU DeviceId
        DeviceId getGPUDevice() const
        {
#ifdef HAVE_CUDA
            if (getCUDABackend() != nullptr)
            {
                return DeviceId::cuda(0);
            }
#endif
#ifdef HAVE_ROCM
            if (getROCmBackend() != nullptr)
            {
                return DeviceId::rocm(0);
            }
#endif
            // Should not reach here if hasGPU() returned true
            return DeviceId::cpu();
        }
    };

    // =========================================================================
    // Parameter Validation Tests
    // =========================================================================

    /**
     * @brief Verify allocateOnDevice() rejects CPU device
     *
     * allocateOnDevice() must be called with a GPU device. CPU targets
     * should return false since there's no "allocation" to do.
     */
    TEST_F(Test__TensorAllocateOnDevice, RejectsCPUDevice)
    {
        DeviceId cpu_device = DeviceId::cpu();

        bool result = tensor_->allocateOnDevice(cpu_device);

        EXPECT_FALSE(result) << "allocateOnDevice() should fail for CPU device";
    }

    // =========================================================================
    // GPU Allocation Tests (require actual GPU)
    // =========================================================================

    /**
     * @brief Verify allocateOnDevice() allocates GPU buffer
     *
     * After calling allocateOnDevice(), gpu_data_ptr() should return
     * a non-null pointer.
     */
    TEST_F(Test__TensorAllocateOnDevice, AllocatesGPUBuffer)
    {
        if (!hasGPU())
        {
            GTEST_SKIP() << "No GPU backend available";
        }

        DeviceId gpu_device = getGPUDevice();

        // Initially should not be on GPU
        EXPECT_EQ(tensor_->gpu_data_ptr(), nullptr) << "Tensor should start with no GPU buffer";

        // Allocate on device
        bool result = tensor_->allocateOnDevice(gpu_device);

        ASSERT_TRUE(result) << "allocateOnDevice() should succeed";
        EXPECT_NE(tensor_->gpu_data_ptr(), nullptr) << "gpu_data_ptr() should be non-null after allocation";
    }

    /**
     * @brief Verify allocateOnDevice() sets device_valid_ to false
     *
     * After allocation, device_valid_ should be false because no data
     * has been uploaded yet. The kernel will write to the buffer.
     */
    TEST_F(Test__TensorAllocateOnDevice, DeviceValidIsFalseAfterAllocation)
    {
        if (!hasGPU())
        {
            GTEST_SKIP() << "No GPU backend available";
        }

        DeviceId gpu_device = getGPUDevice();

        bool result = tensor_->allocateOnDevice(gpu_device);
        ASSERT_TRUE(result);

        // device_valid_ should be false - kernel hasn't written yet
        EXPECT_FALSE(tensor_->isDeviceValid()) << "isDeviceValid() should return false after allocateOnDevice()";
    }

    /**
     * @brief Verify allocateOnDevice() does NOT change host_valid_
     *
     * host_valid_ should remain unchanged (typically true) since we
     * didn't modify any host data.
     */
    TEST_F(Test__TensorAllocateOnDevice, HostValidUnchanged)
    {
        if (!hasGPU())
        {
            GTEST_SKIP() << "No GPU backend available";
        }

        DeviceId gpu_device = getGPUDevice();

        // Host should be valid before (we just created and filled the tensor)
        EXPECT_TRUE(tensor_->isOnCPU()) << "Tensor should start with valid host data";

        bool result = tensor_->allocateOnDevice(gpu_device);
        ASSERT_TRUE(result);

        // Host should STILL be valid - we didn't touch host data
        EXPECT_TRUE(tensor_->isOnCPU()) << "Host data should remain valid after allocateOnDevice()";
    }

    /**
     * @brief Verify allocateOnDevice() reuses existing allocation
     *
     * Calling allocateOnDevice() twice on the same device should not
     * allocate new memory - it should reuse the existing buffer.
     */
    TEST_F(Test__TensorAllocateOnDevice, ReusesExistingAllocation)
    {
        if (!hasGPU())
        {
            GTEST_SKIP() << "No GPU backend available";
        }

        DeviceId gpu_device = getGPUDevice();

        // First allocation
        bool result1 = tensor_->allocateOnDevice(gpu_device);
        ASSERT_TRUE(result1);
        void *ptr1 = tensor_->gpu_data_ptr();
        ASSERT_NE(ptr1, nullptr);

        // Second allocation - should reuse
        bool result2 = tensor_->allocateOnDevice(gpu_device);
        ASSERT_TRUE(result2);
        void *ptr2 = tensor_->gpu_data_ptr();

        EXPECT_EQ(ptr1, ptr2) << "Second allocateOnDevice() should reuse existing buffer";
    }

    /**
     * @brief Compare allocateOnDevice() vs ensureOnDevice() behavior
     *
     * ensureOnDevice() should upload data (device_valid_ = true)
     * allocateOnDevice() should NOT upload (device_valid_ = false)
     */
    TEST_F(Test__TensorAllocateOnDevice, NoUploadUnlikeEnsureOnDevice)
    {
        if (!hasGPU())
        {
            GTEST_SKIP() << "No GPU backend available";
        }

        DeviceId gpu_device = getGPUDevice();

        // Test allocateOnDevice - should NOT set device_valid_
        {
            auto tensor_alloc = std::make_unique<FP32Tensor>(std::vector<size_t>{16, 16});
            float *data = tensor_alloc->mutable_data();
            for (size_t i = 0; i < 256; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            bool result = tensor_alloc->allocateOnDevice(gpu_device);
            ASSERT_TRUE(result);
            EXPECT_FALSE(tensor_alloc->isDeviceValid()) << "allocateOnDevice should NOT set device_valid_";

            // Clean up GPU memory
            tensor_alloc->releaseDeviceMemory();
        }

        // Test ensureOnDevice - SHOULD set device_valid_
        {
            auto tensor_ensure = std::make_unique<FP32Tensor>(std::vector<size_t>{16, 16});
            float *data = tensor_ensure->mutable_data();
            for (size_t i = 0; i < 256; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            bool result = tensor_ensure->ensureOnDevice(gpu_device);
            ASSERT_TRUE(result);
            EXPECT_TRUE(tensor_ensure->isDeviceValid()) << "ensureOnDevice SHOULD set device_valid_";

            // Clean up GPU memory
            tensor_ensure->releaseDeviceMemory();
        }
    }

    /**
     * @brief Verify allocateOnDevice() works with different tensor sizes
     *
     * Test with small, medium, and large tensors to ensure allocation
     * works correctly across different memory sizes.
     */
    TEST_F(Test__TensorAllocateOnDevice, WorksWithDifferentSizes)
    {
        if (!hasGPU())
        {
            GTEST_SKIP() << "No GPU backend available";
        }

        DeviceId gpu_device = getGPUDevice();

        // Small tensor (256 bytes)
        {
            auto small = std::make_unique<FP32Tensor>(std::vector<size_t>{8, 8});
            EXPECT_TRUE(small->allocateOnDevice(gpu_device)) << "Small tensor allocation failed";
            EXPECT_NE(small->gpu_data_ptr(), nullptr);
            small->releaseDeviceMemory();
        }

        // Medium tensor (1 MB)
        {
            auto medium = std::make_unique<FP32Tensor>(std::vector<size_t>{512, 512});
            EXPECT_TRUE(medium->allocateOnDevice(gpu_device)) << "Medium tensor allocation failed";
            EXPECT_NE(medium->gpu_data_ptr(), nullptr);
            medium->releaseDeviceMemory();
        }

        // Large tensor (16 MB)
        {
            auto large = std::make_unique<FP32Tensor>(std::vector<size_t>{2048, 2048});
            EXPECT_TRUE(large->allocateOnDevice(gpu_device)) << "Large tensor allocation failed";
            EXPECT_NE(large->gpu_data_ptr(), nullptr);
            large->releaseDeviceMemory();
        }
    }

    /**
     * @brief Verify device memory is released after releaseDeviceMemory()
     *
     * After calling releaseDeviceMemory(), gpu_data_ptr() should be null,
     * and allocateOnDevice() should allocate fresh memory.
     */
    TEST_F(Test__TensorAllocateOnDevice, AllocateAfterRelease)
    {
        if (!hasGPU())
        {
            GTEST_SKIP() << "No GPU backend available";
        }

        DeviceId gpu_device = getGPUDevice();

        // First allocation
        bool result1 = tensor_->allocateOnDevice(gpu_device);
        ASSERT_TRUE(result1);
        void *ptr1 = tensor_->gpu_data_ptr();

        // Release
        tensor_->releaseDeviceMemory();
        EXPECT_EQ(tensor_->gpu_data_ptr(), nullptr) << "gpu_data_ptr() should be null after release";

        // Allocate again
        bool result2 = tensor_->allocateOnDevice(gpu_device);
        ASSERT_TRUE(result2);
        void *ptr2 = tensor_->gpu_data_ptr();
        EXPECT_NE(ptr2, nullptr) << "Should be able to allocate after release";

        // Note: ptr1 and ptr2 may or may not be the same (allocator-dependent)
    }

} // namespace llaminar2::test
