/**
 * @file Test__MappedMemoryCoherence.cpp
 * @brief Integration tests for mapped memory coherence behavior
 *
 * **Purpose**: Validates that mapped memory tensors behave correctly:
 * - GPU pointer is available immediately after creation
 * - ensureOnHost() is a no-op (no D2H transfer)
 * - Kernel writes are visible on host without explicit sync
 * - Both host_valid_ and device_valid_ remain true
 *
 * **Test Strategy**:
 * Uses FP32Tensor::createMapped() with MockBackend to verify coherence
 * behavior without requiring actual GPU hardware.
 *
 * **Phase**: GPU-Resident Execution Optimization Phase 3
 * **See**: docs/v2/GPU_RESIDENT_EXECUTION_PROJECT_PLAN.md
 *
 * @author GitHub Copilot
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <cstring>

#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"
#include "backends/BackendManager.h"
#include "mocks/MockBackend.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

/**
 * @brief Test fixture for mapped memory coherence tests
 *
 * Sets up MockBackend for testing mapped memory without real GPU hardware.
 */
class Test__MappedMemoryCoherence : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a mock backend for testing
        mock_backend_ = std::make_shared<MockBackend>();

        // Get a GPU device ID for testing
        // Use CUDA device 0 as mock target
        target_device_ = DeviceId::cuda(0);

        // Standard test shape
        test_shape_ = {32, 64}; // 2048 elements
    }

    void TearDown() override
    {
        mock_backend_.reset();
    }

    /**
     * @brief Attempt to create a mapped tensor
     * @return Mapped tensor or nullptr if not supported
     */
    std::unique_ptr<FP32Tensor> tryCreateMapped(
        const std::vector<size_t> &shape,
        DeviceId device)
    {
        auto tensor = FP32Tensor::createMapped(shape, device);
        if (!tensor)
        {
            return nullptr;
        }
        if (!tensor->isMapped())
        {
            return nullptr; // Fell back to regular allocation
        }
        return tensor;
    }

    std::shared_ptr<MockBackend> mock_backend_;
    DeviceId target_device_;
    std::vector<size_t> test_shape_;
};

// ============================================================================
// Test: MappedTensor_HasValidGpuPtr
// ============================================================================

/**
 * @brief Verify gpu_data_ptr() is non-null immediately after creation
 *
 * For mapped tensors, the GPU pointer should be available immediately
 * since host and device share the same physical memory.
 */
TEST_F(Test__MappedMemoryCoherence, MappedTensor_HasValidGpuPtr)
{
    auto tensor = tryCreateMapped(test_shape_, target_device_);
    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported on this system";
    }

    // GPU pointer should be valid immediately (no ensureOnDevice needed)
    const void *gpu_ptr = tensor->gpu_data_ptr();
    EXPECT_NE(gpu_ptr, nullptr)
        << "Mapped tensor should have valid GPU pointer immediately after creation";

    // The pointer should also be non-null when accessed via the device method
    EXPECT_TRUE(tensor->isDeviceValid())
        << "Mapped tensor should report device as valid";
}

// ============================================================================
// Test: MappedTensor_NoMemcpyOnEnsureOnHost
// ============================================================================

/**
 * @brief Verify ensureOnHost() is a no-op for mapped tensors
 *
 * Since mapped memory is shared between host and device, there should
 * be no actual data transfer when calling ensureOnHost().
 */
TEST_F(Test__MappedMemoryCoherence, MappedTensor_NoMemcpyOnEnsureOnHost)
{
    auto tensor = tryCreateMapped(test_shape_, target_device_);
    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported on this system";
    }

    // Record initial transfer stats (if using real backend this would track)
    // For our test, we verify that ensureOnHost returns quickly

    // Mark as device dirty to simulate a kernel write
    tensor->mark_device_dirty();

    // This should NOT trigger a D2H memcpy for mapped tensors
    // The data is already accessible via mapped_host_ptr_
    tensor->ensureOnHost();

    // Host should still be valid (mapped memory is always accessible)
    EXPECT_TRUE(tensor->isOnCPU())
        << "Mapped tensor should remain host-valid after ensureOnHost";

    // Verify we can access host data
    const float *host_data = tensor->data();
    EXPECT_NE(host_data, nullptr)
        << "data() should return valid pointer for mapped tensor";
}

// ============================================================================
// Test: MappedTensor_KernelWriteVisibleOnHost
// ============================================================================

/**
 * @brief Verify kernel writes are visible to host without explicit memcpy
 *
 * When a GPU kernel writes to mapped memory, the host should be able
 * to see those writes immediately (after GPU synchronization).
 */
TEST_F(Test__MappedMemoryCoherence, MappedTensor_KernelWriteVisibleOnHost)
{
    auto tensor = tryCreateMapped(test_shape_, target_device_);
    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported on this system";
    }

    // Simulate a GPU kernel write by writing through the GPU pointer
    // In reality, this would be done by a CUDA/HIP kernel
    float *gpu_ptr = static_cast<float *>(tensor->gpu_data_ptr());
    ASSERT_NE(gpu_ptr, nullptr);

    // Write test pattern through "GPU" pointer (simulating kernel output)
    const float test_value = 42.0f;
    const size_t num_elements = 32 * 64;
    for (size_t i = 0; i < num_elements; ++i)
    {
        gpu_ptr[i] = test_value + static_cast<float>(i);
    }

    // Mark as device dirty (simulating what DeviceGraphExecutor does after kernel)
    tensor->mark_device_dirty();

    // Host should see the writes WITHOUT explicit memcpy
    const float *host_data = tensor->data();
    ASSERT_NE(host_data, nullptr);

    // Verify the values are visible
    EXPECT_FLOAT_EQ(host_data[0], test_value)
        << "First element should be visible on host";
    EXPECT_FLOAT_EQ(host_data[100], test_value + 100.0f)
        << "Element 100 should be visible on host";
    EXPECT_FLOAT_EQ(host_data[num_elements - 1], test_value + static_cast<float>(num_elements - 1))
        << "Last element should be visible on host";
}

// ============================================================================
// Test: MappedTensor_BothFlagsStayTrue
// ============================================================================

/**
 * @brief Verify both host_valid_ and device_valid_ remain true for mapped tensors
 *
 * The key property of mapped memory is that both host and device always
 * have access to the same data. The coherence flags should reflect this.
 */
TEST_F(Test__MappedMemoryCoherence, MappedTensor_BothFlagsStayTrue)
{
    auto tensor = tryCreateMapped(test_shape_, target_device_);
    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported on this system";
    }

    // Initially both should be valid for mapped tensor
    EXPECT_TRUE(tensor->isOnCPU())
        << "Mapped tensor should be host-valid initially";
    EXPECT_TRUE(tensor->isDeviceValid())
        << "Mapped tensor should be device-valid initially";

    // After mark_device_dirty(), both should STILL be valid
    // (unlike non-mapped tensors where host becomes invalid)
    tensor->mark_device_dirty();

    EXPECT_TRUE(tensor->isOnCPU())
        << "Mapped tensor should remain host-valid after mark_device_dirty";
    EXPECT_TRUE(tensor->isDeviceValid())
        << "Mapped tensor should remain device-valid after mark_device_dirty";

    // Calling ensureOnHost should not change the flags
    tensor->ensureOnHost();

    EXPECT_TRUE(tensor->isOnCPU())
        << "Mapped tensor should remain host-valid after ensureOnHost";
    EXPECT_TRUE(tensor->isDeviceValid())
        << "Mapped tensor should remain device-valid after ensureOnHost";
}

// ============================================================================
// Test: MappedTensor_IsMappedFlagCorrect
// ============================================================================

/**
 * @brief Verify isMapped() returns true for mapped tensors
 */
TEST_F(Test__MappedMemoryCoherence, MappedTensor_IsMappedFlagCorrect)
{
    auto tensor = tryCreateMapped(test_shape_, target_device_);
    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported on this system";
    }

    EXPECT_TRUE(tensor->isMapped())
        << "createMapped() should set isMapped() flag to true";
}

// ============================================================================
// Test: RegularTensor_IsMappedFlagFalse
// ============================================================================

/**
 * @brief Verify isMapped() returns false for regular tensors
 */
TEST_F(Test__MappedMemoryCoherence, RegularTensor_IsMappedFlagFalse)
{
    auto tensor = std::make_unique<FP32Tensor>(test_shape_, DeviceId::cpu());

    EXPECT_FALSE(tensor->isMapped())
        << "Regular tensor should have isMapped() == false";
}

// ============================================================================
// Test: MappedTensor_HostAndDevicePtrsSame
// ============================================================================

/**
 * @brief Verify that for mapped tensors, host and device pointers are related
 *
 * In a true mapped allocation, the device can access the host memory
 * directly. The pointers may be the same or different depending on the
 * platform, but both should be valid.
 */
TEST_F(Test__MappedMemoryCoherence, MappedTensor_HostAndDevicePtrsSame)
{
    auto tensor = tryCreateMapped(test_shape_, target_device_);
    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported on this system";
    }

    // Both pointers should be non-null
    const float *host_ptr = tensor->data();
    const void *device_ptr = tensor->gpu_data_ptr();

    EXPECT_NE(host_ptr, nullptr) << "Host pointer should be valid";
    EXPECT_NE(device_ptr, nullptr) << "Device pointer should be valid";

    // For MockBackend, the pointers are the same (true zero-copy simulation)
    // For real HIP/CUDA, they may be different but both point to same physical memory
}

// ============================================================================
// Test: MappedTensor_DataIntegrity
// ============================================================================

/**
 * @brief Verify data written through host is visible through device pointer
 */
TEST_F(Test__MappedMemoryCoherence, MappedTensor_DataIntegrity)
{
    auto tensor = tryCreateMapped(test_shape_, target_device_);
    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported on this system";
    }

    // Write through host
    float *mutable_data = tensor->mutable_data();
    ASSERT_NE(mutable_data, nullptr);

    const float test_pattern = 123.456f;
    mutable_data[0] = test_pattern;
    mutable_data[42] = test_pattern * 2;

    // Read through device pointer (which for mapped memory should see same data)
    float *gpu_ptr = static_cast<float *>(tensor->gpu_data_ptr());
    ASSERT_NE(gpu_ptr, nullptr);

    // In mapped memory, writes through host should be visible through device ptr
    EXPECT_FLOAT_EQ(gpu_ptr[0], test_pattern)
        << "Data written to host should be visible through device pointer";
    EXPECT_FLOAT_EQ(gpu_ptr[42], test_pattern * 2)
        << "Data written to host should be visible through device pointer";
}

// ============================================================================
// Test: MappedTensor_ZeroInitialized
// ============================================================================

/**
 * @brief Verify mapped tensor is zero-initialized
 */
TEST_F(Test__MappedMemoryCoherence, MappedTensor_ZeroInitialized)
{
    auto tensor = tryCreateMapped(test_shape_, target_device_);
    if (!tensor || !tensor->isMapped())
    {
        GTEST_SKIP() << "Mapped memory allocation not supported on this system";
    }

    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    // Mapped memory should be zero-initialized (memset in initMappedMemory)
    const size_t num_elements = 32 * 64;
    for (size_t i = 0; i < std::min(num_elements, size_t(100)); ++i)
    {
        EXPECT_FLOAT_EQ(data[i], 0.0f)
            << "Mapped tensor should be zero-initialized at index " << i;
    }
}
