/**
 * @file Test__CUDATensorTransfer.cpp
 * @brief Test GPU tensor upload/download operations
 *
 * **Purpose**: Validate TensorBase::ensureOnDevice() and ensureOnHost() for
 * FP32, BF16, FP16 tensors using CUDA backend.
 *
 * **Tests**:
 * - Upload FP32 tensor to GPU, verify device pointer non-null (isOnGPU)
 * - Round-trip: upload → download, verify bit-exact match
 * - Large tensor transfers (multi-MB)
 * - State tracking (isOnGPU)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

// Include project headers BEFORE CUDATestUtils.h (provides include paths)
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h" // DeviceManager
#include "execution/local_execution/device/DeviceContext.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif

// Now include test utils (uses headers above)
#include "../utils/CUDATestUtils.h"

#include <vector>
#include <cstring>

using namespace llaminar2;
using namespace llaminar2::test::cuda;

// ============================================================================
// FP32 Tensor Transfer Tests
// ============================================================================

class Test__CUDATensorTransfer : public CUDATestBase
{
};

TEST_F(Test__CUDATensorTransfer, FP32_Upload)
{
    // Create FP32 tensor with known values
    const size_t rows = 128;
    const size_t cols = 256;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
    ASSERT_NE(tensor, nullptr);

    // Fill with test pattern
    float *data = tensor->mutable_data();
    ASSERT_NE(data, nullptr);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>(i) * 0.001f;
    }

    // Upload to GPU
    ASSERT_FALSE(tensor->isOnGPU()) << "Tensor should start on host";
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_)) << "GPU upload failed";
    EXPECT_TRUE(tensor->isOnGPU()) << "Tensor should be on GPU after ensureOnDevice";
}

TEST_F(Test__CUDATensorTransfer, FP32_RoundTrip)
{
    // Create tensor with known values
    const size_t rows = 64;
    const size_t cols = 128;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
    float *data = tensor->mutable_data();

    // Fill with test pattern
    std::vector<float> original(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        float val = static_cast<float>(i) * 0.5f - 100.0f;
        data[i] = val;
        original[i] = val;
    }

    // Upload to GPU
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->isOnGPU());

    // Download back to host
    ASSERT_TRUE(tensor->ensureOnHost());

    // Verify data matches original
    const float *result = tensor->data();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(Test__CUDATensorTransfer, FP32_LargeTensor)
{
    // Test with 16MB tensor (4M floats)
    const size_t rows = 2048;
    const size_t cols = 2048;
    const size_t total = rows * cols;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
    float *data = tensor->mutable_data();

    // Fill with pattern
    std::vector<float> original(total);
    for (size_t i = 0; i < total; ++i)
    {
        data[i] = static_cast<float>(i % 1000) * 0.01f;
        original[i] = data[i];
    }

    // Upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Download
    ASSERT_TRUE(tensor->ensureOnHost());

    // Verify (sample check for speed)
    const float *result = tensor->data();
    for (size_t i = 0; i < total; i += 1000)
    {
        EXPECT_FLOAT_EQ(result[i], original[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(Test__CUDATensorTransfer, FP32_AlreadyOnDevice)
{
    // Test that second ensureOnDevice is a no-op
    const size_t rows = 32;
    const size_t cols = 32;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // First upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->isOnGPU());

    // Second upload should succeed (no-op)
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->isOnGPU());
}

// ============================================================================
// BF16 Tensor Transfer Tests
// ============================================================================

TEST_F(Test__CUDATensorTransfer, BF16_Upload)
{
    const size_t rows = 64;
    const size_t cols = 128;

    auto tensor = std::make_unique<BF16Tensor>(std::vector<size_t>{rows, cols});
    ASSERT_NE(tensor, nullptr);

    // Fill with test pattern (BF16 data)
    uint16_t *data = tensor->mutable_bf16_data();
    ASSERT_NE(data, nullptr);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        // Store simple pattern as BF16 bits
        data[i] = static_cast<uint16_t>(i & 0xFFFF);
    }

    // Upload to GPU
    ASSERT_FALSE(tensor->isOnGPU());
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->isOnGPU());
}

TEST_F(Test__CUDATensorTransfer, BF16_RoundTrip)
{
    const size_t rows = 32;
    const size_t cols = 64;

    auto tensor = std::make_unique<BF16Tensor>(std::vector<size_t>{rows, cols});
    uint16_t *data = tensor->mutable_bf16_data();

    std::vector<uint16_t> original(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        uint16_t val = static_cast<uint16_t>(i % 65536);
        data[i] = val;
        original[i] = val;
    }

    // Round trip
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->ensureOnHost());

    // Verify BF16 data
    const uint16_t *result = tensor->bf16_data();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_EQ(result[i], original[i])
            << "BF16 mismatch at index " << i;
    }
}

// ============================================================================
// FP16 Tensor Transfer Tests
// ============================================================================

TEST_F(Test__CUDATensorTransfer, FP16_Upload)
{
    const size_t rows = 64;
    const size_t cols = 128;

    auto tensor = std::make_unique<FP16Tensor>(std::vector<size_t>{rows, cols});
    uint16_t *data = tensor->mutable_fp16_data();

    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<uint16_t>(i & 0xFFFF);
    }

    ASSERT_FALSE(tensor->isOnGPU());
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->isOnGPU());
}

TEST_F(Test__CUDATensorTransfer, FP16_RoundTrip)
{
    const size_t rows = 32;
    const size_t cols = 64;

    auto tensor = std::make_unique<FP16Tensor>(std::vector<size_t>{rows, cols});
    uint16_t *data = tensor->mutable_fp16_data();

    std::vector<uint16_t> original(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        uint16_t val = static_cast<uint16_t>(i % 65536);
        data[i] = val;
        original[i] = val;
    }

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->ensureOnHost());

    const uint16_t *result = tensor->fp16_data();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_EQ(result[i], original[i])
            << "FP16 mismatch at index " << i;
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(Test__CUDATensorTransfer, SmallTensor_1x1)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1, 1});
    tensor->mutable_data()[0] = 3.14159f;

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->ensureOnHost());

    EXPECT_FLOAT_EQ(tensor->data()[0], 3.14159f);
}

TEST_F(Test__CUDATensorTransfer, SingleRow)
{
    // Single row vector (common in decode phase)
    const size_t cols = 896; // Typical hidden dimension

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1, cols});
    float *data = tensor->mutable_data();

    std::vector<float> original(cols);
    for (size_t i = 0; i < cols; ++i)
    {
        data[i] = static_cast<float>(i) * 0.01f;
        original[i] = data[i];
    }

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->ensureOnHost());

    const float *result = tensor->data();
    for (size_t i = 0; i < cols; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i]);
    }
}

TEST_F(Test__CUDATensorTransfer, TallSkinny)
{
    // Tall skinny matrix (embedding table access pattern)
    const size_t rows = 8192;
    const size_t cols = 64;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
    float *data = tensor->mutable_data();

    std::vector<float> original(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>(i % 1000) * 0.001f;
        original[i] = data[i];
    }

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->ensureOnHost());

    const float *result = tensor->data();
    // Sample check
    for (size_t i = 0; i < rows * cols; i += 500)
    {
        EXPECT_FLOAT_EQ(result[i], original[i]);
    }
}
