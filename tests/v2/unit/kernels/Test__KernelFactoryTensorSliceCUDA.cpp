/**
 * @file Test__KernelFactoryTensorSliceCUDA.cpp
 * @brief Unit tests for TensorSlice handling in KernelFactory dispatch path
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests that KernelFactory correctly unwraps TensorSlice wrappers when
 * creating GEMM kernels. This is critical for MPI tensor parallelism
 * where weights are wrapped in TensorSlice for sharding.
 *
 * Regression test for issue: "unsupported tensor type 10 for device CUDA:0"
 * Root cause: dynamic_cast<Q4_0Tensor*>(TensorSlice*) returns nullptr
 * Solution: Unwrap TensorSlice via slice->inner() before type dispatch
 *
 * These tests verify the unwrapping logic works correctly. The actual
 * CUDA kernel creation is tested in integration tests with real GPUs.
 */

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar::v2::kernels;

namespace
{
    // Helper to check CUDA availability at runtime
    bool hasCUDA()
    {
#ifdef HAVE_CUDA
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
#else
        return false;
#endif
    }

    ITensorGemm *getPreparedKernel(const TensorBase *tensor, DeviceId device_id = DeviceId::cpu())
    {
        static std::vector<std::pair<const TensorBase *, std::shared_ptr<KernelFactory::PreparedGemmHandle>>> handles;
        for (auto &[key, handle] : handles)
        {
            if (key == tensor && handle->device_id == device_id)
                return KernelFactory::getOrCreateGemmEngine(handle.get());
        }

        auto prepared = KernelFactory::prepareGemmHandleLocal(tensor, device_id);
        if (!prepared)
            return nullptr;
        auto *kernel = KernelFactory::getOrCreateGemmEngine(prepared.get());
        handles.emplace_back(tensor, std::move(prepared));
        return kernel;
    }
} // namespace

class KernelFactoryTensorSliceCUDATest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        KernelFactory::clearCache();
        DeviceManager::instance();
    }

    void TearDown() override
    {
        KernelFactory::clearCache();
    }

    // Helper to create Q4_0 tensor with random data
    std::unique_ptr<Q4_0Tensor> createQ4_0(size_t rows, size_t cols)
    {
        const size_t block_size = 32;
        const size_t bytes_per_block = 18; // Q4_0 block: 2 bytes scale + 16 bytes data
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw_data)
        {
            byte = static_cast<uint8_t>(dist(gen));
        }
        // Set scale values to reasonable floats
        for (size_t b = 0; b < num_blocks; ++b)
        {
            uint16_t scale_bits = 0x3C00; // ~1.0 in FP16
            memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<Q4_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    // Helper to create Q8_0 tensor with random data
    std::unique_ptr<Q8_0Tensor> createQ8_0(size_t rows, size_t cols)
    {
        const size_t block_size = 32;
        const size_t bytes_per_block = 34; // Q8_0 block: 2 bytes scale + 32 bytes data
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw_data)
        {
            byte = static_cast<uint8_t>(dist(gen));
        }
        for (size_t b = 0; b < num_blocks; ++b)
        {
            uint16_t scale_bits = 0x3C00;
            memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<Q8_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    // Helper to create IQ4_NL tensor with random data
    std::unique_ptr<IQ4_NLTensor> createIQ4_NL(size_t rows, size_t cols)
    {
        const size_t block_size = 32;
        // IQ4_NL block: 2 bytes scale + 16 bytes qs (4-bit indices)
        const size_t bytes_per_block = 18;
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block);

        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw_data)
        {
            byte = static_cast<uint8_t>(dist(gen));
        }
        for (size_t b = 0; b < num_blocks; ++b)
        {
            uint16_t scale_bits = 0x3C00;
            memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<IQ4_NLTensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    // Helper to wrap tensor in TensorSlice with unique_ptr<TensorBase>
    std::unique_ptr<TensorSlice> wrapInTensorSlice(std::unique_ptr<TensorBase> tensor, SliceMode mode)
    {
        size_t rows = tensor->shape()[0];
        size_t cols = tensor->shape()[1];

        SliceMetadata metadata;
        metadata.mode = mode;
        metadata.original_rows = rows;
        metadata.original_cols = cols;
        metadata.slice_start = 0;
        metadata.slice_end = (mode == SliceMode::ROW_PARALLEL) ? rows : cols;
        metadata.inner_is_presliced = false;

        return std::make_unique<TensorSlice>(std::move(tensor), std::move(metadata));
    }
};

// =============================================================================
// TensorSlice Type Detection Tests
// =============================================================================

TEST_F(KernelFactoryTensorSliceCUDATest, TensorSliceReportsInnerNativeType)
{
    // Verify TensorSlice correctly delegates native_type() to inner tensor
    auto inner = createQ4_0(1024, 896);
    ASSERT_EQ(inner->native_type(), TensorType::Q4_0);

    auto slice = wrapInTensorSlice(std::move(inner), SliceMode::COLUMN_PARALLEL);
    ASSERT_NE(slice, nullptr);

    // TensorSlice.native_type() should delegate to inner
    EXPECT_EQ(slice->native_type(), TensorType::Q4_0)
        << "TensorSlice should report inner tensor's native type";
}

TEST_F(KernelFactoryTensorSliceCUDATest, DirectCastToInnerTypeFails)
{
    // This is the root cause of the bug: dynamic_cast<Q4_0Tensor*>(TensorSlice*) fails
    auto inner = createQ4_0(512, 256);
    auto slice = wrapInTensorSlice(std::move(inner), SliceMode::COLUMN_PARALLEL);

    // Direct cast from TensorSlice to Q4_0Tensor should fail
    auto *direct_cast = dynamic_cast<Q4_0Tensor *>(slice.get());
    EXPECT_EQ(direct_cast, nullptr)
        << "Direct dynamic_cast from TensorSlice to Q4_0Tensor should return nullptr";
}

TEST_F(KernelFactoryTensorSliceCUDATest, CastThroughInnerSucceeds)
{
    // The fix: cast via slice->inner()
    auto inner = createQ4_0(512, 256);
    auto slice = wrapInTensorSlice(std::move(inner), SliceMode::COLUMN_PARALLEL);

    auto *slice_ptr = dynamic_cast<TensorSlice *>(slice.get());
    ASSERT_NE(slice_ptr, nullptr);

    // Cast from inner tensor should succeed
    auto *inner_cast = dynamic_cast<Q4_0Tensor *>(slice_ptr->inner());
    EXPECT_NE(inner_cast, nullptr)
        << "dynamic_cast from TensorSlice::inner() to Q4_0Tensor should succeed";
}

// =============================================================================
// CPU Dispatch Tests (baseline - always works)
// =============================================================================

TEST_F(KernelFactoryTensorSliceCUDATest, CPUDispatchHandlesTensorSliceQ4_0)
{
    // Create Q4_0 tensor and wrap in TensorSlice
    auto inner = createQ4_0(1024, 896);
    auto slice = wrapInTensorSlice(std::move(inner), SliceMode::COLUMN_PARALLEL);
    ASSERT_NE(slice, nullptr);

    // Should work on CPU (baseline)
    auto *kernel = getPreparedKernel(slice.get(), DeviceId::cpu());
    EXPECT_NE(kernel, nullptr) << "CPU GEMM kernel creation should succeed with TensorSlice";
}

TEST_F(KernelFactoryTensorSliceCUDATest, CPUDispatchHandlesTensorSliceQ8_0)
{
    auto inner = createQ8_0(1024, 896);
    auto slice = wrapInTensorSlice(std::move(inner), SliceMode::ROW_PARALLEL);
    ASSERT_NE(slice, nullptr);

    auto *kernel = getPreparedKernel(slice.get(), DeviceId::cpu());
    EXPECT_NE(kernel, nullptr) << "CPU GEMM kernel creation should succeed with TensorSlice wrapping Q8_0";
}

TEST_F(KernelFactoryTensorSliceCUDATest, CPUDispatchHandlesTensorSliceIQ4_NL)
{
    auto inner = createIQ4_NL(1024, 896);
    auto slice = wrapInTensorSlice(std::move(inner), SliceMode::COLUMN_PARALLEL);
    ASSERT_NE(slice, nullptr);

    auto *kernel = getPreparedKernel(slice.get(), DeviceId::cpu());
    EXPECT_NE(kernel, nullptr) << "CPU GEMM kernel creation should succeed with TensorSlice wrapping IQ4_NL";
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(KernelFactoryTensorSliceCUDATest, DirectQ4_0TensorStillWorks)
{
    // Ensure we didn't break direct tensor handling (non-sliced)
    auto tensor = createQ4_0(1024, 896);
    ASSERT_NE(tensor, nullptr);

    auto *kernel = getPreparedKernel(tensor.get(), DeviceId::cpu());
    EXPECT_NE(kernel, nullptr)
        << "Direct Q4_0Tensor (not wrapped) should still work";
}

TEST_F(KernelFactoryTensorSliceCUDATest, KernelCachingWithTensorSlice)
{
    auto inner = createQ4_0(1024, 896);
    auto slice = wrapInTensorSlice(std::move(inner), SliceMode::COLUMN_PARALLEL);

    // Create kernel twice - should be cached
    auto *k1 = getPreparedKernel(slice.get(), DeviceId::cpu());
    auto *k2 = getPreparedKernel(slice.get(), DeviceId::cpu());

    EXPECT_NE(k1, nullptr);
    EXPECT_EQ(k1, k2) << "Same TensorSlice should return cached kernel";
}

TEST_F(KernelFactoryTensorSliceCUDATest, DifferentTensorSlicesGetDifferentKernels)
{
    auto inner1 = createQ4_0(1024, 896);
    auto slice1 = wrapInTensorSlice(std::move(inner1), SliceMode::COLUMN_PARALLEL);

    auto inner2 = createQ4_0(1024, 896);
    auto slice2 = wrapInTensorSlice(std::move(inner2), SliceMode::COLUMN_PARALLEL);

    auto *k1 = getPreparedKernel(slice1.get(), DeviceId::cpu());
    auto *k2 = getPreparedKernel(slice2.get(), DeviceId::cpu());

    EXPECT_NE(k1, nullptr);
    EXPECT_NE(k2, nullptr);
    EXPECT_NE(k1, k2) << "Different TensorSlice instances should get different kernels";
}

// =============================================================================
// CUDA Dispatch Tests (requires actual GPU)
// =============================================================================

#ifdef HAVE_CUDA

TEST_F(KernelFactoryTensorSliceCUDATest, CUDADispatchUnwrapsTensorSliceQ4_0)
{
    // This is the main regression test for the TensorSlice issue
    // When weights are wrapped in TensorSlice for MPI sharding,
    // CUDA dispatch must unwrap to find the inner Q4_0Tensor

    if (!hasCUDA())
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Find first CUDA device
    auto cuda_devices = DeviceManager::instance().get_devices_by_type(ComputeBackendType::GPU_CUDA);
    if (cuda_devices.empty())
    {
        GTEST_SKIP() << "No CUDA device in DeviceManager";
    }
    DeviceId cuda_device = DeviceId::cuda(DeviceManager::instance().devices()[cuda_devices[0]].device_id);

    auto inner = createQ4_0(1024, 896);
    // Transfer tensor to CUDA so KernelFactory uses CUDA dispatch path
    ASSERT_TRUE(inner->ensureOnDevice(cuda_device))
        << "Failed to transfer tensor to " << cuda_device.toString();
    auto slice = wrapInTensorSlice(std::move(inner), SliceMode::COLUMN_PARALLEL);
    ASSERT_NE(slice, nullptr);

    // This should succeed - previously failed with "unsupported tensor type 10"
    auto *kernel = getPreparedKernel(slice.get(), cuda_device);
    EXPECT_NE(kernel, nullptr)
        << "CUDA GEMM kernel creation should succeed with TensorSlice wrapping Q4_0Tensor. "
        << "If this fails with 'unsupported tensor type 10', the TensorSlice unwrapping fix is missing.";
}

TEST_F(KernelFactoryTensorSliceCUDATest, CUDADispatchUnwrapsTensorSliceQ8_0)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Find first CUDA device
    auto cuda_devices = DeviceManager::instance().get_devices_by_type(ComputeBackendType::GPU_CUDA);
    if (cuda_devices.empty())
    {
        GTEST_SKIP() << "No CUDA device in DeviceManager";
    }
    DeviceId cuda_device = DeviceId::cuda(DeviceManager::instance().devices()[cuda_devices[0]].device_id);

    auto inner = createQ8_0(1024, 896);
    ASSERT_TRUE(inner->ensureOnDevice(cuda_device))
        << "Failed to transfer tensor to " << cuda_device.toString();
    auto slice = wrapInTensorSlice(std::move(inner), SliceMode::ROW_PARALLEL);
    ASSERT_NE(slice, nullptr);

    auto *kernel = getPreparedKernel(slice.get(), cuda_device);
    EXPECT_NE(kernel, nullptr)
        << "CUDA GEMM kernel creation should succeed with TensorSlice wrapping Q8_0Tensor";
}

#endif // HAVE_CUDA
