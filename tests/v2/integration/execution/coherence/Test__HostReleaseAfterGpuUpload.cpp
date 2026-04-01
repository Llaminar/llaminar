/**
 * @file Test__HostReleaseAfterGpuUpload.cpp
 * @brief Regression tests for CUDA pinned memory corruption when releasing host data
 *
 * **Bug reproduced**: After uploading a tensor to GPU via ensureOnDevice() (which pins
 * host memory via cudaHostRegister), calling release_host_weight_data() without first
 * unpinning caused CUDA internal tracking corruption. Subsequent cudaMalloc calls
 * would fail with "resource already mapped".
 *
 * **Root cause**: release_raw_data() freed the host buffer that was still registered
 * as pinned memory. CUDA's tracking of pinned regions became inconsistent, causing
 * all subsequent device memory operations to fail.
 *
 * **Fix verified**: release_host_weight_data() now calls unpinHostMemory() before
 * release_raw_data() in all 26 tensor type implementations.
 *
 * @see src/v2/tensors/TensorClasses.h - release_host_weight_data() implementations
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include "execution/local_execution/device/DeviceContext.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif

// Test utils
#include "../../../utils/CUDATestUtils.h"

#include <vector>
#include <cstring>
#include <random>

using namespace llaminar2;
using namespace llaminar2::test::cuda;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__HostReleaseAfterGpuUpload : public CUDATestBase
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

    void fillSequential(FP32Tensor *tensor, float start = 0.0f, float step = 1.0f)
    {
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < tensor->numel(); ++i)
        {
            data[i] = start + static_cast<float>(i) * step;
        }
    }
};

// ============================================================================
// Core Regression: Pinned memory corruption
// ============================================================================

/**
 * @brief REGRESSION: release_host_weight_data() after ensureOnDevice() must not
 *        corrupt CUDA state.
 *
 * Before the fix, this sequence would corrupt CUDA memory tracking:
 *   1. ensureOnDevice() → cudaHostRegister() pins host buffer
 *   2. release_host_weight_data() → frees host buffer WITHOUT cudaHostUnregister()
 *   3. Any subsequent cudaMalloc → "resource already mapped" error
 *
 * After the fix, unpinHostMemory() is called before release_raw_data().
 */
TEST_F(Test__HostReleaseAfterGpuUpload, ReleaseHostData_DoesNotCorruptCudaState)
{
    // Create and upload tensor to GPU
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{128, 128});
    fillSequential(tensor.get());

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->isDeviceValid());
    ASSERT_NE(tensor->gpu_data_ptr(), nullptr);

    // Release host data (the operation that used to corrupt CUDA)
    tensor->release_host_weight_data();

    // Verify host data is released
    EXPECT_TRUE(tensor->is_raw_data_released()) << "Host data should be marked as released";

    // GPU data should still be valid after host release
    EXPECT_TRUE(tensor->isDeviceValid()) << "GPU data should remain valid after host release";
    EXPECT_NE(tensor->gpu_data_ptr(), nullptr) << "GPU pointer should still be non-null";

    // CRITICAL: Subsequent CUDA operations must succeed.
    // Before the fix, this would fail with "resource already mapped"
#ifdef HAVE_CUDA
    void *test_ptr = nullptr;
    cudaError_t err = cudaMalloc(&test_ptr, 1024);
    EXPECT_EQ(err, cudaSuccess)
        << "cudaMalloc after host release should succeed, but got: "
        << cudaGetErrorString(err);
    if (test_ptr)
    {
        cudaFree(test_ptr);
    }
#endif
}

/**
 * @brief Multiple tensors: releasing host data for several GPU-uploaded tensors
 *        must not cause cumulative CUDA corruption.
 */
TEST_F(Test__HostReleaseAfterGpuUpload, MultipleRelease_NoCumulativeCorruption)
{
    constexpr int NUM_TENSORS = 5;
    std::vector<std::unique_ptr<FP32Tensor>> tensors;

    // Create and upload multiple tensors
    for (int i = 0; i < NUM_TENSORS; ++i)
    {
        auto tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{64, static_cast<size_t>(64 + i * 32)});
        fillSequential(tensor.get(), static_cast<float>(i));
        ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_))
            << "Upload failed for tensor " << i;
        tensors.push_back(std::move(tensor));
    }

    // Release host data for all tensors
    for (int i = 0; i < NUM_TENSORS; ++i)
    {
        tensors[i]->release_host_weight_data();
        EXPECT_TRUE(tensors[i]->is_raw_data_released())
            << "Tensor " << i << " should be released";
        EXPECT_TRUE(tensors[i]->isDeviceValid())
            << "Tensor " << i << " GPU data should remain valid";
    }

    // CRITICAL: CUDA state should be clean after releasing all
#ifdef HAVE_CUDA
    void *test_ptr = nullptr;
    cudaError_t err = cudaMalloc(&test_ptr, 4096);
    EXPECT_EQ(err, cudaSuccess)
        << "cudaMalloc after multiple host releases should succeed, but got: "
        << cudaGetErrorString(err);
    if (test_ptr)
    {
        cudaFree(test_ptr);
    }
#endif
}

/**
 * @brief After host release, GPU pointer remains valid and D2H fails gracefully.
 *
 * release_host_weight_data() swaps the host AlignedVector with an empty one,
 * so raw_host_data_ptr() returns nullptr. Subsequent ensureOnHost() cannot
 * download because there's no host buffer to write into. This test verifies:
 * 1. GPU pointer and validity flag survive host release
 * 2. ensureOnHost() returns false (no host buffer) rather than corrupting state
 * 3. CUDA operations remain functional after the failed ensureOnHost()
 */
TEST_F(Test__HostReleaseAfterGpuUpload, GpuPointerSurvives_EnsureOnHostFailsGracefully)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get(), 0.0f, 1.0f);

    // Upload to GPU, then mark device authoritative (host stale)
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_FALSE(tensor->isOnCPU()) << "Host should be stale after mark_device_dirty";
    ASSERT_TRUE(tensor->isDeviceValid());

    // Release host data — frees the host_data_ AlignedVector
    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());

    // GPU data must survive host release
    EXPECT_TRUE(tensor->isDeviceValid()) << "GPU validity flag must persist";
    EXPECT_NE(tensor->gpu_data_ptr(), nullptr) << "GPU pointer must persist";

    // ensureOnHost() should fail gracefully — host buffer is gone,
    // raw_host_data_ptr() returns nullptr, so D2H has nowhere to write.
    EXPECT_FALSE(tensor->ensureOnHost())
        << "ensureOnHost() must fail when host buffer was released";

    // CUDA must remain operational despite the failed ensureOnHost()
#ifdef HAVE_CUDA
    void *test_ptr = nullptr;
    cudaError_t err = cudaMalloc(&test_ptr, 1024);
    EXPECT_EQ(err, cudaSuccess)
        << "CUDA must be clean after failed ensureOnHost, got: "
        << cudaGetErrorString(err);
    if (test_ptr)
        cudaFree(test_ptr);
#endif
}

/**
 * @brief Releasing host data for a tensor that was never uploaded should be safe.
 *        No CUDA corruption because no pinning ever occurred.
 */
TEST_F(Test__HostReleaseAfterGpuUpload, ReleaseWithoutUpload_SafeNoop)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    // Never uploaded - release should be safe
    EXPECT_FALSE(tensor->isDeviceValid());

    tensor->release_host_weight_data();

    EXPECT_TRUE(tensor->is_raw_data_released());
    EXPECT_FALSE(tensor->isDeviceValid()) << "No GPU data should exist";

    // CUDA state should be clean
#ifdef HAVE_CUDA
    void *test_ptr = nullptr;
    cudaError_t err = cudaMalloc(&test_ptr, 1024);
    EXPECT_EQ(err, cudaSuccess)
        << "cudaMalloc should succeed after releasing non-uploaded tensor";
    if (test_ptr)
    {
        cudaFree(test_ptr);
    }
#endif
}

/**
 * @brief Double release must be safe (idempotent).
 *        Second call should not attempt to unpin already-freed memory.
 */
TEST_F(Test__HostReleaseAfterGpuUpload, DoubleRelease_Idempotent)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // First release
    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());

    // Second release - should not crash or corrupt CUDA
    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());

    // CUDA state should still be clean
#ifdef HAVE_CUDA
    void *test_ptr = nullptr;
    cudaError_t err = cudaMalloc(&test_ptr, 1024);
    EXPECT_EQ(err, cudaSuccess)
        << "cudaMalloc should succeed after double release";
    if (test_ptr)
    {
        cudaFree(test_ptr);
    }
#endif
}

/**
 * @brief Interleaved upload-release pattern: upload tensor A, release A,
 *        upload tensor B, release B - CUDA must not accumulate errors.
 */
TEST_F(Test__HostReleaseAfterGpuUpload, InterleavedUploadRelease_NoCudaErrors)
{
    for (int i = 0; i < 10; ++i)
    {
        auto tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{32, static_cast<size_t>(32 + i * 8)});
        fillSequential(tensor.get(), static_cast<float>(i));

        ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_))
            << "Upload failed at iteration " << i;
        ASSERT_TRUE(tensor->isDeviceValid());

        tensor->release_host_weight_data();
        EXPECT_TRUE(tensor->is_raw_data_released());

        // Tensor goes out of scope - destructor must properly free GPU memory
    }

    // After all tensors are destroyed, CUDA should be clean
#ifdef HAVE_CUDA
    void *test_ptr = nullptr;
    cudaError_t err = cudaMalloc(&test_ptr, 8192);
    EXPECT_EQ(err, cudaSuccess)
        << "cudaMalloc should succeed after interleaved upload-release cycles, got: "
        << cudaGetErrorString(err);
    if (test_ptr)
    {
        cudaFree(test_ptr);
    }
#endif
}

/**
 * @brief Large tensor upload+release to stress-test pinned memory tracking.
 *        Uses a realistically-sized weight tensor (similar to attention projection).
 */
TEST_F(Test__HostReleaseAfterGpuUpload, LargeTensor_NoPinnedMemoryCorruption)
{
    // ~64MB tensor (similar to an attention weight matrix in a 7B model)
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{4096, 4096});
    fillSequential(tensor.get(), 0.0f, 0.0001f);

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->isDeviceValid());

    // Release host data
    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());
    EXPECT_TRUE(tensor->isDeviceValid());

    // Allocate another large chunk - would fail if pinned memory tracking is corrupt
#ifdef HAVE_CUDA
    void *test_ptr = nullptr;
    cudaError_t err = cudaMalloc(&test_ptr, 16 * 1024 * 1024); // 16MB
    EXPECT_EQ(err, cudaSuccess)
        << "Large cudaMalloc should succeed after large tensor host release, got: "
        << cudaGetErrorString(err);
    if (test_ptr)
    {
        cudaFree(test_ptr);
    }
#endif
}

/**
 * @brief Tensor upload → release → re-upload cycle.
 *        After releasing host data and invalidating GPU, re-uploading should
 *        fail (host data is gone) but must NOT corrupt CUDA state.
 */
TEST_F(Test__HostReleaseAfterGpuUpload, ReuploadAfterRelease_GracefulFailure)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    // Upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->isDeviceValid());

    // Modify host to invalidate device (forces re-upload on next ensureOnDevice)
    tensor->mutable_data()[0] = 999.0f;
    EXPECT_FALSE(tensor->isDeviceValid()) << "Device should be stale after host modification";

    // Release host data — now host buffer is empty
    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());

    // Re-upload must fail — raw_host_data_ptr() returns nullptr
    // The critical assertion: CUDA state must NOT be corrupted.
    EXPECT_FALSE(tensor->ensureOnDevice(gpu_device_))
        << "Re-upload must fail when host data has been released";

    // CUDA must still be operational
#ifdef HAVE_CUDA
    void *test_ptr = nullptr;
    cudaError_t err = cudaMalloc(&test_ptr, 1024);
    EXPECT_EQ(err, cudaSuccess)
        << "CUDA state must not be corrupted after failed re-upload, got: "
        << cudaGetErrorString(err);
    if (test_ptr)
    {
        cudaFree(test_ptr);
    }
#endif
}

/**
 * @brief Quantized Q8_0 weight tensor: upload → release → verify CUDA clean.
 *
 * The original bug affected all 26 tensor types. Q8_0 is the primary quantized
 * weight format and exercises a different release_host_weight_data() override
 * that also clears dequant_cache_ and mmap_owner_.
 */
TEST_F(Test__HostReleaseAfterGpuUpload, Q8_0Tensor_ReleaseAfterUpload_NoCudaCorruption)
{
    // Build a Q8_0 tensor with valid block data
    const size_t rows = 64;
    const size_t cols = 256; // Multiple of block_size=32
    const size_t blocks_per_row = cols / 32;
    const size_t total_blocks = rows * blocks_per_row;
    const size_t block_bytes = sizeof(Q8_0Block);

    std::vector<uint8_t> raw_data(total_blocks * block_bytes);
    for (size_t i = 0; i < total_blocks; ++i)
    {
        auto *block = reinterpret_cast<Q8_0Block *>(raw_data.data() + i * block_bytes);
        block->d = 0.1f;
        for (int j = 0; j < 32; ++j)
            block->qs[j] = static_cast<int8_t>(j - 16);
    }

    auto tensor = std::make_unique<Q8_0Tensor>(
        std::vector<size_t>{rows, cols}, raw_data);
    ASSERT_FALSE(tensor->is_raw_data_released());

    // Upload to GPU — this pins host memory via cudaHostRegister
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_))
        << "Q8_0 upload to CUDA failed";
    ASSERT_TRUE(tensor->isDeviceValid());
    ASSERT_NE(tensor->gpu_data_ptr(), nullptr);

    // Release host data (unpin + free blocks + clear dequant cache + mmap owner)
    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());
    EXPECT_TRUE(tensor->isDeviceValid())
        << "Q8_0 GPU data should survive host release";

    // CUDA must be clean — the original bug would cause "resource already mapped" here
#ifdef HAVE_CUDA
    void *test_ptr = nullptr;
    cudaError_t err = cudaMalloc(&test_ptr, 4096);
    EXPECT_EQ(err, cudaSuccess)
        << "cudaMalloc after Q8_0 host release must succeed, got: "
        << cudaGetErrorString(err);
    if (test_ptr)
        cudaFree(test_ptr);
#endif
}
