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
 * **Fix verified**: host-storage release and every concrete tensor destructor
 * retire the exact registration in its owning GPU context before storage can
 * disappear. The exhaustive matrix below covers every loader-visible
 * quantized codebook plus floating, integer, Q16, and TurboQuant storage.
 *
 * @see src/v2/tensors/TensorClasses.h - release_host_weight_data() implementations
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "backends/ComputeBackend.h"
#include "execution/local_execution/device/DeviceContext.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif

// Test utils
#include "../../../utils/CUDATestUtils.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ScopedGPUStream.h"

#include <functional>
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
    using TensorCreator = std::function<std::unique_ptr<TensorBase>()>;

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

#ifdef HAVE_CUDA
    /**
     * @brief Assert that CUDA no longer owns a host registration at an address.
     *
     * CUDA may report ordinary malloc storage either as
     * `cudaMemoryTypeUnregistered` or with `cudaErrorInvalidValue`, depending
     * on runtime version. A live `cudaMemoryTypeHost` result is the defect:
     * freeing that allocation would leave a DMA mapping over recyclable heap
     * pages.
     */
    void expectCudaHostAddressUnregistered(
        const void *address,
        const char *format_label)
    {
        cudaPointerAttributes attributes{};
        const cudaError_t status =
            cudaPointerGetAttributes(&attributes, address);
        if (status == cudaSuccess)
        {
            EXPECT_NE(attributes.type, cudaMemoryTypeHost)
                << format_label << " left address " << address
                << " registered as CUDA host memory";
            return;
        }

        EXPECT_EQ(status, cudaErrorInvalidValue)
            << format_label << " pointer query failed unexpectedly: "
            << cudaGetErrorString(status);
        (void)cudaGetLastError();
    }
#endif
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
    llaminar2::test::ScopedGPUStream producer_stream(gpu_device_);
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_, producer_stream.get()));
    TransferEngine::publishCurrentDeviceWrite(
        tensor,
        producer_stream.get());
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

/**
 * @brief Destroy every concrete host-storage family while another CUDA device
 *        is current and prove no registration survives.
 *
 * The production LocalTP failure was device-order dependent: a tensor was
 * registered by CUDA:0, teardown ran while CUDA:1 was current, and an ignored
 * unregister failure left the freed pages pinned. Prefix-cache vectors later
 * reused a mixture of registered and pageable pages, making an otherwise valid
 * D2H return `cudaErrorInvalidValue`. This sweep reproduces that ownership
 * transition for every model codebook and every non-codebook tensor storage
 * family that can travel through TransferEngine.
 */
TEST_F(Test__HostReleaseAfterGpuUpload, EveryFormatDestructorRetiresOwningDeviceRegistration)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "Built without CUDA support";
#else
    if (device_count_ < 2)
        GTEST_SKIP() << "Cross-device registration retirement requires two CUDA devices";

    std::vector<std::pair<const char *, TensorCreator>> cases;
    for (const auto &format : llaminar2::test::quantizedVerifierFormats())
    {
        cases.emplace_back(
            format.label,
            [creator = format.create]()
            {
                return creator({1u, 256u}, 0x51a7u);
            });
    }

    cases.emplace_back("FP32", []()
                       { return llaminar2::test::TestTensorFactory::createFP32Random({1u, 256u}); });
    cases.emplace_back("FP16", []()
                       { return llaminar2::test::TestTensorFactory::createFP16Random({1u, 256u}); });
    cases.emplace_back("BF16", []()
                       { return llaminar2::test::TestTensorFactory::createBF16Random({1u, 256u}); });
    cases.emplace_back("INT8", []()
                       { return std::make_unique<INT8Tensor>(std::vector<size_t>{1u, 256u}); });
    cases.emplace_back("INT32", []()
                       { return std::make_unique<INT32Tensor>(std::vector<size_t>{1u, 256u}); });
    cases.emplace_back("Q16_1", []()
                       { return llaminar2::test::TestTensorFactory::createQ16_1Random({1u, 256u}); });
    cases.emplace_back("TQ4", []()
                       { return std::make_unique<TQ4Tensor>(std::vector<size_t>{1u, 256u}, 64); });
    cases.emplace_back("TQ8", []()
                       { return std::make_unique<TQ8Tensor>(std::vector<size_t>{1u, 256u}, 64); });

    llaminar2::test::ScopedGPUStream upload_stream(gpu_device_);
    for (const auto &[label, create] : cases)
    {
        SCOPED_TRACE(label);
        std::unique_ptr<TensorBase> tensor = create();
        ASSERT_NE(tensor, nullptr);
        const void *const registered_address = tensor->raw_data();
        ASSERT_NE(registered_address, nullptr);
        ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_, upload_stream.get()));

        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        tensor.reset();
        expectCudaHostAddressUnregistered(registered_address, label);
    }
#endif
}

/**
 * @brief Release every loader-visible weight format while a foreign CUDA
 *        device is current and prove the live tensor no longer owns pinned pages.
 *
 * This complements destructor coverage by exercising the earlier production
 * reclamation boundary used after persistent device weights are prepared.
 */
TEST_F(Test__HostReleaseAfterGpuUpload, EveryWeightFormatReleaseRetiresOwningDeviceRegistration)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "Built without CUDA support";
#else
    if (device_count_ < 2)
        GTEST_SKIP() << "Cross-device registration retirement requires two CUDA devices";

    std::vector<std::pair<const char *, TensorCreator>> cases;
    for (const auto &format : llaminar2::test::quantizedVerifierFormats())
    {
        cases.emplace_back(
            format.label,
            [creator = format.create]()
            {
                return creator({1u, 256u}, 0x6e91u);
            });
    }
    cases.emplace_back("FP32", []()
                       { return llaminar2::test::TestTensorFactory::createFP32Random({1u, 256u}); });
    cases.emplace_back("FP16", []()
                       { return llaminar2::test::TestTensorFactory::createFP16Random({1u, 256u}); });
    cases.emplace_back("BF16", []()
                       { return llaminar2::test::TestTensorFactory::createBF16Random({1u, 256u}); });
    cases.emplace_back("INT8", []()
                       { return std::make_unique<INT8Tensor>(std::vector<size_t>{1u, 256u}); });
    cases.emplace_back("INT32", []()
                       { return std::make_unique<INT32Tensor>(std::vector<size_t>{1u, 256u}); });
    cases.emplace_back("Q16_1", []()
                       { return llaminar2::test::TestTensorFactory::createQ16_1Random({1u, 256u}); });
    cases.emplace_back("TQ4", []()
                       { return std::make_unique<TQ4Tensor>(std::vector<size_t>{1u, 256u}, 64); });
    cases.emplace_back("TQ8", []()
                       { return std::make_unique<TQ8Tensor>(std::vector<size_t>{1u, 256u}, 64); });

    llaminar2::test::ScopedGPUStream upload_stream(gpu_device_);
    for (const auto &[label, create] : cases)
    {
        SCOPED_TRACE(label);
        std::unique_ptr<TensorBase> tensor = create();
        ASSERT_NE(tensor, nullptr);
        const void *const registered_address = tensor->raw_data();
        ASSERT_NE(registered_address, nullptr);
        ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_, upload_stream.get()));

        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        tensor->release_host_weight_data();
        EXPECT_TRUE(tensor->is_raw_data_released());
        expectCudaHostAddressUnregistered(registered_address, label);
    }
#endif
}

/**
 * @brief Reproduce the LocalTP failure through the real TensorSlice delegation.
 *
 * TensorSlice owns no host bytes itself: ensureOnDevice registers its inner
 * tensor. Host release must therefore delegate the entire transition to that
 * same inner owner, including event retirement and unregistration, before the
 * inner FP32 vector is reclaimed.
 */
TEST_F(Test__HostReleaseAfterGpuUpload, TensorSliceReleaseRetiresInnerRegistrationOnOwningDevice)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "Built without CUDA support";
#else
    if (device_count_ < 2)
        GTEST_SKIP() << "Cross-device registration retirement requires two CUDA devices";

    auto inner = llaminar2::test::TestTensorFactory::createFP32Random(
        {8u, 3072u});
    const void *const registered_address = inner->raw_data();
    const SliceMetadata metadata = SliceMetadata::forRowParallel(
        8u, 3072u, 1, 2, true);
    std::unique_ptr<TensorBase> inner_storage = std::move(inner);
    TensorSlice slice(std::move(inner_storage), metadata);
    llaminar2::test::ScopedGPUStream upload_stream(gpu_device_);
    ASSERT_TRUE(slice.ensureOnDevice(gpu_device_, upload_stream.get()));

    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    slice.release_host_weight_data();
    EXPECT_TRUE(slice.is_raw_data_released());
    expectCudaHostAddressUnregistered(registered_address, "TensorSlice<FP32>");
#endif
}
