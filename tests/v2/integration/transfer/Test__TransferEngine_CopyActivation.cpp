/**
 * @file Test__TransferEngine_CopyActivation.cpp
 * @brief Integration tests for TransferEngine::copyActivation() on real GPUs.
 *
 * copyActivation() performs a tensor→tensor copy INTO the destination tensor's
 * buffer on a target device, auto-selecting the optimal transport:
 *   - same physical GPU            → device-to-device copy (intra-VRAM)
 *   - same-vendor, different GPU   → peer copy (NCCL/RCCL or peer DMA)
 *   - cross-vendor GPU (CUDA↔ROCm) → host-staged bounce (no direct path)
 *   - source on host                → direct H2D upload
 *
 * These paths require real device backends, so they live in the integration
 * suite (the pure host/guard paths are covered by the unit suite in
 * unit/transfer/Test__TransferEngine). Every GPU-dependent test guards on
 * hardware availability with GTEST_SKIP so the suite passes on CPU-only hosts.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "v2/tensors/TensorClasses.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/collective/BackendRouter.h"
#include "v2/transfer/TransferEngine.h"
#include "v2/transfer/TransferMethod.h"

using namespace llaminar2;

/**
 * @brief Fixture detecting available CUDA/ROCm devices for copyActivation tests.
 *
 * Mirrors the detection strategy of Test__TensorBase_TransferTo: probes the
 * registered backends and device counts so individual tests can skip cleanly
 * when the required hardware (single GPU, two same-vendor GPUs, or both
 * vendors) is not present.
 */
class Test__TransferEngine_CopyActivation : public ::testing::Test
{
protected:
    void SetUp() override
    {
#ifdef HAVE_CUDA
        if (auto *cuda = getCUDABackend(); cuda != nullptr)
        {
            cuda_available_ = true;
            cuda_device_ = DeviceId::cuda(0);
            // Query the backend directly (hipGetDeviceCount/cudaGetDeviceCount) rather
            // than DeviceManager, which would require an explicit initialize() call.
            if (cuda->deviceCount() >= 2)
            {
                cuda_device_1_ = DeviceId::cuda(1);
                multi_cuda_ = true;
            }
        }
#endif
#ifdef HAVE_ROCM
        if (auto *rocm = getROCmBackend(); rocm != nullptr)
        {
            rocm_available_ = true;
            rocm_device_ = DeviceId::rocm(0);
            if (rocm->deviceCount() >= 2)
            {
                rocm_device_1_ = DeviceId::rocm(1);
                multi_rocm_ = true;
            }
        }
#endif

        // Same-vendor peer copies route through the collective backend router
        // (NCCL/RCCL or peer DMA). Initialize the global router so those paths
        // can be exercised when 2+ same-vendor GPUs are present.
        if (multi_cuda_ || multi_rocm_)
        {
            GlobalBackendRouter::initForTests();
        }
    }

    /// @brief Create an FP32 tensor filled with a deterministic ramp pattern.
    std::unique_ptr<FP32Tensor> makePatternTensor()
    {
        auto t = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 128});
        float *data = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
        {
            data[i] = static_cast<float>(i) * 0.001f;
        }
        return t;
    }

    /// @brief Create an FP32 tensor whose host buffer is filled with a poison
    ///        sentinel (-1.0f) that is neither zero nor any ramp value.
    ///
    /// Using a poison value (rather than zeros) makes verifyPattern() impossible
    /// to pass trivially: the test can only succeed if a real device→host
    /// download returns the data that was genuinely transferred into the
    /// destination's *device* buffer. If no transfer occurred, the host buffer
    /// would still hold the sentinel and verification would fail.
    std::unique_ptr<FP32Tensor> makePoisonTensor()
    {
        auto t = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 128});
        float *data = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
        {
            data[i] = -1.0f;
        }
        return t;
    }

    /**
     * @brief Make a tensor authoritative on a GPU with its ramp pattern resident.
     *
     * Uploads the host pattern to the device and marks the device copy as the
     * authoritative source, so copyActivation will read from device memory.
     */
    bool makeGpuResident(FP32Tensor *t, DeviceId device)
    {
        if (!t->ensureOnDevice(device))
        {
            return false;
        }
        t->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        return t->isDeviceAuthoritative(device);
    }

    /// @brief Sync a tensor to host and verify it matches the ramp pattern.
    bool verifyPattern(FP32Tensor *t)
    {
        if (!t->ensureOnHost())
        {
            return false;
        }
        const float *data = t->data();
        for (size_t i = 0; i < t->numel(); ++i)
        {
            if (std::abs(data[i] - static_cast<float>(i) * 0.001f) > 1e-6f)
            {
                return false;
            }
        }
        return true;
    }

    bool cuda_available_ = false;
    bool rocm_available_ = false;
    bool multi_cuda_ = false;
    bool multi_rocm_ = false;

    DeviceId cuda_device_ = DeviceId::cpu();
    DeviceId cuda_device_1_ = DeviceId::cpu();
    DeviceId rocm_device_ = DeviceId::cpu();
    DeviceId rocm_device_1_ = DeviceId::cpu();
};

// =============================================================================
// Host source → GPU (direct H2D)
// =============================================================================

TEST_F(Test__TransferEngine_CopyActivation, HostSource_to_CUDA_H2D)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Source stays host-authoritative; destination is fresh.
    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), cuda_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::HOST_TO_DEVICE);
    EXPECT_TRUE(dst->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

TEST_F(Test__TransferEngine_CopyActivation, HostSource_to_ROCm_H2D)
{
    if (!rocm_available_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), rocm_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::HOST_TO_DEVICE);
    EXPECT_TRUE(dst->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

// =============================================================================
// Same physical GPU → device-to-device copy
// =============================================================================

TEST_F(Test__TransferEngine_CopyActivation, SamePhysicalGPU_CUDA_D2D)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Source resident & authoritative on cuda:0; destination targets cuda:0 too.
    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), cuda_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), cuda_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
    EXPECT_TRUE(dst->isDeviceAuthoritative(cuda_device_));
    // Intra-GPU copy: both src and dst live on the same physical card.
    EXPECT_TRUE(src->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

TEST_F(Test__TransferEngine_CopyActivation, SamePhysicalGPU_ROCm_D2D)
{
    if (!rocm_available_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), rocm_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), rocm_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
    EXPECT_TRUE(dst->isDeviceAuthoritative(rocm_device_));
    // Intra-GPU copy: both src and dst live on the same physical card.
    EXPECT_TRUE(src->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

// =============================================================================
// Same-vendor, different GPU → peer copy (NCCL/RCCL or peer DMA)
// =============================================================================

TEST_F(Test__TransferEngine_CopyActivation, CUDA_to_CUDA_Peer)
{
    if (!multi_cuda_)
    {
        GTEST_SKIP() << "Need 2+ CUDA devices";
    }
    if (GlobalBackendRouter::get() == nullptr)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Distinct physical cards: src on cuda:0, dst on cuda:1.
    ASSERT_NE(cuda_device_.gpu_ordinal(), cuda_device_1_.gpu_ordinal());

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), cuda_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), cuda_device_1_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
    // dst landed on the *other* physical card; src stayed on its own card.
    EXPECT_TRUE(dst->isDeviceAuthoritative(cuda_device_1_));
    EXPECT_FALSE(dst->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(src->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

TEST_F(Test__TransferEngine_CopyActivation, ROCm_to_ROCm_Peer)
{
    if (!multi_rocm_)
    {
        GTEST_SKIP() << "Need 2+ ROCm devices";
    }
    if (GlobalBackendRouter::get() == nullptr)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Distinct physical cards: src on rocm:0, dst on rocm:1.
    ASSERT_NE(rocm_device_.gpu_ordinal(), rocm_device_1_.gpu_ordinal());

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), rocm_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), rocm_device_1_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
    // dst landed on the *other* physical card; src stayed on its own card.
    EXPECT_TRUE(dst->isDeviceAuthoritative(rocm_device_1_));
    EXPECT_FALSE(dst->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(src->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

// =============================================================================
// Cross-vendor (CUDA↔ROCm) → host-staged bounce
// =============================================================================

TEST_F(Test__TransferEngine_CopyActivation, CrossVendor_CUDA_to_ROCm_HostStaged)
{
    if (!cuda_available_ || !rocm_available_)
    {
        GTEST_SKIP() << "Need both CUDA and ROCm devices";
    }

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), cuda_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), rocm_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::HOST_STAGED);
    // Source was on a CUDA card; destination is now authoritative on a ROCm card.
    EXPECT_TRUE(dst->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(src->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

TEST_F(Test__TransferEngine_CopyActivation, CrossVendor_ROCm_to_CUDA_HostStaged)
{
    if (!cuda_available_ || !rocm_available_)
    {
        GTEST_SKIP() << "Need both CUDA and ROCm devices";
    }

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), rocm_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), cuda_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::HOST_STAGED);
    // Source was on a ROCm card; destination is now authoritative on a CUDA card.
    EXPECT_TRUE(dst->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(src->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}
