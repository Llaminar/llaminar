/**
 * @file Test__TensorBase_TransferTo.cpp
 * @brief Integration tests for TransferEngine activation movement
 *
 * Tests Phase 2 of GPU-Native Tensor Coherence:
 * - Direct same-vendor GPU-to-GPU transfers
 * - Deliberately host-staged cross-vendor transfers
 * - Precondition validation
 * - Multi-hop transfers
 */

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"
#include "v2/tensors/TensorClasses.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/collective/BackendRouter.h"
#include "../../utils/ScopedGPUStream.h"

using namespace llaminar2;

class Test__TensorBase_TransferTo : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a simple FP32 tensor for testing
        tensor_ = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 128});

        // Initialize with known pattern for verification
        float *data = tensor_->mutable_data();
        for (size_t i = 0; i < tensor_->numel(); ++i)
        {
            data[i] = static_cast<float>(i) * 0.001f;
        }

        // Store original data for verification
        original_data_.assign(data, data + tensor_->numel());

        // Detect available GPU backends
#ifdef HAVE_CUDA
        if (getCUDABackend() != nullptr)
        {
            cuda_available_ = true;
            cuda_device_ = DeviceId::cuda(0);
            cuda_stream_ =
                std::make_unique<llaminar2::test::ScopedGPUStream>(cuda_device_);

            // Check for second CUDA device
            if (DeviceManager::instance().cuda_device_count() >= 2)
            {
                cuda_device_1_ = DeviceId::cuda(1);
                multi_cuda_ = true;
            }
        }
#endif
#ifdef HAVE_ROCM
        if (getROCmBackend() != nullptr)
        {
            rocm_available_ = true;
            rocm_device_ = DeviceId::rocm(0);
            rocm_stream_ =
                std::make_unique<llaminar2::test::ScopedGPUStream>(rocm_device_);

            // Check for second ROCm device
            if (DeviceManager::instance().rocm_device_count() >= 2)
            {
                rocm_device_1_ = DeviceId::rocm(1);
                multi_rocm_ = true;
            }
        }
#endif
    }

    void *streamFor(DeviceId device) const
    {
        if (device.type == DeviceType::CUDA && cuda_stream_)
            return cuda_stream_->get();
        if (device.type == DeviceType::ROCm && rocm_stream_)
            return rocm_stream_->get();
        throw std::runtime_error(
            "No explicit transfer-test stream for requested GPU");
    }

    /**
     * @brief Verify tensor data matches original pattern after transfer
     *
     * Syncs to host and compares element-by-element with tolerance.
     */
    bool verifyData()
    {
        // Sync to host to verify
        if (!tensor_->ensureOnHost())
        {
            return false;
        }
        const float *data = tensor_->data();
        for (size_t i = 0; i < tensor_->numel(); ++i)
        {
            if (std::abs(data[i] - original_data_[i]) > 1e-6f)
            {
                return false;
            }
        }
        return true;
    }

    std::unique_ptr<FP32Tensor> tensor_;
    std::vector<float> original_data_;

    // Device availability flags
    bool cuda_available_ = false;
    bool rocm_available_ = false;
    bool multi_cuda_ = false;
    bool multi_rocm_ = false;

    DeviceId cuda_device_ = DeviceId::cpu();
    DeviceId cuda_device_1_ = DeviceId::cpu();
    DeviceId rocm_device_ = DeviceId::cpu();
    DeviceId rocm_device_1_ = DeviceId::cpu();
    std::unique_ptr<llaminar2::test::ScopedGPUStream> cuda_stream_;
    std::unique_ptr<llaminar2::test::ScopedGPUStream> rocm_stream_;
};

// =============================================================================
// Precondition Tests
// =============================================================================

TEST_F(Test__TensorBase_TransferTo, HostSourceUploads)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    EXPECT_TRUE(tensor_->isHostAuthoritative());

    const auto result = TransferEngine::instance().transferActivation(
        tensor_.get(),
        cuda_device_);
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(verifyData());
}

TEST_F(Test__TensorBase_TransferTo, SynchronizedSourceCanMove)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // An upload leaves matching host and device copies. TransferEngine may
    // select either valid source; callers do not need to invent authority.
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    EXPECT_TRUE(tensor_->isSynced());

    const auto result = TransferEngine::instance().transferActivation(
        tensor_.get(),
        cuda_device_);
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_TRUE(verifyData());
}

TEST_F(Test__TensorBase_TransferTo, SameDevice_NoOp)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Setup: upload and mark dirty
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    TransferEngine::publishCurrentDeviceWrite(tensor_, streamFor(cuda_device_));
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));

    // Transfer to same device should succeed (no-op)
    const auto result = TransferEngine::instance().transferActivation(
        tensor_.get(),
        cuda_device_);
    EXPECT_TRUE(result.success) << result.error;
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));
}

TEST_F(Test__TensorBase_TransferTo, CPUTargetDownloads)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Setup: upload and mark dirty
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    TransferEngine::publishCurrentDeviceWrite(tensor_, streamFor(cuda_device_));

    const auto result = TransferEngine::instance().transferActivation(
        tensor_.get(),
        DeviceId::cpu());
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_TRUE(tensor_->hostValid());
}

// =============================================================================
// CUDA-to-CUDA Tests
// =============================================================================

TEST_F(Test__TensorBase_TransferTo, CUDA_to_CUDA)
{
    if (!multi_cuda_)
    {
        GTEST_SKIP() << "Need 2+ CUDA devices";
    }

    // Setup: upload and mark dirty on source
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    TransferEngine::publishCurrentDeviceWrite(tensor_, streamFor(cuda_device_));
    ASSERT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));

    // Check if GlobalBackendRouter is initialized
    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Transfer
    const auto result = TransferEngine::instance().transferActivation(
        tensor_.get(),
        cuda_device_1_);
    ASSERT_TRUE(result.success) << result.error;

    // Verify state
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_1_));
    EXPECT_FALSE(tensor_->isDeviceAuthoritative(cuda_device_));
    EXPECT_FALSE(tensor_->isHostAuthoritative());

    // Verify data integrity
    EXPECT_TRUE(verifyData());
}

// =============================================================================
// ROCm-to-ROCm Tests
// =============================================================================

TEST_F(Test__TensorBase_TransferTo, ROCm_to_ROCm)
{
    if (!multi_rocm_)
    {
        GTEST_SKIP() << "Need 2+ ROCm devices";
    }

    // Setup: upload and mark dirty on source
    ASSERT_TRUE(tensor_->ensureOnDevice(rocm_device_));
    TransferEngine::publishCurrentDeviceWrite(tensor_, streamFor(rocm_device_));

    // Check if GlobalBackendRouter is initialized
    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Transfer
    const auto result = TransferEngine::instance().transferActivation(
        tensor_.get(),
        rocm_device_1_);
    ASSERT_TRUE(result.success) << result.error;

    // Verify state
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(rocm_device_1_));
    EXPECT_TRUE(verifyData());
}

// =============================================================================
// Cross-Vendor Tests
// =============================================================================

TEST_F(Test__TensorBase_TransferTo, CUDA_to_ROCm)
{
    if (!cuda_available_ || !rocm_available_)
    {
        GTEST_SKIP() << "Need both CUDA and ROCm";
    }

    // Check if GlobalBackendRouter is initialized
    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Setup: upload and mark dirty on CUDA
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    TransferEngine::publishCurrentDeviceWrite(tensor_, streamFor(cuda_device_));

    // Transfer CUDA -> ROCm
    const auto result = TransferEngine::instance().transferActivation(
        tensor_.get(),
        rocm_device_);
    ASSERT_TRUE(result.success) << result.error;

    // Verify state
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(verifyData());
}

TEST_F(Test__TensorBase_TransferTo, ROCm_to_CUDA)
{
    if (!cuda_available_ || !rocm_available_)
    {
        GTEST_SKIP() << "Need both CUDA and ROCm";
    }

    // Check if GlobalBackendRouter is initialized
    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Setup: upload and mark dirty on ROCm
    ASSERT_TRUE(tensor_->ensureOnDevice(rocm_device_));
    TransferEngine::publishCurrentDeviceWrite(tensor_, streamFor(rocm_device_));

    // Transfer ROCm -> CUDA
    const auto result = TransferEngine::instance().transferActivation(
        tensor_.get(),
        cuda_device_);
    ASSERT_TRUE(result.success) << result.error;

    // Verify state
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(verifyData());
}

// =============================================================================
// Multi-Hop Tests
// =============================================================================

TEST_F(Test__TensorBase_TransferTo, MultiHop_CUDA_ROCm_CUDA)
{
    if (!cuda_available_ || !rocm_available_)
    {
        GTEST_SKIP() << "Need both CUDA and ROCm";
    }

    // Check if GlobalBackendRouter is initialized
    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // CUDA -> ROCm
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    TransferEngine::publishCurrentDeviceWrite(tensor_, streamFor(cuda_device_));
    const auto cuda_to_rocm =
        TransferEngine::instance().transferActivation(
            tensor_.get(),
            rocm_device_);
    ASSERT_TRUE(cuda_to_rocm.success) << cuda_to_rocm.error;
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(rocm_device_));

    // ROCm -> CUDA (round-trip)
    const auto rocm_to_cuda =
        TransferEngine::instance().transferActivation(
            tensor_.get(),
            cuda_device_);
    ASSERT_TRUE(rocm_to_cuda.success) << rocm_to_cuda.error;
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));

    // Verify data survived round-trip
    EXPECT_TRUE(verifyData());
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
