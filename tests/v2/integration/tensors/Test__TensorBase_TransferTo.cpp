/**
 * @file Test__TensorBase_TransferTo.cpp
 * @brief Unit tests for TensorBase::transferTo() direct GPU-to-GPU transfer
 *
 * Tests Phase 2 of GPU-Native Tensor Coherence:
 * - transferTo() for direct GPU-to-GPU transfers
 * - copyTo() for non-authoritative copy operations
 * - Precondition validation
 * - Cross-vendor transfers (CUDA↔ROCm)
 * - Multi-hop transfers
 */

#include <gtest/gtest.h>
#include "v2/tensors/TensorClasses.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/collective/BackendRouter.h"

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

            // Check for second ROCm device
            if (DeviceManager::instance().rocm_device_count() >= 2)
            {
                rocm_device_1_ = DeviceId::rocm(1);
                multi_rocm_ = true;
            }
        }
#endif
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
};

// =============================================================================
// Precondition Tests
// =============================================================================

TEST_F(Test__TensorBase_TransferTo, NotOnGPU_Fails)
{
    // Tensor is on host, not GPU - should fail
    EXPECT_TRUE(tensor_->isHostAuthoritative());

    DeviceId target = DeviceId::cuda(0);
    EXPECT_FALSE(tensor_->transferTo(target));
}

TEST_F(Test__TensorBase_TransferTo, NotDeviceAuthoritative_Fails)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Upload to GPU but don't mark dirty (host still authoritative)
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    EXPECT_TRUE(tensor_->isHostAuthoritative());

    // Transfer should fail because device isn't authoritative
    DeviceId target = DeviceId::cuda(1);
    EXPECT_FALSE(tensor_->transferTo(target));
}

TEST_F(Test__TensorBase_TransferTo, SameDevice_NoOp)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Setup: upload and mark dirty
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    tensor_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));

    // Transfer to same device should succeed (no-op)
    EXPECT_TRUE(tensor_->transferTo(cuda_device_));
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));
}

TEST_F(Test__TensorBase_TransferTo, CPUTarget_Fails)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Setup: upload and mark dirty
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    tensor_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Transfer to CPU should fail (use ensureOnHost instead)
    EXPECT_FALSE(tensor_->transferTo(DeviceId::cpu()));
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
    tensor_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));

    // Check if GlobalBackendRouter is initialized
    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Transfer
    ASSERT_TRUE(tensor_->transferTo(cuda_device_1_));

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
    tensor_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Check if GlobalBackendRouter is initialized
    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Transfer
    ASSERT_TRUE(tensor_->transferTo(rocm_device_1_));

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

    // Check if HOST backend supports this transfer
    auto *backend = router->getBackendForCopy(cuda_device_, rocm_device_);
    if (!backend || !backend->supportsCopy(cuda_device_, rocm_device_))
    {
        GTEST_SKIP() << "HOST backend doesn't support CUDA->ROCm";
    }

    // Setup: upload and mark dirty on CUDA
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    tensor_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Transfer CUDA -> ROCm
    ASSERT_TRUE(tensor_->transferTo(rocm_device_));

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

    // Check if HOST backend supports this transfer
    auto *backend = router->getBackendForCopy(rocm_device_, cuda_device_);
    if (!backend || !backend->supportsCopy(rocm_device_, cuda_device_))
    {
        GTEST_SKIP() << "HOST backend doesn't support ROCm->CUDA";
    }

    // Setup: upload and mark dirty on ROCm
    ASSERT_TRUE(tensor_->ensureOnDevice(rocm_device_));
    tensor_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Transfer ROCm -> CUDA
    ASSERT_TRUE(tensor_->transferTo(cuda_device_));

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

    // Check if HOST backend supports both directions
    auto *backend1 = router->getBackendForCopy(cuda_device_, rocm_device_);
    auto *backend2 = router->getBackendForCopy(rocm_device_, cuda_device_);
    if (!backend1 || !backend2 ||
        !backend1->supportsCopy(cuda_device_, rocm_device_) ||
        !backend2->supportsCopy(rocm_device_, cuda_device_))
    {
        GTEST_SKIP() << "HOST backend doesn't support bidirectional CUDA<->ROCm";
    }

    // CUDA -> ROCm
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    tensor_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_TRUE(tensor_->transferTo(rocm_device_));
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(rocm_device_));

    // ROCm -> CUDA (round-trip)
    ASSERT_TRUE(tensor_->transferTo(cuda_device_));
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));

    // Verify data survived round-trip
    EXPECT_TRUE(verifyData());
}

// =============================================================================
// copyTo() Tests (Non-Authoritative Copy)
// =============================================================================

TEST_F(Test__TensorBase_TransferTo, CopyTo_KeepsSourceAuthoritative)
{
    if (!multi_cuda_)
    {
        GTEST_SKIP() << "Need 2+ CUDA devices";
    }

    // Check if GlobalBackendRouter is initialized
    auto *router = GlobalBackendRouter::get();
    if (!router)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Setup: upload and mark dirty on source
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    tensor_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));

    // Copy to second device
    ASSERT_TRUE(tensor_->copyTo(cuda_device_1_));

    // Source should STILL be authoritative
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));
    EXPECT_FALSE(tensor_->isDeviceAuthoritative(cuda_device_1_));
}

TEST_F(Test__TensorBase_TransferTo, CopyTo_NotAuthoritative_Fails)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Tensor is on host (not device authoritative)
    EXPECT_TRUE(tensor_->isHostAuthoritative());

    // copyTo should fail
    EXPECT_FALSE(tensor_->copyTo(DeviceId::cuda(1)));
}

// =============================================================================
// Host-Only Tests (API Surface Verification)
// =============================================================================

TEST_F(Test__TensorBase_TransferTo, API_Surface_TransferTo_Exists)
{
    // Verify the API compiles - tensor is on host, so this should fail cleanly
    DeviceId target = DeviceId::cuda(0);
    bool result = tensor_->transferTo(target);
    EXPECT_FALSE(result); // Should fail - not on GPU
}

TEST_F(Test__TensorBase_TransferTo, API_Surface_CopyTo_Exists)
{
    // Verify the API compiles - tensor is on host, so this should fail cleanly
    DeviceId target = DeviceId::cuda(0);
    bool result = tensor_->copyTo(target);
    EXPECT_FALSE(result); // Should fail - not on GPU
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
