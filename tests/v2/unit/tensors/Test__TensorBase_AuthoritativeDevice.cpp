/**
 * @file Test__TensorBase_AuthoritativeDevice.cpp
 * @brief Unit tests for TensorBase authoritative device tracking
 * 
 * Tests the new multi-device coherence API:
 * - getAuthoritativeDevice()
 * - isHostAuthoritative()
 * - isDeviceAuthoritative()
 * 
 * These methods enable direct GPU-to-GPU transfers without host staging.
 */

#include <gtest/gtest.h>
#include "v2/tensors/TensorClasses.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"

using namespace llaminar2;

class Test__TensorBase_AuthoritativeDevice : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple FP32 tensor for testing (32 rows x 64 cols)
        tensor_ = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 64});
        
        // Initialize with known pattern
        float* data = tensor_->mutable_data();
        for (size_t i = 0; i < tensor_->numel(); ++i) {
            data[i] = static_cast<float>(i);
        }
        
        // Check for available GPU backends
#ifdef HAVE_CUDA
        if (getCUDABackend() != nullptr) {
            cuda_available_ = true;
            cuda_device_ = DeviceId::cuda(0);
        }
#endif
#ifdef HAVE_ROCM
        if (getROCmBackend() != nullptr) {
            rocm_available_ = true;
            rocm_device_ = DeviceId::rocm(0);
        }
#endif
    }
    
    std::unique_ptr<FP32Tensor> tensor_;
    bool cuda_available_ = false;
    bool rocm_available_ = false;
    DeviceId cuda_device_ = DeviceId::cpu();
    DeviceId rocm_device_ = DeviceId::cpu();
};

// =============================================================================
// Initial State Tests
// =============================================================================

TEST_F(Test__TensorBase_AuthoritativeDevice, InitialState_HostAuthoritative) {
    // New tensor should have host as authoritative
    EXPECT_TRUE(tensor_->isHostAuthoritative());
    EXPECT_FALSE(tensor_->getAuthoritativeDevice().has_value());
}

TEST_F(Test__TensorBase_AuthoritativeDevice, InitialState_NotDeviceAuthoritative) {
    // New tensor should NOT be device authoritative for any device
    EXPECT_FALSE(tensor_->isDeviceAuthoritative(DeviceId::cpu()));
    EXPECT_FALSE(tensor_->isDeviceAuthoritative(DeviceId::cuda(0)));
    EXPECT_FALSE(tensor_->isDeviceAuthoritative(DeviceId::rocm(0)));
}

// =============================================================================
// mark_device_dirty() Tests
// =============================================================================

TEST_F(Test__TensorBase_AuthoritativeDevice, MarkDeviceDirty_SetsAuthoritative_CUDA) {
    if (!cuda_available_) {
        GTEST_SKIP() << "No CUDA device available";
    }
    
    // Upload to GPU
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    
    // Before mark_device_dirty, host is still authoritative
    EXPECT_TRUE(tensor_->isHostAuthoritative());
    
    // Mark device dirty
    tensor_->mark_device_dirty();
    
    // Now device should be authoritative
    EXPECT_FALSE(tensor_->isHostAuthoritative());
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));
    EXPECT_EQ(*tensor_->getAuthoritativeDevice(), cuda_device_);
}

TEST_F(Test__TensorBase_AuthoritativeDevice, MarkDeviceDirty_SetsAuthoritative_ROCm) {
    if (!rocm_available_) {
        GTEST_SKIP() << "No ROCm device available";
    }
    
    // Upload to GPU
    ASSERT_TRUE(tensor_->ensureOnDevice(rocm_device_));
    
    // Before mark_device_dirty, host is still authoritative
    EXPECT_TRUE(tensor_->isHostAuthoritative());
    
    // Mark device dirty
    tensor_->mark_device_dirty();
    
    // Now device should be authoritative
    EXPECT_FALSE(tensor_->isHostAuthoritative());
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(rocm_device_));
    EXPECT_EQ(*tensor_->getAuthoritativeDevice(), rocm_device_);
}

// =============================================================================
// ensureOnHost() Tests
// =============================================================================

TEST_F(Test__TensorBase_AuthoritativeDevice, EnsureOnHost_ClearsAuthoritative_CUDA) {
    if (!cuda_available_) {
        GTEST_SKIP() << "No CUDA device available";
    }
    
    // Upload and mark dirty
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    tensor_->mark_device_dirty();
    ASSERT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));
    
    // Sync back to host
    ASSERT_TRUE(tensor_->ensureOnHost());
    
    // Host should now be authoritative
    EXPECT_TRUE(tensor_->isHostAuthoritative());
    EXPECT_FALSE(tensor_->getAuthoritativeDevice().has_value());
}

TEST_F(Test__TensorBase_AuthoritativeDevice, EnsureOnHost_ClearsAuthoritative_ROCm) {
    if (!rocm_available_) {
        GTEST_SKIP() << "No ROCm device available";
    }
    
    // Upload and mark dirty
    ASSERT_TRUE(tensor_->ensureOnDevice(rocm_device_));
    tensor_->mark_device_dirty();
    ASSERT_TRUE(tensor_->isDeviceAuthoritative(rocm_device_));
    
    // Sync back to host
    ASSERT_TRUE(tensor_->ensureOnHost());
    
    // Host should now be authoritative
    EXPECT_TRUE(tensor_->isHostAuthoritative());
    EXPECT_FALSE(tensor_->getAuthoritativeDevice().has_value());
}

// =============================================================================
// ensureOnDevice() Tests (Should NOT change authoritative)
// =============================================================================

TEST_F(Test__TensorBase_AuthoritativeDevice, EnsureOnDevice_DoesNotChangeAuthoritative) {
    if (!cuda_available_) {
        GTEST_SKIP() << "No CUDA device available";
    }
    
    // Initially host authoritative
    EXPECT_TRUE(tensor_->isHostAuthoritative());
    
    // Upload to GPU (H2D copy)
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    
    // Host should STILL be authoritative (we just made a copy to GPU)
    EXPECT_TRUE(tensor_->isHostAuthoritative());
    EXPECT_FALSE(tensor_->isDeviceAuthoritative(cuda_device_));
}

// =============================================================================
// Cross-Vendor Tests
// =============================================================================

TEST_F(Test__TensorBase_AuthoritativeDevice, CrossVendor_AuthoritativeTracking) {
    if (!cuda_available_ || !rocm_available_) {
        GTEST_SKIP() << "Need both CUDA and ROCm for cross-vendor test";
    }
    
    // Start: host authoritative
    EXPECT_TRUE(tensor_->isHostAuthoritative());
    
    // Upload to CUDA
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    EXPECT_TRUE(tensor_->isHostAuthoritative());  // Still host
    
    // Mark CUDA dirty
    tensor_->mark_device_dirty();
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));
    EXPECT_FALSE(tensor_->isDeviceAuthoritative(rocm_device_));
    
    // Sync to host
    ASSERT_TRUE(tensor_->ensureOnHost());
    EXPECT_TRUE(tensor_->isHostAuthoritative());
    
    // Upload to ROCm
    ASSERT_TRUE(tensor_->ensureOnDevice(rocm_device_));
    EXPECT_TRUE(tensor_->isHostAuthoritative());  // Still host
    
    // Mark ROCm dirty
    tensor_->mark_device_dirty();
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(rocm_device_));
    EXPECT_FALSE(tensor_->isDeviceAuthoritative(cuda_device_));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__TensorBase_AuthoritativeDevice, IsDeviceAuthoritative_WrongDevice_ReturnsFalse) {
    if (!cuda_available_) {
        GTEST_SKIP() << "No CUDA device available";
    }
    
    // Upload and mark dirty on CUDA:0
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    tensor_->mark_device_dirty();
    
    // Check for different device
    DeviceId other_cuda = DeviceId::cuda(99);  // Non-existent
    EXPECT_FALSE(tensor_->isDeviceAuthoritative(other_cuda));
    EXPECT_FALSE(tensor_->isDeviceAuthoritative(DeviceId::rocm(0)));
}

TEST_F(Test__TensorBase_AuthoritativeDevice, MultipleMarkDirty_LastWins) {
    if (!cuda_available_) {
        GTEST_SKIP() << "No CUDA device available";
    }
    
    // Upload to CUDA
    ASSERT_TRUE(tensor_->ensureOnDevice(cuda_device_));
    tensor_->mark_device_dirty();
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));
    
    // Mark dirty again (should still be same device)
    tensor_->mark_device_dirty();
    EXPECT_TRUE(tensor_->isDeviceAuthoritative(cuda_device_));
}

// =============================================================================
// Host-Only Tests (No GPU Required)
// =============================================================================

TEST_F(Test__TensorBase_AuthoritativeDevice, HostOnlyTensor_AlwaysHostAuthoritative) {
    // A tensor that's never uploaded to GPU should always be host authoritative
    auto host_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{16, 32});
    
    EXPECT_TRUE(host_tensor->isHostAuthoritative());
    EXPECT_FALSE(host_tensor->getAuthoritativeDevice().has_value());
    
    // Multiple reads shouldn't change authoritative state
    [[maybe_unused]] const float* data = host_tensor->data();
    EXPECT_TRUE(host_tensor->isHostAuthoritative());
    
    // Multiple writes shouldn't change authoritative state
    [[maybe_unused]] float* mdata = host_tensor->mutable_data();
    EXPECT_TRUE(host_tensor->isHostAuthoritative());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
