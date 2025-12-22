/**
 * @file Test__TensorFactory_DeviceIndex.cpp
 * @brief Unit tests for TensorFactory activation tensor device index assignment
 * @author David Sanftenberg
 *
 * Tests the fix for a bug where Q8_1, BF16, and FP16 activation tensors were
 * created with device_index=-1 regardless of the device_idx parameter passed
 * to createActivation(). This caused prepareActivationForDevice() to attempt
 * spurious device transfers even when source and target were both CPU device 0.
 *
 * Bug: https://github.com/llaminar/llaminar/issues/XXX (if applicable)
 * Fix: TensorFactory::createActivation now calls set_device(device_idx) after
 *      creating non-FP32 tensor types.
 */

#include <gtest/gtest.h>
#include <memory>
#include "v2/tensors/TensorFactory.h"
#include "v2/tensors/Tensors.h"
#include "v2/utils/MPIContext.h"
#include "v2/execution/RuntimeConfig.h"

using namespace llaminar2;

class Test__TensorFactory_DeviceIndex : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Mock MPI context (rank 0, size 1)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
        factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<TensorFactory> factory_;
};

// =============================================================================
// Test: Q8_1 activation tensors respect device_idx parameter
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, Q8_1_ActivationHasCorrectDeviceIndex_Device0)
{
    // Create Q8_1 activation tensor on device 0 (CPU)
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::Q8_1,
        0 // device_idx = 0
    );

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->device_index(), 0) << "Q8_1 tensor should have device_index=0 when created with device_idx=0";
    EXPECT_EQ(tensor->native_type(), TensorType::Q8_1);
}

TEST_F(Test__TensorFactory_DeviceIndex, Q8_1_ActivationHasCorrectDeviceIndex_DeviceMinus1)
{
    // Create Q8_1 activation tensor with default device (-1)
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::Q8_1,
        -1 // device_idx = -1 (explicit CPU)
    );

    ASSERT_NE(tensor, nullptr);
    // When device_idx is -1, we should NOT call set_device, so it stays at constructor default
    EXPECT_EQ(tensor->device_index(), -1) << "Q8_1 tensor should have device_index=-1 when created with device_idx=-1";
}

TEST_F(Test__TensorFactory_DeviceIndex, Q8_1_ActivationHasCorrectDeviceIndex_Device1)
{
    // Create Q8_1 activation tensor on device 1 (hypothetical second CPU/GPU)
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::Q8_1,
        1 // device_idx = 1
    );

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->device_index(), 1) << "Q8_1 tensor should have device_index=1 when created with device_idx=1";
}

// =============================================================================
// Test: BF16 activation tensors respect device_idx parameter
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, BF16_ActivationHasCorrectDeviceIndex_Device0)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::BF16,
        0);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->device_index(), 0) << "BF16 tensor should have device_index=0 when created with device_idx=0";
    EXPECT_EQ(tensor->native_type(), TensorType::BF16);
}

TEST_F(Test__TensorFactory_DeviceIndex, BF16_ActivationHasCorrectDeviceIndex_DeviceMinus1)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::BF16,
        -1);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->device_index(), -1) << "BF16 tensor should have device_index=-1 when created with device_idx=-1";
}

// =============================================================================
// Test: FP16 activation tensors respect device_idx parameter
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, FP16_ActivationHasCorrectDeviceIndex_Device0)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::FP16,
        0);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->device_index(), 0) << "FP16 tensor should have device_index=0 when created with device_idx=0";
    EXPECT_EQ(tensor->native_type(), TensorType::FP16);
}

TEST_F(Test__TensorFactory_DeviceIndex, FP16_ActivationHasCorrectDeviceIndex_DeviceMinus1)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::FP16,
        -1);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->device_index(), -1) << "FP16 tensor should have device_index=-1 when created with device_idx=-1";
}

// =============================================================================
// Test: FP32 activation tensors (baseline - should already work)
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, FP32_ActivationHasCorrectDeviceIndex_Device0)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::FP32,
        0);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->device_index(), 0) << "FP32 tensor should have device_index=0 when created with device_idx=0";
    EXPECT_EQ(tensor->native_type(), TensorType::FP32);
}

TEST_F(Test__TensorFactory_DeviceIndex, FP32_ActivationHasCorrectDeviceIndex_DeviceMinus1)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::FP32,
        -1);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->device_index(), -1) << "FP32 tensor should have device_index=-1 when created with device_idx=-1";
}

// =============================================================================
// Test: Device consistency across tensor types
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, AllPrecisions_ConsistentDeviceIndex)
{
    // Verify all precision types get consistent device indices
    // This is critical for heterogeneous pipelines that route activations between devices

    const int target_device = 0;

    auto fp32 = factory_->createActivation({32, 896}, ActivationPrecision::FP32, target_device);
    auto bf16 = factory_->createActivation({32, 896}, ActivationPrecision::BF16, target_device);
    auto fp16 = factory_->createActivation({32, 896}, ActivationPrecision::FP16, target_device);
    auto q8_1 = factory_->createActivation({32, 896}, ActivationPrecision::Q8_1, target_device);

    ASSERT_NE(fp32, nullptr);
    ASSERT_NE(bf16, nullptr);
    ASSERT_NE(fp16, nullptr);
    ASSERT_NE(q8_1, nullptr);

    EXPECT_EQ(fp32->device_index(), target_device);
    EXPECT_EQ(bf16->device_index(), target_device);
    EXPECT_EQ(fp16->device_index(), target_device);
    EXPECT_EQ(q8_1->device_index(), target_device);
}

// =============================================================================
// Test: Regression test for the original bug scenario
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, RegressionTest_Q8_1_NoSpuriousTransfer)
{
    // This test captures the original bug scenario:
    // 1. Pipeline creates Q8_1 activation with device_idx=0
    // 2. prepareActivationForDevice checks if current_device != target_device
    // 3. OLD BUG: Q8_1 tensor had device_index=-1, so -1 != 0 triggered transfer
    // 4. FIX: Q8_1 tensor now has device_index=0, so 0 == 0, no transfer needed

    const int pipeline_device = 0; // Simulates pipeline's device_idx_

    // Create activation as pipeline would
    auto current_hidden = factory_->createActivation(
        {32, 896},
        ActivationPrecision::Q8_1,
        pipeline_device);

    ASSERT_NE(current_hidden, nullptr);

    // Simulate the check in prepareActivationForDevice
    int current_device = current_hidden->device_index();
    int target_device = pipeline_device;

    // With the fix, these should match - no transfer needed
    EXPECT_EQ(current_device, target_device)
        << "Q8_1 activation tensor's device_index should match the device_idx it was created with. "
        << "Mismatch would cause prepareActivationForDevice to attempt spurious transfer.";
}

TEST_F(Test__TensorFactory_DeviceIndex, RegressionTest_MultipleAllocations_ConsistentDevice)
{
    // Test that multiple allocations with the same device_idx are consistent
    // This catches any potential state leakage or inconsistent behavior

    const int target_device = 0;

    for (int i = 0; i < 5; ++i)
    {
        auto tensor = factory_->createActivation(
            {static_cast<size_t>(32 + i * 10), 896},
            ActivationPrecision::Q8_1,
            target_device);

        ASSERT_NE(tensor, nullptr) << "Allocation " << i << " failed";
        EXPECT_EQ(tensor->device_index(), target_device)
            << "Allocation " << i << " has wrong device_index";
    }
}
