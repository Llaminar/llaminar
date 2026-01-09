/**
 * @file Test__TensorFactory_DeviceIndex.cpp
 * @brief Unit tests for TensorFactory activation tensor device placement
 * @author David Sanftenberg
 *
 * Tests that TensorFactory correctly assigns device placement to activation tensors.
 *
 * With the DeviceId refactor:
 * - Legacy device_idx values -1 and 0 both map to CPU (DeviceId::cpu())
 * - device_idx >= 1 maps to GPU (DeviceId::cuda(device_idx - 1))
 * - Tests now use is_on_cpu()/is_on_gpu() for CPU/GPU checks
 * - Tests use home_device() for type-safe device identification
 */

#include <gtest/gtest.h>
#include <memory>
#include "v2/tensors/TensorFactory.h"
#include "v2/tensors/Tensors.h"
#include "v2/utils/MPIContext.h"
#include "v2/execution/RuntimeConfig.h"
#include "v2/backends/DeviceId.h"

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
// Test: Q8_1 activation tensors respect device placement
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, Q8_1_ActivationHasCorrectDeviceIndex_Device0)
{
    // Create Q8_1 activation tensor on CPU
    // With DeviceId refactor: use DeviceId::cpu() directly
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::Q8_1,
        DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_cpu()) << "Q8_1 tensor should be on CPU when created with device_idx=0";
    EXPECT_EQ(tensor->home_device(), DeviceId::cpu()) << "Q8_1 tensor should have DeviceId::cpu()";
    EXPECT_EQ(tensor->native_type(), TensorType::Q8_1);
}

TEST_F(Test__TensorFactory_DeviceIndex, Q8_1_ActivationHasCorrectDeviceIndex_DeviceMinus1)
{
    // Create Q8_1 activation tensor with default device (CPU)
    // With DeviceId refactor: use DeviceId::cpu() directly
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::Q8_1,
        DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_cpu()) << "Q8_1 tensor should be on CPU when created with device_idx=-1";
    EXPECT_EQ(tensor->home_device(), DeviceId::cpu());
}

TEST_F(Test__TensorFactory_DeviceIndex, Q8_1_ActivationHasCorrectDeviceIndex_Device1)
{
    // Create Q8_1 activation tensor on GPU:0
    // With DeviceId refactor: use DeviceId::cuda(0) directly
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::Q8_1,
        DeviceId::cuda(0));

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_gpu()) << "Q8_1 tensor should be on GPU when created with device_idx=1";
    EXPECT_EQ(tensor->home_device(), DeviceId::cuda(0)) << "Q8_1 tensor should have DeviceId::cuda(0)";
}

// =============================================================================
// Test: BF16 activation tensors respect device placement
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, BF16_ActivationHasCorrectDeviceIndex_Device0)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::BF16,
        DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_cpu()) << "BF16 tensor should be on CPU when created with device_idx=0";
    EXPECT_EQ(tensor->home_device(), DeviceId::cpu());
    EXPECT_EQ(tensor->native_type(), TensorType::BF16);
}

TEST_F(Test__TensorFactory_DeviceIndex, BF16_ActivationHasCorrectDeviceIndex_DeviceMinus1)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::BF16,
        DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_cpu()) << "BF16 tensor should be on CPU when created with device_idx=-1";
    EXPECT_EQ(tensor->home_device(), DeviceId::cpu());
}

// =============================================================================
// Test: FP16 activation tensors respect device placement
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, FP16_ActivationHasCorrectDeviceIndex_Device0)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::FP16,
        DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_cpu()) << "FP16 tensor should be on CPU when created with device_idx=0";
    EXPECT_EQ(tensor->home_device(), DeviceId::cpu());
    EXPECT_EQ(tensor->native_type(), TensorType::FP16);
}

TEST_F(Test__TensorFactory_DeviceIndex, FP16_ActivationHasCorrectDeviceIndex_DeviceMinus1)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::FP16,
        DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_cpu()) << "FP16 tensor should be on CPU when created with device_idx=-1";
    EXPECT_EQ(tensor->home_device(), DeviceId::cpu());
}

// =============================================================================
// Test: FP32 activation tensors (baseline)
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, FP32_ActivationHasCorrectDeviceIndex_Device0)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::FP32,
        DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_cpu()) << "FP32 tensor should be on CPU when created with device_idx=0";
    EXPECT_EQ(tensor->home_device(), DeviceId::cpu());
    EXPECT_EQ(tensor->native_type(), TensorType::FP32);
}

TEST_F(Test__TensorFactory_DeviceIndex, FP32_ActivationHasCorrectDeviceIndex_DeviceMinus1)
{
    auto tensor = factory_->createActivation(
        {32, 896},
        ActivationPrecision::FP32,
        DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_cpu()) << "FP32 tensor should be on CPU when created with device_idx=-1";
    EXPECT_EQ(tensor->home_device(), DeviceId::cpu());
}

// =============================================================================
// Test: Device consistency across tensor types
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, AllPrecisions_ConsistentDeviceIndex)
{
    // Verify all precision types get consistent device placement
    // This is critical for heterogeneous pipelines that route activations between devices

    auto fp32 = factory_->createActivation({32, 896}, ActivationPrecision::FP32, DeviceId::cpu());
    auto bf16 = factory_->createActivation({32, 896}, ActivationPrecision::BF16, DeviceId::cpu());
    auto fp16 = factory_->createActivation({32, 896}, ActivationPrecision::FP16, DeviceId::cpu());
    auto q8_1 = factory_->createActivation({32, 896}, ActivationPrecision::Q8_1, DeviceId::cpu());

    ASSERT_NE(fp32, nullptr);
    ASSERT_NE(bf16, nullptr);
    ASSERT_NE(fp16, nullptr);
    ASSERT_NE(q8_1, nullptr);

    // All should be on CPU (DeviceId::cpu())
    EXPECT_TRUE(fp32->is_on_cpu());
    EXPECT_TRUE(bf16->is_on_cpu());
    EXPECT_TRUE(fp16->is_on_cpu());
    EXPECT_TRUE(q8_1->is_on_cpu());

    // All should have identical DeviceId
    EXPECT_EQ(fp32->home_device(), bf16->home_device());
    EXPECT_EQ(bf16->home_device(), fp16->home_device());
    EXPECT_EQ(fp16->home_device(), q8_1->home_device());
}

// =============================================================================
// Test: Regression test - CPU tensors don't need device transfer
// =============================================================================

TEST_F(Test__TensorFactory_DeviceIndex, RegressionTest_Q8_1_NoSpuriousTransfer)
{
    // This test captures the original bug scenario:
    // 1. Pipeline creates Q8_1 activation with device_idx=0 (CPU)
    // 2. prepareActivationForDevice checks if tensors need transfer
    // 3. With DeviceId: all CPU tensors have is_on_cpu()=true, no spurious transfers

    auto current_hidden = factory_->createActivation(
        {32, 896},
        ActivationPrecision::Q8_1,
        DeviceId::cpu());

    ASSERT_NE(current_hidden, nullptr);

    // The key check: tensor should be on CPU
    EXPECT_TRUE(current_hidden->is_on_cpu())
        << "Q8_1 activation tensor should be on CPU when created with DeviceId::cpu(). "
        << "This prevents spurious device transfers.";

    // Using DeviceId for explicit comparison
    EXPECT_EQ(current_hidden->home_device(), DeviceId::cpu());
}

TEST_F(Test__TensorFactory_DeviceIndex, RegressionTest_MultipleAllocations_ConsistentDevice)
{
    // Test that multiple allocations are consistently on CPU

    for (int i = 0; i < 5; ++i)
    {
        auto tensor = factory_->createActivation(
            {static_cast<size_t>(32 + i * 10), 896},
            ActivationPrecision::Q8_1,
            DeviceId::cpu());

        ASSERT_NE(tensor, nullptr) << "Allocation " << i << " failed";
        EXPECT_TRUE(tensor->is_on_cpu())
            << "Allocation " << i << " should be on CPU";
        EXPECT_EQ(tensor->home_device(), DeviceId::cpu())
            << "Allocation " << i << " has wrong device";
    }
}
