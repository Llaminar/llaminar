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
#include "v2/execution/config/RuntimeConfig.h"
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

    std::shared_ptr<IMPIContext> mpi_ctx_;
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

/**
 * @brief Prove that every native weight codebook can adopt a caller-owned byte buffer.
 *
 * Expert-selection loading already has the exact packed bytes in a temporary
 * vector. Pointer identity is the observable contract that the factory moves
 * that allocation into the tensor instead of copying tens of gigabytes of
 * model data a second time.
 */
TEST_F(Test__TensorFactory_DeviceIndex, AllNativeWeightCodebooksAdoptRvalueStorage)
{
    struct QuantizedCase
    {
        TensorType type;
        size_t elements;
        size_t bytes;
    };

    const std::vector<QuantizedCase> cases = {
        {TensorType::IQ4_NL, IQ4_NLBlock::BLOCK_SIZE, sizeof(IQ4_NLBlock)},
        {TensorType::Q8_0, Q8_0Block::BLOCK_SIZE, sizeof(Q8_0Block)},
        {TensorType::Q4_0, Q4_0Block::BLOCK_SIZE, sizeof(Q4_0Block)},
        {TensorType::Q4_1, Q4_1Block::BLOCK_SIZE, sizeof(Q4_1Block)},
        {TensorType::Q5_0, Q5_0Block::BLOCK_SIZE, sizeof(Q5_0Block)},
        {TensorType::Q5_1, Q5_1Block::BLOCK_SIZE, sizeof(Q5_1Block)},
        {TensorType::Q6_K, Q6_KBlock::BLOCK_SIZE, sizeof(Q6_KBlock)},
        {TensorType::Q2_K, Q2_KBlock::BLOCK_SIZE, sizeof(Q2_KBlock)},
        {TensorType::Q5_K, Q5_KBlock::BLOCK_SIZE, sizeof(Q5_KBlock)},
        {TensorType::Q3_K, Q3_KBlock::BLOCK_SIZE, sizeof(Q3_KBlock)},
        {TensorType::Q4_K, Q4_KBlock::BLOCK_SIZE, sizeof(Q4_KBlock)},
        {TensorType::Q8_K, Q8_KBlock::BLOCK_SIZE, sizeof(Q8_KBlock)},
        {TensorType::IQ4_XS, IQ4_XSBlock::BLOCK_SIZE, sizeof(IQ4_XSBlock)},
        {TensorType::IQ2_XXS, IQ2_XXSBlock::BLOCK_SIZE, sizeof(IQ2_XXSBlock)},
        {TensorType::IQ2_XS, IQ2_XSBlock::BLOCK_SIZE, sizeof(IQ2_XSBlock)},
        {TensorType::IQ3_XXS, IQ3_XXSBlock::BLOCK_SIZE, sizeof(IQ3_XXSBlock)},
        {TensorType::IQ2_S, IQ2_SBlock::BLOCK_SIZE, sizeof(IQ2_SBlock)},
        {TensorType::IQ3_S, IQ3_SBlock::BLOCK_SIZE, sizeof(IQ3_SBlock)},
        {TensorType::IQ1_S, IQ1_SBlock::BLOCK_SIZE, sizeof(IQ1_SBlock)},
        {TensorType::IQ1_M, IQ1_MBlock::BLOCK_SIZE, sizeof(IQ1_MBlock)},
    };

    for (const QuantizedCase &quantized : cases)
    {
        SCOPED_TRACE(static_cast<int>(quantized.type));
        AlignedVector<uint8_t> source;
        source.resize_uninitialized(quantized.bytes);
        std::fill(source.begin(), source.end(), uint8_t{0x5a});
        const uint8_t *const allocation = source.data();

        std::unique_ptr<TensorBase> tensor = factory_->createQuantizedOwned(
            quantized.type,
            {1u, quantized.elements},
            std::move(source));

        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(tensor->native_type(), quantized.type);
        EXPECT_EQ(tensor->raw_data(), allocation);
        EXPECT_EQ(tensor->size_bytes(), quantized.bytes);
    }
}

/**
 * @brief Prove that native floating-point weight storage is adopted in place.
 */
TEST_F(Test__TensorFactory_DeviceIndex, FloatingPointWeightsAdoptAlignedStorage)
{
    AlignedVector<float> fp32_data;
    fp32_data.resize_uninitialized(32u);
    std::fill(fp32_data.begin(), fp32_data.end(), 1.25f);
    const float *const fp32_allocation = fp32_data.data();
    std::unique_ptr<FP32Tensor> fp32 = factory_->createFP32Owned(
        {32u}, std::move(fp32_data));
    ASSERT_NE(fp32, nullptr);
    EXPECT_EQ(fp32->raw_data(), fp32_allocation);
    EXPECT_EQ(fp32->size_bytes(), 32u * sizeof(float));

    AlignedVector<uint16_t> fp16_data;
    fp16_data.resize_uninitialized(32u);
    std::fill(fp16_data.begin(), fp16_data.end(), uint16_t{0x3c00});
    const uint16_t *const fp16_allocation = fp16_data.data();
    std::unique_ptr<FP16Tensor> fp16 = factory_->createFP16Owned(
        {32u}, std::move(fp16_data));
    ASSERT_NE(fp16, nullptr);
    EXPECT_EQ(fp16->raw_data(), fp16_allocation);
    EXPECT_EQ(fp16->size_bytes(), 32u * sizeof(uint16_t));

    AlignedVector<uint16_t> bf16_data;
    bf16_data.resize_uninitialized(32u);
    std::fill(bf16_data.begin(), bf16_data.end(), uint16_t{0x3f80});
    const uint16_t *const bf16_allocation = bf16_data.data();
    std::unique_ptr<BF16Tensor> bf16 = factory_->createBF16Owned(
        {32u}, std::move(bf16_data));
    ASSERT_NE(bf16, nullptr);
    EXPECT_EQ(bf16->raw_data(), bf16_allocation);
    EXPECT_EQ(bf16->size_bytes(), 32u * sizeof(uint16_t));
}
