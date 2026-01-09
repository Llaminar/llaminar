/**
 * @file Test__TensorFactory_Q16BlockSize.cpp
 * @brief Unit tests for TensorFactory Q16_1 variable block size creation
 *
 * Tests the createQ16_1(shape, block_size, device_idx) overload added in Phase 3
 * for supporting models with different head dimensions (64, 128, 192).
 */

#include <gtest/gtest.h>
#include <memory>
#include "v2/tensors/TensorFactory.h"
#include "v2/tensors/Tensors.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/utils/MPIContext.h"
#include "v2/backends/DeviceId.h"

using namespace llaminar2;

class Test__TensorFactory_Q16BlockSize : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
        factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<TensorFactory> factory_;
};

// =============================================================================
// Default Block Size Tests
// =============================================================================

TEST_F(Test__TensorFactory_Q16BlockSize, DefaultBlockSizeIs32)
{
    // Default createQ16_1 should create BLOCK_32 tensor
    auto tensor = factory_->createQ16_1({4, 128}, DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->block_size(), 32);
    EXPECT_EQ(tensor->q16_block_size(), Q16BlockSize::BLOCK_32);
}

// =============================================================================
// Explicit Block Size Creation Tests
// =============================================================================

TEST_F(Test__TensorFactory_Q16BlockSize, CreateBlock64)
{
    auto tensor = factory_->createQ16_1({4, 64}, Q16BlockSize::BLOCK_64, DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->block_size(), 64);
    EXPECT_EQ(tensor->q16_block_size(), Q16BlockSize::BLOCK_64);
    EXPECT_EQ(tensor->blocks_per_row(), 1); // 64 / 64 = 1
}

TEST_F(Test__TensorFactory_Q16BlockSize, CreateBlock128)
{
    auto tensor = factory_->createQ16_1({4, 128}, Q16BlockSize::BLOCK_128, DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->block_size(), 128);
    EXPECT_EQ(tensor->q16_block_size(), Q16BlockSize::BLOCK_128);
    EXPECT_EQ(tensor->blocks_per_row(), 1); // 128 / 128 = 1
}

// =============================================================================
// Device Placement Tests (ensure block_size overload respects device placement)
// =============================================================================

TEST_F(Test__TensorFactory_Q16BlockSize, BlockSizeOverloadRespectsDeviceIdx)
{
    auto tensor = factory_->createQ16_1({4, 128}, Q16BlockSize::BLOCK_128, DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_cpu()) << "Tensor should be on CPU with DeviceId::cpu()";
    EXPECT_EQ(tensor->block_size(), 128);
}

TEST_F(Test__TensorFactory_Q16BlockSize, BlockSizeOverloadDefaultDevice)
{
    auto tensor = factory_->createQ16_1({4, 128}, Q16BlockSize::BLOCK_128);

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->is_on_cpu()) << "Tensor should be on CPU with default device";
    EXPECT_EQ(tensor->block_size(), 128);
}

// =============================================================================
// Memory Allocation Verification
// =============================================================================

TEST_F(Test__TensorFactory_Q16BlockSize, MemorySizesCorrect)
{
    const size_t rows = 8;

    // BLOCK_64: 136 bytes per block, 1 block per row for 64-elem rows
    {
        auto tensor = factory_->createQ16_1({rows, 64}, Q16BlockSize::BLOCK_64, DeviceId::cpu());
        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(tensor->size_bytes(), rows * 1 * sizeof(Q16_1Block_64));
        EXPECT_EQ(tensor->size_bytes(), rows * 136);
    }

    // BLOCK_128: 264 bytes per block, 1 block per row for 128-elem rows
    {
        auto tensor = factory_->createQ16_1({rows, 128}, Q16BlockSize::BLOCK_128, DeviceId::cpu());
        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(tensor->size_bytes(), rows * 1 * sizeof(Q16_1Block_128));
        EXPECT_EQ(tensor->size_bytes(), rows * 264);
    }
}

// =============================================================================
// Real Model Configuration Tests
// =============================================================================

TEST_F(Test__TensorFactory_Q16BlockSize, Qwen2_05B_HeadDim64)
{
    // Qwen2.5-0.5B: head_dim=64, ideal for BLOCK_64
    const size_t batch_size = 1;
    const size_t n_heads = 14;
    const size_t head_dim = 64;

    auto tensor = factory_->createQ16_1(
        {batch_size * n_heads, head_dim},
        Q16BlockSize::BLOCK_64,
        DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->block_size(), 64);
    EXPECT_EQ(tensor->blocks_per_row(), 1); // One block per head
    EXPECT_EQ(tensor->total_blocks(), n_heads);
}

TEST_F(Test__TensorFactory_Q16BlockSize, Llama3_HeadDim128)
{
    // Llama3: head_dim=128, ideal for BLOCK_128
    const size_t batch_size = 1;
    const size_t n_heads = 32;
    const size_t head_dim = 128;

    auto tensor = factory_->createQ16_1(
        {batch_size * n_heads, head_dim},
        Q16BlockSize::BLOCK_128,
        DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->block_size(), 128);
    EXPECT_EQ(tensor->blocks_per_row(), 1); // One block per head
    EXPECT_EQ(tensor->total_blocks(), n_heads);
}

// Note: DeepSeek V3 MLA uses separate NOPE (head_dim=128) + ROPE (head_dim=64)
// tensors with independent scales, not a combined 192-dim block.
// See PROJECT_Q16_INTEGER_ATTENTION_V2.md MLA Architecture section.

// =============================================================================
// Multiple Blocks Per Row Tests
// =============================================================================

TEST_F(Test__TensorFactory_Q16BlockSize, MultipleBlocksPerRow)
{
    const size_t rows = 4;

    // 256 elements with BLOCK_128 = 2 blocks per row
    {
        auto tensor = factory_->createQ16_1({rows, 256}, Q16BlockSize::BLOCK_128, DeviceId::cpu());
        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(tensor->blocks_per_row(), 2);
        EXPECT_EQ(tensor->total_blocks(), rows * 2);
    }
}
