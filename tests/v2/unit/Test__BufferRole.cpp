/**
 * @file Test__BufferRole.cpp
 * @brief Unit tests for BufferRole enum and buffer descriptor types
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "execution/BufferRole.h"

using namespace llaminar2;

// =============================================================================
// BufferRole Enum Tests
// =============================================================================

TEST(Test__BufferRole, RoleNameConversion)
{
    EXPECT_STREQ(bufferRoleName(BufferRole::INPUT), "INPUT");
    EXPECT_STREQ(bufferRoleName(BufferRole::OUTPUT), "OUTPUT");
    EXPECT_STREQ(bufferRoleName(BufferRole::INOUT), "INOUT");
    EXPECT_STREQ(bufferRoleName(BufferRole::SCRATCH), "SCRATCH");
    EXPECT_STREQ(bufferRoleName(BufferRole::WEIGHT), "WEIGHT");
}

// =============================================================================
// BufferTensorType Tests
// =============================================================================

TEST(Test__BufferRole, TensorTypeNameConversion)
{
    EXPECT_STREQ(bufferTensorTypeName(BufferTensorType::FP32), "FP32");
    EXPECT_STREQ(bufferTensorTypeName(BufferTensorType::FP16), "FP16");
    EXPECT_STREQ(bufferTensorTypeName(BufferTensorType::BF16), "BF16");
    EXPECT_STREQ(bufferTensorTypeName(BufferTensorType::Q8_1), "Q8_1");
    EXPECT_STREQ(bufferTensorTypeName(BufferTensorType::Q8_0), "Q8_0");
    EXPECT_STREQ(bufferTensorTypeName(BufferTensorType::Q4_0), "Q4_0");
    EXPECT_STREQ(bufferTensorTypeName(BufferTensorType::IQ4_NL), "IQ4_NL");
    EXPECT_STREQ(bufferTensorTypeName(BufferTensorType::INT32), "INT32");
}

TEST(Test__BufferRole, TensorTypeSizes)
{
    EXPECT_EQ(bufferTensorTypeSize(BufferTensorType::FP32), 4);
    EXPECT_EQ(bufferTensorTypeSize(BufferTensorType::FP16), 2);
    EXPECT_EQ(bufferTensorTypeSize(BufferTensorType::BF16), 2);
    EXPECT_EQ(bufferTensorTypeSize(BufferTensorType::INT32), 4);

    // Block-quantized types (block size in bytes)
    EXPECT_EQ(bufferTensorTypeSize(BufferTensorType::Q8_1), 36);
    EXPECT_EQ(bufferTensorTypeSize(BufferTensorType::Q8_0), 34);
    EXPECT_EQ(bufferTensorTypeSize(BufferTensorType::Q4_0), 18);
    EXPECT_EQ(bufferTensorTypeSize(BufferTensorType::IQ4_NL), 18);
}

// =============================================================================
// BufferDescriptor Tests
// =============================================================================

TEST(Test__BufferDescriptor, DefaultConstruction)
{
    BufferDescriptor desc;

    EXPECT_TRUE(desc.name.empty());
    EXPECT_EQ(desc.role, BufferRole::SCRATCH);
    EXPECT_TRUE(desc.shape.empty());
    EXPECT_EQ(desc.tensor_type, BufferTensorType::FP32);
    EXPECT_TRUE(desc.required);
    EXPECT_EQ(desc.alignment, 64);
    EXPECT_EQ(desc.device_idx, -1);
}

TEST(Test__BufferDescriptor, BuilderPatternInput)
{
    auto desc = BufferDescriptor::input("activations", {16, 896}, BufferTensorType::FP32);

    EXPECT_EQ(desc.name, "activations");
    EXPECT_EQ(desc.role, BufferRole::INPUT);
    EXPECT_EQ(desc.shape.size(), 2);
    EXPECT_EQ(desc.shape[0], 16);
    EXPECT_EQ(desc.shape[1], 896);
    EXPECT_EQ(desc.tensor_type, BufferTensorType::FP32);
    EXPECT_TRUE(desc.required);
}

TEST(Test__BufferDescriptor, BuilderPatternOutput)
{
    auto desc = BufferDescriptor::output("logits", {1, 151936});

    EXPECT_EQ(desc.name, "logits");
    EXPECT_EQ(desc.role, BufferRole::OUTPUT);
    EXPECT_EQ(desc.numel(), 151936);
}

TEST(Test__BufferDescriptor, BuilderPatternInout)
{
    auto desc = BufferDescriptor::inout("residual", {32, 4096}, BufferTensorType::Q8_1);

    EXPECT_EQ(desc.name, "residual");
    EXPECT_EQ(desc.role, BufferRole::INOUT);
    EXPECT_EQ(desc.tensor_type, BufferTensorType::Q8_1);
}

TEST(Test__BufferDescriptor, BuilderPatternScratch)
{
    auto desc = BufferDescriptor::scratch("workspace", {1024, 1024});

    EXPECT_EQ(desc.name, "workspace");
    EXPECT_EQ(desc.role, BufferRole::SCRATCH);
    EXPECT_TRUE(desc.isAliasable());
}

TEST(Test__BufferDescriptor, BuilderPatternWeight)
{
    auto desc = BufferDescriptor::weight("gamma", {4096}, BufferTensorType::FP32);

    EXPECT_EQ(desc.name, "gamma");
    EXPECT_EQ(desc.role, BufferRole::WEIGHT);
    EXPECT_TRUE(desc.isReadOnly());
}

TEST(Test__BufferDescriptor, NumelCalculation)
{
    // 2D tensor
    auto desc2d = BufferDescriptor::scratch("test", {16, 896});
    EXPECT_EQ(desc2d.numel(), 16 * 896);

    // 3D tensor
    auto desc3d = BufferDescriptor::scratch("test", {14, 32, 64});
    EXPECT_EQ(desc3d.numel(), 14 * 32 * 64);

    // 1D tensor
    auto desc1d = BufferDescriptor::scratch("test", {4096});
    EXPECT_EQ(desc1d.numel(), 4096);

    // Empty shape
    BufferDescriptor empty;
    EXPECT_EQ(empty.numel(), 0);
}

TEST(Test__BufferDescriptor, SizeBytesCalculationFP32)
{
    auto desc = BufferDescriptor::scratch("test", {16, 896}, BufferTensorType::FP32);

    // FP32: 4 bytes per element
    EXPECT_EQ(desc.sizeBytes(), 16 * 896 * 4);
}

TEST(Test__BufferDescriptor, SizeBytesCalculationFP16)
{
    auto desc = BufferDescriptor::scratch("test", {16, 896}, BufferTensorType::FP16);

    // FP16: 2 bytes per element
    EXPECT_EQ(desc.sizeBytes(), 16 * 896 * 2);
}

TEST(Test__BufferDescriptor, SizeBytesCalculationQ8_1)
{
    // Q8_1: 36 bytes per 32-element block
    auto desc = BufferDescriptor::scratch("test", {32}, BufferTensorType::Q8_1);
    EXPECT_EQ(desc.sizeBytes(), 36); // Exactly 1 block

    // 64 elements = 2 blocks
    auto desc2 = BufferDescriptor::scratch("test", {64}, BufferTensorType::Q8_1);
    EXPECT_EQ(desc2.sizeBytes(), 72);

    // 33 elements = 2 blocks (rounds up)
    auto desc3 = BufferDescriptor::scratch("test", {33}, BufferTensorType::Q8_1);
    EXPECT_EQ(desc3.sizeBytes(), 72);
}

TEST(Test__BufferDescriptor, IsAliasableOnlyScratch)
{
    EXPECT_FALSE(BufferDescriptor::input("a", {1}).isAliasable());
    EXPECT_FALSE(BufferDescriptor::output("a", {1}).isAliasable());
    EXPECT_FALSE(BufferDescriptor::inout("a", {1}).isAliasable());
    EXPECT_TRUE(BufferDescriptor::scratch("a", {1}).isAliasable());
    EXPECT_FALSE(BufferDescriptor::weight("a", {1}).isAliasable());
}

TEST(Test__BufferDescriptor, IsReadOnly)
{
    EXPECT_TRUE(BufferDescriptor::input("a", {1}).isReadOnly());
    EXPECT_FALSE(BufferDescriptor::output("a", {1}).isReadOnly());
    EXPECT_FALSE(BufferDescriptor::inout("a", {1}).isReadOnly());
    EXPECT_FALSE(BufferDescriptor::scratch("a", {1}).isReadOnly());
    EXPECT_TRUE(BufferDescriptor::weight("a", {1}).isReadOnly());
}

// =============================================================================
// StageBufferRequirements Tests
// =============================================================================

TEST(Test__StageBufferRequirements, DefaultConstruction)
{
    StageBufferRequirements reqs;

    EXPECT_TRUE(reqs.empty());
    EXPECT_EQ(reqs.size(), 0);
}

TEST(Test__StageBufferRequirements, BuilderChaining)
{
    StageBufferRequirements reqs;
    reqs.addInput("input", {16, 896})
        .addOutput("output", {16, 896})
        .addScratch("workspace", {1024})
        .addWeight("gamma", {896});

    EXPECT_FALSE(reqs.empty());
    EXPECT_EQ(reqs.size(), 4);
}

TEST(Test__StageBufferRequirements, GetByRole)
{
    StageBufferRequirements reqs;
    reqs.addInput("in1", {16, 896})
        .addInput("in2", {16, 896})
        .addOutput("out", {16, 896})
        .addScratch("scratch1", {1024})
        .addScratch("scratch2", {2048})
        .addScratch("scratch3", {512});

    auto inputs = reqs.getByRole(BufferRole::INPUT);
    EXPECT_EQ(inputs.size(), 2);

    auto outputs = reqs.getByRole(BufferRole::OUTPUT);
    EXPECT_EQ(outputs.size(), 1);

    auto scratch = reqs.getByRole(BufferRole::SCRATCH);
    EXPECT_EQ(scratch.size(), 3);

    auto weights = reqs.getByRole(BufferRole::WEIGHT);
    EXPECT_EQ(weights.size(), 0);
}

TEST(Test__StageBufferRequirements, GetByName)
{
    StageBufferRequirements reqs;
    reqs.addInput("activations", {16, 896})
        .addOutput("logits", {16, 151936});

    auto *found = reqs.getByName("activations");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->name, "activations");
    EXPECT_EQ(found->role, BufferRole::INPUT);

    auto *found2 = reqs.getByName("logits");
    ASSERT_NE(found2, nullptr);
    EXPECT_EQ(found2->role, BufferRole::OUTPUT);

    auto *notFound = reqs.getByName("nonexistent");
    EXPECT_EQ(notFound, nullptr);
}

TEST(Test__StageBufferRequirements, TotalBytesCalculation)
{
    StageBufferRequirements reqs;
    reqs.addInput("input", {16, 896}, BufferTensorType::FP32)     // 16*896*4 = 57344
        .addOutput("output", {16, 896}, BufferTensorType::FP32)   // 16*896*4 = 57344
        .addScratch("workspace", {1024}, BufferTensorType::FP32); // 1024*4 = 4096

    EXPECT_EQ(reqs.totalInputBytes(), 57344);
    EXPECT_EQ(reqs.totalOutputBytes(), 57344);
    EXPECT_EQ(reqs.totalScratchBytes(), 4096);
    EXPECT_EQ(reqs.totalBytes(), 57344 + 57344 + 4096);
}

TEST(Test__StageBufferRequirements, CustomDescriptor)
{
    BufferDescriptor custom;
    custom.name = "custom_buffer";
    custom.role = BufferRole::SCRATCH;
    custom.shape = {128, 128};
    custom.tensor_type = BufferTensorType::BF16;
    custom.required = false;
    custom.alignment = 128;
    custom.device_idx = 0;

    StageBufferRequirements reqs;
    reqs.add(std::move(custom));

    auto *found = reqs.getByName("custom_buffer");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->tensor_type, BufferTensorType::BF16);
    EXPECT_FALSE(found->required);
    EXPECT_EQ(found->alignment, 128);
    EXPECT_EQ(found->device_idx, 0);
}

// =============================================================================
// Real-World Usage Pattern Tests
// =============================================================================

TEST(Test__BufferRole, RMSNormStagePattern)
{
    // Simulating what RMSNormStage::getBufferRequirements() would return
    const size_t seq_len = 16;
    const size_t d_model = 896;

    StageBufferRequirements reqs;
    reqs.addInput("input", {seq_len, d_model})
        .addOutput("output", {seq_len, d_model})
        .addWeight("gamma", {d_model});

    EXPECT_EQ(reqs.size(), 3);
    EXPECT_EQ(reqs.getByRole(BufferRole::INPUT).size(), 1);
    EXPECT_EQ(reqs.getByRole(BufferRole::OUTPUT).size(), 1);
    EXPECT_EQ(reqs.getByRole(BufferRole::WEIGHT).size(), 1);
}

TEST(Test__BufferRole, SwiGLUStagePattern)
{
    // Simulating what SwiGLUStage::getBufferRequirements() would return
    const size_t seq_len = 16;
    const size_t d_ff = 4864;

    StageBufferRequirements reqs;
    reqs.addInput("gate", {seq_len, d_ff})
        .addInput("up", {seq_len, d_ff})
        .addOutput("output", {seq_len, d_ff});

    EXPECT_EQ(reqs.size(), 3);
    EXPECT_EQ(reqs.totalInputBytes(), 2 * seq_len * d_ff * 4);
    EXPECT_EQ(reqs.totalOutputBytes(), seq_len * d_ff * 4);
}

TEST(Test__BufferRole, GEMMStagePattern)
{
    // Simulating what GEMMStage::getBufferRequirements() would return
    const int m = 16, n = 4864, k = 896;

    StageBufferRequirements reqs;
    reqs.addInput("A", {static_cast<size_t>(m), static_cast<size_t>(k)})
        .addWeight("B", {static_cast<size_t>(k), static_cast<size_t>(n)}, BufferTensorType::Q8_1)
        .addOutput("C", {static_cast<size_t>(m), static_cast<size_t>(n)});

    EXPECT_EQ(reqs.size(), 3);

    auto *weight = reqs.getByName("B");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->tensor_type, BufferTensorType::Q8_1);
}

TEST(Test__BufferRole, AttentionStagePattern)
{
    // Simulating attention stage buffer requirements
    const size_t seq_len = 16;
    const size_t kv_len = 64; // Cached tokens
    const size_t n_heads = 14;
    const size_t head_dim = 64;

    StageBufferRequirements reqs;
    reqs.addInput("Q", {seq_len, n_heads * head_dim})
        .addInput("K", {kv_len, n_heads * head_dim})
        .addInput("V", {kv_len, n_heads * head_dim})
        .addOutput("output", {seq_len, n_heads * head_dim})
        .addScratch("scores", {n_heads, seq_len, kv_len})
        .addScratch("context", {seq_len, head_dim});

    EXPECT_EQ(reqs.size(), 6);
    EXPECT_EQ(reqs.getByRole(BufferRole::INPUT).size(), 3);
    EXPECT_EQ(reqs.getByRole(BufferRole::SCRATCH).size(), 2);

    // All scratch buffers should be aliasable
    auto scratch = reqs.getByRole(BufferRole::SCRATCH);
    for (auto *buf : scratch)
    {
        EXPECT_TRUE(buf->isAliasable());
    }
}

TEST(Test__BufferRole, ResidualAddPattern)
{
    // Residual add uses INOUT pattern
    const size_t seq_len = 16;
    const size_t d_model = 896;

    StageBufferRequirements reqs;
    reqs.addInput("input", {seq_len, d_model})
        .addInout("residual", {seq_len, d_model});

    EXPECT_EQ(reqs.size(), 2);

    auto *residual = reqs.getByName("residual");
    ASSERT_NE(residual, nullptr);
    EXPECT_EQ(residual->role, BufferRole::INOUT);
    EXPECT_FALSE(residual->isAliasable()); // INOUT cannot be aliased
    EXPECT_FALSE(residual->isReadOnly());  // INOUT is read-write
}
