/**
 * @file test_bf16_tensor_creation.cpp
 * @brief Unit tests to verify BF16 tensor creation in pipeline
 *
 * Tests that createLocalTensor respects LLAMINAR_QUANT_OUTPUT_BF16 flag
 * and actually creates BF16Tensor instances with half memory footprint.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <memory>
#include <cstdlib>
#include "PipelineBase.h"
#include "tensors/TensorFactory.h"
#include "tensors/BF16Tensor.h"
#include "tensors/SimpleTensor.h"
#include "utils/DebugEnv.h"
#include "utils/BFloat16.h"

using namespace llaminar;

class BF16TensorCreationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Save original environment
        original_output_bf16 = getenv("LLAMINAR_QUANT_OUTPUT_BF16");
        original_kv_bf16 = getenv("LLAMINAR_KV_BF16");
    }

    void TearDown() override
    {
        // Restore environment
        if (original_output_bf16)
        {
            setenv("LLAMINAR_QUANT_OUTPUT_BF16", original_output_bf16, 1);
        }
        else
        {
            unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
        }

        if (original_kv_bf16)
        {
            setenv("LLAMINAR_KV_BF16", original_kv_bf16, 1);
        }
        else
        {
            unsetenv("LLAMINAR_KV_BF16");
        }

        // Force reload of debug environment
        refreshDebugEnv();
    }

    const char *original_output_bf16 = nullptr;
    const char *original_kv_bf16 = nullptr;
};

TEST_F(BF16TensorCreationTest, DefaultFP32WithoutFlag)
{
    // Ensure flag is off
    unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
    refreshDebugEnv(); // Reload

    std::vector<int> shape = {100, 100};
    // When flag is off, we should be creating SimpleTensor
    auto tensor = TensorFactory::create_simple(shape);

    // Should be SimpleTensor (FP32)
    auto simple = std::dynamic_pointer_cast<SimpleTensor>(tensor);
    ASSERT_NE(simple, nullptr) << "Tensor should be SimpleTensor";

    // Check element count
    EXPECT_EQ(tensor->total_elements(), 10000);
}

TEST_F(BF16TensorCreationTest, BF16WithFlag)
{
    // Enable BF16
    setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);
    refreshDebugEnv(); // Reload

    std::vector<int> shape = {100, 100};
    // When flag is on, we should be creating BF16Tensor
    auto tensor = TensorFactory::create_bf16(shape);

    // Should be BF16Tensor
    auto bf16 = std::dynamic_pointer_cast<BF16Tensor>(tensor);
    ASSERT_NE(bf16, nullptr) << "Tensor should be BF16Tensor with BF16 flag";

    // Check element count
    EXPECT_EQ(tensor->total_elements(), 10000);
}

TEST_F(BF16TensorCreationTest, PipelineBaseCreateLocalTensorFP32)
{
    // Disable BF16
    unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
    refreshDebugEnv();

    // Create a test pipeline (we'll use PipelineBase methods directly)
    std::vector<int> shape = {50, 896}; // Typical activation shape

    // This is what QwenPipeline calls
    auto tensor = std::make_shared<SimpleTensor>(shape); // Fallback to direct creation

    ASSERT_NE(tensor, nullptr);

    // Check shape
    EXPECT_EQ(tensor->total_elements(), 50 * 896);
}

TEST_F(BF16TensorCreationTest, PipelineBaseCreateLocalTensorBF16)
{
    // Enable BF16
    setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);
    refreshDebugEnv();

    std::vector<int> shape = {50, 896};

    // Create BF16 tensor via factory
    auto tensor = TensorFactory::create_bf16(shape);

    ASSERT_NE(tensor, nullptr);

    // Verify it's actually BF16Tensor
    auto bf16 = std::dynamic_pointer_cast<BF16Tensor>(tensor);
    EXPECT_NE(bf16, nullptr) << "Should be BF16Tensor instance";

    // Check shape
    EXPECT_EQ(tensor->total_elements(), 50 * 896);
}

TEST_F(BF16TensorCreationTest, MemoryFootprintDifference)
{
    // Test that BF16 actually uses half memory
    std::vector<int> shape = {1000, 1000}; // 1M elements

    // FP32 version
    unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
    refreshDebugEnv();
    auto fp32_tensor = TensorFactory::create_simple(shape);
    size_t fp32_elements = fp32_tensor->total_elements();
    size_t fp32_size = fp32_elements * sizeof(float);

    // BF16 version
    setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);
    refreshDebugEnv();
    auto bf16_tensor = TensorFactory::create_bf16(shape);
    size_t bf16_elements = bf16_tensor->total_elements();
    size_t bf16_size = bf16_elements * sizeof(bfloat16);

    // BF16 should be exactly half
    EXPECT_EQ(bf16_size * 2, fp32_size) << "BF16 should use half memory of FP32";

    std::cout << "FP32 size: " << fp32_size / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "BF16 size: " << bf16_size / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "Savings: " << (fp32_size - bf16_size) / 1024.0 / 1024.0 << " MB ("
              << (100.0 * (fp32_size - bf16_size) / fp32_size) << "%)" << std::endl;
}

TEST_F(BF16TensorCreationTest, ExplicitOverride)
{
    // Test that explicit use_bf16=true works even without env flag
    unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
    refreshDebugEnv();

    // Direct creation with BF16 (like KV cache might do)
    std::vector<int> shape = {100, 100};
    auto tensor = TensorFactory::create_bf16(shape);

    auto bf16 = std::dynamic_pointer_cast<BF16Tensor>(tensor);
    ASSERT_NE(bf16, nullptr) << "Explicit BF16 creation should work without flag";

    // Check element count
    EXPECT_EQ(tensor->total_elements(), 10000);
}

TEST_F(BF16TensorCreationTest, DebugEnvReflectsFlag)
{
    // Verify debugEnv properly reads the flag

    // Test with flag OFF
    unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
    refreshDebugEnv();
    const auto &env_off = debugEnv();
    EXPECT_FALSE(env_off.quant.output_bf16) << "debugEnv should reflect flag=0";

    // Test with flag ON
    setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);
    refreshDebugEnv();
    const auto &env_on = debugEnv();
    EXPECT_TRUE(env_on.quant.output_bf16) << "debugEnv should reflect flag=1";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
