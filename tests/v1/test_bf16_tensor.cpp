/**
 * @file test_bf16_tensor.cpp
 * @brief Unit tests for BF16Tensor class (Phase 5)
 * @author David Sanftenberg
 * @date October 20, 2025
 *
 * Tests BF16 activation storage:
 * - Basic tensor operations (construction, copy, fill)
 * - FP32↔BF16 conversion accuracy
 * - NUMA first-touch initialization
 * - Batch operations (stack, get_batch)
 * - TensorFactory integration
 */

#include <gtest/gtest.h>
#include "../src/tensors/BF16Tensor.h"
#include "../src/tensors/TensorFactory.h"
#include "../src/tensors/SimpleTensor.h"
#include "../src/utils/BFloat16.h"
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

using namespace llaminar;

class BF16TensorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure NUMA first-touch is enabled for tests
        setenv("LLAMINAR_NUMA_FIRST_TOUCH", "1", 1);
    }

    void TearDown() override
    {
        unsetenv("LLAMINAR_NUMA_FIRST_TOUCH");
    }

    // Helper: Compute relative error between two values
    float relativeError(float a, float b)
    {
        if (std::abs(a) < 1e-8f && std::abs(b) < 1e-8f)
            return 0.0f; // Both near zero
        return std::abs(a - b) / (std::abs(a) + std::abs(b) + 1e-8f);
    }

    // Helper: Compute max relative error across arrays
    float maxRelativeError(const float *a, const float *b, size_t count)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            max_err = std::max(max_err, relativeError(a[i], b[i]));
        }
        return max_err;
    }
};

// =============================================================================
// Test 1: Basic Construction
// =============================================================================

TEST_F(BF16TensorTest, BasicConstruction)
{
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{8, 896});

    EXPECT_EQ(tensor->size(), 8 * 896);
    EXPECT_EQ(tensor->shape()[0], 8);
    EXPECT_EQ(tensor->shape()[1], 896);
    EXPECT_EQ(tensor->ndim(), 2);
    EXPECT_EQ(tensor->type_name(), "BF16Tensor");
    EXPECT_FALSE(tensor->is_distributed());
}

TEST_F(BF16TensorTest, ZeroInitialization)
{
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});

    const bfloat16 *data = tensor->bf16_data();
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_FLOAT_EQ(static_cast<float>(data[i]), 0.0f);
    }
}

TEST_F(BF16TensorTest, InvalidDimensions)
{
    EXPECT_THROW(
        auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{-1, 896}),
        std::invalid_argument);
}

// =============================================================================
// Test 2: FP32 ↔ BF16 Conversion
// =============================================================================

TEST_F(BF16TensorTest, FP32ToBF16Conversion)
{
    std::vector<float> fp32_data(100);
    std::iota(fp32_data.begin(), fp32_data.end(), 0.0f);

    auto bf16_tensor = std::make_shared<BF16Tensor>(
        std::vector<int>{10, 10}, fp32_data);

    // Convert back to FP32 and compare
    std::vector<float> fp32_recovered(100);
    bf16_tensor->to_fp32(fp32_recovered.data(), 100);

    // BF16 has ~7 bits mantissa → ~3-4 decimal digits precision
    // For small integers, expect exact match or very close
    for (int i = 0; i < 100; ++i)
    {
        float rel_err = relativeError(fp32_data[i], fp32_recovered[i]);
        EXPECT_LT(rel_err, 0.01f) << "Index " << i << ": " << fp32_data[i] << " vs " << fp32_recovered[i];
    }
}

TEST_F(BF16TensorTest, BF16PrecisionCharacteristics)
{
    // Test BF16 representable range and precision
    std::vector<float> test_values = {
        0.0f,         // Zero
        1.0f,         // Unity
        -1.0f,        // Negative
        3.14159f,     // Pi (should round)
        1e-8f,        // Small number (preserved)
        1e20f,        // Large number (preserved due to wide exponent)
        1.0f + 1e-3f, // Near 1.0 (BF16 precision limit)
        1.0f + 1e-4f  // Below BF16 precision (will round)
    };

    for (float val : test_values)
    {
        bfloat16 bf16 = bfloat16::from_float(val);
        float recovered = static_cast<float>(bf16);

        // BF16 has ~0.8% relative precision (1/128 from 7 mantissa bits)
        if (std::abs(val) > 1e-6f)
        {
            float rel_err = std::abs(val - recovered) / std::abs(val);
            EXPECT_LT(rel_err, 0.01f) << "Value: " << val << " → " << recovered;
        }
    }
}

TEST_F(BF16TensorTest, ConversionSizeMismatch)
{
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});
    std::vector<float> wrong_size(50);

    EXPECT_THROW(
        tensor->from_fp32(wrong_size.data(), wrong_size.size()),
        std::invalid_argument);
}

// =============================================================================
// Test 3: Tensor Operations
// =============================================================================

TEST_F(BF16TensorTest, ZeroOperation)
{
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});

    // Fill with non-zero values
    tensor->fill(3.14f);

    // Zero out
    tensor->zero();

    const bfloat16 *data = tensor->bf16_data();
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_FLOAT_EQ(static_cast<float>(data[i]), 0.0f);
    }
}

TEST_F(BF16TensorTest, FillOperation)
{
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});
    tensor->fill(2.5f);

    const bfloat16 *data = tensor->bf16_data();
    for (int i = 0; i < 100; ++i)
    {
        // BF16 should represent 2.5 exactly (small power of 2)
        EXPECT_NEAR(static_cast<float>(data[i]), 2.5f, 0.01f);
    }
}

TEST_F(BF16TensorTest, CopyOperation)
{
    auto tensor1 = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});
    tensor1->fill(1.5f);

    auto tensor2 = tensor1->copy();

    EXPECT_NE(tensor1.get(), tensor2.get()); // Different instances
    EXPECT_EQ(tensor1->shape(), tensor2->shape());

    auto bf16_copy = TensorFactory::to_bf16_tensor(tensor2);
    ASSERT_NE(bf16_copy, nullptr);

    const bfloat16 *data1 = tensor1->bf16_data();
    const bfloat16 *data2 = bf16_copy->bf16_data();

    for (int i = 0; i < 100; ++i)
    {
        EXPECT_EQ(data1[i].data, data2[i].data);
    }
}

TEST_F(BF16TensorTest, CopyFromFP32)
{
    auto fp32_tensor = std::make_shared<SimpleTensor>(std::vector<int>{10, 10});
    std::fill(fp32_tensor->data(), fp32_tensor->data() + 100, 3.0f);

    auto bf16_tensor = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});
    bf16_tensor->copy_from(*fp32_tensor);

    const bfloat16 *data = bf16_tensor->bf16_data();
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_NEAR(static_cast<float>(data[i]), 3.0f, 0.01f);
    }
}

TEST_F(BF16TensorTest, CopyFromBF16)
{
    auto bf16_tensor1 = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});
    bf16_tensor1->fill(2.0f);

    auto bf16_tensor2 = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});
    bf16_tensor2->copy_from(*bf16_tensor1);

    const bfloat16 *data = bf16_tensor2->bf16_data();
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_NEAR(static_cast<float>(data[i]), 2.0f, 0.01f);
    }
}

TEST_F(BF16TensorTest, CopyShapeMismatch)
{
    auto tensor1 = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});
    auto tensor2 = std::make_shared<BF16Tensor>(std::vector<int>{5, 5});

    EXPECT_THROW(
        tensor1->copy_from(*tensor2),
        std::invalid_argument);
}

// =============================================================================
// Test 4: NUMA First-Touch
// =============================================================================

TEST_F(BF16TensorTest, NUMAFirstTouchLargeTensor)
{
    // Create large tensor (>128KB) to trigger NUMA first-touch
    // 128KB = 64K bfloat16 elements
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{512, 256});

    // Verify size exceeds threshold
    EXPECT_GE(tensor->size() * sizeof(bfloat16), 128 * 1024);

    // Verify zero-initialization (NUMA first-touch should apply)
    const bfloat16 *data = tensor->bf16_data();
    EXPECT_FLOAT_EQ(static_cast<float>(data[0]), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(data[1000]), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(data[tensor->size() - 1]), 0.0f);
}

TEST_F(BF16TensorTest, NUMAFirstTouchSmallTensor)
{
    // Small tensor (<128KB) should skip parallel init
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});

    // Verify size below threshold
    EXPECT_LT(tensor->size() * sizeof(bfloat16), 128 * 1024);

    // Still zero-initialized (single-threaded path)
    const bfloat16 *data = tensor->bf16_data();
    for (int i = 0; i < tensor->size(); ++i)
    {
        EXPECT_FLOAT_EQ(static_cast<float>(data[i]), 0.0f);
    }
}

// =============================================================================
// Test 5: Batch Operations
// =============================================================================

TEST_F(BF16TensorTest, BatchSizeQuery3D)
{
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{4, 8, 896});
    EXPECT_EQ(tensor->batch_size(), 4);
}

TEST_F(BF16TensorTest, BatchSizeQuery2D)
{
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{8, 896});
    EXPECT_EQ(tensor->batch_size(), 1); // Implicit batch=1
}

TEST_F(BF16TensorTest, BatchSizeQuery1D)
{
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{896});
    EXPECT_EQ(tensor->batch_size(), 1); // Implicit batch=1
}

TEST_F(BF16TensorTest, GetBatchExtraction)
{
    // Create 3D tensor [2, 3, 4]
    auto batched = std::make_shared<BF16Tensor>(std::vector<int>{2, 3, 4});

    // Fill with unique pattern
    bfloat16 *data = batched->bf16_data();
    for (int i = 0; i < 24; ++i)
    {
        data[i] = bfloat16::from_float(static_cast<float>(i));
    }

    // Extract batch 0
    auto batch0 = batched->get_batch(0);
    ASSERT_NE(batch0, nullptr);
    EXPECT_EQ(batch0->shape(), std::vector<int>({3, 4}));

    const bfloat16 *batch0_data = batch0->bf16_data();
    for (int i = 0; i < 12; ++i)
    {
        EXPECT_FLOAT_EQ(static_cast<float>(batch0_data[i]), static_cast<float>(i));
    }

    // Extract batch 1
    auto batch1 = batched->get_batch(1);
    const bfloat16 *batch1_data = batch1->bf16_data();
    for (int i = 0; i < 12; ++i)
    {
        EXPECT_FLOAT_EQ(static_cast<float>(batch1_data[i]), static_cast<float>(i + 12));
    }
}

TEST_F(BF16TensorTest, GetBatchInvalidIndex)
{
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{2, 3, 4});

    EXPECT_THROW(tensor->get_batch(-1), std::out_of_range);
    EXPECT_THROW(tensor->get_batch(2), std::out_of_range);
}

TEST_F(BF16TensorTest, GetBatchRequires3D)
{
    auto tensor = std::make_shared<BF16Tensor>(std::vector<int>{3, 4}); // 2D

    EXPECT_THROW(tensor->get_batch(0), std::runtime_error);
}

TEST_F(BF16TensorTest, StackBatchSingleSequence)
{
    std::vector<std::shared_ptr<BF16Tensor>> sequences;
    sequences.push_back(std::make_shared<BF16Tensor>(std::vector<int>{3, 4}));
    sequences[0]->fill(1.0f);

    auto batched = BF16Tensor::stack_batch(sequences);

    ASSERT_NE(batched, nullptr);
    EXPECT_EQ(batched->shape(), std::vector<int>({1, 3, 4}));
    EXPECT_EQ(batched->batch_size(), 1);
}

TEST_F(BF16TensorTest, StackBatchMultipleSequences)
{
    std::vector<std::shared_ptr<BF16Tensor>> sequences;

    for (int i = 0; i < 3; ++i)
    {
        auto seq = std::make_shared<BF16Tensor>(std::vector<int>{2, 3});
        seq->fill(static_cast<float>(i));
        sequences.push_back(seq);
    }

    auto batched = BF16Tensor::stack_batch(sequences);

    ASSERT_NE(batched, nullptr);
    EXPECT_EQ(batched->shape(), std::vector<int>({3, 2, 3}));
    EXPECT_EQ(batched->batch_size(), 3);

    // Verify data integrity
    const bfloat16 *data = batched->bf16_data();
    for (int b = 0; b < 3; ++b)
    {
        for (int i = 0; i < 6; ++i)
        {
            float expected = static_cast<float>(b);
            float actual = static_cast<float>(data[b * 6 + i]);
            EXPECT_NEAR(actual, expected, 0.01f);
        }
    }
}

TEST_F(BF16TensorTest, StackBatchEmpty)
{
    std::vector<std::shared_ptr<BF16Tensor>> sequences;
    auto result = BF16Tensor::stack_batch(sequences);
    EXPECT_EQ(result, nullptr);
}

TEST_F(BF16TensorTest, StackBatchShapeMismatch)
{
    std::vector<std::shared_ptr<BF16Tensor>> sequences;
    sequences.push_back(std::make_shared<BF16Tensor>(std::vector<int>{2, 3}));
    sequences.push_back(std::make_shared<BF16Tensor>(std::vector<int>{3, 4})); // Different shape

    EXPECT_THROW(BF16Tensor::stack_batch(sequences), std::invalid_argument);
}

// =============================================================================
// Test 6: TensorFactory Integration
// =============================================================================

TEST_F(BF16TensorTest, FactoryCreateBF16)
{
    auto tensor = TensorFactory::create_bf16(std::vector<int>{8, 896});

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->type_name(), "BF16Tensor");
    EXPECT_EQ(tensor->size(), 8 * 896);
}

TEST_F(BF16TensorTest, FactoryCreateBF16WithData)
{
    std::vector<float> fp32_data(100);
    std::iota(fp32_data.begin(), fp32_data.end(), 0.0f);

    auto tensor = TensorFactory::create_bf16(std::vector<int>{10, 10}, fp32_data);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->type_name(), "BF16Tensor");

    auto bf16_tensor = TensorFactory::to_bf16_tensor(tensor);
    const bfloat16 *data = bf16_tensor->bf16_data();

    for (int i = 0; i < 10; ++i)
    {
        float rel_err = relativeError(static_cast<float>(i), static_cast<float>(data[i]));
        EXPECT_LT(rel_err, 0.01f);
    }
}

TEST_F(BF16TensorTest, FactoryConvertToBF16FromFP32)
{
    auto fp32_tensor = std::make_shared<SimpleTensor>(std::vector<int>{10, 10});
    std::fill(fp32_tensor->data(), fp32_tensor->data() + 100, 2.5f);

    auto bf16_tensor_base = TensorFactory::convert_to_bf16(fp32_tensor);
    auto bf16_tensor = TensorFactory::to_bf16_tensor(bf16_tensor_base);

    ASSERT_NE(bf16_tensor, nullptr);
    const bfloat16 *data = bf16_tensor->bf16_data();

    for (int i = 0; i < 100; ++i)
    {
        EXPECT_NEAR(static_cast<float>(data[i]), 2.5f, 0.01f);
    }
}

TEST_F(BF16TensorTest, FactoryConvertToBF16FromBF16)
{
    auto bf16_tensor1 = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});
    bf16_tensor1->fill(3.0f);

    auto bf16_tensor2_base = TensorFactory::convert_to_bf16(bf16_tensor1);

    // Should return same instance (already BF16)
    EXPECT_EQ(bf16_tensor1.get(), bf16_tensor2_base.get());
}

TEST_F(BF16TensorTest, FactoryToBF16Tensor)
{
    auto tensor = TensorFactory::create_bf16(std::vector<int>{10, 10});
    auto bf16_tensor = TensorFactory::to_bf16_tensor(tensor);

    ASSERT_NE(bf16_tensor, nullptr);
    EXPECT_EQ(bf16_tensor->type_name(), "BF16Tensor");
}

TEST_F(BF16TensorTest, FactoryToBF16TensorFromFP32)
{
    auto fp32_tensor = std::make_shared<SimpleTensor>(std::vector<int>{10, 10});
    std::fill(fp32_tensor->data(), fp32_tensor->data() + 100, 1.5f);

    auto bf16_tensor = TensorFactory::to_bf16_tensor(fp32_tensor);

    ASSERT_NE(bf16_tensor, nullptr);
    const bfloat16 *data = bf16_tensor->bf16_data();

    for (int i = 0; i < 100; ++i)
    {
        EXPECT_NEAR(static_cast<float>(data[i]), 1.5f, 0.01f);
    }
}

// =============================================================================
// Test 7: TensorBase Compatibility
// =============================================================================

TEST_F(BF16TensorTest, TensorBaseDataAccess)
{
    auto bf16_tensor = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});
    bf16_tensor->fill(2.0f);

    // Access via TensorBase interface (triggers FP32 cache)
    TensorBase *base = bf16_tensor.get();
    const float *fp32_data = base->data();

    ASSERT_NE(fp32_data, nullptr);

    for (int i = 0; i < 100; ++i)
    {
        EXPECT_NEAR(fp32_data[i], 2.0f, 0.01f);
    }
}

TEST_F(BF16TensorTest, TensorBaseCacheInvalidation)
{
    auto bf16_tensor = std::make_shared<BF16Tensor>(std::vector<int>{10, 10});
    bf16_tensor->fill(1.0f);

    // Access cache
    TensorBase *base = bf16_tensor.get();
    const float *fp32_data1 = base->data();
    EXPECT_NEAR(fp32_data1[0], 1.0f, 0.01f);

    // Modify tensor (should invalidate cache)
    bf16_tensor->fill(2.0f);

    // Re-access cache (should be updated)
    const float *fp32_data2 = base->data();
    EXPECT_NEAR(fp32_data2[0], 2.0f, 0.01f);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
