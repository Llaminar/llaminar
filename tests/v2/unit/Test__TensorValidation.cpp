/**
 * @file Test__TensorValidation.cpp
 * @brief Unit tests for TensorValidation utilities
 * @author GitHub Copilot
 * @date December 2025
 *
 * Tests the debug-only tensor validation utilities that detect
 * uninitialized, zero, and corrupted tensors.
 */

#include <gtest/gtest.h>
#include "../../../src/v2/tensors/TensorValidation.h"
#include "../../../src/v2/tensors/Tensors.h"
#include "../utils/TestTensorFactory.h"

#include <cmath>
#include <limits>

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class TensorValidationTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// tensorAppearsZero Tests
// =============================================================================

TEST_F(TensorValidationTest, DetectsNullTensor)
{
    EXPECT_TRUE(tensorAppearsZero(nullptr));
}

TEST_F(TensorValidationTest, DetectsZeroTensor)
{
    // Create a tensor filled with zeros
    auto zero_tensor = TestTensorFactory::createFP32({32, 64});
    float *data = zero_tensor->mutable_data();
    for (size_t i = 0; i < zero_tensor->numel(); ++i)
    {
        data[i] = 0.0f;
    }

    EXPECT_TRUE(tensorAppearsZero(zero_tensor.get()));
}

TEST_F(TensorValidationTest, PassesNonZeroTensor)
{
    // Create a tensor with random non-zero values
    auto valid_tensor = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f);

    EXPECT_FALSE(tensorAppearsZero(valid_tensor.get()));
}

TEST_F(TensorValidationTest, DetectsSingleNonZero)
{
    // Create mostly-zero tensor with one non-zero value
    auto tensor = TestTensorFactory::createFP32({1000});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = 0.0f;
    }
    data[500] = 1.0f; // Single non-zero in middle

    // With sufficient sampling, should detect the non-zero
    EXPECT_FALSE(tensorAppearsZero(tensor.get(), 1000));
}

TEST_F(TensorValidationTest, LargeTensorSampling)
{
    // Create large tensor - sampling should still work
    auto large_tensor = TestTensorFactory::createFP32Random({1024, 1024}, -1.0f, 1.0f);

    // Should detect non-zero even with sampling
    EXPECT_FALSE(tensorAppearsZero(large_tensor.get(), 100));
}

// =============================================================================
// tensorHasNaNOrInf Tests
// =============================================================================

TEST_F(TensorValidationTest, DetectsNaN)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f);
    float *data = tensor->mutable_data();
    data[100] = std::numeric_limits<float>::quiet_NaN();

    EXPECT_TRUE(tensorHasNaNOrInf(tensor.get()));
}

TEST_F(TensorValidationTest, DetectsPosInf)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f);
    float *data = tensor->mutable_data();
    data[100] = std::numeric_limits<float>::infinity();

    EXPECT_TRUE(tensorHasNaNOrInf(tensor.get()));
}

TEST_F(TensorValidationTest, DetectsNegInf)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f);
    float *data = tensor->mutable_data();
    data[100] = -std::numeric_limits<float>::infinity();

    EXPECT_TRUE(tensorHasNaNOrInf(tensor.get()));
}

TEST_F(TensorValidationTest, PassesValidTensor)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f);

    EXPECT_FALSE(tensorHasNaNOrInf(tensor.get()));
}

TEST_F(TensorValidationTest, NullTensorNoNaN)
{
    // Null tensor doesn't have NaN (it has nothing)
    EXPECT_FALSE(tensorHasNaNOrInf(nullptr));
}

// =============================================================================
// validateTensorNotZero Tests (Warning Path)
// =============================================================================

TEST_F(TensorValidationTest, ValidateLogs_ZeroTensor)
{
    auto zero_tensor = TestTensorFactory::createFP32({32, 64});
    float *data = zero_tensor->mutable_data();
    for (size_t i = 0; i < zero_tensor->numel(); ++i)
    {
        data[i] = 0.0f;
    }

    // This should log a warning but not throw
    // We can't easily capture log output, so just verify it doesn't crash
    EXPECT_NO_THROW(validateTensorNotZero(zero_tensor.get(), "test_tensor", "TestStage"));
}

TEST_F(TensorValidationTest, ValidateLogs_NullTensor)
{
    EXPECT_NO_THROW(validateTensorNotZero(nullptr, "null_tensor", "TestStage"));
}

TEST_F(TensorValidationTest, ValidatePasses_ValidTensor)
{
    auto valid_tensor = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f);
    EXPECT_NO_THROW(validateTensorNotZero(valid_tensor.get(), "valid_tensor", "TestStage"));
}

// =============================================================================
// assertTensorValid Tests (Throw Path)
// =============================================================================

TEST_F(TensorValidationTest, AssertThrows_NullTensor)
{
    EXPECT_THROW(assertTensorValid(nullptr, "null_tensor", "TestStage"),
                 std::runtime_error);
}

TEST_F(TensorValidationTest, AssertThrows_ZeroTensor)
{
    auto zero_tensor = TestTensorFactory::createFP32({32, 64});
    float *data = zero_tensor->mutable_data();
    for (size_t i = 0; i < zero_tensor->numel(); ++i)
    {
        data[i] = 0.0f;
    }

    EXPECT_THROW(assertTensorValid(zero_tensor.get(), "zero_tensor", "TestStage"),
                 std::runtime_error);
}

TEST_F(TensorValidationTest, AssertThrows_NaNTensor)
{
    auto tensor = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f);
    float *data = tensor->mutable_data();
    data[0] = std::numeric_limits<float>::quiet_NaN();

    EXPECT_THROW(assertTensorValid(tensor.get(), "nan_tensor", "TestStage"),
                 std::runtime_error);
}

TEST_F(TensorValidationTest, AssertPasses_ValidTensor)
{
    auto valid_tensor = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f);
    EXPECT_NO_THROW(assertTensorValid(valid_tensor.get(), "valid_tensor", "TestStage"));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(TensorValidationTest, EmptyTensor)
{
    auto empty_tensor = TestTensorFactory::createFP32({0});

    // Empty tensor is considered "zero" (has no non-zero elements)
    EXPECT_TRUE(tensorAppearsZero(empty_tensor.get()));
    EXPECT_FALSE(tensorHasNaNOrInf(empty_tensor.get()));
}

TEST_F(TensorValidationTest, SmallTensor)
{
    // Single-element tensor
    auto tiny = TestTensorFactory::createFP32({1});
    tiny->mutable_data()[0] = 42.0f;

    EXPECT_FALSE(tensorAppearsZero(tiny.get()));
    EXPECT_FALSE(tensorHasNaNOrInf(tiny.get()));
}

TEST_F(TensorValidationTest, VerySmallNonZero)
{
    // Tensor with very small but non-zero values
    auto tensor = TestTensorFactory::createFP32({100});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = 1e-38f; // Very small but not zero
    }

    EXPECT_FALSE(tensorAppearsZero(tensor.get()));
}
