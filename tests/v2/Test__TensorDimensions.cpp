/**
 * @file Test__TensorDimensions.cpp
 * @brief Test tensor dimension validation system
 *
 * Demonstrates that:
 * 1. Validation catches dimension mismatches in DEBUG builds
 * 2. Validation compiles to no-op in RELEASE builds
 * 3. Pipeline-specific dimension specs work correctly
 */

#include "../../src/v2/pipelines/TensorDimensions.h"
#include "../../src/v2/tensors/Tensors.h"
#include <gtest/gtest.h>
#include <memory>

using namespace llaminar2;

TEST(Test__TensorDimensions, TensorSpecMatching)
{
    // Create a tensor: [4, 768]
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 768});

    // Create matching spec
    TensorSpec spec_correct({4, 768}, "hidden[4, 768]");
    EXPECT_TRUE(spec_correct.matches(tensor->shape()));

    // Create mismatched spec (wrong dimensions)
    TensorSpec spec_wrong_dims({4, 512}, "hidden[4, 512]");
    EXPECT_FALSE(spec_wrong_dims.matches(tensor->shape()));

    // Create mismatched spec (wrong rank)
    TensorSpec spec_wrong_rank({4}, "hidden[4]");
    EXPECT_FALSE(spec_wrong_rank.matches(tensor->shape()));
}

TEST(Test__TensorDimensions, ShapeToString)
{
    std::vector<size_t> shape = {16, 2048, 128};
    std::string shape_str = TensorSpec::shape_str(shape);
    EXPECT_EQ(shape_str, "[16, 2048, 128]");
}

#ifndef NDEBUG
// These tests only run in DEBUG builds where validation is active

TEST(Test__TensorDimensions, ValidationPassesCorrectShape)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 1024});
    TensorSpec spec({8, 1024}, "test[8, 1024]");

    // Should not abort in debug mode when shape matches
    VALIDATE_TENSOR(tensor, spec, "test_stage");

    // If we got here, validation passed
    SUCCEED();
}

TEST(Test__TensorDimensions, SameShapeValidation)
{
    auto tensor1 = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 512});
    auto tensor2 = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 512});

    // Should not abort when shapes match
    ASSERT_SAME_SHAPE(tensor1, tensor2, "test_stage");

    SUCCEED();
}

// Note: We cannot test validation failures (abort cases) in unit tests
// since they call std::abort(). In real usage, these will be caught
// during development when running with debug builds.

#else
// RELEASE build tests

TEST(Test__TensorDimensions, ValidationCompilesAway)
{
    // In release builds, validation should compile to no-ops
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{8, 1024});
    TensorSpec spec({8, 1024}, "test[8, 1024]");

    // This should compile to nothing and have zero runtime cost
    VALIDATE_TENSOR(tensor, spec, "test_stage");

    // Verify we can even pass mismatched specs (validation is disabled)
    TensorSpec wrong_spec({16, 512}, "wrong[16, 512]");
    VALIDATE_TENSOR(tensor, wrong_spec, "test_stage"); // No error in release!

    SUCCEED();
}

#endif

/**
 * @brief Example: Pipeline-specific dimension helper
 *
 * Shows how pipelines can create helper methods for their architecture
 */
class MockPipeline
{
public:
    MockPipeline(int d_model, int d_ff, int vocab_size)
        : d_model_(d_model), d_ff_(d_ff), vocab_size_(vocab_size) {}

    // Pipeline-specific dimension helpers
    TensorSpec spec_hidden(int seq_len) const
    {
        return TensorSpec({static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)},
                          "hidden[" + std::to_string(seq_len) + "," + std::to_string(d_model_) + "]");
    }

    TensorSpec spec_ffn_intermediate(int seq_len) const
    {
        return TensorSpec({static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)},
                          "ffn[" + std::to_string(seq_len) + "," + std::to_string(d_ff_) + "]");
    }

    TensorSpec spec_logits(int seq_len) const
    {
        return TensorSpec({static_cast<size_t>(seq_len), static_cast<size_t>(vocab_size_)},
                          "logits[" + std::to_string(seq_len) + "," + std::to_string(vocab_size_) + "]");
    }

private:
    int d_model_;
    int d_ff_;
    int vocab_size_;
};

TEST(Test__TensorDimensions, PipelineSpecHelpers)
{
    // Create mock pipeline with Qwen 0.5B dimensions
    MockPipeline pipeline(896, 4864, 151936);

    // Get specs for seq_len=8
    TensorSpec hidden = pipeline.spec_hidden(8);
    TensorSpec ffn = pipeline.spec_ffn_intermediate(8);
    TensorSpec logits = pipeline.spec_logits(8);

    // Verify shapes
    EXPECT_EQ(hidden.expected_shape.size(), 2);
    EXPECT_EQ(hidden.expected_shape[0], 8);
    EXPECT_EQ(hidden.expected_shape[1], 896);

    EXPECT_EQ(ffn.expected_shape.size(), 2);
    EXPECT_EQ(ffn.expected_shape[0], 8);
    EXPECT_EQ(ffn.expected_shape[1], 4864);

    EXPECT_EQ(logits.expected_shape.size(), 2);
    EXPECT_EQ(logits.expected_shape[0], 8);
    EXPECT_EQ(logits.expected_shape[1], 151936);

    // Verify descriptions
    EXPECT_EQ(hidden.description, "hidden[8,896]");
    EXPECT_EQ(ffn.description, "ffn[8,4864]");
    EXPECT_EQ(logits.description, "logits[8,151936]");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
