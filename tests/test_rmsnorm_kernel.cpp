#include <gtest/gtest.h>
#include "../src/kernels/RMSNormKernel.h"
#include "graph_compute.h"
#include <memory>
#include <cmath>
#include <chrono>

using namespace llaminar;

class RMSNormKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        kernel = std::make_unique<RMSNormKernel>();
    }

    std::unique_ptr<RMSNormKernel> kernel;

    std::shared_ptr<Tensor> createTensor(const std::vector<int> &shape, const std::vector<float> &data)
    {
        auto tensor = std::make_shared<Tensor>();
        tensor->shape = shape;
        tensor->data = data;
        return tensor;
    }

    void assertTensorEqual(const Tensor &actual, const std::vector<float> &expected, float tolerance = 1e-6f)
    {
        ASSERT_EQ(actual.data.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_NEAR(actual.data[i], expected[i], tolerance)
                << "Mismatch at index " << i << ": expected " << expected[i]
                << ", got " << actual.data[i];
        }
    }
};

TEST_F(RMSNormKernelTest, BasicNormalization)
{
    // Test data: [1, 2, 3, 4] with weight [1, 1, 1, 1]
    auto input = createTensor({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto weight = createTensor({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto output = createTensor({1, 4}, {0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // Expected: normalized [1,2,3,4] = [1,2,3,4] / sqrt(mean([1,4,9,16])) = [1,2,3,4] / sqrt(7.5) = [1,2,3,4] / 2.738613
    std::vector<float> expected = {0.365148f, 0.730297f, 1.095445f, 1.460593f};
    assertTensorEqual(*output, expected, 1e-5f);
}

TEST_F(RMSNormKernelTest, WithScaling)
{
    // Test with different weight values
    auto input = createTensor({1, 3}, {1.0f, 2.0f, 3.0f});
    auto weight = createTensor({3}, {2.0f, 0.5f, 1.5f});
    auto output = createTensor({1, 3}, {0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // Expected computation:
    // RMS = sqrt(mean([1,4,9])) = sqrt(14/3) ≈ 2.16
    // normalized = [1,2,3] / 2.16 ≈ [0.463, 0.926, 1.389]
    // scaled = [0.463*2, 0.926*0.5, 1.389*1.5] = [0.926, 0.463, 2.084]
    std::vector<float> expected = {0.925820f, 0.462910f, 2.083095f};
    assertTensorEqual(*output, expected, 1e-5f);
}

TEST_F(RMSNormKernelTest, MultipleSequences)
{
    // Test with multiple sequences (batch processing)
    auto input = createTensor({2, 3}, {1.0f, 2.0f, 3.0f,
                                       4.0f, 5.0f, 6.0f});
    auto weight = createTensor({3}, {1.0f, 1.0f, 1.0f});
    auto output = createTensor({2, 3}, {0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // First sequence RMS = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
    // Second sequence RMS = sqrt((16+25+36)/3) = sqrt(77/3) ≈ 5.07
    std::vector<float> expected = {
        0.462910f, 0.925820f, 1.388730f, // First sequence
        0.788811f, 0.986014f, 1.183217f  // Second sequence
    };
    assertTensorEqual(*output, expected, 1e-5f);
}

TEST_F(RMSNormKernelTest, ZeroInput)
{
    // Test edge case with zero input
    auto input = createTensor({1, 3}, {0.0f, 0.0f, 0.0f});
    auto weight = createTensor({3}, {1.0f, 1.0f, 1.0f});
    auto output = createTensor({1, 3}, {0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // Should handle division by zero gracefully (add epsilon)
    std::vector<float> expected = {0.0f, 0.0f, 0.0f};
    assertTensorEqual(*output, expected, 1e-6f);
}

TEST_F(RMSNormKernelTest, InputValidation)
{
    // Test with wrong number of inputs
    auto input = createTensor({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto output = createTensor({1, 4}, {0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input}; // Missing weight
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

TEST_F(RMSNormKernelTest, ShapeValidation)
{
    // Test with mismatched shapes
    auto input = createTensor({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto weight = createTensor({3}, {1.0f, 1.0f, 1.0f}); // Wrong size
    auto output = createTensor({1, 4}, {0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

// Performance test (optional - can be enabled/disabled)
TEST_F(RMSNormKernelTest, DISABLED_PerformanceTest)
{
    // Large tensor for performance testing
    const int seq_len = 1000;
    const int hidden_size = 4096;

    std::vector<float> input_data(seq_len * hidden_size, 1.0f);
    std::vector<float> weight_data(hidden_size, 1.0f);
    std::vector<float> output_data(seq_len * hidden_size, 0.0f);

    auto input = createTensor({seq_len, hidden_size}, input_data);
    auto weight = createTensor({hidden_size}, weight_data);
    auto output = createTensor({seq_len, hidden_size}, output_data);

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    auto start = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(kernel->execute(inputs, outputs));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "RMSNorm performance test: " << duration.count() << " ms" << std::endl;
}