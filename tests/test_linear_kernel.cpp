#include <gtest/gtest.h>
#include "../src/kernels/LinearKernel.h"
#include "graph_compute.h"
#include <memory>
#include <chrono>

using namespace llaminar;

class LinearKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        kernel = std::make_unique<LinearKernel>();
    }

    std::unique_ptr<LinearKernel> kernel;

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

TEST_F(LinearKernelTest, BasicMatrixMultiplication)
{
    // Test: [1, 2] @ [[1, 3], [2, 4]] = [5, 11]
    auto input = createTensor({1, 2}, {1.0f, 2.0f});
    auto weight = createTensor({2, 2}, {1.0f, 3.0f, 2.0f, 4.0f}); // Column-major order
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    std::vector<float> expected = {5.0f, 11.0f};
    assertTensorEqual(*output, expected);
}

TEST_F(LinearKernelTest, WithBias)
{
    // Test: [1, 2] @ [[1, 3], [2, 4]] + [10, 20] = [15, 31]
    auto input = createTensor({1, 2}, {1.0f, 2.0f});
    auto weight = createTensor({2, 2}, {1.0f, 3.0f, 2.0f, 4.0f});
    auto bias = createTensor({2}, {10.0f, 20.0f});
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    std::vector<float> expected = {15.0f, 31.0f};
    assertTensorEqual(*output, expected);
}

TEST_F(LinearKernelTest, BatchProcessing)
{
    // Test batch: [[1, 2], [3, 4]] @ [[1, 3], [2, 4]] = [[5, 11], [11, 25]]
    auto input = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto weight = createTensor({2, 2}, {1.0f, 3.0f, 2.0f, 4.0f});
    auto output = createTensor({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    std::vector<float> expected = {5.0f, 11.0f, 11.0f, 25.0f};
    assertTensorEqual(*output, expected);
}

TEST_F(LinearKernelTest, DifferentDimensions)
{
    // Test: [1, 2, 3] @ [[1], [2], [3]] = [14]
    auto input = createTensor({1, 3}, {1.0f, 2.0f, 3.0f});
    auto weight = createTensor({3, 1}, {1.0f, 2.0f, 3.0f});
    auto output = createTensor({1, 1}, {0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    std::vector<float> expected = {14.0f}; // 1*1 + 2*2 + 3*3 = 14
    assertTensorEqual(*output, expected);
}

TEST_F(LinearKernelTest, ZeroWeights)
{
    // Test with zero weights
    auto input = createTensor({1, 2}, {1.0f, 2.0f});
    auto weight = createTensor({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    std::vector<float> expected = {0.0f, 0.0f};
    assertTensorEqual(*output, expected);
}

TEST_F(LinearKernelTest, IdentityTransform)
{
    // Test identity matrix
    auto input = createTensor({1, 3}, {1.0f, 2.0f, 3.0f});
    auto weight = createTensor({3, 3}, {1.0f, 0.0f, 0.0f,
                                        0.0f, 1.0f, 0.0f,
                                        0.0f, 0.0f, 1.0f});
    auto output = createTensor({1, 3}, {0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    std::vector<float> expected = {1.0f, 2.0f, 3.0f};
    assertTensorEqual(*output, expected);
}

TEST_F(LinearKernelTest, LargeNumbers)
{
    // Test with larger numbers to check numerical stability
    auto input = createTensor({1, 2}, {100.0f, 200.0f});
    auto weight = createTensor({2, 2}, {0.01f, 0.03f, 0.02f, 0.04f});
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // 100*0.01 + 200*0.02 = 1 + 4 = 5
    // 100*0.03 + 200*0.04 = 3 + 8 = 11
    std::vector<float> expected = {5.0f, 11.0f};
    assertTensorEqual(*output, expected);
}

TEST_F(LinearKernelTest, InputValidation)
{
    // Test with wrong number of inputs
    auto input = createTensor({1, 2}, {1.0f, 2.0f});
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input}; // Missing weight
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

TEST_F(LinearKernelTest, DimensionMismatch)
{
    // Test with incompatible dimensions
    auto input = createTensor({1, 3}, {1.0f, 2.0f, 3.0f});
    auto weight = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}); // Wrong input size
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

TEST_F(LinearKernelTest, BiasShapeMismatch)
{
    // Test with wrong bias shape
    auto input = createTensor({1, 2}, {1.0f, 2.0f});
    auto weight = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto bias = createTensor({2}, {1.0f, 2.0f}); // Should be size 3
    auto output = createTensor({1, 3}, {0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight, bias};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

// Performance test (optional)
TEST_F(LinearKernelTest, DISABLED_PerformanceTest)
{
    const int batch_size = 32;
    const int input_size = 1024;
    const int output_size = 2048;

    std::vector<float> input_data(batch_size * input_size, 0.1f);
    std::vector<float> weight_data(input_size * output_size, 0.01f);
    std::vector<float> output_data(batch_size * output_size, 0.0f);

    auto input = createTensor({batch_size, input_size}, input_data);
    auto weight = createTensor({input_size, output_size}, weight_data);
    auto output = createTensor({batch_size, output_size}, output_data);

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    auto start = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(kernel->execute(inputs, outputs));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Linear kernel performance test: " << duration.count() << " ms" << std::endl;
}