#include <gtest/gtest.h>
#include "../src/kernels/MLPKernel.h"
#include "graph_compute.h"
#include <memory>
#include <chrono>
#include <cmath>

using namespace llaminar;

class MLPKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        kernel = std::make_unique<MLPKernel>();
    }

    std::unique_ptr<MLPKernel> kernel;

    std::shared_ptr<Tensor> createTensor(const std::vector<int> &shape, const std::vector<float> &data)
    {
        auto tensor = std::make_shared<Tensor>();
        tensor->shape = shape;
        tensor->data = data;
        return tensor;
    }

    void assertTensorEqual(const Tensor &actual, const std::vector<float> &expected, float tolerance = 1e-5f)
    {
        ASSERT_EQ(actual.data.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_NEAR(actual.data[i], expected[i], tolerance)
                << "Mismatch at index " << i << ": expected " << expected[i]
                << ", got " << actual.data[i];
        }
    }

    // Helper function to compute SiLU activation: x * sigmoid(x)
    float silu(float x)
    {
        return x / (1.0f + expf(-x));
    }
};

TEST_F(MLPKernelTest, BasicSwiGLUComputation)
{
    // Test basic SwiGLU: gate=silu(input @ gate_weight), up=input @ up_weight, output=(gate * up) @ down_weight

    // Input: [1, 2] (1x2)
    auto input = createTensor({1, 2}, {1.0f, 2.0f});

    // Gate weight: [[1, 0], [0, 1]] (2x2, identity for simplicity)
    auto gate_weight = createTensor({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});

    // Up weight: [[2, 0], [0, 2]] (2x2, scale by 2)
    auto up_weight = createTensor({2, 2}, {2.0f, 0.0f, 0.0f, 2.0f});

    // Down weight: [[0.5, 0], [0, 0.5]] (2x2, scale by 0.5)
    auto down_weight = createTensor({2, 2}, {0.5f, 0.0f, 0.0f, 0.5f});

    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, gate_weight, up_weight, down_weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // Expected computation:
    // gate_proj = [1, 2] @ [[1,0],[0,1]] = [1, 2]
    // gate = silu([1, 2]) = [silu(1), silu(2)] ≈ [0.731, 1.762]
    // up_proj = [1, 2] @ [[2,0],[0,2]] = [2, 4]
    // combined = gate * up_proj = [0.731*2, 1.762*4] = [1.462, 7.048]
    // output = [1.462, 7.048] @ [[0.5,0],[0,0.5]] = [0.731, 3.524]

    float expected_gate_0 = silu(1.0f);
    float expected_gate_1 = silu(2.0f);
    std::vector<float> expected = {
        expected_gate_0 * 2.0f * 0.5f, // 0.731 * 2 * 0.5 ≈ 0.731
        expected_gate_1 * 4.0f * 0.5f  // 1.762 * 4 * 0.5 ≈ 3.524
    };

    assertTensorEqual(*output, expected);
}

TEST_F(MLPKernelTest, WithBiases)
{
    // Test MLP with bias terms
    auto input = createTensor({1, 2}, {1.0f, 1.0f});

    // Simple weights (identity)
    auto gate_weight = createTensor({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto up_weight = createTensor({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto down_weight = createTensor({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});

    // Biases
    auto gate_bias = createTensor({2}, {0.5f, -0.5f});
    auto up_bias = createTensor({2}, {1.0f, 2.0f});
    auto down_bias = createTensor({2}, {0.1f, 0.2f});

    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {
        input, gate_weight, up_weight, down_weight,
        gate_bias, up_bias, down_bias};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // Expected computation:
    // gate_proj = [1, 1] + [0.5, -0.5] = [1.5, 0.5]
    // gate = silu([1.5, 0.5])
    // up_proj = [1, 1] + [1, 2] = [2, 3]
    // combined = gate * up_proj
    // output = combined + [0.1, 0.2]

    float gate_0 = silu(1.5f);
    float gate_1 = silu(0.5f);
    std::vector<float> expected = {
        gate_0 * 2.0f + 0.1f,
        gate_1 * 3.0f + 0.2f};

    assertTensorEqual(*output, expected);
}

TEST_F(MLPKernelTest, BatchProcessing)
{
    // Test with batch input
    auto input = createTensor({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});

    // Identity weights for simplicity
    auto gate_weight = createTensor({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto up_weight = createTensor({2, 2}, {2.0f, 0.0f, 0.0f, 2.0f});
    auto down_weight = createTensor({2, 2}, {0.5f, 0.0f, 0.0f, 0.5f});

    auto output = createTensor({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, gate_weight, up_weight, down_weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // First row: [1, 0]
    // gate = silu([1, 0]) = [silu(1), silu(0)]
    // up = [2, 0]
    // combined = [silu(1)*2, silu(0)*0] = [silu(1)*2, 0]
    // output = [silu(1)*2*0.5, 0] = [silu(1), 0]

    // Second row: [0, 1]
    // gate = silu([0, 1]) = [silu(0), silu(1)]
    // up = [0, 2]
    // combined = [0, silu(1)*2]
    // output = [0, silu(1)]

    float silu_1 = silu(1.0f);
    float silu_0 = silu(0.0f);
    std::vector<float> expected = {
        silu_1, silu_0, // First batch
        silu_0, silu_1  // Second batch
    };

    assertTensorEqual(*output, expected);
}

TEST_F(MLPKernelTest, ZeroInput)
{
    // Test with zero input
    auto input = createTensor({1, 2}, {0.0f, 0.0f});

    auto gate_weight = createTensor({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto up_weight = createTensor({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto down_weight = createTensor({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});

    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, gate_weight, up_weight, down_weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // gate_proj = [0, 0], gate = silu([0, 0]) = [0, 0]
    // up_proj = [0, 0]
    // combined = [0, 0]
    // output = [0, 0]
    std::vector<float> expected = {0.0f, 0.0f};
    assertTensorEqual(*output, expected);
}

TEST_F(MLPKernelTest, SiLUActivationFunction)
{
    // Test specifically the SiLU activation behavior
    auto input = createTensor({1, 4}, {-2.0f, -1.0f, 1.0f, 2.0f});

    // Identity weights to isolate SiLU testing
    auto gate_weight = createTensor({4, 4}, {1.0f, 0.0f, 0.0f, 0.0f,
                                             0.0f, 1.0f, 0.0f, 0.0f,
                                             0.0f, 0.0f, 1.0f, 0.0f,
                                             0.0f, 0.0f, 0.0f, 1.0f});
    auto up_weight = createTensor({4, 4}, {1.0f, 0.0f, 0.0f, 0.0f,
                                           0.0f, 1.0f, 0.0f, 0.0f,
                                           0.0f, 0.0f, 1.0f, 0.0f,
                                           0.0f, 0.0f, 0.0f, 1.0f});
    auto down_weight = createTensor({4, 4}, {1.0f, 0.0f, 0.0f, 0.0f,
                                             0.0f, 1.0f, 0.0f, 0.0f,
                                             0.0f, 0.0f, 1.0f, 0.0f,
                                             0.0f, 0.0f, 0.0f, 1.0f});

    auto output = createTensor({1, 4}, {0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, gate_weight, up_weight, down_weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // Expected: silu(input) * input = silu(x) * x for each x
    std::vector<float> expected = {
        silu(-2.0f) * (-2.0f),
        silu(-1.0f) * (-1.0f),
        silu(1.0f) * 1.0f,
        silu(2.0f) * 2.0f};

    assertTensorEqual(*output, expected);
}

TEST_F(MLPKernelTest, InputValidation)
{
    // Test with insufficient inputs
    auto input = createTensor({1, 2}, {1.0f, 2.0f});
    auto gate_weight = createTensor({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, gate_weight}; // Missing weights
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

TEST_F(MLPKernelTest, DimensionMismatch)
{
    // Test with incompatible weight dimensions
    auto input = createTensor({1, 3}, {1.0f, 2.0f, 3.0f});
    auto gate_weight = createTensor({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}); // Wrong input size
    auto up_weight = createTensor({3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
    auto down_weight = createTensor({2, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    auto output = createTensor({1, 3}, {0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, gate_weight, up_weight, down_weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

// Performance test
TEST_F(MLPKernelTest, DISABLED_PerformanceTest)
{
    const int batch_size = 32;
    const int hidden_size = 1024;
    const int intermediate_size = 2048;

    std::vector<float> input_data(batch_size * hidden_size, 0.1f);
    std::vector<float> gate_weight_data(hidden_size * intermediate_size, 0.01f);
    std::vector<float> up_weight_data(hidden_size * intermediate_size, 0.01f);
    std::vector<float> down_weight_data(intermediate_size * hidden_size, 0.01f);
    std::vector<float> output_data(batch_size * hidden_size, 0.0f);

    auto input = createTensor({batch_size, hidden_size}, input_data);
    auto gate_weight = createTensor({hidden_size, intermediate_size}, gate_weight_data);
    auto up_weight = createTensor({hidden_size, intermediate_size}, up_weight_data);
    auto down_weight = createTensor({intermediate_size, hidden_size}, down_weight_data);
    auto output = createTensor({batch_size, hidden_size}, output_data);

    std::vector<std::shared_ptr<Tensor>> inputs = {input, gate_weight, up_weight, down_weight};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    auto start = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(kernel->execute(inputs, outputs));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "MLP kernel performance test: " << duration.count() << " ms" << std::endl;
}