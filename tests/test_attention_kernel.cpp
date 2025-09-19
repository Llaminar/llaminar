#include <gtest/gtest.h>
#include "../src/kernels/AttentionKernel.h"
#include "graph_compute.h"
#include <memory>
#include <chrono>
#include <cmath>

using namespace llaminar;

class AttentionKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        kernel = std::make_unique<AttentionKernel>();
        kernel->setHeadDimensions(2, 2, 2); // 2 heads, 2 KV heads, 2 dimensions per head
    }

    std::unique_ptr<AttentionKernel> kernel;

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
};

TEST_F(AttentionKernelTest, BasicAttentionComputation)
{
    // Simple attention test with small dimensions
    const int seq_len = 1;
    const int hidden_size = 4; // 2 heads * 2 head_dim
    const int max_seq_len = 10;

    // Input: [1, 4]
    auto input = createTensor({seq_len, hidden_size}, {1.0f, 2.0f, 3.0f, 4.0f});

    // Identity weights for Q, K, V projections
    auto q_weight = createTensor({hidden_size, hidden_size}, {1.0f, 0.0f, 0.0f, 0.0f,
                                                              0.0f, 1.0f, 0.0f, 0.0f,
                                                              0.0f, 0.0f, 1.0f, 0.0f,
                                                              0.0f, 0.0f, 0.0f, 1.0f});

    auto k_weight = createTensor({hidden_size, hidden_size}, {1.0f, 0.0f, 0.0f, 0.0f,
                                                              0.0f, 1.0f, 0.0f, 0.0f,
                                                              0.0f, 0.0f, 1.0f, 0.0f,
                                                              0.0f, 0.0f, 0.0f, 1.0f});

    auto v_weight = createTensor({hidden_size, hidden_size}, {1.0f, 0.0f, 0.0f, 0.0f,
                                                              0.0f, 1.0f, 0.0f, 0.0f,
                                                              0.0f, 0.0f, 1.0f, 0.0f,
                                                              0.0f, 0.0f, 0.0f, 1.0f});

    auto output_weight = createTensor({hidden_size, hidden_size}, {1.0f, 0.0f, 0.0f, 0.0f,
                                                                   0.0f, 1.0f, 0.0f, 0.0f,
                                                                   0.0f, 0.0f, 1.0f, 0.0f,
                                                                   0.0f, 0.0f, 0.0f, 1.0f});

    // KV cache (initially zeros)
    auto k_cache = createTensor({max_seq_len, hidden_size}, std::vector<float>(max_seq_len * hidden_size, 0.0f));
    auto v_cache = createTensor({max_seq_len, hidden_size}, std::vector<float>(max_seq_len * hidden_size, 0.0f));

    auto output = createTensor({seq_len, hidden_size}, {0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {
        input, q_weight, k_weight, v_weight, output_weight, k_cache, v_cache};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // With the simplified attention implementation, we expect some combination of Q and V
    // The exact values depend on the implementation details
    ASSERT_EQ(output->data.size(), 4);

    // Verify the output is not all zeros (attention should produce some result)
    bool hasNonZero = false;
    for (float val : output->data)
    {
        if (std::abs(val) > 1e-6f)
        {
            hasNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZero);
}

TEST_F(AttentionKernelTest, KVCacheUpdate)
{
    // Test that KV cache is properly updated
    const int seq_len = 2;
    const int hidden_size = 4;
    const int max_seq_len = 10;

    auto input = createTensor({seq_len, hidden_size}, {1.0f, 2.0f, 3.0f, 4.0f,
                                                       5.0f, 6.0f, 7.0f, 8.0f});

    // Identity weights
    auto q_weight = createTensor({hidden_size, hidden_size}, {1.0f, 0.0f, 0.0f, 0.0f,
                                                              0.0f, 1.0f, 0.0f, 0.0f,
                                                              0.0f, 0.0f, 1.0f, 0.0f,
                                                              0.0f, 0.0f, 0.0f, 1.0f});

    auto k_weight = q_weight;
    auto v_weight = q_weight;
    auto output_weight = q_weight;

    auto k_cache = createTensor({max_seq_len, hidden_size}, std::vector<float>(max_seq_len * hidden_size, 0.0f));
    auto v_cache = createTensor({max_seq_len, hidden_size}, std::vector<float>(max_seq_len * hidden_size, 0.0f));

    auto output = createTensor({seq_len, hidden_size}, std::vector<float>(seq_len * hidden_size, 0.0f));

    std::vector<std::shared_ptr<Tensor>> inputs = {
        input, q_weight, k_weight, v_weight, output_weight, k_cache, v_cache};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // Check that K and V caches have been updated with input values
    // First sequence should be in cache positions 0-1
    for (int i = 0; i < seq_len * hidden_size; ++i)
    {
        EXPECT_NE(k_cache->data[i], 0.0f) << "K cache should be updated at position " << i;
        EXPECT_NE(v_cache->data[i], 0.0f) << "V cache should be updated at position " << i;
    }
}

TEST_F(AttentionKernelTest, MultipleSequenceLength)
{
    // Test with longer sequence
    const int seq_len = 3;
    const int hidden_size = 4;
    const int max_seq_len = 10;

    auto input = createTensor({seq_len, hidden_size}, {1.0f, 0.0f, 0.0f, 0.0f,
                                                       0.0f, 1.0f, 0.0f, 0.0f,
                                                       0.0f, 0.0f, 1.0f, 0.0f});

    // Simple scaling weights
    auto q_weight = createTensor({hidden_size, hidden_size}, {2.0f, 0.0f, 0.0f, 0.0f,
                                                              0.0f, 2.0f, 0.0f, 0.0f,
                                                              0.0f, 0.0f, 2.0f, 0.0f,
                                                              0.0f, 0.0f, 0.0f, 2.0f});

    auto k_weight = createTensor({hidden_size, hidden_size}, {1.0f, 0.0f, 0.0f, 0.0f,
                                                              0.0f, 1.0f, 0.0f, 0.0f,
                                                              0.0f, 0.0f, 1.0f, 0.0f,
                                                              0.0f, 0.0f, 0.0f, 1.0f});

    auto v_weight = k_weight;
    auto output_weight = k_weight;

    auto k_cache = createTensor({max_seq_len, hidden_size}, std::vector<float>(max_seq_len * hidden_size, 0.0f));
    auto v_cache = createTensor({max_seq_len, hidden_size}, std::vector<float>(max_seq_len * hidden_size, 0.0f));

    auto output = createTensor({seq_len, hidden_size}, std::vector<float>(seq_len * hidden_size, 0.0f));

    std::vector<std::shared_ptr<Tensor>> inputs = {
        input, q_weight, k_weight, v_weight, output_weight, k_cache, v_cache};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // Verify output has reasonable values
    ASSERT_EQ(output->data.size(), seq_len * hidden_size);

    // Check that some computation occurred
    float sum = 0.0f;
    for (float val : output->data)
    {
        sum += std::abs(val);
    }
    EXPECT_GT(sum, 0.0f) << "Output should not be all zeros";
}

TEST_F(AttentionKernelTest, InputValidation)
{
    // Test with insufficient inputs
    auto input = createTensor({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto weight = createTensor({4, 4}, std::vector<float>(16, 1.0f));
    auto output = createTensor({1, 4}, {0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {input, weight}; // Missing inputs
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

TEST_F(AttentionKernelTest, OutputValidation)
{
    // Test with wrong number of outputs
    const int seq_len = 1;
    const int hidden_size = 4;
    const int max_seq_len = 10;

    auto input = createTensor({seq_len, hidden_size}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto q_weight = createTensor({hidden_size, hidden_size}, std::vector<float>(16, 1.0f));
    auto k_weight = createTensor({hidden_size, hidden_size}, std::vector<float>(16, 1.0f));
    auto v_weight = createTensor({hidden_size, hidden_size}, std::vector<float>(16, 1.0f));
    auto output_weight = createTensor({hidden_size, hidden_size}, std::vector<float>(16, 1.0f));
    auto k_cache = createTensor({max_seq_len, hidden_size}, std::vector<float>(max_seq_len * hidden_size, 0.0f));
    auto v_cache = createTensor({max_seq_len, hidden_size}, std::vector<float>(max_seq_len * hidden_size, 0.0f));

    auto output1 = createTensor({seq_len, hidden_size}, std::vector<float>(seq_len * hidden_size, 0.0f));
    auto output2 = createTensor({seq_len, hidden_size}, std::vector<float>(seq_len * hidden_size, 0.0f));

    std::vector<std::shared_ptr<Tensor>> inputs = {
        input, q_weight, k_weight, v_weight, output_weight, k_cache, v_cache};
    std::vector<std::shared_ptr<Tensor>> outputs = {output1, output2}; // Too many outputs

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

TEST_F(AttentionKernelTest, DimensionMismatch)
{
    // Test with mismatched tensor dimensions
    const int seq_len = 1;
    const int hidden_size = 4;
    const int max_seq_len = 10;

    auto input = createTensor({seq_len, hidden_size}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto q_weight = createTensor({3, 4}, std::vector<float>(12, 1.0f)); // Wrong input dimension
    auto k_weight = createTensor({hidden_size, hidden_size}, std::vector<float>(16, 1.0f));
    auto v_weight = createTensor({hidden_size, hidden_size}, std::vector<float>(16, 1.0f));
    auto output_weight = createTensor({hidden_size, hidden_size}, std::vector<float>(16, 1.0f));
    auto k_cache = createTensor({max_seq_len, hidden_size}, std::vector<float>(max_seq_len * hidden_size, 0.0f));
    auto v_cache = createTensor({max_seq_len, hidden_size}, std::vector<float>(max_seq_len * hidden_size, 0.0f));

    auto output = createTensor({seq_len, hidden_size}, std::vector<float>(seq_len * hidden_size, 0.0f));

    std::vector<std::shared_ptr<Tensor>> inputs = {
        input, q_weight, k_weight, v_weight, output_weight, k_cache, v_cache};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    // Should fail due to dimension mismatch
    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

TEST_F(AttentionKernelTest, HeadDimensionConfiguration)
{
    // Test that head dimensions can be configured
    kernel->setHeadDimensions(4, 2, 64); // 4 heads, 2 KV heads, 64 dims per head

    // This test mainly verifies the setter works without errors
    // More comprehensive testing would require a more detailed attention implementation
    SUCCEED();
}

// Performance test
TEST_F(AttentionKernelTest, DISABLED_PerformanceTest)
{
    const int seq_len = 128;
    const int hidden_size = 512;
    const int max_seq_len = 2048;

    kernel->setHeadDimensions(8, 8, 64); // 8 heads, 8 KV heads, 64 dims per head

    std::vector<float> input_data(seq_len * hidden_size, 0.1f);
    std::vector<float> weight_data(hidden_size * hidden_size, 0.01f);
    std::vector<float> cache_data(max_seq_len * hidden_size, 0.0f);

    auto input = createTensor({seq_len, hidden_size}, input_data);
    auto q_weight = createTensor({hidden_size, hidden_size}, weight_data);
    auto k_weight = createTensor({hidden_size, hidden_size}, weight_data);
    auto v_weight = createTensor({hidden_size, hidden_size}, weight_data);
    auto output_weight = createTensor({hidden_size, hidden_size}, weight_data);
    auto k_cache = createTensor({max_seq_len, hidden_size}, cache_data);
    auto v_cache = createTensor({max_seq_len, hidden_size}, cache_data);

    auto output = createTensor({seq_len, hidden_size}, std::vector<float>(seq_len * hidden_size, 0.0f));

    std::vector<std::shared_ptr<Tensor>> inputs = {
        input, q_weight, k_weight, v_weight, output_weight, k_cache, v_cache};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    auto start = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(kernel->execute(inputs, outputs));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Attention kernel performance test: " << duration.count() << " ms" << std::endl;
}