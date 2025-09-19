#include <gtest/gtest.h>
#include "../src/kernels/EmbeddingKernel.h"
#include "graph_compute.h"
#include <memory>
#include <chrono>

using namespace llaminar;

class EmbeddingKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        kernel = std::make_unique<EmbeddingKernel>();
    }

    std::unique_ptr<EmbeddingKernel> kernel;

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

TEST_F(EmbeddingKernelTest, BasicEmbeddingLookup)
{
    // Embedding table: 4 tokens, 3 dimensions each
    auto embedding_table = createTensor({4, 3}, {
                                                    1.0f, 2.0f, 3.0f,   // Token 0
                                                    4.0f, 5.0f, 6.0f,   // Token 1
                                                    7.0f, 8.0f, 9.0f,   // Token 2
                                                    10.0f, 11.0f, 12.0f // Token 3
                                                });

    // Token IDs: [1, 3, 0]
    auto token_ids = createTensor({3}, {1.0f, 3.0f, 0.0f}); // As floats (will be cast to int)

    auto output = createTensor({3, 3}, std::vector<float>(9, 0.0f));

    std::vector<std::shared_ptr<Tensor>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    // Expected: embeddings for tokens [1, 3, 0]
    std::vector<float> expected = {
        4.0f, 5.0f, 6.0f,    // Token 1
        10.0f, 11.0f, 12.0f, // Token 3
        1.0f, 2.0f, 3.0f     // Token 0
    };
    assertTensorEqual(*output, expected);
}

TEST_F(EmbeddingKernelTest, SingleTokenLookup)
{
    // Simple case: lookup one token
    auto embedding_table = createTensor({3, 2}, {
                                                    1.5f, 2.5f, // Token 0
                                                    3.5f, 4.5f, // Token 1
                                                    5.5f, 6.5f  // Token 2
                                                });

    auto token_ids = createTensor({1}, {2.0f}); // Token 2
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    std::vector<float> expected = {5.5f, 6.5f};
    assertTensorEqual(*output, expected);
}

TEST_F(EmbeddingKernelTest, RepeatedTokens)
{
    // Test with repeated token IDs
    auto embedding_table = createTensor({3, 2}, {
                                                    10.0f, 20.0f, // Token 0
                                                    30.0f, 40.0f, // Token 1
                                                    50.0f, 60.0f  // Token 2
                                                });

    auto token_ids = createTensor({4}, {0.0f, 1.0f, 0.0f, 2.0f});
    auto output = createTensor({4, 2}, std::vector<float>(8, 0.0f));

    std::vector<std::shared_ptr<Tensor>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    std::vector<float> expected = {
        10.0f, 20.0f, // Token 0
        30.0f, 40.0f, // Token 1
        10.0f, 20.0f, // Token 0 (repeated)
        50.0f, 60.0f  // Token 2
    };
    assertTensorEqual(*output, expected);
}

TEST_F(EmbeddingKernelTest, LargeEmbeddingDimension)
{
    // Test with larger embedding dimension
    const int vocab_size = 3;
    const int embed_dim = 5;

    auto embedding_table = createTensor({vocab_size, embed_dim}, {
                                                                     1.0f, 1.1f, 1.2f, 1.3f, 1.4f, // Token 0
                                                                     2.0f, 2.1f, 2.2f, 2.3f, 2.4f, // Token 1
                                                                     3.0f, 3.1f, 3.2f, 3.3f, 3.4f  // Token 2
                                                                 });

    auto token_ids = createTensor({2}, {1.0f, 2.0f});
    auto output = createTensor({2, embed_dim}, std::vector<float>(10, 0.0f));

    std::vector<std::shared_ptr<Tensor>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    std::vector<float> expected = {
        2.0f, 2.1f, 2.2f, 2.3f, 2.4f, // Token 1
        3.0f, 3.1f, 3.2f, 3.3f, 3.4f  // Token 2
    };
    assertTensorEqual(*output, expected);
}

TEST_F(EmbeddingKernelTest, OutOfBoundsToken)
{
    // Test with out-of-bounds token ID (should be handled gracefully)
    auto embedding_table = createTensor({2, 2}, {
                                                    1.0f, 2.0f, // Token 0
                                                    3.0f, 4.0f  // Token 1
                                                });

    auto token_ids = createTensor({1}, {5.0f}); // Out of bounds
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    // Should return false due to bounds checking
    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

TEST_F(EmbeddingKernelTest, NegativeTokenId)
{
    // Test with negative token ID
    auto embedding_table = createTensor({2, 2}, {
                                                    1.0f, 2.0f, // Token 0
                                                    3.0f, 4.0f  // Token 1
                                                });

    auto token_ids = createTensor({1}, {-1.0f}); // Negative
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    // Should return false due to bounds checking
    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

TEST_F(EmbeddingKernelTest, FloatToIntConversion)
{
    // Test that float token IDs are properly converted to integers
    auto embedding_table = createTensor({3, 2}, {
                                                    10.0f, 20.0f, // Token 0
                                                    30.0f, 40.0f, // Token 1
                                                    50.0f, 60.0f  // Token 2
                                                });

    auto token_ids = createTensor({3}, {0.9f, 1.1f, 2.7f}); // Should become [0, 1, 2]
    auto output = createTensor({3, 2}, std::vector<float>(6, 0.0f));

    std::vector<std::shared_ptr<Tensor>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_TRUE(kernel->execute(inputs, outputs));

    std::vector<float> expected = {
        10.0f, 20.0f, // Token 0 (from 0.9)
        30.0f, 40.0f, // Token 1 (from 1.1)
        50.0f, 60.0f  // Token 2 (from 2.7)
    };
    assertTensorEqual(*output, expected);
}

TEST_F(EmbeddingKernelTest, InputValidation)
{
    // Test with wrong number of inputs
    auto embedding_table = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto output = createTensor({1, 2}, {0.0f, 0.0f});

    std::vector<std::shared_ptr<Tensor>> inputs = {embedding_table}; // Missing token_ids
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

TEST_F(EmbeddingKernelTest, OutputShapeValidation)
{
    // Test with wrong output shape
    auto embedding_table = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto token_ids = createTensor({2}, {0.0f, 1.0f});
    auto output = createTensor({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}); // Wrong embed_dim

    std::vector<std::shared_ptr<Tensor>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    ASSERT_FALSE(kernel->execute(inputs, outputs));
}

// Performance test
TEST_F(EmbeddingKernelTest, DISABLED_PerformanceTest)
{
    const int vocab_size = 50000;
    const int embed_dim = 512;
    const int seq_len = 2048;

    // Create large embedding table
    std::vector<float> table_data(vocab_size * embed_dim);
    for (int i = 0; i < vocab_size * embed_dim; ++i)
    {
        table_data[i] = static_cast<float>(i % 100) * 0.01f;
    }

    // Create token sequence
    std::vector<float> token_data(seq_len);
    for (int i = 0; i < seq_len; ++i)
    {
        token_data[i] = static_cast<float>(i % vocab_size);
    }

    auto embedding_table = createTensor({vocab_size, embed_dim}, table_data);
    auto token_ids = createTensor({seq_len}, token_data);
    auto output = createTensor({seq_len, embed_dim}, std::vector<float>(seq_len * embed_dim, 0.0f));

    std::vector<std::shared_ptr<Tensor>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<Tensor>> outputs = {output};

    auto start = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(kernel->execute(inputs, outputs));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Embedding kernel performance test: " << duration.count() << " ms" << std::endl;
}