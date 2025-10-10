/**
 * @file test_mpi_attention_real_weights.cpp
 * @brief Test MPIAttentionKernel with real GGUF model weights
 * @author David Sanftenberg
 *
 * This test loads actual model weights from GGUF files to test attention
 * with real, validated data instead of synthetic test data.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <filesystem>
#include "kernels/MPIAttentionKernel.h"
#include "model_loader.h"
#include "tensors/tensor_factory.h"
#include "logger.h"

using namespace llaminar;

namespace
{

    struct TestMPIContext
    {
        int rank;
        int world_size;

        TestMPIContext()
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        }
    };

    /**
     * Find a test model file
     */
    std::string findTestModel()
    {
        const char *candidates[] = {
            "models/qwen2.5-0.5b-instruct-q4_0.gguf",
            "models/qwen2.5-0.5b-instruct-q3_k_m.gguf",
            "models/qwen2.5-0.5b-instruct-q2_k.gguf",
            "../models/qwen2.5-0.5b-instruct-q4_0.gguf"};

        for (const char *path : candidates)
        {
            if (std::filesystem::exists(path))
            {
                return path;
            }
        }
        return "";
    }

    /**
     * Extract a layer's attention weights from loaded model
     */
    struct AttentionWeights
    {
        std::shared_ptr<TensorBase> wq;
        std::shared_ptr<TensorBase> wk;
        std::shared_ptr<TensorBase> wv;
        std::shared_ptr<TensorBase> wo;
        std::shared_ptr<TensorBase> bq; // May be null if no bias
        std::shared_ptr<TensorBase> bk;
        std::shared_ptr<TensorBase> bv;

        int d_model;
        int n_head;
        int n_head_kv;
        int head_dim;
    };

    AttentionWeights extractAttentionWeights(ModelLoader &loader, int layer_idx = 0)
    {
        const auto &model = loader.getModel();
        AttentionWeights weights;

        // Get model config from GGUF model struct fields
        weights.d_model = model.embedding_length;
        weights.n_head = model.head_count;
        weights.n_head_kv = model.head_count_kv;
        weights.head_dim = weights.d_model / weights.n_head;

        // Find attention weights for the specified layer
        std::string layer_prefix = "blk." + std::to_string(layer_idx) + ".attn_";

        // Load and dequantize weights using ModelLoader API
        try
        {
            weights.wq = loader.loadTensor(layer_prefix + "q.weight");
            weights.wk = loader.loadTensor(layer_prefix + "k.weight");
            weights.wv = loader.loadTensor(layer_prefix + "v.weight");
            weights.wo = loader.loadTensor(layer_prefix + "output.weight");
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Failed to load attention weights for layer " +
                                     std::to_string(layer_idx) + ": " + e.what());
        }

        // Qwen models don't have attention biases - leave as nullptr
        weights.bq = nullptr;
        weights.bk = nullptr;
        weights.bv = nullptr;

        return weights;
    }

} // anonymous namespace

class MPIAttentionRealWeightsTest : public ::testing::Test
{
protected:
    TestMPIContext mpi;

    void SetUp() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

/**
 * Test basic execution with real model weights
 */
TEST_F(MPIAttentionRealWeightsTest, BasicExecutionWithRealWeights)
{
    std::string model_path = findTestModel();
    if (model_path.empty())
    {
        GTEST_SKIP() << "No test model found in models/ directory";
    }

    if (mpi.rank == 0)
    {
        LOG_INFO("Loading model from: " << model_path);
    }

    // Load model
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to load model";

    // Extract layer 0 attention weights
    auto weights = extractAttentionWeights(loader, 0);

    if (mpi.rank == 0)
    {
        LOG_INFO("Model config: d_model=" << weights.d_model
                                          << ", n_head=" << weights.n_head
                                          << ", n_head_kv=" << weights.n_head_kv
                                          << ", head_dim=" << weights.head_dim);
        LOG_INFO("Weight shapes:");
        LOG_INFO("  wq: [" << weights.wq->shape()[0] << ", " << weights.wq->shape()[1] << "]");
        LOG_INFO("  wk: [" << weights.wk->shape()[0] << ", " << weights.wk->shape()[1] << "]");
        LOG_INFO("  wv: [" << weights.wv->shape()[0] << ", " << weights.wv->shape()[1] << "]");
        LOG_INFO("  wo: [" << weights.wo->shape()[0] << ", " << weights.wo->shape()[1] << "]");
    }

    // Create test input (small sequence)
    const int seq_len = 4;
    auto input = TensorFactory::create_simple({seq_len, weights.d_model});

    // Fill input with small random values
    std::srand(42);
    for (int i = 0; i < input->size(); i++)
    {
        input->data()[i] = (std::rand() / (float)RAND_MAX) * 0.1f - 0.05f;
    }

    // Create KV cache
    auto k_cache = TensorFactory::create_simple({seq_len, weights.n_head_kv * weights.head_dim});
    auto v_cache = TensorFactory::create_simple({seq_len, weights.n_head_kv * weights.head_dim});
    k_cache->zero();
    v_cache->zero();

    // Create kernel
    MPIAttentionKernel kernel(weights.n_head, weights.n_head_kv, weights.head_dim);

    // Prepare inputs
    std::vector<std::shared_ptr<TensorBase>> inputs = {
        input,
        weights.wq, weights.wk, weights.wv, weights.wo,
        weights.bq, weights.bk, weights.bv,
        k_cache, v_cache};

    // Create output
    auto output = TensorFactory::create_simple({seq_len, weights.d_model});
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    // Execute
    bool success = kernel.execute(inputs, outputs);
    ASSERT_TRUE(success) << "Kernel execution failed with real weights";

    // Verify output is reasonable
    float sum = 0.0f;
    float max_val = -std::numeric_limits<float>::infinity();
    float min_val = std::numeric_limits<float>::infinity();

    for (int i = 0; i < output->size(); i++)
    {
        float val = output->data()[i];
        sum += std::abs(val);
        max_val = std::max(max_val, val);
        min_val = std::min(min_val, val);

        // Check for NaN/Inf
        ASSERT_FALSE(std::isnan(val)) << "Output contains NaN at index " << i;
        ASSERT_FALSE(std::isinf(val)) << "Output contains Inf at index " << i;
    }

    if (mpi.rank == 0)
    {
        LOG_INFO("Output statistics:");
        LOG_INFO("  Sum of abs values: " << sum);
        LOG_INFO("  Range: [" << min_val << ", " << max_val << "]");
    }

    // Output should be non-zero
    EXPECT_GT(sum, 0.0f) << "Output should be non-zero";

    // Output should be in reasonable range (not exploded)
    EXPECT_LT(max_val, 100.0f) << "Output values too large";
    EXPECT_GT(min_val, -100.0f) << "Output values too small";
}

/**
 * Test deterministic output with same inputs
 */
TEST_F(MPIAttentionRealWeightsTest, DeterministicOutput)
{
    std::string model_path = findTestModel();
    if (model_path.empty())
    {
        GTEST_SKIP() << "No test model found";
    }

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));
    auto weights = extractAttentionWeights(loader, 0);

    const int seq_len = 4;
    auto input = TensorFactory::create_simple({seq_len, weights.d_model});

    // Use fixed seed for reproducible input
    std::srand(12345);
    for (int i = 0; i < input->size(); i++)
    {
        input->data()[i] = (std::rand() / (float)RAND_MAX) * 0.1f - 0.05f;
    }

    auto k_cache = TensorFactory::create_simple({seq_len, weights.n_head_kv * weights.head_dim});
    auto v_cache = TensorFactory::create_simple({seq_len, weights.n_head_kv * weights.head_dim});
    k_cache->zero();
    v_cache->zero();

    MPIAttentionKernel kernel(weights.n_head, weights.n_head_kv, weights.head_dim);

    std::vector<std::shared_ptr<TensorBase>> inputs = {
        input,
        weights.wq, weights.wk, weights.wv, weights.wo,
        weights.bq, weights.bk, weights.bv,
        k_cache, v_cache};

    // Run twice
    auto output1 = TensorFactory::create_simple({seq_len, weights.d_model});
    auto output2 = TensorFactory::create_simple({seq_len, weights.d_model});

    std::vector<std::shared_ptr<TensorBase>> outputs1 = {output1};
    std::vector<std::shared_ptr<TensorBase>> outputs2 = {output2};

    ASSERT_TRUE(kernel.execute(inputs, outputs1));
    ASSERT_TRUE(kernel.execute(inputs, outputs2));

    // Outputs should be identical
    for (int i = 0; i < output1->size(); i++)
    {
        EXPECT_FLOAT_EQ(output1->data()[i], output2->data()[i])
            << "Outputs differ at index " << i;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Only rank 0 prints test output
    if (rank != 0)
    {
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
