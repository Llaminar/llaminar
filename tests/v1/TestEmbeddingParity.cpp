/**
 * @file TestEmbeddingParity.cpp
 * @brief Minimal test to compare embedding lookup between Llaminar and PyTorch
 *
 * This is a focused diagnostic to find the root cause of EMBEDDING divergence.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <fstream>
#include <cmath>

#include "../src/Logger.h"
#include "../src/ModelLoader.h"
#include "../src/operators/MPIEmbeddingOperator.h"
#include "../src/TensorFactory.h"

// Test tokens: [1, 2, 3, 4, 5]
const std::vector<int> TEST_TOKENS = {1, 2, 3, 4, 5};

TEST(EmbeddingParity, CompareWithPyTorch)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Load model
    const char *model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    ModelLoader loader(model_path);
    if (!loader.load())
    {
        FAIL() << "Failed to load model";
    }

    auto weights = loader.getWeights();
    auto embedding_table = weights->get("token_embd.weight");

    if (!embedding_table)
    {
        FAIL() << "Embedding table not found";
    }

    LOG_INFO("Embedding table shape: " << embedding_table->shape()[0]
                                       << " x " << embedding_table->shape()[1]);

    // Create embedding operator
    MPIEmbeddingOperator emb_kernel(151936, 896);

    // Create input/output tensors
    auto token_tensor = TensorFactory::create_simple({static_cast<int>(TEST_TOKENS.size()), 1});
    for (size_t i = 0; i < TEST_TOKENS.size(); ++i)
    {
        token_tensor->data()[i] = static_cast<float>(TEST_TOKENS[i]);
    }

    auto output = TensorFactory::create_simple({static_cast<int>(TEST_TOKENS.size()), 896});

    // Execute embedding lookup
    std::vector<std::shared_ptr<TensorBase>> inputs = {token_tensor, embedding_table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    ASSERT_TRUE(emb_kernel.execute(inputs, outputs));

    if (rank == 0)
    {
        LOG_INFO("Llaminar embedding for token 0 (first 20 dims):");
        for (int i = 0; i < 20; ++i)
        {
            std::cout << output->data()[i] << " ";
        }
        std::cout << std::endl;

        LOG_INFO("Llaminar embedding for token 1 (first 20 dims):");
        for (int i = 0; i < 20; ++i)
        {
            std::cout << output->data()[896 + i] << " ";
        }
        std::cout << std::endl;

        // Load PyTorch reference
        // TODO: Actually load and compare the .npy file
        std::cout << "\nTo compare with PyTorch:" << std::endl;
        std::cout << "python3 -c \"import numpy as np; emb = np.load('/tmp/pytorch_snapshots_openblas/EMBEDDING_-1.npy'); print('PyTorch token 0:', emb[0,0,:20]); print('PyTorch token 1:', emb[0,1,:20])\"" << std::endl;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        Logger::init(LogLevel::INFO);
    }
    else
    {
        Logger::init(LogLevel::WARN);
    }

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
