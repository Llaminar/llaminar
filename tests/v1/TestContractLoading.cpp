/**
 * @file TestContractLoading.cpp
 * @brief Simple test to verify contract-driven weight loading
 * @author David Sanftenberg
 */

#include "ModelLoader.h"
#include "WeightContracts.h"
#include "TransformerConfig.h"
#include "Logger.h"
#include <mpi.h>
#include <iostream>
#include <memory>

int main(int, char **)
{
    // Initialize MPI
    MPI_Init(NULL, NULL);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    try
    {
        // Hard-coded Qwen 0.5B config (from qwen_pipeline.cpp)
        TransformerLayerConfig config;
        config.d_model = 896;
        config.n_head = 14;
        config.n_head_kv = 2;
        config.head_dim = 64;
        config.d_ff = 4864;
        config.vocab_size = 151666;
        config.max_seq_len = 131072;

        const char *model_path = "../models/qwen2.5-0.5b-instruct-q2_k.gguf";

        if (mpi_rank == 0)
        {
            std::cout << "Testing contract-driven loading with " << mpi_size << " ranks" << std::endl;
            std::cout << "Model: " << model_path << std::endl;
        }

        ModelLoader loader;
        if (!loader.loadModel(model_path))
        {
            std::cerr << "[Rank " << mpi_rank << "] Failed to load model!" << std::endl;
            MPI_Finalize();
            return 1;
        }

        if (mpi_rank == 0)
        {
            std::cout << "\nConfig: d_model=" << config.d_model
                      << " n_head=" << config.n_head
                      << " n_head_kv=" << config.n_head_kv
                      << " head_dim=" << config.head_dim
                      << " d_ff=" << config.d_ff << std::endl;
        }

        // Get contracts
        auto contracts = llaminar::getQwenWeightContracts();

        // Test loading layer 0 weights
        const int layer = 0;

        if (mpi_rank == 0)
        {
            std::cout << "\n=== Loading Layer " << layer << " Weights ===" << std::endl;
        }

        // Load Q weight
        auto wq = contracts.layer_weights[1].load(loader, config, mpi_rank, mpi_size, layer);
        std::cout << "[Rank " << mpi_rank << "] Q weight shape: ["
                  << wq->shape()[0] << ", " << wq->shape()[1] << "]" << std::endl;

        // Load K weight
        auto wk = contracts.layer_weights[2].load(loader, config, mpi_rank, mpi_size, layer);
        std::cout << "[Rank " << mpi_rank << "] K weight shape: ["
                  << wk->shape()[0] << ", " << wk->shape()[1] << "]" << std::endl;

        // Load V weight
        auto wv = contracts.layer_weights[3].load(loader, config, mpi_rank, mpi_size, layer);
        std::cout << "[Rank " << mpi_rank << "] V weight shape: ["
                  << wv->shape()[0] << ", " << wv->shape()[1] << "]" << std::endl;

        // Load O weight
        auto wo = contracts.layer_weights[4].load(loader, config, mpi_rank, mpi_size, layer);
        std::cout << "[Rank " << mpi_rank << "] O weight shape: ["
                  << wo->shape()[0] << ", " << wo->shape()[1] << "]" << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);

        if (mpi_rank == 0)
        {
            std::cout << "\n✅ All weights loaded successfully!" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Rank " << mpi_rank << "] ERROR: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
