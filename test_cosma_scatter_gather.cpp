/**
 * @file test_cosma_scatter_gather.cpp
 * @brief Minimal reproducer for COSMA scatter/gather coordinate mapping
 */

#include "src/cosma_prefill_manager.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

using namespace llaminar;

bool test_roundtrip(int m, int k, int rank, int world_size)
{
    if (rank == 0)
    {
        std::cout << "\n=== Testing " << m << "x" << k << " matrix ===" << std::endl;
    }

    // Create test input (same on all ranks)
    std::vector<float> input(m * k);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &val : input)
    {
        val = dist(gen);
    }

    // Get CosmaPrefillManager instance
    CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
    auto strategy = mgr.strategy_for(m, k, k);

    // Scatter
    auto cosma_view = mgr.convert_activation_operand(input.data(), m, k, strategy);

    // Gather
    std::vector<float> output(m * k, 0.0f);
    mgr.to_row_major(cosma_view, output.data());

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        double max_abs = 0.0;
        int mismatches = 0;

        for (int i = 0; i < m * k; ++i)
        {
            double err = std::abs(input[i] - output[i]);
            if (err > 1e-5)
            {
                mismatches++;
                if (mismatches <= 3)
                {
                    std::cout << "  Mismatch [" << i << "]: "
                              << input[i] << " → " << output[i]
                              << " (err=" << err << ")" << std::endl;
                }
            }
            max_abs = std::max(max_abs, err);
        }

        std::cout << "  Max error: " << max_abs << std::endl;
        std::cout << "  Mismatches: " << mismatches << " / " << (m * k) << std::endl;
        std::cout << "  " << ((max_abs < 1e-5) ? "✅ PASSED" : "❌ FAILED") << std::endl;

        return max_abs < 1e-5;
    }

    return true;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0)
    {
        std::cout << "=== COSMA Scatter/Gather Test ===" << std::endl;
        std::cout << "Ranks: " << world_size << std::endl;
    }

    bool all_passed = true;
    all_passed &= test_roundtrip(8, 8, rank, world_size);
    all_passed &= test_roundtrip(32, 896, rank, world_size);
    all_passed &= test_roundtrip(32, 10000, rank, world_size);

    if (rank == 0)
    {
        std::cout << "\n"
                  << (all_passed ? "✅ ALL PASSED" : "❌ FAILED") << std::endl;
    }

    MPI_Finalize();
    return all_passed ? 0 : 1;
}
