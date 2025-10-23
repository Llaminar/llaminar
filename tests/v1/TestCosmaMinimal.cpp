// Minimal COSMA Custom Layout Test
//
// This test validates that COSMA's custom layout feature works
// with simple row-major matrices using the C interface.

#include <iostream>
#include <vector>
#include <mpi.h>
#include <cosma/cinterface.hpp>
#include <cmath>

void test_cosma_custom_layout_minimal()
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Test small matrices first: A(4x4) * B(4x4) = C(4x4)
    const int m = 4, n = 4, k = 4;

    // Initialize matrices (same on all ranks for simplicity)
    std::vector<float> A(m * k, 1.0f); // All ones
    std::vector<float> B(k * n, 2.0f); // All twos
    std::vector<float> C(m * n, 0.0f); // Zeros

    // Expected result: C = A * B = 4 * 2 = 8 (for all elements)

    // Create simple layout: each rank owns all data (replicated)
    // This is not distributed, but tests the interface

    std::vector<int> rowsplit = {0, m};   // Single row block
    std::vector<int> colsplit = {0, k};   // Single col block for A
    std::vector<int> colsplit_B = {0, n}; // Single col block for B
    std::vector<int> colsplit_C = {0, n}; // Single col block for C

    std::vector<int> owners = {0}; // Rank 0 owns the single block

    // Only rank 0 has the block
    ::block block_A = {A.data(), k, 0, 0};
    ::block block_B = {B.data(), n, 0, 0};
    ::block block_C = {C.data(), n, 0, 0};

    ::layout layout_A = {
        1,                             // rowblocks
        1,                             // colblocks
        rowsplit.data(),               // rowsplit
        colsplit.data(),               // colsplit
        owners.data(),                 // owners
        rank == 0 ? 1 : 0,             // nlocalblocks
        rank == 0 ? &block_A : nullptr // localblocks
    };

    ::layout layout_B = {
        1, 1,
        colsplit.data(), // Use k for rows of B
        colsplit_B.data(),
        owners.data(),
        rank == 0 ? 1 : 0,
        rank == 0 ? &block_B : nullptr};

    ::layout layout_C = {
        1, 1,
        rowsplit.data(),
        colsplit_C.data(),
        owners.data(),
        rank == 0 ? 1 : 0,
        rank == 0 ? &block_C : nullptr};

    // Perform multiplication
    char transa = 'N', transb = 'N';
    float alpha = 1.0f, beta = 0.0f;

    if (rank == 0)
    {
        std::cout << "Testing minimal COSMA custom layout..." << std::endl;
    }

    try
    {
        smultiply_using_layout(MPI_COMM_WORLD,
                               &transa, &transb,
                               &alpha,
                               &layout_A, &layout_B,
                               &beta,
                               &layout_C);

        if (rank == 0)
        {
            std::cout << "COSMA call completed successfully!" << std::endl;

            // Check result
            bool correct = true;
            for (int i = 0; i < m * n; ++i)
            {
                if (std::abs(C[i] - 8.0f) > 1e-5f)
                {
                    correct = false;
                    std::cout << "Error: C[" << i << "] = " << C[i] << ", expected 8.0" << std::endl;
                }
            }

            if (correct)
            {
                std::cout << "✓ COSMA custom layout test PASSED!" << std::endl;
            }
            else
            {
                std::cout << "✗ COSMA custom layout test FAILED!" << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        if (rank == 0)
        {
            std::cout << "✗ COSMA custom layout test FAILED with exception: " << e.what() << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    test_cosma_custom_layout_minimal();

    MPI_Finalize();
    return 0;
}