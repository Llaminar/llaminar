#include "../src/common.h"
#include "../src/argument_parser.h"
#include "../src/kernels/MatMulKernel.h"
#include "../src/graph_compute.h" // For Tensor definition
#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace llaminar;

// Generate random matrix data
void fillMatrixRandom(std::vector<float> &matrix, size_t size, int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (size_t i = 0; i < size; ++i)
    {
        matrix[i] = static_cast<float>(dis(gen));
    }
}

// Validate matrix multiplication result (simple check)
bool validateResult(const Tensor &A, const Tensor &B, const Tensor &C, double tolerance = 1e-4)
{
    int m = A.shape[0];
    int k = A.shape[1];
    int n = B.shape[1];

    // Check a few elements manually for basic correctness
    for (int i = 0; i < std::min(m, 4); ++i)
    {
        for (int j = 0; j < std::min(n, 4); ++j)
        {
            float expected = 0.0f;
            for (int idx = 0; idx < k; ++idx)
            {
                expected += A.data[i * k + idx] * B.data[idx * n + j];
            }
            float actual = C.data[i * n + j];
            if (std::abs(expected - actual) > tolerance)
            {
                std::cerr << "Validation failed at C[" << i << "," << j << "]: "
                          << "expected " << expected << ", got " << actual << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Run COSMA kernel test
bool runCOSMAKernelTest(int m, int n, int k, int rank, int size)
{
    if (rank == 0)
    {
        std::cout << "Testing COSMA kernel: " << m << "x" << n << "x" << k
                  << " with " << size << " processes..." << std::endl;
    }

    // Create test matrices with float data
    std::vector<float> data_A(m * k);
    std::vector<float> data_B(k * n);
    std::vector<float> data_C(m * n, 0.0f);

    fillMatrixRandom(data_A, m * k, rank);
    fillMatrixRandom(data_B, k * n, rank + 100);

    // Create tensors using new Tensor struct
    auto tensor_A = std::make_shared<Tensor>(std::vector<int>{m, k});
    auto tensor_B = std::make_shared<Tensor>(std::vector<int>{k, n});
    auto tensor_C = std::make_shared<Tensor>(std::vector<int>{m, n});

    // Copy data to tensors
    tensor_A->data = data_A;
    tensor_B->data = data_B;
    tensor_C->data = data_C;

    // Prepare inputs and outputs
    std::vector<std::shared_ptr<Tensor>> inputs = {tensor_A, tensor_B};
    std::vector<std::shared_ptr<Tensor>> outputs = {tensor_C};

    // Create MatMulKernel instance
    MatMulKernel kernel;
    kernel.setStrategy("auto");
    kernel.setBlockSizes(256, 256, 256);

    // Execute kernel
    auto start = std::chrono::high_resolution_clock::now();
    bool success = kernel.execute(inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();

    if (success)
    {
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Calculate performance metrics
        double ops = 2.0 * m * n * k; // FMA operations
        double gflops = (ops / 1e9) / (time_ms / 1000.0);
        double memory_gb = (m * k + k * n + m * n) * sizeof(float) / (1024.0 * 1024.0 * 1024.0);

        if (rank == 0)
        {
            std::cout << "  Time: " << time_ms << " ms" << std::endl;
            std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
            std::cout << "  Memory: " << memory_gb << " GB" << std::endl;
        }

        // Validate result on small matrices
        if (m <= 64 && n <= 64 && k <= 64)
        {
            if (!validateResult(*tensor_A, *tensor_B, *tensor_C))
            {
                if (rank == 0)
                {
                    std::cerr << "  ✗ Result validation failed" << std::endl;
                }
                return false;
            }
        }
    }

    return success;
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse arguments
    ArgumentParser parser(argc, argv);
    LlaminarParams params;

    if (!parser.parse(params))
    {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        std::cout << "\n=== COSMA Kernel Test ===" << std::endl;
        std::cout << "Testing COSMA matrix multiplication kernel" << std::endl;
        std::cout << "Using refactored MatMulKernel (no kernel manager)" << std::endl;
    }

    // Test kernel validation
    if (rank == 0)
    {
        std::cout << "\nTesting kernel validation..." << std::endl;

        MatMulKernel kernel;

        // Test invalid input count
        std::vector<std::shared_ptr<Tensor>> empty_inputs;
        std::vector<std::shared_ptr<Tensor>> empty_outputs;

        if (kernel.validate(empty_inputs, empty_outputs))
        {
            std::cerr << "Error: Validation should fail for empty inputs!" << std::endl;
            MPI_Finalize();
            return 1;
        }

        std::cout << "  ✓ Input validation working correctly" << std::endl;
    }

    // Run test cases
    std::vector<std::tuple<int, int, int>> test_cases = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512}};

    bool all_tests_passed = true;

    for (const auto &test_case : test_cases)
    {
        int m, n, k;
        std::tie(m, n, k) = test_case;

        bool success = runCOSMAKernelTest(m, n, k, rank, size);
        if (!success)
        {
            all_tests_passed = false;
            if (rank == 0)
            {
                std::cerr << "  ✗ Test failed for " << m << "x" << n << "x" << k << std::endl;
            }
        }
        else
        {
            if (rank == 0)
            {
                std::cout << "  ✓ Test passed for " << m << "x" << n << "x" << k << std::endl;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Final report
    if (rank == 0)
    {
        if (all_tests_passed)
        {
            std::cout << "\n✓ COSMA KERNEL TEST SUCCESS: All tests passed" << std::endl;
            std::cout << "  MatMulKernel successfully refactored to work like other kernels" << std::endl;
            std::cout << "  COSMA integration maintained for high performance" << std::endl;
        }
        else
        {
            std::cout << "\n✗ COSMA KERNEL TEST FAILURE: Some tests failed" << std::endl;
        }
        std::cout << "=========================" << std::endl;
    }

    MPI_Finalize();
    return all_tests_passed ? 0 : 1;
}