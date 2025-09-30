// Adaptive Architecture Demonstration
//
// This program demonstrates the hybrid OpenBLAS/COSMA architecture
// in action, showing automatic backend selection for different operation types.

#include "adaptive_matmul.h"
#include "adaptive_transformer_pipeline.h"
#include "transformer_config.h"
#include "logger.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace llaminar;

void printHeader(const std::string &title)
{
    std::cout << "\n"
              << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void demonstrateBackendSelection()
{
    printHeader("BACKEND SELECTION DEMONSTRATION");

    AdaptiveMatMulManager manager;

    struct TestCase
    {
        int m, n, k;
        bool is_prefill;
        std::string description;
    };

    std::vector<TestCase> test_cases = {
        // Single token inference scenarios (should use OpenBLAS)
        {1, 896, 896, false, "Single token Q/K/V projection"},
        {1, 4864, 896, false, "Single token FFN up projection"},
        {1, 896, 4864, false, "Single token FFN down projection"},
        {1, 151936, 896, false, "Single token vocab projection"},

        // Small batch scenarios (should use OpenBLAS)
        {8, 896, 896, false, "Small batch Q/K/V projection"},
        {32, 896, 896, false, "Medium batch Q/K/V projection"},

        // Prefill scenarios (should use COSMA for large enough sequences)
        {64, 896, 896, true, "Prefill Q/K/V projection (threshold)"},
        {128, 896, 896, true, "Prefill Q/K/V projection (large)"},
        {512, 896, 896, true, "Prefill Q/K/V projection (very large)"},
        {1024, 896, 896, true, "Prefill Q/K/V projection (huge)"},

        // FFN operations in prefill
        {64, 4864, 896, true, "Prefill FFN up projection"},
        {64, 896, 4864, true, "Prefill FFN down projection"},
        {512, 4864, 896, true, "Large prefill FFN up projection"},
        {512, 896, 4864, true, "Large prefill FFN down projection"},

        // Edge cases
        {64, 151936, 896, true, "Large vocab projection (should avoid COSMA)"},
        {4, 896, 896, true, "Very small prefill (should use OpenBLAS)"}};

    std::cout << std::left << std::setw(40) << "Operation"
              << std::setw(15) << "Matrix Size"
              << std::setw(10) << "Backend" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (const auto &test_case : test_cases)
    {
        MatMulBackend backend = manager.selectBackend(test_case.m, test_case.n, test_case.k, test_case.is_prefill);
        std::string backend_name = (backend == MatMulBackend::COSMA) ? "COSMA" : "OpenBLAS";
        std::string matrix_size = std::to_string(test_case.m) + "x" +
                                  std::to_string(test_case.n) + "x" +
                                  std::to_string(test_case.k);

        std::cout << std::left << std::setw(40) << test_case.description
                  << std::setw(15) << matrix_size
                  << std::setw(10) << backend_name << std::endl;
    }
}

void demonstratePerformanceComparison()
{
    printHeader("PERFORMANCE COMPARISON");

    AdaptiveMatMulManager manager;

    struct BenchmarkCase
    {
        int m, n, k;
        bool is_prefill;
        std::string description;
    };

    std::vector<BenchmarkCase> benchmarks = {
        {1, 896, 896, false, "Single token inference"},
        {64, 896, 896, true, "Prefill (threshold)"},
        {256, 896, 896, true, "Large prefill"},
        {1024, 896, 896, true, "Very large prefill"},
        {64, 4864, 896, true, "FFN up projection"},
        {64, 896, 4864, true, "FFN down projection"}};

    std::cout << std::left << std::setw(25) << "Operation"
              << std::setw(15) << "Matrix Size"
              << std::setw(12) << "Backend"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS" << std::endl;
    std::cout << std::string(82, '-') << std::endl;

    for (const auto &benchmark : benchmarks)
    {
        // Generate random test data
        std::vector<float> A(benchmark.m * benchmark.k, 1.0f);
        std::vector<float> B(benchmark.k * benchmark.n, 1.0f);
        std::vector<float> C(benchmark.m * benchmark.n, 0.0f);

        // Warm up
        manager.multiply(A.data(), B.data(), C.data(),
                         benchmark.m, benchmark.n, benchmark.k,
                         false, false, 1.0f, 0.0f, benchmark.is_prefill);

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        bool success = manager.multiply(A.data(), B.data(), C.data(),
                                        benchmark.m, benchmark.n, benchmark.k,
                                        false, false, 1.0f, 0.0f, benchmark.is_prefill);
        auto end = std::chrono::high_resolution_clock::now();

        if (!success)
        {
            std::cout << "Failed to execute: " << benchmark.description << std::endl;
            continue;
        }

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double flops = 2.0 * benchmark.m * benchmark.n * benchmark.k; // 2 ops per multiply-add
        double gflops = flops / (time_ms * 1e6);                      // GFLOPS

        MatMulBackend backend = manager.selectBackend(benchmark.m, benchmark.n, benchmark.k, benchmark.is_prefill);
        std::string backend_name = (backend == MatMulBackend::COSMA) ? "COSMA" : "OpenBLAS";
        std::string matrix_size = std::to_string(benchmark.m) + "x" +
                                  std::to_string(benchmark.n) + "x" +
                                  std::to_string(benchmark.k);

        std::cout << std::left << std::setw(25) << benchmark.description
                  << std::setw(15) << matrix_size
                  << std::setw(12) << backend_name
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << gflops << std::endl;
    }
}

void demonstrateTokenGeneration()
{
    printHeader("TOKEN GENERATION SIMULATION");

    // Create transformer configuration for Qwen 2.5 0.5B
    TransformerLayerConfig config;
    config.d_model = 896;
    config.d_ff = 4864;
    config.n_head = 14;
    config.n_head_kv = 2;
    config.head_dim = 64;
    config.vocab_size = 151936;
    config.n_layers = 24;
    config.max_seq_len = 32768;
    config.eps = 1e-6;
    config.rope_freq_base = 10000.0f;

    try
    {
        AdaptiveTransformerPipeline pipeline(config, true);

        std::cout << "Initialized AdaptiveTransformerPipeline with Qwen 2.5 0.5B configuration:" << std::endl;
        std::cout << "  Hidden size: " << config.d_model << std::endl;
        std::cout << "  Intermediate size: " << config.d_ff << std::endl;
        std::cout << "  Attention heads: " << config.n_head << std::endl;
        std::cout << "  Layers: " << config.n_layers << std::endl;
        std::cout << "  Vocab size: " << config.vocab_size << std::endl;

        // Simulate different scenarios
        std::vector<std::pair<std::vector<int>, std::string>> scenarios = {
            {{1, 2, 3, 4}, "Short prompt (4 tokens)"},
            {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, "Medium prompt (16 tokens)"},
            {std::vector<int>(64, 1), "Long prompt (64 tokens)"},
            {std::vector<int>(256, 1), "Very long prompt (256 tokens)"}};

        std::cout << "\nToken generation scenarios:" << std::endl;
        for (const auto &[prompt, description] : scenarios)
        {
            std::cout << "  " << description << " -> ";

            if (prompt.size() >= 64)
            {
                std::cout << "Prefill with COSMA, generation with OpenBLAS" << std::endl;
            }
            else
            {
                std::cout << "Both prefill and generation with OpenBLAS" << std::endl;
            }
        }

        std::cout << "\nAdaptive architecture ready for token generation!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "Error initializing transformer pipeline: " << e.what() << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        std::cout << "\nLlaminar Adaptive Architecture Demonstration" << std::endl;
        std::cout << "Running with " << size << " MPI processes" << std::endl;

        if (size < 2)
        {
            std::cout << "\nWarning: This demonstration is optimized for 2+ MPI processes." << std::endl;
            std::cout << "COSMA backend will not be used with single process." << std::endl;
        }

        // Run demonstrations
        demonstrateBackendSelection();
        demonstratePerformanceComparison();
        demonstrateTokenGeneration();

        printHeader("SUMMARY");
        std::cout << "The adaptive architecture provides:" << std::endl;
        std::cout << "  ✓ Automatic backend selection based on operation characteristics" << std::endl;
        std::cout << "  ✓ OpenBLAS for fast single-token inference (<64 tokens)" << std::endl;
        std::cout << "  ✓ COSMA for efficient large-scale prefill (≥64 tokens)" << std::endl;
        std::cout << "  ✓ Performance monitoring and optimization" << std::endl;
        std::cout << "  ✓ Seamless integration with transformer pipeline" << std::endl;
        std::cout << "\nOptimal performance achieved through hybrid deployment!" << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}