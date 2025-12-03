/**
 * @file Perf__Q8_1_QuantizeSIMD.cpp
 * @brief Performance benchmarks for Q8_1 quantization SIMD implementations
 * @author David Sanftenberg
 *
 * Measures throughput (GB/s) for:
 *   - quantize_single_block_scalar()
 *   - quantize_single_block_avx2()
 *   - quantize_single_block_avx512()
 *
 * Reports input bandwidth (FP32 data processed) and output bandwidth (Q8_1 blocks produced).
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "tensors/SIMDHelpers.h"

using namespace llaminar2;
using namespace llaminar2::simd;

// ============================================================================
// Performance Test Fixture
// ============================================================================

class Q8_1_QuantizeSIMD_Perf : public ::testing::Test
{
protected:
    static constexpr int BLOCK_SIZE = 32;
    static constexpr size_t WARMUP_ITERATIONS = 10;
    static constexpr size_t BENCHMARK_ITERATIONS = 100;

    // Test sizes: 1MB, 4MB, 16MB, 64MB of FP32 input data
    static constexpr size_t SIZE_1MB = 1024 * 1024 / sizeof(float);       // 256K floats
    static constexpr size_t SIZE_4MB = 4 * 1024 * 1024 / sizeof(float);   // 1M floats
    static constexpr size_t SIZE_16MB = 16 * 1024 * 1024 / sizeof(float); // 4M floats
    static constexpr size_t SIZE_64MB = 64 * 1024 * 1024 / sizeof(float); // 16M floats

    std::mt19937 rng_{42};

    void SetUp() override
    {
        // Ensure consistent RNG state
        rng_.seed(42);
    }

    /**
     * @brief Allocate aligned buffer filled with random FP32 data
     */
    std::vector<float> allocate_random_input(size_t num_floats)
    {
        std::vector<float> data(num_floats + 64 / sizeof(float)); // Extra for alignment
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = dist(rng_);
        }
        return data;
    }

    /**
     * @brief Get aligned pointer from vector
     */
    float *get_aligned_ptr(std::vector<float> &data)
    {
        uintptr_t addr = reinterpret_cast<uintptr_t>(data.data());
        uintptr_t aligned = (addr + 63) & ~63ULL;
        return reinterpret_cast<float *>(aligned);
    }

    /**
     * @brief Allocate output buffer for Q8_1 blocks
     */
    std::vector<Q8_1Block> allocate_output(size_t num_blocks)
    {
        return std::vector<Q8_1Block>(num_blocks);
    }

    /**
     * @brief Print benchmark result in formatted table row
     */
    void print_result(const std::string &impl_name,
                      size_t input_bytes,
                      size_t output_bytes,
                      double elapsed_ms,
                      size_t iterations)
    {
        double total_input_bytes = static_cast<double>(input_bytes) * iterations;
        double total_output_bytes = static_cast<double>(output_bytes) * iterations;
        double elapsed_sec = elapsed_ms / 1000.0;

        double input_gbps = (total_input_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec;
        double output_gbps = (total_output_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec;
        double throughput_gflops = (total_input_bytes / sizeof(float)) / (elapsed_sec * 1e9);

        std::cout << "│ " << std::setw(12) << std::left << impl_name
                  << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << input_gbps
                  << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << output_gbps
                  << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << elapsed_ms
                  << " │ " << std::setw(12) << std::right << std::fixed << std::setprecision(3) << throughput_gflops
                  << " │" << std::endl;
    }

    /**
     * @brief Print table header
     */
    void print_header(const std::string &test_name, size_t input_mb)
    {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ " << std::setw(74) << std::left << test_name << "║" << std::endl;
        std::cout << "║ Input Size: " << std::setw(4) << input_mb << " MB"
                  << "                                                        ║" << std::endl;
        std::cout << "╠══════════════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "│ Implementation │ Input GB/s │ Output GB/s │   Time (ms) │ GElements/s  │" << std::endl;
        std::cout << "├────────────────┼────────────┼─────────────┼─────────────┼──────────────┤" << std::endl;
    }

    /**
     * @brief Print table footer
     */
    void print_footer()
    {
        std::cout << "╚══════════════════════════════════════════════════════════════════════════╝" << std::endl;
    }

    /**
     * @brief Benchmark scalar implementation
     */
    double benchmark_scalar(float *input, Q8_1Block *output, size_t num_blocks)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_single_block_scalar(&input[i * BLOCK_SIZE], output[i], BLOCK_SIZE);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_single_block_scalar(&input[i * BLOCK_SIZE], output[i], BLOCK_SIZE);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

#if defined(__AVX2__)
    /**
     * @brief Benchmark AVX2 implementation
     */
    double benchmark_avx2(float *input, Q8_1Block *output, size_t num_blocks)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_single_block_avx2(&input[i * BLOCK_SIZE], output[i]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_single_block_avx2(&input[i * BLOCK_SIZE], output[i]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }
#endif

#if defined(__AVX512F__)
    /**
     * @brief Benchmark AVX512 implementation
     */
    double benchmark_avx512(float *input, Q8_1Block *output, size_t num_blocks)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_single_block_avx512(&input[i * BLOCK_SIZE], output[i]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_single_block_avx512(&input[i * BLOCK_SIZE], output[i]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }
#endif

    /**
     * @brief Run benchmark suite for a given input size
     */
    void run_benchmark_suite(size_t num_floats, const std::string &test_name)
    {
        // Ensure num_floats is a multiple of BLOCK_SIZE
        num_floats = (num_floats / BLOCK_SIZE) * BLOCK_SIZE;
        size_t num_blocks = num_floats / BLOCK_SIZE;

        size_t input_bytes = num_floats * sizeof(float);
        size_t output_bytes = num_blocks * sizeof(Q8_1Block);
        size_t input_mb = input_bytes / (1024 * 1024);

        // Allocate buffers
        auto input_vec = allocate_random_input(num_floats);
        float *input = get_aligned_ptr(input_vec);
        auto output = allocate_output(num_blocks);

        print_header(test_name, input_mb);

        // Scalar benchmark
        {
            double elapsed = benchmark_scalar(input, output.data(), num_blocks);
            print_result("Scalar", input_bytes, output_bytes, elapsed, BENCHMARK_ITERATIONS);
        }

#if defined(__AVX2__)
        // AVX2 benchmark
        {
            double elapsed = benchmark_avx2(input, output.data(), num_blocks);
            print_result("AVX2", input_bytes, output_bytes, elapsed, BENCHMARK_ITERATIONS);
        }
#endif

#if defined(__AVX512F__)
        // AVX512 benchmark
        {
            double elapsed = benchmark_avx512(input, output.data(), num_blocks);
            print_result("AVX512", input_bytes, output_bytes, elapsed, BENCHMARK_ITERATIONS);
        }
#endif

        print_footer();
    }
};

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Perf, Throughput_1MB)
{
    run_benchmark_suite(SIZE_1MB, "Q8_1 Quantization Throughput - 1MB");
}

TEST_F(Q8_1_QuantizeSIMD_Perf, Throughput_4MB)
{
    run_benchmark_suite(SIZE_4MB, "Q8_1 Quantization Throughput - 4MB");
}

TEST_F(Q8_1_QuantizeSIMD_Perf, Throughput_16MB)
{
    run_benchmark_suite(SIZE_16MB, "Q8_1 Quantization Throughput - 16MB");
}

TEST_F(Q8_1_QuantizeSIMD_Perf, Throughput_64MB)
{
    run_benchmark_suite(SIZE_64MB, "Q8_1 Quantization Throughput - 64MB");
}

// ============================================================================
// Scaling Test: Measure how performance scales with data size
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Perf, ScalingAnalysis)
{
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    Q8_1 Quantization Scaling Analysis                    ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "│ Size (MB) │ Scalar GB/s │  AVX2 GB/s  │ AVX512 GB/s │ AVX2 Speedup │ AVX512 Speedup │" << std::endl;
    std::cout << "├───────────┼─────────────┼─────────────┼─────────────┼──────────────┼────────────────┤" << std::endl;

    std::vector<size_t> sizes = {
        256 * 1024,       // 1 MB
        1024 * 1024,      // 4 MB
        4 * 1024 * 1024,  // 16 MB
        16 * 1024 * 1024, // 64 MB
    };

    for (size_t num_floats : sizes)
    {
        num_floats = (num_floats / BLOCK_SIZE) * BLOCK_SIZE;
        size_t num_blocks = num_floats / BLOCK_SIZE;
        size_t input_bytes = num_floats * sizeof(float);
        size_t input_mb = input_bytes / (1024 * 1024);

        auto input_vec = allocate_random_input(num_floats);
        float *input = get_aligned_ptr(input_vec);
        auto output = allocate_output(num_blocks);

        double scalar_ms = benchmark_scalar(input, output.data(), num_blocks);
        double scalar_gbps = (static_cast<double>(input_bytes) * BENCHMARK_ITERATIONS / (1024.0 * 1024.0 * 1024.0)) / (scalar_ms / 1000.0);

        double avx2_gbps = 0.0, avx512_gbps = 0.0;
        double avx2_speedup = 0.0, avx512_speedup = 0.0;

#if defined(__AVX2__)
        double avx2_ms = benchmark_avx2(input, output.data(), num_blocks);
        avx2_gbps = (static_cast<double>(input_bytes) * BENCHMARK_ITERATIONS / (1024.0 * 1024.0 * 1024.0)) / (avx2_ms / 1000.0);
        avx2_speedup = scalar_ms / avx2_ms;
#endif

#if defined(__AVX512F__)
        double avx512_ms = benchmark_avx512(input, output.data(), num_blocks);
        avx512_gbps = (static_cast<double>(input_bytes) * BENCHMARK_ITERATIONS / (1024.0 * 1024.0 * 1024.0)) / (avx512_ms / 1000.0);
        avx512_speedup = scalar_ms / avx512_ms;
#endif

        std::cout << "│ " << std::setw(9) << std::right << input_mb
                  << " │ " << std::setw(11) << std::right << std::fixed << std::setprecision(2) << scalar_gbps
                  << " │ " << std::setw(11) << std::right << std::fixed << std::setprecision(2) << avx2_gbps
                  << " │ " << std::setw(11) << std::right << std::fixed << std::setprecision(2) << avx512_gbps
                  << " │ " << std::setw(12) << std::right << std::fixed << std::setprecision(2) << avx2_speedup << "x"
                  << " │ " << std::setw(14) << std::right << std::fixed << std::setprecision(2) << avx512_speedup << "x"
                  << " │" << std::endl;
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
}

// ============================================================================
// Latency Test: Measure per-block latency in nanoseconds
// ============================================================================

TEST_F(Q8_1_QuantizeSIMD_Perf, PerBlockLatency)
{
    constexpr size_t NUM_BLOCKS = 10000;
    constexpr size_t LATENCY_ITERATIONS = 1000;

    auto input_vec = allocate_random_input(NUM_BLOCKS * BLOCK_SIZE);
    float *input = get_aligned_ptr(input_vec);
    auto output = allocate_output(NUM_BLOCKS);

    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    Q8_1 Quantization Per-Block Latency                   ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "│ Implementation │  Latency (ns/block)  │  Blocks/µs  │  Elements/cycle*  │" << std::endl;
    std::cout << "├────────────────┼──────────────────────┼─────────────┼───────────────────┤" << std::endl;

    // Scalar
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < LATENCY_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < NUM_BLOCKS; ++i)
            {
                quantize_single_block_scalar(&input[i * BLOCK_SIZE], output[i], BLOCK_SIZE);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        double total_ns = std::chrono::duration<double, std::nano>(end - start).count();
        double ns_per_block = total_ns / (NUM_BLOCKS * LATENCY_ITERATIONS);
        double blocks_per_us = 1000.0 / ns_per_block;
        // Assuming ~3 GHz CPU, estimate elements per cycle
        double elements_per_cycle = (BLOCK_SIZE * 3.0) / ns_per_block;

        std::cout << "│ " << std::setw(14) << std::left << "Scalar"
                  << " │ " << std::setw(20) << std::right << std::fixed << std::setprecision(2) << ns_per_block
                  << " │ " << std::setw(11) << std::right << std::fixed << std::setprecision(2) << blocks_per_us
                  << " │ " << std::setw(17) << std::right << std::fixed << std::setprecision(2) << elements_per_cycle
                  << " │" << std::endl;
    }

#if defined(__AVX2__)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < LATENCY_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < NUM_BLOCKS; ++i)
            {
                quantize_single_block_avx2(&input[i * BLOCK_SIZE], output[i]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        double total_ns = std::chrono::duration<double, std::nano>(end - start).count();
        double ns_per_block = total_ns / (NUM_BLOCKS * LATENCY_ITERATIONS);
        double blocks_per_us = 1000.0 / ns_per_block;
        double elements_per_cycle = (BLOCK_SIZE * 3.0) / ns_per_block;

        std::cout << "│ " << std::setw(14) << std::left << "AVX2"
                  << " │ " << std::setw(20) << std::right << std::fixed << std::setprecision(2) << ns_per_block
                  << " │ " << std::setw(11) << std::right << std::fixed << std::setprecision(2) << blocks_per_us
                  << " │ " << std::setw(17) << std::right << std::fixed << std::setprecision(2) << elements_per_cycle
                  << " │" << std::endl;
    }
#endif

#if defined(__AVX512F__)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < LATENCY_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < NUM_BLOCKS; ++i)
            {
                quantize_single_block_avx512(&input[i * BLOCK_SIZE], output[i]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        double total_ns = std::chrono::duration<double, std::nano>(end - start).count();
        double ns_per_block = total_ns / (NUM_BLOCKS * LATENCY_ITERATIONS);
        double blocks_per_us = 1000.0 / ns_per_block;
        double elements_per_cycle = (BLOCK_SIZE * 3.0) / ns_per_block;

        std::cout << "│ " << std::setw(14) << std::left << "AVX512"
                  << " │ " << std::setw(20) << std::right << std::fixed << std::setprecision(2) << ns_per_block
                  << " │ " << std::setw(11) << std::right << std::fixed << std::setprecision(2) << blocks_per_us
                  << " │ " << std::setw(17) << std::right << std::fixed << std::setprecision(2) << elements_per_cycle
                  << " │" << std::endl;
    }
#endif

    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "* Elements/cycle estimated assuming 3 GHz CPU frequency" << std::endl;
}
