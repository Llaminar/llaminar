/**
 * @file Perf__Q16_1_DecodeSIMD.cpp
 * @brief Performance benchmarks for Q16_1 block decode SIMD implementations
 * @author David Sanftenberg
 *
 * Measures throughput (GB/s) for:
 *   - decode_q16_1_block_to_fp32_scalar()
 *   - decode_q16_1_block_to_fp32_avx2()
 *   - decode_q16_1_block_to_fp32_avx512()
 *
 * Reports input bandwidth (Q16_1 blocks processed) and output bandwidth (FP32 data produced).
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

class Q16_1_DecodeSIMD_Perf : public ::testing::Test
{
protected:
    static constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE; // 32 elements
    static constexpr size_t WARMUP_ITERATIONS = 100;
    static constexpr size_t BENCHMARK_ITERATIONS = 1000;

    // Test sizes in number of Q16_1 blocks
    static constexpr size_t NUM_BLOCKS_SMALL = 1024;       // 32K elements, ~128KB output
    static constexpr size_t NUM_BLOCKS_MEDIUM = 16 * 1024; // 512K elements, ~2MB output
    static constexpr size_t NUM_BLOCKS_LARGE = 64 * 1024;  // 2M elements, ~8MB output

    std::mt19937 rng_{42};

    void SetUp() override
    {
        rng_.seed(42);
    }

    /**
     * @brief Allocate Q16_1 blocks with random data
     */
    std::vector<Q16_1Block> allocate_random_input(size_t num_blocks)
    {
        std::vector<Q16_1Block> blocks(num_blocks);
        std::uniform_real_distribution<float> scale_dist(0.001f, 1.0f);
        std::uniform_int_distribution<int16_t> qs_dist(-32767, 32767);

        for (size_t i = 0; i < num_blocks; ++i)
        {
            blocks[i].d = scale_dist(rng_);
            blocks[i].sum_qs = 0;
            int32_t sum = 0;
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                blocks[i].qs[j] = qs_dist(rng_);
                sum += blocks[i].qs[j];
            }
            blocks[i].sum_qs = static_cast<int32_t>(sum);
        }
        return blocks;
    }

    /**
     * @brief Allocate aligned output buffer for FP32 data
     */
    std::vector<float> allocate_output(size_t num_blocks)
    {
        // 64-byte aligned allocation for AVX-512
        size_t num_floats = num_blocks * BLOCK_SIZE;
        std::vector<float> output(num_floats + 16); // Extra for alignment padding
        return output;
    }

    /**
     * @brief Get 64-byte aligned pointer from vector
     */
    float *get_aligned_ptr(std::vector<float> &data)
    {
        uintptr_t addr = reinterpret_cast<uintptr_t>(data.data());
        uintptr_t aligned = (addr + 63) & ~63ULL;
        return reinterpret_cast<float *>(aligned);
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
        double ns_per_block = (elapsed_ms * 1e6) / (iterations * (input_bytes / sizeof(Q16_1Block)));

        std::cout << "│ " << std::setw(12) << std::left << impl_name
                  << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << input_gbps
                  << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << output_gbps
                  << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << elapsed_ms
                  << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << ns_per_block
                  << " │" << std::endl;
    }

    /**
     * @brief Print table header
     */
    void print_header(const std::string &test_name, size_t num_blocks)
    {
        size_t input_kb = (num_blocks * sizeof(Q16_1Block)) / 1024;
        size_t output_kb = (num_blocks * BLOCK_SIZE * sizeof(float)) / 1024;

        std::cout << "\n╔════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ " << std::setw(72) << std::left << test_name << "║" << std::endl;
        std::cout << "║ Blocks: " << std::setw(8) << num_blocks
                  << " | Input: " << std::setw(6) << input_kb << " KB"
                  << " | Output: " << std::setw(6) << output_kb << " KB"
                  << "              ║" << std::endl;
        std::cout << "╠════════════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "│ Implementation │ Input GB/s │ Output GB/s │   Time (ms) │  ns/block  │" << std::endl;
        std::cout << "├────────────────┼────────────┼─────────────┼─────────────┼────────────┤" << std::endl;
    }

    /**
     * @brief Print table footer with speedup summary
     */
    void print_footer(double scalar_ms, double avx2_ms, double avx512_ms)
    {
        std::cout << "├────────────────┴────────────┴─────────────┴─────────────┴────────────┤" << std::endl;
        std::cout << "│ Speedups:";

        if (avx2_ms > 0 && scalar_ms > 0)
        {
            double avx2_speedup = scalar_ms / avx2_ms;
            std::cout << "  AVX2/Scalar: " << std::fixed << std::setprecision(2) << avx2_speedup << "x";
        }

        if (avx512_ms > 0 && avx2_ms > 0)
        {
            double avx512_avx2_speedup = avx2_ms / avx512_ms;
            std::cout << "  | AVX512/AVX2: " << std::fixed << std::setprecision(2) << avx512_avx2_speedup << "x";
        }

        if (avx512_ms > 0 && scalar_ms > 0)
        {
            double avx512_scalar_speedup = scalar_ms / avx512_ms;
            std::cout << "  | AVX512/Scalar: " << std::fixed << std::setprecision(2) << avx512_scalar_speedup << "x";
        }

        std::cout << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════════════════╝" << std::endl;
    }

    /**
     * @brief Benchmark scalar implementation
     */
    double benchmark_scalar(const Q16_1Block *input, float *output, size_t num_blocks)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                decode_q16_1_block_to_fp32_scalar(input[i], &output[i * BLOCK_SIZE]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                decode_q16_1_block_to_fp32_scalar(input[i], &output[i * BLOCK_SIZE]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

#if defined(__AVX2__)
    /**
     * @brief Benchmark AVX2 implementation
     */
    double benchmark_avx2(const Q16_1Block *input, float *output, size_t num_blocks)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                decode_q16_1_block_to_fp32_avx2(input[i], &output[i * BLOCK_SIZE]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                decode_q16_1_block_to_fp32_avx2(input[i], &output[i * BLOCK_SIZE]);
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
    double benchmark_avx512(const Q16_1Block *input, float *output, size_t num_blocks)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                decode_q16_1_block_to_fp32_avx512(input[i], &output[i * BLOCK_SIZE]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                decode_q16_1_block_to_fp32_avx512(input[i], &output[i * BLOCK_SIZE]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }
#endif

    /**
     * @brief Run full benchmark suite for a given number of blocks
     */
    void run_benchmark_suite(size_t num_blocks, const std::string &size_label)
    {
        auto input = allocate_random_input(num_blocks);
        auto output_storage = allocate_output(num_blocks);
        float *output = get_aligned_ptr(output_storage);

        size_t input_bytes = num_blocks * sizeof(Q16_1Block);
        size_t output_bytes = num_blocks * BLOCK_SIZE * sizeof(float);

        print_header("Q16_1 Block Decode Performance - " + size_label, num_blocks);

        // Scalar benchmark
        double scalar_ms = benchmark_scalar(input.data(), output, num_blocks);
        print_result("Scalar", input_bytes, output_bytes, scalar_ms, BENCHMARK_ITERATIONS);

        double avx2_ms = 0.0;
        double avx512_ms = 0.0;

#if defined(__AVX2__)
        avx2_ms = benchmark_avx2(input.data(), output, num_blocks);
        print_result("AVX2", input_bytes, output_bytes, avx2_ms, BENCHMARK_ITERATIONS);
#endif

#if defined(__AVX512F__)
        avx512_ms = benchmark_avx512(input.data(), output, num_blocks);
        print_result("AVX512", input_bytes, output_bytes, avx512_ms, BENCHMARK_ITERATIONS);
#endif

        print_footer(scalar_ms, avx2_ms, avx512_ms);
    }
};

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(Q16_1_DecodeSIMD_Perf, Small_1K_Blocks)
{
    run_benchmark_suite(NUM_BLOCKS_SMALL, "Small (1K blocks)");
}

TEST_F(Q16_1_DecodeSIMD_Perf, Medium_16K_Blocks)
{
    run_benchmark_suite(NUM_BLOCKS_MEDIUM, "Medium (16K blocks)");
}

TEST_F(Q16_1_DecodeSIMD_Perf, Large_64K_Blocks)
{
    run_benchmark_suite(NUM_BLOCKS_LARGE, "Large (64K blocks)");
}

// ============================================================================
// Speedup Assertion Tests
// ============================================================================

#if defined(__AVX2__)
TEST_F(Q16_1_DecodeSIMD_Perf, AVX2_FasterThanScalar)
{
    auto input = allocate_random_input(NUM_BLOCKS_MEDIUM);
    auto output_storage = allocate_output(NUM_BLOCKS_MEDIUM);
    float *output = get_aligned_ptr(output_storage);

    double scalar_ms = benchmark_scalar(input.data(), output, NUM_BLOCKS_MEDIUM);
    double avx2_ms = benchmark_avx2(input.data(), output, NUM_BLOCKS_MEDIUM);

    double speedup = scalar_ms / avx2_ms;

    std::cout << "\n[Speedup] AVX2 vs Scalar: " << std::fixed << std::setprecision(2)
              << speedup << "x (scalar=" << scalar_ms << "ms, avx2=" << avx2_ms << "ms)\n";

    // AVX2 processes 8 int16→float at a time vs 1 for scalar
    // Expect at least 2x speedup (conservative due to memory bandwidth limits)
    EXPECT_GT(speedup, 2.0) << "AVX2 should be at least 2x faster than scalar";
    EXPECT_LT(speedup, 10.0) << "AVX2 speedup suspiciously high (>10x)";
}
#endif

#if defined(__AVX512F__)
TEST_F(Q16_1_DecodeSIMD_Perf, AVX512_FasterThanAVX2)
{
    auto input = allocate_random_input(NUM_BLOCKS_MEDIUM);
    auto output_storage = allocate_output(NUM_BLOCKS_MEDIUM);
    float *output = get_aligned_ptr(output_storage);

    double avx2_ms = benchmark_avx2(input.data(), output, NUM_BLOCKS_MEDIUM);
    double avx512_ms = benchmark_avx512(input.data(), output, NUM_BLOCKS_MEDIUM);

    double speedup = avx2_ms / avx512_ms;

    std::cout << "\n[Speedup] AVX512 vs AVX2: " << std::fixed << std::setprecision(2)
              << speedup << "x (avx2=" << avx2_ms << "ms, avx512=" << avx512_ms << "ms)\n";

    // AVX512 processes 16 int16→float at a time vs 8 for AVX2
    // However, memory bandwidth often limits gains. Expect at least 1.1x.
    EXPECT_GT(speedup, 1.1) << "AVX512 should be at least 1.1x faster than AVX2";
    EXPECT_LT(speedup, 3.0) << "AVX512 speedup suspiciously high (>3x)";
}

TEST_F(Q16_1_DecodeSIMD_Perf, AVX512_FasterThanScalar)
{
    auto input = allocate_random_input(NUM_BLOCKS_MEDIUM);
    auto output_storage = allocate_output(NUM_BLOCKS_MEDIUM);
    float *output = get_aligned_ptr(output_storage);

    double scalar_ms = benchmark_scalar(input.data(), output, NUM_BLOCKS_MEDIUM);
    double avx512_ms = benchmark_avx512(input.data(), output, NUM_BLOCKS_MEDIUM);

    double speedup = scalar_ms / avx512_ms;

    std::cout << "\n[Speedup] AVX512 vs Scalar: " << std::fixed << std::setprecision(2)
              << speedup << "x (scalar=" << scalar_ms << "ms, avx512=" << avx512_ms << "ms)\n";

    // AVX512 processes 16 int16→float at a time vs 1 for scalar
    // Expect at least 3x speedup overall
    EXPECT_GT(speedup, 3.0) << "AVX512 should be at least 3x faster than scalar";
    EXPECT_LT(speedup, 20.0) << "AVX512 speedup suspiciously high (>20x)";
}
#endif

// ============================================================================
// Correctness Sanity Check
// ============================================================================

TEST_F(Q16_1_DecodeSIMD_Perf, Correctness_AllImplementationsMatch)
{
    constexpr size_t NUM_TEST_BLOCKS = 100;
    auto input = allocate_random_input(NUM_TEST_BLOCKS);

    std::vector<float> scalar_output(NUM_TEST_BLOCKS * BLOCK_SIZE);
    std::vector<float> avx2_output(NUM_TEST_BLOCKS * BLOCK_SIZE);
    std::vector<float> avx512_output(NUM_TEST_BLOCKS * BLOCK_SIZE);

    // Decode with scalar
    for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
    {
        decode_q16_1_block_to_fp32_scalar(input[i], &scalar_output[i * BLOCK_SIZE]);
    }

#if defined(__AVX2__)
    // Decode with AVX2
    for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
    {
        decode_q16_1_block_to_fp32_avx2(input[i], &avx2_output[i * BLOCK_SIZE]);
    }

    // Compare AVX2 vs Scalar
    for (size_t i = 0; i < NUM_TEST_BLOCKS * BLOCK_SIZE; ++i)
    {
        ASSERT_FLOAT_EQ(scalar_output[i], avx2_output[i])
            << "AVX2 mismatch at index " << i;
    }
    std::cout << "[Correctness] AVX2 matches Scalar: PASS\n";
#endif

#if defined(__AVX512F__)
    // Decode with AVX512
    for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
    {
        decode_q16_1_block_to_fp32_avx512(input[i], &avx512_output[i * BLOCK_SIZE]);
    }

    // Compare AVX512 vs Scalar
    for (size_t i = 0; i < NUM_TEST_BLOCKS * BLOCK_SIZE; ++i)
    {
        ASSERT_FLOAT_EQ(scalar_output[i], avx512_output[i])
            << "AVX512 mismatch at index " << i;
    }
    std::cout << "[Correctness] AVX512 matches Scalar: PASS\n";
#endif
}
