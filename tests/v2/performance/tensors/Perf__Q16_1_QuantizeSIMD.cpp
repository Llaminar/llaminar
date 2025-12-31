/**
 * @file Perf__Q16_1_QuantizeSIMD.cpp
 * @brief Performance benchmarks for Q16_1 quantization SIMD implementations
 * @author David Sanftenberg
 *
 * Measures throughput (GB/s) for:
 *   - quantize_fp32_to_q16_1_block_scalar()
 *   - quantize_fp32_to_q16_1_block_avx2()
 *   - quantize_fp32_to_q16_1_block_avx512()
 *
 * Reports input bandwidth (FP32 data processed) and output bandwidth (Q16_1 blocks produced).
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

class Q16_1_QuantizeSIMD_Perf : public ::testing::Test
{
protected:
    static constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE; // 32 elements
    static constexpr size_t WARMUP_ITERATIONS = 100;
    static constexpr size_t BENCHMARK_ITERATIONS = 1000;

    // Test sizes in number of blocks
    static constexpr size_t NUM_BLOCKS_SMALL = 1024;       // 32K elements, ~128KB input
    static constexpr size_t NUM_BLOCKS_MEDIUM = 16 * 1024; // 512K elements, ~2MB input
    static constexpr size_t NUM_BLOCKS_LARGE = 64 * 1024;  // 2M elements, ~8MB input

    std::mt19937 rng_{42};

    void SetUp() override
    {
        rng_.seed(42);
    }

    /**
     * @brief Allocate FP32 buffer with random data
     */
    std::vector<float> allocate_random_input(size_t num_blocks)
    {
        size_t num_floats = num_blocks * BLOCK_SIZE;
        // Extra padding for alignment
        std::vector<float> data(num_floats + 16);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

        for (size_t i = 0; i < num_floats; ++i)
        {
            data[i] = dist(rng_);
        }
        return data;
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
     * @brief Allocate output buffer for Q16_1 blocks
     */
    std::vector<Q16_1Block> allocate_output(size_t num_blocks)
    {
        return std::vector<Q16_1Block>(num_blocks);
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
        double ns_per_block = (elapsed_ms * 1e6) / (iterations * (input_bytes / (BLOCK_SIZE * sizeof(float))));

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
        size_t input_kb = (num_blocks * BLOCK_SIZE * sizeof(float)) / 1024;
        size_t output_kb = (num_blocks * sizeof(Q16_1Block)) / 1024;

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
    double benchmark_scalar(const float *input, Q16_1Block *output, size_t num_blocks)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_scalar(&input[i * BLOCK_SIZE], output[i]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_scalar(&input[i * BLOCK_SIZE], output[i]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

#if defined(__AVX2__)
    /**
     * @brief Benchmark AVX2 implementation
     */
    double benchmark_avx2(const float *input, Q16_1Block *output, size_t num_blocks)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_avx2(&input[i * BLOCK_SIZE], output[i]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_avx2(&input[i * BLOCK_SIZE], output[i]);
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
    double benchmark_avx512(const float *input, Q16_1Block *output, size_t num_blocks)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_avx512(&input[i * BLOCK_SIZE], output[i]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_avx512(&input[i * BLOCK_SIZE], output[i]);
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
        auto input_storage = allocate_random_input(num_blocks);
        const float *input = get_aligned_ptr(input_storage);
        auto output = allocate_output(num_blocks);

        size_t input_bytes = num_blocks * BLOCK_SIZE * sizeof(float);
        size_t output_bytes = num_blocks * sizeof(Q16_1Block);

        print_header("Q16_1 Block Quantize Performance - " + size_label, num_blocks);

        // Scalar benchmark
        double scalar_ms = benchmark_scalar(input, output.data(), num_blocks);
        print_result("Scalar", input_bytes, output_bytes, scalar_ms, BENCHMARK_ITERATIONS);

        double avx2_ms = 0.0;
        double avx512_ms = 0.0;

#if defined(__AVX2__)
        avx2_ms = benchmark_avx2(input, output.data(), num_blocks);
        print_result("AVX2", input_bytes, output_bytes, avx2_ms, BENCHMARK_ITERATIONS);
#endif

#if defined(__AVX512F__)
        avx512_ms = benchmark_avx512(input, output.data(), num_blocks);
        print_result("AVX512", input_bytes, output_bytes, avx512_ms, BENCHMARK_ITERATIONS);
#endif

        print_footer(scalar_ms, avx2_ms, avx512_ms);
    }
};

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(Q16_1_QuantizeSIMD_Perf, Small_1K_Blocks)
{
    run_benchmark_suite(NUM_BLOCKS_SMALL, "Small (1K blocks)");
}

TEST_F(Q16_1_QuantizeSIMD_Perf, Medium_16K_Blocks)
{
    run_benchmark_suite(NUM_BLOCKS_MEDIUM, "Medium (16K blocks)");
}

TEST_F(Q16_1_QuantizeSIMD_Perf, Large_64K_Blocks)
{
    run_benchmark_suite(NUM_BLOCKS_LARGE, "Large (64K blocks)");
}

// ============================================================================
// Speedup Assertion Tests
// ============================================================================

#if defined(__AVX2__)
TEST_F(Q16_1_QuantizeSIMD_Perf, AVX2_FasterThanScalar)
{
    auto input_storage = allocate_random_input(NUM_BLOCKS_MEDIUM);
    const float *input = get_aligned_ptr(input_storage);
    auto output = allocate_output(NUM_BLOCKS_MEDIUM);

    double scalar_ms = benchmark_scalar(input, output.data(), NUM_BLOCKS_MEDIUM);
    double avx2_ms = benchmark_avx2(input, output.data(), NUM_BLOCKS_MEDIUM);

    double speedup = scalar_ms / avx2_ms;

    std::cout << "\n[Speedup] AVX2 vs Scalar: " << std::fixed << std::setprecision(2)
              << speedup << "x (scalar=" << scalar_ms << "ms, avx2=" << avx2_ms << "ms)\n";

    // Quantization is more compute-intensive than decode (find max, scale, clamp, sum)
    // Expect at least 2x speedup
    EXPECT_GT(speedup, 2.0) << "AVX2 should be at least 2x faster than scalar";
    EXPECT_LT(speedup, 12.0) << "AVX2 speedup suspiciously high (>12x)";
}
#endif

#if defined(__AVX512F__)
TEST_F(Q16_1_QuantizeSIMD_Perf, AVX512_FasterThanAVX2)
{
    auto input_storage = allocate_random_input(NUM_BLOCKS_MEDIUM);
    const float *input = get_aligned_ptr(input_storage);
    auto output = allocate_output(NUM_BLOCKS_MEDIUM);

    double avx2_ms = benchmark_avx2(input, output.data(), NUM_BLOCKS_MEDIUM);
    double avx512_ms = benchmark_avx512(input, output.data(), NUM_BLOCKS_MEDIUM);

    double speedup = avx2_ms / avx512_ms;

    std::cout << "\n[Speedup] AVX512 vs AVX2: " << std::fixed << std::setprecision(2)
              << speedup << "x (avx2=" << avx2_ms << "ms, avx512=" << avx512_ms << "ms)\n";

    // AVX512 processes 32 floats in 2 ZMM vs 4 YMM for AVX2
    // With compiler optimizations, the gap narrows - expect at least parity
    EXPECT_GE(speedup, 0.95) << "AVX512 should be at least as fast as AVX2";
    EXPECT_LT(speedup, 3.0) << "AVX512 speedup suspiciously high (>3x)";
}

TEST_F(Q16_1_QuantizeSIMD_Perf, AVX512_FasterThanScalar)
{
    auto input_storage = allocate_random_input(NUM_BLOCKS_MEDIUM);
    const float *input = get_aligned_ptr(input_storage);
    auto output = allocate_output(NUM_BLOCKS_MEDIUM);

    double scalar_ms = benchmark_scalar(input, output.data(), NUM_BLOCKS_MEDIUM);
    double avx512_ms = benchmark_avx512(input, output.data(), NUM_BLOCKS_MEDIUM);

    double speedup = scalar_ms / avx512_ms;

    std::cout << "\n[Speedup] AVX512 vs Scalar: " << std::fixed << std::setprecision(2)
              << speedup << "x (scalar=" << scalar_ms << "ms, avx512=" << avx512_ms << "ms)\n";

    // Expect at least 3x speedup overall
    EXPECT_GT(speedup, 3.0) << "AVX512 should be at least 3x faster than scalar";
    EXPECT_LT(speedup, 30.0) << "AVX512 speedup suspiciously high (>30x)";
}
#endif

// ============================================================================
// Correctness Sanity Check
// ============================================================================

TEST_F(Q16_1_QuantizeSIMD_Perf, Correctness_AllImplementationsMatch)
{
    constexpr size_t NUM_TEST_BLOCKS = 100;
    auto input_storage = allocate_random_input(NUM_TEST_BLOCKS);
    const float *input = input_storage.data();

    std::vector<Q16_1Block> scalar_output(NUM_TEST_BLOCKS);
    std::vector<Q16_1Block> avx2_output(NUM_TEST_BLOCKS);
    std::vector<Q16_1Block> avx512_output(NUM_TEST_BLOCKS);

    // Quantize with scalar
    for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
    {
        quantize_fp32_to_q16_1_block_scalar(&input[i * BLOCK_SIZE], scalar_output[i]);
    }

#if defined(__AVX2__)
    // Quantize with AVX2
    for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
    {
        quantize_fp32_to_q16_1_block_avx2(&input[i * BLOCK_SIZE], avx2_output[i]);
    }

    // Compare AVX2 vs Scalar
    for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
    {
        ASSERT_FLOAT_EQ(scalar_output[i].d, avx2_output[i].d)
            << "AVX2 scale mismatch at block " << i;
        ASSERT_EQ(scalar_output[i].sum_qs, avx2_output[i].sum_qs)
            << "AVX2 sum_qs mismatch at block " << i;
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            ASSERT_EQ(scalar_output[i].qs[j], avx2_output[i].qs[j])
                << "AVX2 qs mismatch at block " << i << ", element " << j;
        }
    }
    std::cout << "[Correctness] AVX2 matches Scalar: PASS\n";
#endif

#if defined(__AVX512F__)
    // Quantize with AVX512
    for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
    {
        quantize_fp32_to_q16_1_block_avx512(&input[i * BLOCK_SIZE], avx512_output[i]);
    }

    // Compare AVX512 vs Scalar
    for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
    {
        ASSERT_FLOAT_EQ(scalar_output[i].d, avx512_output[i].d)
            << "AVX512 scale mismatch at block " << i;
        ASSERT_EQ(scalar_output[i].sum_qs, avx512_output[i].sum_qs)
            << "AVX512 sum_qs mismatch at block " << i;
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            ASSERT_EQ(scalar_output[i].qs[j], avx512_output[i].qs[j])
                << "AVX512 qs mismatch at block " << i << ", element " << j;
        }
    }
    std::cout << "[Correctness] AVX512 matches Scalar: PASS\n";
#endif
}

// ============================================================================
// Round-Trip Correctness Test (Quantize -> Dequantize)
// ============================================================================

TEST_F(Q16_1_QuantizeSIMD_Perf, RoundTrip_QuantizeDequantize)
{
    constexpr size_t NUM_TEST_BLOCKS = 50;
    auto input_storage = allocate_random_input(NUM_TEST_BLOCKS);
    const float *input = input_storage.data();

    std::vector<Q16_1Block> blocks(NUM_TEST_BLOCKS);
    std::vector<float> decoded(NUM_TEST_BLOCKS * BLOCK_SIZE);

    // Quantize using auto-dispatch
    for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
    {
        quantize_fp32_to_q16_1_block(&input[i * BLOCK_SIZE], blocks[i]);
    }

    // Decode using auto-dispatch
    for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
    {
        decode_q16_1_block_to_fp32(blocks[i], &decoded[i * BLOCK_SIZE]);
    }

    // Check reconstruction error
    double max_error = 0.0;
    double total_error = 0.0;
    for (size_t i = 0; i < NUM_TEST_BLOCKS * BLOCK_SIZE; ++i)
    {
        double error = std::abs(static_cast<double>(input[i]) - static_cast<double>(decoded[i]));
        max_error = std::max(max_error, error);
        total_error += error;
    }
    double avg_error = total_error / (NUM_TEST_BLOCKS * BLOCK_SIZE);

    std::cout << "[Round-Trip] Max error: " << max_error << ", Avg error: " << avg_error << "\n";

    // Q16_1 has 16-bit precision, expect very low error
    // Max error should be roughly scale/2 ≈ max_abs/(2*32767)
    // For values in [-10, 10], max_error ~ 10/(2*32767) ~ 0.00015
    EXPECT_LT(max_error, 0.001) << "Round-trip max error too large";
    EXPECT_LT(avg_error, 0.0005) << "Round-trip avg error too large";
}
