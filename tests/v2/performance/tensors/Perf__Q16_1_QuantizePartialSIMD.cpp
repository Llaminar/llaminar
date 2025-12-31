/**
 * @file Perf__Q16_1_QuantizePartialSIMD.cpp
 * @brief Performance benchmarks for Q16_1 partial block quantization SIMD implementations
 *
 * Measures throughput for partial block quantization (1-31 elements):
 *   - quantize_fp32_to_q16_1_block_partial_scalar()
 *   - quantize_fp32_to_q16_1_block_partial_avx2()
 *   - quantize_fp32_to_q16_1_block_partial_avx512()
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

class Q16_1_QuantizePartialSIMD_Perf : public ::testing::Test
{
protected:
    static constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE; // 32 elements
    static constexpr size_t WARMUP_ITERATIONS = 1000;
    static constexpr size_t BENCHMARK_ITERATIONS = 10000;

    // Test with various partial sizes
    static constexpr size_t NUM_BLOCKS = 1024;

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
     * @brief Print benchmark result
     */
    void print_result(const std::string &impl_name,
                      size_t partial_size,
                      double elapsed_ms,
                      size_t iterations,
                      size_t num_blocks)
    {
        double ns_per_block = (elapsed_ms * 1e6) / (iterations * num_blocks);
        double blocks_per_sec = (iterations * num_blocks * 1000.0) / elapsed_ms;

        std::cout << "│ " << std::setw(12) << std::left << impl_name
                  << " │ " << std::setw(6) << std::right << partial_size
                  << " │ " << std::setw(12) << std::right << std::fixed << std::setprecision(2) << ns_per_block
                  << " │ " << std::setw(12) << std::right << std::fixed << std::setprecision(0) << blocks_per_sec / 1e6
                  << " │" << std::endl;
    }

    /**
     * @brief Print table header
     */
    void print_header(const std::string &test_name)
    {
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ " << std::setw(61) << std::left << test_name << "║" << std::endl;
        std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "│ Implementation │ Count  │   ns/block   │ M blocks/sec │" << std::endl;
        std::cout << "├────────────────┼────────┼──────────────┼──────────────┤" << std::endl;
    }

    /**
     * @brief Print table footer with speedup
     */
    void print_footer(double scalar_ms, double avx2_ms, double avx512_ms)
    {
        std::cout << "├────────────────┴────────┴──────────────┴──────────────┤" << std::endl;
        std::cout << "│ Speedups:";

        if (avx2_ms > 0 && scalar_ms > 0)
        {
            double avx2_speedup = scalar_ms / avx2_ms;
            std::cout << "  AVX2: " << std::fixed << std::setprecision(2) << avx2_speedup << "x";
        }

        if (avx512_ms > 0 && scalar_ms > 0)
        {
            double avx512_speedup = scalar_ms / avx512_ms;
            std::cout << "  | AVX512: " << std::fixed << std::setprecision(2) << avx512_speedup << "x";
        }

        std::cout << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    }

    /**
     * @brief Benchmark scalar implementation for a specific partial size
     */
    double benchmark_scalar(const float *input, Q16_1Block *output, size_t num_blocks, size_t partial_size)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_partial_scalar(&input[i * BLOCK_SIZE], partial_size, output[i]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_partial_scalar(&input[i * BLOCK_SIZE], partial_size, output[i]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }

#if defined(__AVX2__)
    /**
     * @brief Benchmark AVX2 implementation
     */
    double benchmark_avx2(const float *input, Q16_1Block *output, size_t num_blocks, size_t partial_size)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_partial_avx2(&input[i * BLOCK_SIZE], partial_size, output[i]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_partial_avx2(&input[i * BLOCK_SIZE], partial_size, output[i]);
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
    double benchmark_avx512(const float *input, Q16_1Block *output, size_t num_blocks, size_t partial_size)
    {
        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_partial_avx512(&input[i * BLOCK_SIZE], partial_size, output[i]);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                quantize_fp32_to_q16_1_block_partial_avx512(&input[i * BLOCK_SIZE], partial_size, output[i]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count();
    }
#endif

    /**
     * @brief Run benchmark suite for a specific partial size
     */
    void run_benchmark_for_size(size_t partial_size, const std::string &label)
    {
        auto input_storage = allocate_random_input(NUM_BLOCKS);
        const float *input = get_aligned_ptr(input_storage);
        auto output = allocate_output(NUM_BLOCKS);

        print_header("Q16_1 Partial Block Quantize - " + label);

        double scalar_ms = benchmark_scalar(input, output.data(), NUM_BLOCKS, partial_size);
        print_result("Scalar", partial_size, scalar_ms, BENCHMARK_ITERATIONS, NUM_BLOCKS);

        double avx2_ms = 0.0;
        double avx512_ms = 0.0;

#if defined(__AVX2__)
        avx2_ms = benchmark_avx2(input, output.data(), NUM_BLOCKS, partial_size);
        print_result("AVX2", partial_size, avx2_ms, BENCHMARK_ITERATIONS, NUM_BLOCKS);
#endif

#if defined(__AVX512F__)
        avx512_ms = benchmark_avx512(input, output.data(), NUM_BLOCKS, partial_size);
        print_result("AVX512", partial_size, avx512_ms, BENCHMARK_ITERATIONS, NUM_BLOCKS);
#endif

        print_footer(scalar_ms, avx2_ms, avx512_ms);
    }
};

// ============================================================================
// Performance Tests for Various Partial Sizes
// ============================================================================

TEST_F(Q16_1_QuantizePartialSIMD_Perf, Partial_1_Element)
{
    run_benchmark_for_size(1, "1 element");
}

TEST_F(Q16_1_QuantizePartialSIMD_Perf, Partial_7_Elements)
{
    run_benchmark_for_size(7, "7 elements");
}

TEST_F(Q16_1_QuantizePartialSIMD_Perf, Partial_8_Elements)
{
    run_benchmark_for_size(8, "8 elements (1 YMM)");
}

TEST_F(Q16_1_QuantizePartialSIMD_Perf, Partial_15_Elements)
{
    run_benchmark_for_size(15, "15 elements");
}

TEST_F(Q16_1_QuantizePartialSIMD_Perf, Partial_16_Elements)
{
    run_benchmark_for_size(16, "16 elements (1 ZMM)");
}

TEST_F(Q16_1_QuantizePartialSIMD_Perf, Partial_24_Elements)
{
    run_benchmark_for_size(24, "24 elements (3 YMM)");
}

TEST_F(Q16_1_QuantizePartialSIMD_Perf, Partial_31_Elements)
{
    run_benchmark_for_size(31, "31 elements (max partial)");
}

// ============================================================================
// Speedup Assertion Tests
// ============================================================================

#if defined(__AVX2__)
TEST_F(Q16_1_QuantizePartialSIMD_Perf, AVX2_FasterThanScalar_16Elements)
{
    auto input_storage = allocate_random_input(NUM_BLOCKS);
    const float *input = get_aligned_ptr(input_storage);
    auto output = allocate_output(NUM_BLOCKS);

    double scalar_ms = benchmark_scalar(input, output.data(), NUM_BLOCKS, 16);
    double avx2_ms = benchmark_avx2(input, output.data(), NUM_BLOCKS, 16);

    double speedup = scalar_ms / avx2_ms;

    std::cout << "\n[Speedup] AVX2 vs Scalar (16 elements): " << std::fixed << std::setprecision(2)
              << speedup << "x\n";

    // For partial blocks, expect at least some speedup
    EXPECT_GE(speedup, 1.0) << "AVX2 should be at least as fast as scalar";
}
#endif

#if defined(__AVX512F__)
TEST_F(Q16_1_QuantizePartialSIMD_Perf, AVX512_FasterThanScalar_24Elements)
{
    auto input_storage = allocate_random_input(NUM_BLOCKS);
    const float *input = get_aligned_ptr(input_storage);
    auto output = allocate_output(NUM_BLOCKS);

    double scalar_ms = benchmark_scalar(input, output.data(), NUM_BLOCKS, 24);
    double avx512_ms = benchmark_avx512(input, output.data(), NUM_BLOCKS, 24);

    double speedup = scalar_ms / avx512_ms;

    std::cout << "\n[Speedup] AVX512 vs Scalar (24 elements): " << std::fixed << std::setprecision(2)
              << speedup << "x\n";

    // AVX512 with masked ops should provide speedup for larger partials
    EXPECT_GE(speedup, 1.0) << "AVX512 should be at least as fast as scalar";
}
#endif

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_F(Q16_1_QuantizePartialSIMD_Perf, Correctness_AllImplementationsMatch)
{
    constexpr size_t NUM_TEST_BLOCKS = 100;
    auto input_storage = allocate_random_input(NUM_TEST_BLOCKS);
    const float *input = input_storage.data();

    // Test various partial sizes
    std::vector<size_t> test_sizes = {1, 7, 8, 15, 16, 17, 24, 31};

    for (size_t partial_size : test_sizes)
    {
        std::vector<Q16_1Block> scalar_output(NUM_TEST_BLOCKS);
        std::vector<Q16_1Block> avx2_output(NUM_TEST_BLOCKS);
        std::vector<Q16_1Block> avx512_output(NUM_TEST_BLOCKS);

        // Quantize with scalar
        for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
        {
            quantize_fp32_to_q16_1_block_partial_scalar(&input[i * BLOCK_SIZE], partial_size, scalar_output[i]);
        }

#if defined(__AVX2__)
        // Quantize with AVX2
        for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
        {
            quantize_fp32_to_q16_1_block_partial_avx2(&input[i * BLOCK_SIZE], partial_size, avx2_output[i]);
        }

        // Compare AVX2 vs Scalar
        for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
        {
            ASSERT_FLOAT_EQ(scalar_output[i].d, avx2_output[i].d)
                << "AVX2 scale mismatch at block " << i << " (partial_size=" << partial_size << ")";
            ASSERT_EQ(scalar_output[i].sum_qs, avx2_output[i].sum_qs)
                << "AVX2 sum_qs mismatch at block " << i << " (partial_size=" << partial_size << ")";
            for (size_t j = 0; j < BLOCK_SIZE; ++j)
            {
                ASSERT_EQ(scalar_output[i].qs[j], avx2_output[i].qs[j])
                    << "AVX2 qs mismatch at block " << i << ", element " << j
                    << " (partial_size=" << partial_size << ")";
            }
        }
#endif

#if defined(__AVX512F__)
        // Quantize with AVX512
        for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
        {
            quantize_fp32_to_q16_1_block_partial_avx512(&input[i * BLOCK_SIZE], partial_size, avx512_output[i]);
        }

        // Compare AVX512 vs Scalar
        for (size_t i = 0; i < NUM_TEST_BLOCKS; ++i)
        {
            ASSERT_FLOAT_EQ(scalar_output[i].d, avx512_output[i].d)
                << "AVX512 scale mismatch at block " << i << " (partial_size=" << partial_size << ")";
            ASSERT_EQ(scalar_output[i].sum_qs, avx512_output[i].sum_qs)
                << "AVX512 sum_qs mismatch at block " << i << " (partial_size=" << partial_size << ")";
            for (size_t j = 0; j < BLOCK_SIZE; ++j)
            {
                ASSERT_EQ(scalar_output[i].qs[j], avx512_output[i].qs[j])
                    << "AVX512 qs mismatch at block " << i << ", element " << j
                    << " (partial_size=" << partial_size << ")";
            }
        }
#endif
    }

    std::cout << "[Correctness] All implementations match for partial sizes: ";
    for (size_t s : test_sizes)
        std::cout << s << " ";
    std::cout << "- PASS\n";
}

TEST_F(Q16_1_QuantizePartialSIMD_Perf, Correctness_ZeroFillVerification)
{
    constexpr size_t partial_size = 17;
    auto input_storage = allocate_random_input(1);
    const float *input = input_storage.data();

    Q16_1Block block;
    quantize_fp32_to_q16_1_block_partial(input, partial_size, block);

    // Verify elements beyond partial_size are zero
    for (size_t i = partial_size; i < BLOCK_SIZE; ++i)
    {
        ASSERT_EQ(block.qs[i], 0) << "Element " << i << " should be zero (partial_size=" << partial_size << ")";
    }

    std::cout << "[Correctness] Zero-fill verification for partial_size=" << partial_size << " - PASS\n";
}

TEST_F(Q16_1_QuantizePartialSIMD_Perf, RoundTrip_PartialQuantizeDequantize)
{
    constexpr size_t partial_size = 20;
    auto input_storage = allocate_random_input(1);
    const float *input = input_storage.data();

    Q16_1Block block;
    quantize_fp32_to_q16_1_block_partial(input, partial_size, block);

    float decoded[BLOCK_SIZE];
    decode_q16_1_block_to_fp32(block, decoded);

    // Check reconstruction error for valid elements
    double max_error = 0.0;
    for (size_t i = 0; i < partial_size; ++i)
    {
        double error = std::abs(static_cast<double>(input[i]) - static_cast<double>(decoded[i]));
        max_error = std::max(max_error, error);
    }

    // Verify decoded zeros for invalid elements
    for (size_t i = partial_size; i < BLOCK_SIZE; ++i)
    {
        ASSERT_FLOAT_EQ(decoded[i], 0.0f) << "Decoded element " << i << " should be zero";
    }

    std::cout << "[Round-Trip] Max error for " << partial_size << " elements: " << max_error << "\n";
    EXPECT_LT(max_error, 0.001) << "Round-trip error too large";
}
