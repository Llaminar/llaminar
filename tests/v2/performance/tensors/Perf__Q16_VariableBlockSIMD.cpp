/**
 * @file Perf__Q16_VariableBlockSIMD.cpp
 * @brief Performance tests for variable block size Q16 quantization (32, 64, 128)
 *
 * Tests the templated SIMD quantization functions that work with:
 * - Q16_1Block (32 elements)
 * - Q16_1Block_64 (64 elements)
 * - Q16_1Block_128 (128 elements)
 */

#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::simd;

class Q16_VariableBlockSIMD_Perf : public ::testing::Test
{
protected:
    static constexpr size_t NUM_BLOCKS = 8192;
    static constexpr size_t NUM_ITERATIONS = 100;

    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};

    template <typename BlockType>
    void generate_fp32_input(std::vector<float> &input)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        input.resize(NUM_BLOCKS * BLOCK_SIZE);
        for (auto &v : input)
        {
            v = dist(rng);
        }
    }

    template <typename BlockType>
    double benchmark_quantize_scalar(const std::vector<float> &input, std::vector<BlockType> &output)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        output.resize(NUM_BLOCKS);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                quantize_fp32_to_q16_block_scalar<BlockType>(input.data() + b * BLOCK_SIZE, output[b]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }

#if defined(__AVX2__)
    template <typename BlockType>
    double benchmark_quantize_avx2(const std::vector<float> &input, std::vector<BlockType> &output)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        output.resize(NUM_BLOCKS);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                quantize_fp32_to_q16_block_avx2<BlockType>(input.data() + b * BLOCK_SIZE, output[b]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }
#endif

#if defined(__AVX512F__)
    template <typename BlockType>
    double benchmark_quantize_avx512(const std::vector<float> &input, std::vector<BlockType> &output)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        output.resize(NUM_BLOCKS);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                quantize_fp32_to_q16_block_avx512<BlockType>(input.data() + b * BLOCK_SIZE, output[b]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }
#endif

    template <typename BlockType>
    bool verify_correctness(const std::vector<float> &input,
                            const std::vector<BlockType> &scalar_output,
                            const std::vector<BlockType> &simd_output,
                            const std::string &simd_name)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        for (size_t b = 0; b < NUM_BLOCKS; ++b)
        {
            // Check scale factor
            if (std::abs(scalar_output[b].d - simd_output[b].d) > 1e-6f)
            {
                std::cerr << simd_name << " block " << b << " d mismatch: "
                          << scalar_output[b].d << " vs " << simd_output[b].d << std::endl;
                return false;
            }

            // Check sum_qs
            if (scalar_output[b].sum_qs != simd_output[b].sum_qs)
            {
                std::cerr << simd_name << " block " << b << " sum_qs mismatch: "
                          << scalar_output[b].sum_qs << " vs " << simd_output[b].sum_qs << std::endl;
                return false;
            }

            // Check quantized values
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
            {
                if (scalar_output[b].qs[i] != simd_output[b].qs[i])
                {
                    std::cerr << simd_name << " block " << b << " qs[" << i << "] mismatch: "
                              << scalar_output[b].qs[i] << " vs " << simd_output[b].qs[i] << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    void print_results(const std::string &block_name, size_t block_size,
                       double scalar_ms, double avx2_ms, double avx512_ms)
    {
        const double input_bytes = NUM_BLOCKS * block_size * sizeof(float);
        const double input_mb = input_bytes / (1024.0 * 1024.0);

        std::cout << "\n╔════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ Q16 Variable Block Quantize Performance - " << std::setw(20) << std::left << block_name << "        ║\n";
        std::cout << "║ Blocks: " << std::setw(8) << NUM_BLOCKS
                  << " | Block Size: " << std::setw(4) << block_size
                  << " | Input: " << std::setw(6) << std::fixed << std::setprecision(1) << input_mb << " MB      ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "│ Implementation │   Time (ms) │    GB/s    │ Speedup vs Scalar        │\n";
        std::cout << "├────────────────┼─────────────┼────────────┼──────────────────────────┤\n";

        auto print_row = [&](const char *name, double ms, double speedup)
        {
            double gb_per_sec = (input_bytes / (1e9)) / (ms / 1000.0);
            std::cout << "│ " << std::setw(14) << std::left << name << " │ "
                      << std::setw(11) << std::right << std::fixed << std::setprecision(2) << ms << " │ "
                      << std::setw(10) << std::fixed << std::setprecision(2) << gb_per_sec << " │ "
                      << std::setw(24) << std::fixed << std::setprecision(2) << speedup << "x │\n";
        };

        print_row("Scalar", scalar_ms, 1.0);
        if (avx2_ms > 0)
            print_row("AVX2", avx2_ms, scalar_ms / avx2_ms);
        if (avx512_ms > 0)
            print_row("AVX512", avx512_ms, scalar_ms / avx512_ms);

        std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";
    }
};

// ============================================================================
// Block Size 32 Tests
// ============================================================================

TEST_F(Q16_VariableBlockSIMD_Perf, Block32_Quantize)
{
    std::vector<float> input;
    generate_fp32_input<Q16_1Block>(input);

    std::vector<Q16_1Block> scalar_out, avx2_out, avx512_out;

    double scalar_ms = benchmark_quantize_scalar<Q16_1Block>(input, scalar_out);
    double avx2_ms = 0, avx512_ms = 0;

#if defined(__AVX2__)
    avx2_ms = benchmark_quantize_avx2<Q16_1Block>(input, avx2_out);
    EXPECT_TRUE(verify_correctness(input, scalar_out, avx2_out, "AVX2"));
#endif

#if defined(__AVX512F__)
    avx512_ms = benchmark_quantize_avx512<Q16_1Block>(input, avx512_out);
    EXPECT_TRUE(verify_correctness(input, scalar_out, avx512_out, "AVX512"));
#endif

    print_results("Q16_1Block (32)", Q16_1Block::BLOCK_SIZE, scalar_ms, avx2_ms, avx512_ms);

    // Verify speedups
#if defined(__AVX2__)
    EXPECT_GT(scalar_ms / avx2_ms, 1.5) << "AVX2 should be at least 1.5x faster than scalar";
#endif
#if defined(__AVX512F__)
    EXPECT_GT(scalar_ms / avx512_ms, 2.0) << "AVX512 should be at least 2x faster than scalar";
#endif
}

// ============================================================================
// Block Size 64 Tests
// ============================================================================

TEST_F(Q16_VariableBlockSIMD_Perf, Block64_Quantize)
{
    std::vector<float> input;
    generate_fp32_input<Q16_1Block_64>(input);

    std::vector<Q16_1Block_64> scalar_out, avx2_out, avx512_out;

    double scalar_ms = benchmark_quantize_scalar<Q16_1Block_64>(input, scalar_out);
    double avx2_ms = 0, avx512_ms = 0;

#if defined(__AVX2__)
    avx2_ms = benchmark_quantize_avx2<Q16_1Block_64>(input, avx2_out);
    EXPECT_TRUE(verify_correctness(input, scalar_out, avx2_out, "AVX2"));
#endif

#if defined(__AVX512F__)
    avx512_ms = benchmark_quantize_avx512<Q16_1Block_64>(input, avx512_out);
    EXPECT_TRUE(verify_correctness(input, scalar_out, avx512_out, "AVX512"));
#endif

    print_results("Q16_1Block_64 (64)", Q16_1Block_64::BLOCK_SIZE, scalar_ms, avx2_ms, avx512_ms);

    // Verify speedups
#if defined(__AVX2__)
    EXPECT_GT(scalar_ms / avx2_ms, 1.5) << "AVX2 should be at least 1.5x faster than scalar";
#endif
#if defined(__AVX512F__)
    EXPECT_GT(scalar_ms / avx512_ms, 2.0) << "AVX512 should be at least 2x faster than scalar";
#endif
}

// ============================================================================
// Block Size 128 Tests
// ============================================================================

TEST_F(Q16_VariableBlockSIMD_Perf, Block128_Quantize)
{
    std::vector<float> input;
    generate_fp32_input<Q16_1Block_128>(input);

    std::vector<Q16_1Block_128> scalar_out, avx2_out, avx512_out;

    double scalar_ms = benchmark_quantize_scalar<Q16_1Block_128>(input, scalar_out);
    double avx2_ms = 0, avx512_ms = 0;

#if defined(__AVX2__)
    avx2_ms = benchmark_quantize_avx2<Q16_1Block_128>(input, avx2_out);
    EXPECT_TRUE(verify_correctness(input, scalar_out, avx2_out, "AVX2"));
#endif

#if defined(__AVX512F__)
    avx512_ms = benchmark_quantize_avx512<Q16_1Block_128>(input, avx512_out);
    EXPECT_TRUE(verify_correctness(input, scalar_out, avx512_out, "AVX512"));
#endif

    print_results("Q16_1Block_128 (128)", Q16_1Block_128::BLOCK_SIZE, scalar_ms, avx2_ms, avx512_ms);

    // Verify speedups
#if defined(__AVX2__)
    EXPECT_GT(scalar_ms / avx2_ms, 1.5) << "AVX2 should be at least 1.5x faster than scalar";
#endif
#if defined(__AVX512F__)
    EXPECT_GT(scalar_ms / avx512_ms, 2.0) << "AVX512 should be at least 2x faster than scalar";
#endif
}

// ============================================================================
// Partial Block Tests (Variable Block Sizes)
// ============================================================================

TEST_F(Q16_VariableBlockSIMD_Perf, Block64_Partial_Quantize)
{
    // Test partial block with 48 elements (75% full)
    constexpr size_t PARTIAL_COUNT = 48;
    std::vector<float> input(NUM_BLOCKS * Q16_1Block_64::BLOCK_SIZE);
    for (auto &v : input)
    {
        v = dist(rng);
    }

    std::vector<Q16_1Block_64> scalar_out(NUM_BLOCKS), avx2_out(NUM_BLOCKS), avx512_out(NUM_BLOCKS);

    // Scalar
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        for (size_t b = 0; b < NUM_BLOCKS; ++b)
        {
            quantize_fp32_to_q16_block_partial_scalar<Q16_1Block_64>(
                input.data() + b * Q16_1Block_64::BLOCK_SIZE, PARTIAL_COUNT, scalar_out[b]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;

    double avx2_ms = 0, avx512_ms = 0;

#if defined(__AVX2__)
    start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        for (size_t b = 0; b < NUM_BLOCKS; ++b)
        {
            quantize_fp32_to_q16_block_partial_avx2<Q16_1Block_64>(
                input.data() + b * Q16_1Block_64::BLOCK_SIZE, PARTIAL_COUNT, avx2_out[b]);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    avx2_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    EXPECT_TRUE(verify_correctness(input, scalar_out, avx2_out, "AVX2 Partial"));
#endif

#if defined(__AVX512F__)
    start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        for (size_t b = 0; b < NUM_BLOCKS; ++b)
        {
            quantize_fp32_to_q16_block_partial_avx512<Q16_1Block_64>(
                input.data() + b * Q16_1Block_64::BLOCK_SIZE, PARTIAL_COUNT, avx512_out[b]);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    avx512_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    EXPECT_TRUE(verify_correctness(input, scalar_out, avx512_out, "AVX512 Partial"));
#endif

    std::cout << "\n╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Q16_1Block_64 Partial Quantize (48/64 elements)                          ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "│ Scalar: " << std::fixed << std::setprecision(2) << scalar_ms << " ms";
    if (avx2_ms > 0)
        std::cout << " | AVX2: " << avx2_ms << " ms (" << (scalar_ms / avx2_ms) << "x)";
    if (avx512_ms > 0)
        std::cout << " | AVX512: " << avx512_ms << " ms (" << (scalar_ms / avx512_ms) << "x)";
    std::cout << "\n╚════════════════════════════════════════════════════════════════════════╝\n";
}

TEST_F(Q16_VariableBlockSIMD_Perf, Block128_Partial_Quantize)
{
    // Test partial block with 96 elements (75% full)
    constexpr size_t PARTIAL_COUNT = 96;
    std::vector<float> input(NUM_BLOCKS * Q16_1Block_128::BLOCK_SIZE);
    for (auto &v : input)
    {
        v = dist(rng);
    }

    std::vector<Q16_1Block_128> scalar_out(NUM_BLOCKS), avx2_out(NUM_BLOCKS), avx512_out(NUM_BLOCKS);

    // Scalar
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        for (size_t b = 0; b < NUM_BLOCKS; ++b)
        {
            quantize_fp32_to_q16_block_partial_scalar<Q16_1Block_128>(
                input.data() + b * Q16_1Block_128::BLOCK_SIZE, PARTIAL_COUNT, scalar_out[b]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;

    double avx2_ms = 0, avx512_ms = 0;

#if defined(__AVX2__)
    start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        for (size_t b = 0; b < NUM_BLOCKS; ++b)
        {
            quantize_fp32_to_q16_block_partial_avx2<Q16_1Block_128>(
                input.data() + b * Q16_1Block_128::BLOCK_SIZE, PARTIAL_COUNT, avx2_out[b]);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    avx2_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    EXPECT_TRUE(verify_correctness(input, scalar_out, avx2_out, "AVX2 Partial"));
#endif

#if defined(__AVX512F__)
    start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        for (size_t b = 0; b < NUM_BLOCKS; ++b)
        {
            quantize_fp32_to_q16_block_partial_avx512<Q16_1Block_128>(
                input.data() + b * Q16_1Block_128::BLOCK_SIZE, PARTIAL_COUNT, avx512_out[b]);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    avx512_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    EXPECT_TRUE(verify_correctness(input, scalar_out, avx512_out, "AVX512 Partial"));
#endif

    std::cout << "\n╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Q16_1Block_128 Partial Quantize (96/128 elements)                        ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "│ Scalar: " << std::fixed << std::setprecision(2) << scalar_ms << " ms";
    if (avx2_ms > 0)
        std::cout << " | AVX2: " << avx2_ms << " ms (" << (scalar_ms / avx2_ms) << "x)";
    if (avx512_ms > 0)
        std::cout << " | AVX512: " << avx512_ms << " ms (" << (scalar_ms / avx512_ms) << "x)";
    std::cout << "\n╚════════════════════════════════════════════════════════════════════════╝\n";
}
