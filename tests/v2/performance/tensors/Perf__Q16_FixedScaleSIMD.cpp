/**
 * @file Perf__Q16_FixedScaleSIMD.cpp
 * @brief Performance tests for fixed-scale Q16 quantization (VNNI-safe)
 *
 * Benchmarks the SIMD implementations of fixed-scale quantization
 * which uses a fixed scale and clips to VNNI-safe INT16 limits.
 */

#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::simd;

class Q16_FixedScaleSIMD_Perf : public ::testing::Test
{
protected:
    static constexpr size_t NUM_BLOCKS = 8192;
    static constexpr size_t NUM_ITERATIONS = 100;

    // Fixed-scale parameters (typical for KV cache quantization)
    static constexpr float KV_CACHE_SCALE = 8.0f;
    static constexpr int16_t MAX_SAFE_INT16_64 = 23170;  // head_dim=64
    static constexpr int16_t MAX_SAFE_INT16_128 = 16383; // head_dim=128

    const float d = KV_CACHE_SCALE / 32767.0f;
    const float inv_d = 32767.0f / KV_CACHE_SCALE;

    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f}; // Some values will be clipped

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
    double benchmark_fixed_scale_scalar(const std::vector<float> &input, std::vector<BlockType> &output,
                                        int16_t max_safe)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        output.resize(NUM_BLOCKS);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                quantize_fp32_to_q16_block_fixed_scale_scalar<BlockType>(
                    input.data() + b * BLOCK_SIZE, output[b], d, inv_d, max_safe);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }

#if defined(__AVX2__)
    template <typename BlockType>
    double benchmark_fixed_scale_avx2(const std::vector<float> &input, std::vector<BlockType> &output,
                                      int16_t max_safe)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        output.resize(NUM_BLOCKS);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                quantize_fp32_to_q16_block_fixed_scale_avx2<BlockType>(
                    input.data() + b * BLOCK_SIZE, output[b], d, inv_d, max_safe);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }
#endif

#if defined(__AVX512F__)
    template <typename BlockType>
    double benchmark_fixed_scale_avx512(const std::vector<float> &input, std::vector<BlockType> &output,
                                        int16_t max_safe)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        output.resize(NUM_BLOCKS);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                quantize_fp32_to_q16_block_fixed_scale_avx512<BlockType>(
                    input.data() + b * BLOCK_SIZE, output[b], d, inv_d, max_safe);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }
#endif

    void print_results_header(const char *block_name, size_t block_size)
    {
        const double input_mb = (NUM_BLOCKS * block_size * sizeof(float)) / (1024.0 * 1024.0);
        std::cout << "\n╔════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ Q16 Fixed-Scale Quantize Performance - " << std::setw(20) << block_name << "      ║\n";
        std::cout << "║ Blocks: " << std::setw(8) << NUM_BLOCKS
                  << " | Block Size: " << std::setw(4) << block_size
                  << " | Input: " << std::fixed << std::setprecision(1) << std::setw(6) << input_mb << " MB"
                  << "      ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "│ Implementation │   Time (ms) │    GB/s    │ Speedup vs Scalar        │\n";
        std::cout << "├────────────────┼─────────────┼────────────┼──────────────────────────┤\n";
    }

    void print_result(const char *name, double time_ms, double scalar_time_ms, size_t block_size)
    {
        const double bytes = NUM_BLOCKS * block_size * sizeof(float);
        const double gb_per_sec = (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
        const double speedup = scalar_time_ms / time_ms;

        std::cout << "│ " << std::setw(14) << name << " │"
                  << std::fixed << std::setprecision(2) << std::setw(12) << time_ms << " │"
                  << std::setw(11) << gb_per_sec << " │"
                  << std::setw(24) << std::setprecision(2) << speedup << "x │\n";
    }

    void print_results_footer()
    {
        std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";
    }
};

// ============================================================================
// Block32 Fixed-Scale Performance
// ============================================================================

TEST_F(Q16_FixedScaleSIMD_Perf, Block32_FixedScale)
{
    std::vector<float> input;
    generate_fp32_input<Q16_1Block>(input);

    std::vector<Q16_1Block> output;

    // Warmup
    benchmark_fixed_scale_scalar<Q16_1Block>(input, output, MAX_SAFE_INT16_128);

    double scalar_time = benchmark_fixed_scale_scalar<Q16_1Block>(input, output, MAX_SAFE_INT16_128);

    print_results_header("Q16_1Block (32)", Q16_1Block::BLOCK_SIZE);
    print_result("Scalar", scalar_time, scalar_time, Q16_1Block::BLOCK_SIZE);

#if defined(__AVX2__)
    double avx2_time = benchmark_fixed_scale_avx2<Q16_1Block>(input, output, MAX_SAFE_INT16_128);
    print_result("AVX2", avx2_time, scalar_time, Q16_1Block::BLOCK_SIZE);
#endif

#if defined(__AVX512F__)
    double avx512_time = benchmark_fixed_scale_avx512<Q16_1Block>(input, output, MAX_SAFE_INT16_128);
    print_result("AVX512", avx512_time, scalar_time, Q16_1Block::BLOCK_SIZE);
#endif

    print_results_footer();
}

// ============================================================================
// Block64 Fixed-Scale Performance
// ============================================================================

TEST_F(Q16_FixedScaleSIMD_Perf, Block64_FixedScale)
{
    std::vector<float> input;
    generate_fp32_input<Q16_1Block_64>(input);

    std::vector<Q16_1Block_64> output;

    // Warmup
    benchmark_fixed_scale_scalar<Q16_1Block_64>(input, output, MAX_SAFE_INT16_64);

    double scalar_time = benchmark_fixed_scale_scalar<Q16_1Block_64>(input, output, MAX_SAFE_INT16_64);

    print_results_header("Q16_1Block_64 (64)", Q16_1Block_64::BLOCK_SIZE);
    print_result("Scalar", scalar_time, scalar_time, Q16_1Block_64::BLOCK_SIZE);

#if defined(__AVX2__)
    double avx2_time = benchmark_fixed_scale_avx2<Q16_1Block_64>(input, output, MAX_SAFE_INT16_64);
    print_result("AVX2", avx2_time, scalar_time, Q16_1Block_64::BLOCK_SIZE);
#endif

#if defined(__AVX512F__)
    double avx512_time = benchmark_fixed_scale_avx512<Q16_1Block_64>(input, output, MAX_SAFE_INT16_64);
    print_result("AVX512", avx512_time, scalar_time, Q16_1Block_64::BLOCK_SIZE);
#endif

    print_results_footer();
}

// ============================================================================
// Block128 Fixed-Scale Performance
// ============================================================================

TEST_F(Q16_FixedScaleSIMD_Perf, Block128_FixedScale)
{
    std::vector<float> input;
    generate_fp32_input<Q16_1Block_128>(input);

    std::vector<Q16_1Block_128> output;

    // Warmup
    benchmark_fixed_scale_scalar<Q16_1Block_128>(input, output, MAX_SAFE_INT16_128);

    double scalar_time = benchmark_fixed_scale_scalar<Q16_1Block_128>(input, output, MAX_SAFE_INT16_128);

    print_results_header("Q16_1Block_128 (128)", Q16_1Block_128::BLOCK_SIZE);
    print_result("Scalar", scalar_time, scalar_time, Q16_1Block_128::BLOCK_SIZE);

#if defined(__AVX2__)
    double avx2_time = benchmark_fixed_scale_avx2<Q16_1Block_128>(input, output, MAX_SAFE_INT16_128);
    print_result("AVX2", avx2_time, scalar_time, Q16_1Block_128::BLOCK_SIZE);
#endif

#if defined(__AVX512F__)
    double avx512_time = benchmark_fixed_scale_avx512<Q16_1Block_128>(input, output, MAX_SAFE_INT16_128);
    print_result("AVX512", avx512_time, scalar_time, Q16_1Block_128::BLOCK_SIZE);
#endif

    print_results_footer();
}

// ============================================================================
// Comparison: Fixed-Scale vs Adaptive Quantization
// ============================================================================

TEST_F(Q16_FixedScaleSIMD_Perf, FixedScale_vs_Adaptive_Block64)
{
    std::vector<float> input;
    generate_fp32_input<Q16_1Block_64>(input);

    std::vector<Q16_1Block_64> output;

    // Warmup
    benchmark_fixed_scale_scalar<Q16_1Block_64>(input, output, MAX_SAFE_INT16_64);

    std::cout << "\n╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Fixed-Scale vs Adaptive Quantization - Q16_1Block_64                   ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "│ Note: Fixed-scale is FASTER because it skips max-finding pass          │\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";

    // Benchmark adaptive (existing)
    auto benchmark_adaptive = [&]() -> double
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                quantize_fp32_to_q16_block<Q16_1Block_64>(
                    input.data() + b * Q16_1Block_64::BLOCK_SIZE, output[b]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    };

    double adaptive_time = benchmark_adaptive();

#if defined(__AVX512F__)
    double fixed_time = benchmark_fixed_scale_avx512<Q16_1Block_64>(input, output, MAX_SAFE_INT16_64);
#elif defined(__AVX2__)
    double fixed_time = benchmark_fixed_scale_avx2<Q16_1Block_64>(input, output, MAX_SAFE_INT16_64);
#else
    double fixed_time = benchmark_fixed_scale_scalar<Q16_1Block_64>(input, output, MAX_SAFE_INT16_64);
#endif

    const double bytes = NUM_BLOCKS * Q16_1Block_64::BLOCK_SIZE * sizeof(float);
    const double adaptive_gbps = (bytes / (1024.0 * 1024.0 * 1024.0)) / (adaptive_time / 1000.0);
    const double fixed_gbps = (bytes / (1024.0 * 1024.0 * 1024.0)) / (fixed_time / 1000.0);

    std::cout << "│ Adaptive (best SIMD) │ " << std::fixed << std::setprecision(2)
              << std::setw(8) << adaptive_time << " ms │ " << std::setw(8) << adaptive_gbps << " GB/s │\n";
    std::cout << "│ Fixed-Scale (best)   │ " << std::fixed << std::setprecision(2)
              << std::setw(8) << fixed_time << " ms │ " << std::setw(8) << fixed_gbps << " GB/s │\n";
    std::cout << "│ Fixed-Scale Speedup  │ "
              << std::setw(8) << (adaptive_time / fixed_time) << "x                         │\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";
}

// ============================================================================
// Partial Block Performance
// ============================================================================

TEST_F(Q16_FixedScaleSIMD_Perf, Block64_Partial_FixedScale)
{
    const size_t count = 48; // 48 of 64 elements
    std::vector<float> input;
    generate_fp32_input<Q16_1Block_64>(input);

    std::vector<Q16_1Block_64> output(NUM_BLOCKS);

    // Benchmark scalar partial
    auto benchmark_partial_scalar = [&]() -> double
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                quantize_fp32_to_q16_block_partial_fixed_scale_scalar<Q16_1Block_64>(
                    input.data() + b * Q16_1Block_64::BLOCK_SIZE, count, output[b],
                    d, inv_d, MAX_SAFE_INT16_64);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    };

    double scalar_time = benchmark_partial_scalar();

    std::cout << "\n╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Q16_1Block_64 Partial Fixed-Scale (48/64 elements)                     ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";

#if defined(__AVX2__)
    auto benchmark_partial_avx2 = [&]() -> double
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                quantize_fp32_to_q16_block_partial_fixed_scale_avx2<Q16_1Block_64>(
                    input.data() + b * Q16_1Block_64::BLOCK_SIZE, count, output[b],
                    d, inv_d, MAX_SAFE_INT16_64);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    };

    double avx2_time = benchmark_partial_avx2();
#endif

#if defined(__AVX512F__)
    auto benchmark_partial_avx512 = [&]() -> double
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                quantize_fp32_to_q16_block_partial_fixed_scale_avx512<Q16_1Block_64>(
                    input.data() + b * Q16_1Block_64::BLOCK_SIZE, count, output[b],
                    d, inv_d, MAX_SAFE_INT16_64);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    };

    double avx512_time = benchmark_partial_avx512();
#endif

    std::cout << "│ Scalar: " << std::fixed << std::setprecision(2) << scalar_time << " ms";
#if defined(__AVX2__)
    std::cout << " | AVX2: " << avx2_time << " ms (" << (scalar_time / avx2_time) << "x)";
#endif
#if defined(__AVX512F__)
    std::cout << " | AVX512: " << avx512_time << " ms (" << (scalar_time / avx512_time) << "x)";
#endif
    std::cout << "\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";
}
