/**
 * @file Perf__Q16_DecodeSIMD.cpp
 * @brief Performance tests for Q16 block decode (dequantization) to FP32
 *
 * Benchmarks the SIMD implementations of Q16 block decoding across all
 * supported block sizes (32, 64, 128).
 *
 * @author Llaminar Team
 * @date 2025-12-31
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

class Q16_DecodeSIMD_Perf : public ::testing::Test
{
protected:
    static constexpr size_t NUM_BLOCKS = 8192;
    static constexpr size_t NUM_ITERATIONS = 100;

    std::mt19937 rng{42};
    std::uniform_int_distribution<int16_t> int16_dist{-32000, 32000};
    std::uniform_real_distribution<float> scale_dist{0.0001f, 0.1f};

    template <typename BlockType>
    void generate_blocks(std::vector<BlockType> &blocks)
    {
        blocks.resize(NUM_BLOCKS);
        for (auto &block : blocks)
        {
            block.d = scale_dist(rng);
            int32_t sum = 0;
            for (size_t i = 0; i < BlockType::BLOCK_SIZE; ++i)
            {
                block.qs[i] = int16_dist(rng);
                sum += block.qs[i];
            }
            block.sum_qs = sum;
        }
    }

    template <typename BlockType>
    double benchmark_scalar(const std::vector<BlockType> &input)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        std::vector<float> output(NUM_BLOCKS * BLOCK_SIZE);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                decode_q16_block_to_fp32_scalar<BlockType>(input[b], output.data() + b * BLOCK_SIZE);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }

#if defined(__AVX2__)
    template <typename BlockType>
    double benchmark_avx2(const std::vector<BlockType> &input)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        std::vector<float> output(NUM_BLOCKS * BLOCK_SIZE);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                decode_q16_block_to_fp32_avx2<BlockType>(input[b], output.data() + b * BLOCK_SIZE);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }
#endif

#if defined(__AVX512F__)
    template <typename BlockType>
    double benchmark_avx512(const std::vector<BlockType> &input)
    {
        constexpr size_t BLOCK_SIZE = BlockType::BLOCK_SIZE;
        std::vector<float> output(NUM_BLOCKS * BLOCK_SIZE);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < NUM_BLOCKS; ++b)
            {
                decode_q16_block_to_fp32_avx512<BlockType>(input[b], output.data() + b * BLOCK_SIZE);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    }
#endif

    void print_results_header(const char *block_name, size_t block_size)
    {
        const double output_mb = (NUM_BLOCKS * block_size * sizeof(float)) / (1024.0 * 1024.0);
        std::cout << "\n╔════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ Q16 Decode (Dequant) Performance - " << std::setw(24) << block_name << "      ║\n";
        std::cout << "║ Blocks: " << std::setw(8) << NUM_BLOCKS
                  << " | Block Size: " << std::setw(4) << block_size
                  << " | Output: " << std::fixed << std::setprecision(1) << std::setw(6) << output_mb << " MB"
                  << "     ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "│ Implementation │   Time (ms) │    GB/s    │ Speedup vs Scalar        │\n";
        std::cout << "├────────────────┼─────────────┼────────────┼──────────────────────────┤\n";
    }

    void print_result(const char *name, double time_ms, double scalar_time_ms, size_t block_size)
    {
        // Output bytes (FP32 output)
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
// Block32 Decode Performance
// ============================================================================

TEST_F(Q16_DecodeSIMD_Perf, Block32_Decode)
{
    std::vector<Q16_1Block> blocks;
    generate_blocks<Q16_1Block>(blocks);

    // Warmup
    benchmark_scalar<Q16_1Block>(blocks);

    double scalar_time = benchmark_scalar<Q16_1Block>(blocks);

    print_results_header("Q16_1Block (32)", Q16_1Block::BLOCK_SIZE);
    print_result("Scalar", scalar_time, scalar_time, Q16_1Block::BLOCK_SIZE);

#if defined(__AVX2__)
    double avx2_time = benchmark_avx2<Q16_1Block>(blocks);
    print_result("AVX2", avx2_time, scalar_time, Q16_1Block::BLOCK_SIZE);
#endif

#if defined(__AVX512F__)
    double avx512_time = benchmark_avx512<Q16_1Block>(blocks);
    print_result("AVX512", avx512_time, scalar_time, Q16_1Block::BLOCK_SIZE);

    // NOTE: No strict performance assertions for decode - operation is memory-bound
    // (~15-17 GB/s saturates single-core DDR4 bandwidth), so all implementations
    // converge to similar throughput. Scalar code is auto-vectorized by compiler.
    (void)avx2_time; // Suppress unused warning
#endif

    print_results_footer();
}

// ============================================================================
// Block64 Decode Performance
// ============================================================================

TEST_F(Q16_DecodeSIMD_Perf, Block64_Decode)
{
    std::vector<Q16_1Block_64> blocks;
    generate_blocks<Q16_1Block_64>(blocks);

    // Warmup
    benchmark_scalar<Q16_1Block_64>(blocks);

    double scalar_time = benchmark_scalar<Q16_1Block_64>(blocks);

    print_results_header("Q16_1Block_64 (64)", Q16_1Block_64::BLOCK_SIZE);
    print_result("Scalar", scalar_time, scalar_time, Q16_1Block_64::BLOCK_SIZE);

#if defined(__AVX2__)
    double avx2_time = benchmark_avx2<Q16_1Block_64>(blocks);
    print_result("AVX2", avx2_time, scalar_time, Q16_1Block_64::BLOCK_SIZE);
#endif

#if defined(__AVX512F__)
    double avx512_time = benchmark_avx512<Q16_1Block_64>(blocks);
    print_result("AVX512", avx512_time, scalar_time, Q16_1Block_64::BLOCK_SIZE);

    // NOTE: No strict performance assertions - decode is memory-bound
    (void)avx2_time; // Suppress unused warning
#endif

    print_results_footer();
}

// ============================================================================
// Block128 Decode Performance
// ============================================================================

TEST_F(Q16_DecodeSIMD_Perf, Block128_Decode)
{
    std::vector<Q16_1Block_128> blocks;
    generate_blocks<Q16_1Block_128>(blocks);

    // Warmup
    benchmark_scalar<Q16_1Block_128>(blocks);

    double scalar_time = benchmark_scalar<Q16_1Block_128>(blocks);

    print_results_header("Q16_1Block_128 (128)", Q16_1Block_128::BLOCK_SIZE);
    print_result("Scalar", scalar_time, scalar_time, Q16_1Block_128::BLOCK_SIZE);

#if defined(__AVX2__)
    double avx2_time = benchmark_avx2<Q16_1Block_128>(blocks);
    print_result("AVX2", avx2_time, scalar_time, Q16_1Block_128::BLOCK_SIZE);
#endif

#if defined(__AVX512F__)
    double avx512_time = benchmark_avx512<Q16_1Block_128>(blocks);
    print_result("AVX512", avx512_time, scalar_time, Q16_1Block_128::BLOCK_SIZE);

    // NOTE: No strict performance assertions - decode is memory-bound
    (void)avx2_time; // Suppress unused warning
#endif

    print_results_footer();
}

// ============================================================================
// Summary Comparison Across Block Sizes
// ============================================================================

TEST_F(Q16_DecodeSIMD_Perf, Summary_AllBlockSizes)
{
    std::cout << "\n╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Q16 Decode Performance Summary (Best SIMD)                   ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "│ Block Size │ Scalar (GB/s) │ Best SIMD (GB/s) │ Speedup              │\n";
    std::cout << "├────────────┼───────────────┼──────────────────┼──────────────────────┤\n";

    auto print_summary_row = [](const char *name, size_t block_size, double scalar_time, double best_simd_time)
    {
        const double bytes = NUM_BLOCKS * block_size * sizeof(float);
        const double scalar_gbs = (bytes / (1024.0 * 1024.0 * 1024.0)) / (scalar_time / 1000.0);
        const double simd_gbs = (bytes / (1024.0 * 1024.0 * 1024.0)) / (best_simd_time / 1000.0);
        const double speedup = scalar_time / best_simd_time;

        std::cout << "│ " << std::setw(10) << name << " │"
                  << std::fixed << std::setprecision(2) << std::setw(14) << scalar_gbs << " │"
                  << std::setw(17) << simd_gbs << " │"
                  << std::setw(19) << speedup << "x │\n";
    };

    // Block32
    {
        std::vector<Q16_1Block> blocks;
        generate_blocks<Q16_1Block>(blocks);
        double scalar_time = benchmark_scalar<Q16_1Block>(blocks);
#if defined(__AVX512F__)
        double best_time = benchmark_avx512<Q16_1Block>(blocks);
#elif defined(__AVX2__)
        double best_time = benchmark_avx2<Q16_1Block>(blocks);
#else
        double best_time = scalar_time;
#endif
        print_summary_row("Block32", Q16_1Block::BLOCK_SIZE, scalar_time, best_time);
    }

    // Block64
    {
        std::vector<Q16_1Block_64> blocks;
        generate_blocks<Q16_1Block_64>(blocks);
        double scalar_time = benchmark_scalar<Q16_1Block_64>(blocks);
#if defined(__AVX512F__)
        double best_time = benchmark_avx512<Q16_1Block_64>(blocks);
#elif defined(__AVX2__)
        double best_time = benchmark_avx2<Q16_1Block_64>(blocks);
#else
        double best_time = scalar_time;
#endif
        print_summary_row("Block64", Q16_1Block_64::BLOCK_SIZE, scalar_time, best_time);
    }

    // Block128
    {
        std::vector<Q16_1Block_128> blocks;
        generate_blocks<Q16_1Block_128>(blocks);
        double scalar_time = benchmark_scalar<Q16_1Block_128>(blocks);
#if defined(__AVX512F__)
        double best_time = benchmark_avx512<Q16_1Block_128>(blocks);
#elif defined(__AVX2__)
        double best_time = benchmark_avx2<Q16_1Block_128>(blocks);
#else
        double best_time = scalar_time;
#endif
        print_summary_row("Block128", Q16_1Block_128::BLOCK_SIZE, scalar_time, best_time);
    }

    std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";
}
