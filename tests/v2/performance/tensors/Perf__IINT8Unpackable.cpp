/**
 * @file Perf__IINT8Unpackable.cpp
 * @brief Standardized performance benchmark for IINT8Unpackable superblock unpacking
 * @author David Sanftenberg
 *
 * This benchmark provides a consistent, reproducible measurement of unpack throughput
 * across all quantized tensor types implementing IINT8Unpackable.
 *
 * Test Matrix (standardized):
 * - Tensor size: 2048 x 2048 elements
 * - Warmup iterations: 2
 * - Benchmark iterations: 10
 * - API: unpack_superblock_to_int8() (superblock API)
 * - Metric: Average throughput (GB/s) over 10 runs
 *
 * Formats tested:
 * - Simple formats (32-element blocks): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, IQ4_NL
 * - K-Quant formats (256-element superblocks): Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
 * - IQuant formats (256-element superblocks): IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ1_S, IQ1_M
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */

#include "backends/GPUDeviceProcessLifetime.h"
#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <type_traits>

#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"

using namespace llaminar2;

// ============================================================================
// Standardized Test Configuration
// ============================================================================
static constexpr size_t BENCH_ROWS = 2048;
static constexpr size_t BENCH_COLS = 2048;
static constexpr int WARMUP_ITERS = 2;
static constexpr int BENCH_ITERS = 10;

/**
 * @brief Result structure for a single format benchmark
 */
struct BenchResult
{
    std::string format_name;
    size_t superblock_size;
    double avg_time_us;     ///< Average time per full tensor unpack (µs)
    double throughput_gbps; ///< Output throughput (GB/s)
    size_t total_elements;  ///< Total elements (512 * 512 = 262144)
};

class Perf__IINT8Unpackable : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    // ========================================================================
    // Tensor Creation Helpers
    // ========================================================================

    template <typename T, typename = void>
    struct has_d_member : std::false_type
    {
    };

    template <typename T>
    struct has_d_member<T, std::void_t<decltype(std::declval<T>().d)>> : std::true_type
    {
    };

    template <typename BlockT>
    std::vector<uint8_t> create_random_blocks(size_t num_blocks)
    {
        std::vector<uint8_t> raw(num_blocks * sizeof(BlockT));
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &byte : raw)
        {
            byte = static_cast<uint8_t>(dist(rng_));
        }
        if constexpr (has_d_member<BlockT>::value)
        {
            for (size_t i = 0; i < num_blocks; ++i)
            {
                BlockT *block = reinterpret_cast<BlockT *>(raw.data() + i * sizeof(BlockT));
                block->d = 0x3C00; // FP16 1.0
            }
        }
        return raw;
    }

    // Factory methods for each tensor type
    std::shared_ptr<Q4_0Tensor> create_q4_0(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q4_0Block::BLOCK_SIZE;
        auto raw = create_random_blocks<Q4_0Block>(rows * blocks_per_row);
        return std::make_shared<Q4_0Tensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<Q4_1Tensor> create_q4_1(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q4_1Block::BLOCK_SIZE;
        auto raw = create_random_blocks<Q4_1Block>(rows * blocks_per_row);
        return std::make_shared<Q4_1Tensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<Q5_0Tensor> create_q5_0(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q5_0Block::BLOCK_SIZE;
        auto raw = create_random_blocks<Q5_0Block>(rows * blocks_per_row);
        return std::make_shared<Q5_0Tensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<Q5_1Tensor> create_q5_1(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q5_1Block::BLOCK_SIZE;
        auto raw = create_random_blocks<Q5_1Block>(rows * blocks_per_row);
        return std::make_shared<Q5_1Tensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<Q8_0Tensor> create_q8_0(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q8_0Block::BLOCK_SIZE;
        auto raw = create_random_blocks<Q8_0Block>(rows * blocks_per_row);
        return std::make_shared<Q8_0Tensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<Q8_1Tensor> create_q8_1(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q8_1Block::BLOCK_SIZE;
        auto raw = create_random_blocks<Q8_1Block>(rows * blocks_per_row);
        return std::make_shared<Q8_1Tensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<IQ4_NLTensor> create_iq4_nl(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / IQ4_NLBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<IQ4_NLBlock>(rows * blocks_per_row);
        return std::make_shared<IQ4_NLTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<Q2_KTensor> create_q2_k(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q2_KBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<Q2_KBlock>(rows * blocks_per_row);
        return std::make_shared<Q2_KTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<Q3_KTensor> create_q3_k(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q3_KBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<Q3_KBlock>(rows * blocks_per_row);
        return std::make_shared<Q3_KTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<Q4_KTensor> create_q4_k(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q4_KBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<Q4_KBlock>(rows * blocks_per_row);
        return std::make_shared<Q4_KTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<Q5_KTensor> create_q5_k(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q5_KBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<Q5_KBlock>(rows * blocks_per_row);
        return std::make_shared<Q5_KTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<Q6_KTensor> create_q6_k(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / Q6_KBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<Q6_KBlock>(rows * blocks_per_row);
        return std::make_shared<Q6_KTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<IQ4_XSTensor> create_iq4_xs(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / IQ4_XSBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<IQ4_XSBlock>(rows * blocks_per_row);
        return std::make_shared<IQ4_XSTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<IQ2_XXSTensor> create_iq2_xxs(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / IQ2_XXSBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<IQ2_XXSBlock>(rows * blocks_per_row);
        return std::make_shared<IQ2_XXSTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<IQ2_XSTensor> create_iq2_xs(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / IQ2_XSBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<IQ2_XSBlock>(rows * blocks_per_row);
        return std::make_shared<IQ2_XSTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<IQ2_STensor> create_iq2_s(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / IQ2_SBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<IQ2_SBlock>(rows * blocks_per_row);
        return std::make_shared<IQ2_STensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<IQ3_XXSTensor> create_iq3_xxs(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / IQ3_XXSBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<IQ3_XXSBlock>(rows * blocks_per_row);
        return std::make_shared<IQ3_XXSTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<IQ3_STensor> create_iq3_s(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / IQ3_SBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<IQ3_SBlock>(rows * blocks_per_row);
        return std::make_shared<IQ3_STensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<IQ1_STensor> create_iq1_s(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / IQ1_SBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<IQ1_SBlock>(rows * blocks_per_row);
        return std::make_shared<IQ1_STensor>(std::vector<size_t>{rows, cols}, raw);
    }

    std::shared_ptr<IQ1_MTensor> create_iq1_m(size_t rows, size_t cols)
    {
        size_t blocks_per_row = cols / IQ1_MBlock::BLOCK_SIZE;
        auto raw = create_random_blocks<IQ1_MBlock>(rows * blocks_per_row);
        return std::make_shared<IQ1_MTensor>(std::vector<size_t>{rows, cols}, raw);
    }

    // ========================================================================
    // Benchmark Runner (Superblock API)
    // ========================================================================

    /**
     * @brief Benchmark unpack_superblock_to_int8() for a tensor
     *
     * Uses the standardized test matrix:
     * - 2 warmup iterations
     * - 10 benchmark iterations
     * - Returns average time over 10 runs
     */
    BenchResult benchmark_superblock(
        const std::string &format_name,
        const IINT8Unpackable *unpackable,
        size_t rows,
        size_t cols)
    {
        size_t superblock_size = unpackable->superblock_size();
        size_t superblocks_per_row = cols / superblock_size;
        size_t total_elements = rows * cols;

        // Allocate output buffers
        std::vector<int8_t> output(superblock_size);
        std::vector<float> scales(8);
        std::vector<float> mins(8);

        // Warmup (2 iterations)
        for (int iter = 0; iter < WARMUP_ITERS; ++iter)
        {
            for (size_t row = 0; row < rows; ++row)
            {
                for (size_t sb = 0; sb < superblocks_per_row; ++sb)
                {
                    unpackable->unpack_superblock_to_int8(row, sb, output.data(), scales.data(), mins.data());
                }
            }
        }

        // Benchmark (10 iterations)
        std::vector<double> times_us;
        times_us.reserve(BENCH_ITERS);

        for (int iter = 0; iter < BENCH_ITERS; ++iter)
        {
            auto start = std::chrono::high_resolution_clock::now();

            for (size_t row = 0; row < rows; ++row)
            {
                for (size_t sb = 0; sb < superblocks_per_row; ++sb)
                {
                    unpackable->unpack_superblock_to_int8(row, sb, output.data(), scales.data(), mins.data());
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(end - start).count();
            times_us.push_back(us);
        }

        // Compute average
        double avg_us = std::accumulate(times_us.begin(), times_us.end(), 0.0) / times_us.size();

        // Compute throughput (GB/s of int8 output)
        double bytes_output = static_cast<double>(total_elements);
        double seconds = avg_us / 1e6;
        double throughput_gbps = (bytes_output / seconds) / 1e9;

        return BenchResult{
            format_name,
            superblock_size,
            avg_us,
            throughput_gbps,
            total_elements};
    }

    // ========================================================================
    // Result Printing
    // ========================================================================

    void print_results(const std::vector<BenchResult> &results)
    {
        std::cout << "\n";
        std::cout << "╔══════════════════╦══════════════╦════════════════╦══════════════════╗\n";
        std::cout << "║ Format           ║ SuperBlk Sz  ║ Avg Time (µs)  ║ Throughput       ║\n";
        std::cout << "╠══════════════════╬══════════════╬════════════════╬══════════════════╣\n";

        for (const auto &r : results)
        {
            std::cout << "║ " << std::left << std::setw(16) << r.format_name
                      << " ║ " << std::right << std::setw(12) << r.superblock_size
                      << " ║ " << std::fixed << std::setprecision(2) << std::setw(14) << r.avg_time_us
                      << " ║ " << std::setprecision(2) << std::setw(12) << r.throughput_gbps << " GB/s ║\n";
        }

        std::cout << "╚══════════════════╩══════════════╩════════════════╩══════════════════╝\n";
        std::cout << "\n";

        // Find best and worst
        auto best = std::max_element(results.begin(), results.end(),
                                     [](const BenchResult &a, const BenchResult &b)
                                     {
                                         return a.throughput_gbps < b.throughput_gbps;
                                     });
        auto worst = std::min_element(results.begin(), results.end(),
                                      [](const BenchResult &a, const BenchResult &b)
                                      {
                                          return a.throughput_gbps < b.throughput_gbps;
                                      });

        std::cout << "Best:  " << best->format_name << " @ " << std::fixed << std::setprecision(2)
                  << best->throughput_gbps << " GB/s\n";
        std::cout << "Worst: " << worst->format_name << " @ " << std::fixed << std::setprecision(2)
                  << worst->throughput_gbps << " GB/s\n";
        std::cout << "Ratio: " << std::setprecision(1) << (best->throughput_gbps / worst->throughput_gbps) << "x\n";
    }
};

// ============================================================================
// Single Comprehensive Benchmark Test
// ============================================================================

/**
 * @brief Comprehensive benchmark of ALL IINT8Unpackable formats
 *
 * Test Matrix:
 * - Size: 2048 x 2048 elements
 * - Warmup: 2 iterations
 * - Benchmark: 10 iterations
 * - API: unpack_superblock_to_int8()
 * - Metric: Average throughput (GB/s) over 10 runs
 */
TEST_F(Perf__IINT8Unpackable, SuperblockUnpackThroughput)
{
    const size_t total_elements = BENCH_ROWS * BENCH_COLS;
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           IINT8Unpackable Superblock Unpack Benchmark                ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Test Matrix:                                                         ║\n";
    std::cout << "║   Tensor Size:      " << BENCH_ROWS << " x " << BENCH_COLS
              << " = " << std::setw(10) << total_elements << " elements"
              << std::string(17 - std::to_string(total_elements).length(), ' ') << "║\n";
    std::cout << "║   Warmup:           " << WARMUP_ITERS << " iterations                                     ║\n";
    std::cout << "║   Benchmark:        " << BENCH_ITERS << " iterations                                    ║\n";
    std::cout << "║   API:              unpack_superblock_to_int8()                      ║\n";
    std::cout << "║   Metric:           Average throughput (GB/s) over " << BENCH_ITERS << " runs           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    std::vector<BenchResult> results;

    // ---- Simple Formats (32-element blocks, superblock_size=32) ----
    std::cout << "\n=== Simple Formats (32-element superblocks) ===\n";
    {
        auto tensor = create_q4_0(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q4_0 does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q4_0", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_q4_1(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q4_1 does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q4_1", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_q5_0(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q5_0 does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q5_0", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_q5_1(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q5_1 does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q5_1", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_q8_0(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q8_0 does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q8_0", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_q8_1(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q8_1 does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q8_1", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_iq4_nl(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "IQ4_NL does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("IQ4_NL", unpackable, BENCH_ROWS, BENCH_COLS));
    }

    // ---- K-Quant Formats (256-element superblocks) ----
    std::cout << "\n=== K-Quant Formats (256-element superblocks) ===\n";
    {
        auto tensor = create_q2_k(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q2_K does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q2_K", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_q3_k(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q3_K does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q3_K", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_q4_k(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q4_K does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q4_K", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_q5_k(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q5_K does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q5_K", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_q6_k(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "Q6_K does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("Q6_K", unpackable, BENCH_ROWS, BENCH_COLS));
    }

    // ---- IQuant Formats (256-element superblocks) ----
    std::cout << "\n=== IQuant Formats (256-element superblocks) ===\n";
    {
        auto tensor = create_iq4_xs(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "IQ4_XS does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("IQ4_XS", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_iq3_s(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "IQ3_S does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("IQ3_S", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_iq3_xxs(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "IQ3_XXS does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("IQ3_XXS", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_iq2_s(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "IQ2_S does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("IQ2_S", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_iq2_xs(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "IQ2_XS does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("IQ2_XS", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_iq2_xxs(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "IQ2_XXS does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("IQ2_XXS", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_iq1_s(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "IQ1_S does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("IQ1_S", unpackable, BENCH_ROWS, BENCH_COLS));
    }
    {
        auto tensor = create_iq1_m(BENCH_ROWS, BENCH_COLS);
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr) << "IQ1_M does not implement IINT8Unpackable";
        results.push_back(benchmark_superblock("IQ1_M", unpackable, BENCH_ROWS, BENCH_COLS));
    }

    // Sort by throughput (descending)
    std::sort(results.begin(), results.end(),
              [](const BenchResult &a, const BenchResult &b)
              {
                  return a.throughput_gbps > b.throughput_gbps;
              });

    print_results(results);

    // Print ranking
    std::cout << "\n=== Ranking by Throughput ===\n";
    for (size_t i = 0; i < results.size(); ++i)
    {
        double speedup = results[i].throughput_gbps / results.back().throughput_gbps;
        std::cout << std::setw(2) << (i + 1) << ". " << std::left << std::setw(12) << results[i].format_name
                  << " " << std::fixed << std::setprecision(2) << std::setw(8) << results[i].throughput_gbps
                  << " GB/s  (" << std::setprecision(1) << speedup << "x vs slowest)\n";
    }
}

int main(int argc, char **argv)
{
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
