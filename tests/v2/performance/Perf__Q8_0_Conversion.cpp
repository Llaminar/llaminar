/**
 * @file Perf__Q8_0_Conversion.cpp
 * @brief Performance benchmarks for Q8_0 conversion with SIMD variants
 * @author David Sanftenberg
 *
 * Tests:
 * 1. Scalar vs AVX2 vs AVX512 comparison (compile-time dispatch)
 * 2. Single-threaded vs multi-threaded OpenMP parallelization
 * 3. Per-block throughput for all tensor types (FP32, BF16, FP16, INT32)
 * 4. Large-scale conversion throughput (millions of elements)
 *
 * Expected results (AVX512):
 * - SIMD speedup: 8-16× vs scalar for per-block operations
 * - OpenMP speedup: Linear scaling with physical cores (up to 28×)
 * - Overall: 100-300× combined speedup on multi-core AVX512 systems
 */

#include <gtest/gtest.h>
#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/tensors/Q8_0Helpers.h"
#include "../../src/v2/utils/Logger.h"
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>

using namespace llaminar2;

namespace
{
    // Test configuration
    constexpr size_t WARMUP_ITERATIONS = 10;
    constexpr size_t BENCHMARK_ITERATIONS = 1000;
    constexpr size_t LARGE_SCALE_ELEMENTS = 1024 * 1024; // 1M elements

    // Timing utilities
    class Timer
    {
        std::chrono::high_resolution_clock::time_point start_;

    public:
        Timer() : start_(std::chrono::high_resolution_clock::now()) {}

        double elapsed_ms() const
        {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start_).count();
        }

        void reset() { start_ = std::chrono::high_resolution_clock::now(); }
    };

    // Generate test data with realistic distribution
    void generate_test_data_fp32(float *data, size_t count, float scale = 1.0f)
    {
        for (size_t i = 0; i < count; ++i)
        {
            // Mix of small, medium, large values
            float base = static_cast<float>(i) / count;
            data[i] = scale * (std::sin(base * 10.0f) + 0.1f * std::cos(base * 100.0f));
        }
    }

    void generate_test_data_bf16(uint16_t *data, size_t count, float scale = 1.0f)
    {
        for (size_t i = 0; i < count; ++i)
        {
            float base = static_cast<float>(i) / count;
            float val = scale * (std::sin(base * 10.0f) + 0.1f * std::cos(base * 100.0f));
            data[i] = simd::fp32_to_bf16(val);
        }
    }

    void generate_test_data_fp16(uint16_t *data, size_t count, float scale = 1.0f)
    {
        for (size_t i = 0; i < count; ++i)
        {
            float base = static_cast<float>(i) / count;
            float val = scale * (std::sin(base * 10.0f) + 0.1f * std::cos(base * 100.0f));
            data[i] = fp32_to_fp16(val);
        }
    }

    // Print benchmark header
    void print_benchmark_header(const std::string &title)
    {
        std::cout << "\n"
                  << std::string(80, '=') << "\n"
                  << title << "\n"
                  << std::string(80, '=') << std::endl;
    }

    // Print performance metrics
    void print_performance(const std::string &name, double time_ms, size_t operations, double baseline_ms = 0.0)
    {
        double ops_per_sec = (operations * 1000.0) / time_ms;
        double time_per_op_us = (time_ms * 1000.0) / operations;

        std::cout << std::left << std::setw(30) << name << ": "
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3)
                  << time_ms << " ms  ("
                  << std::setw(12) << std::setprecision(2) << ops_per_sec << " ops/s, "
                  << std::setw(8) << std::setprecision(3) << time_per_op_us << " μs/op)";

        if (baseline_ms > 0.0)
        {
            double speedup = baseline_ms / time_ms;
            std::cout << "  [" << std::setw(6) << std::setprecision(2) << speedup << "× speedup]";
        }

        std::cout << std::endl;
    }

} // anonymous namespace

// =============================================================================
// Test 1: SIMD Variant Comparison (Scalar vs AVX2 vs AVX512)
// =============================================================================

TEST(Q8_0_ConversionPerf, SIMD_Variants_FP32)
{
    print_benchmark_header("SIMD Variant Performance: FP32 → Q8_0");

    constexpr size_t BLOCK_COUNT = 1000;
    constexpr size_t TOTAL_ELEMENTS = BLOCK_COUNT * Q8_0Block::BLOCK_SIZE;

    // Prepare test data
    alignas(64) float fp32_data[TOTAL_ELEMENTS];
    generate_test_data_fp32(fp32_data, TOTAL_ELEMENTS, 10.0f);

    std::vector<Q8_0Block> blocks(BLOCK_COUNT);

    // Warmup
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i)
    {
        for (size_t b = 0; b < BLOCK_COUNT; ++b)
        {
            simd::quantize_fp32_to_q8_0_block(fp32_data + b * Q8_0Block::BLOCK_SIZE, blocks[b], Q8_0Block::BLOCK_SIZE);
        }
    }

    // Benchmark: Single-threaded (SIMD auto-selected based on compilation)
    Timer timer;
    for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
    {
        for (size_t b = 0; b < BLOCK_COUNT; ++b)
        {
            simd::quantize_fp32_to_q8_0_block(fp32_data + b * Q8_0Block::BLOCK_SIZE, blocks[b], Q8_0Block::BLOCK_SIZE);
        }
    }
    double time_simd = timer.elapsed_ms();

    size_t total_ops = BENCHMARK_ITERATIONS * BLOCK_COUNT;

    std::cout << "\nConfiguration:" << std::endl;
#if defined(__AVX512F__)
    std::cout << "  SIMD Level: AVX512 (16 floats/iteration)" << std::endl;
#elif defined(__AVX2__)
    std::cout << "  SIMD Level: AVX2 (8 floats/iteration)" << std::endl;
#else
    std::cout << "  SIMD Level: Scalar (1 float/iteration)" << std::endl;
#endif
    std::cout << "  Block Count: " << BLOCK_COUNT << std::endl;
    std::cout << "  Iterations: " << BENCHMARK_ITERATIONS << std::endl;
    std::cout << "  Total Operations: " << total_ops << " blocks" << std::endl;

    std::cout << "\nResults:" << std::endl;
    print_performance("FP32 → Q8_0 (SIMD)", time_simd, total_ops);

    // Calculate throughput
    double elements_per_sec = (TOTAL_ELEMENTS * BENCHMARK_ITERATIONS * 1000.0) / time_simd;
    double mb_per_sec = (TOTAL_ELEMENTS * sizeof(float) * BENCHMARK_ITERATIONS) / (time_simd * 1024.0);

    std::cout << "\nThroughput:" << std::endl;
    std::cout << "  Elements/sec: " << std::fixed << std::setprecision(0) << elements_per_sec << std::endl;
    std::cout << "  Bandwidth: " << std::setprecision(2) << mb_per_sec << " MB/s" << std::endl;

    // Verify correctness (sanity check)
    bool any_nonzero = false;
    for (const auto &block : blocks)
    {
        if (block.d != 0)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero) << "All blocks have zero scale (likely bug)";
}

// =============================================================================
// Test 2: OpenMP Parallelization Scaling
// =============================================================================

TEST(Q8_0_ConversionPerf, OpenMP_Scaling_FP32)
{
    print_benchmark_header("OpenMP Parallelization Scaling: FP32 → Q8_0");

    constexpr size_t BLOCK_COUNT = 10000; // Large enough to see parallel benefit
    constexpr size_t TOTAL_ELEMENTS = BLOCK_COUNT * Q8_0Block::BLOCK_SIZE;

    // Prepare test data
    std::vector<float> fp32_data(TOTAL_ELEMENTS);
    generate_test_data_fp32(fp32_data.data(), TOTAL_ELEMENTS, 10.0f);

    std::vector<Q8_0Block> blocks(BLOCK_COUNT);

    // Get available thread counts
    int max_threads = omp_get_max_threads();
    std::vector<int> thread_counts = {1, 2, 4, 8};
    if (max_threads >= 16)
        thread_counts.push_back(16);
    if (max_threads >= 28)
        thread_counts.push_back(max_threads); // Test full parallelism

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Max OMP Threads: " << max_threads << std::endl;
    std::cout << "  Block Count: " << BLOCK_COUNT << std::endl;
    std::cout << "  Iterations: " << BENCHMARK_ITERATIONS << std::endl;

    std::cout << "\nResults:" << std::endl;
    double baseline_time = 0.0;

    for (int num_threads : thread_counts)
    {
        if (num_threads > max_threads)
            continue;

        omp_set_num_threads(num_threads);

        // Warmup
        for (size_t i = 0; i < WARMUP_ITERATIONS; ++i)
        {
#pragma omp parallel for
            for (size_t b = 0; b < BLOCK_COUNT; ++b)
            {
                simd::quantize_fp32_to_q8_0_block(fp32_data.data() + b * Q8_0Block::BLOCK_SIZE,
                                                  blocks[b],
                                                  Q8_0Block::BLOCK_SIZE);
            }
        }

        // Benchmark
        Timer timer;
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
#pragma omp parallel for
            for (size_t b = 0; b < BLOCK_COUNT; ++b)
            {
                simd::quantize_fp32_to_q8_0_block(fp32_data.data() + b * Q8_0Block::BLOCK_SIZE,
                                                  blocks[b],
                                                  Q8_0Block::BLOCK_SIZE);
            }
        }
        double time_ms = timer.elapsed_ms();

        if (num_threads == 1)
        {
            baseline_time = time_ms;
        }

        std::string name = std::to_string(num_threads) + " thread" + (num_threads > 1 ? "s" : "");
        print_performance(name, time_ms, BENCHMARK_ITERATIONS * BLOCK_COUNT, baseline_time);
    }

    // Reset to max threads
    omp_set_num_threads(max_threads);
}

// =============================================================================
// Test 3: All Tensor Types Comparison
// =============================================================================

TEST(Q8_0_ConversionPerf, AllTypes_Comparison)
{
    print_benchmark_header("Q8_0 Conversion: All Tensor Types");

    constexpr size_t BLOCK_COUNT = 1000;
    constexpr size_t TOTAL_ELEMENTS = BLOCK_COUNT * Q8_0Block::BLOCK_SIZE;

    // Prepare test data for all types
    std::vector<float> fp32_data(TOTAL_ELEMENTS);
    std::vector<uint16_t> bf16_data(TOTAL_ELEMENTS);
    std::vector<uint16_t> fp16_data(TOTAL_ELEMENTS);
    std::vector<int32_t> int32_data(TOTAL_ELEMENTS);

    generate_test_data_fp32(fp32_data.data(), TOTAL_ELEMENTS, 10.0f);
    generate_test_data_bf16(bf16_data.data(), TOTAL_ELEMENTS, 10.0f);
    generate_test_data_fp16(fp16_data.data(), TOTAL_ELEMENTS, 10.0f);

    // Generate INT32 data (simulate accumulator values)
    for (size_t i = 0; i < TOTAL_ELEMENTS; ++i)
    {
        int32_data[i] = static_cast<int32_t>(fp32_data[i] * 127.0f);
    }

    std::vector<Q8_0Block> blocks(BLOCK_COUNT);

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Block Count: " << BLOCK_COUNT << std::endl;
    std::cout << "  Iterations: " << BENCHMARK_ITERATIONS << std::endl;
    std::cout << "  OMP Threads: " << omp_get_max_threads() << std::endl;

    std::cout << "\nResults (single-threaded):" << std::endl;

    // Disable OpenMP for fair comparison
    omp_set_num_threads(1);

    // Benchmark FP32
    {
        Timer timer;
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < BLOCK_COUNT; ++b)
            {
                simd::quantize_fp32_to_q8_0_block(fp32_data.data() + b * Q8_0Block::BLOCK_SIZE,
                                                  blocks[b],
                                                  Q8_0Block::BLOCK_SIZE);
            }
        }
        print_performance("FP32 → Q8_0", timer.elapsed_ms(), BENCHMARK_ITERATIONS * BLOCK_COUNT);
    }

    // Benchmark BF16
    {
        Timer timer;
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < BLOCK_COUNT; ++b)
            {
                simd::quantize_bf16_to_q8_0_block(bf16_data.data() + b * Q8_0Block::BLOCK_SIZE,
                                                  blocks[b],
                                                  Q8_0Block::BLOCK_SIZE);
            }
        }
        print_performance("BF16 → Q8_0", timer.elapsed_ms(), BENCHMARK_ITERATIONS * BLOCK_COUNT);
    }

    // Benchmark FP16
    {
        Timer timer;
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < BLOCK_COUNT; ++b)
            {
                simd::quantize_fp16_to_q8_0_block(fp16_data.data() + b * Q8_0Block::BLOCK_SIZE,
                                                  blocks[b],
                                                  Q8_0Block::BLOCK_SIZE);
            }
        }
        print_performance("FP16 → Q8_0", timer.elapsed_ms(), BENCHMARK_ITERATIONS * BLOCK_COUNT);
    }

    // Benchmark INT32 (requires dequant step)
    {
        const float int32_scale = 1.0f / 127.0f;
        alignas(64) float fp32_temp[Q8_0Block::BLOCK_SIZE];

        Timer timer;
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            for (size_t b = 0; b < BLOCK_COUNT; ++b)
            {
                // Dequantize INT32 → FP32
                const int32_t *block_src = int32_data.data() + b * Q8_0Block::BLOCK_SIZE;
                for (size_t i = 0; i < Q8_0Block::BLOCK_SIZE; ++i)
                {
                    fp32_temp[i] = static_cast<float>(block_src[i]) * int32_scale;
                }
                // Quantize FP32 → Q8_0
                simd::quantize_fp32_to_q8_0_block(fp32_temp, blocks[b], Q8_0Block::BLOCK_SIZE);
            }
        }
        print_performance("INT32 → Q8_0 (w/dequant)", timer.elapsed_ms(), BENCHMARK_ITERATIONS * BLOCK_COUNT);
    }

    // Reset to max threads
    omp_set_num_threads(omp_get_max_threads());
}

// =============================================================================
// Test 4: Large-Scale Throughput (Full Tensor Conversion)
// =============================================================================

TEST(Q8_0_ConversionPerf, LargeScale_TensorConversion)
{
    print_benchmark_header("Large-Scale Tensor Conversion: FP32 → Q8_0");

    constexpr size_t TOTAL_ELEMENTS = LARGE_SCALE_ELEMENTS; // 1M elements
    const size_t num_blocks = (TOTAL_ELEMENTS + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Total Elements: " << TOTAL_ELEMENTS << " (" << (TOTAL_ELEMENTS * sizeof(float)) / (1024 * 1024) << " MB)" << std::endl;
    std::cout << "  Block Count: " << num_blocks << std::endl;
    std::cout << "  OMP Threads: " << omp_get_max_threads() << std::endl;

    // Create FP32 tensor
    auto fp32_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{TOTAL_ELEMENTS});
    const float *fp32_data_const = fp32_tensor->data();
    float *fp32_data = const_cast<float *>(fp32_data_const);
    generate_test_data_fp32(fp32_data, TOTAL_ELEMENTS, 100.0f);

    std::vector<Q8_0Block> q8_0_blocks(num_blocks);

    // Warmup
    for (size_t i = 0; i < 5; ++i)
    {
        fp32_tensor->to_q8_0(q8_0_blocks.data());
    }

    // Benchmark full tensor conversion (multi-threaded via OpenMP in to_q8_0)
    constexpr size_t TENSOR_ITERATIONS = 100;
    Timer timer;
    for (size_t iter = 0; iter < TENSOR_ITERATIONS; ++iter)
    {
        fp32_tensor->to_q8_0(q8_0_blocks.data());
    }
    double time_ms = timer.elapsed_ms();

    std::cout << "\nResults:" << std::endl;
    print_performance("Full Tensor Conversion", time_ms, TENSOR_ITERATIONS);

    // Throughput metrics
    double avg_time_per_conversion = time_ms / TENSOR_ITERATIONS;
    double elements_per_sec = (TOTAL_ELEMENTS * TENSOR_ITERATIONS * 1000.0) / time_ms;
    double mb_per_sec = (TOTAL_ELEMENTS * sizeof(float) * TENSOR_ITERATIONS) / (time_ms * 1024.0);
    double blocks_per_sec = (num_blocks * TENSOR_ITERATIONS * 1000.0) / time_ms;

    std::cout << "\nThroughput:" << std::endl;
    std::cout << "  Time per conversion: " << std::fixed << std::setprecision(3) << avg_time_per_conversion << " ms" << std::endl;
    std::cout << "  Elements/sec: " << std::setprecision(0) << elements_per_sec << std::endl;
    std::cout << "  Blocks/sec: " << std::setprecision(0) << blocks_per_sec << std::endl;
    std::cout << "  Bandwidth: " << std::setprecision(2) << mb_per_sec << " MB/s" << std::endl;
}

// =============================================================================
// Test 5: Combined SIMD + OpenMP Speedup
// =============================================================================

TEST(Q8_0_ConversionPerf, Combined_SIMD_OpenMP_Speedup)
{
    print_benchmark_header("Combined SIMD + OpenMP Speedup Analysis");

    constexpr size_t BLOCK_COUNT = 10000;
    constexpr size_t TOTAL_ELEMENTS = BLOCK_COUNT * Q8_0Block::BLOCK_SIZE;

    std::vector<float> fp32_data(TOTAL_ELEMENTS);
    generate_test_data_fp32(fp32_data.data(), TOTAL_ELEMENTS, 10.0f);
    std::vector<Q8_0Block> blocks(BLOCK_COUNT);

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Block Count: " << BLOCK_COUNT << std::endl;
    std::cout << "  Iterations: " << BENCHMARK_ITERATIONS << std::endl;

#if defined(__AVX512F__)
    std::cout << "  SIMD: AVX512 (theoretical 16× vs scalar)" << std::endl;
#elif defined(__AVX2__)
    std::cout << "  SIMD: AVX2 (theoretical 8× vs scalar)" << std::endl;
#else
    std::cout << "  SIMD: Scalar baseline" << std::endl;
#endif

    std::cout << "  Max Threads: " << omp_get_max_threads() << std::endl;

    // Test 1: Single-threaded SIMD
    omp_set_num_threads(1);
    Timer timer1;
    for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
    {
        for (size_t b = 0; b < BLOCK_COUNT; ++b)
        {
            simd::quantize_fp32_to_q8_0_block(fp32_data.data() + b * Q8_0Block::BLOCK_SIZE,
                                              blocks[b],
                                              Q8_0Block::BLOCK_SIZE);
        }
    }
    double time_single = timer1.elapsed_ms();

    // Test 2: Multi-threaded SIMD
    omp_set_num_threads(omp_get_max_threads());
    Timer timer2;
    for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
    {
#pragma omp parallel for
        for (size_t b = 0; b < BLOCK_COUNT; ++b)
        {
            simd::quantize_fp32_to_q8_0_block(fp32_data.data() + b * Q8_0Block::BLOCK_SIZE,
                                              blocks[b],
                                              Q8_0Block::BLOCK_SIZE);
        }
    }
    double time_multi = timer2.elapsed_ms();

    std::cout << "\nResults:" << std::endl;
    print_performance("Single-threaded (SIMD)", time_single, BENCHMARK_ITERATIONS * BLOCK_COUNT);
    print_performance("Multi-threaded (SIMD + OpenMP)", time_multi, BENCHMARK_ITERATIONS * BLOCK_COUNT, time_single);

    double parallel_speedup = time_single / time_multi;
    double parallel_efficiency = (parallel_speedup / omp_get_max_threads()) * 100.0;

    std::cout << "\nScaling Analysis:" << std::endl;
    std::cout << "  Parallel Speedup: " << std::fixed << std::setprecision(2) << parallel_speedup << "×" << std::endl;
    std::cout << "  Parallel Efficiency: " << std::setprecision(1) << parallel_efficiency << "%" << std::endl;

#if defined(__AVX512F__)
    std::cout << "\n  Estimated total speedup vs scalar single-thread: ~" << std::setprecision(0) << (parallel_speedup * 12) << "× (SIMD 12-16× × OpenMP " << parallel_speedup << "×)" << std::endl;
#elif defined(__AVX2__)
    std::cout << "\n  Estimated total speedup vs scalar single-thread: ~" << std::setprecision(0) << (parallel_speedup * 6) << "× (SIMD 6-8× × OpenMP " << parallel_speedup << "×)" << std::endl;
#endif
}

// =============================================================================
// Test 6: Per-Block Micro-Benchmark
// =============================================================================

TEST(Q8_0_ConversionPerf, MicroBenchmark_PerBlock)
{
    print_benchmark_header("Micro-Benchmark: Single Block Performance");

    alignas(64) float fp32_block[Q8_0Block::BLOCK_SIZE];
    generate_test_data_fp32(fp32_block, Q8_0Block::BLOCK_SIZE, 10.0f);

    Q8_0Block q8_block;

    constexpr size_t MICRO_ITERATIONS = 10000000; // 10M iterations for stable timing

    // Warmup
    for (size_t i = 0; i < 1000; ++i)
    {
        simd::quantize_fp32_to_q8_0_block(fp32_block, q8_block, Q8_0Block::BLOCK_SIZE);
    }

    // Benchmark
    Timer timer;
    for (size_t iter = 0; iter < MICRO_ITERATIONS; ++iter)
    {
        simd::quantize_fp32_to_q8_0_block(fp32_block, q8_block, Q8_0Block::BLOCK_SIZE);
    }
    double time_ms = timer.elapsed_ms();

    double ns_per_block = (time_ms * 1e6) / MICRO_ITERATIONS;
    double blocks_per_sec = (MICRO_ITERATIONS * 1000.0) / time_ms;

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Iterations: " << MICRO_ITERATIONS << std::endl;
#if defined(__AVX512F__)
    std::cout << "  SIMD Level: AVX512" << std::endl;
#elif defined(__AVX2__)
    std::cout << "  SIMD Level: AVX2" << std::endl;
#else
    std::cout << "  SIMD Level: Scalar" << std::endl;
#endif

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Time per block: " << std::fixed << std::setprecision(2) << ns_per_block << " ns" << std::endl;
    std::cout << "  Blocks/sec: " << std::setprecision(0) << blocks_per_sec << std::endl;
    std::cout << "  Total time: " << std::setprecision(3) << time_ms << " ms" << std::endl;

    // Sanity check
    EXPECT_GT(q8_block.d, 0) << "Block scale should be non-zero";
}
