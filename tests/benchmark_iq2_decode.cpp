/**
 * @file benchmark_iq2_decode.cpp
 * @brief Performance benchmarks for IQ2 family SIMD optimizations
 * @author David Sanftenberg
 * 
 * Measures decode throughput for IQ2_XXS, IQ2_XS, IQ2_S formats with:
 * - Single-threaded performance (AVX2 vs scalar)
 * - Multi-threaded scaling (OMP thread count sweep)
 * - Comparison against llama.cpp baseline (future)
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <omp.h>
#include "../src/tensors/IQ2_XXSTensor.h"
#include "../src/tensors/IQ2_XSTensor.h"
#include "../src/tensors/IQ2_STensor.h"

namespace {

/**
 * @brief Benchmark parameters
 */
constexpr size_t WARMUP_ITERATIONS = 10;
constexpr size_t BENCH_ITERATIONS = 100;
constexpr size_t TENSOR_ROWS = 256;     // ~4K elements per row for IQ2_XXS
constexpr size_t TENSOR_COLS = 4096;    // Total: 1M elements

/**
 * @brief Timer utility
 */
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

/**
 * @brief Benchmark IQ2_XXS decode throughput
 */
TEST(IQ2_DecodeBenchmark, IQ2_XXS_SingleThread) {
    // Disable OMP for single-thread test
    omp_set_num_threads(1);
    
    // Create tensor (random data is fine for throughput test)
    std::vector<int> shape = {static_cast<int>(TENSOR_ROWS), static_cast<int>(TENSOR_COLS)};
    size_t block_size = 66;  // IQ2_XXS block size
    size_t num_blocks = (TENSOR_ROWS * TENSOR_COLS + 255) / 256;
    std::vector<uint8_t> data(num_blocks * block_size);
    
    // Initialize with random data
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(rand() % 256);
    }
    
    auto tensor = std::make_shared<llaminar::IQ2_XXSTensor>(shape, data);
    std::vector<float> output(TENSOR_ROWS * TENSOR_COLS);
    
    // Warmup
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
        tensor->decode_to_fp32(output.data());
    }
    
    // Benchmark
    Timer timer;
    timer.start();
    for (size_t i = 0; i < BENCH_ITERATIONS; ++i) {
        tensor->decode_to_fp32(output.data());
    }
    double elapsed_ms = timer.elapsed_ms();
    
    // Calculate throughput
    size_t total_elements = TENSOR_ROWS * TENSOR_COLS * BENCH_ITERATIONS;
    double throughput_melem_per_sec = (total_elements / 1e6) / (elapsed_ms / 1e3);
    double avg_time_us = (elapsed_ms * 1000.0) / BENCH_ITERATIONS;
    
    std::cout << "\n=== IQ2_XXS Single-Thread Performance ===\n";
    std::cout << "Tensor size: " << TENSOR_ROWS << " × " << TENSOR_COLS << " = " 
              << (TENSOR_ROWS * TENSOR_COLS / 1e6) << " M elements\n";
    std::cout << "Iterations: " << BENCH_ITERATIONS << "\n";
    std::cout << "Total time: " << elapsed_ms << " ms\n";
    std::cout << "Avg decode time: " << avg_time_us << " µs\n";
    std::cout << "Throughput: " << throughput_melem_per_sec << " Melem/s\n";
    std::cout << "==========================================\n\n";
    
    // Sanity check: throughput should be reasonable (>10 Melem/s on modern CPU)
    EXPECT_GT(throughput_melem_per_sec, 10.0);
}

/**
 * @brief Benchmark IQ2_XS decode throughput
 */
TEST(IQ2_DecodeBenchmark, IQ2_XS_SingleThread) {
    omp_set_num_threads(1);
    
    std::vector<int> shape = {static_cast<int>(TENSOR_ROWS), static_cast<int>(TENSOR_COLS)};
    size_t block_size = 74;  // IQ2_XS block size
    size_t num_blocks = (TENSOR_ROWS * TENSOR_COLS + 255) / 256;
    std::vector<uint8_t> data(num_blocks * block_size);
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(rand() % 256);
    }
    
    auto tensor = std::make_shared<llaminar::IQ2_XSTensor>(shape, data);
    std::vector<float> output(TENSOR_ROWS * TENSOR_COLS);
    
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
        tensor->decode_to_fp32(output.data());
    }
    
    Timer timer;
    timer.start();
    for (size_t i = 0; i < BENCH_ITERATIONS; ++i) {
        tensor->decode_to_fp32(output.data());
    }
    double elapsed_ms = timer.elapsed_ms();
    
    size_t total_elements = TENSOR_ROWS * TENSOR_COLS * BENCH_ITERATIONS;
    double throughput_melem_per_sec = (total_elements / 1e6) / (elapsed_ms / 1e3);
    double avg_time_us = (elapsed_ms * 1000.0) / BENCH_ITERATIONS;
    
    std::cout << "\n=== IQ2_XS Single-Thread Performance ===\n";
    std::cout << "Tensor size: " << TENSOR_ROWS << " × " << TENSOR_COLS << " = " 
              << (TENSOR_ROWS * TENSOR_COLS / 1e6) << " M elements\n";
    std::cout << "Iterations: " << BENCH_ITERATIONS << "\n";
    std::cout << "Total time: " << elapsed_ms << " ms\n";
    std::cout << "Avg decode time: " << avg_time_us << " µs\n";
    std::cout << "Throughput: " << throughput_melem_per_sec << " Melem/s\n";
    std::cout << "=========================================\n\n";
    
    EXPECT_GT(throughput_melem_per_sec, 10.0);
}

/**
 * @brief Benchmark IQ2_S decode throughput
 */
TEST(IQ2_DecodeBenchmark, IQ2_S_SingleThread) {
    omp_set_num_threads(1);
    
    std::vector<int> shape = {static_cast<int>(TENSOR_ROWS), static_cast<int>(TENSOR_COLS)};
    size_t block_size = 82;  // IQ2_S block size
    size_t num_blocks = (TENSOR_ROWS * TENSOR_COLS + 255) / 256;
    std::vector<uint8_t> data(num_blocks * block_size);
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(rand() % 256);
    }
    
    auto tensor = std::make_shared<llaminar::IQ2_STensor>(shape, data);
    std::vector<float> output(TENSOR_ROWS * TENSOR_COLS);
    
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
        tensor->decode_to_fp32(output.data());
    }
    
    Timer timer;
    timer.start();
    for (size_t i = 0; i < BENCH_ITERATIONS; ++i) {
        tensor->decode_to_fp32(output.data());
    }
    double elapsed_ms = timer.elapsed_ms();
    
    size_t total_elements = TENSOR_ROWS * TENSOR_COLS * BENCH_ITERATIONS;
    double throughput_melem_per_sec = (total_elements / 1e6) / (elapsed_ms / 1e3);
    double avg_time_us = (elapsed_ms * 1000.0) / BENCH_ITERATIONS;
    
    std::cout << "\n=== IQ2_S Single-Thread Performance ===\n";
    std::cout << "Tensor size: " << TENSOR_ROWS << " × " << TENSOR_COLS << " = " 
              << (TENSOR_ROWS * TENSOR_COLS / 1e6) << " M elements\n";
    std::cout << "Iterations: " << BENCH_ITERATIONS << "\n";
    std::cout << "Total time: " << elapsed_ms << " ms\n";
    std::cout << "Avg decode time: " << avg_time_us << " µs\n";
    std::cout << "Throughput: " << throughput_melem_per_sec << " Melem/s\n";
    std::cout << "========================================\n\n";
    
    EXPECT_GT(throughput_melem_per_sec, 10.0);
}

/**
 * @brief Benchmark multi-threaded scaling
 */
TEST(IQ2_DecodeBenchmark, IQ2_XXS_MultiThread) {
    std::vector<int> shape = {static_cast<int>(TENSOR_ROWS), static_cast<int>(TENSOR_COLS)};
    size_t block_size = 66;
    size_t num_blocks = (TENSOR_ROWS * TENSOR_COLS + 255) / 256;
    std::vector<uint8_t> data(num_blocks * block_size);
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(rand() % 256);
    }
    
    auto tensor = std::make_shared<llaminar::IQ2_XXSTensor>(shape, data);
    std::vector<float> output(TENSOR_ROWS * TENSOR_COLS);
    
    std::cout << "\n=== IQ2_XXS Multi-Thread Scaling ===\n";
    std::cout << "Tensor size: " << TENSOR_ROWS << " × " << TENSOR_COLS << " = " 
              << (TENSOR_ROWS * TENSOR_COLS / 1e6) << " M elements\n\n";
    
    std::vector<int> thread_counts = {1, 2, 4, 8};
    double baseline_throughput = 0.0;
    
    for (int num_threads : thread_counts) {
        omp_set_num_threads(num_threads);
        
        // Warmup
        for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
            tensor->decode_to_fp32(output.data());
        }
        
        // Benchmark
        Timer timer;
        timer.start();
        for (size_t i = 0; i < BENCH_ITERATIONS; ++i) {
            tensor->decode_to_fp32(output.data());
        }
        double elapsed_ms = timer.elapsed_ms();
        
        size_t total_elements = TENSOR_ROWS * TENSOR_COLS * BENCH_ITERATIONS;
        double throughput = (total_elements / 1e6) / (elapsed_ms / 1e3);
        
        if (num_threads == 1) {
            baseline_throughput = throughput;
        }
        double speedup = throughput / baseline_throughput;
        
        std::cout << num_threads << " threads: " 
                  << throughput << " Melem/s (speedup: " << speedup << "×)\n";
    }
    std::cout << "====================================\n\n";
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
