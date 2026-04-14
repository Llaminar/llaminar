/**
 * @file Perf__CUDAVectorAddKernels.cpp
 * @brief Performance benchmarks for CUDA vector addition kernels used in HOST backend reductions
 *
 * Measures throughput (GB/s) for in-place vector addition at realistic reduction sizes:
 *   - Small (896 elements): hidden_size after attention output projection
 *   - Medium (4,864 elements): intermediate_size after FFN gate/up
 *   - Large (57,344 elements): LM head output (vocab_size=57,344 for Qwen2)
 *   - Bulk: 1M-16M elements for stress testing
 *
 * Tests both scalar and vectorized kernel variants.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

// Include the kernels header
#include "kernels/cuda/ops/CUDAVectorAddKernels.h"

using namespace llaminar2;
using namespace llaminar2::cuda;

namespace
{

    // ============================================================================
    // Performance Test Configuration
    // ============================================================================

    constexpr int WARMUP_ITERATIONS = 50;
    constexpr int BENCHMARK_ITERATIONS = 1000;

    // Realistic sizes from Qwen2.5-0.5B
    constexpr size_t SIZE_HIDDEN = 896;           // hidden_size
    constexpr size_t SIZE_INTERMEDIATE = 4864;    // intermediate_size
    constexpr size_t SIZE_VOCAB = 57344;          // vocab_size (151936 for full, 57344 common)
    constexpr size_t SIZE_1M = 1024 * 1024;       // 1M elements
    constexpr size_t SIZE_4M = 4 * 1024 * 1024;   // 4M elements
    constexpr size_t SIZE_16M = 16 * 1024 * 1024; // 16M elements

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class CUDAVectorAddPerf : public ::testing::Test
    {
    protected:
        cudaStream_t stream_ = nullptr;
        int device_ = -1;
        cudaDeviceProp props_;

        void SetUp() override
        {
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_count == 0)
            {
                GTEST_SKIP() << "No CUDA devices available";
            }

            // Select first device
            device_ = 0;
            err = cudaSetDevice(device_);
            if (err != cudaSuccess)
            {
                GTEST_SKIP() << "Failed to set CUDA device: " << cudaGetErrorString(err);
            }

            err = cudaGetDeviceProperties(&props_, device_);
            if (err != cudaSuccess)
            {
                GTEST_SKIP() << "Failed to get device properties: " << cudaGetErrorString(err);
            }

            err = cudaStreamCreate(&stream_);
            if (err != cudaSuccess)
            {
                GTEST_SKIP() << "Failed to create CUDA stream: " << cudaGetErrorString(err);
            }

            // Print device info
            std::cout << "\n┌─────────────────────────────────────────────────────────────────┐" << std::endl;
            std::cout << "│              CUDA VECTOR ADD KERNEL PERFORMANCE                 │" << std::endl;
            std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
            std::cout << "│ Device: " << std::setw(54) << std::left << props_.name << "│" << std::endl;
            std::cout << "│ Memory Bus Width: " << std::setw(41) << std::right
                      << props_.memoryBusWidth << " bits │" << std::endl;
            std::cout << "│ Warmup: " << std::setw(8) << WARMUP_ITERATIONS
                      << " iterations, Benchmark: " << std::setw(8) << BENCHMARK_ITERATIONS << " iterations │" << std::endl;
            std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;
        }

        void TearDown() override
        {
            if (stream_)
            {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }
        }

        /**
         * @brief Allocate device memory with random FP32 data
         */
        float *allocateDeviceBuffer(size_t count, std::mt19937 &rng)
        {
            std::vector<float> host_data(count);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &v : host_data)
            {
                v = dist(rng);
            }

            float *d_ptr = nullptr;
            cudaMalloc(&d_ptr, count * sizeof(float));
            cudaMemcpy(d_ptr, host_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);
            return d_ptr;
        }

        /**
         * @brief Benchmark in-place addition with timing
         */
        struct BenchResult
        {
            double avg_time_us;
            double throughput_gbps;
            double ops_per_sec;
        };

        BenchResult benchmarkInplaceAdd(float *d_output, const float *d_input, size_t count)
        {
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                launchVectorAddInplace_f32(d_output, d_input, count, stream_);
            }
            cudaStreamSynchronize(stream_);

            // Benchmark
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, stream_);
            for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
            {
                launchVectorAddInplace_f32(d_output, d_input, count, stream_);
            }
            cudaEventRecord(stop, stream_);
            cudaStreamSynchronize(stream_);

            float elapsed_ms = 0;
            cudaEventElapsedTime(&elapsed_ms, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            BenchResult result;
            result.avg_time_us = (elapsed_ms * 1000.0) / BENCHMARK_ITERATIONS;

            // Throughput: read input (count * 4 bytes) + read/write output (count * 4 bytes * 2)
            // Total: 3 * count * 4 bytes per operation
            double bytes_per_op = 3.0 * count * sizeof(float);
            double total_bytes = bytes_per_op * BENCHMARK_ITERATIONS;
            result.throughput_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

            // Operations per second
            result.ops_per_sec = BENCHMARK_ITERATIONS / (elapsed_ms / 1000.0);

            return result;
        }

        /**
         * @brief Print result row
         */
        void printResult(const std::string &name, size_t count, const BenchResult &result)
        {
            std::cout << "│ " << std::setw(20) << std::left << name
                      << " │ " << std::setw(12) << std::right << count
                      << " │ " << std::setw(10) << std::fixed << std::setprecision(2) << result.avg_time_us
                      << " │ " << std::setw(10) << std::fixed << std::setprecision(2) << result.throughput_gbps
                      << " │ " << std::setw(12) << std::scientific << std::setprecision(2) << result.ops_per_sec
                      << " │" << std::endl;
        }

        void printTableHeader()
        {
            std::cout << "\n┌──────────────────────┬──────────────┬────────────┬────────────┬──────────────┐" << std::endl;
            std::cout << "│ Kernel / Size        │    Elements  │  Time (μs) │   GB/s     │    Ops/sec   │" << std::endl;
            std::cout << "├──────────────────────┼──────────────┼────────────┼────────────┼──────────────┤" << std::endl;
        }

        void printTableFooter()
        {
            std::cout << "└──────────────────────┴──────────────┴────────────┴────────────┴──────────────┘" << std::endl;
        }
    };

    // ============================================================================
    // FP32 In-place Addition Benchmarks (used by HOST backend reductions)
    // ============================================================================

    TEST_F(CUDAVectorAddPerf, FP32_InplaceAdd_RealisticSizes)
    {
        std::mt19937 rng(42);

        // Test sizes that match real inference workloads
        std::vector<std::pair<std::string, size_t>> sizes = {
            {"hidden (896)", SIZE_HIDDEN},
            {"intermediate (4864)", SIZE_INTERMEDIATE},
            {"vocab (57344)", SIZE_VOCAB},
            {"batch_hidden (8*896)", 8 * SIZE_HIDDEN}, // Batch of 8 tokens
            {"batch_interm (8*4864)", 8 * SIZE_INTERMEDIATE},
        };

        printTableHeader();

        for (const auto &[name, count] : sizes)
        {
            float *d_output = allocateDeviceBuffer(count, rng);
            float *d_input = allocateDeviceBuffer(count, rng);

            auto result = benchmarkInplaceAdd(d_output, d_input, count);
            printResult(name, count, result);

            cudaFree(d_output);
            cudaFree(d_input);
        }

        printTableFooter();
    }

    TEST_F(CUDAVectorAddPerf, FP32_InplaceAdd_BulkSizes)
    {
        std::mt19937 rng(42);

        // Stress test with large buffers
        std::vector<std::pair<std::string, size_t>> sizes = {
            {"1M elements", SIZE_1M},
            {"4M elements", SIZE_4M},
            {"16M elements", SIZE_16M},
        };

        printTableHeader();

        for (const auto &[name, count] : sizes)
        {
            float *d_output = allocateDeviceBuffer(count, rng);
            float *d_input = allocateDeviceBuffer(count, rng);

            auto result = benchmarkInplaceAdd(d_output, d_input, count);
            printResult(name, count, result);

            cudaFree(d_output);
            cudaFree(d_input);
        }

        printTableFooter();

        // Print memory bus info (clock rate not available in all CUDA headers)
        std::cout << "\n[Info] Memory bus width: " << props_.memoryBusWidth << " bits" << std::endl;
    }

    TEST_F(CUDAVectorAddPerf, FP32_InplaceAdd_ScalarVsVectorized)
    {
        std::mt19937 rng(42);

        // Test sizes that force scalar path vs vectorized path
        // Scalar: count < 1024 OR not divisible by 4
        // Vectorized: count >= 1024 AND divisible by 4

        std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│                    SCALAR vs VECTORIZED KERNEL COMPARISON                           │" << std::endl;
        std::cout << "│ Scalar path:     count < 1024 OR count % 4 != 0                                     │" << std::endl;
        std::cout << "│ Vectorized path: count >= 1024 AND count % 4 == 0                                   │" << std::endl;
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────────┘" << std::endl;

        std::vector<std::tuple<std::string, size_t, std::string>> sizes = {
            {"512 (scalar)", 512, "scalar"},
            {"1024 (vectorized)", 1024, "vectorized"},
            {"1023 (scalar)", 1023, "scalar"},
            {"4096 (vectorized)", 4096, "vectorized"},
            {"4097 (scalar)", 4097, "scalar"},
        };

        printTableHeader();

        for (const auto &[name, count, kernel_type] : sizes)
        {
            float *d_output = allocateDeviceBuffer(count, rng);
            float *d_input = allocateDeviceBuffer(count, rng);

            auto result = benchmarkInplaceAdd(d_output, d_input, count);
            printResult(name + " [" + kernel_type + "]", count, result);

            cudaFree(d_output);
            cudaFree(d_input);
        }

        printTableFooter();
    }

    TEST_F(CUDAVectorAddPerf, FP32_InplaceAdd_PCIeBAR_SimulatedReduction)
    {
        std::mt19937 rng(42);

        // Simulate a typical HOST backend allreduce pattern:
        // - One GPU writes to host staging buffer
        // - Other GPU reads and adds to local buffer
        // For 2-way TP, we do 1 reduction per collective

        // Typical allreduce sizes for Qwen2.5-0.5B:
        // - Attention output: 896 * seq_len
        // - FFN down: 896 * seq_len
        // - LM head: 151936 (vocab)

        constexpr size_t SEQ_LEN = 1; // Decode phase
        std::vector<std::pair<std::string, size_t>> allreduce_sizes = {
            {"attn_out (decode)", SIZE_HIDDEN * SEQ_LEN},
            {"ffn_down (decode)", SIZE_HIDDEN * SEQ_LEN},
            {"lm_head (decode)", 151936}, // Full vocab
        };

        std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│                    HOST BACKEND ALLREDUCE REDUCTION SIMULATION                      │" << std::endl;
        std::cout << "│ Simulates the CUDA-side reduction kernel during 2-way TP allreduce                  │" << std::endl;
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────────┘" << std::endl;

        printTableHeader();

        for (const auto &[name, count] : allreduce_sizes)
        {
            float *d_output = allocateDeviceBuffer(count, rng);
            float *d_input = allocateDeviceBuffer(count, rng);

            auto result = benchmarkInplaceAdd(d_output, d_input, count);
            printResult(name, count, result);

            cudaFree(d_output);
            cudaFree(d_input);
        }

        printTableFooter();
    }

    // ============================================================================
    // Latency-focused test (important for small reductions)
    // ============================================================================

    TEST_F(CUDAVectorAddPerf, FP32_InplaceAdd_KernelLaunchOverhead)
    {
        std::mt19937 rng(42);

        // Very small sizes to measure kernel launch overhead
        constexpr size_t TINY_SIZE = 64;
        constexpr int LATENCY_ITERATIONS = 10000;

        float *d_output = allocateDeviceBuffer(TINY_SIZE, rng);
        float *d_input = allocateDeviceBuffer(TINY_SIZE, rng);

        // Warmup
        for (int i = 0; i < 100; ++i)
        {
            launchVectorAddInplace_f32(d_output, d_input, TINY_SIZE, stream_);
        }
        cudaStreamSynchronize(stream_);

        // Measure single-kernel latency
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream_);
        for (int i = 0; i < LATENCY_ITERATIONS; ++i)
        {
            launchVectorAddInplace_f32(d_output, d_input, TINY_SIZE, stream_);
        }
        cudaEventRecord(stop, stream_);
        cudaStreamSynchronize(stream_);

        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        double avg_latency_us = (elapsed_ms * 1000.0) / LATENCY_ITERATIONS;

        std::cout << "\n┌─────────────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│                    KERNEL LAUNCH OVERHEAD                        │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│ Size: " << std::setw(8) << TINY_SIZE << " elements (scalar path)                      │" << std::endl;
        std::cout << "│ Iterations: " << std::setw(8) << LATENCY_ITERATIONS << "                                        │" << std::endl;
        std::cout << "│ Average latency: " << std::setw(8) << std::fixed << std::setprecision(3) << avg_latency_us << " μs                               │" << std::endl;
        std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_output);
        cudaFree(d_input);

        // Assert reasonable launch overhead (should be < 10us)
        EXPECT_LT(avg_latency_us, 10.0) << "Kernel launch overhead seems too high";
    }

} // namespace

#else
// No CUDA - skip all tests

TEST(CUDAVectorAddPerf, DISABLED_NoCUDA)
{
    GTEST_SKIP() << "CUDA not available";
}

#endif // HAVE_CUDA
