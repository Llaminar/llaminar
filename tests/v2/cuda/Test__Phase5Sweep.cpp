/**
 * @file Test__Phase5Sweep.cpp
 * @brief Auto-tuning sweep for Phase 5 GEMM configurations
 *
 * Benchmarks Phase 5 configurations to find optimal shared memory / occupancy
 * tradeoffs. Tests hypothesis that single buffering may outperform double
 * buffering due to better occupancy.
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#include "CudaGemmJITPhase5.h"
#include "Phase5ConfigSpace.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>

using namespace llaminar2::cuda;

class Phase5SweepTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check for CUDA device
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            GTEST_SKIP() << "No CUDA device available";
        }

        // Allocate test matrices (same size as Phase 4 benchmarks)
        M_ = 1024;
        N_ = 896;
        K_ = 896;

        // Allocate A (FP32 activation - Phase 5 uses FP32 input!)
        size_t a_size = M_ * K_ * sizeof(float);
        cudaMalloc(&d_A_, a_size);

        // Allocate B (IQ4_NL weights) - 4.5 bits per element
        // Each block is 32 elements = 18 bytes
        int blocks_per_row = (K_ + 31) / 32;
        size_t b_size = N_ * blocks_per_row * 18; // IQ4_NL block size
        cudaMalloc(&d_B_iq4nl_, b_size);

        // Allocate C (FP32 output)
        size_t c_size = M_ * N_ * sizeof(float);
        cudaMalloc(&d_C_, c_size);

        // Initialize activations in COLUMN-MAJOR format A[K][M] for coalesced loads
        std::vector<float> h_A(M_ * K_);
        for (int k = 0; k < K_; ++k)
        {
            for (int m = 0; m < M_; ++m)
            {
                h_A[k * M_ + m] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        cudaMemcpy(d_A_, h_A.data(), a_size, cudaMemcpyHostToDevice);

        // Initialize B with random IQ4_NL blocks
        std::vector<uint8_t> h_B(b_size);
        for (auto &val : h_B)
        {
            val = rand() % 256;
        }
        cudaMemcpy(d_B_iq4nl_, h_B.data(), b_size, cudaMemcpyHostToDevice);

        // Zero output
        cudaMemset(d_C_, 0, c_size);

        cudaDeviceSynchronize();
    }

    void TearDown() override
    {
        if (d_A_)
            cudaFree(d_A_);
        if (d_B_iq4nl_)
            cudaFree(d_B_iq4nl_);
        if (d_C_)
            cudaFree(d_C_);
    }

    struct BenchmarkResult
    {
        std::string config_id;
        Phase5GemmConfig config;
        float time_ms;
        float tflops;
        size_t smem_bytes;
        int estimated_occupancy;
        bool success;
        std::string error_msg;

        bool operator<(const BenchmarkResult &other) const
        {
            return tflops > other.tflops; // Sort descending
        }
    };

    CudaGemmConfigPhase5 toFullConfig(const Phase5GemmConfig &simple)
    {
        return CudaGemmConfigPhase5(
            simple.tile_m, simple.tile_n, simple.tile_k,
            simple.sub_k,
            simple.mma_m, simple.mma_n,
            simple.buffer_stages);
    }

    BenchmarkResult benchmarkConfig(const Phase5GemmConfig &config)
    {
        BenchmarkResult result;
        result.config = config;

        // Convert to full config for JIT
        CudaGemmConfigPhase5 full_config = toFullConfig(config);
        result.config_id = full_config.config_id();
        result.smem_bytes = full_config.shared_memory_bytes();
        result.estimated_occupancy = full_config.estimate_occupancy_blocks_per_sm();
        result.success = false;

        try
        {
            // JIT compile kernel
            auto &jit = CudaGemmJITPhase5::instance();
            CUfunction kernel = jit.getKernel(full_config);

            // Set up launch configuration
            int blocks_m = (M_ + config.tile_m - 1) / config.tile_m;
            int blocks_n = (N_ + config.tile_n - 1) / config.tile_n;
            dim3 grid(blocks_n, blocks_m);
            dim3 block(config.threads_per_block);

            // Warmup launches
            for (int i = 0; i < 5; ++i)
            {
                void *args[] = {&d_A_, &d_B_iq4nl_, &d_C_, &M_, &N_, &K_};
                cuLaunchKernel(kernel,
                               grid.x, grid.y, grid.z,
                               block.x, block.y, block.z,
                               full_config.shared_memory_bytes(), nullptr,
                               args, nullptr);
            }
            cudaDeviceSynchronize();

            // Timed launches - use CUDA events for accurate GPU timing
            constexpr int num_iters = 20;

            cudaEvent_t start_event, stop_event;
            cudaEventCreate(&start_event);
            cudaEventCreate(&stop_event);

            cudaEventRecord(start_event);
            for (int i = 0; i < num_iters; ++i)
            {
                void *args[] = {&d_A_, &d_B_iq4nl_, &d_C_, &M_, &N_, &K_};
                cuLaunchKernel(kernel,
                               grid.x, grid.y, grid.z,
                               block.x, block.y, block.z,
                               full_config.shared_memory_bytes(), nullptr,
                               args, nullptr);
            }
            cudaEventRecord(stop_event);
            cudaEventSynchronize(stop_event);

            float total_ms = 0.0f;
            cudaEventElapsedTime(&total_ms, start_event, stop_event);
            result.time_ms = total_ms / num_iters;

            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);

            // TFLOPS calculation
            double flops = 2.0 * M_ * N_ * K_; // MAC = multiply + add
            result.tflops = flops / (result.time_ms * 1e9);

            result.success = true;
        }
        catch (const std::exception &e)
        {
            result.error_msg = e.what();
        }

        return result;
    }

    void printResults(const std::vector<BenchmarkResult> &results)
    {
        std::cout << "\n========================================\n";
        std::cout << "Phase 5 Configuration Sweep Results\n";
        std::cout << "========================================\n\n";
        std::cout << "Matrix size: " << M_ << " x " << N_ << " x " << K_ << "\n";
        std::cout << "Phase 4 baseline: 8.75 TFLOPS\n\n";

        // Print header
        std::cout << std::left << std::setw(50) << "Config"
                  << std::right << std::setw(10) << "Time(ms)"
                  << std::setw(12) << "TFLOPS"
                  << std::setw(12) << "Speedup"
                  << std::setw(10) << "Smem(KB)"
                  << std::setw(10) << "Occ(blk)"
                  << "\n";
        std::cout << std::string(104, '-') << "\n";

        // Print results
        for (const auto &r : results)
        {
            if (!r.success)
            {
                std::cout << std::left << std::setw(50) << r.config_id
                          << " FAILED: " << r.error_msg << "\n";
                continue;
            }

            float speedup = r.tflops / 8.75f; // vs Phase 4 baseline
            std::cout << std::left << std::setw(50) << r.config_id
                      << std::right << std::setw(10) << std::fixed << std::setprecision(3) << r.time_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.tflops
                      << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
                      << std::setw(10) << (r.smem_bytes / 1024)
                      << std::setw(10) << r.estimated_occupancy
                      << "\n";
        }

        // Summary statistics
        auto successful = results;
        successful.erase(std::remove_if(successful.begin(), successful.end(),
                                        [](const BenchmarkResult &r)
                                        { return !r.success; }),
                         successful.end());

        if (!successful.empty())
        {
            std::cout << "\n========================================\n";
            std::cout << "Summary\n";
            std::cout << "========================================\n";
            std::cout << "Successful configs: " << successful.size() << "/" << results.size() << "\n";
            std::cout << "Best: " << successful[0].config_id << " @ "
                      << successful[0].tflops << " TFLOPS ("
                      << (successful[0].tflops / 8.75f) << "x vs Phase 4)\n";

            // Check if single buffering outperforms double
            auto best_single = std::find_if(successful.begin(), successful.end(),
                                            [](const BenchmarkResult &r)
                                            { return r.config.buffer_stages == 1; });
            auto best_double = std::find_if(successful.begin(), successful.end(),
                                            [](const BenchmarkResult &r)
                                            { return r.config.buffer_stages == 2; });

            if (best_single != successful.end() && best_double != successful.end())
            {
                std::cout << "\nBuffering comparison:\n";
                std::cout << "  Single buffer: " << best_single->tflops << " TFLOPS\n";
                std::cout << "  Double buffer: " << best_double->tflops << " TFLOPS\n";
                if (best_single->tflops > best_double->tflops)
                {
                    float improvement = (best_single->tflops / best_double->tflops - 1.0f) * 100.0f;
                    std::cout << "  → Single buffering wins by " << std::fixed << std::setprecision(1)
                              << improvement << "%! (Occupancy hypothesis VALIDATED)\n";
                }
            }
        }
    }

    void saveCSV(const std::vector<BenchmarkResult> &results, const std::string &filename)
    {
        std::ofstream csv(filename);
        csv << "config_id,tile_m,tile_n,tile_k,sub_k,mma_m,mma_n,buffer_stages,"
            << "threads_per_block,smem_bytes,estimated_occupancy,time_ms,tflops,success\n";

        for (const auto &r : results)
        {
            csv << r.config_id << ","
                << r.config.tile_m << "," << r.config.tile_n << "," << r.config.tile_k << ","
                << r.config.sub_k << ","
                << r.config.mma_m << "," << r.config.mma_n << ","
                << r.config.buffer_stages << ","
                << r.config.threads_per_block << ","
                << r.smem_bytes << ","
                << r.estimated_occupancy << ","
                << (r.success ? std::to_string(r.time_ms) : "N/A") << ","
                << (r.success ? std::to_string(r.tflops) : "N/A") << ","
                << (r.success ? "1" : "0") << "\n";
        }
    }

    int M_, N_, K_;
    void *d_A_ = nullptr;
    void *d_B_iq4nl_ = nullptr;
    void *d_C_ = nullptr;
};

TEST_F(Phase5SweepTest, FocusedSweep)
{
    std::cout << "\n=== Phase 5 Focused Configuration Sweep ===\n";
    std::cout << "Testing 11 hand-picked configurations to validate hypotheses:\n";
    std::cout << "  1. Single buffering may outperform double (better occupancy)\n";
    std::cout << "  2. Larger tiles may improve Tensor Core utilization\n";
    std::cout << "  3. Optimal SUB_K granularity\n\n";

    // Generate focused config space
    auto configs = generate_phase5_config_space_focused();

    std::cout << "Found " << configs.size() << " valid configurations.\n";
    std::cout << "Benchmarking on " << M_ << "x" << N_ << "x" << K_ << " matrix...\n\n";

    // Benchmark all configs
    std::vector<BenchmarkResult> results;
    for (const auto &config : configs)
    {
        std::cout << "Testing: " << config.config_id() << " ... " << std::flush;
        auto result = benchmarkConfig(config);
        results.push_back(result);

        if (result.success)
        {
            std::cout << result.tflops << " TFLOPS\n";
        }
        else
        {
            std::cout << "FAILED: " << result.error_msg << "\n";
        }
    }

    // Sort by performance
    std::sort(results.begin(), results.end());

    // Print results
    printResults(results);

    // Save CSV
    saveCSV(results, "phase5_focused_sweep_results.csv");
    std::cout << "\nResults saved to: phase5_focused_sweep_results.csv\n";

    // Assertions
    auto successful = results;
    successful.erase(std::remove_if(successful.begin(), successful.end(),
                                    [](const BenchmarkResult &r)
                                    { return !r.success; }),
                     successful.end());

    ASSERT_FALSE(successful.empty()) << "All configurations failed!";

    // Expect at least one config to match or beat Phase 4
    float best_tflops = successful[0].tflops;
    std::cout << "\n=== Final Verdict ===\n";
    if (best_tflops >= 8.75f)
    {
        std::cout << "✓ SUCCESS: Best config achieves " << best_tflops
                  << " TFLOPS (≥8.75 Phase 4 baseline)\n";
    }
    else
    {
        std::cout << "⚠ Best config achieves " << best_tflops
                  << " TFLOPS (below 8.75 Phase 4 baseline)\n";
        std::cout << "  This indicates configuration space may need expansion.\n";
    }

    // Minimum success: At least 1 valid config
    // Target success: At least 1 config ≥ Phase 4 (8.75 TFLOPS)
    // Stretch success: At least 1 config ≥ 11.4 TFLOPS (+30%)
    EXPECT_GE(best_tflops, 8.75f) << "Expected at least one config to match Phase 4";

    if (best_tflops >= 11.4f)
    {
        std::cout << "🎉 STRETCH GOAL ACHIEVED: +30% improvement over Phase 4!\n";
    }
}

TEST_F(Phase5SweepTest, DISABLED_FullSweep)
{
    // This test is disabled by default (takes ~8-12 hours)
    // Run with: --gtest_also_run_disabled_tests

    std::cout << "\n=== Phase 5 Full Configuration Sweep ===\n";
    std::cout << "Generating full configuration space...\n";

    auto configs = generate_phase5_config_space();
    std::cout << "Found " << configs.size() << " valid configurations.\n";
    std::cout << "This will take approximately " << (configs.size() * 1.5) / 60.0
              << " hours to complete.\n\n";

    std::vector<BenchmarkResult> results;
    int count = 0;
    for (const auto &config : configs)
    {
        count++;
        if (count % 50 == 0)
        {
            std::cout << "Progress: " << count << "/" << configs.size() << "\n";
        }

        auto result = benchmarkConfig(config);
        results.push_back(result);
    }

    std::sort(results.begin(), results.end());
    printResults(results);
    saveCSV(results, "phase5_full_sweep_results.csv");
}
