/**
 * @file Perf__Q8_1KernelSweep.cpp
 * @brief Kernel parameter sweep for Q8_1 GEMM (MR × NR tuning)
 * @author David Sanftenberg
 *
 * Sweeps MR (M register blocking) and NR (N register blocking) from 8 to 128
 * to find optimal microkernel configuration for the target architecture.
 *
 * Key Parameters:
 * - MR: M register blocking (rows per microkernel)
 * - NR: N register blocking (columns per microkernel)
 *
 * Sweep Configuration:
 * - MR values: {8, 16, 24, 32, 48, 64, 96, 128}
 * - NR values: {8, 16, 24, 32, 48, 64, 96, 128}
 * - Tests all combinations (64 total configurations)
 * - Includes asymmetric configurations (MR ≠ NR)
 *
 * Expected Results:
 * - Baseline: MR=32, NR=64 → ~470 GFLOPS
 * - Optimal: TBD (depends on CPU cache/register characteristics)
 * - Trends: Larger MR/NR reduce loop overhead but increase register pressure
 */

#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>

#include "kernels/cpu/gemm_v2/Q8_1GemmKernel.h"
#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"

using namespace llaminar2;

/**
 * @struct KernelConfig
 * @brief Configuration for a single kernel parameter sweep point
 */
struct KernelConfig
{
    int MR;        // M register blocking
    int NR;        // N register blocking
    double gflops; // Measured throughput
    double avg_ms; // Average time per iteration
    bool valid;    // Whether configuration compiled/ran successfully
};

class Q8_1KernelSweep : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Load Q8_0 model for weights
        model_path_ = "models/qwen2.5-0.5b-instruct-q8_0.gguf";
        loader_ = std::make_unique<ModelLoader>();

        if (!loader_->loadModel(model_path_))
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }

        // Setup test matrices (same as LargeBatchedPrefill)
        setupTestMatrices();
    }

    void setupTestMatrices()
    {
        // Load Q8_0 weight tensor
        auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
        ASSERT_NE(wq_template, nullptr);
        ASSERT_EQ(wq_template->native_type(), TensorType::Q8_0);

        q8_0_template_ = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);
        ASSERT_NE(q8_0_template_, nullptr);

        // Matrix dimensions (same as performance test)
        M_ = 4096; // Large batch
        N_ = 896;  // d_model
        K_ = 896;  // input features

        ASSERT_EQ(K_ % 32, 0) << "K must be multiple of block size";

        // Create Q8_1 activation tensor (A matrix)
        const size_t template_rows = q8_0_template_->shape()[0];
        const size_t rows_per_tile = template_rows;
        const size_t num_tiles = (M_ + rows_per_tile - 1) / rows_per_tile;

        const void *q8_0_template_data = q8_0_template_->get_raw_block_at(0, 0);
        const size_t K_blocks = K_ / 32;
        const size_t blocks_per_row = K_blocks;

        std::vector<uint8_t> A_q8_0_data(M_ * blocks_per_row * sizeof(Q8_0Block));

        // Tile the template
        for (size_t tile = 0; tile < num_tiles; ++tile)
        {
            const size_t dst_row_start = tile * rows_per_tile;
            const size_t rows_to_copy = std::min(rows_per_tile, M_ - dst_row_start);
            const size_t bytes_to_copy = rows_to_copy * blocks_per_row * sizeof(Q8_0Block);

            std::memcpy(A_q8_0_data.data() + dst_row_start * blocks_per_row * sizeof(Q8_0Block),
                        q8_0_template_data,
                        bytes_to_copy);
        }

        // Create Q8_0 → FP32 → Q8_1
        auto q8_0_A = std::make_unique<Q8_0Tensor>(std::vector<size_t>{M_, K_}, A_q8_0_data);
        std::vector<float> A_fp32(M_ * K_);

        for (size_t i = 0; i < M_; ++i)
        {
            for (size_t kb = 0; kb < K_blocks; ++kb)
            {
                const Q8_0Block *block = reinterpret_cast<const Q8_0Block *>(
                    q8_0_A->get_raw_block_at(i, kb));
                float scale = fp16_to_fp32(block->d);

                for (size_t k_in = 0; k_in < 32; ++k_in)
                {
                    A_fp32[i * K_ + kb * 32 + k_in] = static_cast<float>(block->qs[k_in]) * scale;
                }
            }
        }

        q8_1_A_ = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M_, K_});
        ASSERT_NE(q8_1_A_, nullptr);

        // Output buffer
        C_.resize(M_ * N_, 0.0f);
    }

    /**
     * @brief Benchmark a specific kernel configuration
     */
    template <int MR, int NR>
    KernelConfig benchmarkConfig()
    {
        KernelConfig config;
        config.MR = MR;
        config.NR = NR;
        config.valid = true;

        try
        {
            // Use specialized kernel for this configuration
            using KernelType = Q8_1GemmKernelTemplate<MR, NR>;

            // Warmup
            constexpr int WARMUP = 5;
            for (int i = 0; i < WARMUP; ++i)
            {
                std::fill(C_.begin(), C_.end(), 0.0f);
                KernelType::gemm(M_, N_, K_, *q8_1_A_, *q8_0_template_, C_.data(), N_);
            }

            // Timed iterations
            constexpr int ITERATIONS = 20;
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < ITERATIONS; ++i)
            {
                std::fill(C_.begin(), C_.end(), 0.0f);
                KernelType::gemm(M_, N_, K_, *q8_1_A_, *q8_0_template_, C_.data(), N_);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            config.avg_ms = total_ms / ITERATIONS;

            // Calculate GFLOPS
            double flops = 2.0 * M_ * N_ * K_;
            config.gflops = (flops / 1e9) / (config.avg_ms / 1000.0);
        }
        catch (const std::exception &e)
        {
            std::cerr << "  MR=" << MR << " NR=" << NR << " FAILED: " << e.what() << std::endl;
            config.valid = false;
            config.gflops = 0.0;
            config.avg_ms = 0.0;
        }

        return config;
    }

    std::string model_path_;
    std::unique_ptr<ModelLoader> loader_;
    std::shared_ptr<Q8_0Tensor> q8_0_template_;
    std::shared_ptr<Q8_1Tensor> q8_1_A_; // Changed to shared_ptr to match quantize_from_fp32 return type
    std::vector<float> C_;
    int M_, N_, K_;
};

/**
 * @brief Comprehensive kernel parameter sweep
 *
 * Tests all combinations of MR and NR from {8, 16, 24, 32, 48, 64, 96, 128}
 * Total: 64 configurations (8 MR values × 8 NR values)
 */
TEST_F(Q8_1KernelSweep, ComprehensiveSweep)
{
    std::cout << "\n=== Q8_1 GEMM Kernel Parameter Sweep ===" << std::endl;
    std::cout << "Matrix: M=" << M_ << " N=" << N_ << " K=" << K_ << std::endl;
    std::cout << "Sweeping MR × NR configurations from 8 to 128" << std::endl;
    std::cout << "Baseline: MR=32 NR=64 → ~470 GFLOPS" << std::endl;
    std::cout << "\nTesting 64 configurations..." << std::endl;

    // Sweep values
    std::vector<int> sweep_values = {8, 16, 24, 32, 48, 64, 96, 128};
    std::vector<KernelConfig> results;

    // Run sweep
    for (int MR : sweep_values)
    {
        for (int NR : sweep_values)
        {
            KernelConfig config;

            // Dispatch to templated benchmark based on MR/NR
            if (MR == 8 && NR == 8)
                config = benchmarkConfig<8, 8>();
            else if (MR == 8 && NR == 16)
                config = benchmarkConfig<8, 16>();
            else if (MR == 8 && NR == 24)
                config = benchmarkConfig<8, 24>();
            else if (MR == 8 && NR == 32)
                config = benchmarkConfig<8, 32>();
            else if (MR == 8 && NR == 48)
                config = benchmarkConfig<8, 48>();
            else if (MR == 8 && NR == 64)
                config = benchmarkConfig<8, 64>();
            else if (MR == 8 && NR == 96)
                config = benchmarkConfig<8, 96>();
            else if (MR == 8 && NR == 128)
                config = benchmarkConfig<8, 128>();
            else if (MR == 16 && NR == 8)
                config = benchmarkConfig<16, 8>();
            else if (MR == 16 && NR == 16)
                config = benchmarkConfig<16, 16>();
            else if (MR == 16 && NR == 24)
                config = benchmarkConfig<16, 24>();
            else if (MR == 16 && NR == 32)
                config = benchmarkConfig<16, 32>();
            else if (MR == 16 && NR == 48)
                config = benchmarkConfig<16, 48>();
            else if (MR == 16 && NR == 64)
                config = benchmarkConfig<16, 64>();
            else if (MR == 16 && NR == 96)
                config = benchmarkConfig<16, 96>();
            else if (MR == 16 && NR == 128)
                config = benchmarkConfig<16, 128>();
            else if (MR == 24 && NR == 8)
                config = benchmarkConfig<24, 8>();
            else if (MR == 24 && NR == 16)
                config = benchmarkConfig<24, 16>();
            else if (MR == 24 && NR == 24)
                config = benchmarkConfig<24, 24>();
            else if (MR == 24 && NR == 32)
                config = benchmarkConfig<24, 32>();
            else if (MR == 24 && NR == 48)
                config = benchmarkConfig<24, 48>();
            else if (MR == 24 && NR == 64)
                config = benchmarkConfig<24, 64>();
            else if (MR == 24 && NR == 96)
                config = benchmarkConfig<24, 96>();
            else if (MR == 24 && NR == 128)
                config = benchmarkConfig<24, 128>();
            else if (MR == 32 && NR == 8)
                config = benchmarkConfig<32, 8>();
            else if (MR == 32 && NR == 16)
                config = benchmarkConfig<32, 16>();
            else if (MR == 32 && NR == 24)
                config = benchmarkConfig<32, 24>();
            else if (MR == 32 && NR == 32)
                config = benchmarkConfig<32, 32>();
            else if (MR == 32 && NR == 48)
                config = benchmarkConfig<32, 48>();
            else if (MR == 32 && NR == 64)
                config = benchmarkConfig<32, 64>();
            else if (MR == 32 && NR == 96)
                config = benchmarkConfig<32, 96>();
            else if (MR == 32 && NR == 128)
                config = benchmarkConfig<32, 128>();
            else if (MR == 48 && NR == 8)
                config = benchmarkConfig<48, 8>();
            else if (MR == 48 && NR == 16)
                config = benchmarkConfig<48, 16>();
            else if (MR == 48 && NR == 24)
                config = benchmarkConfig<48, 24>();
            else if (MR == 48 && NR == 32)
                config = benchmarkConfig<48, 32>();
            else if (MR == 48 && NR == 48)
                config = benchmarkConfig<48, 48>();
            else if (MR == 48 && NR == 64)
                config = benchmarkConfig<48, 64>();
            else if (MR == 48 && NR == 96)
                config = benchmarkConfig<48, 96>();
            else if (MR == 48 && NR == 128)
                config = benchmarkConfig<48, 128>();
            else if (MR == 64 && NR == 8)
                config = benchmarkConfig<64, 8>();
            else if (MR == 64 && NR == 16)
                config = benchmarkConfig<64, 16>();
            else if (MR == 64 && NR == 24)
                config = benchmarkConfig<64, 24>();
            else if (MR == 64 && NR == 32)
                config = benchmarkConfig<64, 32>();
            else if (MR == 64 && NR == 48)
                config = benchmarkConfig<64, 48>();
            else if (MR == 64 && NR == 64)
                config = benchmarkConfig<64, 64>();
            else if (MR == 64 && NR == 96)
                config = benchmarkConfig<64, 96>();
            else if (MR == 64 && NR == 128)
                config = benchmarkConfig<64, 128>();
            else if (MR == 96 && NR == 8)
                config = benchmarkConfig<96, 8>();
            else if (MR == 96 && NR == 16)
                config = benchmarkConfig<96, 16>();
            else if (MR == 96 && NR == 24)
                config = benchmarkConfig<96, 24>();
            else if (MR == 96 && NR == 32)
                config = benchmarkConfig<96, 32>();
            else if (MR == 96 && NR == 48)
                config = benchmarkConfig<96, 48>();
            else if (MR == 96 && NR == 64)
                config = benchmarkConfig<96, 64>();
            else if (MR == 96 && NR == 96)
                config = benchmarkConfig<96, 96>();
            else if (MR == 96 && NR == 128)
                config = benchmarkConfig<96, 128>();
            else if (MR == 128 && NR == 8)
                config = benchmarkConfig<128, 8>();
            else if (MR == 128 && NR == 16)
                config = benchmarkConfig<128, 16>();
            else if (MR == 128 && NR == 24)
                config = benchmarkConfig<128, 24>();
            else if (MR == 128 && NR == 32)
                config = benchmarkConfig<128, 32>();
            else if (MR == 128 && NR == 48)
                config = benchmarkConfig<128, 48>();
            else if (MR == 128 && NR == 64)
                config = benchmarkConfig<128, 64>();
            else if (MR == 128 && NR == 96)
                config = benchmarkConfig<128, 96>();
            else if (MR == 128 && NR == 128)
                config = benchmarkConfig<128, 128>();

            results.push_back(config);

            if (config.valid)
            {
                std::cout << "  MR=" << std::setw(3) << MR << " NR=" << std::setw(3) << NR
                          << " → " << std::setw(6) << std::fixed << std::setprecision(1)
                          << config.gflops << " GFLOPS  (" << std::setw(6) << std::setprecision(2)
                          << config.avg_ms << " ms)" << std::endl;
            }
        }
    }

    // Find best configuration
    auto best = std::max_element(results.begin(), results.end(),
                                 [](const KernelConfig &a, const KernelConfig &b)
                                 {
                                     return a.gflops < b.gflops;
                                 });

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total configurations tested: " << results.size() << std::endl;
    std::cout << "Valid configurations: "
              << std::count_if(results.begin(), results.end(),
                               [](const KernelConfig &c)
                               { return c.valid; })
              << std::endl;

    if (best != results.end() && best->valid)
    {
        std::cout << "\n✅ Best Configuration:" << std::endl;
        std::cout << "  MR=" << best->MR << " NR=" << best->NR << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
                  << best->gflops << " GFLOPS" << std::endl;
        std::cout << "  Avg time:   " << std::fixed << std::setprecision(2)
                  << best->avg_ms << " ms" << std::endl;

        // Compare to baseline (32×64)
        auto baseline = std::find_if(results.begin(), results.end(),
                                     [](const KernelConfig &c)
                                     {
                                         return c.MR == 32 && c.NR == 64 && c.valid;
                                     });
        if (baseline != results.end())
        {
            double improvement = ((best->gflops - baseline->gflops) / baseline->gflops) * 100.0;
            std::cout << "  vs Baseline (32×64): " << std::showpos << std::fixed << std::setprecision(1)
                      << improvement << "%" << std::noshowpos << std::endl;
        }
    }

    // Export results to CSV
    std::ofstream csv("q8_1_kernel_sweep_results.csv");
    csv << "MR,NR,GFLOPS,Time_ms,Valid\n";
    for (const auto &cfg : results)
    {
        csv << cfg.MR << "," << cfg.NR << "," << std::fixed << std::setprecision(2)
            << cfg.gflops << "," << cfg.avg_ms << "," << (cfg.valid ? "1" : "0") << "\n";
    }
    csv.close();
    std::cout << "\n📊 Results exported to: q8_1_kernel_sweep_results.csv" << std::endl;

    // Test should pass if we have at least one valid configuration
    EXPECT_TRUE(std::any_of(results.begin(), results.end(),
                            [](const KernelConfig &c)
                            { return c.valid; }))
        << "No valid kernel configurations found!";
}
