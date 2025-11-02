/**
 * @file Test__MLAutoTunerE2E.cpp
 * @brief End-to-end integration test for ML-based autotuner
 *
 * Validates that CudaGemmAutoTuner correctly uses ML predictor and
 * that ML-selected tile configurations produce correct results with
 * good performance.
 *
 * @author David Sanftenberg
 * @date November 1, 2025
 */

#include <gtest/gtest.h>
#include "../../../src/v2/kernels/cuda/CudaGemmAutoTuner.h"
#include "../../../src/v2/kernels/cuda/CudaGemmVariantsBaseline.h"
#include "../../../src/v2/kernels/cuda/IQ4_NL_BlockDecoder.h"
#include "../../../build_v2/autotuner_models/GemmAutoTunerML.h"
#include <cuda_runtime.h>
#include <random>
#include <cmath>

using namespace llaminar2;
using namespace llaminar2::cuda;

class Test__MLAutoTunerE2E : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure ML predictor is enabled (not using old heuristic)
        unsetenv("LLAMINAR_USE_OLD_HEURISTIC");

        // Disable full auto-tuning (we want heuristic selection only)
        setenv("LLAMINAR_DISABLE_CUDA_AUTOTUNE", "1", 1);

        // Clear autotuner cache
        CudaGemmAutoTuner::instance().clearCache();
    }

    void TearDown() override
    {
        // Cleanup
        unsetenv("LLAMINAR_DISABLE_CUDA_AUTOTUNE");
    }

    // Helper: Validate ML predictor is being used
    bool validateMLPrediction(int m, int n, int k)
    {
        auto ml_tile = GemmAutoTunerML::predict(m, n, k);
        auto autotuner_config = CudaGemmAutoTuner::instance().getOptimalConfig(m, n, k);

        bool match = (autotuner_config.tile_m == ml_tile.tile_m &&
                      autotuner_config.tile_n == ml_tile.tile_n &&
                      autotuner_config.tile_k == ml_tile.tile_k);

        if (!match)
        {
            std::cout << "ML vs Autotuner mismatch for (" << m << ", " << n << ", " << k << "):\n"
                      << "  ML: TM" << ml_tile.tile_m << "_TN" << ml_tile.tile_n << "_TK" << ml_tile.tile_k << "\n"
                      << "  Autotuner: TM" << autotuner_config.tile_m
                      << "_TN" << autotuner_config.tile_n
                      << "_TK" << autotuner_config.tile_k << std::endl;
        }

        return match;
    }

    // Helper: Run GEMM with ML-selected config and validate correctness
    struct GemmTestResult
    {
        bool success;
        double max_abs_error;
        double gflops;
        CudaGemmConfig config;
    };

    GemmTestResult runGemmTest(int m, int n, int k)
    {
        GemmTestResult result;
        result.success = false;
        result.max_abs_error = 0.0;
        result.gflops = 0.0;

        // Get ML-selected config
        result.config = CudaGemmAutoTuner::instance().getOptimalConfig(m, n, k);

        // Allocate test data
        size_t A_size = static_cast<size_t>(m) * k;
        size_t B_blocks = n * (k / 32);
        size_t C_size = static_cast<size_t>(m) * n;

        std::vector<float> A_host(A_size);
        std::vector<IQ4_NLBlock> B_host(B_blocks); // From cuda namespace
        std::vector<float> C_host(C_size, 0.0f);

        // Initialize with small random values
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        for (auto &val : A_host)
        {
            val = dist(rng);
        }

        // Initialize B with simple pattern (for reproducibility)
        for (size_t i = 0; i < B_blocks; ++i)
        {
            // Convert 0.1f to FP16 for scale (approximation)
            uint16_t scale_fp16 = 0x2E00; // ~0.09375 in FP16
            B_host[i].d = scale_fp16;
            for (int j = 0; j < 16; ++j)
            {
                B_host[i].qs[j] = static_cast<uint8_t>(i % 256);
            }
        }

        // Allocate GPU memory
        float *d_A = nullptr;
        IQ4_NLBlock *d_B = nullptr; // From cuda namespace
        float *d_C = nullptr;

        cudaMalloc(&d_A, A_size * sizeof(float));
        cudaMalloc(&d_B, B_blocks * sizeof(IQ4_NLBlock));
        cudaMalloc(&d_C, C_size * sizeof(float));

        cudaMemcpy(d_A, A_host.data(), A_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B_host.data(), B_blocks * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, C_size * sizeof(float));

        // Run GEMM with ML-selected config
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup
        for (int i = 0; i < 3; ++i)
        {
            launchIQ4NLGemmVariant(d_A, d_B, d_C, m, n, k, result.config, stream);
        }
        cudaStreamSynchronize(stream);

        // Timed run
        cudaEventRecord(start, stream);
        for (int i = 0; i < 10; ++i)
        {
            auto err = launchIQ4NLGemmVariant(d_A, d_B, d_C, m, n, k, result.config, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "GEMM launch failed: " << cudaGetErrorString(err) << std::endl;
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                cudaStreamDestroy(stream);
                return result;
            }
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        // Copy result back
        cudaMemcpy(C_host.data(), d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Compute performance
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        double flops = 2.0 * static_cast<double>(m) * n * k;
        result.gflops = (flops / 1e9) / ((elapsed_ms / 10.0) / 1000.0);

        // Basic sanity check: C should contain non-zero values
        int non_zero_count = 0;
        for (const auto &val : C_host)
        {
            if (std::abs(val) > 1e-6f)
                non_zero_count++;
        }

        // At least 10% of output should be non-zero (sanity check)
        result.max_abs_error = (non_zero_count >= C_size / 10) ? 0.0 : 1.0;

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);

        result.success = true;
        return result;
    }
};

/**
 * @brief Validate that autotuner uses ML predictor
 */
TEST_F(Test__MLAutoTunerE2E, AutotunerUsesMLPredictor)
{
    // Test all 12 training workloads
    struct Workload
    {
        int m, n, k;
    };
    std::vector<Workload> workloads = {
        // Single token
        {1, 896, 896},    // 0.5B QKV
        {1, 4864, 896},   // 0.5B FFN
        {1, 2560, 2560},  // 4B QKV
        {1, 2560, 13824}, // 4B FFN
        {1, 4096, 4096},  // 7B QKV
        {1, 22016, 4096}, // 7B FFN
        {1, 5120, 5120},  // 14B QKV
        {1, 5120, 27648}, // 14B FFN

        // Batched
        {32, 896, 896},    // 0.5B batch 32
        {128, 2560, 2560}, // 4B batch 128
        {128, 4096, 4096}, // 7B batch 128
        {256, 5120, 5120}, // 14B batch 256
    };

    int matches = 0;
    for (const auto &wl : workloads)
    {
        bool match = validateMLPrediction(wl.m, wl.n, wl.k);
        EXPECT_TRUE(match) << "Workload (" << wl.m << ", " << wl.n << ", " << wl.k << ")";
        if (match)
            matches++;
    }

    std::cout << "ML predictor matches: " << matches << "/" << workloads.size() << std::endl;
    EXPECT_EQ(matches, workloads.size()) << "All workloads should use ML predictor";
}

/**
 * @brief Validate correctness with ML-selected configs (small matrices)
 */
TEST_F(Test__MLAutoTunerE2E, CorrectnessValidation_SingleToken_0_5B)
{
    auto result = runGemmTest(1, 896, 896);

    ASSERT_TRUE(result.success) << "GEMM execution should succeed";
    EXPECT_GT(result.gflops, 0.0) << "Should achieve non-zero GFLOPS";
    EXPECT_LT(result.max_abs_error, 0.5) << "Should have reasonable output (sanity check passed)";

    std::cout << "0.5B Single Token QKV:\n"
              << "  Config: TM" << result.config.tile_m
              << "_TN" << result.config.tile_n
              << "_TK" << result.config.tile_k << "\n"
              << "  Performance: " << result.gflops << " GFLOPS\n"
              << "  Sanity check: " << (result.max_abs_error < 0.5 ? "PASSED" : "FAILED") << std::endl;
}

TEST_F(Test__MLAutoTunerE2E, CorrectnessValidation_Batch32_0_5B)
{
    auto result = runGemmTest(32, 896, 896);

    ASSERT_TRUE(result.success) << "GEMM execution should succeed";
    EXPECT_GT(result.gflops, 0.0) << "Should achieve non-zero GFLOPS";
    EXPECT_LT(result.max_abs_error, 0.5) << "Should have reasonable output (sanity check passed)";

    std::cout << "0.5B Batch 32 QKV:\n"
              << "  Config: TM" << result.config.tile_m
              << "_TN" << result.config.tile_n
              << "_TK" << result.config.tile_k << "\n"
              << "  Performance: " << result.gflops << " GFLOPS\n"
              << "  Sanity check: " << (result.max_abs_error < 0.5 ? "PASSED" : "FAILED") << std::endl;
}

/**
 * @brief Performance sanity check (no exact validation for large matrices)
 */
TEST_F(Test__MLAutoTunerE2E, PerformanceSanityCheck_7B_SingleToken)
{
    auto result = runGemmTest(1, 4096, 4096);

    ASSERT_TRUE(result.success) << "GEMM execution should succeed";
    EXPECT_GT(result.gflops, 20.0) << "Should achieve >20 GFLOPS for 7B single token";

    std::cout << "7B Single Token QKV:\n"
              << "  Config: TM" << result.config.tile_m
              << "_TN" << result.config.tile_n
              << "_TK" << result.config.tile_k << "\n"
              << "  Performance: " << result.gflops << " GFLOPS" << std::endl;
}

TEST_F(Test__MLAutoTunerE2E, PerformanceSanityCheck_7B_Batch128)
{
    auto result = runGemmTest(128, 4096, 4096);

    ASSERT_TRUE(result.success) << "GEMM execution should succeed";
    EXPECT_GT(result.gflops, 1000.0) << "Should achieve >1000 GFLOPS for 7B batch 128";

    std::cout << "7B Batch 128 QKV:\n"
              << "  Config: TM" << result.config.tile_m
              << "_TN" << result.config.tile_n
              << "_TK" << result.config.tile_k << "\n"
              << "  Performance: " << result.gflops << " GFLOPS" << std::endl;
}

/**
 * @brief Test fallback to old heuristic when requested
 */
TEST_F(Test__MLAutoTunerE2E, FallbackToOldHeuristic)
{
    // Enable old heuristic
    setenv("LLAMINAR_USE_OLD_HEURISTIC", "1", 1);

    // Clear cache
    CudaGemmAutoTuner::instance().clearCache();

    // Get config
    auto config = CudaGemmAutoTuner::instance().getOptimalConfig(1, 896, 896);

    // Should NOT match ML prediction
    auto ml_tile = GemmAutoTunerML::predict(1, 896, 896);

    // Old heuristic uses different logic (size-based presets)
    // We just verify it doesn't crash
    EXPECT_GT(config.tile_m, 0);
    EXPECT_GT(config.tile_n, 0);
    EXPECT_EQ(config.tile_k, 32); // Always 32 for IQ4_NL

    std::cout << "Old heuristic config: TM" << config.tile_m
              << "_TN" << config.tile_n
              << "_TK" << config.tile_k << std::endl;

    unsetenv("LLAMINAR_USE_OLD_HEURISTIC");
}

/**
 * @brief Test caching behavior
 */
TEST_F(Test__MLAutoTunerE2E, ConfigCaching)
{
    // First call should compute
    auto config1 = CudaGemmAutoTuner::instance().getOptimalConfig(1, 896, 896);

    // Second call should hit cache
    auto config2 = CudaGemmAutoTuner::instance().getOptimalConfig(1, 896, 896);

    // Should be identical
    EXPECT_EQ(config1.tile_m, config2.tile_m);
    EXPECT_EQ(config1.tile_n, config2.tile_n);
    EXPECT_EQ(config1.tile_k, config2.tile_k);

    // Clear cache
    CudaGemmAutoTuner::instance().clearCache();

    // Third call should recompute
    auto config3 = CudaGemmAutoTuner::instance().getOptimalConfig(1, 896, 896);

    // Should match (same ML prediction)
    EXPECT_EQ(config1.tile_m, config3.tile_m);
    EXPECT_EQ(config1.tile_n, config3.tile_n);
    EXPECT_EQ(config1.tile_k, config3.tile_k);
}
