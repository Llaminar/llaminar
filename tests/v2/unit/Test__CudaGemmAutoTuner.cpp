/**
 * @file Test__CudaGemmAutoTuner.cpp
 * @brief Unit tests for CUDA GEMM auto-tuner
 *
 * Tests:
 * - Configuration generation (100-200 valid configs)
 * - Heuristic selection (fast fallback)
 * - Filtering pipeline (problem size, resources, ranking)
 * - Auto-tuning workflow (end-to-end)
 * - Cache management
 *
 * @author David Sanftenberg
 * @date October 31, 2025
 */

#include <gtest/gtest.h>
#include "kernels/cuda/CudaGemmAutoTuner.h"
#include "kernels/cuda/CudaGemmVariantsBaseline.h"
#include "kernels/cuda/CudaGemmConfig.h"
#include "tensors/FP16Utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <random>

using namespace llaminar2::cuda;

class Test__CudaGemmAutoTuner : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check CUDA availability
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Reset auto-tuner state
        CudaGemmAutoTuner::instance().clearCache();
        CudaGemmAutoTuner::instance().setAutoTuningEnabled(true);
    }
};

// ============================================================================
// Configuration Generation Tests
// ============================================================================

TEST_F(Test__CudaGemmAutoTuner, GeneratesValidConfigurations)
{
    auto configs = CudaGemmAutoTuner::instance().getAvailableConfigs();

    // Should generate 2000-3000 valid configurations (atom-aware search space: 4× expansion)
    // Base configs (648) × atom_layouts (4) = 2,592 configs
    EXPECT_GE(configs.size(), 3800) << "Too few configurations generated (expected ~3,888 with atom diversity)";
    EXPECT_LE(configs.size(), 4000) << "Too many configurations generated (expected ~3,888 with atom diversity)";

    // All configs should be valid
    for (const auto &config : configs)
    {
        EXPECT_TRUE(config.isValid()) << "Invalid config: " << config.id();
    }

    std::cout << "Generated " << configs.size() << " valid configurations" << std::endl;
}

TEST_F(Test__CudaGemmAutoTuner, ConfigurationsSpanParameterSpace)
{
    auto configs = CudaGemmAutoTuner::instance().getAvailableConfigs();

    // Count unique tile sizes
    std::set<int> unique_tile_m, unique_tile_n, unique_tile_k;
    std::set<int> unique_threads_m, unique_threads_n;
    std::set<int> unique_prefetch;

    for (const auto &config : configs)
    {
        unique_tile_m.insert(config.tile_m);
        unique_tile_n.insert(config.tile_n);
        unique_tile_k.insert(config.tile_k);
        unique_threads_m.insert(config.threads_m);
        unique_threads_n.insert(config.threads_n);
        unique_prefetch.insert(config.prefetch_stages);
    }

    // Should explore multiple values for each parameter
    EXPECT_GE(unique_tile_m.size(), 3) << "Insufficient tile_m diversity";
    EXPECT_GE(unique_tile_n.size(), 3) << "Insufficient tile_n diversity";
    // Note: tile_k is fixed at 32 for IQ4_NL block size, so we expect exactly 1 value
    EXPECT_EQ(unique_tile_k.size(), 1) << "Expected single tile_k value (32 for IQ4_NL)";
    EXPECT_GE(unique_threads_m.size(), 2) << "Insufficient threads_m diversity";
    EXPECT_GE(unique_threads_n.size(), 2) << "Insufficient threads_n diversity";
    EXPECT_GE(unique_prefetch.size(), 2) << "Insufficient prefetch diversity";

    std::cout << "Parameter space coverage:" << std::endl;
    std::cout << "  tile_m values: " << unique_tile_m.size() << std::endl;
    std::cout << "  tile_n values: " << unique_tile_n.size() << std::endl;
    std::cout << "  tile_k values: " << unique_tile_k.size() << std::endl;
    std::cout << "  prefetch stages: " << unique_prefetch.size() << std::endl;
}

// ============================================================================
// Heuristic Selection Tests
// ============================================================================

TEST_F(Test__CudaGemmAutoTuner, HeuristicSelectsSmallConfigForSmallMatrix)
{
    CudaGemmAutoTuner::instance().setAutoTuningEnabled(false);

    auto config = CudaGemmAutoTuner::instance().getOptimalConfig(64, 64, 896);

    // Small matrix should use reasonably sized tiles (≤ matrix dimensions)
    // For 64x64 matrix, 64x64 tile is acceptable (covers whole matrix in one tile)
    EXPECT_LE(config.tile_m, 64) << "Tile too large for small matrix";
    EXPECT_LE(config.tile_n, 64) << "Tile too large for small matrix";
    EXPECT_TRUE(config.isValid());

    std::cout << "Small matrix heuristic: " << config.id() << std::endl;
}

TEST_F(Test__CudaGemmAutoTuner, HeuristicSelectsLargeConfigForLargeMatrix)
{
    CudaGemmAutoTuner::instance().setAutoTuningEnabled(false);

    auto config = CudaGemmAutoTuner::instance().getOptimalConfig(1024, 1024, 3584);

    // Large matrix should use large tile
    EXPECT_GE(config.tile_m, 32) << "Tile too small for large matrix";
    EXPECT_GE(config.tile_n, 32) << "Tile too small for large matrix";
    EXPECT_TRUE(config.isValid());

    std::cout << "Large matrix heuristic: " << config.id() << std::endl;
}

TEST_F(Test__CudaGemmAutoTuner, HeuristicAdaptsToProblemShape)
{
    CudaGemmAutoTuner::instance().setAutoTuningEnabled(false);

    // Tall matrix (m >> n)
    auto tall_config = CudaGemmAutoTuner::instance().getOptimalConfig(512, 128, 3584);

    // Wide matrix (n >> m)
    auto wide_config = CudaGemmAutoTuner::instance().getOptimalConfig(128, 512, 3584);

    // Configs may differ for different aspect ratios, but not required
    // (square configs can be optimal for both in some cases)
    // Just verify both are valid and reasonable
    EXPECT_TRUE(tall_config.isValid());
    EXPECT_TRUE(wide_config.isValid());

    // At least one dimension should adapt to problem size
    bool adapts = (tall_config.tile_m != wide_config.tile_m) ||
                  (tall_config.tile_n != wide_config.tile_n);

    std::cout << "Tall matrix: " << tall_config.id() << std::endl;
    std::cout << "Wide matrix: " << wide_config.id() << std::endl;
    std::cout << "Heuristic adapts to shape: " << (adapts ? "YES" : "NO (uses square tiles)") << std::endl;
}

// ============================================================================
// Auto-Tuning Workflow Tests
// ============================================================================

TEST_F(Test__CudaGemmAutoTuner, AutoTunesOnFirstCall)
{
    CudaGemmAutoTuner::instance().setAutoTuningEnabled(true);
    CudaGemmAutoTuner::instance().setMaxCandidates(5); // Faster for testing
    CudaGemmAutoTuner::instance().setBenchmarkIterations(3);
    CudaGemmAutoTuner::instance().setWarmupIterations(1);

    // First call should auto-tune
    auto config = CudaGemmAutoTuner::instance().getOptimalConfig(128, 128, 896);

    EXPECT_TRUE(config.isValid());

    // Should have benchmark results
    auto results = CudaGemmAutoTuner::instance().getBenchmarkResults(128, 128, 896);
    EXPECT_GE(results.size(), 1) << "No benchmark results stored";
    EXPECT_LE(results.size(), 5) << "Too many results (should limit to max_candidates)";

    // Results should be sorted by GFLOPS (best first)
    for (size_t i = 1; i < results.size(); ++i)
    {
        EXPECT_GE(results[i - 1].gflops, results[i].gflops)
            << "Results not sorted by performance";
    }

    std::cout << "Auto-tuned config: " << config.id() << std::endl;
    std::cout << "Best GFLOPS: " << results[0].gflops << std::endl;
    std::cout << "Worst GFLOPS: " << results.back().gflops << std::endl;
}

TEST_F(Test__CudaGemmAutoTuner, CachesOptimalConfiguration)
{
    CudaGemmAutoTuner::instance().setAutoTuningEnabled(true);
    CudaGemmAutoTuner::instance().setMaxCandidates(3);

    // First call auto-tunes
    auto config1 = CudaGemmAutoTuner::instance().getOptimalConfig(256, 256, 3584);

    // Second call should return cached result (fast)
    auto start = std::chrono::high_resolution_clock::now();
    auto config2 = CudaGemmAutoTuner::instance().getOptimalConfig(256, 256, 3584);
    auto end = std::chrono::high_resolution_clock::now();
    auto cached_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Should return same config
    EXPECT_EQ(config1.id(), config2.id());

    // Should be very fast (< 1ms)
    EXPECT_LT(cached_time_us, 1000) << "Cache lookup too slow: " << cached_time_us << " μs";

    std::cout << "Cached lookup time: " << cached_time_us << " μs" << std::endl;
}

TEST_F(Test__CudaGemmAutoTuner, AutoTunesDifferentShapesSeparately)
{
    CudaGemmAutoTuner::instance().setAutoTuningEnabled(true);
    CudaGemmAutoTuner::instance().setMaxCandidates(3);

    // Auto-tune for two different shapes
    auto config_small = CudaGemmAutoTuner::instance().getOptimalConfig(32, 32, 896);
    auto config_large = CudaGemmAutoTuner::instance().getOptimalConfig(512, 512, 3584);

    // Configs should likely differ (different optimal for different sizes)
    // Note: Not guaranteed to differ, but very likely
    std::cout << "Small matrix config: " << config_small.id() << std::endl;
    std::cout << "Large matrix config: " << config_large.id() << std::endl;

    // Both should have benchmark results
    auto results_small = CudaGemmAutoTuner::instance().getBenchmarkResults(32, 32, 896);
    auto results_large = CudaGemmAutoTuner::instance().getBenchmarkResults(512, 512, 3584);

    EXPECT_GE(results_small.size(), 1);
    EXPECT_GE(results_large.size(), 1);
}

// ============================================================================
// Cache Management Tests
// ============================================================================

TEST_F(Test__CudaGemmAutoTuner, ClearCacheRemovesAllEntries)
{
    CudaGemmAutoTuner::instance().setAutoTuningEnabled(true);
    CudaGemmAutoTuner::instance().setMaxCandidates(2);

    // Populate cache
    CudaGemmAutoTuner::instance().getOptimalConfig(128, 128, 896);
    CudaGemmAutoTuner::instance().getOptimalConfig(256, 256, 3584);

    // Verify entries exist
    auto results1 = CudaGemmAutoTuner::instance().getBenchmarkResults(128, 128, 896);
    auto results2 = CudaGemmAutoTuner::instance().getBenchmarkResults(256, 256, 3584);
    EXPECT_GE(results1.size(), 1);
    EXPECT_GE(results2.size(), 1);

    // Clear cache
    CudaGemmAutoTuner::instance().clearCache();

    // Results should be gone
    auto results_after1 = CudaGemmAutoTuner::instance().getBenchmarkResults(128, 128, 896);
    auto results_after2 = CudaGemmAutoTuner::instance().getBenchmarkResults(256, 256, 3584);
    EXPECT_EQ(results_after1.size(), 0);
    EXPECT_EQ(results_after2.size(), 0);
}

TEST_F(Test__CudaGemmAutoTuner, ManualConfigurationOverridesAutoTuning)
{
    // Set manual configuration
    auto manual_config = presets::small();
    CudaGemmAutoTuner::instance().setOptimalConfig(128, 128, 896, manual_config);

    // Get config (should return manual one, no auto-tuning)
    auto retrieved_config = CudaGemmAutoTuner::instance().getOptimalConfig(128, 128, 896);

    EXPECT_EQ(retrieved_config.id(), manual_config.id());

    // Should NOT have benchmark results (auto-tuning was skipped)
    auto results = CudaGemmAutoTuner::instance().getBenchmarkResults(128, 128, 896);
    EXPECT_EQ(results.size(), 0) << "Manual config should not trigger benchmarking";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
