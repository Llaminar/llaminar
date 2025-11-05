/**
 * @file Test__CudaGemmAutoTuner.cpp
 * @brief Unit tests for CudaGemmAutoTuner class
 *
 * Tests cover:
 * - Configuration generation and filtering
 * - Heuristic ranking (manual, ML, and NN modes)
 * - Cache management
 * - Environment variable handling
 * - Fallback behavior when ONNX/ML unavailable
 *
 * @author David Sanftenberg
 * @date November 3, 2025
 */

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include "../../../../../src/v2/kernels/cuda/CudaGemmAutoTuner.h"
#include "../../../../../src/v2/kernels/cuda/CudaGemmConfig.h"

#ifdef HAVE_ONNX_RUNTIME
#include "../../../../../src/v2/kernels/cuda/CudaGemmNeuralNetwork.h"
#endif

using namespace llaminar2::cuda;

/**
 * @brief Test fixture for CudaGemmAutoTuner tests
 */
class Test__CudaGemmAutoTuner : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clear any cached configs before each test
        CudaGemmAutoTuner::instance().clearCache();
    }

    void TearDown() override
    {
        // Clean up environment variables
        unsetenv("LLAMINAR_USE_ML_HEURISTIC");
        unsetenv("LLAMINAR_USE_NN_HEURISTIC");
        unsetenv("LLAMINAR_DISABLE_CUDA_AUTOTUNE");
        unsetenv("LLAMINAR_CUDA_AUTOTUNE_CANDIDATES");
    }
};

// ============================================================================
// Configuration Generation Tests
// ============================================================================

/**
 * @brief Test that autotuner generates valid configurations
 */
TEST_F(Test__CudaGemmAutoTuner, GeneratesValidConfigurations)
{
    auto &tuner = CudaGemmAutoTuner::instance();

    // Disable auto-tuning to just get heuristic selection
    tuner.setAutoTuningEnabled(false);

    // Get config for a small problem
    auto config = tuner.getOptimalConfig(1, 896, 896);

    // Verify config has valid parameters
    EXPECT_GT(config.tile_m, 0);
    EXPECT_GT(config.tile_n, 0);
    EXPECT_GT(config.tile_k, 0);
    EXPECT_GT(config.threads_m, 0);
    EXPECT_GT(config.threads_n, 0);
    EXPECT_GT(config.work_per_thread_m, 0);
    EXPECT_GT(config.work_per_thread_n, 0);

    // Thread block size should be reasonable (32-1024)
    int thread_block_size = config.threads_m * config.threads_n;
    EXPECT_GE(thread_block_size, 32);
    EXPECT_LE(thread_block_size, 1024);

    // Config should pass its own validation
    EXPECT_TRUE(config.isValid());
}

/**
 * @brief Test that different problem sizes get different configurations
 */
TEST_F(Test__CudaGemmAutoTuner, AdaptsToProblemsSize)
{
    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    // Small single-token problem
    auto config_small = tuner.getOptimalConfig(1, 896, 896);

    // Large batch problem
    auto config_large = tuner.getOptimalConfig(128, 896, 896);

    // Configs should be different (at least some parameters)
    bool configs_differ = (config_small.tile_m != config_large.tile_m) ||
                          (config_small.tile_n != config_large.tile_n) ||
                          (config_small.tile_k != config_large.tile_k) ||
                          (config_small.threads_m != config_large.threads_m) ||
                          (config_small.threads_n != config_large.threads_n);

    EXPECT_TRUE(configs_differ) << "Small and large problems should use different configs";
}

// ============================================================================
// Cache Management Tests
// ============================================================================

/**
 * @brief Test that cache stores and retrieves configurations
 */
TEST_F(Test__CudaGemmAutoTuner, CacheStoresAndRetrieves)
{
    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    // Get config (will be cached)
    auto config1 = tuner.getOptimalConfig(1, 1280, 1280);

    // Get same config again (should be from cache)
    auto config2 = tuner.getOptimalConfig(1, 1280, 1280);

    // Should be identical
    EXPECT_EQ(config1.id(), config2.id());
}

/**
 * @brief Test that clearCache() actually clears the cache
 */
TEST_F(Test__CudaGemmAutoTuner, ClearCacheWorks)
{
    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    // Cache a config
    auto config1 = tuner.getOptimalConfig(1, 2048, 2048);

    // Clear cache
    tuner.clearCache();

    // Get config again - should work (regenerate from heuristic)
    auto config2 = tuner.getOptimalConfig(1, 2048, 2048);

    // Should be valid (same heuristic should produce same config)
    EXPECT_TRUE(config2.isValid());
    EXPECT_EQ(config1.id(), config2.id()) << "Heuristic should be deterministic";
}

// ============================================================================
// Heuristic Ranking Tests
// ============================================================================

/**
 * @brief Test manual heuristic mode (default)
 */
TEST_F(Test__CudaGemmAutoTuner, ManualHeuristicMode)
{
    unsetenv("LLAMINAR_USE_ML_HEURISTIC");
    unsetenv("LLAMINAR_USE_NN_HEURISTIC");

    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    // Should use manual heuristic (no crashes)
    auto config = tuner.getOptimalConfig(1, 1280, 1280);
    EXPECT_TRUE(config.tile_m > 0); // Got a valid config
}

/**
 * @brief Test ML heuristic mode
 */
TEST_F(Test__CudaGemmAutoTuner, MLHeuristicMode)
{
    setenv("LLAMINAR_USE_ML_HEURISTIC", "1", 1);
    unsetenv("LLAMINAR_USE_NN_HEURISTIC");

    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    // Should use ML heuristic (lookup table)
    auto config = tuner.getOptimalConfig(1, 1280, 1280);
    EXPECT_TRUE(config.tile_m > 0); // Got a valid config
}

#ifdef HAVE_ONNX_RUNTIME
/**
 * @brief Test NN heuristic mode (requires ONNX Runtime)
 */
TEST_F(Test__CudaGemmAutoTuner, NNHeuristicMode)
{
    unsetenv("LLAMINAR_USE_ML_HEURISTIC");
    setenv("LLAMINAR_USE_NN_HEURISTIC", "1", 1);

    // Verify NN is initialized
    auto &nn = CudaGemmNeuralNetwork::instance();
    ASSERT_TRUE(nn.isInitialized()) << "ONNX neural network must be initialized for this test";

    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    // Should use NN ranking model
    auto config = tuner.getOptimalConfig(1, 1280, 1280);
    EXPECT_TRUE(config.tile_m > 0); // Got a valid config
}

/**
 * @brief Test NN ranking returns consistent scores
 */
TEST_F(Test__CudaGemmAutoTuner, NNRankingIsConsistent)
{
    auto &nn = CudaGemmNeuralNetwork::instance();
    ASSERT_TRUE(nn.isInitialized());

    // Use explicitly different configs
    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    // Get configs for very different problem sizes
    auto config_small = tuner.getOptimalConfig(1, 896, 896);     // Small single token
    auto config_batch = tuner.getOptimalConfig(128, 5120, 5120); // Large batch

    // Verify configs are actually different
    ASSERT_NE(config_small.id(), config_batch.id()) << "Should have different configs for test";

    int m = 1, n = 1280, k = 1280;

    // Rank same config twice
    double score1_a = nn.rankConfig(config_small, m, n, k);
    double score1_b = nn.rankConfig(config_small, m, n, k);

    // Should be identical (deterministic)
    EXPECT_DOUBLE_EQ(score1_a, score1_b) << "NN ranking should be deterministic";

    // Different config should get different score (most likely)
    double score2 = nn.rankConfig(config_batch, m, n, k);
    // Note: It's theoretically possible (but very unlikely) that two different
    // configs could get the exact same score, so we don't enforce strict inequality
    // We just verify the NN can produce scores
    EXPECT_GT(score2, 0.0) << "NN should produce positive ranking scores";
}
#endif

/**
 * @brief Test fallback behavior when NN/ML unavailable
 */
TEST_F(Test__CudaGemmAutoTuner, FallbackToManualHeuristic)
{
    // Force NN mode even if not available
    setenv("LLAMINAR_USE_NN_HEURISTIC", "1", 1);

    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    // Should fall back to manual heuristic gracefully (no crash)
    auto config = tuner.getOptimalConfig(1, 896, 896);
    EXPECT_TRUE(config.tile_m > 0); // Got a valid config
}

// ============================================================================
// Environment Variable Tests
// ============================================================================

/**
 * @brief Test LLAMINAR_DISABLE_CUDA_AUTOTUNE disables benchmarking
 */
TEST_F(Test__CudaGemmAutoTuner, DisableAutotuneEnvVar)
{
    setenv("LLAMINAR_DISABLE_CUDA_AUTOTUNE", "1", 1);

    auto &tuner = CudaGemmAutoTuner::instance();

    // Should skip benchmarking and use heuristic only
    auto config = tuner.getOptimalConfig(1, 1280, 1280);
    EXPECT_TRUE(config.tile_m > 0);

    // getBenchmarkResults should be empty (no benchmarking was done)
    auto results = tuner.getBenchmarkResults(1, 1280, 1280);
    EXPECT_TRUE(results.empty()) << "Should not have benchmark results when autotuning disabled";

    unsetenv("LLAMINAR_DISABLE_CUDA_AUTOTUNE");
}

// ============================================================================
// Configuration Validation Tests
// ============================================================================

/**
 * @brief Test that generated configs respect hardware constraints
 */
TEST_F(Test__CudaGemmAutoTuner, ConfigsRespectHardwareConstraints)
{
    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    std::vector<std::tuple<int, int, int>> test_shapes = {
        {1, 896, 896},     // 0.5B single token
        {1, 1280, 1280},   // 1.5B single token
        {32, 2048, 2048},  // 4B batch
        {128, 5120, 5120}, // 14B batch
    };

    for (const auto &[m, n, k] : test_shapes)
    {
        auto config = tuner.getOptimalConfig(m, n, k);

        // Thread block size must be multiple of warp size (32)
        int threads_per_block = config.threads_m * config.threads_n;
        EXPECT_EQ(threads_per_block % 32, 0)
            << "Thread block size must be multiple of 32 for shape [" << m << "×" << n << "×" << k << "]";

        // Thread block size must not exceed 1024
        EXPECT_LE(threads_per_block, 1024)
            << "Thread block size must be ≤1024 for shape [" << m << "×" << n << "×" << k << "]";

        // Shared memory should be reasonable (≤48KB typical limit)
        int smem_bytes = (config.tile_m * config.tile_k + config.tile_k * config.tile_n) * 4; // 4 bytes per float
        EXPECT_LE(smem_bytes, 49152)
            << "Shared memory usage should be ≤48KB for shape [" << m << "×" << n << "×" << k << "]";

        // Vectorization must be power of 2 (1, 2, 4)
        EXPECT_TRUE(config.vectorize_load == 1 || config.vectorize_load == 2 || config.vectorize_load == 4)
            << "Vectorization must be 1, 2, or 4 for shape [" << m << "×" << n << "×" << k << "]";
    }
}

/**
 * @brief Test that configs are valid
 */
TEST_F(Test__CudaGemmAutoTuner, ConfigsAreValid)
{
    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    auto config = tuner.getOptimalConfig(1, 2048, 2048);

    // tile_m must equal threads_m * work_per_thread_m
    EXPECT_EQ(config.tile_m, config.threads_m * config.work_per_thread_m)
        << "tile_m must equal threads_m × work_per_thread_m";

    // tile_n must equal threads_n * work_per_thread_n
    EXPECT_EQ(config.tile_n, config.threads_n * config.work_per_thread_n)
        << "tile_n × work_per_thread_n";

    // Should pass validation
    EXPECT_TRUE(config.isValid()) << "Config should pass internal validation";
}

// ============================================================================
// Stress Tests
// ============================================================================

/**
 * @brief Test autotuner with various problem sizes
 */
TEST_F(Test__CudaGemmAutoTuner, HandlesVariousProblemSizes)
{
    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    std::vector<std::tuple<int, int, int>> problem_sizes = {
        // Square matrices
        {1, 512, 512},
        {1, 1024, 1024},
        {1, 2048, 2048},
        {1, 4096, 4096},

        // Rectangular matrices (QKV projections)
        {1, 896, 896},
        {1, 1280, 1280},
        {1, 2048, 2048},
        {1, 5120, 5120},

        // Wide matrices (FFN gate)
        {1, 4864, 896},
        {1, 6912, 1280},

        // Tall matrices (FFN down)
        {1, 896, 4864},
        {1, 1280, 6912},

        // Batch processing
        {8, 896, 896},
        {32, 1280, 1280},
        {128, 2048, 2048},
    };

    for (const auto &[m, n, k] : problem_sizes)
    {
        SCOPED_TRACE("Testing shape [" + std::to_string(m) + "×" + std::to_string(n) + "×" + std::to_string(k) + "]");

        auto config = tuner.getOptimalConfig(m, n, k);

        // Should get a valid config
        EXPECT_GT(config.tile_m, 0);
        EXPECT_GT(config.tile_n, 0);
        EXPECT_GT(config.tile_k, 0);

        // Config should be valid
        EXPECT_TRUE(config.isValid()) << "Config should be valid for shape [" << m << "×" << n << "×" << k << "]";
    }
}

/**
 * @brief Test that autotuner is thread-safe (concurrent access)
 */
TEST_F(Test__CudaGemmAutoTuner, ThreadSafeConcurrentAccess)
{
    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    // Pre-populate cache for one shape
    auto config_cached = tuner.getOptimalConfig(1, 1024, 1024);

    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};

    // Launch multiple threads accessing different shapes
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i)
    {
        threads.emplace_back([&, i]()
                             {
            try {
                int size = 512 + i * 128;
                auto config = tuner.getOptimalConfig(1, size, size);
                if (config.tile_m > 0) {
                    success_count++;
                }
            } catch (...) {
                error_count++;
            } });
    }

    // Wait for all threads
    for (auto &t : threads)
    {
        t.join();
    }

    // All threads should succeed
    EXPECT_EQ(success_count, 8) << "All threads should successfully get configs";
    EXPECT_EQ(error_count, 0) << "No threads should encounter errors";
}

// ============================================================================
// Regression Tests
// ============================================================================

/**
 * @brief Test that known good shapes produce reasonable configs
 */
TEST_F(Test__CudaGemmAutoTuner, KnownGoodShapes)
{
    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    // Qwen 0.5B single token (896×896)
    auto config_05b = tuner.getOptimalConfig(1, 896, 896);
    EXPECT_TRUE(config_05b.tile_m >= 16 && config_05b.tile_m <= 64) << "0.5B should use reasonable tile_m";
    EXPECT_TRUE(config_05b.tile_n >= 16 && config_05b.tile_n <= 64) << "0.5B should use reasonable tile_n";

    // Qwen 14B single token (5120×5120)
    auto config_14b = tuner.getOptimalConfig(1, 5120, 5120);
    EXPECT_TRUE(config_14b.tile_m >= 16 && config_14b.tile_m <= 128) << "14B should use reasonable tile_m";
    EXPECT_TRUE(config_14b.tile_n >= 16 && config_14b.tile_n <= 128) << "14B should use reasonable tile_n";

    // Batch processing should use larger tiles
    auto config_batch = tuner.getOptimalConfig(128, 2048, 2048);
    EXPECT_TRUE(config_batch.tile_m >= 32 || config_batch.tile_n >= 32)
        << "Batch processing should use larger tiles";
}

/**
 * @brief Test that config IDs are unique and stable
 */
TEST_F(Test__CudaGemmAutoTuner, ConfigIDsAreStable)
{
    auto &tuner = CudaGemmAutoTuner::instance();
    tuner.setAutoTuningEnabled(false);

    auto config1 = tuner.getOptimalConfig(1, 1280, 1280);
    std::string id1 = config1.id();

    // Clear cache and get config again
    tuner.clearCache();
    auto config2 = tuner.getOptimalConfig(1, 1280, 1280);
    std::string id2 = config2.id();

    // Same shape should produce same config ID (deterministic heuristic)
    EXPECT_EQ(id1, id2) << "Config IDs should be deterministic for same shape";

    // ID should contain recognizable config parameters
    EXPECT_TRUE(id1.find("tile_") != std::string::npos) << "Config ID should contain 'tile_'";
    EXPECT_TRUE(id1.find("threads_") != std::string::npos) << "Config ID should contain 'threads_'";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
