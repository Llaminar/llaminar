/**
 * @file Test__StreamingEnv.cpp
 * @brief Unit tests for StreamingEnv environment variable parsing
 * @author GitHub Copilot
 * @date January 2026
 *
 * Tests the weight streaming configuration system:
 * - StreamingEnv environment variable parsing
 * - createStreamingConfigFromEnv() helper function
 * - Edge cases and validation
 *
 * @see DebugEnv.h for StreamingEnv struct
 * @see StreamingConfigFromEnv.h for createStreamingConfigFromEnv()
 */

#include <gtest/gtest.h>
#include "utils/DebugEnv.h"
#include "loaders/StreamingConfigFromEnv.h"
#include <cstdlib>

namespace llaminar2::test
{

    // =============================================================================
    // StreamingEnv Tests
    // =============================================================================

    class StreamingEnvTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Save original env vars (from first test in suite only - 
            // subsequent SetUp calls get already-cleared state)
            if (!saved_once_)
            {
                saved_enabled_ = getEnvSafe("LLAMINAR_WEIGHT_STREAMING");
                saved_memory_ = getEnvSafe("LLAMINAR_STREAM_MEMORY_MB");
                saved_prefetch_ = getEnvSafe("LLAMINAR_STREAM_PREFETCH_DEPTH");
                saved_policy_ = getEnvSafe("LLAMINAR_STREAM_EVICTION_POLICY");
                saved_verbose_ = getEnvSafe("LLAMINAR_STREAM_VERBOSE");
                saved_once_ = true;
            }

            // Clear env vars for clean test state
            clearEnvVars();
            
            // Also reload DebugEnv to clear any cached state
            mutableDebugEnv().reload();
        }

        void TearDown() override
        {
            // Clear env vars (for next test isolation)
            clearEnvVars();
            
            // Reload debug env to restore original state
            mutableDebugEnv().reload();
        }
        
        // Static cleanup to restore original env at suite end
        static void TearDownTestSuite()
        {
            // Restore original env vars
            restoreEnv("LLAMINAR_WEIGHT_STREAMING", saved_enabled_);
            restoreEnv("LLAMINAR_STREAM_MEMORY_MB", saved_memory_);
            restoreEnv("LLAMINAR_STREAM_PREFETCH_DEPTH", saved_prefetch_);
            restoreEnv("LLAMINAR_STREAM_EVICTION_POLICY", saved_policy_);
            restoreEnv("LLAMINAR_STREAM_VERBOSE", saved_verbose_);
            mutableDebugEnv().reload();
        }

        static std::string getEnvSafe(const char *name)
        {
            const char *val = std::getenv(name);
            return val ? val : "";
        }

        static void clearEnvVars()
        {
            unsetenv("LLAMINAR_WEIGHT_STREAMING");
            unsetenv("LLAMINAR_STREAM_MEMORY_MB");
            unsetenv("LLAMINAR_STREAM_PREFETCH_DEPTH");
            unsetenv("LLAMINAR_STREAM_EVICTION_POLICY");
            unsetenv("LLAMINAR_STREAM_VERBOSE");
        }

        static void restoreEnv(const char *name, const std::string &value)
        {
            if (value.empty())
            {
                unsetenv(name);
            }
            else
            {
                setenv(name, value.c_str(), 1);
            }
        }

        // Helper to get a fresh config with current env vars
        StreamingEnv freshConfig()
        {
            StreamingEnv cfg;
            cfg.reload();
            return cfg;
        }

    private:
        static std::string saved_enabled_;
        static std::string saved_memory_;
        static std::string saved_prefetch_;
        static std::string saved_policy_;
        static std::string saved_verbose_;
        static bool saved_once_;
    };
    
    // Static member definitions
    std::string StreamingEnvTest::saved_enabled_;
    std::string StreamingEnvTest::saved_memory_;
    std::string StreamingEnvTest::saved_prefetch_;
    std::string StreamingEnvTest::saved_policy_;
    std::string StreamingEnvTest::saved_verbose_;
    bool StreamingEnvTest::saved_once_ = false;

    // =============================================================================
    // Default Value Tests
    // =============================================================================

    TEST_F(StreamingEnvTest, DefaultValues_DisabledByDefault)
    {
        auto cfg = freshConfig();

        EXPECT_FALSE(cfg.enabled) << "Streaming should be disabled by default";
        EXPECT_EQ(cfg.memory_budget_mb, 0u) << "Memory budget should be 0 (auto) by default";
        EXPECT_EQ(cfg.prefetch_depth, 1) << "Prefetch depth should be 1 by default";
        EXPECT_EQ(cfg.eviction_policy, "lru") << "Eviction policy should be 'lru' by default";
        EXPECT_FALSE(cfg.verbose) << "Verbose logging should be disabled by default";
    }

    // =============================================================================
    // LLAMINAR_WEIGHT_STREAMING Tests
    // =============================================================================

    TEST_F(StreamingEnvTest, Enabled_ParsesTrue)
    {
        setenv("LLAMINAR_WEIGHT_STREAMING", "1", 1);
        auto cfg = freshConfig();

        EXPECT_TRUE(cfg.enabled);
    }

    TEST_F(StreamingEnvTest, Enabled_ParsesFalse)
    {
        setenv("LLAMINAR_WEIGHT_STREAMING", "0", 1);
        auto cfg = freshConfig();

        EXPECT_FALSE(cfg.enabled);
    }

    TEST_F(StreamingEnvTest, Enabled_NonZeroIsTrue)
    {
        setenv("LLAMINAR_WEIGHT_STREAMING", "42", 1);
        auto cfg = freshConfig();

        EXPECT_TRUE(cfg.enabled);
    }

    // =============================================================================
    // LLAMINAR_STREAM_MEMORY_MB Tests
    // =============================================================================

    TEST_F(StreamingEnvTest, MemoryBudget_ParsesValue)
    {
        setenv("LLAMINAR_STREAM_MEMORY_MB", "4096", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.memory_budget_mb, 4096u);
    }

    TEST_F(StreamingEnvTest, MemoryBudget_ParsesZero)
    {
        setenv("LLAMINAR_STREAM_MEMORY_MB", "0", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.memory_budget_mb, 0u);
    }

    TEST_F(StreamingEnvTest, MemoryBudget_ParsesLargeValue)
    {
        setenv("LLAMINAR_STREAM_MEMORY_MB", "65536", 1); // 64GB
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.memory_budget_mb, 65536u);
    }

    // =============================================================================
    // LLAMINAR_STREAM_PREFETCH_DEPTH Tests
    // =============================================================================

    TEST_F(StreamingEnvTest, PrefetchDepth_ParsesValue)
    {
        setenv("LLAMINAR_STREAM_PREFETCH_DEPTH", "3", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.prefetch_depth, 3);
    }

    TEST_F(StreamingEnvTest, PrefetchDepth_ParsesZero)
    {
        setenv("LLAMINAR_STREAM_PREFETCH_DEPTH", "0", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.prefetch_depth, 0);
    }

    TEST_F(StreamingEnvTest, PrefetchDepth_NegativeClampsToZero)
    {
        setenv("LLAMINAR_STREAM_PREFETCH_DEPTH", "-1", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.prefetch_depth, 0) << "Negative prefetch depth should clamp to 0";
    }

    // =============================================================================
    // LLAMINAR_STREAM_EVICTION_POLICY Tests
    // =============================================================================

    TEST_F(StreamingEnvTest, EvictionPolicy_ParsesLRU)
    {
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "lru", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.eviction_policy, "lru");
    }

    TEST_F(StreamingEnvTest, EvictionPolicy_ParsesFIFO)
    {
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "fifo", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.eviction_policy, "fifo");
    }

    TEST_F(StreamingEnvTest, EvictionPolicy_ParsesNone)
    {
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "none", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.eviction_policy, "none");
    }

    TEST_F(StreamingEnvTest, EvictionPolicy_CaseInsensitive)
    {
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "LRU", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.eviction_policy, "lru") << "Policy should be normalized to lowercase";
    }

    TEST_F(StreamingEnvTest, EvictionPolicy_MixedCase)
    {
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "FiFo", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.eviction_policy, "fifo");
    }

    TEST_F(StreamingEnvTest, EvictionPolicy_InvalidKeepsDefault)
    {
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "invalid_policy", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.eviction_policy, "lru") << "Invalid policy should keep default 'lru'";
    }

    TEST_F(StreamingEnvTest, EvictionPolicy_EmptyKeepsDefault)
    {
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "", 1);
        auto cfg = freshConfig();

        EXPECT_EQ(cfg.eviction_policy, "lru") << "Empty policy should keep default 'lru'";
    }

    // =============================================================================
    // LLAMINAR_STREAM_VERBOSE Tests
    // =============================================================================

    TEST_F(StreamingEnvTest, Verbose_ParsesTrue)
    {
        setenv("LLAMINAR_STREAM_VERBOSE", "1", 1);
        auto cfg = freshConfig();

        EXPECT_TRUE(cfg.verbose);
    }

    TEST_F(StreamingEnvTest, Verbose_ParsesFalse)
    {
        setenv("LLAMINAR_STREAM_VERBOSE", "0", 1);
        auto cfg = freshConfig();

        EXPECT_FALSE(cfg.verbose);
    }

    // =============================================================================
    // Combined Configuration Tests
    // =============================================================================

    TEST_F(StreamingEnvTest, FullConfiguration)
    {
        setenv("LLAMINAR_WEIGHT_STREAMING", "1", 1);
        setenv("LLAMINAR_STREAM_MEMORY_MB", "8192", 1);
        setenv("LLAMINAR_STREAM_PREFETCH_DEPTH", "2", 1);
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "fifo", 1);
        setenv("LLAMINAR_STREAM_VERBOSE", "1", 1);

        auto cfg = freshConfig();

        EXPECT_TRUE(cfg.enabled);
        EXPECT_EQ(cfg.memory_budget_mb, 8192u);
        EXPECT_EQ(cfg.prefetch_depth, 2);
        EXPECT_EQ(cfg.eviction_policy, "fifo");
        EXPECT_TRUE(cfg.verbose);
    }

    // =============================================================================
    // DebugEnv Integration Tests
    // =============================================================================

    TEST_F(StreamingEnvTest, DebugEnv_ContainsStreamingMember)
    {
        setenv("LLAMINAR_WEIGHT_STREAMING", "1", 1);
        setenv("LLAMINAR_STREAM_MEMORY_MB", "2048", 1);

        mutableDebugEnv().reload();
        const auto &env = debugEnv();

        EXPECT_TRUE(env.streaming.enabled);
        EXPECT_EQ(env.streaming.memory_budget_mb, 2048u);
    }

    TEST_F(StreamingEnvTest, DebugEnv_ReloadUpdatesStreaming)
    {
        // Initial state with cleared vars (SetUp already cleared)
        EXPECT_FALSE(debugEnv().streaming.enabled);

        // Change env and reload
        setenv("LLAMINAR_WEIGHT_STREAMING", "1", 1);
        mutableDebugEnv().reload();

        EXPECT_TRUE(debugEnv().streaming.enabled);
    }

    // =============================================================================
    // createStreamingConfigFromEnv() Tests
    // =============================================================================

    TEST_F(StreamingEnvTest, CreateConfig_DefaultValues)
    {
        // SetUp already cleared vars and reloaded DebugEnv
        auto config = createStreamingConfigFromEnv();

        EXPECT_EQ(config.gpu_memory_budget, 0u) << "0 = auto-detect";
        EXPECT_EQ(config.prefetch_depth, 1u);
        EXPECT_TRUE(config.enable_prefetch) << "Prefetch enabled when depth > 0";
        EXPECT_EQ(config.eviction_policy, StreamingEvictionPolicy::LRU);
        EXPECT_FALSE(config.log_transfer_stats);
    }

    TEST_F(StreamingEnvTest, CreateConfig_MemoryConversion)
    {
        setenv("LLAMINAR_STREAM_MEMORY_MB", "4096", 1);
        mutableDebugEnv().reload();

        auto config = createStreamingConfigFromEnv();

        // 4096 MB = 4096 * 1024 * 1024 bytes
        // Use size_t to avoid overflow
        size_t expected = static_cast<size_t>(4096) * 1024 * 1024;
        EXPECT_EQ(config.gpu_memory_budget, expected);
    }

    TEST_F(StreamingEnvTest, CreateConfig_PrefetchDisabledWhenZero)
    {
        setenv("LLAMINAR_STREAM_PREFETCH_DEPTH", "0", 1);
        mutableDebugEnv().reload();

        auto config = createStreamingConfigFromEnv();

        EXPECT_EQ(config.prefetch_depth, 0u);
        EXPECT_FALSE(config.enable_prefetch) << "Prefetch disabled when depth = 0";
    }

    TEST_F(StreamingEnvTest, CreateConfig_EvictionPolicyFIFO)
    {
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "fifo", 1);
        mutableDebugEnv().reload();

        auto config = createStreamingConfigFromEnv();

        EXPECT_EQ(config.eviction_policy, StreamingEvictionPolicy::FIFO);
    }

    TEST_F(StreamingEnvTest, CreateConfig_EvictionPolicyNone)
    {
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "none", 1);
        mutableDebugEnv().reload();

        auto config = createStreamingConfigFromEnv();

        EXPECT_EQ(config.eviction_policy, StreamingEvictionPolicy::NONE);
    }

    TEST_F(StreamingEnvTest, CreateConfig_VerboseEnablesLogging)
    {
        setenv("LLAMINAR_STREAM_VERBOSE", "1", 1);
        mutableDebugEnv().reload();

        auto config = createStreamingConfigFromEnv();

        EXPECT_TRUE(config.log_transfer_stats);
    }

    TEST_F(StreamingEnvTest, CreateConfig_FullConfiguration)
    {
        setenv("LLAMINAR_WEIGHT_STREAMING", "1", 1);
        setenv("LLAMINAR_STREAM_MEMORY_MB", "16384", 1); // 16 GB
        setenv("LLAMINAR_STREAM_PREFETCH_DEPTH", "3", 1);
        setenv("LLAMINAR_STREAM_EVICTION_POLICY", "lru", 1);
        setenv("LLAMINAR_STREAM_VERBOSE", "1", 1);
        mutableDebugEnv().reload();

        auto config = createStreamingConfigFromEnv();

        // Use size_t to avoid overflow
        size_t expected = static_cast<size_t>(16384) * 1024 * 1024;
        EXPECT_EQ(config.gpu_memory_budget, expected);
        EXPECT_EQ(config.prefetch_depth, 3u);
        EXPECT_TRUE(config.enable_prefetch);
        EXPECT_EQ(config.eviction_policy, StreamingEvictionPolicy::LRU);
        EXPECT_TRUE(config.log_transfer_stats);
    }

    // =============================================================================
    // isWeightStreamingEnabled() Tests
    // =============================================================================

    TEST_F(StreamingEnvTest, IsEnabled_ReturnsFalseByDefault)
    {
        // SetUp already cleared env vars and reloaded
        EXPECT_FALSE(isWeightStreamingEnabled());
    }

    TEST_F(StreamingEnvTest, IsEnabled_ReturnsTrueWhenSet)
    {
        setenv("LLAMINAR_WEIGHT_STREAMING", "1", 1);
        mutableDebugEnv().reload();

        EXPECT_TRUE(isWeightStreamingEnabled());
    }

} // namespace llaminar2::test
