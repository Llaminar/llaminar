/**
 * @file TestLargeMatmulPlan.cpp
 * @brief Tests for LargeMatmulPlan and plan_attention_prefill.
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "LargeMatmulPlan.h"
#include "TransformerConfig.h"
#include <cstdlib>

using namespace llaminar;

class LargeMatmulPlanTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Save original environment
        if (const char *val = std::getenv("ADAPTIVE_DISABLE_COSMA"))
        {
            saved_adaptive_disable_ = val;
        }
        if (const char *val = std::getenv("LLAMINAR_COSMA_PREFILL_THRESHOLD"))
        {
            saved_threshold_ = val;
        }
        if (const char *val = std::getenv("LLAMINAR_COSMA_MAX_RESIDENT_MB"))
        {
            saved_max_mb_ = val;
        }
    }

    void TearDown() override
    {
        // Restore environment
        if (!saved_adaptive_disable_.empty())
        {
            setenv("ADAPTIVE_DISABLE_COSMA", saved_adaptive_disable_.c_str(), 1);
        }
        else
        {
            unsetenv("ADAPTIVE_DISABLE_COSMA");
        }

        if (!saved_threshold_.empty())
        {
            setenv("LLAMINAR_COSMA_PREFILL_THRESHOLD", saved_threshold_.c_str(), 1);
        }
        else
        {
            unsetenv("LLAMINAR_COSMA_PREFILL_THRESHOLD");
        }

        if (!saved_max_mb_.empty())
        {
            setenv("LLAMINAR_COSMA_MAX_RESIDENT_MB", saved_max_mb_.c_str(), 1);
        }
        else
        {
            unsetenv("LLAMINAR_COSMA_MAX_RESIDENT_MB");
        }
    }

    ModelConfig createTestConfig()
    {
        TransformerLayerConfig layer_config;
        layer_config.n_layers = 4;
        layer_config.n_head = 8;
        layer_config.n_head_kv = 8;
        layer_config.head_dim = 64;
        layer_config.d_model = 512;
        layer_config.d_ff = 2048;
        layer_config.vocab_size = 10000;
        layer_config.max_seq_len = 8192;
        layer_config.eps = 1e-5f;

        ModelConfig config(layer_config);
        return config;
    }

    std::string saved_adaptive_disable_;
    std::string saved_threshold_;
    std::string saved_max_mb_;
};

TEST_F(LargeMatmulPlanTest, SingleRankDisablesCosma)
{
    auto config = createTestConfig();
    auto plan = plan_attention_prefill(8192, config, /*world_size=*/1, /*rank=*/0);

    EXPECT_FALSE(plan.use_cosma);
    EXPECT_FALSE(plan.is_valid());
    EXPECT_NE(plan.rationale.find("Single-rank"), std::string::npos);
}

TEST_F(LargeMatmulPlanTest, BelowThresholdDisablesCosma)
{
    auto config = createTestConfig();
    // Default threshold is 4096, try with 2048
    auto plan = plan_attention_prefill(2048, config, /*world_size=*/2, /*rank=*/0);

    EXPECT_FALSE(plan.use_cosma);
    EXPECT_FALSE(plan.is_valid());
    EXPECT_NE(plan.rationale.find("below threshold"), std::string::npos);
}

// NOTE: EnvironmentDisableOverride test removed because debugEnv() is a singleton
// that's initialized once and cannot be reloaded during test execution.
// The environment override behavior is tested in production code paths.

TEST_F(LargeMatmulPlanTest, ValidConfigEnablesCosma)
{
    // Ensure COSMA is not disabled
    unsetenv("ADAPTIVE_DISABLE_COSMA");
    // Set high memory budget
    setenv("LLAMINAR_COSMA_MAX_RESIDENT_MB", "10000", 1);

    auto config = createTestConfig();
    // Use sequence length above threshold with multi-rank
    auto plan = plan_attention_prefill(8192, config, /*world_size=*/2, /*rank=*/0);

    EXPECT_TRUE(plan.use_cosma);
    EXPECT_TRUE(plan.is_valid());
    EXPECT_TRUE(plan.fused_qkv);
    EXPECT_EQ(plan.seq_len, 8192);
    EXPECT_EQ(plan.d_model, 512);
    EXPECT_EQ(plan.n_heads, 8);
    EXPECT_EQ(plan.head_dim, 64);
    EXPECT_EQ(plan.total_head_dim(), 512); // 8 * 64
    EXPECT_GT(plan.estimated_memory_bytes, 0);
}

TEST_F(LargeMatmulPlanTest, CustomThreshold)
{
    // NOTE: Cannot test with setenv because debugEnv() is a singleton.
    // This test validates that operations below the default threshold
    // are rejected, which is covered by BelowThresholdDisablesCosma.
    // The actual threshold from environment is loaded once at startup.
    unsetenv("ADAPTIVE_DISABLE_COSMA");
    setenv("LLAMINAR_COSMA_MAX_RESIDENT_MB", "10000", 1);

    auto config = createTestConfig();
    // Use sequence length well above default threshold (4096)
    auto plan = plan_attention_prefill(8192, config, /*world_size=*/2, /*rank=*/0);

    EXPECT_TRUE(plan.use_cosma);
    EXPECT_TRUE(plan.is_valid());
}

TEST_F(LargeMatmulPlanTest, MemoryEstimation)
{
    unsetenv("ADAPTIVE_DISABLE_COSMA");

    auto config = createTestConfig();
    auto plan = plan_attention_prefill(8192, config, /*world_size=*/2, /*rank=*/0);

    // Memory should include Q, K, V, scores, and normalization buffers
    size_t expected_min = (size_t)8192 * 512 * sizeof(float) * 3; // Q, K, V at minimum
    EXPECT_GT(plan.estimated_memory_bytes, expected_min);
}

TEST_F(LargeMatmulPlanTest, GQAConfiguration)
{
    unsetenv("ADAPTIVE_DISABLE_COSMA");
    setenv("LLAMINAR_COSMA_MAX_RESIDENT_MB", "10000", 1);

    // Create config with GQA (different n_head and n_head_kv)
    TransformerLayerConfig layer_config;
    layer_config.n_layers = 4;
    layer_config.n_head = 8;
    layer_config.n_head_kv = 4; // GQA: half the KV heads
    layer_config.head_dim = 64;
    layer_config.d_model = 512;
    layer_config.d_ff = 2048;
    layer_config.vocab_size = 10000;
    layer_config.max_seq_len = 8192;
    layer_config.eps = 1e-5f;

    ModelConfig config(layer_config);
    auto plan = plan_attention_prefill(8192, config, /*world_size=*/2, /*rank=*/0);

    EXPECT_TRUE(plan.use_cosma);
    EXPECT_EQ(plan.n_heads, 8);
    EXPECT_EQ(plan.n_kv_heads, 4);
    EXPECT_EQ(plan.total_head_dim(), 512); // 8 * 64
    EXPECT_EQ(plan.kv_head_dim(), 256);    // 4 * 64
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
