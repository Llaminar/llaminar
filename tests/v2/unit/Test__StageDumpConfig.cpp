/**
 * @file Test__StageDumpConfig.cpp
 * @brief Unit tests for StageDumpConfig environment variable parsing and filtering
 *
 * Tests the new LLAMINAR_STAGE_DUMP_NAMES feature for stage name-based filtering.
 */

#include <gtest/gtest.h>
#include "utils/DebugEnv.h"
#include <cstdlib>

namespace llaminar2
{
    namespace test
    {

        class StageDumpConfigTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Clear all stage dump environment variables before each test
                unsetenv("LLAMINAR_STAGE_DUMP_ENABLED");
                unsetenv("LLAMINAR_STAGE_DUMP_DIR");
                unsetenv("LLAMINAR_STAGE_DUMP_TYPES");
                unsetenv("LLAMINAR_STAGE_DUMP_NAMES");
                unsetenv("LLAMINAR_STAGE_DUMP_LAYERS");
                unsetenv("LLAMINAR_STAGE_DUMP_ITERATION");
                unsetenv("LLAMINAR_STAGE_DUMP_RANK");
                unsetenv("LLAMINAR_STAGE_DUMP_MAX");
                unsetenv("LLAMINAR_STAGE_DUMP_INPUTS");
                unsetenv("LLAMINAR_STAGE_DUMP_OUTPUTS");
                unsetenv("LLAMINAR_STAGE_DUMP_WEIGHTS");
            }

            void TearDown() override
            {
                // Clean up environment after tests
                SetUp();
            }

            // Helper to reload the config
            StageDumpConfig freshConfig()
            {
                StageDumpConfig cfg;
                cfg.reload();
                return cfg;
            }
        };

        // =============================================================================
        // Basic Enable/Disable Tests
        // =============================================================================

        TEST_F(StageDumpConfigTest, DisabledByDefault)
        {
            auto cfg = freshConfig();
            EXPECT_FALSE(cfg.enabled);
        }

        TEST_F(StageDumpConfigTest, CanBeEnabled)
        {
            setenv("LLAMINAR_STAGE_DUMP_ENABLED", "1", 1);
            auto cfg = freshConfig();
            EXPECT_TRUE(cfg.enabled);
        }

        TEST_F(StageDumpConfigTest, EnabledZeroIsDisabled)
        {
            setenv("LLAMINAR_STAGE_DUMP_ENABLED", "0", 1);
            auto cfg = freshConfig();
            EXPECT_FALSE(cfg.enabled);
        }

        // =============================================================================
        // Stage Name Filtering Tests (New Feature)
        // =============================================================================

        TEST_F(StageDumpConfigTest, DumpAllNamesByDefault)
        {
            auto cfg = freshConfig();
            EXPECT_TRUE(cfg.dump_all_names);
            EXPECT_TRUE(cfg.dump_names.empty());
        }

        TEST_F(StageDumpConfigTest, SingleNameFilter)
        {
            setenv("LLAMINAR_STAGE_DUMP_NAMES", "layer0_attention", 1);
            auto cfg = freshConfig();

            EXPECT_FALSE(cfg.dump_all_names);
            EXPECT_EQ(cfg.dump_names.size(), 1u);
            EXPECT_TRUE(cfg.dump_names.count("layer0_attention") > 0);
        }

        TEST_F(StageDumpConfigTest, MultipleNameFilter)
        {
            setenv("LLAMINAR_STAGE_DUMP_NAMES", "layer0_attention,layer1_ffn,embed", 1);
            auto cfg = freshConfig();

            EXPECT_FALSE(cfg.dump_all_names);
            EXPECT_EQ(cfg.dump_names.size(), 3u);
            EXPECT_TRUE(cfg.dump_names.count("layer0_attention") > 0);
            EXPECT_TRUE(cfg.dump_names.count("layer1_ffn") > 0);
            EXPECT_TRUE(cfg.dump_names.count("embed") > 0);
        }

        TEST_F(StageDumpConfigTest, ExplicitAllNames)
        {
            setenv("LLAMINAR_STAGE_DUMP_NAMES", "all", 1);
            auto cfg = freshConfig();

            EXPECT_TRUE(cfg.dump_all_names);
            EXPECT_TRUE(cfg.dump_names.empty());
        }

        TEST_F(StageDumpConfigTest, ShouldDumpNameWithFilter)
        {
            setenv("LLAMINAR_STAGE_DUMP_NAMES", "layer0_attention,layer2_attention", 1);
            auto cfg = freshConfig();

            EXPECT_TRUE(cfg.shouldDumpName("layer0_attention"));
            EXPECT_TRUE(cfg.shouldDumpName("layer2_attention"));
            EXPECT_FALSE(cfg.shouldDumpName("layer1_attention"));
            EXPECT_FALSE(cfg.shouldDumpName("layer0_ffn"));
        }

        TEST_F(StageDumpConfigTest, ShouldDumpNameWithSubstringMatch)
        {
            // Filter by partial name - should match any stage containing the substring
            setenv("LLAMINAR_STAGE_DUMP_NAMES", "fused_attn_wo", 1);
            auto cfg = freshConfig();

            // Substring "fused_attn_wo" should match these node names
            EXPECT_TRUE(cfg.shouldDumpName("layer0_fused_attn_wo"));
            EXPECT_TRUE(cfg.shouldDumpName("layer5_fused_attn_wo"));
            EXPECT_TRUE(cfg.shouldDumpName("layer23_fused_attn_wo"));
            EXPECT_TRUE(cfg.shouldDumpName("prefill_fused_attn_wo"));

            // Should NOT match these
            EXPECT_FALSE(cfg.shouldDumpName("layer0_attention"));
            EXPECT_FALSE(cfg.shouldDumpName("layer0_qkv_proj"));
            EXPECT_FALSE(cfg.shouldDumpName("layer0_ffn_norm"));
        }

        TEST_F(StageDumpConfigTest, ShouldDumpNameWithLayerSubstringMatch)
        {
            // Filter by layer prefix - dump all layer0 stages
            setenv("LLAMINAR_STAGE_DUMP_NAMES", "layer0_", 1);
            auto cfg = freshConfig();

            EXPECT_TRUE(cfg.shouldDumpName("layer0_fused_attn_wo"));
            EXPECT_TRUE(cfg.shouldDumpName("layer0_attention"));
            EXPECT_TRUE(cfg.shouldDumpName("layer0_qkv_proj"));
            EXPECT_TRUE(cfg.shouldDumpName("layer0_ffn_norm"));

            EXPECT_FALSE(cfg.shouldDumpName("layer1_attention"));
            EXPECT_FALSE(cfg.shouldDumpName("layer10_attention")); // layer10 != layer0_
            EXPECT_FALSE(cfg.shouldDumpName("embedding"));
        }

        TEST_F(StageDumpConfigTest, ShouldDumpNameWithMultipleSubstrings)
        {
            // Multiple substring filters
            setenv("LLAMINAR_STAGE_DUMP_NAMES", "fused_attn_wo,ffn_norm", 1);
            auto cfg = freshConfig();

            EXPECT_TRUE(cfg.shouldDumpName("layer0_fused_attn_wo"));
            EXPECT_TRUE(cfg.shouldDumpName("layer5_ffn_norm"));

            EXPECT_FALSE(cfg.shouldDumpName("layer0_qkv_proj"));
            EXPECT_FALSE(cfg.shouldDumpName("embedding"));
        }

        TEST_F(StageDumpConfigTest, ShouldDumpNameWithoutFilter)
        {
            // No NAMES env var set - should dump all names
            auto cfg = freshConfig();

            EXPECT_TRUE(cfg.shouldDumpName("layer0_attention"));
            EXPECT_TRUE(cfg.shouldDumpName("any_stage_name"));
            EXPECT_TRUE(cfg.shouldDumpName(""));
        }

        TEST_F(StageDumpConfigTest, NameFilterWithWhitespace)
        {
            setenv("LLAMINAR_STAGE_DUMP_NAMES", "  layer0_attention , layer1_ffn  ", 1);
            auto cfg = freshConfig();

            EXPECT_FALSE(cfg.dump_all_names);
            EXPECT_EQ(cfg.dump_names.size(), 2u);
            EXPECT_TRUE(cfg.shouldDumpName("layer0_attention"));
            EXPECT_TRUE(cfg.shouldDumpName("layer1_ffn"));
        }

        // =============================================================================
        // Full shouldDump Integration Tests
        // =============================================================================

        TEST_F(StageDumpConfigTest, ShouldDumpWithNameAndTypeFilter)
        {
            setenv("LLAMINAR_STAGE_DUMP_ENABLED", "1", 1);
            setenv("LLAMINAR_STAGE_DUMP_TYPES", "FUSED_ATTENTION_WO", 1);
            setenv("LLAMINAR_STAGE_DUMP_NAMES", "layer0_attention", 1);
            setenv("LLAMINAR_STAGE_DUMP_LAYERS", "0", 1);
            auto cfg = freshConfig();

            // Should pass: correct type, name, layer
            EXPECT_TRUE(cfg.shouldDump("FUSED_ATTENTION_WO", "layer0_attention", 0, -1, 0));

            // Should fail: wrong name
            EXPECT_FALSE(cfg.shouldDump("FUSED_ATTENTION_WO", "layer1_attention", 0, -1, 0));

            // Should fail: wrong type
            EXPECT_FALSE(cfg.shouldDump("GEMM", "layer0_attention", 0, -1, 0));

            // Should fail: wrong layer
            EXPECT_FALSE(cfg.shouldDump("FUSED_ATTENTION_WO", "layer0_attention", 1, -1, 0));
        }

        TEST_F(StageDumpConfigTest, ShouldDumpWhenDisabled)
        {
            // Not enabled
            setenv("LLAMINAR_STAGE_DUMP_NAMES", "layer0_attention", 1);
            auto cfg = freshConfig();

            EXPECT_FALSE(cfg.shouldDump("FUSED_ATTENTION_WO", "layer0_attention", 0, -1, 0));
        }

        TEST_F(StageDumpConfigTest, ShouldDumpLegacyNoNameFilter)
        {
            setenv("LLAMINAR_STAGE_DUMP_ENABLED", "1", 1);
            setenv("LLAMINAR_STAGE_DUMP_TYPES", "GEMM", 1);
            auto cfg = freshConfig();

            // Legacy call without stage name should still work (dump_all_names=true)
            EXPECT_TRUE(cfg.shouldDump("GEMM", 0, -1, 0));
            EXPECT_FALSE(cfg.shouldDump("RMS_NORM", 0, -1, 0));
        }

        // =============================================================================
        // Type Filter Tests (Existing Functionality Regression)
        // =============================================================================

        TEST_F(StageDumpConfigTest, DumpAllTypesByDefault)
        {
            auto cfg = freshConfig();
            EXPECT_TRUE(cfg.dump_all_types);
            EXPECT_TRUE(cfg.shouldDumpType("GEMM"));
            EXPECT_TRUE(cfg.shouldDumpType("ATTENTION"));
        }

        TEST_F(StageDumpConfigTest, SingleTypeFilter)
        {
            setenv("LLAMINAR_STAGE_DUMP_TYPES", "GEMM", 1);
            auto cfg = freshConfig();

            EXPECT_FALSE(cfg.dump_all_types);
            EXPECT_TRUE(cfg.shouldDumpType("GEMM"));
            EXPECT_FALSE(cfg.shouldDumpType("ATTENTION"));
        }

        // =============================================================================
        // Layer Filter Tests (Existing Functionality Regression)
        // =============================================================================

        TEST_F(StageDumpConfigTest, DumpAllLayersByDefault)
        {
            auto cfg = freshConfig();
            EXPECT_TRUE(cfg.dump_all_layers);
            EXPECT_TRUE(cfg.shouldDumpLayer(0));
            EXPECT_TRUE(cfg.shouldDumpLayer(23));
        }

        TEST_F(StageDumpConfigTest, SingleLayerFilter)
        {
            setenv("LLAMINAR_STAGE_DUMP_LAYERS", "0", 1);
            auto cfg = freshConfig();

            EXPECT_FALSE(cfg.dump_all_layers);
            EXPECT_TRUE(cfg.shouldDumpLayer(0));
            EXPECT_FALSE(cfg.shouldDumpLayer(1));
        }

        TEST_F(StageDumpConfigTest, MultipleLayerFilter)
        {
            setenv("LLAMINAR_STAGE_DUMP_LAYERS", "0,5,10", 1);
            auto cfg = freshConfig();

            EXPECT_TRUE(cfg.shouldDumpLayer(0));
            EXPECT_TRUE(cfg.shouldDumpLayer(5));
            EXPECT_TRUE(cfg.shouldDumpLayer(10));
            EXPECT_FALSE(cfg.shouldDumpLayer(1));
            EXPECT_FALSE(cfg.shouldDumpLayer(6));
        }

        // =============================================================================
        // Iteration Filter Tests
        // =============================================================================

        TEST_F(StageDumpConfigTest, DumpAllIterationsByDefault)
        {
            auto cfg = freshConfig();
            EXPECT_TRUE(cfg.dump_all_iterations);
            EXPECT_TRUE(cfg.shouldDumpIteration(-1)); // prefill
            EXPECT_TRUE(cfg.shouldDumpIteration(0));
            EXPECT_TRUE(cfg.shouldDumpIteration(100));
        }

        TEST_F(StageDumpConfigTest, SingleIterationFilter)
        {
            setenv("LLAMINAR_STAGE_DUMP_ITERATION", "0", 1);
            auto cfg = freshConfig();

            EXPECT_FALSE(cfg.dump_all_iterations);
            EXPECT_TRUE(cfg.shouldDumpIteration(0));
            EXPECT_FALSE(cfg.shouldDumpIteration(1));
            // Prefill (-1) always passes
            EXPECT_TRUE(cfg.shouldDumpIteration(-1));
        }

        // =============================================================================
        // Rank Filter Tests
        // =============================================================================

        TEST_F(StageDumpConfigTest, Rank0ByDefault)
        {
            auto cfg = freshConfig();
            EXPECT_EQ(cfg.dump_rank, 0);
            EXPECT_TRUE(cfg.shouldDumpRank(0));
            EXPECT_FALSE(cfg.shouldDumpRank(1));
        }

        TEST_F(StageDumpConfigTest, AllRanks)
        {
            setenv("LLAMINAR_STAGE_DUMP_RANK", "-1", 1);
            auto cfg = freshConfig();

            EXPECT_EQ(cfg.dump_rank, -1);
            EXPECT_TRUE(cfg.shouldDumpRank(0));
            EXPECT_TRUE(cfg.shouldDumpRank(1));
            EXPECT_TRUE(cfg.shouldDumpRank(7));
        }

    } // namespace test
} // namespace llaminar2
