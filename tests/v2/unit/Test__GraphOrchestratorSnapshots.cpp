/**
 * @file Test__GraphOrchestratorSnapshots.cpp
 * @brief Unit tests for GraphOrchestrator snapshot capture API
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the pipeline-compatible snapshot capture API added to GraphOrchestrator
 * for E2E parity testing migration.
 */

#include <gtest/gtest.h>
#include "execution/GraphOrchestrator.h"
#include <vector>
#include <string>
#include <algorithm>

using namespace llaminar2;

// =============================================================================
// Stage Name Conversion Tests
// =============================================================================

/**
 * @brief Test stage name to snapshot key conversion
 *
 * Verifies the static conversion function translates graph-style names
 * to pipeline-style snapshot keys.
 */
TEST(Test__GraphOrchestratorSnapshots, StageNameConversion_GlobalStages)
{
    // Global stage mappings
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("embedding"), "EMBEDDING");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("final_norm"), "FINAL_NORM");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("lm_head"), "LM_HEAD");
}

TEST(Test__GraphOrchestratorSnapshots, StageNameConversion_AttentionStages)
{
    // Layer 0 attention stages - using actual graph stage names from Qwen2Graph.cpp
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_attn_norm"),
              "layer0_ATTENTION_NORM");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_q_proj"),
              "layer0_Q_PROJECTION");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_k_proj"),
              "layer0_K_PROJECTION");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_v_proj"),
              "layer0_V_PROJECTION");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_q_rope"),
              "layer0_Q_ROPE");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_k_rope"),
              "layer0_K_ROPE");
    // Graph uses "attention", not "attn_compute"
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_attention"),
              "layer0_ATTENTION_CONTEXT");
    // Graph uses "wo_proj", not "attn_proj"
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_wo_proj"),
              "layer0_ATTENTION_OUTPUT");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_attn_residual"),
              "layer0_ATTENTION_RESIDUAL");
}

TEST(Test__GraphOrchestratorSnapshots, StageNameConversion_FFNStages)
{
    // Layer 0 FFN stages - using actual graph stage names from Qwen2Graph.cpp
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_ffn_norm"),
              "layer0_FFN_NORM");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_ffn_gate"),
              "layer0_FFN_GATE");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_ffn_up"),
              "layer0_FFN_UP");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_swiglu"),
              "layer0_FFN_SWIGLU");
    // Graph uses "down_proj", not "ffn_down"
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_down_proj"),
              "layer0_FFN_DOWN");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer0_ffn_residual"),
              "layer0_FFN_RESIDUAL");
}

TEST(Test__GraphOrchestratorSnapshots, StageNameConversion_MultipleLayerIndices)
{
    // Different layer indices - using actual graph stage names
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer5_q_proj"),
              "layer5_Q_PROJECTION");
    // Graph uses "down_proj", not "ffn_down"
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer23_down_proj"),
              "layer23_FFN_DOWN");
    EXPECT_EQ(GraphOrchestrator::convertStageNameToSnapshotKey("layer99_attn_norm"),
              "layer99_ATTENTION_NORM");
}

TEST(Test__GraphOrchestratorSnapshots, StageNameConversion_UnknownStage)
{
    // Unknown stages should be uppercased
    std::string result = GraphOrchestrator::convertStageNameToSnapshotKey("unknown_stage");
    EXPECT_EQ(result, "UNKNOWN_STAGE");
}

// =============================================================================
// Snapshot API State Tests
// =============================================================================

/**
 * @brief Test snapshot enable/disable state tracking
 */
TEST(Test__GraphOrchestratorSnapshots, SnapshotState_InitiallyDisabled)
{
    // Create minimal config for orchestrator
    Qwen2GraphConfig config;
    config.d_model = 896;
    config.n_layers = 24;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.d_ff = 4864;
    config.vocab_size = 151936;
    config.max_seq_len = 2048;

    GraphOrchestrator orchestrator(config);

    // Should be disabled by default
    EXPECT_FALSE(orchestrator.isSnapshotCaptureEnabled());
    EXPECT_TRUE(orchestrator.getSnapshotKeys().empty());
}

TEST(Test__GraphOrchestratorSnapshots, SnapshotState_EnableDisable)
{
    Qwen2GraphConfig config;
    config.d_model = 896;
    config.n_layers = 24;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.d_ff = 4864;
    config.vocab_size = 151936;
    config.max_seq_len = 2048;

    GraphOrchestrator orchestrator(config);

    // Enable
    orchestrator.enableSnapshotCapture();
    EXPECT_TRUE(orchestrator.isSnapshotCaptureEnabled());

    // Disable
    orchestrator.disableSnapshotCapture();
    EXPECT_FALSE(orchestrator.isSnapshotCaptureEnabled());
}

TEST(Test__GraphOrchestratorSnapshots, GetSnapshot_ReturnsNullForMissingKey)
{
    Qwen2GraphConfig config;
    config.d_model = 896;
    config.n_layers = 24;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.d_ff = 4864;
    config.vocab_size = 151936;
    config.max_seq_len = 2048;

    GraphOrchestrator orchestrator(config);
    orchestrator.enableSnapshotCapture();

    size_t size = 0;
    const float *data = orchestrator.getSnapshot("nonexistent_key", size);

    EXPECT_EQ(data, nullptr);
    EXPECT_EQ(size, 0);
}

// =============================================================================
// Summary
// =============================================================================
// These tests verify the pipeline-compatible snapshot API added to GraphOrchestrator.
// Full integration testing with actual graph execution requires model weights and
// is covered by E2E tests in tests/v2/e2e/.
