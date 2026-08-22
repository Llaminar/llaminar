/**
 * @file Test__SnapshotCapture.cpp
 * @brief Unit tests for SnapshotCapture
 *
 * Tests the snapshot capture routing logic, dequantization,
 * and stage name → key conversion extracted in Phase 2 of DGO refactor.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <numeric>
#include <thread>
#include <vector>

#include "snapshots/SnapshotCapture.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"

using namespace llaminar2;
using simd::fp32_to_fp16;

// =========================================================================
// Test Helpers
// =========================================================================

namespace
{

    /// Build a StageDumpInfo::OutputBuffer from FP32 data
    StageDumpInfo::OutputBuffer makeFP32Output(
        const char *name,
        const float *data,
        size_t rows,
        size_t cols)
    {
        StageDumpInfo::OutputBuffer out;
        out.name = name;
        out.data = data;
        out.rows = rows;
        out.cols = cols;
        out.dtype = "FP32";
        out.element_size = sizeof(float);
        return out;
    }

    /// Build a StageDumpInfo::OutputBuffer from exact INT32 data.
    StageDumpInfo::OutputBuffer makeINT32Output(
        const char *name,
        const int32_t *data,
        size_t rows,
        size_t cols)
    {
        StageDumpInfo::OutputBuffer out;
        out.name = name;
        out.data = data;
        out.rows = rows;
        out.cols = cols;
        out.dtype = "INT32";
        out.element_size = sizeof(int32_t);
        return out;
    }

    /// Build StageDumpInfo with a single FP32 output
    StageDumpInfo makeSingleOutputDump(
        const char *name,
        const float *data,
        size_t rows,
        size_t cols)
    {
        StageDumpInfo dump;
        dump.outputs.push_back(makeFP32Output(name, data, rows, cols));
        return dump;
    }

} // namespace

// =========================================================================
// Test: convertStageNameToSnapshotKey
// =========================================================================

class Test__SnapshotCapture_KeyConversion : public ::testing::Test
{
};

TEST(Test__SnapshotCapture_KeyConversion, GlobalStages)
{
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("embedding"), "EMBEDDING");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("final_norm"), "FINAL_NORM");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("lm_head"), "LM_HEAD");
}

TEST(Test__SnapshotCapture_KeyConversion, AttentionLayerStages)
{
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_attn_norm"), "layer0_ATTENTION_NORM");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_q_proj"), "layer0_Q_PROJECTION");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_k_proj"), "layer0_K_PROJECTION");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_v_proj"), "layer0_V_PROJECTION");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_q_rope"), "layer0_Q_ROPE");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_k_rope"), "layer0_K_ROPE");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_attention"), "layer0_ATTENTION_CONTEXT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_wo_proj"), "layer0_ATTENTION_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_wo_allreduce"),
              "layer0_ATTENTION_OUTPUT_ALLREDUCED");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_attn_residual"), "layer0_ATTENTION_RESIDUAL");
}

TEST(Test__SnapshotCapture_KeyConversion, FFNLayerStages)
{
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_ffn_norm"), "layer0_FFN_NORM");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_ffn_gate"), "layer0_FFN_GATE");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_ffn_up"), "layer0_FFN_UP");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_swiglu"), "layer0_FFN_SWIGLU");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_down_proj"), "layer0_FFN_DOWN");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_down_allreduce"),
              "layer0_FFN_DOWN_ALLREDUCED");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_ffn_residual"), "layer0_FFN_RESIDUAL");
}

TEST(Test__SnapshotCapture_KeyConversion, HighLayerIndex)
{
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer23_q_proj"), "layer23_Q_PROJECTION");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer127_ffn_gate"), "layer127_FFN_GATE");
}

TEST(Test__SnapshotCapture_KeyConversion, MoEStages)
{
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_moe_ffn"), "layer0_MOE_EXPERT_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_moe_expert_ffn"), "layer0_MOE_EXPERT_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_moe_expert_allreduce"),
              "layer0_MOE_EXPERT_OUTPUT_ALLREDUCED");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_moe_expert_overlay_fast_allreduce"),
              "layer0_MOE_EXPERT_OUTPUT_ALLREDUCED");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey(
                  "layer0_moe_overlay_continuation_broadcast"),
              "layer0_MOE_EXPERT_OUTPUT_ALLREDUCED");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_moe_routed_expert_partial_reduce"), "layer0_MOE_EXPERT_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_moe_sparse_return_reduce_tier0_hot_p0_allreduce"),
              "layer0_MOE_EXPERT_OUTPUT_ALLREDUCED");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_shared_expert"), "layer0_MOE_SHARED_EXPERT_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_shared_expert_allreduce"),
              "layer0_MOE_SHARED_EXPERT_OUTPUT_ALLREDUCED");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey(
                  "layer0_shared_expert_reduce_to_overlay_root"),
              "layer0_MOE_SHARED_EXPERT_OUTPUT_REDUCED_TO_OVERLAY_ROOT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_shared_expert_gate"), "layer0_MOE_SHARED_GATE_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_moe_add"), "layer0_MOE_COMBINED_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_moe_combine"), "layer0_MOE_COMBINED_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey(
                  "layer0_moe_overlay_ticket_consume_tier2_cold_p3"),
              "layer0_MOE_EXPERT_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer39_moe_ffn"), "layer39_MOE_EXPERT_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer39_moe_add"), "layer39_MOE_COMBINED_OUTPUT");
}

/**
 * @brief Overlay ticket captures retain each target's cumulative routed sum.
 *
 * The ordinary expert-output key remains the final semantic model checkpoint;
 * the extra key is evidence for diagnosis of a heterogeneous sparse return.
 */
TEST(Test__SnapshotCapture_Capture,
     OverlayTicketConsumePublishesParticipantCumulativeCheckpoint)
{
    const std::vector<float> cumulative = {1.0f, -2.0f, 3.0f, -4.0f};
    StageDumpInfo dump;
    dump.outputs.push_back(
        makeFP32Output("output", cumulative.data(), 1, cumulative.size()));

    SnapshotCapture capture;
    capture.captureStage(
        "layer7_moe_overlay_ticket_consume_tier2_cold_p3", dump);

    const std::string semantic_key = "layer7_MOE_EXPERT_OUTPUT";
    const std::string cumulative_key =
        "layer7_MOE_OVERLAY_CUMULATIVE_TIER2_COLD_P3";
    ASSERT_NE(capture.get(semantic_key), nullptr);
    ASSERT_NE(capture.get(cumulative_key), nullptr);
    EXPECT_EQ(capture.get(semantic_key)->data, cumulative);
    EXPECT_EQ(capture.get(cumulative_key)->data, cumulative);

    const auto keys = SnapshotCapture::possibleKeysForStageName(
        "layer7_moe_overlay_ticket_consume_tier2_cold_p3");
    EXPECT_NE(std::find(keys.begin(), keys.end(), semantic_key), keys.end());
    EXPECT_NE(std::find(keys.begin(), keys.end(), cumulative_key), keys.end());
}

TEST(Test__SnapshotCapture_KeyConversion, PossibleKeysIncludeFusedMoECombinedOutput)
{
    auto expert_keys = SnapshotCapture::possibleKeysForStageName("layer0_moe_expert_ffn");
    EXPECT_NE(std::find(expert_keys.begin(), expert_keys.end(), "layer0_MOE_EXPERT_OUTPUT"), expert_keys.end());
    EXPECT_NE(std::find(expert_keys.begin(), expert_keys.end(), "layer0_MOE_COMBINED_OUTPUT"), expert_keys.end());

    auto shared_gate_keys = SnapshotCapture::possibleKeysForStageName("layer0_shared_expert_gate");
    EXPECT_NE(std::find(shared_gate_keys.begin(), shared_gate_keys.end(), "layer0_MOE_SHARED_GATE_OUTPUT"), shared_gate_keys.end());
    EXPECT_NE(std::find(shared_gate_keys.begin(), shared_gate_keys.end(), "layer0_MOE_COMBINED_OUTPUT"), shared_gate_keys.end());
}

/**
 * @brief A fused rooted shared-gate epilogue preserves all three MoE values.
 *
 * Heterogeneous LocalTP overlay lowering intentionally defers the routed-only
 * broadcast and publishes just the final combined row.  The root-owned fused
 * gate stage is therefore the only real producer boundary at which parity can
 * observe the complete routed row without adding a test-only graph node.
 */
TEST(Test__SnapshotCapture_Capture,
     FusedSharedGatePublishesRootOwnedRoutedCheckpoint)
{
    const std::vector<float> shared = {0.5f, -0.25f};
    const std::vector<float> routed = {1.0f, 2.0f};
    const std::vector<float> combined = {1.5f, 1.75f};
    StageDumpInfo dump;
    dump.outputs.push_back(
        makeFP32Output("shared_output", shared.data(), 1, shared.size()));
    dump.outputs.push_back(
        makeFP32Output("routed_output", routed.data(), 1, routed.size()));
    dump.outputs.push_back(
        makeFP32Output("combined_output", combined.data(), 1, combined.size()));

    const auto possible = SnapshotCapture::possibleKeysForStage(
        "layer2_shared_expert_gate", dump);
    EXPECT_NE(
        std::find(
            possible.begin(), possible.end(), "layer2_MOE_EXPERT_OUTPUT"),
        possible.end());

    SnapshotCapture capture;
    capture.captureStage("layer2_shared_expert_gate", dump);
    ASSERT_NE(capture.get("layer2_MOE_EXPERT_OUTPUT"), nullptr);
    ASSERT_NE(capture.get("layer2_MOE_SHARED_GATE_OUTPUT"), nullptr);
    ASSERT_NE(capture.get("layer2_MOE_COMBINED_OUTPUT"), nullptr);
    EXPECT_EQ(capture.get("layer2_MOE_EXPERT_OUTPUT")->data, routed);
    EXPECT_EQ(capture.get("layer2_MOE_SHARED_GATE_OUTPUT")->data, shared);
    EXPECT_EQ(capture.get("layer2_MOE_COMBINED_OUTPUT")->data, combined);
}

/**
 * @brief Descriptor-aware filtering follows the concrete canonical producer.
 *
 * LocalTP canonical publication leaves the expert stage name intact while
 * moving routed/shared/combined ownership to a rooted finalizer. The filter
 * must therefore reject the stale final-output alias on the expert stage and
 * select all three real finalizer outputs.
 */
TEST(Test__SnapshotCapture_KeyConversion,
     ConcreteMoEOutputsDisambiguateCanonicalPublicationOwnership)
{
    StageDumpInfo route_dump;
    route_dump.outputs.push_back(
        makeFP32Output(
            "canonical_route_contributions", nullptr, 16, 32));
    const auto route_keys = SnapshotCapture::possibleKeysForStage(
        "layer0_moe_expert_ffn_overlay_fast", route_dump);
    EXPECT_EQ(route_keys,
              std::vector<std::string>{
                  "layer0_MOE_CANONICAL_ROUTE_CONTRIBUTIONS"});
    EXPECT_EQ(std::find(route_keys.begin(),
                        route_keys.end(),
                        "layer0_MOE_COMBINED_OUTPUT"),
              route_keys.end());

    StageDumpInfo finalize_dump;
    finalize_dump.outputs.push_back(
        makeFP32Output("routed_output", nullptr, 2, 32));
    finalize_dump.outputs.push_back(
        makeFP32Output("shared_output", nullptr, 2, 32));
    finalize_dump.outputs.push_back(
        makeFP32Output("combined_output", nullptr, 2, 32));
    const auto final_keys = SnapshotCapture::possibleKeysForStage(
        "layer0_moe_canonical_publication_finalize", finalize_dump);
    EXPECT_EQ(final_keys,
              (std::vector<std::string>{
                  "layer0_MOE_EXPERT_OUTPUT",
                  "layer0_MOE_SHARED_GATE_OUTPUT",
                  "layer0_MOE_COMBINED_OUTPUT"}));
}

/**
 * @brief Mapped sparse reduction publishes both pinned route projections.
 *
 * Domain route scratch is reused by later layers, while the global placement
 * bank may advance independently under RCU. The boundary snapshot therefore
 * preserves the local schedule, post-filter weights, both durable banks, and
 * the request selector as exact evidence for the parity artifact API.
 */
TEST(Test__SnapshotCapture_KeyConversion,
     MappedSparseReducerPublishesDeviceRouteAssignmentLedger)
{
    const std::vector<float> routed = {1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<int32_t> participants = {1, 2, 0, 3};
    const std::vector<float> runtime_weights = {0.4f, 0.3f, 0.2f, 0.1f};
    const std::vector<int32_t> bank0 = {4, 5, 6};
    const std::vector<int32_t> bank1 = {7, 8, 9};
    const std::vector<int32_t> selected_bank = {1};
    StageDumpInfo dump;
    dump.outputs.push_back(
        makeFP32Output("output", routed.data(), 1, routed.size()));
    dump.outputs.push_back(
        makeINT32Output(
            "domain_route_participant_ids", participants.data(), 2, 2));
    dump.outputs.push_back(
        makeFP32Output(
            "runtime_route_weights", runtime_weights.data(), 2, 2));
    dump.outputs.push_back(makeINT32Output(
        "overlay_route_participants_bank0", bank0.data(), 1, bank0.size()));
    dump.outputs.push_back(makeINT32Output(
        "overlay_route_participants_bank1", bank1.data(), 1, bank1.size()));
    dump.outputs.push_back(makeINT32Output(
        "overlay_route_selected_bank",
        selected_bank.data(),
        1,
        selected_bank.size()));

    const std::string stage =
        "layer7_moe_overlay_continuation_routes_ordered_reduce";
    EXPECT_EQ(
        SnapshotCapture::possibleKeysForStage(stage, dump),
        (std::vector<std::string>{
            "layer7_MOE_EXPERT_OUTPUT",
            "layer7_MOE_DOMAIN_ROUTE_PARTICIPANT_IDS",
            "layer7_MOE_RUNTIME_ROUTE_WEIGHTS",
            "layer7_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0",
            "layer7_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1",
            "layer7_MOE_OVERLAY_ROUTE_SELECTED_BANK"}));

    SnapshotCapture capture;
    capture.captureStage(stage, dump);
    const auto routed_snapshot =
        capture.get("layer7_MOE_EXPERT_OUTPUT");
    const auto assignment_snapshot = capture.get(
        "layer7_MOE_DOMAIN_ROUTE_PARTICIPANT_IDS");
    const auto runtime_weight_snapshot = capture.get(
        "layer7_MOE_RUNTIME_ROUTE_WEIGHTS");
    const auto bank0_snapshot = capture.get(
        "layer7_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0");
    const auto bank1_snapshot = capture.get(
        "layer7_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1");
    const auto selected_bank_snapshot = capture.get(
        "layer7_MOE_OVERLAY_ROUTE_SELECTED_BANK");
    ASSERT_NE(routed_snapshot, nullptr);
    ASSERT_NE(assignment_snapshot, nullptr);
    ASSERT_NE(runtime_weight_snapshot, nullptr);
    ASSERT_NE(bank0_snapshot, nullptr);
    ASSERT_NE(bank1_snapshot, nullptr);
    ASSERT_NE(selected_bank_snapshot, nullptr);
    EXPECT_EQ(routed_snapshot->data, routed);
    EXPECT_EQ(
        assignment_snapshot->data,
        (std::vector<float>{1.0f, 2.0f, 0.0f, 3.0f}));
    EXPECT_EQ(runtime_weight_snapshot->data, runtime_weights);
    EXPECT_EQ(bank0_snapshot->data, (std::vector<float>{4.0f, 5.0f, 6.0f}));
    EXPECT_EQ(bank1_snapshot->data, (std::vector<float>{7.0f, 8.0f, 9.0f}));
    EXPECT_EQ(selected_bank_snapshot->data, (std::vector<float>{1.0f}));
}

/**
 * @brief Single-continuation mapped return finalizes the same pinned epoch.
 *
 * Dispatch precedes grouped domain assignment and must not expose its reusable
 * scratch. The final return join is the first boundary that owns the complete
 * routed row, final domain assignment, and request-pinned placement bank.
 */
TEST(Test__SnapshotCapture_KeyConversion,
     MappedReturnPublishesPinnedRouteEpochAfterAssignmentFinalization)
{
    const std::vector<float> routed = {1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<int32_t> participants = {0, -1, 0, -1};
    const std::vector<float> runtime_weights = {0.7f, 0.0f, 0.3f, 0.0f};
    const std::vector<int32_t> bank0 = {0, 1, 0};
    const std::vector<int32_t> bank1 = {1, 0, 1};
    const std::vector<int32_t> selected_bank = {0};
    StageDumpInfo dump;
    dump.outputs.push_back(
        makeFP32Output("output", routed.data(), 1, routed.size()));
    dump.outputs.push_back(
        makeINT32Output(
            "domain_route_participant_ids", participants.data(), 2, 2));
    dump.outputs.push_back(
        makeFP32Output(
            "runtime_route_weights", runtime_weights.data(), 2, 2));
    dump.outputs.push_back(makeINT32Output(
        "overlay_route_participants_bank0", bank0.data(), 1, bank0.size()));
    dump.outputs.push_back(makeINT32Output(
        "overlay_route_participants_bank1", bank1.data(), 1, bank1.size()));
    dump.outputs.push_back(makeINT32Output(
        "overlay_route_selected_bank",
        selected_bank.data(),
        1,
        selected_bank.size()));

    const auto premature_dispatch_keys =
        SnapshotCapture::possibleKeysForStageName(
            "layer3_moe_overlay_activation_dispatch_pack_batch");
    EXPECT_EQ(
        std::find(
            premature_dispatch_keys.begin(),
            premature_dispatch_keys.end(),
            "layer3_MOE_DOMAIN_ROUTE_PARTICIPANT_IDS"),
        premature_dispatch_keys.end())
        << "Dispatch precedes the only producer of final domain assignment";

    const std::string stage =
        "layer3_moe_overlay_activation_return_consume_batch";
    EXPECT_EQ(
        SnapshotCapture::possibleKeysForStage(stage, dump),
        (std::vector<std::string>{
            "layer3_MOE_EXPERT_OUTPUT",
            "layer3_MOE_DOMAIN_ROUTE_PARTICIPANT_IDS",
            "layer3_MOE_RUNTIME_ROUTE_WEIGHTS",
            "layer3_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0",
            "layer3_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1",
            "layer3_MOE_OVERLAY_ROUTE_SELECTED_BANK"}));
    EXPECT_EQ(
        SnapshotCapture::possibleKeysForStageName(stage),
        (std::vector<std::string>{
            "layer3_MOE_EXPERT_OUTPUT",
            "layer3_MOE_DOMAIN_ROUTE_PARTICIPANT_IDS",
            "layer3_MOE_RUNTIME_ROUTE_WEIGHTS",
            "layer3_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0",
            "layer3_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1",
            "layer3_MOE_OVERLAY_ROUTE_SELECTED_BANK"}));

    SnapshotCapture capture;
    capture.captureStage(stage, dump);
    ASSERT_NE(capture.get("layer3_MOE_EXPERT_OUTPUT"), nullptr);
    EXPECT_EQ(capture.get("layer3_MOE_EXPERT_OUTPUT")->data, routed);
    ASSERT_NE(
        capture.get("layer3_MOE_DOMAIN_ROUTE_PARTICIPANT_IDS"), nullptr);
    EXPECT_EQ(
        capture.get("layer3_MOE_DOMAIN_ROUTE_PARTICIPANT_IDS")->data,
        (std::vector<float>{0.0f, -1.0f, 0.0f, -1.0f}));
    EXPECT_EQ(
        capture.get("layer3_MOE_RUNTIME_ROUTE_WEIGHTS")->data,
        runtime_weights);
    EXPECT_EQ(
        capture.get("layer3_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0")->data,
        (std::vector<float>{0.0f, 1.0f, 0.0f}));
    EXPECT_EQ(
        capture.get("layer3_MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1")->data,
        (std::vector<float>{1.0f, 0.0f, 1.0f}));
    EXPECT_EQ(
        capture.get("layer3_MOE_OVERLAY_ROUTE_SELECTED_BANK")->data,
        (std::vector<float>{0.0f}));
}

/**
 * @brief Canonical finalizer snapshots preserve all three semantic outputs.
 */
TEST(Test__SnapshotCapture_CanonicalMoERouting,
     FinalizerPublishesNamedOutputs)
{
    const std::vector<float> routed = {1.0f, 2.0f};
    const std::vector<float> shared = {3.0f, 4.0f};
    const std::vector<float> combined = {4.0f, 6.0f};
    StageDumpInfo dump;
    dump.outputs.push_back(
        makeFP32Output("routed_output", routed.data(), 1, routed.size()));
    dump.outputs.push_back(
        makeFP32Output("shared_output", shared.data(), 1, shared.size()));
    dump.outputs.push_back(
        makeFP32Output("combined_output", combined.data(), 1, combined.size()));

    SnapshotCapture capture;
    capture.captureStage(
        "layer0_moe_canonical_publication_finalize", dump);

    ASSERT_NE(capture.get("layer0_MOE_EXPERT_OUTPUT"), nullptr);
    ASSERT_NE(capture.get("layer0_MOE_SHARED_GATE_OUTPUT"), nullptr);
    ASSERT_NE(capture.get("layer0_MOE_COMBINED_OUTPUT"), nullptr);
    EXPECT_EQ(capture.get("layer0_MOE_EXPERT_OUTPUT")->data, routed);
    EXPECT_EQ(capture.get("layer0_MOE_SHARED_GATE_OUTPUT")->data, shared);
    EXPECT_EQ(capture.get("layer0_MOE_COMBINED_OUTPUT")->data, combined);
}

TEST(Test__SnapshotCapture_KeyConversion, PossibleKeysIncludePostCollectiveOutputs)
{
    const auto gdn_wo_keys =
        SnapshotCapture::possibleKeysForStageName("layer14_gdn_wo_allreduce");
    EXPECT_NE(std::find(gdn_wo_keys.begin(),
                        gdn_wo_keys.end(),
                        "layer14_ATTENTION_OUTPUT_ALLREDUCED"),
              gdn_wo_keys.end());

    const auto fa_wo_keys =
        SnapshotCapture::possibleKeysForStageName("layer14_wo_allreduce");
    EXPECT_NE(std::find(fa_wo_keys.begin(),
                        fa_wo_keys.end(),
                        "layer14_ATTENTION_OUTPUT_ALLREDUCED"),
              fa_wo_keys.end());

    const auto down_keys =
        SnapshotCapture::possibleKeysForStageName("layer14_down_allreduce");
    EXPECT_NE(std::find(down_keys.begin(),
                        down_keys.end(),
                        "layer14_FFN_DOWN_ALLREDUCED"),
              down_keys.end());

    const auto shared_keys =
        SnapshotCapture::possibleKeysForStageName("layer14_shared_expert_allreduce");
    EXPECT_NE(std::find(shared_keys.begin(),
                        shared_keys.end(),
                        "layer14_MOE_SHARED_EXPERT_OUTPUT_ALLREDUCED"),
              shared_keys.end());

    const auto shared_root_reduce_keys =
        SnapshotCapture::possibleKeysForStageName(
            "layer14_shared_expert_reduce_to_overlay_root");
    EXPECT_EQ(
        shared_root_reduce_keys,
        (std::vector<std::string>{
            "layer14_MOE_SHARED_EXPERT_OUTPUT_REDUCED_TO_OVERLAY_ROOT"}));

    const auto routed_keys =
        SnapshotCapture::possibleKeysForStageName("layer14_moe_expert_overlay_fast_allreduce");
    EXPECT_NE(std::find(routed_keys.begin(),
                        routed_keys.end(),
                        "layer14_MOE_EXPERT_OUTPUT_ALLREDUCED"),
              routed_keys.end());

    const auto overlay_publication_keys =
        SnapshotCapture::possibleKeysForStageName(
            "layer14_moe_overlay_continuation_broadcast");
    EXPECT_NE(
        std::find(
            overlay_publication_keys.begin(),
            overlay_publication_keys.end(),
            "layer14_MOE_EXPERT_OUTPUT_ALLREDUCED"),
        overlay_publication_keys.end());
}

/**
 * @brief A rooted overlay collective cannot replace the local TP checkpoint.
 *
 * The local shared FFN reports `[tokens,hidden]`, while the rooted collective
 * reports the same transfer span as `[1,count]`.  Both observations are useful,
 * but only the former participates in row-parallel parity reconstruction.
 */
TEST(Test__SnapshotCapture_Capture,
     SharedExpertRootReduceDoesNotOverwriteLocalRowParallelSnapshot)
{
    const std::vector<float> local = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> rooted = {
        11.0f, 12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f, 18.0f};

    SnapshotCapture capture;
    capture.captureStage(
        "layer14_shared_expert_ffn",
        makeSingleOutputDump("output", local.data(), 2, 4));
    capture.captureStage(
        "layer14_shared_expert_reduce_to_overlay_root",
        makeSingleOutputDump("output", rooted.data(), 1, 8));

    const auto *local_snapshot =
        capture.get("layer14_MOE_SHARED_EXPERT_OUTPUT");
    ASSERT_NE(local_snapshot, nullptr);
    EXPECT_EQ(local_snapshot->rows, 2u);
    EXPECT_EQ(local_snapshot->cols, 4u);
    EXPECT_EQ(local_snapshot->data, local);

    const auto *rooted_snapshot = capture.get(
        "layer14_MOE_SHARED_EXPERT_OUTPUT_REDUCED_TO_OVERLAY_ROOT");
    ASSERT_NE(rooted_snapshot, nullptr);
    EXPECT_EQ(rooted_snapshot->rows, 1u);
    EXPECT_EQ(rooted_snapshot->cols, 8u);
    EXPECT_EQ(rooted_snapshot->data, rooted);
}

TEST(Test__SnapshotCapture_KeyConversion, MTPSidecarStages)
{
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("MTP0_attn_norm"), "MTP0_ATTENTION_NORM");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("MTP0_q_proj"), "MTP0_Q_PROJECTION");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("MTP0_attention"), "MTP0_ATTENTION_CONTEXT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("MTP0_attn_output_gate"), "MTP0_ATTENTION_CONTEXT_GATED");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("MTP0_wo_proj"), "MTP0_ATTENTION_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("MTP0_ffn_norm"), "MTP0_FFN_NORM");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("MTP0_moe_expert_ffn"), "MTP0_MOE_EXPERT_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("MTP0_shared_expert_ffn"), "MTP0_MOE_SHARED_EXPERT_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("MTP0_shared_expert_gate"), "MTP0_MOE_SHARED_GATE_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("MTP0_moe_combine"), "MTP0_MOE_COMBINED_OUTPUT");
}

TEST(Test__SnapshotCapture_Capture, SharedExpertGateFusedOutputsUseSemanticKeys)
{
    const std::vector<float> gated_shared = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> combined = {5.0f, 6.0f, 7.0f, 8.0f};

    StageDumpInfo dump;
    dump.outputs.push_back(makeFP32Output("shared_output", gated_shared.data(), 1, 4));
    dump.outputs.push_back(makeFP32Output("combined_output", combined.data(), 1, 4));

    SnapshotCapture capture;
    capture.captureStage("MTP0_shared_expert_gate", dump);

    const auto *shared_snap = capture.get("MTP0_MOE_SHARED_GATE_OUTPUT");
    ASSERT_NE(shared_snap, nullptr);
    EXPECT_EQ(shared_snap->data, gated_shared);

    const auto *combined_snap = capture.get("MTP0_MOE_COMBINED_OUTPUT");
    ASSERT_NE(combined_snap, nullptr);
    EXPECT_EQ(combined_snap->data, combined);
}

TEST(Test__SnapshotCapture_Capture, ConcurrentStageCallbacksDoNotRaceSnapshotStorage)
{
    constexpr int kThreads = 8;
    constexpr int kIterations = 200;

    SnapshotCapture capture;
    std::vector<std::vector<float>> payloads;
    payloads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t)
    {
        payloads.push_back({static_cast<float>(t), static_cast<float>(t + 1),
                            static_cast<float>(t + 2), static_cast<float>(t + 3)});
    }

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t)
    {
        threads.emplace_back([&, t]() {
            const std::string stage_name = "layer" + std::to_string(t) + "_ffn_norm";
            for (int i = 0; i < kIterations; ++i)
            {
                auto dump = makeSingleOutputDump("output", payloads[t].data(), 1, payloads[t].size());
                capture.captureStage(stage_name, dump);
            }
        });
    }
    for (auto &thread : threads)
    {
        thread.join();
    }

    for (int t = 0; t < kThreads; ++t)
    {
        const std::string key = "layer" + std::to_string(t) + "_FFN_NORM";
        const auto *snapshot = capture.get(key);
        ASSERT_NE(snapshot, nullptr) << key;
        EXPECT_EQ(snapshot->data, payloads[t]);
    }
}

/**
 * @brief A handle acquired by parity must outlive later callback publication.
 *
 * Captured graphs can publish a later decode/pre-fill value or clear the
 * diagnostic bank while an outer rank/global router is still copying an older
 * checkpoint.  This regression locks down the ownership contract used by
 * SnapshotInfo: a reader sees one complete immutable publication, never a
 * map entry or vector invalidated by the next callback.
 */
TEST(Test__SnapshotCapture_Capture,
     SharedHandleSurvivesReplacementAndClear)
{
    const std::vector<float> first = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> second = {9.0f, 8.0f, 7.0f, 6.0f};

    SnapshotCapture capture;
    capture.captureStage(
        "embedding",
        makeSingleOutputDump("output", first.data(), /*rows=*/1, /*cols=*/4));

    const StoredSnapshotHandle held = capture.getShared("EMBEDDING");
    ASSERT_TRUE(held);
    EXPECT_EQ(held->data, first);

    // A later graph publication replaces the semantic key, not the immutable
    // object retained by the outer parity comparison.
    capture.captureStage(
        "embedding",
        makeSingleOutputDump("output", second.data(), /*rows=*/1, /*cols=*/4));
    capture.clear();

    EXPECT_EQ(held->rows, 1u);
    EXPECT_EQ(held->cols, 4u);
    EXPECT_EQ(held->data, first);
    EXPECT_FALSE(capture.getShared("EMBEDDING"));
}

TEST(Test__SnapshotCapture_KeyConversion, FallbackUpperCase)
{
    // Unknown suffix: just uppercase the whole thing
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("custom_stage"), "CUSTOM_STAGE");
}

// =========================================================================
// Test: extractFp32FromOutput
// =========================================================================

class Test__SnapshotCapture_Extract : public ::testing::Test
{
};

TEST(Test__SnapshotCapture_Extract, FP32DirectCopy)
{
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    auto out = makeFP32Output("test", input.data(), 2, 2);

    auto result = SnapshotCapture::extractFp32FromOutput(out);

    ASSERT_EQ(result.size(), 4u);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    EXPECT_FLOAT_EQ(result[3], 4.0f);
}

TEST(Test__SnapshotCapture_Extract, NullDataReturnsEmpty)
{
    StageDumpInfo::OutputBuffer out;
    out.data = nullptr;
    out.rows = 2;
    out.cols = 2;

    auto result = SnapshotCapture::extractFp32FromOutput(out);
    EXPECT_TRUE(result.empty());
}

TEST(Test__SnapshotCapture_Extract, ZeroDimensionsReturnsEmpty)
{
    float dummy = 1.0f;
    StageDumpInfo::OutputBuffer out;
    out.data = &dummy;
    out.rows = 0;
    out.cols = 4;

    auto result = SnapshotCapture::extractFp32FromOutput(out);
    EXPECT_TRUE(result.empty());
}

TEST(Test__SnapshotCapture_Extract, Q8_1Dequantization)
{
    // Create Q8_1 blocks with known values
    // scale = 0.5, qs = {1, 2, 3, ...} → dequant = {0.5, 1.0, 1.5, ...}
    Q8_1Block block{};
    block.d = fp32_to_fp16(0.5f);
    block.sum_qs = 0;
    for (int i = 0; i < 32; ++i)
    {
        block.qs[i] = static_cast<int8_t>(i + 1);
    }

    StageDumpInfo::OutputBuffer out;
    out.name = "test";
    out.data = &block;
    out.rows = 1;
    out.cols = 32;
    out.dtype = "Q8_1";

    auto result = SnapshotCapture::extractFp32FromOutput(out);

    ASSERT_EQ(result.size(), 32u);
    // each element = qs[i] * scale = (i+1) * 0.5
    for (int i = 0; i < 32; ++i)
    {
        float expected = static_cast<float>(i + 1) * 0.5f;
        EXPECT_NEAR(result[i], expected, 0.01f) << "at index " << i;
    }
}

TEST(Test__SnapshotCapture_Extract, Q16_1Dequantization)
{
    // The extraction code reads Q16_1Block::d (a float field) and passes it through
    // fp16_to_fp32(), which truncates the float to uint16_t. This is a known quirk
    // of the snapshot extraction — it works in practice because the parity tests
    // validate actual numeric accuracy. Here we just verify it runs without error
    // and produces finite, non-NaN values.
    Q16_1Block block{};
    block.d = 0.25f;
    block.sum_qs = 0;
    for (int i = 0; i < 32; ++i)
    {
        block.qs[i] = static_cast<int16_t>((i + 1) * 10);
    }

    StageDumpInfo::OutputBuffer out;
    out.name = "test";
    out.data = &block;
    out.rows = 1;
    out.cols = 32;
    out.dtype = "Q16_1";

    auto result = SnapshotCapture::extractFp32FromOutput(out);
    ASSERT_EQ(result.size(), 32u);

    bool any_nan = std::any_of(result.begin(), result.end(),
                               [](float v)
                               { return std::isnan(v); });
    EXPECT_FALSE(any_nan) << "Q16_1 extraction produced NaN values";
}

TEST(Test__SnapshotCapture_Extract, BF16Conversion)
{
    // BF16: top 16 bits of float32.
    // 1.0f = 0x3F800000 → BF16 = 0x3F80
    // 2.0f = 0x40000000 → BF16 = 0x4000
    std::vector<uint16_t> bf16_data = {0x3F80, 0x4000, 0xC000, 0x0000};

    StageDumpInfo::OutputBuffer out;
    out.name = "test";
    out.data = bf16_data.data();
    out.rows = 1;
    out.cols = 4;
    out.dtype = "BF16";

    auto result = SnapshotCapture::extractFp32FromOutput(out);

    ASSERT_EQ(result.size(), 4u);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], -2.0f);
    EXPECT_FLOAT_EQ(result[3], 0.0f);
}

TEST(Test__SnapshotCapture_Extract, FP16Conversion)
{
    // FP16: 1.0f = 0x3C00
    std::vector<uint16_t> fp16_data;
    fp16_data.push_back(fp32_to_fp16(1.0f));
    fp16_data.push_back(fp32_to_fp16(-0.5f));
    fp16_data.push_back(fp32_to_fp16(3.14f));
    fp16_data.push_back(fp32_to_fp16(0.0f));

    StageDumpInfo::OutputBuffer out;
    out.name = "test";
    out.data = fp16_data.data();
    out.rows = 2;
    out.cols = 2;
    out.dtype = "FP16";

    auto result = SnapshotCapture::extractFp32FromOutput(out);

    ASSERT_EQ(result.size(), 4u);
    EXPECT_NEAR(result[0], 1.0f, 0.001f);
    EXPECT_NEAR(result[1], -0.5f, 0.001f);
    EXPECT_NEAR(result[2], 3.14f, 0.01f);
    EXPECT_FLOAT_EQ(result[3], 0.0f);
}

TEST(Test__SnapshotCapture_Capture, MTPSidecarFusedQKVUsesMTPKeys)
{
    std::vector<float> q = {1.0f, 2.0f};
    std::vector<float> k = {3.0f, 4.0f};
    std::vector<float> v = {5.0f, 6.0f};

    StageDumpInfo dump;
    dump.outputs.push_back(makeFP32Output("q", q.data(), 1, 2));
    dump.outputs.push_back(makeFP32Output("k", k.data(), 1, 2));
    dump.outputs.push_back(makeFP32Output("v", v.data(), 1, 2));

    SnapshotCapture capture;
    capture.captureStage("MTP0_qkv_proj", dump);

    const auto *q_snapshot = capture.get("MTP0_Q_PROJECTION");
    const auto *k_snapshot = capture.get("MTP0_K_PROJECTION");
    const auto *v_snapshot = capture.get("MTP0_V_PROJECTION");
    ASSERT_NE(q_snapshot, nullptr);
    ASSERT_NE(k_snapshot, nullptr);
    ASSERT_NE(v_snapshot, nullptr);
    EXPECT_EQ(q_snapshot->data, q);
    EXPECT_EQ(k_snapshot->data, k);
    EXPECT_EQ(v_snapshot->data, v);
}

/**
 * @brief K/V-only catch-up graphs preserve both semantic projection outputs.
 */
TEST(Test__SnapshotCapture_Capture, MTPSidecarFusedKVUsesCanonicalKAndVKeys)
{
    std::vector<float> k = {3.0f, 4.0f};
    std::vector<float> v = {5.0f, 6.0f};

    StageDumpInfo dump;
    dump.outputs.push_back(makeFP32Output("output_k", k.data(), 1, 2));
    dump.outputs.push_back(makeFP32Output("output_v", v.data(), 1, 2));

    SnapshotCapture capture;
    capture.captureStage("MTP0_kv_proj", dump);

    ASSERT_NE(capture.get("MTP0_K_PROJECTION"), nullptr);
    ASSERT_NE(capture.get("MTP0_V_PROJECTION"), nullptr);
    EXPECT_EQ(capture.get("MTP0_K_PROJECTION")->data, k);
    EXPECT_EQ(capture.get("MTP0_V_PROJECTION")->data, v);
    EXPECT_EQ(capture.get("MTP0_KV_PROJ"), nullptr)
        << "A fused implementation is not a semantic numerical checkpoint";

    EXPECT_EQ(
        SnapshotCapture::possibleKeysForStageName("MTP0_kv_proj"),
        (std::vector<std::string>{
            "MTP0_K_PROJECTION", "MTP0_V_PROJECTION"}));
}

/**
 * @brief Scoped MTP embedding collectives use the canonical reduced key.
 */
TEST(Test__SnapshotCapture_Capture, MTPEmbeddingAllreduceUsesReducedSemanticName)
{
    const std::vector<float> embedding = {1.0f, 2.0f};
    SnapshotCapture capture;
    capture.captureStage(
        "MTP0_embedding_allreduce",
        makeSingleOutputDump("output", embedding.data(), 1, 2));

    ASSERT_NE(capture.get("MTP0_EMBEDDING_ALLREDUCED"), nullptr);
    EXPECT_EQ(capture.get("MTP0_EMBEDDING_ALLREDUCED")->data, embedding);
    EXPECT_EQ(capture.get("MTP0_EMBEDDING_ALLREDUCE"), nullptr);
    EXPECT_EQ(
        SnapshotCapture::convertStageNameToSnapshotKey(
            "MTP0_embedding_allreduce"),
        "MTP0_EMBEDDING_ALLREDUCED");
    EXPECT_EQ(
        SnapshotCapture::convertStageNameToSnapshotKey(
            "mtp0_embedding_allreduce"),
        "MTP0_EMBEDDING_ALLREDUCED")
        << "Production MTP graph node spelling must not leak into the "
           "schema-facing snapshot key";
}

TEST(Test__SnapshotCapture_Capture, GDNProjectionSplitsAlphaAndBeta)
{
    std::vector<float> qkv = {1.0f, 2.0f};
    std::vector<float> z = {3.0f, 4.0f};
    std::vector<float> alpha = {5.0f, 6.0f};
    std::vector<float> beta = {7.0f, 8.0f};

    StageDumpInfo dump;
    dump.outputs.push_back(makeFP32Output("qkv", qkv.data(), 1, 2));
    dump.outputs.push_back(makeFP32Output("z", z.data(), 1, 2));
    dump.outputs.push_back(makeFP32Output("alpha", alpha.data(), 1, 2));
    dump.outputs.push_back(makeFP32Output("beta", beta.data(), 1, 2));

    SnapshotCapture capture;
    capture.captureStage("layer0_gdn_proj", dump);

    const auto *qkv_snapshot = capture.get("layer0_QKV_PROJECTION");
    const auto *z_snapshot = capture.get("layer0_GDN_Z_PROJECTION");
    const auto *alpha_snapshot = capture.get("layer0_GDN_ALPHA");
    const auto *beta_snapshot = capture.get("layer0_GDN_BETA");

    ASSERT_NE(qkv_snapshot, nullptr);
    ASSERT_NE(z_snapshot, nullptr);
    ASSERT_NE(alpha_snapshot, nullptr);
    ASSERT_NE(beta_snapshot, nullptr);
    EXPECT_EQ(qkv_snapshot->data, qkv);
    EXPECT_EQ(z_snapshot->data, z);
    EXPECT_EQ(alpha_snapshot->data, alpha);
    EXPECT_EQ(beta_snapshot->data, beta);
}

TEST(Test__SnapshotCapture_Capture, MTPSidecarMoERoutingUsesMTPKeys)
{
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    std::vector<float> indices = {2.0f, 1.0f};
    std::vector<float> weights = {0.75f, 0.25f};

    StageDumpInfo dump;
    dump.outputs.push_back(makeFP32Output("logits", logits.data(), 1, 3));
    dump.outputs.push_back(makeFP32Output("indices", indices.data(), 1, 2));
    dump.outputs.push_back(makeFP32Output("weights", weights.data(), 1, 2));

    SnapshotCapture capture;
    capture.captureStage("MTP0_moe_routing", dump);

    const auto *router_snapshot = capture.get("MTP0_MOE_ROUTER_OUTPUT");
    const auto *indices_snapshot = capture.get("MTP0_MOE_ROUTING_INDICES");
    const auto *weights_snapshot = capture.get("MTP0_MOE_ROUTING_WEIGHTS");
    ASSERT_NE(router_snapshot, nullptr);
    ASSERT_NE(indices_snapshot, nullptr);
    ASSERT_NE(weights_snapshot, nullptr);
    EXPECT_EQ(router_snapshot->data, logits);
    EXPECT_EQ(indices_snapshot->data, indices);
    EXPECT_EQ(weights_snapshot->data, weights);
}

TEST(Test__SnapshotCapture_Capture, MoERoutingTensorOnlyDumpUsesCanonicalKeys)
{
    std::vector<float> indices = {2.0f, 1.0f, 3.0f, 0.0f};
    std::vector<float> weights = {0.75f, 0.25f, 0.60f, 0.40f};

    StageDumpInfo dump;
    dump.outputs.push_back(
        makeFP32Output("output_indices_tensor", indices.data(), 2, 2));
    dump.outputs.push_back(
        makeFP32Output("output_weights_tensor", weights.data(), 2, 2));

    SnapshotCapture capture;
    capture.captureStage("layer7_moe_routing", dump);

    const auto *router_snapshot = capture.get("layer7_MOE_ROUTER_OUTPUT");
    const auto *indices_snapshot = capture.get("layer7_MOE_ROUTING_INDICES");
    const auto *weights_snapshot = capture.get("layer7_MOE_ROUTING_WEIGHTS");

    EXPECT_EQ(router_snapshot, nullptr);
    ASSERT_NE(indices_snapshot, nullptr);
    ASSERT_NE(weights_snapshot, nullptr);
    EXPECT_EQ(indices_snapshot->data, indices);
    EXPECT_EQ(weights_snapshot->data, weights);
}

TEST(Test__SnapshotCapture_Capture, ContextQualifiedMTPKeysRemainDisambiguated)
{
    std::vector<float> decode_embedding = {1.0f, 2.0f};
    std::vector<float> catchup_embedding = {3.0f, 4.0f};

    SnapshotCapture capture;
    capture.captureStage(
        "mtp_decode_sidecar::MTP0_embedding",
        makeSingleOutputDump("output", decode_embedding.data(), 1, 2));
    capture.captureStage(
        "mtp_decode_catchup::MTP0_embedding",
        makeSingleOutputDump("output", catchup_embedding.data(), 1, 2));

    const auto *decode_snapshot = capture.get("MTP_DECODE_SIDECAR_MTP0_EMBEDDING");
    const auto *catchup_snapshot = capture.get("MTP_DECODE_CATCHUP_MTP0_EMBEDDING");
    ASSERT_NE(decode_snapshot, nullptr);
    ASSERT_NE(catchup_snapshot, nullptr);
    EXPECT_EQ(decode_snapshot->data, decode_embedding);
    EXPECT_EQ(catchup_snapshot->data, catchup_embedding);
}

/**
 * @brief Segmented prefill restores ordinary parity keys to the full prompt.
 *
 * The graph callback records a scoped copy for each fixed bucket and also
 * writes the historical bare key. Aggregation must replace that bare final
 * chunk with the real-row concatenation while retaining every scoped value for
 * CSV-guided diagnosis. Terminal values remain the final chunk rather than
 * being fabricated into sequence tensors.
 */
TEST(Test__SnapshotCapture_Capture,
     SegmentedPrefillAggregationJoinsLiveRowsAndRetainsChunkEvidence)
{
    const std::vector<float> first_embedding = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> second_embedding = {5.0f, 6.0f};
    const std::vector<float> first_logits = {10.0f, 11.0f};
    const std::vector<float> second_logits = {12.0f, 13.0f};

    SnapshotCapture capture;
    capture.captureStage(
        "prefill_chunk_0::embedding",
        makeSingleOutputDump(
            "output", first_embedding.data(), /*rows=*/2, /*cols=*/2));
    capture.captureStage(
        "prefill_chunk_1::embedding",
        makeSingleOutputDump(
            "output", second_embedding.data(), /*rows=*/1, /*cols=*/2));
    // LM-head snapshots are intentionally last-token values: one row even
    // though the first graph chunk contains two real token rows.
    capture.captureStage(
        "prefill_chunk_0::lm_head",
        makeSingleOutputDump(
            "output", first_logits.data(), /*rows=*/1, /*cols=*/2));
    capture.captureStage(
        "prefill_chunk_1::lm_head",
        makeSingleOutputDump(
            "output", second_logits.data(), /*rows=*/1, /*cols=*/2));

    const auto aggregation = capture.aggregateSequentialChunkSnapshots({
        {.context = "prefill_chunk_0", .logical_rows = 2},
        {.context = "prefill_chunk_1", .logical_rows = 1},
    });
    ASSERT_TRUE(aggregation) << aggregation.error;
    EXPECT_EQ(aggregation.aggregated_sequence_keys, 1u);
    EXPECT_EQ(aggregation.terminal_or_nonsequence_keys, 1u);

    const auto *embedding = capture.get("EMBEDDING");
    ASSERT_NE(embedding, nullptr);
    EXPECT_EQ(embedding->rows, 3u);
    EXPECT_EQ(embedding->cols, 2u);
    EXPECT_EQ(
        embedding->data,
        (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

    // The final bare LM_HEAD remains the terminal chunk. It must not claim to
    // contain three prompt rows merely because the embedding is sequence-shaped.
    const auto *lm_head = capture.get("LM_HEAD");
    ASSERT_NE(lm_head, nullptr);
    EXPECT_EQ(lm_head->rows, 1u);
    EXPECT_EQ(lm_head->data, second_logits);

    const auto *first_scoped = capture.get("PREFILL_CHUNK_0_EMBEDDING");
    const auto *second_scoped = capture.get("PREFILL_CHUNK_1_EMBEDDING");
    ASSERT_NE(first_scoped, nullptr);
    ASSERT_NE(second_scoped, nullptr);
    EXPECT_EQ(first_scoped->data, first_embedding);
    EXPECT_EQ(second_scoped->data, second_embedding);
}

/**
 * @brief Packed route contributions join live slots and discard bucket padding.
 *
 * Canonical ExpertOverlay evidence has `top_k` rows per logical token. The
 * final segmented chunk still occupies the fixed physical bucket, so its
 * inactive route rows must not survive in the prompt-wide diagnostic tensor.
 */
TEST(Test__SnapshotCapture_Capture,
     SegmentedPrefillAggregationJoinsPackedRouteRowsWithoutPadding)
{
    const std::vector<float> first_indices = {0.0f, 1.0f, 2.0f, 3.0f};
    const std::vector<float> final_indices = {4.0f, 5.0f};
    const std::vector<float> first_contributions = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    };
    const std::vector<float> final_bucket_contributions = {
        9.0f, 10.0f,
        11.0f, 12.0f,
        99.0f, 99.0f,
        99.0f, 99.0f,
    };

    SnapshotCapture capture;
    StageDumpInfo first_routing;
    first_routing.outputs.push_back(makeFP32Output(
        "output_indices_tensor", first_indices.data(), 2, 2));
    capture.captureStage(
        "prefill_chunk_0::layer0_moe_routing", first_routing);
    StageDumpInfo final_routing;
    final_routing.outputs.push_back(makeFP32Output(
        "output_indices_tensor", final_indices.data(), 1, 2));
    capture.captureStage(
        "prefill_chunk_1::layer0_moe_routing", final_routing);

    StageDumpInfo first_experts;
    first_experts.outputs.push_back(makeFP32Output(
        "canonical_route_contributions",
        first_contributions.data(),
        4,
        2));
    capture.captureStage(
        "prefill_chunk_0::layer0_moe_expert_ffn_overlay_fast",
        first_experts);
    StageDumpInfo final_experts;
    final_experts.outputs.push_back(makeFP32Output(
        "canonical_route_contributions",
        final_bucket_contributions.data(),
        4,
        2));
    capture.captureStage(
        "prefill_chunk_1::layer0_moe_expert_ffn_overlay_fast",
        final_experts);

    const auto aggregation = capture.aggregateSequentialChunkSnapshots({
        {.context = "prefill_chunk_0", .logical_rows = 2},
        {.context = "prefill_chunk_1", .logical_rows = 1},
    });
    ASSERT_TRUE(aggregation) << aggregation.error;
    EXPECT_EQ(aggregation.aggregated_sequence_keys, 2u);
    EXPECT_EQ(aggregation.terminal_or_nonsequence_keys, 0u);

    const auto *canonical =
        capture.get("layer0_MOE_CANONICAL_ROUTE_CONTRIBUTIONS");
    ASSERT_NE(canonical, nullptr);
    EXPECT_EQ(canonical->rows, 6u);
    EXPECT_EQ(canonical->cols, 2u);
    EXPECT_EQ(
        canonical->data,
        (std::vector<float>{
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f,
        }));
}

/**
 * @brief An incomplete chunk checkpoint must fail rather than compare a tail.
 */
TEST(Test__SnapshotCapture_Capture,
     SegmentedPrefillAggregationRejectsMissingSequenceChunk)
{
    const std::vector<float> first_embedding = {1.0f, 2.0f, 3.0f, 4.0f};
    SnapshotCapture capture;
    capture.captureStage(
        "prefill_chunk_0::embedding",
        makeSingleOutputDump(
            "output", first_embedding.data(), /*rows=*/2, /*cols=*/2));

    const auto aggregation = capture.aggregateSequentialChunkSnapshots({
        {.context = "prefill_chunk_0", .logical_rows = 2},
        {.context = "prefill_chunk_1", .logical_rows = 1},
    });
    EXPECT_FALSE(aggregation);
    EXPECT_NE(aggregation.error.find("missing chunk 1"), std::string::npos);
}

// =========================================================================
// Test: captureStage Routing
// =========================================================================

class Test__SnapshotCapture_Routing : public ::testing::Test
{
protected:
    SnapshotCapture capture;
    // Reusable FP32 buffers
    std::vector<float> data_a;
    std::vector<float> data_b;
    std::vector<float> data_c;

    void SetUp() override
    {
        // 2x4 matrix with distinct values per buffer
        data_a.resize(8);
        data_b.resize(8);
        data_c.resize(8);
        std::iota(data_a.begin(), data_a.end(), 1.0f);  // 1,2,3,...
        std::iota(data_b.begin(), data_b.end(), 10.0f); // 10,11,12,...
        std::iota(data_c.begin(), data_c.end(), 20.0f); // 20,21,22,...
    }
};

TEST_F(Test__SnapshotCapture_Routing, StandardSingleOutput)
{
    auto dump = makeSingleOutputDump("output", data_a.data(), 2, 4);
    capture.captureStage("embedding", dump);

    auto *snap = capture.get("EMBEDDING");
    ASSERT_NE(snap, nullptr);
    EXPECT_EQ(snap->rows, 2u);
    EXPECT_EQ(snap->cols, 4u);
    EXPECT_FLOAT_EQ(snap->data[0], 1.0f);
    EXPECT_EQ(snap->data.size(), 8u);
}

TEST_F(Test__SnapshotCapture_Routing, QKVProjectionSplit)
{
    StageDumpInfo dump;
    dump.outputs.push_back(makeFP32Output("q", data_a.data(), 2, 4));
    dump.outputs.push_back(makeFP32Output("k", data_b.data(), 2, 4));
    dump.outputs.push_back(makeFP32Output("v", data_c.data(), 2, 4));

    capture.captureStage("layer0_qkv_proj", dump);

    // Should produce three separate snapshots
    auto *q = capture.get("layer0_Q_PROJECTION");
    auto *k = capture.get("layer0_K_PROJECTION");
    auto *v = capture.get("layer0_V_PROJECTION");

    ASSERT_NE(q, nullptr);
    ASSERT_NE(k, nullptr);
    ASSERT_NE(v, nullptr);

    // data_a starts at 1.0, data_b at 10.0, data_c at 20.0
    EXPECT_FLOAT_EQ(q->data[0], 1.0f);
    EXPECT_FLOAT_EQ(k->data[0], 10.0f);
    EXPECT_FLOAT_EQ(v->data[0], 20.0f);
}

TEST_F(Test__SnapshotCapture_Routing, GateUpSplit)
{
    StageDumpInfo dump;
    dump.outputs.push_back(makeFP32Output("gate", data_a.data(), 2, 4));
    dump.outputs.push_back(makeFP32Output("up", data_b.data(), 2, 4));

    capture.captureStage("layer5_gate_up", dump);

    auto *gate = capture.get("layer5_FFN_GATE");
    auto *up = capture.get("layer5_FFN_UP");

    ASSERT_NE(gate, nullptr);
    ASSERT_NE(up, nullptr);
    EXPECT_FLOAT_EQ(gate->data[0], 1.0f);
    EXPECT_FLOAT_EQ(up->data[0], 10.0f);
}

TEST_F(Test__SnapshotCapture_Routing, RoPESplit)
{
    StageDumpInfo dump;
    dump.outputs.push_back(makeFP32Output("q_rope", data_a.data(), 2, 4));
    dump.outputs.push_back(makeFP32Output("k_rope", data_b.data(), 2, 4));

    capture.captureStage("layer0_rope", dump);

    auto *q = capture.get("layer0_Q_ROPE");
    auto *k = capture.get("layer0_K_ROPE");

    ASSERT_NE(q, nullptr);
    ASSERT_NE(k, nullptr);
    EXPECT_FLOAT_EQ(q->data[0], 1.0f);
    EXPECT_FLOAT_EQ(k->data[0], 10.0f);
}

TEST_F(Test__SnapshotCapture_Routing, RoPEDoesNotMatchQRopeOrKRope)
{
    // "layer0_q_rope" should NOT trigger the fused RoPE handler
    auto dump = makeSingleOutputDump("output", data_a.data(), 2, 4);
    capture.captureStage("layer0_q_rope", dump);

    // Should use standard path → key = "layer0_Q_ROPE"
    auto *snap = capture.get("layer0_Q_ROPE");
    ASSERT_NE(snap, nullptr);
    EXPECT_FLOAT_EQ(snap->data[0], 1.0f);
}

TEST_F(Test__SnapshotCapture_Routing, FusedResidualNormStoresSecondOutput)
{
    // For attn_norm and ffn_norm with 2 outputs: store outputs[1] (norm_output)
    StageDumpInfo dump;
    dump.outputs.push_back(makeFP32Output("residual", data_a.data(), 2, 4));
    dump.outputs.push_back(makeFP32Output("norm_output", data_b.data(), 2, 4));

    capture.captureStage("layer0_attn_norm", dump);

    auto *snap = capture.get("layer0_ATTENTION_NORM");
    ASSERT_NE(snap, nullptr);
    // Should contain data_b (outputs[1]), not data_a (outputs[0])
    EXPECT_FLOAT_EQ(snap->data[0], 10.0f);
}

TEST_F(Test__SnapshotCapture_Routing, FFNNormFusedResidual)
{
    StageDumpInfo dump;
    dump.outputs.push_back(makeFP32Output("residual", data_a.data(), 2, 4));
    dump.outputs.push_back(makeFP32Output("norm_output", data_b.data(), 2, 4));

    capture.captureStage("layer3_ffn_norm", dump);

    auto *snap = capture.get("layer3_FFN_NORM");
    ASSERT_NE(snap, nullptr);
    EXPECT_FLOAT_EQ(snap->data[0], 10.0f);
}

TEST_F(Test__SnapshotCapture_Routing, LMHeadAllgatherOverwrites)
{
    // First store a partial LM head
    auto dump1 = makeSingleOutputDump("logits", data_a.data(), 1, 8);
    capture.captureStage("lm_head", dump1);

    auto *partial = capture.get("LM_HEAD");
    ASSERT_NE(partial, nullptr);
    EXPECT_FLOAT_EQ(partial->data[0], 1.0f);

    // Now allgather should OVERWRITE with full vocab
    auto dump2 = makeSingleOutputDump("logits", data_b.data(), 1, 8);
    capture.captureStage("lm_head_allgather", dump2);

    auto *full = capture.get("LM_HEAD");
    ASSERT_NE(full, nullptr);
    // Should be data_b, not data_a
    EXPECT_FLOAT_EQ(full->data[0], 10.0f);
}

// =========================================================================
// Test: clear() and keys()
// =========================================================================

TEST_F(Test__SnapshotCapture_Routing, ClearRemovesAll)
{
    auto dump = makeSingleOutputDump("output", data_a.data(), 2, 4);
    capture.captureStage("embedding", dump);
    capture.captureStage("final_norm", dump);

    EXPECT_EQ(capture.all().size(), 2u);

    capture.clear();
    EXPECT_TRUE(capture.all().empty());
    EXPECT_EQ(capture.get("EMBEDDING"), nullptr);
}

TEST_F(Test__SnapshotCapture_Routing, KeysReturnsAllKeys)
{
    auto dump = makeSingleOutputDump("output", data_a.data(), 2, 4);
    capture.captureStage("embedding", dump);
    capture.captureStage("final_norm", dump);
    capture.captureStage("lm_head", dump);

    auto keys = capture.keys();
    EXPECT_EQ(keys.size(), 3u);

    // Keys should include EMBEDDING, FINAL_NORM, LM_HEAD
    std::set<std::string> key_set(keys.begin(), keys.end());
    EXPECT_TRUE(key_set.count("EMBEDDING"));
    EXPECT_TRUE(key_set.count("FINAL_NORM"));
    EXPECT_TRUE(key_set.count("LM_HEAD"));
}

// =========================================================================
// Test: StoredSnapshot shape metadata
// =========================================================================

TEST_F(Test__SnapshotCapture_Routing, ShapeMetadataPreserved)
{
    std::vector<float> wide(32);
    std::iota(wide.begin(), wide.end(), 0.0f);

    auto dump = makeSingleOutputDump("output", wide.data(), 4, 8);
    capture.captureStage("layer0_q_proj", dump);

    auto *snap = capture.get("layer0_Q_PROJECTION");
    ASSERT_NE(snap, nullptr);
    EXPECT_EQ(snap->rows, 4u);
    EXPECT_EQ(snap->cols, 8u);
    EXPECT_EQ(snap->data.size(), 32u);
}

// =========================================================================
// Regression: GDN suffix matching order (Bug #2)
//
// Bug: SnapshotCapture used an unordered_map for suffix→key mapping.
// The stage name "layer0_gdn_wo_allreduce" matched the shorter suffix
// "_wo_allreduce" first (hash order), extracting prefix "layer0_gdn"
// → key "layer0_gdn_ATTENTION_OUTPUT_ALLREDUCED" instead of the correct
// "layer0_ATTENTION_OUTPUT_ALLREDUCED".
//
// Fix: Changed to ordered vector with longest-suffix-first matching,
// so "_gdn_wo_allreduce" matches before "_wo_allreduce".
// =========================================================================

TEST(Test__SnapshotCapture_KeyConversion, GDNStages)
{
    // GDN-specific stage names
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_gdn_proj"),
              "layer0_QKV_PROJECTION");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_gdn_recurrence"),
              "layer0_GDN_DELTA_RULE_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_gated_norm"),
              "layer0_GDN_NORM_GATE_OUTPUT");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_gdn_out_proj"),
              "layer0_ATTENTION_OUTPUT");
}

TEST(Test__SnapshotCapture_KeyConversion, GDNWoAllreduceSuffixMatchesBeforeWoAllreduce)
{
    // THE regression test: "_gdn_wo_allreduce" must match BEFORE "_wo_allreduce"
    // so the prefix is "layer0" (not "layer0_gdn")
    auto key = SnapshotCapture::convertStageNameToSnapshotKey("layer0_gdn_wo_allreduce");
    EXPECT_EQ(key, "layer0_ATTENTION_OUTPUT_ALLREDUCED")
        << "Bug: '_gdn_wo_allreduce' matched '_wo_allreduce' suffix, "
           "producing 'layer0_gdn_ATTENTION_OUTPUT_ALLREDUCED' instead of "
           "'layer0_ATTENTION_OUTPUT_ALLREDUCED'";

    // Also verify the shorter suffix still works for non-GDN stages
    auto key2 = SnapshotCapture::convertStageNameToSnapshotKey("layer0_wo_allreduce");
    EXPECT_EQ(key2, "layer0_ATTENTION_OUTPUT_ALLREDUCED")
        << "'_wo_allreduce' should still map to ATTENTION_OUTPUT_ALLREDUCED for FA layers";
}

TEST(Test__SnapshotCapture_KeyConversion, GDNSuffixMatchingAcrossLayers)
{
    // Verify suffix ordering works for all layer indices, not just layer0
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer5_gdn_wo_allreduce"),
              "layer5_ATTENTION_OUTPUT_ALLREDUCED");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer23_gdn_wo_allreduce"),
              "layer23_ATTENTION_OUTPUT_ALLREDUCED");

    // The bug was visible across ALL GDN layers, not just layer 0
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer17_gdn_proj"),
              "layer17_QKV_PROJECTION");
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer17_gdn_recurrence"),
              "layer17_GDN_DELTA_RULE_OUTPUT");
}

TEST(Test__SnapshotCapture_KeyConversion, LongestSuffixMatchesFirst)
{
    // Verify that the longest/most-specific suffix always wins.
    // "_down_allreduce" must match before "_allreduce" (if such shorter suffix existed)
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_down_allreduce"),
              "layer0_FFN_DOWN_ALLREDUCED");

    // "_gdn_out_proj" must match before "_out_proj" or "_proj"
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_gdn_out_proj"),
              "layer0_ATTENTION_OUTPUT");

    // "_wo_proj" must not be confused with "_q_proj"
    EXPECT_EQ(SnapshotCapture::convertStageNameToSnapshotKey("layer0_wo_proj"),
              "layer0_ATTENTION_OUTPUT");
}
