/**
 * @file Test__Qwen4ExpMTPAdmissionCases.cpp
 * @brief Device-free source/depth/topology declarations for gold-free sidecar admission.
 *
 * The real model tests own physical preparation. These tests validate only the
 * fixed input projection, keeping the complete standard mathematical matrix and
 * main20 definitions unchanged and making no backend or model availability claim.
 */
#include "integration/parity/qwen4exp/Qwen4ExpMTPAdmissionCases.h"
#include <gtest/gtest.h>
#include <array>
#include <set>
namespace llaminar2::test::parity::qwen4exp
{
/** @test Eight sidecar cases cannot append or relabel the existing main20 cells. */
TEST(Qwen4ExpMTPAdmissionCases, EightPairedCasesPreserveMainTwentyAndExplicitStorage)
{
    EXPECT_EQ(qwen4ExpProductionCases().size(), 20u);
    ASSERT_EQ(qwen4ExpMTPAdmissionCases().size(), 8u);
    std::set<std::string> names;
    std::set<std::pair<std::string, ImmutableRowStoragePreference>> axes;
    for (const auto &cell : qwen4ExpMTPAdmissionCases())
    {
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_TRUE(cell.testName().ends_with("MTPR1Fixed1Retained1"));
        EXPECT_EQ(cell.main.mtp, ModelParityMTP::Off);
        EXPECT_EQ(cell.main.retained_mtp_draft_capacity, 0);
        EXPECT_EQ(cell.main.model.maximum_mtp_draft_depth, 0);
        EXPECT_FALSE(cell.main.model.shared_predictor);
        EXPECT_EQ(cell.predictor.path(), kPredictorModelPath);
        axes.emplace(cell.main.topology.test_id, cell.main.ngram_table_policy->storage);
        if (cell.main.expert_overlay) EXPECT_EQ(*cell.main.expert_overlay, ModelParityExpertOverlayPolicy{});
    }
    EXPECT_EQ(axes.size(), 8u);
}
/** @test Active and inactive sources retain the exact main identity and independent depth envelopes. */
TEST(Qwen4ExpMTPAdmissionCases, FixedOneAndExplicitNoMTPHaveIndependentRetainedCapacity)
{
    for (const auto &cell : qwen4ExpMTPAdmissionCases())
        for (const auto mode : {Qwen4ExpMTPAdmissionExecution::FixedOne, Qwen4ExpMTPAdmissionExecution::Inactive})
        {
            const auto config = cell.config(mode);
            const bool active = mode == Qwen4ExpMTPAdmissionExecution::FixedOne;
            EXPECT_EQ(config.model_path, kMainModelPath);
            ASSERT_TRUE(config.shared_predictor);
            EXPECT_EQ(config.shared_predictor->path(), kPredictorModelPath);
            EXPECT_EQ(config.ngram_table_policy, *cell.main.ngram_table_policy);
            EXPECT_EQ(config.max_seq_len, 4096);
            EXPECT_EQ(config.batch_size, 1);
            EXPECT_EQ(config.mtp.enabled, active);
            EXPECT_EQ(config.mtp_activation_policy, active ? MTPActivationPolicy::Enabled : MTPActivationPolicy::Disabled);
            EXPECT_EQ(config.mtp.draft_tokens, 1);
            EXPECT_EQ(config.mtp.graph_capacity_draft_tokens, active ? 1 : 0);
            EXPECT_EQ(config.mtp.max_request_batch, 1);
            EXPECT_EQ(config.mtp.depth_policy.mode, MTPDepthPolicyMode::Fixed);
            EXPECT_EQ(resolveMTPMaximumExecutionDraftDepth(config.mtp), 1);
            EXPECT_EQ(retainsMTPGraphCapacity(config.mtp), active);
            EXPECT_EQ(config.prefix_cache.block_size, 9);
        }
}
/** @test Existing8/12/28 PP remains complete and puts the sidecar only at the main terminal stage. */
TEST(Qwen4ExpMTPAdmissionCases, HybridKeepsOriginalExplicitPipelineAndRejectsForeignIntent)
{
    for (const auto &cell : qwen4ExpMTPAdmissionCases())
    {
        if (cell.main.topology.kind != ModelParityTopologyKind::RankLocalPipelineParallel) continue;
        const auto config = cell.config(Qwen4ExpMTPAdmissionExecution::FixedOne);
        ASSERT_EQ(config.pp_stage_definitions.size(), 3u);
        ASSERT_EQ(config.domain_definitions.size(), 3u);
        constexpr std::array first{0, 8, 20};
        constexpr std::array last{7, 19, 47};
        const std::array devices{GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::cpu(0)};
        for (std::size_t index = 0; index < first.size(); ++index)
        {
            EXPECT_EQ(config.pp_stage_definitions[index].first_layer, first[index]);
            EXPECT_EQ(config.pp_stage_definitions[index].last_layer, last[index]);
            EXPECT_EQ(config.domain_definitions[index].devices, (std::vector{devices[index]}));
        }
    }
    EXPECT_THROW(qwen4ExpMTPAdmissionCases().front().config(
        static_cast<Qwen4ExpMTPAdmissionExecution>(255)), std::invalid_argument);
    auto foreign = qwen4ExpMTPAdmissionCases().front();
    foreign.main.ngram_table_policy = ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::Automatic};
    EXPECT_THROW(foreign.config(Qwen4ExpMTPAdmissionExecution::FixedOne), std::invalid_argument);
}
} // namespace llaminar2::test::parity::qwen4exp
