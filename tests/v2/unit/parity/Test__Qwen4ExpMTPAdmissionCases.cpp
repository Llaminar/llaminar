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
#include "config/OrchestrationConfigDocument.h"
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
        for (const auto mode : {Qwen4ExpMTPAdmissionExecution::Active, Qwen4ExpMTPAdmissionExecution::Inactive})
        {
            const auto config = cell.config(mode);
            const bool active = mode == Qwen4ExpMTPAdmissionExecution::Active;
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
        const auto config = cell.config(Qwen4ExpMTPAdmissionExecution::Active);
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
    EXPECT_THROW(foreign.config(Qwen4ExpMTPAdmissionExecution::Active), std::invalid_argument);
}
/** @test A1 runtime policies preserve the Off gold identity and explicit R1 capacity. */
TEST(Qwen4ExpMTPAdmissionCases, FullROCmTP4RuntimePoliciesKeepNumericalOffSeparate)
{
    const auto &cases = qwen4ExpFullROCmTP4MTPAdmissionCases();
    ASSERT_EQ(cases.size(), 5u);
    EXPECT_EQ(std::string_view(qwen4ExpMTPRuntimePurposeName(Qwen4ExpMTPAdmissionPurpose::RuntimeQualification)),
              "runtime_qualification");
    EXPECT_EQ(std::string_view(qwen4ExpMTPRuntimePurposeName(Qwen4ExpMTPAdmissionPurpose::ProductionServing)),
              "production_serving");
    EXPECT_THROW(qwen4ExpMTPRuntimePurposeName(Qwen4ExpMTPAdmissionPurpose::AdmissionDiagnostic),
                 std::invalid_argument);
    EXPECT_THROW(qwen4ExpMTPRuntimePurposeName(static_cast<Qwen4ExpMTPAdmissionPurpose>(255)),
                 std::invalid_argument);
    const auto original = qwen4ExpFullROCmTP4ProductionCases().front().generation_workload;
    EXPECT_EQ(original.readinessSeconds(), 60);
    for (const auto &other : qwen4ExpMTPAdmissionCases())
        EXPECT_EQ(other.main.generation_workload.readinessSeconds(), 60);
    EXPECT_EQ(qwen4ExpFullROCmTP4ProductionCases().size(), 1u);
    std::set<std::string> names;
    for (const auto &cell : cases)
    {
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_EQ(cell.main.generation_workload, original.withReadiness(300));
        EXPECT_EQ(cell.main.generation_workload.maximumTokens(), original.maximumTokens());
        EXPECT_EQ(cell.main.generation_workload.minimumTokens(), original.minimumTokens());
        EXPECT_EQ(cell.main.mtp, ModelParityMTP::Off);
        EXPECT_EQ(cell.main.model.maximum_mtp_draft_depth, 0);
        EXPECT_EQ(cell.main.retained_mtp_draft_capacity, 0);
        const auto config = cell.config(Qwen4ExpMTPAdmissionExecution::Active);
        EXPECT_TRUE(config.deterministic);
        const auto production = cell.config(Qwen4ExpMTPAdmissionExecution::Active,
            Qwen4ExpMTPAdmissionPurpose::RuntimeQualification);
        EXPECT_FALSE(production.deterministic);
        EXPECT_EQ(production.mtp.verify_mode, modelParityHttpMTPVerifyMode());
        EXPECT_EQ(config.mtp.verify_mode, MTPVerifyMode::Greedy);
        EXPECT_EQ(production.mtp.draft_tokens, config.mtp.draft_tokens);
        const auto defaults = OrchestrationConfig::defaults();
        EXPECT_EQ(config.prefix_cache.block_size,
                  productionParityPrefixRestoreProofBlockSize(cell.main.model.token_ids.size()));
        EXPECT_EQ(production.prefix_cache.block_size, defaults.prefix_cache.block_size);
        EXPECT_NE(production.prefix_cache.block_size, config.prefix_cache.block_size);
        EXPECT_EQ(production.moe_rebalance.mode, config.moe_rebalance.mode);
        EXPECT_EQ(production.moe_rebalance.window_size, defaults.moe_rebalance.window_size);
        EXPECT_EQ(production.moe_rebalance.dynamic_min_improvement_per_mille,
                  defaults.moe_rebalance.dynamic_min_improvement_per_mille);
        auto expected_qualification = config;
        expected_qualification.prefix_cache = defaults.prefix_cache;
        expected_qualification.moe_rebalance = defaults.moe_rebalance;
        expected_qualification.moe_rebalance.mode = config.moe_rebalance.mode;
        expected_qualification.mtp.verify_mode = modelParityHttpMTPVerifyMode();
        expected_qualification.deterministic = false;
        EXPECT_EQ(serializeOrchestrationConfig(production),
                  serializeOrchestrationConfig(expected_qualification));
        const auto workload = cell.config(Qwen4ExpMTPAdmissionExecution::Active,
            Qwen4ExpMTPAdmissionPurpose::ProductionServing);
        // Exact document equality guards every topology, source, precision,
        // ownership, resource and state field, allowing only the startup-depth
        // tuning distinction for the already-authored Dynamic axis.
        auto expected_workload = production;
        if (cell.policy == ModelParityMTP::DynamicDepth)
            expected_workload.mtp.depth_policy.initial_depth = MTPDepthPolicyConfig{}.initial_depth;
        EXPECT_EQ(serializeOrchestrationConfig(workload), serializeOrchestrationConfig(expected_workload));
        EXPECT_EQ(workload.mtp.depth_policy.initial_depth, MTPDepthPolicyConfig{}.initial_depth);
        const auto workload_document = serializeOrchestrationConfig(workload);
        EXPECT_EQ(serializeOrchestrationConfig(deserializeOrchestrationConfig(workload_document)),
                  workload_document);
        // Poison short-proof geometry and cadence: neither production purpose
        // may import these values. The diagnostic still obeys the exact prompt.
        auto poisoned = cell;
        poisoned.main.model.token_ids.resize(17, 1);
        poisoned.main.dynamic_rebalance.window_size = 7;
        const auto poisoned_diagnostic = poisoned.config(Qwen4ExpMTPAdmissionExecution::Active);
        EXPECT_EQ(poisoned_diagnostic.prefix_cache.block_size, 17);
        EXPECT_EQ(poisoned_diagnostic.moe_rebalance.window_size, 7);
        EXPECT_NE(poisoned_diagnostic.moe_rebalance.window_size, config.moe_rebalance.window_size);
        EXPECT_EQ(serializeOrchestrationConfig(poisoned.config(Qwen4ExpMTPAdmissionExecution::Active,
            Qwen4ExpMTPAdmissionPurpose::RuntimeQualification)), serializeOrchestrationConfig(production));
        EXPECT_EQ(serializeOrchestrationConfig(poisoned.config(Qwen4ExpMTPAdmissionExecution::Active,
            Qwen4ExpMTPAdmissionPurpose::ProductionServing)), workload_document);
        for (const auto purpose : {Qwen4ExpMTPAdmissionPurpose::RuntimeQualification,
                                  Qwen4ExpMTPAdmissionPurpose::ProductionServing})
        {
            const auto inactive = cell.config(Qwen4ExpMTPAdmissionExecution::Inactive, purpose);
            const auto active_config = cell.config(Qwen4ExpMTPAdmissionExecution::Active, purpose);
            auto expected_inactive = active_config;
            expected_inactive.mtp.enabled = false;
            expected_inactive.mtp_activation_policy = MTPActivationPolicy::Disabled;
            expected_inactive.mtp.graph_capacity_draft_tokens = 0;
            EXPECT_EQ(serializeOrchestrationConfig(inactive), serializeOrchestrationConfig(expected_inactive));
        }
        EXPECT_THROW(cell.config(Qwen4ExpMTPAdmissionExecution::Active,
            static_cast<Qwen4ExpMTPAdmissionPurpose>(255)), std::invalid_argument);
        const auto document = serializeOrchestrationConfig(production);
        const auto decoded = deserializeOrchestrationConfig(document);
        EXPECT_EQ(serializeOrchestrationConfig(decoded), document);
        EXPECT_FALSE(decoded.deterministic);
        EXPECT_EQ(decoded.mtp.terminal_head_policy, MTPTerminalHeadPolicy::VocabularySharded);
        EXPECT_TRUE(config.mtp.enabled);
        EXPECT_EQ(config.mtp.max_request_batch, 1);
        EXPECT_EQ(config.batch_size, 1);
        EXPECT_EQ(config.mtp.graph_capacity_draft_tokens, 15);
        EXPECT_EQ(config.mtp.terminal_head_policy, MTPTerminalHeadPolicy::VocabularySharded);
        EXPECT_EQ(config.mtp.draft_tokens, cell.requestedDepth());
        EXPECT_EQ(config.mtp.depth_policy.mode, cell.policy == ModelParityMTP::DynamicDepth
            ? MTPDepthPolicyMode::Dynamic : MTPDepthPolicyMode::Fixed);
        if (cell.policy == ModelParityMTP::DynamicDepth)
        {
            EXPECT_EQ(config.mtp.depth_policy.min_depth, 1);
            EXPECT_EQ(config.mtp.depth_policy.max_depth, 15);
            EXPECT_EQ(config.mtp.depth_policy.initial_depth, 15);
        }
        EXPECT_EQ(config.model_path, kMainModelPath);
        ASSERT_TRUE(config.shared_predictor);
        EXPECT_EQ(config.shared_predictor->path(), kPredictorModelPath);
        EXPECT_EQ(config.ngram_table_policy.storage, ImmutableRowStoragePreference::RAM);
        ASSERT_TRUE(config.moe_routed_expert_plan);
        const auto &plan = *config.moe_routed_expert_plan;
        ASSERT_EQ(plan.domains.front().participants.size(), 4u);
        for (int ordinal = 0; ordinal < 4; ++ordinal)
            EXPECT_EQ(plan.domains.front().participants[ordinal], GlobalDeviceAddress::rocm(ordinal));
        EXPECT_EQ(config.activation_precision, "fp32");
        EXPECT_EQ(config.kv_cache_precision, "fp16");
        EXPECT_EQ(plan.domains.front().routed_compute_policy,
                  RoutedExpertComputePolicy::GateUpOwnedDownColumns);
        EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(), DenseParallelPolicy::TensorParallel);
        auto invalid = cell;
        invalid.retained_depth = 0;
        EXPECT_THROW(invalid.config(Qwen4ExpMTPAdmissionExecution::Active), std::invalid_argument);
        invalid = cell;
        invalid.policy = static_cast<ModelParityMTP>(255);
        EXPECT_THROW(invalid.config(Qwen4ExpMTPAdmissionExecution::Active), std::invalid_argument);
    }
}
} // namespace llaminar2::test::parity::qwen4exp
