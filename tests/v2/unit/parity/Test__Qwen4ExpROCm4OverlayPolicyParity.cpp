/**
 * @file Test__Qwen4ExpROCm4OverlayPolicyParity.cpp
 * @brief Device-free explicit ROCm4 dense/routed source and paired-binding controls.
 *
 * Actual typed factories, canonical public parser and local profile attachment
 * are exercised without model/reference files, backend initialization or PMA
 * capacity claims. Toy document locators below are metadata only.
 */
#include "integration/parity/qwen4exp/Qwen4ExpLocalPairedCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "config/OrchestrationConfigParser.h"
#include <gtest/gtest.h>
#include <set>
#include <sstream>

namespace llaminar2::test::parity::qwen4exp
{
namespace
{
/** @return Actual public diagnostic projection, parsed without device/model work. */
OrchestrationConfig publicPolicyConfig(const ModelParityCase &cell)
{
    auto args = modelParityDiagnosticBenchmarkArguments(cell);
    args.insert(args.begin(), {"llaminar2", "--model", cell.model.model_path});
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
}
/** @return Canonical synthetic document reference, never an opened file/receipt. */
nlohmann::json policyReference(const char *name)
{
    return {{"path", std::string("synthetic/") + name}, {"sha256", std::string(64, 'a')}};
}
/** @return Existing local binding schema targeting only the three new exact source cases. */
nlohmann::json policyDocument(const QFNCaseFamilies &families)
{
    nlohmann::json bindings = nlohmann::json::array();
    for (const auto &cell : families.at("Qwen4ExpROCm4OverlayPolicy"))
        bindings.push_back({
            {"exact_cell", "Qwen4ExpROCm4OverlayPolicy/Qwen4ExpMainParityTest.ProductionParity/" + cell.testName()},
            {"role", "required"}, {"catalog", policyReference("catalog.json")},
            {"workload", policyReference("manifest.json")}, {"pilot", policyReference("pilot.json")},
            {"benchmark", policyReference("benchmark.json")}, {"arithmetic", {
                {"gdn_projection", "fp16_moe_alpha_beta"}, {"moe_router", "fp16"},
                {"hc_up", "two_term_q8"}, {"gdn_output", "single_q8"},
                {"fa_output", "single_q8"}, {"gdn_qkv", "single_q8"}}}});
    return {{"schema", "llaminar.qfn-local-paired-profiles.v1"},
        {"source", {{"architecture", "qwen4exp"}, {"main_layers", 48}, {"expert_count", 512},
            {"shards", nlohmann::json::array({{{"path", kMainModelPath}},
                nlohmann::json::object(), nlohmann::json::object(), nlohmann::json::object()})},
            {"shared_predictor", {{"path", kPredictorModelPath}}}}}, {"bindings", bindings}};
}
}

/** @test Complete the two-by-two policy coverage without replacing the original A1 control. */
TEST(Qwen4ExpROCm4OverlayPolicy, ThreeAdditionalCasesRetainSourceAndNumericalObligations)
{
    const auto &original = qwen4ExpFullROCmTP4ProductionCases().front();
    const auto original_args = modelParityDiagnosticBenchmarkArguments(original);
    const auto &cells = qwen4ExpROCm4OverlayPolicyProductionCases();
    ASSERT_EQ(cells.size(), 3u);
    std::set<std::pair<DenseParallelPolicy, RoutedExpertComputePolicy>> policies;
    policies.emplace(DenseParallelPolicy::TensorParallel, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
    std::set<std::string> names{original.testName()};
    for (const auto &cell : cells)
    {
        ASSERT_TRUE(cell.topology.expert_overlay_plan);
        const auto &plan = *cell.topology.expert_overlay_plan;
        ASSERT_EQ(plan.domains.size(), 1u);
        EXPECT_TRUE(policies.emplace(plan.continuation_domain_spec.effectiveDensePolicy(),
                                     plan.domains.front().routed_compute_policy).second);
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_EQ(cell.model.model_path, original.model.model_path);
        EXPECT_EQ(cell.model.shared_predictor, original.model.shared_predictor);
        EXPECT_EQ(cell.model.reference_directory, original.model.reference_directory);
        EXPECT_EQ(cell.model.prompt, original.model.prompt);
        EXPECT_EQ(cell.model.token_ids, original.model.token_ids);
        EXPECT_EQ(cell.model.transformer_layers, 48);
        EXPECT_EQ(cell.model.decode_steps, original.model.decode_steps);
        EXPECT_EQ(cell.ngram_table_policy, original.ngram_table_policy);
        EXPECT_EQ(cell.activation_precision, original.activation_precision);
        EXPECT_EQ(cell.kv_cache_precision, original.kv_cache_precision);
        EXPECT_EQ(cell.mtp, ModelParityMTP::Off);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, 0);
        EXPECT_EQ(cell.expert_overlay, original.expert_overlay);
        EXPECT_EQ(cell.collective_evidence_source, original.collective_evidence_source);
        EXPECT_EQ(cell.prefix_restore_geometry, original.prefix_restore_geometry);
        EXPECT_EQ(cell.graph_snapshot_policy.required_prefill_snapshot_keys,
                  original.graph_snapshot_policy.required_prefill_snapshot_keys);
        EXPECT_EQ(cell.graph_snapshot_policy.required_decode_snapshot_keys,
                  original.graph_snapshot_policy.required_decode_snapshot_keys);
        EXPECT_EQ(cell.thresholds.cosine_threshold, original.thresholds.cosine_threshold);
        EXPECT_EQ(cell.thresholds.decode_cosine_threshold, original.thresholds.decode_cosine_threshold);
        EXPECT_EQ(cell.thresholds.kl_threshold, original.thresholds.kl_threshold);
        EXPECT_EQ(cell.thresholds.min_top1_accuracy, original.thresholds.min_top1_accuracy);
        EXPECT_EQ(cell.thresholds.min_top5_accuracy, original.thresholds.min_top5_accuracy);
        ASSERT_TRUE(cell.thresholds.final_logit_mean_kl && original.thresholds.final_logit_mean_kl);
        EXPECT_EQ(cell.thresholds.final_logit_mean_kl->prefill_maximum,
                  original.thresholds.final_logit_mean_kl->prefill_maximum);
        EXPECT_EQ(cell.thresholds.final_logit_mean_kl->decode_maximum,
                  original.thresholds.final_logit_mean_kl->decode_maximum);
        EXPECT_EQ(cell.thresholds.final_logit_mean_kl->decode_steps,
                  original.thresholds.final_logit_mean_kl->decode_steps);
        EXPECT_FALSE(cell.e2e_certification);
        EXPECT_FALSE(cell.paired_workload);
        for (const auto *body : {"ProductionParity", "SnapshotAdmission"})
            EXPECT_LT(std::string("Qwen4ExpROCm4OverlayPolicy_Qwen4ExpMainParityTest_").size() +
                      std::string(body).size() + 1u + cell.testName().size(), 256u);
    }
    EXPECT_EQ(policies.size(), 4u);
    EXPECT_EQ(modelParityDiagnosticBenchmarkArguments(qwen4ExpFullROCmTP4ProductionCases().front()), original_args);
}

/** @test Both actual config paths preserve exact dense/routed policies and complete GPU-only ownership. */
TEST(Qwen4ExpROCm4OverlayPolicy, PublicParserAndTypedProjectionAgreeWithoutCapacityOrCpuSubstitution)
{
    for (const auto &cell : qwen4ExpROCm4OverlayPolicyProductionCases())
        for (const auto &config : {cell.makeOrchestrationConfig(cell.model.model_path, 0), publicPolicyConfig(cell)})
        {
            ASSERT_TRUE(config.moe_routed_expert_plan);
            const auto &plan = *config.moe_routed_expert_plan;
            const auto &expected = *cell.topology.expert_overlay_plan;
            EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(),
                      expected.continuation_domain_spec.effectiveDensePolicy());
            EXPECT_EQ(plan.continuation_domain_spec.hidden_layout, MoEContinuationActivationLayout::ReplicatedHidden);
            ASSERT_EQ(plan.domains.size(), 1u);
            ASSERT_EQ(plan.dense_domains.size(), 1u);
            const auto &domain = plan.domains.front();
            EXPECT_EQ(domain.routed_compute_policy, expected.domains.front().routed_compute_policy);
            EXPECT_EQ(plan.dense_domains.front().routed_compute_policy, domain.routed_compute_policy);
            EXPECT_EQ(domain.scope, ExecutionDomainScope::RANK_LOCAL);
            EXPECT_EQ(domain.backend, CollectiveBackendType::RCCL);
            EXPECT_EQ(domain.routed_phase_policy, RoutedExpertPhasePolicy::Uniform);
            EXPECT_EQ(domain.routed_decode_assignment_policy, RoutedExpertAssignmentPolicy::StaticOwner);
            EXPECT_EQ(domain.routed_prefill_assignment_policy, RoutedExpertAssignmentPolicy::StaticOwner);
            ASSERT_EQ(domain.participants.size(), 4u);
            for (int ordinal = 0; ordinal < 4; ++ordinal)
                EXPECT_EQ(domain.participants[static_cast<std::size_t>(ordinal)], GlobalDeviceAddress::rocm(ordinal));
            ASSERT_EQ(plan.routed_tiers.size(), 1u);
            EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer, 0);
            EXPECT_TRUE(plan.routed_tiers.front().fallback);
            EXPECT_EQ(plan.routed_tiers.front().memory_budget_bytes, 0u);
            EXPECT_TRUE(validateMoERoutedExpertPlacementPlan(plan, {.layer_count=48, .routed_expert_count=512}).ok());
            EXPECT_EQ(config.ngram_table_policy.storage, ImmutableRowStoragePreference::RAM);
            EXPECT_FALSE(config.mtp.enabled);
            EXPECT_TRUE(config.pp_stage_definitions.empty());
        }
}

/** @test The existing opt-in profile mechanism consumes new exact IDs without attaching to older cases. */
TEST(Qwen4ExpROCm4OverlayPolicy, PairedAttachmentIsExplicitAndLeavesExistingFamiliesUnchanged)
{
    const auto original = qfnOriginalCaseFamilies();
    auto document = policyDocument(original);
    const auto attached = qfnAttachLocalPairedCases(original, document);
    for (const auto &[family, cells] : original)
    {
        ASSERT_EQ(attached.at(family).size(), cells.size());
        for (std::size_t index = 0; index < cells.size(); ++index)
        {
            const auto &after = attached.at(family)[index];
            EXPECT_EQ(after.testName(), cells[index].testName());
            EXPECT_EQ(modelParityDiagnosticBenchmarkArguments(after), modelParityDiagnosticBenchmarkArguments(cells[index]));
            if (family == "Qwen4ExpROCm4OverlayPolicy")
            {
                ASSERT_TRUE(after.paired_workload);
                ASSERT_TRUE(after.paired_workload->qfn_arithmetic);
                EXPECT_EQ(after.paired_workload->qfn_arithmetic->hc_up,
                          QuantizedActivationOperandPrecision::TwoTermQ8);
            }
            else EXPECT_FALSE(after.paired_workload);
        }
    }
    document["bindings"][0]["exact_cell"] = "Qwen4ExpROCm4OverlayPolicy/foreign";
    EXPECT_THROW(qfnAttachLocalPairedCases(original, document), std::invalid_argument);
    document = policyDocument(original);
    document["bindings"].push_back(document["bindings"][0]);
    EXPECT_THROW(qfnAttachLocalPairedCases(original, document), std::invalid_argument);
}

/** @test Expert mirroring, within-F sharding and undeclared dense modes cannot enter this comparison. */
TEST(Qwen4ExpROCm4OverlayPolicy, RejectsPoliciesOutsideTheExplicitComparison)
{
    EXPECT_THROW(qwen4ExpROCm4OverlayPolicyDefinition(DenseParallelPolicy::Replicated,
        RoutedExpertComputePolicy::Apportioned), std::invalid_argument);
    EXPECT_THROW(qwen4ExpROCm4OverlayPolicyDefinition(DenseParallelPolicy::TensorParallel,
        RoutedExpertComputePolicy::Replicated), std::invalid_argument);
    EXPECT_THROW(qwen4ExpROCm4OverlayPolicyDefinition(DenseParallelPolicy::TensorParallel,
        RoutedExpertComputePolicy::TensorSharded), std::invalid_argument);
    EXPECT_THROW(qwen4ExpROCm4OverlayPolicyDefinition(DenseParallelPolicy::TensorParallel,
        RoutedExpertComputePolicy::Automatic), std::invalid_argument);
}
} // namespace llaminar2::test::parity::qwen4exp
