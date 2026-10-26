/**
 * @file Test__Qwen4ExpMTPNumericalCandidateCases.cpp
 * @brief Device-free declarations for an explicitly unqualified QFN numerical MTP candidate.
 *
 * Actual source metadata, reference generation, model admission, accepted
 * drafts and native paths are runtime obligations, never inferred from this
 * inventory test. The canonical expander and public projection own all modes.
 */
#include "integration/parity/qwen4exp/Qwen4ExpMTPNumericalCandidateCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "integration/parity/qwen36/Qwen36ModelParityDefinitions.h"
#include "config/OrchestrationConfigParser.h"
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <nlohmann/json.hpp>
#include <iterator>
#include <set>

namespace llaminar2::test::parity::qwen4exp
{
/** @test Exact paired24 identities retain all six policies and leave main20 unchanged. */
TEST(Qwen4ExpMTPNumericalCandidates, CompleteTypedInventoryDoesNotEnableMainMTP)
{
    std::set<std::string> original;
    for (const auto &cell : qwen4ExpProductionCases())
    {
        original.insert(cell.testName());
        EXPECT_FALSE(cell.model.shared_predictor);
        EXPECT_EQ(cell.model.maximum_mtp_draft_depth, 0);
        EXPECT_EQ(cell.mtp, ModelParityMTP::Off);
    }
    EXPECT_EQ(original.size(), 20u);
    const auto &cases = qwen4ExpMTPNumericalCandidateCases();
    ASSERT_EQ(cases.size(), 24u);
    std::set<std::string> names;
    std::map<std::string, std::set<ModelParityMTP>> policies;
    for (const auto &cell : cases)
    {
        EXPECT_TRUE(names.insert(cell.testName()).second);
        policies[cell.topology.test_id].insert(cell.mtp);
        ASSERT_TRUE(cell.model.shared_predictor);
        EXPECT_EQ(cell.model.shared_predictor->path(), kPredictorModelPath);
        EXPECT_EQ(cell.model.model_path, kMainModelPath);
        EXPECT_EQ(cell.model.reference_directory, "parity-results/qwen-flash-next/reference/main-ram-mmap");
        EXPECT_EQ(cell.model.transformer_layers, 48);
        EXPECT_EQ(cell.model.maximum_mtp_draft_depth, 15);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, 15);
        EXPECT_EQ(cell.model.prefix_state, ModelParityPrefixState::HybridRecurrent);
        EXPECT_FALSE(cell.e2e_certification.has_value());
        EXPECT_TRUE(cell.model.mtp_checkpoint_surface.empty()); // Every actual reference boundary is mandatory.
        ASSERT_TRUE(cell.ngram_table_policy);
        EXPECT_EQ(cell.ngram_table_policy->storage, ImmutableRowStoragePreference::MemoryMapped);
        if (cell.expert_overlay)
        {
            EXPECT_EQ(*cell.expert_overlay, ModelParityExpertOverlayPolicy{});
        }
        EXPECT_EQ(cell.thresholds.early_layers_count, 48);
        EXPECT_EQ(cell.thresholds.min_early_layers_passed, 48);
    }
    ASSERT_EQ(policies.size(), 4u);
    const std::set<ModelParityMTP> expected(kCanonicalModelParityMTPPolicies.begin(), kCanonicalModelParityMTPPolicies.end());
    for (const auto &[topology, actual] : policies)
    {
        SCOPED_TRACE(topology);
        EXPECT_EQ(actual, expected);
    }
    EXPECT_EQ(qwen4ExpProductionCases().size(), 20u);
}

/** @test Requested logical depth stays independent of retained15/R1/source48 setup. */
TEST(Qwen4ExpMTPNumericalCandidates, RuntimeProjectionRetainsExplicitSourceAndDepth)
{
    for (const auto &cell : qwen4ExpMTPNumericalCandidateCases())
    {
        const auto config = cell.makeOrchestrationConfig(cell.model.model_path, 0);
        ASSERT_TRUE(config.shared_predictor);
        EXPECT_EQ(config.shared_predictor->path(), kPredictorModelPath);
        EXPECT_EQ(config.batch_size, 1);
        EXPECT_EQ(config.mtp.max_request_batch, 1);
        EXPECT_EQ(config.mtp.enabled, cell.mtpEnabled());
        EXPECT_EQ(config.mtp.graph_capacity_draft_tokens, 15);
        EXPECT_EQ(config.mtp.draft_tokens, std::max(1, cell.requestedMTPDraftDepth()));
        EXPECT_EQ(config.mtp.verify_mode, MTPVerifyMode::Greedy);
        EXPECT_EQ(config.mtp.depth_policy.mode, cell.usesDynamicMTPDepth() ? MTPDepthPolicyMode::Dynamic : MTPDepthPolicyMode::Fixed);
        EXPECT_EQ(config.max_seq_len, 4096);
        EXPECT_TRUE(config.prefix_cache.enabled);
        if (cell.expert_overlay)
        {
            EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Off);
        }
    }
}

/** @test The standard overlay validator rejects stale cadence before any weights or device startup. */
TEST(Qwen4ExpMTPNumericalCandidates, StandardOverlayRequiresCompleteVerifierMaintenanceEnvelope)
{
    auto stale = qwen4ExpMTPNumericalCandidateDefinition(Qwen4ExpParityTopology::CUDACPU);
    stale.dynamic_rebalance.economic_movement.device_min_maintenance_period_tokens = 1;
    try
    {
        (void)expandModelParityDefinition(stale);
        FAIL() << "Stale single-row cadence entered the standard MTP product";
    }
    catch (const std::invalid_argument &error)
    {
        EXPECT_NE(std::string(error.what()).find("recurring maintenance cadence"), std::string::npos);
    }
    EXPECT_EQ(expandModelParityDefinition(qwen4ExpMTPNumericalCandidateDefinition(Qwen4ExpParityTopology::CUDACPU)).size(), 24u);
}
} // namespace llaminar2::test::parity::qwen4exp

namespace llaminar2::test::parity::qwen4exp
{
/** @test The first executable CPU diagnostic has an exact canonical identity and public source mapping. */
TEST(Qwen4ExpMTPNumericalCandidates, FirstCPUCellHasExactIdentityAndPublicPredictorProjection)
{
    const auto &cases = qwen4ExpMTPNumericalCandidateCases();
    const auto first = std::find_if(cases.begin(), cases.end(), [](const auto &cell) {
        return cell.topology.test_id == "CPU0" && cell.mtp == ModelParityMTP::Depth1;
    });
    ASSERT_NE(first, cases.end());
    EXPECT_EQ(first->testName(), "Qwen38FlashNext_UDQ4KXL_CPU0_NgramMmap_ActFP32_KVFP16_MTPDepth1");
    auto args = modelParityServerArguments(*first);
    // This is the canonical runtime-policy projection, not a complete server
    // command: the typed caller owns the main model separately. The predictor
    // remains an explicit policy argument and cannot substitute for that main.
    EXPECT_EQ(std::find(args.begin(), args.end(), "--model"), args.end());
    ASSERT_TRUE(first->model.shared_predictor);
    const auto predictor = std::find(args.begin(), args.end(), "--mtp-model");
    ASSERT_NE(predictor, args.end());
    ASSERT_NE(std::next(predictor), args.end());
    EXPECT_EQ(*std::next(predictor), first->model.shared_predictor->path());
    EXPECT_EQ(first->makeOrchestrationConfig(first->model.model_path, 0).model_path, kMainModelPath);
    // Bind the complete invocation through the public parser, exactly where
    // the source-owned outer command supplies its main path. Never manufacture
    // a parser default or reinterpret the separate predictor as the main.
    args.insert(args.begin(), {"llaminar2", "--model", first->model.model_path});
    std::vector<char *> argv;
    for (auto &argument : args)
        argv.push_back(argument.data());
    const auto parsed = OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
    ASSERT_TRUE(parsed.shared_predictor);
    EXPECT_EQ(parsed.shared_predictor->path(), kPredictorModelPath);
    EXPECT_EQ(parsed.model_path, kMainModelPath);
    EXPECT_EQ(parsed.ngram_table_policy, *first->ngram_table_policy);
    EXPECT_TRUE(parsed.mtp.enabled);
    EXPECT_EQ(parsed.mtp.draft_tokens, 1);
    EXPECT_EQ(parsed.mtp.graph_capacity_draft_tokens, 15);
    EXPECT_EQ(parsed.mtp.depth_policy.mode, MTPDepthPolicyMode::Fixed);
}
} // namespace llaminar2::test::parity::qwen4exp

namespace llaminar2::test::parity::qwen4exp
{
/** @test Both real QFN definitions retain numerical tests but only main owns canonical controls. */
TEST(Qwen4ExpMTPNumericalCandidates, InventoryRoleSeparatesNumericalAndRuntimeControlAuthority)
{
    std::set<std::string> canonical, diagnostics;
    for (const auto &cell : qwen4ExpProductionCases())
    {
        EXPECT_EQ(cell.inventory_role, ModelParityInventoryRole::Canonical);
        std::ostringstream original;
        PrintTo(cell, &original);
        const auto record = nlohmann::json::parse(original.str());
        EXPECT_FALSE(record.contains("inventory_role"));
        EXPECT_TRUE(canonical.insert(record.at("id").get<std::string>()).second);
    }
    for (const auto &cell : qwen4ExpMTPNumericalCandidateCases())
    {
        EXPECT_EQ(cell.inventory_role, ModelParityInventoryRole::NumericalDiagnostic);
        std::ostringstream output;
        PrintTo(cell, &output);
        const auto record = nlohmann::json::parse(output.str());
        EXPECT_EQ(record.at("inventory_role"), "numerical_diagnostic");
        EXPECT_TRUE(diagnostics.insert(record.at("id").get<std::string>()).second);
        EXPECT_TRUE(record.contains("shared_predictor"));
    }
    unsigned overlaps = 0;
    for (const auto &id : diagnostics) overlaps += canonical.count(id);
    EXPECT_EQ(overlaps, 4u); // Distinct paired-source Off records must not be first-wins deduplicated.
    const auto other_definition = qwen36::qwen36DenseParityDefinition(
        qwen36::qwen36SingleDeviceTopology("CUDA0", GlobalDeviceAddress::cuda(0)), "fixture/reference");
    for (const auto &cell : expandModelParityDefinition(other_definition))
    {
        EXPECT_EQ(cell.inventory_role, ModelParityInventoryRole::Canonical);
        EXPECT_TRUE(canonical.insert(cell.testName()).second);
    }
    EXPECT_EQ(canonical.size(), 26u);
    EXPECT_EQ(diagnostics.size(), 24u);
    auto invalid = qwen4ExpMTPNumericalCandidateDefinition(Qwen4ExpParityTopology::CPU);
    invalid.inventory_role = static_cast<ModelParityInventoryRole>(255);
    EXPECT_THROW((void)expandModelParityDefinition(invalid), std::invalid_argument);
    auto invalid_cell = qwen4ExpProductionCases().front();
    invalid_cell.inventory_role = static_cast<ModelParityInventoryRole>(255);
    std::ostringstream rejected;
    EXPECT_THROW(PrintTo(invalid_cell, &rejected), std::invalid_argument);
    EXPECT_TRUE(rejected.str().empty());
}
} // namespace llaminar2::test::parity::qwen4exp
