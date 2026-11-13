/**
 * @file Test__ModelParityPairedWorkload.cpp
 * @brief Device-free propagation and discovery tests of paired document obligations.
 *
 * Document paths/digests below are synthetic metadata, never approvals or model
 * evidence. Initial/calibrated variants retain mutually exclusive document
 * obligations; neither test branch certifies accuracy. Actual canonical definitions and the public parser supply topology;
 * these tests open no model, corpus, backend, calibration or benchmark files.
 */
#include "integration/parity/ModelParityDefinition.h"
#include "integration/parity/qwen4exp/Qwen4ExpProductionParityCases.h"
#include "integration/parity/qwen38/Qwen38ModelParityDefinitions.h"
#include "config/OrchestrationConfigParser.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

namespace llaminar2::test::parity
{
namespace
{
/** @return Synthetic pinned document reference, with no approval assertion. */
ModelParityDocumentReference reference(const char *name)
{
    return {std::string("fixture-metadata/") + name, std::string(64, 'a')};
}
/** @return Complete reviewed-source obligations bound to the original model. */
ModelParityPairedWorkload obligations(const ModelParityModelDefinition &model)
{
    return {.role = ModelParityPairRole::Required,
        .reviewed_main_source = model.model_path,
        .reviewed_predictor_source = model.shared_predictor
            ? std::optional<std::string>(model.shared_predictor->path()) : std::nullopt,
        .accuracy_catalog = reference("catalog.json"),
        .accuracy_workload = reference("workload.json"),
        .accuracy_policy = ModelParityCalibratedRevision{reference("baseline.json"),
            reference("approval.json"), reference("calibration.json")},
        .benchmark_profile = reference("benchmark.json")};
}
/** @return Existing production CPU definition without adding a topology. */
ModelParityDefinition definition()
{
    return qwen4exp::qwen4ExpProductionDefinition(qwen4exp::Qwen4ExpParityTopology::CPU);
}
/** @return Actual ADL discovery object without invoking native discovery. */
nlohmann::json exported(const ModelParityCase &cell)
{
    std::ostringstream out;
    PrintTo(cell, &out);
    return nlohmann::json::parse(out.str());
}
}

TEST(ModelParityPairedWorkload, PropagatesWithoutMultiplyingOrRenamingOriginalCases)
{
    auto original = definition();
    const auto controls = expandModelParityDefinition(original);
    original.paired_workload = obligations(original.model);
    const auto paired = expandModelParityDefinition(original);
    ASSERT_EQ(paired.size(), controls.size());
    ASSERT_FALSE(paired.empty());
    for (std::size_t i = 0; i < paired.size(); ++i)
    {
        EXPECT_EQ(paired[i].testName(), controls[i].testName());
        EXPECT_EQ(paired[i].paired_workload, original.paired_workload);
        auto record = exported(paired[i]);
        EXPECT_EQ(record.at("paired_workload").at("role"), "required");
        record.erase("paired_workload");
        EXPECT_EQ(record, exported(controls[i]));
    }
}

TEST(ModelParityPairedWorkload, BothDiagnosticObligationsReferenceOneOriginalArgumentOwner)
{
    auto source = definition();
    source.paired_workload = obligations(source.model);
    source.paired_workload->role = ModelParityPairRole::Expanded;
    const auto cell = expandModelParityDefinition(source).front();
    const auto record = exported(cell);
    const auto &pair = record.at("paired_workload");
    EXPECT_EQ(pair.at("case_id"), record.at("id"));
    EXPECT_EQ(pair.at("policy"), "diagnostic_obligation");
    EXPECT_EQ(pair.at("role"), "expanded");
    EXPECT_EQ(pair.at("certification_eligible"), false);
    EXPECT_EQ(pair.at("source_fields").front(), "model");
    EXPECT_EQ(pair.at("source_fields").size(), cell.model.shared_predictor ? 2u : 1u);
    EXPECT_EQ(pair.at("accuracy").at("arguments_from"), "diagnostic_benchmark");
    EXPECT_EQ(pair.at("benchmark").at("arguments_from"), "diagnostic_benchmark");
    EXPECT_EQ(pair.at("accuracy").at("catalog").at("path"), source.paired_workload->accuracy_catalog.path);
    EXPECT_EQ(pair.at("accuracy").at("workload").at("sha256"), source.paired_workload->accuracy_workload.sha256);
    const auto &revision = std::get<ModelParityCalibratedRevision>(source.paired_workload->accuracy_policy);
    EXPECT_EQ(pair.at("schema"), 2);
    EXPECT_EQ(pair.at("accuracy").at("policy"), "calibrated_revision");
    EXPECT_EQ(pair.at("accuracy").at("baseline").at("path"), revision.baseline.path);
    EXPECT_EQ(pair.at("accuracy").at("baseline").at("sha256"), revision.baseline.sha256);
    EXPECT_EQ(pair.at("accuracy").at("approval").at("path"), revision.approval.path);
    EXPECT_EQ(pair.at("accuracy").at("calibration").at("path"), revision.calibration.path);
    EXPECT_FALSE(pair.at("accuracy").contains("pilot"));
    EXPECT_EQ(pair.at("benchmark").at("profile").at("path"), source.paired_workload->benchmark_profile.path);
    EXPECT_TRUE(record.at("e2e").is_null());
    EXPECT_TRUE(record.at("benchmark").is_null());
    EXPECT_THROW(modelParityBenchmarkArguments(cell), std::invalid_argument);
    auto args = record.at("diagnostic_benchmark").at("args").get<std::vector<std::string>>();
    EXPECT_EQ(args, modelParityDiagnosticBenchmarkArguments(cell));
    args.insert(args.begin(), "llaminar2");
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    const auto parsed = OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
    const auto expected = cell.makeOrchestrationConfig(cell.model.model_path, 0);
    EXPECT_EQ(parsed.default_backend, expected.default_backend);
    EXPECT_EQ(parsed.device_for_this_rank, expected.device_for_this_rank);
    EXPECT_EQ(parsed.activation_precision, expected.activation_precision);
    EXPECT_EQ(parsed.kv_cache_precision, expected.kv_cache_precision);
    EXPECT_EQ(parsed.ngram_table_policy, expected.ngram_table_policy);
    EXPECT_EQ(parsed.mtp.enabled, expected.mtp.enabled);
}

TEST(ModelParityPairedWorkload, EveryRoleRequiresEveryPinnedDocumentBeforeExpansionAndExport)
{
    for (const auto role : {ModelParityPairRole::Required, ModelParityPairRole::Expanded})
        for (const bool initial : {false, true})
            for (int index = 0; index < (initial ? 4 : 6); ++index)
                for (bool missing_path : {false, true})
                {
                    auto source = definition();
                    source.paired_workload = obligations(source.model);
                    source.paired_workload->role = role;
                    auto &pair = *source.paired_workload;
                    if (initial) pair.accuracy_policy = ModelParityInitialAbsolutePilot{reference("pilot.json")};
                    std::vector<ModelParityDocumentReference *> documents = {
                        &pair.accuracy_catalog, &pair.accuracy_workload, &pair.benchmark_profile};
                    if (auto *pilot = std::get_if<ModelParityInitialAbsolutePilot>(&pair.accuracy_policy))
                        documents.push_back(&pilot->pilot);
                    else
                    {
                        auto &revision = std::get<ModelParityCalibratedRevision>(pair.accuracy_policy);
                        documents.insert(documents.end(), {&revision.baseline, &revision.approval, &revision.calibration});
                    }
                    if (missing_path) documents[index]->path.clear();
                    else documents[index]->sha256 = "not-a-digest";
                    EXPECT_THROW(expandModelParityDefinition(source), std::invalid_argument);
                    auto cell = expandModelParityDefinition(definition()).front();
                    cell.paired_workload = pair;
                    std::ostringstream output;
                    EXPECT_THROW(PrintTo(cell, &output), std::invalid_argument);
                    EXPECT_TRUE(output.str().empty());
                }
}

TEST(ModelParityPairedWorkload, InitialPilotRetainsSourcesAndArgumentsWithoutBaselineClaims)
{
    for (const auto role : {ModelParityPairRole::Required, ModelParityPairRole::Expanded})
    {
        auto source = definition();
        const auto controls = expandModelParityDefinition(source);
        auto pair = obligations(source.model);
        pair.role = role;
        pair.accuracy_policy = ModelParityInitialAbsolutePilot{reference("exact-pilot.json")};
        source.paired_workload = pair;
        const auto cells = expandModelParityDefinition(source);
        ASSERT_EQ(cells.size(), controls.size());
        for (std::size_t i = 0; i < cells.size(); ++i)
        {
            EXPECT_EQ(cells[i].testName(), controls[i].testName());
            EXPECT_EQ(cells[i].paired_workload, source.paired_workload);
            auto record = exported(cells[i]);
            const auto &profile = record.at("paired_workload");
            const auto &accuracy = profile.at("accuracy");
            EXPECT_EQ(profile.at("schema"), 2);
            EXPECT_FALSE(profile.at("certification_eligible").get<bool>());
            EXPECT_EQ(accuracy.at("policy"), "initial_absolute_pilot");
            EXPECT_EQ(accuracy.at("pilot").at("path"), "fixture-metadata/exact-pilot.json");
            EXPECT_EQ(accuracy.at("pilot").at("sha256"), std::string(64, 'a'));
            EXPECT_EQ(accuracy.at("catalog").at("path"), pair.accuracy_catalog.path);
            EXPECT_EQ(accuracy.at("workload").at("sha256"), pair.accuracy_workload.sha256);
            EXPECT_EQ(accuracy.at("arguments_from"), "diagnostic_benchmark");
            EXPECT_EQ(profile.at("benchmark").at("profile").at("path"), pair.benchmark_profile.path);
            EXPECT_EQ(profile.at("benchmark").at("arguments_from"), "diagnostic_benchmark");
            EXPECT_FALSE(accuracy.at("approved_baseline").get<bool>());
            EXPECT_FALSE(accuracy.at("calibrated").get<bool>());
            EXPECT_FALSE(accuracy.at("whole_corpus_accepted").get<bool>());
            for (const char *absent : {"baseline", "approval", "calibration"})
                EXPECT_FALSE(accuracy.contains(absent));
            record.erase("paired_workload");
            EXPECT_EQ(record, exported(controls[i]));
        }
    }
}

TEST(ModelParityPairedWorkload, DefaultCalibratedPolicyRequiresActualBaselineLocator)
{
    auto source = definition();
    source.paired_workload = obligations(source.model);
    auto &revision = std::get<ModelParityCalibratedRevision>(source.paired_workload->accuracy_policy);
    // Complete approval/calibration cannot stand in for the baseline's locator.
    revision.baseline = {};
    EXPECT_THROW(expandModelParityDefinition(source), std::invalid_argument);
    auto cell = expandModelParityDefinition(definition()).front();
    cell.paired_workload = source.paired_workload;
    std::ostringstream output;
    EXPECT_THROW(PrintTo(cell, &output), std::invalid_argument);
    EXPECT_TRUE(output.str().empty());
}

TEST(ModelParityPairedWorkload, ForeignMainPredictorAndUnknownRoleCannotBecomeASecondSource)
{
    for (int mutation = 0; mutation < 4; ++mutation)
    {
        auto source = definition();
        source.paired_workload = obligations(source.model);
        auto &pair = *source.paired_workload;
        if (mutation == 0) pair.reviewed_main_source += ".foreign";
        if (mutation == 1) pair.reviewed_predictor_source = "/foreign/predictor.gguf";
        if (mutation == 2)
        {
            // Test either missing original predictor or an undeclared empty role.
            if (source.model.shared_predictor) pair.reviewed_predictor_source.reset();
            else pair.reviewed_predictor_source = "";
        }
        if (mutation == 3) pair.role = static_cast<ModelParityPairRole>(255);
        EXPECT_THROW(expandModelParityDefinition(source), std::invalid_argument);
        auto cell = expandModelParityDefinition(definition()).front();
        cell.paired_workload = pair;
        std::ostringstream output;
        EXPECT_THROW(PrintTo(cell, &output), std::invalid_argument);
        EXPECT_TRUE(output.str().empty());
    }
}

TEST(ModelParityPairedWorkload, AbsentObligationPreservesExistingCertifiedAutomaticProjection)
{
    int tagged = 0;
    for (const auto &source : qwen38::qwen38DenseMultiDeviceDefinitions())
        for (const auto &cell : expandModelParityDefinition(source))
        {
            ASSERT_FALSE(cell.paired_workload);
            EXPECT_FALSE(exported(cell).contains("paired_workload"));
            if (!cell.e2e_certification) continue;
            ++tagged;
            EXPECT_EQ(modelParityBenchmarkArguments(cell), modelParityBenchmarkSemanticArguments(cell,
                model_parity_runtime_detail::BenchmarkPlacement::AutomaticProduction));
        }
    EXPECT_GT(tagged, 0);
    for (const auto &cell : qwen4exp::qwen4ExpProductionCases())
        EXPECT_FALSE(cell.paired_workload);
}
/** @test The profile, not a corpus/environment override, owns all six candidate selectors. */
TEST(ModelParityPairedWorkload, ExplicitQFNArithmeticRequiresCompleteResolvedTypedSelection)
{
    auto source = definition();
    source.paired_workload = obligations(source.model);
    auto &pair = *source.paired_workload;
    pair.qfn_arithmetic = ModelParityQFNArithmetic{GDNProjectionArithmetic::FP16MoEAlphaBeta,
        MoERouterArithmetic::FP16, QuantizedActivationOperandPrecision::TwoTermQ8,
        QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::SingleQ8,
        QuantizedActivationOperandPrecision::SingleQ8};
    const auto cell = expandModelParityDefinition(source).front();
    const auto arithmetic = exported(cell).at("paired_workload").at("arithmetic");
    EXPECT_EQ(arithmetic.size(), 6u);
    EXPECT_EQ(arithmetic.at("hc_up"), "two_term_q8");
    EXPECT_EQ(arithmetic.at("moe_router"), "fp16");
    EXPECT_EQ(arithmetic.at("gdn_qkv"), "single_q8");
    pair.qfn_arithmetic->moe_router = MoERouterArithmetic::ModelDefault;
    EXPECT_THROW(expandModelParityDefinition(source), std::invalid_argument);
    pair.qfn_arithmetic->moe_router = MoERouterArithmetic::FP16;
    pair.qfn_arithmetic->hc_up = static_cast<QuantizedActivationOperandPrecision>(255);
    EXPECT_THROW(expandModelParityDefinition(source), std::invalid_argument);
    pair.qfn_arithmetic.reset();
    EXPECT_FALSE(exported(expandModelParityDefinition(source).front()).at("paired_workload").contains("arithmetic"));
}

} // namespace llaminar2::test::parity
