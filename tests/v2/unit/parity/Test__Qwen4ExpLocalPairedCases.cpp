/**
 * @file Test__Qwen4ExpLocalPairedCases.cpp
 * @brief Device-free exact-case local profile attachment and no-default-change guards.
 *
 * All document/source descriptors are synthetic passive metadata. The actual
 * canonical case factories and parser-intent helper run without loading a
 * model, opening a reference, initializing a backend or issuing an approval.
 */
#include "integration/parity/qwen4exp/Qwen4ExpLocalPairedCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "tools/ModelKLDQFNContract.h"
#include <gtest/gtest.h>
#include <sstream>

namespace llaminar2::test::parity::qwen4exp
{
namespace
{
using json = nlohmann::json;
/** @return Complete toy references, never files or actual baseline evidence. */
json reference(const char *name) { return {{"path",std::string("synthetic/")+name},{"sha256",std::string(64,'a')}}; }
/** @return One candidate obligation bound to a real original ID without deriving topology from that ID. */
json binding(const std::string &id)
{
    return {{"exact_cell",id},{"role","required"},{"catalog",reference("catalog.json")},
        {"workload",reference("manifest.json")},{"pilot",reference("pilot.json")},
        {"benchmark",reference("benchmark.json")},{"arithmetic",{
            {"gdn_projection","fp16_moe_alpha_beta"},{"moe_router","fp16"},{"hc_up","two_term_q8"},
            {"gdn_output","single_q8"},{"fa_output","single_q8"},{"gdn_qkv","single_q8"}}}};
}
/** @return The five actual source selections; every predicate uses typed original values. */
json document(const QFNCaseFamilies &families)
{
    json selected = json::array();
    for (const auto &[family,cases] : families)
        for (const auto &cell : cases)
        {
            const bool mmap = cell.ngram_table_policy && cell.ngram_table_policy->storage == ImmutableRowStoragePreference::MemoryMapped;
            const bool main = family == "Qwen4ExpMain" && mmap &&
                (cell.topology.kind == ModelParityTopologyKind::SingleDevice ||
                 cell.topology.kind == ModelParityTopologyKind::RankLocalPipelineParallel);
            const bool adaptive = family == "Qwen4ExpGPUAdaptive" && cell.ngram_table_policy &&
                cell.ngram_table_policy->gpu_adaptive.maximum_gpu_cache_rows == 0;
            if (main || adaptive || family == "Qwen4ExpFullROCmTP4")
                selected.push_back(binding(family+"/Qwen4ExpMainParityTest.ProductionParity/"+cell.testName()));
        }
    // Complete identity bytes are frozen by the configure helper. This pure
    // registration fixture deliberately uses no physical source-stat claim.
    return {{"schema","llaminar.qfn-local-paired-profiles.v1"},
        {"source",{{"architecture","qwen4exp"},{"main_layers",48},{"expert_count",512},
            {"shards",json::array({{{"path",kMainModelPath}},json::object(),json::object(),json::object()})},
            {"shared_predictor",{{"path",kPredictorModelPath}}}}},{"bindings",selected}};
}
}

TEST(Qwen4ExpLocalPairedCases, FiveExactCasesKeepAllOrdinaryAxesAndParserIntent)
{
    const auto original = qfnOriginalCaseFamilies();
    auto input = document(original);
    // The source-owned CPU pilot changes numerical acceptance only. Bind the
    // literal original case; ordinary topology, arithmetic and siblings stay
    // under their existing declarative owners.
    const std::string cpu_case = "Qwen4ExpMain/Qwen4ExpMainParityTest.ProductionParity/"
        "Qwen38FlashNext_UDQ4KXL_CPU0_NgramMmap_ActFP32_KVFP16_MTPOff";
    const json cpu_pilot = {{"path","tests/v2/integration/parity/qwen4exp/policies/cpu-mmap-hc-up-two-term-epoch-row-kl-1p2.json"},
        {"sha256","7d9656ef9676f49494a15c21a079d1477093841b0404e42b18099e91a1bf1e3e"}};
    size_t cpu_bindings = 0;
    for (auto &entry : input.at("bindings"))
        if (entry.at("exact_cell") == cpu_case)
        {
            entry["pilot"] = cpu_pilot;
            ++cpu_bindings;
        }
    ASSERT_EQ(cpu_bindings,1u);
    ASSERT_EQ(input.at("bindings").size(),5u);
    const auto arithmetic_before = accuracy::qfnActualArithmeticIntent();
    const auto selected = qfnAttachLocalPairedCases(original,input);
    ASSERT_EQ(selected.size(),original.size());
    size_t profiled = 0;
    size_t deferred = 0;
    for (const auto &[family,cases] : original)
    {
        const auto &result = selected.at(family);
        ASSERT_EQ(result.size(),cases.size());
        for (size_t i=0;i<cases.size();++i)
        {
            EXPECT_EQ(result[i].testName(),cases[i].testName());
            EXPECT_EQ(result[i].graph_snapshot_policy.required_prefill_snapshot_keys,cases[i].graph_snapshot_policy.required_prefill_snapshot_keys);
            EXPECT_EQ(result[i].graph_snapshot_policy.required_decode_snapshot_keys,cases[i].graph_snapshot_policy.required_decode_snapshot_keys);
            if (cases[i].topology.needsObservedPipelineBinding())
            {
                // Registration retains an observed request, not an invented
                // physical rank/device config. Attachment must preserve the
                // entire declaration and its fail-closed runtime boundary.
                ++deferred;
                EXPECT_TRUE(result[i].topology.needsObservedPipelineBinding());
                EXPECT_FALSE(result[i].paired_workload);
                std::ostringstream before_declaration, after_declaration;
                PrintTo(cases[i], &before_declaration);
                PrintTo(result[i], &after_declaration);
                EXPECT_EQ(before_declaration.str(), after_declaration.str());
                EXPECT_THROW((void)cases[i].toTestConfig(), std::logic_error);
                EXPECT_THROW((void)result[i].toTestConfig(), std::logic_error);
                continue;
            }
            const auto before = cases[i].makeOrchestrationConfig(cases[i].model.model_path,0);
            const auto after = result[i].makeOrchestrationConfig(result[i].model.model_path,0);
            EXPECT_EQ(accuracy::qfnCanonicalConfigurationIntent(before),accuracy::qfnCanonicalConfigurationIntent(after));
            if (!result[i].paired_workload)
            {
                EXPECT_EQ(modelParityDiagnosticBenchmarkArguments(result[i]),modelParityDiagnosticBenchmarkArguments(cases[i]));
                EXPECT_EQ(result[i].model.shared_predictor,cases[i].model.shared_predictor);
                continue;
            }
            ++profiled;
            ASSERT_TRUE(result[i].model.shared_predictor);
            EXPECT_EQ(result[i].model.shared_predictor->path(),kPredictorModelPath);
            ASSERT_TRUE(result[i].paired_workload->qfn_arithmetic);
            EXPECT_EQ(result[i].paired_workload->qfn_arithmetic->hc_up,QuantizedActivationOperandPrecision::TwoTermQ8);
            const auto *pilot = std::get_if<ModelParityInitialAbsolutePilot>(&result[i].paired_workload->accuracy_policy);
            ASSERT_NE(pilot,nullptr);
            if (family+"/Qwen4ExpMainParityTest.ProductionParity/"+result[i].testName() == cpu_case)
            {
                EXPECT_EQ(pilot->pilot.path,cpu_pilot.at("path").get<std::string>());
                EXPECT_EQ(pilot->pilot.sha256,cpu_pilot.at("sha256").get<std::string>());
                EXPECT_EQ(result[i].paired_workload->qfn_arithmetic->gdn_output,QuantizedActivationOperandPrecision::SingleQ8);
                EXPECT_EQ(result[i].paired_workload->qfn_arithmetic->gdn_qkv,QuantizedActivationOperandPrecision::SingleQ8);
            }
            else EXPECT_EQ(pilot->pilot.path,"synthetic/pilot.json");
            EXPECT_EQ(result[i].mtp,ModelParityMTP::Off);
            EXPECT_EQ(result[i].retained_mtp_draft_capacity,cases[i].retained_mtp_draft_capacity);
        }
    }
    EXPECT_GT(deferred,0u);
    EXPECT_EQ(profiled,5u);
    EXPECT_EQ(accuracy::qfnActualArithmeticIntent(),arithmetic_before);
    for (const auto &[family,cases] : original)
        for (const auto &cell : cases) EXPECT_FALSE(cell.paired_workload);
}

TEST(Qwen4ExpLocalPairedCases, UnknownUnusedDuplicateAndForeignSourceRejectBeforePublication)
{
    const auto original = qfnOriginalCaseFamilies();
    for (int mutation=0;mutation<6;++mutation)
    {
        auto input = document(original);
        if (mutation==0) input["bindings"][0]["exact_cell"]="Unknown.ProductionParity/Unknown";
        if (mutation==1) input["bindings"].push_back(input["bindings"][0]);
        if (mutation==2) input["source"]["shards"][0]["path"]="foreign.gguf";
        if (mutation==3) input["source"]["shared_predictor"]["path"]="foreign-mtp.gguf";
        if (mutation==4) input["bindings"][0]["arithmetic"]["hc_up"]="foreign";
        if (mutation==5) input["bindings"][0]["role"]="foreign";
        EXPECT_THROW((void)qfnAttachLocalPairedCases(original,input),std::invalid_argument) << mutation;
    }
    auto missing = document(original);
    missing["source"].erase("shared_predictor");
    EXPECT_THROW((void)qfnAttachLocalPairedCases(original,missing),std::invalid_argument);
    for (const auto &[family,cases] : original)
        for (const auto &cell : cases) EXPECT_FALSE(cell.paired_workload);
}

TEST(Qwen4ExpLocalPairedCases, UnsetBuildOptionReturnsTheOriginalFamilyDirectly)
{
#ifndef LLAMINAR_QFN_LOCAL_PAIRED_BINDINGS
    const auto &ordinary = qwen4ExpProductionCases();
    EXPECT_EQ(&qfnLocalPairedFamily("Qwen4ExpMain",ordinary),&ordinary);
#endif
}
/** @test Configuration projection only: actual panel execution remains a separate numerical obligation. */
TEST(Qwen4ExpLocalPairedCases, LocalHotNumericalPoliciesRequireExactPairedBindings)
{
    const auto families = qfnOriginalCaseFamilies();
    const auto& cases = families.at("Qwen4ExpLocalHotNumerical");
    ASSERT_EQ(cases.size(), 6u);
    // Reuse the already-published functional profiles exactly, rather than
    // manufacturing separate numerical capacity or source policies.
    const auto functional = qwen4ExpLocalHotBudgetOrdinaryCases();
    ASSERT_EQ(functional.size(), 5u);
    for (std::size_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(cases[i].testName(), functional[i].testName());
        EXPECT_EQ(cases[i].ngram_table_policy, functional[i].ngram_table_policy);
    }
    const auto cuda = qwen4ExpCUDALocalHotOrdinaryCases();
    ASSERT_EQ(cuda.size(), 2u);
    EXPECT_EQ(cases.back().testName(), cuda.back().testName());
    EXPECT_EQ(cases.back().ngram_table_policy, cuda.back().ngram_table_policy);
    auto input = document(families);
    input["bindings"] = json::array();
    for (const auto& cell : cases)
    {
        EXPECT_EQ(cell.mtp, ModelParityMTP::Off);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, 3);
        EXPECT_EQ(cell.expert_overlay, std::optional{ModelParityExpertOverlayPolicy{}});
        EXPECT_THROW(qfnRequireLocalHotNumericalBinding(cell), std::invalid_argument);
        input["bindings"].push_back(binding(qfnLocalPairedCellId("Qwen4ExpLocalHotNumerical", cell)));
    }
    const auto selected = qfnAttachLocalPairedCases(families, input);
    const auto& bound = selected.at("Qwen4ExpLocalHotNumerical");
    std::set<std::string> identities;
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        EXPECT_TRUE(identities.insert(qfnLocalPairedCellId("Qwen4ExpLocalHotNumerical", bound[i])).second);
        EXPECT_NO_THROW(qfnRequireLocalHotNumericalBinding(bound[i]));
        const auto before = cases[i].makeOrchestrationConfig(cases[i].model.model_path, 0);
        const auto after = bound[i].makeOrchestrationConfig(bound[i].model.model_path, 0);
        EXPECT_EQ(accuracy::qfnCanonicalConfigurationIntent(before), accuracy::qfnCanonicalConfigurationIntent(after));
        EXPECT_EQ(bound[i].testName(), cases[i].testName());
        EXPECT_EQ(bound[i].ngram_table_policy, cases[i].ngram_table_policy);
        EXPECT_EQ(bound[i].graph_snapshot_policy.required_prefill_snapshot_keys,
                  cases[i].graph_snapshot_policy.required_prefill_snapshot_keys);
        EXPECT_EQ(bound[i].graph_snapshot_policy.required_decode_snapshot_keys,
                  cases[i].graph_snapshot_policy.required_decode_snapshot_keys);
    }
    auto wrong = input;
    wrong["bindings"][0]["exact_cell"] = "Qwen4ExpLocalHotNumerical/Qwen4ExpMainParityTest.ProductionParity/" + cases[0].testName();
    EXPECT_THROW(qfnAttachLocalPairedCases(families, wrong), std::invalid_argument);
}

} // namespace llaminar2::test::parity::qwen4exp
