/**
 * @file Test__PrefillNumericalAggregation.cpp
 * @brief Actual generic prefill aggregation rejects positional-slot contamination.
 *
 * The abstract static probe is never instantiated and invokes no SetUp, runner,
 * model loader or backend. Tiny tensors use the existing shared comparator;
 * finalization is the exact typed method called by production prefill parity.
 */
#include "integration/parity/ParityTestBase.h"
#include "integration/parity/qwen4exp/Qwen4ExpPrefillAcceptance.h"
#include "integration/parity/qwen4exp/Qwen4ExpProductionParityCases.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
namespace llaminar2::test::parity {
namespace {
/** @brief Static access only; no fixture instance or execution owner is created. */
class PrefillNumericsProbe : public ParityTestBase {
public:
    using ParityTestBase::finalizePrefillLayerNumerics;
    using ParityTestBase::finalizePrefillSummaryNumerics;
};
/**
 * @brief Compare two tiny semantic tensors through the actual numerical authority.
 * @param name Exact semantic checkpoint name.
 * @param bad Whether to invert the observed tensor for a failing comparison.
 * @return Complete raw comparison at the unchanged configured threshold.
 */
StageComparisonResult semantic(const std::string &name,bool bad=false) {
    const std::array expected{1.0f,2.0f};
    const std::array actual=bad?std::array{-1.0f,-2.0f}:expected;
    return compareParityTensorData(actual.data(),expected.data(),actual.size(),name,0.99f);
}
/**
 * @brief Retain a positional weighted-contribution diagnostic.
 * @param permuted Whether matching expert contributions exchange router slots.
 * @return Raw comparison including its unchanged adjacent-drop diagnostic.
 */
StageComparisonResult routeSlots(bool permuted) {
    const std::array reference{1.0f,0.0f,0.0f,2.0f},changed{0.0f,2.0f,1.0f,0.0f};
    const auto &actual=permuted?changed:reference;
    auto result=compareParityTensorData(actual.data(),reference.data(),reference.size(),"MOE_ROUTE_CONTRIBUTIONS",0.99f);
    result.cosine_drop=permuted?0.77f:0.13f; // Existing raw CSV diagnostic must remain intact.
    return result;
}
/**
 * @brief Build the real result type consumed by the prefill finalizer.
 * @param permuted Whether the diagnostic-only contribution slots exchange order.
 * @return Good semantic bank with an independently retained positional diagnostic.
 */
LayerStats bank(bool permuted) {
    LayerStats layer;layer.layer_idx=3;
    layer.stage_results={semantic("ATTENTION_OUTPUT"),routeSlots(permuted),semantic("MOE_EXPERT_OUTPUT"),
        semantic("MOE_COMBINED_OUTPUT"),semantic("FFN_RESIDUAL")};
    return layer;
}
}

/** @test Actual prefill finalization ignores slot order and preserves raw diagnostics. */
TEST(PrefillNumericalAggregation,PermutedWeightedSlotsPreserveSemanticAggregateAndRawDiagnostics) {
    const std::array nativeRoutes{9.0f,7.0f},referenceRoutes{7.0f,9.0f},participants{0.0f,0.0f};
    const std::array native{0.0f,2.0f,1.0f,0.0f},reference{1.0f,0.0f,0.0f,2.0f};
    const auto aligned=compareRoutedExpertContribution(nativeRoutes,referenceRoutes,participants,7,0,2,native,reference);
    ASSERT_TRUE(aligned.validGeometry());ASSERT_TRUE(aligned.finite());EXPECT_FLOAT_EQ(aligned.cosine_similarity,1.0f);
    auto original=bank(false),permuted=bank(true);const auto raw=permuted.stage_results;
    for(const bool average:{false,true}) {
        ParityConfig config;config.use_avg_cosine=average;
        PrefillNumericsProbe::finalizePrefillLayerNumerics(original,config);
        PrefillNumericsProbe::finalizePrefillLayerNumerics(permuted,config);
        EXPECT_TRUE(original.passed);EXPECT_TRUE(permuted.passed);EXPECT_EQ(permuted.stages_compared,4);
        EXPECT_FLOAT_EQ(permuted.avg_cosine_sim,original.avg_cosine_sim);
        EXPECT_FLOAT_EQ(permuted.min_cosine_sim,original.min_cosine_sim);
        EXPECT_EQ(permuted.worst_stage,original.worst_stage);
        EXPECT_FLOAT_EQ(permuted.max_cosine_drop,original.max_cosine_drop);
        EXPECT_EQ(permuted.max_drop_stage,original.max_drop_stage);
    }
    ASSERT_EQ(permuted.stage_results.size(),raw.size());
    for(std::size_t i=0;i<raw.size();++i) {
        const auto &before=raw[i],&after=permuted.stage_results[i];
        EXPECT_EQ(after.stage_name,before.stage_name);EXPECT_EQ(after.passed,before.passed);
        EXPECT_FLOAT_EQ(after.cosine_similarity,before.cosine_similarity);EXPECT_FLOAT_EQ(after.cosine_drop,before.cosine_drop);
        EXPECT_FLOAT_EQ(after.rel_l2_norm,before.rel_l2_norm);EXPECT_FLOAT_EQ(after.max_abs_diff,before.max_abs_diff);
        EXPECT_EQ(after.total_elements,before.total_elements);
    }
    EXPECT_FALSE(permuted.stage_results[1].passed);EXPECT_FLOAT_EQ(permuted.stage_results[1].cosine_similarity,0.0f);
    EXPECT_FLOAT_EQ(permuted.stage_results[1].cosine_drop,0.77f);
}

/** @test The actual canonical stage CSV stays byte-for-byte identical after aggregation. */
TEST(PrefillNumericalAggregation,RawStageCSVRemainsUnchanged)
{
    struct TemporaryResults {
        std::filesystem::path path;
        /** @brief Create one isolated ordinary host diagnostic directory. */
        TemporaryResults() {
            const auto pattern=(std::filesystem::temp_directory_path()/"llaminar-prefill-numerics-XXXXXX").string();
            std::vector<char> storage(pattern.begin(),pattern.end());storage.push_back(0);
            const auto *created=::mkdtemp(storage.data());
            if(!created)throw std::runtime_error("Cannot create isolated numerical CSV directory");
            path=created;
        }
        /** @brief Remove only this fixture's own generated CSVs, even after assertion failure. */
        ~TemporaryResults() {std::error_code ec;std::filesystem::remove_all(path,ec);}
    } temporary;
    ParityTestSummary summary;summary.layer_stats.push_back(bank(true));
    ASSERT_TRUE(ParityCSVArtifactWriter::writePrefill(temporary.path/"before","CPU",summary));
    PrefillNumericsProbe::finalizePrefillLayerNumerics(summary.layer_stats.front(),ParityConfig{});
    ASSERT_TRUE(ParityCSVArtifactWriter::writePrefill(temporary.path/"after","CPU",summary));
    std::ifstream before(temporary.path/"before/prefill_stages.csv"),after(temporary.path/"after/prefill_stages.csv");
    ASSERT_TRUE(before.good());ASSERT_TRUE(after.good());
    const std::string rawBefore{std::istreambuf_iterator<char>(before),std::istreambuf_iterator<char>()};
    const std::string rawAfter{std::istreambuf_iterator<char>(after),std::istreambuf_iterator<char>()};
    EXPECT_EQ(rawBefore,rawAfter);
    EXPECT_NE(rawAfter.find("MOE_ROUTE_CONTRIBUTIONS"),std::string::npos);
    EXPECT_FALSE(summary.layer_stats.front().stage_results[1].passed);
}

/** @test Expert/combined/HC/final semantic checkpoints retain ordinary strict comparisons. */
TEST(PrefillNumericalAggregation,BadSemanticExpertCombinedHCAndFinalRemainRed) {
    for(const auto *name:{"MOE_EXPERT_OUTPUT","MOE_COMBINED_OUTPUT","GR_READ","GR_WRITE",
                         "ATTENTION_RESIDUAL","FFN_NORM","FFN_RESIDUAL","LM_HEAD"})
    for(const bool average:{false,true}) {
        auto layer=bank(true);layer.stage_results.push_back(semantic(name,true));ParityConfig config;config.use_avg_cosine=average;
        PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,config);
        EXPECT_FALSE(layer.passed)<<name;EXPECT_EQ(layer.worst_stage,name);
        EXPECT_LT(layer.min_cosine_sim,config.cosine_threshold);EXPECT_FALSE(layer.stage_results.back().passed);
    }
}

/** @test Losing/duplicating experts cannot be excused by excluding positional slots. */
TEST(PrefillNumericalAggregation,MissingOrDuplicatedExpertSemanticSumStillFails) {
    const std::array goldSum{1.0f,2.0f},lost{0.0f,2.0f},duplicated{2.0f,2.0f};
    for(const auto &actual:{lost,duplicated}) {
        auto layer=bank(true);
        layer.stage_results[2]=compareParityTensorData(actual.data(),goldSum.data(),actual.size(),"MOE_EXPERT_OUTPUT",0.99f);
        auto routing=semantic("MOE_ROUTING_INDICES");routing.is_routing_stage=true;
        routing.routing_overlap=0.5f;routing.passed=false;layer.stage_results.push_back(routing);
        for(const bool average:{false,true}) {
            ParityConfig config;config.use_avg_cosine=average;PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,config);
            EXPECT_FALSE(layer.passed);EXPECT_FALSE(layer.stage_results[2].passed);
            EXPECT_FALSE(layer.stage_results.back().passed);EXPECT_FLOAT_EQ(layer.stage_results.back().routing_overlap,0.5f);
        }
    }
}

/** @test Comparable min/worst/drop use the same semantic sequence, not raw slot transitions. */
TEST(PrefillNumericalAggregation,ComparableMinWorstAndDropShareTheSameSemanticSequence) {
    auto layer=bank(true);layer.stage_results[0].cosine_similarity=0.9999f;
    layer.stage_results[2].cosine_similarity=0.9980f;layer.stage_results[3].cosine_similarity=0.9990f;
    layer.stage_results[4].cosine_similarity=0.9975f;
    PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,ParityConfig{});
    EXPECT_EQ(layer.worst_stage,"FFN_RESIDUAL");EXPECT_EQ(layer.max_drop_stage,"MOE_EXPERT_OUTPUT");
    EXPECT_FLOAT_EQ(layer.max_cosine_drop,0.9999f-0.9980f);EXPECT_FLOAT_EQ(layer.stage_results[1].cosine_drop,0.77f);
    EXPECT_FLOAT_EQ(layer.avg_cosine_sim,(0.9999f+0.9980f+0.9990f+0.9975f)/4);
}

/** @test Exact integer identities veto even when excluded from floating aggregate metrics. */
TEST(PrefillNumericalAggregation,IntegerMismatchVetoesExcludedCosine) {
    for(const auto *name:{"PLE_HASH_INDICES","QSA_SELECTED_INDICES","MOE_ROUTING_INDICES"})
    for(const bool average:{false,true}) {
        auto layer=bank(true);auto integer=semantic(name);integer.scalar_type="INT64";
        integer.integer_mismatch_count=1;integer.passed=false;integer.cosine_similarity=std::numeric_limits<float>::quiet_NaN();
        layer.stage_results.push_back(integer);ParityConfig config;config.use_avg_cosine=average;
        PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,config);
        EXPECT_FALSE(layer.passed);EXPECT_EQ(layer.stages_compared,4);EXPECT_FLOAT_EQ(layer.avg_cosine_sim,1.0f);
        EXPECT_EQ(*layer.stage_results.back().integer_mismatch_count,1u);
        layer.stage_results.back().passed=true; // A stale flag cannot hide an exact nonzero mismatch count.
        PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,config);EXPECT_FALSE(layer.passed);
        layer.stage_results.back().integer_mismatch_count=0;
        PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,config);EXPECT_TRUE(layer.passed);EXPECT_EQ(layer.stages_compared,4);
    }
}

/** @test Routing-only/identity-only/empty evidence cannot inherit minimum one and certify parity. */
TEST(PrefillNumericalAggregation,NoComparableNumericalStagesFails) {
    StageComparisonResult integer;integer.stage_name="PLE_HASH_INDICES";integer.passed=true;integer.integer_mismatch_count=0;
    StageComparisonResult empty;empty.stage_name="QSA_INDEXER_POOLED_KEYS";empty.passed=true;
    empty.cosine_similarity=std::numeric_limits<float>::quiet_NaN();
    for(const auto &results:{std::vector<StageComparisonResult>{},std::vector{routeSlots(true)},std::vector{integer},
                            std::vector{empty},std::vector{integer,empty,routeSlots(false)}})
    for(const bool average:{false,true}) {
        LayerStats layer;layer.stage_results=results;ParityConfig config;config.use_avg_cosine=average;
        PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,config);EXPECT_EQ(layer.stages_compared,0);EXPECT_FALSE(layer.passed);
    }
}

/** @test Nonfinite continuous output fails despite a stale true flag or otherwise good stages. */
TEST(PrefillNumericalAggregation,NonfiniteContinuousOutputCannotHideBehindGoodStages) {
    for(int mutation=0;mutation<9;++mutation)for(const bool average:{false,true}) {
        auto layer=bank(true);auto bad=semantic("MOE_COMBINED_OUTPUT");
        const auto nan=std::numeric_limits<float>::quiet_NaN(),inf=std::numeric_limits<float>::infinity();
        if(mutation==0)bad.cosine_similarity=nan;if(mutation==1)bad.cosine_similarity=inf;if(mutation==2)bad.cosine_similarity=-inf;
        if(mutation==3)bad.llaminar_stats.nan_count=1;if(mutation==4)bad.llaminar_stats.inf_count=1;
        if(mutation==5)bad.pytorch_stats.nan_count=1;if(mutation==6)bad.pytorch_stats.inf_count=1;
        if(mutation==7)bad.rel_l2_norm=inf;if(mutation==8)bad.max_abs_diff=nan;
        layer.stage_results.push_back(bad);ParityConfig config;config.use_avg_cosine=average;
        PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,config);
        EXPECT_FALSE(layer.passed)<<mutation;EXPECT_EQ(layer.worst_stage,"MOE_COMBINED_OUTPUT");
        EXPECT_TRUE(layer.stage_results.back().passed)<<"Raw evidence is not rewritten";
    }
}
/** @test Excluding positional cosine never exempts its nonempty continuous finite witnesses. */
TEST(PrefillNumericalAggregation,NonfinitePositionalContributionRemainsRed)
{
    for(int mutation=0;mutation<9;++mutation)for(const bool average:{false,true})
    {
        auto layer=bank(true);auto &raw=layer.stage_results[1];
        const auto nan=std::numeric_limits<float>::quiet_NaN(),inf=std::numeric_limits<float>::infinity();
        if(mutation==0)raw.cosine_similarity=nan;
        if(mutation==1)raw.cosine_similarity=inf;
        if(mutation==2)raw.cosine_similarity=-inf;
        if(mutation==3)raw.llaminar_stats.nan_count=1;
        if(mutation==4)raw.llaminar_stats.inf_count=1;
        if(mutation==5)raw.pytorch_stats.nan_count=1;
        if(mutation==6)raw.pytorch_stats.inf_count=1;
        if(mutation==7)raw.rel_l2_norm=inf;
        if(mutation==8)raw.max_abs_diff=nan;
        raw.passed=true; // A stale stage flag cannot authorize nonfinite data.
        ParityConfig config;config.use_avg_cosine=average;
        PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,config);
        EXPECT_FALSE(layer.passed)<<mutation;EXPECT_EQ(layer.stages_compared,4);
        EXPECT_FLOAT_EQ(layer.avg_cosine_sim,1.0f);EXPECT_FLOAT_EQ(layer.min_cosine_sim,1.0f);
        EXPECT_TRUE(layer.stage_results[1].passed)<<"The complete raw comparison stays unchanged";
        if(mutation==0)
        {
            EXPECT_TRUE(std::isnan(layer.stage_results[1].cosine_similarity));
        }
        if(mutation==1)
        {
            EXPECT_EQ(layer.stage_results[1].cosine_similarity,inf);
        }
        if(mutation==2)
        {
            EXPECT_EQ(layer.stage_results[1].cosine_similarity,-inf);
        }
        if(mutation==3)
        {
            EXPECT_EQ(layer.stage_results[1].llaminar_stats.nan_count,1u);
        }
        if(mutation==4)
        {
            EXPECT_EQ(layer.stage_results[1].llaminar_stats.inf_count,1u);
        }
        if(mutation==5)
        {
            EXPECT_EQ(layer.stage_results[1].pytorch_stats.nan_count,1u);
        }
        if(mutation==6)
        {
            EXPECT_EQ(layer.stage_results[1].pytorch_stats.inf_count,1u);
        }
        if(mutation==7)
        {
            EXPECT_EQ(layer.stage_results[1].rel_l2_norm,inf);
        }
        if(mutation==8)
        {
            EXPECT_TRUE(std::isnan(layer.stage_results[1].max_abs_diff));
        }
    }
}

/** @test Genuine routing-metric and authenticated empty sentinels remain non-numerical. */
TEST(PrefillNumericalAggregation,RoutingAndEmptyMetricSentinelsRemainDistinctFromContinuousData)
{
    for(const bool average:{false,true})
    {
        auto layer=bank(true);const auto nan=std::numeric_limits<float>::quiet_NaN();
        for(const auto *name:{"MOE_ROUTING_INDICES","MOE_ROUTING_WEIGHTS"})
        {
            auto routing=semantic(name);routing.is_routing_stage=true;routing.routing_overlap=1.0f;
            routing.cosine_similarity=routing.rel_l2_norm=routing.max_abs_diff=nan;
            layer.stage_results.push_back(routing);
        }
        StageComparisonResult empty;empty.stage_name="QSA_INDEXER_POOLED_KEYS";empty.passed=true;
        empty.cosine_similarity=empty.rel_l2_norm=empty.max_abs_diff=nan;
        layer.stage_results.push_back(empty);
        ParityConfig config;config.use_avg_cosine=average;
        PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,config);
        EXPECT_TRUE(layer.passed);EXPECT_EQ(layer.stages_compared,4);
        EXPECT_FLOAT_EQ(layer.avg_cosine_sim,1.0f);EXPECT_FLOAT_EQ(layer.min_cosine_sim,1.0f);
        EXPECT_TRUE(std::isnan(layer.stage_results.back().cosine_similarity));
        layer.stage_results.back().passed=false;
        PrefillNumericsProbe::finalizePrefillLayerNumerics(layer,config);EXPECT_FALSE(layer.passed);
    }
}


namespace {
/** @brief Build the real ordinary/TP summary with original source indices and good logits. */
template <typename Summary>
Summary completeQwen4ExpSummary()
{
    Summary summary;
    summary.lm_head_passed = true;
    summary.layer_stats.resize(48);
    for (int index = 0; index < 48; ++index)
    {
        summary.layer_stats[index].layer_idx = index;
        summary.layer_stats[index].passed = true;
    }
    return summary;
}

/**
 * @brief Exercise the actual shared finalizer with QFN's exact production acceptance policy.
 * @tparam Summary Original ordinary or TP numerical summary type.
 */
template <typename Summary>
void requireQwen4ExpCompleteGate()
{
    const auto definition = qwen4exp::qwen4ExpProductionDefinition(qwen4exp::Qwen4ExpParityTopology::CPU);
    ParityConfig config;
    config.early_layers_count = definition.thresholds.early_layers_count;
    config.min_early_layers_passed = definition.thresholds.min_early_layers_passed;
    ASSERT_EQ(config.early_layers_count, 48);
    ASSERT_EQ(config.min_early_layers_passed, 48); // Decode still uses this unchanged minimum.
    const auto finalize = [&](Summary &summary, bool embedding)
    {
        PrefillNumericsProbe::finalizePrefillSummaryNumerics(summary, config, embedding,
            qwen4exp::acceptsQwen4ExpCompletePrefill(summary.layer_stats, embedding, summary.lm_head_passed));
    };
    auto summary = completeQwen4ExpSummary<Summary>();
    finalize(summary, true);
    EXPECT_TRUE(summary.overall_passed);
    EXPECT_EQ(summary.early_layers_passed, 49); // Preserve the historical CSV counter.
    for (int failed = 0; failed < 48; ++failed)
    {
        summary = completeQwen4ExpSummary<Summary>();
        summary.layer_stats[failed].passed = false;
        auto legacy = summary;
        PrefillNumericsProbe::finalizePrefillSummaryNumerics(legacy, config, true);
        EXPECT_TRUE(legacy.overall_passed) << failed; // Concrete old-code negative control.
        finalize(summary, true);
        EXPECT_EQ(summary.early_layers_passed, 48);
        EXPECT_FALSE(summary.overall_passed) << failed;
        EXPECT_FALSE(summary.layer_stats[failed].passed);
    }
    summary = completeQwen4ExpSummary<Summary>();
    finalize(summary, false);
    EXPECT_EQ(summary.early_layers_passed, 48);
    EXPECT_FALSE(summary.overall_passed);
    summary = completeQwen4ExpSummary<Summary>();
    summary.lm_head_passed = false;
    finalize(summary, true);
    EXPECT_FALSE(summary.overall_passed);
    summary = completeQwen4ExpSummary<Summary>();
    summary.layer_stats.pop_back();
    finalize(summary, true);
    EXPECT_FALSE(summary.overall_passed);
    summary = completeQwen4ExpSummary<Summary>();
    summary.layer_stats.back().layer_idx = 46;
    finalize(summary, true);
    EXPECT_FALSE(summary.overall_passed);
    summary = completeQwen4ExpSummary<Summary>();
    summary.layer_stats.back().layer_idx = 48; // Sidecar/foreign source cannot replace main47.
    finalize(summary, true);
    EXPECT_FALSE(summary.overall_passed);
    // Retaining all valid originals does not authorize a49th record.
    for (const int extra_index : {48, -1, 17})
    {
        summary = completeQwen4ExpSummary<Summary>();
        auto extra = summary.layer_stats.front();
        extra.layer_idx = extra_index;
        summary.layer_stats.push_back(extra);
        const auto evidence = qwen4exp::acceptsQwen4ExpCompletePrefill(
            summary.layer_stats, true, summary.lm_head_passed);
        EXPECT_FALSE(evidence);
        const std::string reason = evidence.message();
        EXPECT_NE(reason.find(extra_index == 17 ? "duplicate" : "foreign"), std::string::npos);
        EXPECT_NE(reason.find(std::to_string(extra_index)), std::string::npos);
        finalize(summary, true);
        EXPECT_FALSE(summary.overall_passed);
        EXPECT_EQ(summary.layer_stats.size(), 49);
        for (int index = 0; index < 48; ++index)
        {
            EXPECT_EQ(summary.layer_stats[index].layer_idx, index);
            EXPECT_TRUE(summary.layer_stats[index].passed);
        }
    }
    summary = completeQwen4ExpSummary<Summary>();
    summary.layer_stats.pop_back();
    const auto missing = qwen4exp::acceptsQwen4ExpCompletePrefill(summary.layer_stats, true, true);
    EXPECT_FALSE(missing);
    EXPECT_NE(std::string(missing.message()).find("missing decoder index 47"), std::string::npos);
    summary = completeQwen4ExpSummary<Summary>();
    summary.layer_stats[17].passed = false;
    const auto failed = qwen4exp::acceptsQwen4ExpCompletePrefill(summary.layer_stats, true, true);
    EXPECT_FALSE(failed);
    EXPECT_NE(std::string(failed.message()).find("failed decoder index 17"), std::string::npos);
    summary = completeQwen4ExpSummary<Summary>();
    std::reverse(summary.layer_stats.begin(), summary.layer_stats.end());
    finalize(summary, true);
    EXPECT_TRUE(summary.overall_passed);
    EXPECT_EQ(summary.layer_stats.front().layer_idx, 47);
}
} // namespace

/** @test QFN embedding cannot mask any failed, missing or duplicated ordinary decoder. */
TEST(PrefillNumericalAggregation, Qwen4ExpCompleteOrdinaryPrefillCannotSpendEmbeddingOnDecoder)
{
    requireQwen4ExpCompleteGate<ParityTestSummary>();
}

/** @test The same QFN policy joins the actual TP summary finalizer before prefix acceptance. */
TEST(PrefillNumericalAggregation, Qwen4ExpCompleteTPPrefillCannotSpendEmbeddingOnDecoder)
{
    requireQwen4ExpCompleteGate<TPParityTestSummary>();
}

/** @test Non-QFN default-pass semantics and historical embedding-inclusive counters stay intact. */
TEST(PrefillNumericalAggregation, LegacyPrefillCountingAndThresholdsRemainUnchanged)
{
    ParityConfig config;
    config.early_layers_count = config.min_early_layers_passed = 6;
    auto summary = completeQwen4ExpSummary<ParityTestSummary>();
    summary.layer_stats.resize(6);
    summary.layer_stats[2].passed = false;
    PrefillNumericsProbe::finalizePrefillSummaryNumerics(summary, config, true);
    EXPECT_EQ(summary.early_layers_passed, 6);
    EXPECT_EQ(summary.total_layers_passed, 6);
    EXPECT_TRUE(summary.overall_passed);
    PrefillNumericsProbe::finalizePrefillSummaryNumerics(summary, config, false);
    EXPECT_EQ(summary.early_layers_passed, 5);
    EXPECT_FALSE(summary.overall_passed);
}

} // namespace llaminar2::test::parity
