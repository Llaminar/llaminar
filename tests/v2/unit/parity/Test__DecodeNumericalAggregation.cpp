/**
 * @file Test__DecodeNumericalAggregation.cpp
 * @brief Ordinary decode's actual finalizer preserves excluded finite and exact-integer witnesses.
 *
 * An abstract static probe calls the same acceptance method used by runDecodeParity.
 * It is never instantiated: no SetUp, model, runner, MPI or backend is initialized.
 * Small raw comparisons retain their original CSV metrics and verdicts.
 */
#include "integration/parity/ParityTestBase.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <string_view>

namespace llaminar2::test::parity
{
namespace
{
/** @brief Expose only the real static phase finalizers, without constructing an execution fixture. */
class DecodeNumericsProbe : public ParityTestBase
{
public:
    using ParityTestBase::finalizeDecodeLayerNumerics;
    using ParityTestBase::finalizePrefillLayerNumerics;
};
/**
 * @brief Produce a complete raw FP32 comparison through the canonical comparator.
 * @param name Semantic checkpoint. @param bad Whether to invert the actual values.
 * @return Original numerical and finite-data witnesses at the unchanged threshold.
 */
StageComparisonResult semantic(const char *name, bool bad = false)
{
    const std::array expected{1.0f, 2.0f};
    const auto actual = bad ? std::array{-1.0f, -2.0f} : expected;
    return compareParityTensorData(actual.data(), expected.data(), expected.size(), name, 0.99f);
}
/** @return Finite changed-route bank with strict comparable combined and residual outputs. */
LayerStats changedRouteBank()
{
    LayerStats result;
    result.layer_idx = 2;
    result.stage_results = {semantic("ATTENTION_OUTPUT"), semantic("MOE_ROUTE_CONTRIBUTIONS", true),
        semantic("MOE_EXPERT_OUTPUT", true), semantic("MOE_COMBINED_OUTPUT"), semantic("FFN_RESIDUAL")};
    result.stage_results[1].cosine_drop = 0.77f;
    return result;
}
/**
 * @brief Compute the old decode eligibility-only average for an explicit negative-control witness.
 * @param layer Original records. @param exact Whether expert sets match.
 * @return Prior scalar verdict with its unchanged decode threshold, excluding no data witnesses.
 */
bool eligibilityOnlyVerdict(const LayerStats &layer, bool exact)
{
    float sum = 0.0f;
    int count = 0;
    for (const auto &stage : layer.stage_results)
    {
        // The old production loop appended integer observations then continued
        // before adding them to its floating sum.
        if (stage.integer_mismatch_count)
            continue;
        if (parityStageContributesToLayerCosine(stage.stage_name, exact))
        {
            sum += stage.cosine_similarity;
            ++count;
        }
    }
    return count > 0 && sum / count >= ParityConfig{}.decode_cosine_threshold;
}
}

/** @test Finite route changes keep the established decode branch policy and raw diagnostics. */
TEST(DecodeNumericalAggregation,FiniteChangedRoutesKeepSemanticAverageAndRawRecords)
{
    auto layer = changedRouteBank();
    const auto raw = layer.stage_results;
    ASSERT_TRUE(eligibilityOnlyVerdict(layer, false));
    DecodeNumericsProbe::finalizeDecodeLayerNumerics(layer, ParityConfig{}, false);
    EXPECT_TRUE(layer.passed);
    EXPECT_EQ(layer.stages_compared, 3);
    EXPECT_FLOAT_EQ(layer.avg_cosine_sim, 1.0f);
    EXPECT_FLOAT_EQ(layer.min_cosine_sim, 1.0f);
    ASSERT_EQ(layer.stage_results.size(), raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        EXPECT_EQ(layer.stage_results[i].stage_name, raw[i].stage_name);
        EXPECT_EQ(layer.stage_results[i].passed, raw[i].passed);
        EXPECT_FLOAT_EQ(layer.stage_results[i].cosine_similarity, raw[i].cosine_similarity);
        EXPECT_FLOAT_EQ(layer.stage_results[i].cosine_drop, raw[i].cosine_drop);
        EXPECT_FLOAT_EQ(layer.stage_results[i].rel_l2_norm, raw[i].rel_l2_norm);
        EXPECT_FLOAT_EQ(layer.stage_results[i].max_abs_diff, raw[i].max_abs_diff);
    }
    EXPECT_FALSE(layer.stage_results[1].passed);
    EXPECT_FALSE(layer.stage_results[2].passed);
}

/** @test Excluded raw slots and changed-route expert sums cannot hide NaN or infinity. */
TEST(DecodeNumericalAggregation,NonfiniteExcludedFloatingWitnessRejectsOldPassingAverage)
{
    for (const auto *name : {"MOE_ROUTE_CONTRIBUTIONS", "MOE_EXPERT_OUTPUT"})
    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()})
    {
        SCOPED_TRACE(::testing::Message() << name << " invalid=" << invalid);
        auto layer = changedRouteBank();
        const std::array actual{invalid, 2.0f}, reference{1.0f, 2.0f};
        const auto index = std::string_view(name) == "MOE_ROUTE_CONTRIBUTIONS" ? 1u : 2u;
        layer.stage_results[index] = compareParityTensorData(
            actual.data(), reference.data(), actual.size(), name, 0.99f);
        ASSERT_TRUE(eligibilityOnlyVerdict(layer, false));
        DecodeNumericsProbe::finalizeDecodeLayerNumerics(layer, ParityConfig{}, false);
        EXPECT_FALSE(layer.passed);
        EXPECT_EQ(layer.stages_compared, 3);
        EXPECT_FLOAT_EQ(layer.avg_cosine_sim, 1.0f);
    }
}

/** @test Integer mismatches veto independently of cosine eligibility and a stale passed flag. */
TEST(DecodeNumericalAggregation,ExactIntegerMismatchRejectsOldPassingAverage)
{
    for (const auto *name : {"QSA_SELECTED_INDICES", "PLE_HASH_INDICES", "MOE_ROUTING_INDICES"})
    for (const bool stalePassed : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << name << " stalePassed=" << stalePassed);
        auto layer = changedRouteBank();
        auto integer = semantic(name);
        integer.scalar_type = "INT64";
        integer.integer_mismatch_count = 1;
        integer.cosine_similarity = std::numeric_limits<float>::quiet_NaN();
        integer.passed = stalePassed;
        layer.stage_results.push_back(integer);
        ASSERT_TRUE(eligibilityOnlyVerdict(layer, false));
        DecodeNumericsProbe::finalizeDecodeLayerNumerics(layer, ParityConfig{}, false);
        EXPECT_FALSE(layer.passed);
        EXPECT_EQ(layer.stages_compared, 3);
        EXPECT_FLOAT_EQ(layer.avg_cosine_sim, 1.0f);
    }
}

/** @test Independent exact integer success and intentional routing metric sentinels stay valid. */
TEST(DecodeNumericalAggregation,ExactIntegerAndRoutingMetricSentinelsRemainAccepted)
{
    auto layer = changedRouteBank();
    auto integer = semantic("QSA_SELECTED_INDICES");
    integer.integer_mismatch_count = 0;
    integer.cosine_similarity = std::numeric_limits<float>::quiet_NaN();
    layer.stage_results.push_back(integer);
    auto routing = semantic("MOE_ROUTING_WEIGHTS");
    routing.rel_l2_norm = routing.max_abs_diff = std::numeric_limits<float>::quiet_NaN();
    layer.stage_results.push_back(routing);
    DecodeNumericsProbe::finalizeDecodeLayerNumerics(layer, ParityConfig{}, false);
    EXPECT_TRUE(layer.passed);
    EXPECT_EQ(layer.stages_compared, 3);
}

/** @test Complete same-route expert sums and every later semantic output retain strict eligibility. */
TEST(DecodeNumericalAggregation,ExpertCombinedHCFinalAndLogitNumericsStayStrict)
{
    auto sameRoutes = changedRouteBank();
    DecodeNumericsProbe::finalizeDecodeLayerNumerics(sameRoutes, ParityConfig{}, true);
    EXPECT_FALSE(sameRoutes.passed);
    EXPECT_EQ(sameRoutes.stages_compared, 4);
    for (const auto *name : {"MOE_COMBINED_OUTPUT", "GR_WRITE", "FFN_RESIDUAL", "FINAL_NORM", "LM_HEAD"})
    {
        SCOPED_TRACE(name);
        auto layer = changedRouteBank();
        layer.stage_results.push_back(semantic(name, true));
        DecodeNumericsProbe::finalizeDecodeLayerNumerics(layer, ParityConfig{}, false);
        EXPECT_FALSE(layer.passed);
        EXPECT_FALSE(layer.stage_results.back().passed);
    }
    auto prefill = changedRouteBank();
    DecodeNumericsProbe::finalizePrefillLayerNumerics(prefill, ParityConfig{});
    EXPECT_FALSE(prefill.passed);
    EXPECT_EQ(prefill.stages_compared, 4);
}

/** @test Decode keeps its original average/threshold even when prefill selects a minimum. */
TEST(DecodeNumericalAggregation,DecodeAverageAndThresholdAreUnchanged)
{
    LayerStats layer;
    layer.stage_results = {semantic("ATTENTION_OUTPUT"), semantic("MOE_COMBINED_OUTPUT")};
    layer.stage_results.back().cosine_similarity = 0.5f;
    ParityConfig config;
    config.use_avg_cosine = false;
    config.cosine_threshold = 0.999f;
    config.decode_cosine_threshold = 0.75f;
    DecodeNumericsProbe::finalizeDecodeLayerNumerics(layer, config, false);
    EXPECT_TRUE(layer.passed);
    EXPECT_FLOAT_EQ(layer.avg_cosine_sim, 0.75f);
    EXPECT_FLOAT_EQ(layer.min_cosine_sim, 0.5f);
    config.decode_cosine_threshold = 0.76f;
    DecodeNumericsProbe::finalizeDecodeLayerNumerics(layer, config, false);
    EXPECT_FALSE(layer.passed);
}

/** @test Routing-only and empty records cannot inherit a passing numerical minimum. */
TEST(DecodeNumericalAggregation,NoComparableNumericalEvidenceFails)
{
    LayerStats layer;
    layer.stage_results = {semantic("MOE_ROUTING_INDICES"), semantic("MOE_ROUTE_CONTRIBUTIONS")};
    DecodeNumericsProbe::finalizeDecodeLayerNumerics(layer, ParityConfig{}, false);
    EXPECT_FALSE(layer.passed);
    EXPECT_EQ(layer.stages_compared, 0);
    auto empty = semantic("EMPTY");
    empty.total_elements = 0;
    empty.passed = false;
    auto good = changedRouteBank();
    good.stage_results.push_back(empty);
    DecodeNumericsProbe::finalizeDecodeLayerNumerics(good, ParityConfig{}, false);
    EXPECT_FALSE(good.passed);
}
} // namespace llaminar2::test::parity
