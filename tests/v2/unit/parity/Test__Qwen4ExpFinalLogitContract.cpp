/**
 * @file Test__Qwen4ExpFinalLogitContract.cpp
 * @brief Device-free actual-comparator and finalizer regressions for QFN main-logit policy.
 *
 * Tiny real floating comparisons are assigned to source-declared observation
 * roles to test aggregation only. They are not native/model shape evidence.
 * The abstract probe never runs SetUp, opens model data or starts a backend.
 */
#include "integration/parity/ParityTestBase.h"
#include "integration/parity/qwen4exp/Qwen4ExpProductionParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpMTPNumericalCandidateCases.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>

namespace llaminar2::test::parity
{
namespace
{
/** @brief Expose only actual static acceptance and comparison-boundary methods. */
class FinalLogitProbe : public ParityTestBase
{
public:
    using ParityTestBase::acceptsFinalLogitDecodeEvidence;
    using ParityTestBase::acceptsFinalLogitRawEvidence;
    using ParityTestBase::finalizeDecodeSummaryNumerics;
    using ParityTestBase::finalizePrefillLayerNumerics;
    using ParityTestBase::finalizePrefillSummaryNumerics;
    using ParityTestBase::retainMissingRequiredParityStage;
};
/** @return The actual full-main definition's thresholds and publication authority. */
ParityConfig config()
{
    const auto definition = qwen4exp::qwen4ExpProductionDefinition(qwen4exp::Qwen4ExpParityTopology::CPU);
    ParityConfig result;
    result.decode_steps = definition.model.decode_steps;
    result.early_layers_count = definition.thresholds.early_layers_count;
    result.min_early_layers_passed = definition.thresholds.min_early_layers_passed;
    result.kl_threshold = definition.thresholds.kl_threshold;
    result.mtp_kl_threshold = definition.thresholds.mtp_kl_threshold;
    result.final_logit_mean_kl = definition.thresholds.final_logit_mean_kl;
    result.graph_snapshot_policy = definition.graph_snapshot_policy;
    return result;
}
/** @return A real finite comparator result, optionally with a deliberately red raw cosine. */
StageComparisonResult observation(const std::string &name, bool red = false)
{
    const std::array expected{1.0f, 2.0f};
    const auto actual = red ? std::array{-1.0f, -2.0f} : expected;
    return compareParityTensorData(actual.data(), expected.data(), actual.size(), name, .99f);
}
/** @return Complete source-declared layer roles using tiny actual comparisons, not copied stage inventories. */
std::vector<LayerStats> layers(const ParityConfig &contract)
{
    std::vector<LayerStats> result;
    for (int index = 0; index < contract.early_layers_count; ++index)
    {
        LayerStats layer; layer.layer_idx = index;
        const auto prefix = "layer" + std::to_string(index) + "_";
        for (const auto &key : contract.graph_snapshot_policy.required_prefill_snapshot_keys)
            if (key.starts_with(prefix)) layer.stage_results.push_back(observation(key.substr(prefix.size()), true));
        FinalLogitProbe::finalizePrefillLayerNumerics(layer, contract);
        result.push_back(std::move(layer));
    }
    return result;
}
/** @return An actual finite final-row observation with independent raw layer diagnostics. */
DecodeStepStats step(int index, const ParityConfig &contract)
{
    const std::array values{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    DecodeStepStats result;
    result.step_idx = index; result.has_logit_data = true;
    result.logit_evidence_valid = !finalParityLogitRow(values.data(), values.size(), values.size()).empty();
    result.kl_divergence = computeKLDivergence(values.data(), values.data(), values.size(), values.size());
    result.cosine_similarity = 1.f; result.layer_stats = layers(contract);
    return result;
}
/** @return Two actual final-row metric observations under the exact main trajectory declaration. */
DecodeParitySummary decode(const ParityConfig &contract)
{
    DecodeParitySummary result; result.steps_total = 2;
    result.step_stats = {step(0, contract), step(1, contract)};
    return result;
}
/** @return Real finite embedding/final-row witnesses and a complete numerically red raw bank. */
ParityTestSummary prefill(const ParityConfig &contract)
{
    const std::array values{1.f, 2.f};
    const auto witness = compareParityTensorData(values.data(), values.data(), values.size(), "EMBEDDING", .99f);
    ParityTestSummary result;
    result.embedding_evidence_valid = witness.evidence_valid;
    result.lm_head_evidence_valid = witness.evidence_valid;
    result.lm_head_kl = computeKLDivergence(values.data(), values.data(), values.size(), values.size());
    result.lm_head_passed = contract.final_logit_mean_kl->accepts(result.lm_head_kl, true);
    result.layer_stats = layers(contract);
    return result;
}
}

/** @test Main2-step policy survives canonical topology/MTP projection; recursive limits/depth remain independent. */
TEST(Qwen4ExpFinalLogits, CanonicalTwoStepsAndMTPBudgetStayIndependent)
{
    for (auto mode : {qwen4exp::Qwen4ExpParityTopology::CPU, qwen4exp::Qwen4ExpParityTopology::CUDACPU,
                      qwen4exp::Qwen4ExpParityTopology::ROCmCPU, qwen4exp::Qwen4ExpParityTopology::CUDAROCmCPU})
    {
        const auto definition = qwen4exp::qwen4ExpProductionDefinition(mode);
        EXPECT_EQ(definition.model.decode_steps, 2);
        ASSERT_TRUE(definition.thresholds.final_logit_mean_kl);
        EXPECT_FLOAT_EQ(definition.thresholds.final_logit_mean_kl->prefill_maximum, .05f);
        EXPECT_FLOAT_EQ(definition.thresholds.final_logit_mean_kl->decode_maximum, .10f);
        const auto mtp = qwen4exp::qwen4ExpMTPNumericalCandidateDefinition(mode);
        EXPECT_EQ(mtp.model.decode_steps, 2);
        EXPECT_EQ(mtp.model.maximum_mtp_draft_depth, kModelParityRequiredMaximumMTPDepth);
        EXPECT_FLOAT_EQ(mtp.thresholds.mtp_kl_threshold.value_or(mtp.thresholds.kl_threshold), .05f);
        EXPECT_TRUE(mtp.thresholds.final_logit_mean_kl.has_value());
        for (const auto &cell : expandModelParityDefinition(definition))
        {
            const auto projected = cell.toTestConfig();
            EXPECT_EQ(projected.decode_steps, 2);
            ASSERT_TRUE(projected.thresholds.final_logit_mean_kl);
            EXPECT_FLOAT_EQ(projected.thresholds.final_logit_mean_kl->decode_maximum, .10f);
        }
    }
    auto invalid = qwen4exp::qwen4ExpProductionDefinition(qwen4exp::Qwen4ExpParityTopology::CPU);
    invalid.thresholds.final_logit_mean_kl->decode_steps = 3;
    EXPECT_THROW(expandModelParityDefinition(invalid), std::invalid_argument);
    invalid.thresholds.final_logit_mean_kl->decode_steps = 2;
    invalid.thresholds.final_logit_mean_kl->decode_maximum = std::numeric_limits<float>::infinity();
    EXPECT_THROW(expandModelParityDefinition(invalid), std::invalid_argument);
    EXPECT_FALSE(ParityConfig{}.final_logit_mean_kl);
    EXPECT_FALSE(BackendThresholds{}.final_logit_mean_kl);
}

/** @test Vocabulary KL is averaged by original positions; top5 means intersection/5, not reference-token recall. */
TEST(Qwen4ExpFinalLogits, ActualRowMeansAndTopKAgreementHaveTransparentDefinitions)
{
    const std::array reference{6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f};
    const std::array actual{6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f, 6.f, 5.f, 4.f, 3.f, 2.f};
    const float first = computeKLDivergence(actual.data(), reference.data(), 6, 6);
    const float last = computeKLDivergence(actual.data() + 6, reference.data() + 6, 6, 6);
    EXPECT_FLOAT_EQ(computeKLDivergence(actual.data(), reference.data(), 12, 6), (first + last) / 2.f);
    const auto final_row = finalParityLogitRow(actual.data(), actual.size(), 6);
    ASSERT_EQ(final_row.size(), 6u);
    EXPECT_EQ(final_row.data(), actual.data() + 6);
    EXPECT_FLOAT_EQ(computeTopKOverlap(actual.data(), reference.data(), 12, 6, 1), .5f);
    EXPECT_FLOAT_EQ(computeTopKOverlap(actual.data(), reference.data(), 12, 6, 5), .9f);
    EXPECT_FLOAT_EQ(pytorchTop1InLlaminarTopK(actual.data(), reference.data(), 12, 6, 5), .5f);
}

/** @test Each original step must pass independently; neither a good cosine nor across-step mean bypasses KL. */
TEST(Qwen4ExpFinalLogits, EveryStepAndStrictBudgetBoundaryAreRequired)
{
    auto cfg = config(); auto summary = decode(cfg);
    summary.step_stats[0].kl_divergence = .01f;
    summary.step_stats[1].kl_divergence = .15f; // Across-step mean .08 is below .10.
    FinalLogitProbe::finalizeDecodeSummaryNumerics(summary, cfg, true);
    EXPECT_FALSE(summary.overall_passed);
    for (float invalid : {.10f, -.001f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
    {
        summary.step_stats[1].kl_divergence = invalid;
        EXPECT_FALSE(FinalLogitProbe::acceptsFinalLogitDecodeEvidence(summary, *cfg.final_logit_mean_kl));
    }
    summary.step_stats[1].kl_divergence = .099f;
    FinalLogitProbe::finalizeDecodeSummaryNumerics(summary, cfg, true);
    EXPECT_TRUE(summary.overall_passed);
    auto prompt = prefill(cfg); prompt.lm_head_kl = .05f;
    FinalLogitProbe::finalizePrefillSummaryNumerics(prompt, cfg, true, true);
    EXPECT_FALSE(prompt.overall_passed);
}

/** @test Missing/duplicate/truncated final rows reject, including default-invalid evidence at low KL. */
TEST(Qwen4ExpFinalLogits, CompleteOriginalTrajectoryAndFiniteOperandsCannotBeOmitted)
{
    auto cfg = config(); const auto good = decode(cfg);
    for (int mutation = 0; mutation != 6; ++mutation)
    {
        auto summary = good;
        switch (mutation)
        {
        case 0: summary.step_stats.pop_back(); break;
        case 1: summary.steps_total = 1; break;
        case 2: summary.step_stats[1].step_idx = 0; break;
        case 3: summary.step_stats[1].has_logit_data = false; break;
        case 4: summary.step_stats[1].logit_evidence_valid = false; break;
        case 5: summary.step_stats.clear(); summary.steps_total = 0; break;
        }
        EXPECT_FALSE(FinalLogitProbe::acceptsFinalLogitDecodeEvidence(summary, *cfg.final_logit_mean_kl));
    }
    std::array finite{1.f, 2.f, 3.f, 4.f};
    EXPECT_TRUE(finalParityLogitRow(nullptr, 4, 4).empty());
    EXPECT_TRUE(finalParityLogitRow(finite.data(), 3, 4).empty());
    EXPECT_TRUE(finalParityLogitRow(finite.data(), 4, 0).empty());
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
    {
        finite[0] = invalid;
        EXPECT_TRUE(finalParityLogitRow(finite.data(), 4, 4).empty());
    }
}

/** @test Real low-cosine raw results stay red; declared missing/nonfinite evidence still rejects low final KL. */
TEST(Qwen4ExpFinalLogits, RawDiagnosticRedsRemainVisibleButMissingDeclaredStageRejects)
{
    auto cfg = config(); auto summary = prefill(cfg);
    ASSERT_FALSE(summary.layer_stats[0].passed);
    const auto raw = summary.layer_stats[0].stage_results[0];
    FinalLogitProbe::finalizePrefillSummaryNumerics(summary, cfg, false, true);
    EXPECT_TRUE(summary.overall_passed);
    EXPECT_FALSE(summary.layer_stats[0].stage_results[0].passed);
    EXPECT_FLOAT_EQ(summary.layer_stats[0].stage_results[0].cosine_similarity, raw.cosine_similarity);
    const auto name = summary.layer_stats[0].stage_results[0].stage_name;
    summary.layer_stats[0].stage_results.erase(summary.layer_stats[0].stage_results.begin());
    EXPECT_FALSE(FinalLogitProbe::acceptsFinalLogitRawEvidence(summary.layer_stats, cfg, ParityForwardPhase::Prefill));
    EXPECT_TRUE(FinalLogitProbe::retainMissingRequiredParityStage(summary.layer_stats[0], "layer0_" + name, name,
        cfg.graph_snapshot_policy.required_prefill_snapshot_keys));
    FinalLogitProbe::finalizePrefillSummaryNumerics(summary, cfg, true, true);
    EXPECT_FALSE(summary.overall_passed);
    EXPECT_FALSE(summary.layer_stats[0].stage_results.back().evidence_valid);
}

/** @test All48 original identities are required, with no extra/foreign record substituting for missing evidence. */
TEST(Qwen4ExpFinalLogits, MissingDuplicateAndForeignOriginalDecoderRecordsReject)
{
    auto cfg = config(); const auto good = layers(cfg);
    for (int mutation = 0; mutation != 4; ++mutation)
    {
        auto records = good;
        if (mutation == 0) records.pop_back();
        if (mutation == 1) records[47].layer_idx = 46;
        if (mutation == 2) records.push_back(good[0]);
        if (mutation == 3) records[47].layer_idx = -1;
        EXPECT_FALSE(FinalLogitProbe::acceptsFinalLogitRawEvidence(records, cfg, ParityForwardPhase::Prefill));
    }
    auto reordered = good; std::reverse(reordered.begin(), reordered.end());
    EXPECT_TRUE(FinalLogitProbe::acceptsFinalLogitRawEvidence(reordered, cfg, ParityForwardPhase::Prefill));
}

/** @test Nonfinite optional/raw values and exact integer mismatches cannot hide behind acceptable logits. */
TEST(Qwen4ExpFinalLogits, InvalidFloatOrExactIntegerWitnessRejectsAllNumericalWaivers)
{
    auto cfg = config();
    const std::array expected{1.f, 2.f};
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
    {
        for (bool bad_reference : {false, true})
        {
            auto summary = prefill(cfg); const std::array invalid{bad, 2.f};
            summary.layer_stats[0].stage_results.push_back(compareParityTensorData(
                bad_reference ? expected.data() : invalid.data(), bad_reference ? invalid.data() : expected.data(),
                2, "MOE_ROUTE_CONTRIBUTIONS", .99f));
            FinalLogitProbe::finalizePrefillSummaryNumerics(summary, cfg, true, true);
            EXPECT_FALSE(summary.overall_passed);
        }
    }
    auto summary = prefill(cfg);
    auto integer = observation("PLE_HASH_INDICES"); integer.integer_mismatch_count = 1;
    summary.layer_stats[1].stage_results.push_back(integer);
    FinalLogitProbe::finalizePrefillSummaryNumerics(summary, cfg, true, true);
    EXPECT_FALSE(summary.overall_passed);
    summary = prefill(cfg); summary.embedding_evidence_valid = false;
    FinalLogitProbe::finalizePrefillSummaryNumerics(summary, cfg, true, true);
    EXPECT_FALSE(summary.overall_passed);
}

/** @test Different valid expert sets are numerical drift; malformed physical route rows remain invalid. */
TEST(Qwen4ExpFinalLogits, RoutingRepresentationValidityIsSeparateFromExpectedMembership)
{
    const std::array a{1.f, 2.f}, b{1.f, 3.f};
    EXPECT_TRUE(validParityRouteIds(a, 2, 512));
    EXPECT_TRUE(validParityRouteIds(b, 2, 512));
    for (const auto bad : {std::array{1.f, 1.f}, std::array{-1.f, 2.f}, std::array{.5f, 2.f}, std::array{1.f, 512.f},
                           std::array{1.f, std::numeric_limits<float>::quiet_NaN()}})
    {
        EXPECT_FALSE(validParityRouteIds(bad, 2, 512));
    }
    EXPECT_FALSE(validParityRouteIds(a, 3, 512));
    EXPECT_TRUE(parityRouteFrameHasExactExtents(a, b, a, b));
    EXPECT_FALSE(parityRouteFrameHasExactExtents(a, b, std::span<const float>(a).first(1), b));
    EXPECT_FALSE(parityRouteFrameHasExactExtents(a, b, a, std::span<const float>{}));
    const auto crop = RouteContributionReferenceWindow::checked(4, 2, 2);
    const std::array full_ids{9.f, 8.f, 1.f, 3.f};
    const auto same_window = crop.projectIds(full_ids, 2, 2, 2);
    EXPECT_TRUE(parityRouteFrameHasExactExtents(a, b, a, same_window));
    EXPECT_THROW(crop.projectIds(b, 2, 2, 2), std::invalid_argument);
    EXPECT_TRUE(validNormalizedParityWeights(std::array{.3f, .7f}, 2));
    EXPECT_FALSE(validNormalizedParityWeights(std::array{0.f, 0.f}, 2));
    EXPECT_FALSE(validNormalizedParityWeights(std::array{-.1f, .7f}, 2));
    EXPECT_FALSE(validNormalizedParityWeights(std::array{.1f, 1.1f}, 2));
}

/** @test TP and ordinary use the same actual total evidence finalizer, preserving absent-contract legacy gates. */
TEST(Qwen4ExpFinalLogits, TPAndLegacyControlsDoNotAcquireNumericalWaivers)
{
    auto cfg = config(); const auto ordinary = prefill(cfg); TPParityTestSummary tp;
    tp.embedding_evidence_valid = true; tp.lm_head_evidence_valid = true; tp.lm_head_passed = true;
    for (const auto &source : ordinary.layer_stats)
    {
        TPLayerStats layer; layer.layer_idx = source.layer_idx; layer.stages_compared = source.stages_compared;
        for (const auto &record : source.stage_results)
        {
            TPStageComparisonResult combined; combined.stage_name = record.stage_name; combined.combined_result = record;
            layer.stage_results.push_back(std::move(combined));
        }
        tp.layer_stats.push_back(std::move(layer));
    }
    FinalLogitProbe::finalizePrefillSummaryNumerics(tp, cfg, false, true);
    EXPECT_TRUE(tp.overall_passed);
    tp.layer_stats[0].stage_results.pop_back();
    FinalLogitProbe::finalizePrefillSummaryNumerics(tp, cfg, true, true);
    EXPECT_FALSE(tp.overall_passed);
    auto legacy = ordinary; cfg.final_logit_mean_kl.reset();
    FinalLogitProbe::finalizePrefillSummaryNumerics(legacy, cfg, true, true);
    EXPECT_FALSE(legacy.overall_passed);
    auto decoded = decode(config());
    FinalLogitProbe::finalizeDecodeSummaryNumerics(decoded, config(), true);
    EXPECT_TRUE(decoded.overall_passed); // Top1/top5/default top-k counts are deliberately0: report only.
    FinalLogitProbe::finalizeDecodeSummaryNumerics(decoded, cfg, true);
    EXPECT_FALSE(decoded.overall_passed);
}
} // namespace llaminar2::test::parity
