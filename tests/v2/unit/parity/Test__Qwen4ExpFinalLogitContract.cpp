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
#include <memory>
#include <optional>
#include <vector>

namespace llaminar2::test::parity
{
namespace
{
/** @brief Expose only actual static acceptance and comparison-boundary methods. */
class FinalLogitProbe : public ParityTestBase
{
public:
    using ParityTestBase::acceptsFinalLogitDecodeEvidence;
    using ParityTestBase::acceptsOwnedPipelineRawEvidence;
    using ParityTestBase::pipelineParityLayerObservations;
    using ParityTestBase::acceptsFinalLogitSummaryRawEvidence;
    using ParityTestBase::acceptsFinalLogitRawEvidence;
    using ParityTestBase::finalizeDecodeSummaryNumerics;
    using ParityTestBase::finalizeDecodeLayerNumerics;
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
/**
 * @brief A finite red query comparison with independently supplied original Decode history.
 * @param name Original raw/prepared checkpoint identity from the canonical declaration.
 * @param step Original single-row Decode step in this synthetic aggregation fixture.
 * @return Owned authored512-coordinate evidence, with no model/native shape claim.
 * @details Synthetic original prompt2051 makes the first visible horizon2052.
 * The acquired record independently states prefix2051+step and one useful row;
 * the canonical history validator must authenticate their agreement. Numerical
 * red cosine is retained, just like the other raw aggregation witnesses.
 */
StageComparisonResult decodeQueryObservation(const std::string &name, int step)
{
    std::array<float,512> reference;
    std::vector<float> actual(reference.size());
    for(std::size_t i=0;i<reference.size();++i) {
        reference[i]=1.f+float(i%2);actual[i]=-reference[i];
    }
    auto owner=std::make_shared<const std::vector<float>>(std::move(actual));
    SnapshotInfo snapshot{.data=owner->data(),.size=owner->size(),.rows=1,.cols=512,
        .lifetime_owner=owner,.typed_values=std::span<const float>(*owner),
        .semantic_axis=SnapshotSemanticAxis::AuthoredTokenRows,
        .authored_token_rows=SnapshotAuthoredTokenRowsObservation{
            .requests={{.prefixRows=2051+step,.liveRows=1,.rowOffset=0,.rowCapacity=1,.sourceCapacity=4096,
                .firstRequiredLogicalRow=2051,.firstRow=0,.endRow=1,.emptyReason=SnapshotEmptyReason::None}},
            .physicalRows=1,.repeatsPerRow=1}};
    auto result=compareParityTensorData(snapshot.data,reference.data(),reference.size(),name,.99f);
    result.checkpoint_evidence_kind=ParityCheckpointEvidenceKind::AuthoredQuery;
    result.query_work_evidence=inspectQwenSparseQueryEvidence(snapshot,
        {.promptRows=2051,.decodeRowsBefore=std::size_t(step),.liveRows=1,.sourceCapacity=4096},reference.size());
    return result;
}
/**
 * @brief Build every source-declared role using actual finite fixture comparisons.
 * @param contract Original phase-specific checkpoint declaration and thresholds.
 * @param decodeStep Original synthetic Decode step; absent retains Prefill evidence.
 * @return Complete source-indexed layer records, never a copied model inventory.
 */
std::vector<LayerStats> layers(const ParityConfig &contract, std::optional<int> decodeStep=std::nullopt)
{
    std::vector<LayerStats> result;
    for (int index = 0; index < contract.early_layers_count; ++index)
    {
        LayerStats layer; layer.layer_idx = index;
        const auto prefix = "layer" + std::to_string(index) + "_";
        const auto phase=decodeStep?ParityForwardPhase::Decode:ParityForwardPhase::Prefill;
        for (const auto &key : contract.graph_snapshot_policy.requiredSnapshotKeys(phase))
            if (key.starts_with(prefix)) {
                const auto name=key.substr(prefix.size());
                layer.stage_results.push_back(decodeStep && isQwenSparseQueryCheckpoint(name)
                    ?decodeQueryObservation(name,*decodeStep):observation(name,true));
            }
        if(decodeStep)FinalLogitProbe::finalizeDecodeLayerNumerics(layer,contract,true);
        else FinalLogitProbe::finalizePrefillLayerNumerics(layer, contract);
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
    result.cosine_similarity = 1.f; result.layer_stats = layers(contract,index);
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

/** @test Valid final logits cannot waive raw/prepared Q provenance or authenticate a false empty guard. */
TEST(Qwen4ExpFinalLogits, DecodeQueryHistoryCannotBeWaivedByValidFinalRows)
{
    const auto cfg=config();auto good=decode(cfg);
    FinalLogitProbe::finalizeDecodeSummaryNumerics(good,cfg,true);
    ASSERT_TRUE(good.overall_passed);
    for(const auto *name:{"QSA_INDEXER_RAW_QUERY","QSA_INDEXER_QUERY"})for(int mutation=0;mutation<3;++mutation)
    {
        SCOPED_TRACE(::testing::Message()<<name<<" mutation="<<mutation);
        auto changed=good;
        auto &records=changed.step_stats.front().layer_stats.at(3).stage_results;
        const auto found=std::find_if(records.begin(),records.end(),[&](const auto &stage){return stage.stage_name==name;});
        ASSERT_NE(found,records.end());
        ASSERT_TRUE(validParityCheckpointEvidence(*found,true));
        EXPECT_FALSE(found->passed); // The original deliberately red comparison is still visible.
        if(mutation==0)found->query_work_evidence.reset();
        else if(mutation==1)++found->query_work_evidence->history.decodeRowsBefore;
        else {
            // Internally consistent empty metadata from a claimed shorter
            // prefix cannot override the independently supplied original2051.
            auto proof=*found->query_work_evidence;
            proof.observation.data=nullptr;proof.observation.size=proof.observation.rows=0;
            proof.observation.typed_values=std::span<const float>{};
            auto &record=proof.observation.authored_token_rows->requests.front();
            record.prefixRows=2050;record.firstRow=record.endRow=1;
            record.emptyReason=SnapshotEmptyReason::RankingNotRequired;
            ASSERT_TRUE(validSnapshotAuthoredTokenRows(proof.observation));
            proof.unauthored=true;proof.referenceFirstRow=1;
            *found=StageComparisonResult{};found->stage_name=name;
            found->checkpoint_evidence_kind=ParityCheckpointEvidenceKind::UnauthoredQuery;
            found->query_work_evidence=std::move(proof);
        }
        FinalLogitProbe::finalizeDecodeSummaryNumerics(changed,cfg,true);
        EXPECT_FALSE(changed.overall_passed);
    }
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

/** @test Both partial publication authorities retain low raw cosine but honor the declared decode KL. */
TEST(Qwen4ExpFinalLogits, PartialPrefixUsesDeclaredFinalLogitPolicyAndRetainsLegacyCosine)
{
    const std::array reference{1.f,2.f,3.f,4.f};
    const std::array shifted{-99.1f,-98.f,-97.f,-96.f}; // Small probability drift; raw cosine remains red.
    const auto cfg=config();
    const auto graph=comparePrefixPartialLogits(shifted,reference,4,cfg.final_logit_mean_kl,.99f);
    std::vector<float> terminal;terminal.insert(terminal.end(),shifted.begin(),shifted.end());
    terminal.insert(terminal.end(),shifted.begin(),shifted.end());
    const auto publication=comparePrefixPartialLogits(terminal,reference,4,cfg.final_logit_mean_kl,.99f);
    ASSERT_TRUE(graph.evidence_valid);EXPECT_TRUE(graph.passed);EXPECT_LT(graph.minimum_cosine,.99f);
    EXPECT_TRUE(publication.passed);EXPECT_EQ(publication.rows,2u);
    EXPECT_FLOAT_EQ(publication.maximum_kl,graph.maximum_kl);
    EXPECT_FALSE(comparePrefixPartialLogits(shifted,reference,4,std::nullopt,.99f).passed);
    EXPECT_TRUE(comparePrefixPartialLogits(reference,reference,4,std::nullopt,.99f).passed);
}

/** @test Every complete participant row must meet the strict decode budget, never a mean over copies. */
TEST(Qwen4ExpFinalLogits, PartialPrefixRejectsOneBadRowAndStrictKLBoundary)
{
    const std::array reference{1.f,2.f,3.f,4.f};
    const std::array bad{4.f,3.f,2.f,1.f};
    auto cfg=config();
    std::vector<float> rows(reference.begin(),reference.end());rows.insert(rows.end(),bad.begin(),bad.end());
    EXPECT_FALSE(comparePrefixPartialLogits(rows,reference,4,cfg.final_logit_mean_kl,.99f).passed);
    std::swap_ranges(rows.begin(),rows.begin()+4,rows.begin()+4);
    EXPECT_FALSE(comparePrefixPartialLogits(rows,reference,4,cfg.final_logit_mean_kl,.99f).passed);
    const float exact=computeKLDivergence(bad.data(),reference.data(),4,4);
    ASSERT_GT(exact,0.f);cfg.final_logit_mean_kl->decode_maximum=exact;
    EXPECT_FALSE(comparePrefixPartialLogits(bad,reference,4,cfg.final_logit_mean_kl,-1.f).passed);
    cfg.final_logit_mean_kl->decode_maximum=std::nextafter(exact,std::numeric_limits<float>::infinity());
    EXPECT_TRUE(comparePrefixPartialLogits(bad,reference,4,cfg.final_logit_mean_kl,-1.f).passed);
}

/** @test Invalid geometry and any nonfinite source row reject both actual publication variants. */
TEST(Qwen4ExpFinalLogits, PartialPrefixRejectsMissingPartialAndNonfiniteRows)
{
    const std::array reference{1.f,2.f,3.f,4.f};const auto cfg=config();
    EXPECT_FALSE(comparePrefixPartialLogits({},reference,4,cfg.final_logit_mean_kl,.99f).evidence_valid);
    EXPECT_FALSE(comparePrefixPartialLogits(reference,reference,0,cfg.final_logit_mean_kl,.99f).evidence_valid);
    EXPECT_FALSE(comparePrefixPartialLogits(std::span(reference).first(3),reference,4,cfg.final_logit_mean_kl,.99f).evidence_valid);
    EXPECT_FALSE(comparePrefixPartialLogits(reference,std::span(reference).first(3),4,cfg.final_logit_mean_kl,.99f).evidence_valid);
    for(float invalid:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()})
    {
        std::vector<float> rows(reference.begin(),reference.end());rows.insert(rows.end(),reference.begin(),reference.end());
        rows[0]=invalid;
        EXPECT_FALSE(comparePrefixPartialLogits(rows,reference,4,cfg.final_logit_mean_kl,.99f).evidence_valid);
        rows[0]=1.f;rows.back()=invalid;
        EXPECT_FALSE(comparePrefixPartialLogits(rows,reference,4,cfg.final_logit_mean_kl,.99f).passed);
    }
}

} // namespace llaminar2::test::parity

namespace llaminar2::test::parity
{
namespace
{
/** @brief Actual original comparator outputs projected into reversed diagnostic ownership. */
struct SplitPipelineFixture
{
    ParityConfig cfg=config();
    std::array<std::vector<LayerStats>,2> local;
    std::array<std::vector<PipelineParityStageOwnership>,2> scopes;
    std::vector<PipelineParityObservation> headers;
    std::vector<PipelineParityLayerObservation> cells;
    /**
     * @brief Preserve the canonical comparator/required-stage law for both phases.
     * @param phase Original phase. @param index Original Decode index or minus one for Prefill.
     * Rank0 deliberately owns the tail, so no rank-order heuristic can succeed.
     */
    SplitPipelineFixture(ParityForwardPhase phase=ParityForwardPhase::Prefill,int index=-1)
    {
        cfg.uses_cross_rank_pipeline=true;
        scopes[0]={{1,0,24,48,false,true}};
        scopes[1]={{0,1,0,24,true,false}};
        const auto original=layers(cfg,phase==ParityForwardPhase::Decode ? std::optional{index}:std::nullopt);
        const auto boundary=prefill(cfg);
        for(int rank=0;rank<2;++rank)
        {
            for(const auto& layer:original)
                if(layer.layer_idx>=scopes[rank][0].first_layer && layer.layer_idx<scopes[rank][0].last_layer)
                    local[rank].push_back(layer);
            PipelineParityObservation observed;
            observed.rank=rank;observed.world=2;observed.layers=48;observed.phase=static_cast<int>(phase);observed.step=index;
            observed.scope_valid=1;
            observed.local_raw_valid=bool(FinalLogitProbe::acceptsOwnedPipelineRawEvidence(local[rank],scopes[rank],cfg,phase,rank));
            observed.embedding_claims=rank==1;observed.logits_claims=rank==0;
            if(rank==1 && phase==ParityForwardPhase::Prefill)
            {
                observed.embedding_valid=boundary.embedding_evidence_valid;
                observed.embedding_cosine=boundary.embedding_cosine;
            }
            if(rank==0)
            {
                const auto actual=step(index<0?0:index,cfg);
                observed.has_logits=actual.has_logit_data;observed.logits_valid=actual.logit_evidence_valid;
                observed.kl=actual.kl_divergence;observed.cosine=actual.cosine_similarity;
                observed.logit_passed=cfg.final_logit_mean_kl->accepts(observed.kl,phase==ParityForwardPhase::Prefill);
            }
            headers.push_back(observed);
            rebuild(rank);
        }
    }
    /** @brief Reproject changed raw observations through the actual original framework helper.
     * @param rank Actual authored owner whose diagnostic records changed. */
    void rebuild(int rank)
    {
        if(cells.size()!=96)cells.resize(96);
        if(headers.size()>std::size_t(rank))
        {
            const auto phase=static_cast<ParityForwardPhase>(headers[rank].phase);
            headers[rank].local_raw_valid=bool(FinalLogitProbe::acceptsOwnedPipelineRawEvidence(local[rank],scopes[rank],cfg,phase,rank));
            const auto projected=FinalLogitProbe::pipelineParityLayerObservations(local[rank],scopes[rank],48,headers[rank].local_raw_valid,rank);
            std::copy(projected.begin(),projected.end(),cells.begin()+rank*48);
        }
    }
    /** @return Actual merger result; no test-owned acceptance implementation. */
    PipelineParityEvidence merge(ParityForwardPhase phase=ParityForwardPhase::Prefill,int index=-1) const
    { return mergePipelineParityEvidence(headers,cells,48,phase,index); }
};
}

/** @test Both original half-model comparators jointly satisfy the exact same full48 policy. */
TEST(Qwen4ExpFinalLogits, SplitPipelineCompletePartitionAndReversedBoundaryOwners)
{
    SplitPipelineFixture fixture;
    ASSERT_EQ(fixture.local[0].size(),24u);ASSERT_EQ(fixture.local[1].size(),24u);
    const auto evidence=fixture.merge();
    for(int replay=0;replay<3;++replay)
    {
        const auto repeated=fixture.merge();
        EXPECT_EQ(repeated.headRank(),evidence.headRank());
        EXPECT_EQ(repeated.tail().kl,evidence.tail().kl);
    }
    EXPECT_EQ(evidence.headRank(),1);EXPECT_EQ(evidence.tailRank(),0);
    EXPECT_TRUE(evidence.matches(ParityForwardPhase::Prefill,-1,48));
    auto summary=prefill(fixture.cfg);
    summary.layer_stats=fixture.local[0];summary.pipeline_evidence=evidence;
    FinalLogitProbe::finalizePrefillSummaryNumerics(summary,fixture.cfg,false,true);
    EXPECT_TRUE(summary.overall_passed);
    EXPECT_EQ(summary.layer_stats.size(),24u); // Original local CSV rows remain local.
    summary.lm_head_kl=fixture.cfg.final_logit_mean_kl->prefill_maximum;
    FinalLogitProbe::finalizePrefillSummaryNumerics(summary,fixture.cfg,false,true);
    EXPECT_FALSE(summary.overall_passed); // Exact unchanged strict numerical boundary.
    auto unchanged=config();auto local=prefill(unchanged);
    FinalLogitProbe::finalizePrefillSummaryNumerics(local,unchanged,false,true);
    EXPECT_TRUE(local.overall_passed);EXPECT_FALSE(local.pipeline_evidence);
}

/** @test Missing, duplicate, foreign and malformed original raw observations cannot certify a union. */
TEST(Qwen4ExpFinalLogits, SplitPipelineRejectsMissingDuplicateAndForeignRawEvidence)
{
    {
        SplitPipelineFixture fixture;
        const auto retained=fixture.local[0].back();
        fixture.local[0].pop_back();fixture.rebuild(0);
        EXPECT_THROW(fixture.merge(),std::invalid_argument);
        fixture.local[0].push_back(retained);fixture.rebuild(0);
        EXPECT_NO_THROW(fixture.merge()); // Reset restores data, with identical original scope.
    }
    {
        SplitPipelineFixture fixture;fixture.local[0].push_back(fixture.local[0].front());fixture.rebuild(0);
        EXPECT_THROW(fixture.merge(),std::invalid_argument);
    }
    {
        SplitPipelineFixture fixture;fixture.local[0][0]=fixture.local[1][0];fixture.rebuild(0);
        EXPECT_THROW(fixture.merge(),std::invalid_argument);
    }
    {
        SplitPipelineFixture fixture;fixture.local[0][0].stage_results.clear();fixture.rebuild(0);
        EXPECT_THROW(fixture.merge(),std::invalid_argument);
    }
    {
        SplitPipelineFixture fixture;fixture.scopes[0][0].rank=1;
        EXPECT_FALSE(FinalLogitProbe::acceptsOwnedPipelineRawEvidence(fixture.local[0],fixture.scopes[0],fixture.cfg,ParityForwardPhase::Prefill,0));
        EXPECT_THROW(FinalLogitProbe::pipelineParityLayerObservations(fixture.local[0],fixture.scopes[0],48,true,0),std::invalid_argument);
    }
}

/** @test Phase/step/world/boundary claims and literal original source identity stay exact. */
TEST(Qwen4ExpFinalLogits, SplitPipelineRejectsConflictingOwnershipAndStaleIdentity)
{
    for(int fault=0;fault<7;++fault)
    {
        SplitPipelineFixture fixture;
        switch(fault)
        {
        case 0:fixture.headers[0].embedding_claims=1;break;
        case 1:fixture.headers[0].step=0;break;
        case 2:fixture.headers[0].phase=static_cast<int>(ParityForwardPhase::Decode);break;
        case 3:fixture.headers[1].world=3;break;
        case 4:fixture.headers[0].rank=1;break;
        case 5:fixture.cells[0].claims=1;fixture.cells[0].records=1;fixture.cells[0].valid_records=1;break;
        case 6:fixture.headers[0].kl=std::numeric_limits<float>::quiet_NaN();break;
        }
        EXPECT_THROW(fixture.merge(),std::invalid_argument);
    }
    EXPECT_NO_THROW(requirePipelineParityIdentities(std::array<std::string,2>{"original source/reference/request","original source/reference/request"}));
    EXPECT_THROW(requirePipelineParityIdentities(std::array<std::string,2>{"original source/reference/request","other source/reference/request"}),std::invalid_argument);
    EXPECT_THROW(requirePipelineParityIdentities(std::array<std::string,2>{"",""}),std::invalid_argument);
}

/** @test Each original Decode step retains query provenance and the strict per-step logit budget. */
TEST(Qwen4ExpFinalLogits, SplitPipelineDecodeStepIsCompleteAndNeverAveraged)
{
    auto cfg=config();cfg.uses_cross_rank_pipeline=true;
    DecodeParitySummary summary;summary.steps_total=2;
    for(int index=0;index<2;++index)
    {
        SplitPipelineFixture fixture(ParityForwardPhase::Decode,index);
        auto actual=step(index,cfg);actual.layer_stats=fixture.local[0];
        actual.pipeline_evidence=fixture.merge(ParityForwardPhase::Decode,index);
        EXPECT_FALSE(actual.pipeline_evidence->matches(ParityForwardPhase::Decode,index+1,48));
        summary.step_stats.push_back(std::move(actual));
        EXPECT_THROW(fixture.merge(ParityForwardPhase::Decode,index+1),std::invalid_argument);
    }
    FinalLogitProbe::finalizeDecodeSummaryNumerics(summary,cfg,true);
    EXPECT_TRUE(summary.overall_passed);
    summary.step_stats[1].kl_divergence=cfg.final_logit_mean_kl->decode_maximum;
    FinalLogitProbe::finalizeDecodeSummaryNumerics(summary,cfg,true);
    EXPECT_FALSE(summary.overall_passed);
    summary.step_stats[1].pipeline_evidence.reset();
    EXPECT_FALSE(FinalLogitProbe::acceptsFinalLogitSummaryRawEvidence(summary.step_stats[1],cfg,ParityForwardPhase::Decode,1));
}
}

namespace llaminar2::test::parity
{
/** @test Communicated decoder records contain only actual claims/observations, including partial and idle owners. */
TEST(Qwen4ExpFinalLogits, SplitPipelineLiveMetadataRetainsOnlyActualRecordsAcrossResetReplay)
{
    SplitPipelineFixture fixture;
    std::vector<PipelineParityLayerRecord> joined;
    std::array<std::size_t,3> counts{};
    for(std::size_t rank=0;rank<2;++rank)
    {
        const auto compact=compactPipelineParityLayers(std::span(fixture.cells).subspan(rank*48,48));
        EXPECT_EQ(compact.size(),24u);
        counts[rank]=compact.size();joined.insert(joined.end(),compact.begin(),compact.end());
    }
    // A completely idle original rank has no layer payload; only its bounded count/header remains.
    const std::vector<PipelineParityLayerObservation> idle(48);
    EXPECT_TRUE(compactPipelineParityLayers(idle).empty());
    EXPECT_EQ(joined.size()*sizeof(PipelineParityLayerRecord),48u*16u);
    auto expanded=expandPipelineParityLayers(joined,counts,48);
    ASSERT_EQ(expanded.size(),144u);
    for(std::size_t i=0;i<fixture.cells.size();++i)
    {
        EXPECT_EQ(expanded[i].claims,fixture.cells[i].claims);
        EXPECT_EQ(expanded[i].records,fixture.cells[i].records);
        EXPECT_EQ(expanded[i].valid_records,fixture.cells[i].valid_records);
    }
    for(auto& header:fixture.headers)header.world=3;
    PipelineParityObservation empty;empty.rank=2;empty.world=3;empty.layers=48;
    empty.phase=static_cast<int>(ParityForwardPhase::Prefill);empty.step=-1;
    empty.scope_valid=1;empty.local_raw_valid=1;fixture.headers.push_back(empty);
    EXPECT_NO_THROW(mergePipelineParityEvidence(fixture.headers,expanded,48,ParityForwardPhase::Prefill,-1));
    // A large-to-small replay recomputes useful metadata and cannot send old retained capacity.
    std::vector<PipelineParityLayerObservation> partial(48);partial[7]={1,1,1};
    for(int replay=0;replay<3;++replay)
    {
        const auto compact=compactPipelineParityLayers(partial);
        ASSERT_EQ(compact.size(),1u);EXPECT_EQ(compact.front().layer,7u);
        const std::array<std::size_t,3> partial_counts{1,0,0};
        const auto reset=expandPipelineParityLayers(compact,partial_counts,48);
        EXPECT_EQ(reset[7].claims,1u);EXPECT_EQ(reset[8].claims,0u);
        EXPECT_THROW(mergePipelineParityEvidence(fixture.headers,reset,48,ParityForwardPhase::Prefill,-1),std::invalid_argument);
    }
}
/** @test Malformed compact wire extents reject; nonzero invalid observations remain visible to the exact merger. */
TEST(Qwen4ExpFinalLogits, SplitPipelineLiveMetadataRejectsMalformedRecordsWithoutHidingForeignEvidence)
{
    const std::array<std::size_t,2> one{1,0};
    const PipelineParityLayerRecord valid{0,{1,1,1}};
    const std::array<std::size_t,2> two{2,0};
    EXPECT_THROW(expandPipelineParityLayers(std::array{valid,valid},two,48),std::invalid_argument);
    EXPECT_THROW(expandPipelineParityLayers(std::array{PipelineParityLayerRecord{48,{1,1,1}}},one,48),std::invalid_argument);
    EXPECT_THROW(expandPipelineParityLayers(std::array{PipelineParityLayerRecord{0,{}}},one,48),std::invalid_argument);
    EXPECT_THROW(expandPipelineParityLayers(std::array{valid},two,48),std::invalid_argument);
    EXPECT_THROW(expandPipelineParityLayers(std::array{valid},std::array<std::size_t,2>{0,0},48),std::invalid_argument);
    EXPECT_THROW(expandPipelineParityLayers(std::array{valid},one,0),std::invalid_argument);
    SplitPipelineFixture fixture;
    // An unowned actual record must cross the wire and fail the original exact checker.
    fixture.cells[0]={0,1,1};
    const auto first=compactPipelineParityLayers(std::span(fixture.cells).first(48));
    const auto second=compactPipelineParityLayers(std::span(fixture.cells).subspan(48));
    std::vector<PipelineParityLayerRecord> joined(first);joined.insert(joined.end(),second.begin(),second.end());
    const auto expanded=expandPipelineParityLayers(joined,std::array<std::size_t,2>{first.size(),second.size()},48);
    EXPECT_EQ(expanded[0].records,1u);EXPECT_EQ(expanded[0].claims,0u);
    EXPECT_THROW(mergePipelineParityEvidence(fixture.headers,expanded,48,ParityForwardPhase::Prefill,-1),std::invalid_argument);
}
}
