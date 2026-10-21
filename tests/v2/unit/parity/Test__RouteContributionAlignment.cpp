/**
 * @file Test__RouteContributionAlignment.cpp
 * @brief Actual same-row expert diagnostics preserve ordinary numerical acceptance.
 *
 * Static probes construct no fixture/backend/model. The production matcher,
 * comparator and both finalizers consume independent tiny route banks.
 */
#include "integration/parity/ParityTestBase.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <fstream>
#include <cstdlib>
namespace llaminar2::test::parity {
namespace {
/** @brief Expose actual static operations without constructing a backend fixture. */
class AlignedContributionProbe : public ParityTestBase {
public:
    using ParityTestBase::compareExpertAlignedRouteContributions;
    using ParityTestBase::finalizePrefillLayerNumerics;
    using ParityTestBase::finalizeDecodeLayerNumerics;
};
/** @brief Retire only this fixture's actual writer output, with no model/backend setup. */
struct TemporaryCSVFiles {
    std::filesystem::path root;
    /** @brief Create one unique ordinary host directory for actual CSV writer checks. */
    TemporaryCSVFiles() {
        auto name=(std::filesystem::temp_directory_path()/"route-expert-align-XXXXXX").string();
        std::vector<char> storage(name.begin(),name.end());storage.push_back(0);
        const auto *made=::mkdtemp(storage.data());
        if(!made)throw std::runtime_error("Cannot create isolated diagnostic CSV directory");
        root=made;
    }
    /** @brief Delete only this instance's temporary files even after assertion unwinding. */
    ~TemporaryCSVFiles() {std::error_code error;std::filesystem::remove_all(root,error);}
};
/** @brief Retrieve the original raw route line, excluding the separately named derived diagnostic. */
std::string rawRouteCSVLine(const std::filesystem::path &path) {
    std::ifstream stream(path);std::string line;
    while(std::getline(stream,line)) {
        if(line.find(",MOE_ROUTE_CONTRIBUTIONS,")!=std::string::npos)return line;
    }
    throw std::runtime_error("Actual CSV writer omitted the original positional route row");
}
/** @brief Compare one unchanged semantic expert sum through the real authority. */
StageComparisonResult semantic() {
    const std::array values{1.f,2.f};
    return compareParityTensorData(values.data(),values.data(),2,"MOE_EXPERT_OUTPUT",0.99f);
}
}
/** @test Original rows/experts survive a within-set ranking permutation. */
TEST(RouteContributionAlignment,SameRowsAndExpertsAlignWithoutChangingOriginalSlots) {
    const std::array gold_ids{143.f,465.f,499.f,11.f,2.f,7.f};
    const std::array actual_ids{143.f,499.f,465.f,11.f,2.f,7.f};
    const std::array gold{1.f,0.f,2.f,0.f,0.f,3.f,4.f,0.f,0.f,5.f,6.f,0.f};
    const std::array actual{1.f,0.f,0.f,3.f,2.f,0.f,4.f,0.f,0.f,5.f,6.f,0.f};
    const auto raw=compareParityTensorData(actual.data(),gold.data(),gold.size(),"MOE_ROUTE_CONTRIBUTIONS",0.99f);
    ASSERT_FALSE(raw.passed);
    const auto aligned=AlignedContributionProbe::compareExpertAlignedRouteContributions(actual,gold,actual_ids,gold_ids,3,512,0.99f);
    ASSERT_TRUE(aligned);
    EXPECT_TRUE(aligned->passed);
    EXPECT_FLOAT_EQ(aligned->cosine_similarity,1.f);
    EXPECT_FLOAT_EQ(aligned->max_abs_diff,0.f);
    EXPECT_EQ(aligned->total_elements,gold.size());
    EXPECT_EQ(aligned->stage_name,"MOE_ROUTE_CONTRIBUTIONS_EXPERT_ALIGNED");
    EXPECT_EQ(actual_ids[1],499.f);
    EXPECT_EQ(gold_ids[1],465.f);
    EXPECT_EQ(actual[2],0.f);
    EXPECT_EQ(gold[2],2.f);
}
/** @test Per-expert errors remain visible even when the semantic sum is unchanged. */
TEST(RouteContributionAlignment,OpposingExpertErrorsCannotHideBehindAnExactSum) {
    const std::array ids{3.f,8.f};
    const std::array gold{1.f,0.f,0.f,2.f};
    const std::array actual{2.f,0.f,-1.f,2.f};
    const std::array sum_actual{actual[0]+actual[2],actual[1]+actual[3]},sum_gold{gold[0]+gold[2],gold[1]+gold[3]};
    EXPECT_TRUE(compareParityTensorData(sum_actual.data(),sum_gold.data(),2,"MOE_EXPERT_OUTPUT",0.99f).passed);
    const auto aligned=AlignedContributionProbe::compareExpertAlignedRouteContributions(actual,gold,ids,ids,2,512,0.99f);
    ASSERT_TRUE(aligned);
    EXPECT_FALSE(aligned->passed);
    EXPECT_GT(aligned->rel_l2_norm,0.f);
}
/** @test Different valid sets are explicitly unavailable, not intersection-only passes. */
TEST(RouteContributionAlignment,DifferentSetsNeverManufactureAnAlignedComparison) {
    const std::array ids{3.f,8.f};
    const std::array other{3.f,9.f};
    const std::array values{1.f,0.f,0.f,2.f};
    const auto mapping=alignRouteContributionReference(ids,other,values,4,2,512);
    EXPECT_EQ(mapping.status,RouteContributionAlignmentStatus::DifferentExpertSets);
    EXPECT_TRUE(mapping.reference_in_production_order.empty());
    EXPECT_FALSE(AlignedContributionProbe::compareExpertAlignedRouteContributions(values,values,ids,other,2,512,0.99f));
}
/** @test Duplicate/foreign/nonintegral IDs and incomplete geometries reject before comparison. */
TEST(RouteContributionAlignment,MalformedIdsAndMissingRowGeometryReject) {
    const std::array ids{3.f,8.f};
    const std::array values{1.f,0.f,0.f,2.f};
    for(const auto &bad:std::array{std::array{3.f,3.f},std::array{3.f,-1.f},std::array{3.f,512.f},std::array{3.f,8.5f},
            std::array{3.f,std::numeric_limits<float>::infinity()},std::array{3.f,std::numeric_limits<float>::quiet_NaN()}}) {
        EXPECT_THROW(AlignedContributionProbe::compareExpertAlignedRouteContributions(values,values,bad,ids,2,512,0.99f),std::invalid_argument);
        EXPECT_THROW(AlignedContributionProbe::compareExpertAlignedRouteContributions(values,values,ids,bad,2,512,0.99f),std::invalid_argument);
    }
    EXPECT_THROW(alignRouteContributionReference({},ids,values,4,2,512),std::invalid_argument);
    EXPECT_THROW(alignRouteContributionReference(ids,ids,values,3,2,512),std::invalid_argument);
    EXPECT_THROW(alignRouteContributionReference(ids,ids,values,4,0,512),std::invalid_argument);
    EXPECT_THROW(alignRouteContributionReference(ids,ids,values,4,3,512),std::invalid_argument);
    EXPECT_THROW(alignRouteContributionReference(ids,ids,values,4,2,1),std::invalid_argument);
    const std::array a{3.f,8.f,4.f,7.f},later_bad{3.f,9.f,4.f,4.f};
    EXPECT_THROW(alignRouteContributionReference(a,later_bad,values,4,2,512),std::invalid_argument);
}
/** @test A repeated expert in another row cannot repair the current row's missing counterpart. */
TEST(RouteContributionAlignment,RowIdentityCannotBeReplacedByGlobalExpertMembership) {
    const std::array ids{1.f,2.f,3.f,4.f},other{3.f,4.f,1.f,2.f},values{1.f,2.f,3.f,4.f};
    EXPECT_FALSE(AlignedContributionProbe::compareExpertAlignedRouteContributions(values,values,ids,other,2,512,0.99f));
}
/** @test Derived diagnostic rows preserve both phase aggregates and every original raw field. */
TEST(RouteContributionAlignment,DiagnosticsPreserveBothPhaseAggregatesAndRawMetrics) {
    const std::array ids{3.f,8.f};
    const std::array gold{1.f,0.f,0.f,2.f};
    const std::array actual{2.f,0.f,-1.f,2.f};
    auto raw=compareParityTensorData(actual.data(),gold.data(),4,"MOE_ROUTE_CONTRIBUTIONS",0.99f);raw.cosine_drop=0.37f;
    const auto aligned=AlignedContributionProbe::compareExpertAlignedRouteContributions(actual,gold,ids,ids,2,512,0.99f);
    ASSERT_TRUE(aligned);
    for(const bool decode:{false,true}) {
        auto before=LayerStats{.layer_idx=1};before.stage_results={raw,semantic()};auto after=before;
        after.stage_results.push_back(*aligned);ParityConfig config;
        if(decode) {
            AlignedContributionProbe::finalizeDecodeLayerNumerics(before,config,true);
            AlignedContributionProbe::finalizeDecodeLayerNumerics(after,config,true);
        } else {
            AlignedContributionProbe::finalizePrefillLayerNumerics(before,config);
            AlignedContributionProbe::finalizePrefillLayerNumerics(after,config);
        }
        EXPECT_EQ(before.passed,after.passed);
        EXPECT_EQ(before.stages_compared,after.stages_compared);
        EXPECT_FLOAT_EQ(before.avg_cosine_sim,after.avg_cosine_sim);
        EXPECT_FLOAT_EQ(before.min_cosine_sim,after.min_cosine_sim);
        EXPECT_EQ(before.worst_stage,after.worst_stage);
        EXPECT_EQ(before.max_drop_stage,after.max_drop_stage);
        EXPECT_FLOAT_EQ(before.max_cosine_drop,after.max_cosine_drop);
        EXPECT_FALSE(after.stage_results.front().passed);
        EXPECT_FLOAT_EQ(after.stage_results.front().cosine_drop,0.37f);
        EXPECT_FLOAT_EQ(after.stage_results.front().rel_l2_norm,raw.rel_l2_norm);
        EXPECT_FALSE(after.stage_results.back().passed);
    }
    auto nonfinite=actual;nonfinite[0]=std::numeric_limits<float>::quiet_NaN();
    const auto invalid=AlignedContributionProbe::compareExpertAlignedRouteContributions(nonfinite,gold,ids,ids,2,512,0.99f);
    ASSERT_TRUE(invalid);
    EXPECT_FALSE(invalid->passed);
    auto layer=LayerStats{.layer_idx=1};layer.stage_results={raw,semantic(),*invalid};
    AlignedContributionProbe::finalizePrefillLayerNumerics(layer,ParityConfig{});
    EXPECT_FALSE(layer.passed);
}
/** @test Existing CSV schema and original positional bytes survive the additional derived row. */
TEST(RouteContributionAlignment,ActualCSVWriterRetainsOriginalPositionalRowByteForByte) {
    TemporaryCSVFiles files;
    const std::array ids{3.f,8.f};
    const std::array gold{1.f,0.f,0.f,2.f};
    const std::array actual{2.f,0.f,-1.f,2.f};
    auto raw=compareParityTensorData(actual.data(),gold.data(),4,"MOE_ROUTE_CONTRIBUTIONS",0.99f);
    raw.cosine_drop=0.37f;
    auto summary=ParityTestSummary{};auto layer=LayerStats{.layer_idx=1};
    layer.stage_results={raw,semantic()};summary.layer_stats.push_back(layer);
    ASSERT_TRUE(ParityCSVArtifactWriter::writePrefill(files.root/"before","CPU",summary));
    const auto aligned=AlignedContributionProbe::compareExpertAlignedRouteContributions(actual,gold,ids,ids,2,512,0.99f);
    ASSERT_TRUE(aligned);
    summary.layer_stats.front().stage_results.push_back(*aligned);
    AlignedContributionProbe::finalizePrefillLayerNumerics(summary.layer_stats.front(),ParityConfig{});
    ASSERT_TRUE(ParityCSVArtifactWriter::writePrefill(files.root/"after","CPU",summary));
    EXPECT_EQ(rawRouteCSVLine(files.root/"before/prefill_stages.csv"),rawRouteCSVLine(files.root/"after/prefill_stages.csv"));
    std::ifstream before(files.root/"before/prefill_stages.csv"),after(files.root/"after/prefill_stages.csv");
    std::string header_before,header_after;std::getline(before,header_before);std::getline(after,header_after);
    EXPECT_EQ(header_before,header_after);
}
/** @test The original decode crop's element window owns its associated reference IDs too. */
TEST(RouteContributionAlignment,ExistingDecodeCropCarriesTheSameCheckedWindowToReferenceIds) {
    const std::array actual_ids{3.f,8.f};
    const std::array full_reference_ids{1.f,2.f,8.f,3.f};
    const std::array original_reference{10.f,0.f,0.f,20.f,0.f,2.f,1.f,0.f};
    const std::array actual{1.f,0.f,0.f,2.f};
    const auto first=original_reference.size()-actual.size();
    // This is the same checked interval used by the ordinary decode value copy.
    const auto window=RouteContributionReferenceWindow::checked(original_reference.size(),first,actual.size());
    const std::span selected(original_reference.data()+first,actual.size());
    const auto compared=AlignedContributionProbe::compareExpertAlignedRouteContributions(
        actual,selected,actual_ids,full_reference_ids,2,512,0.99f,window);
    ASSERT_TRUE(compared);
    EXPECT_TRUE(compared->passed);
    EXPECT_FLOAT_EQ(compared->max_abs_diff,0.f);
    EXPECT_EQ(full_reference_ids[0],1.f);
    // Missing/incorrect full-frame ownership cannot be repaired by another ID trim.
    EXPECT_THROW(AlignedContributionProbe::compareExpertAlignedRouteContributions(
        actual,selected,actual_ids,actual_ids,2,512,0.99f,window),std::invalid_argument);
    EXPECT_THROW(AlignedContributionProbe::compareExpertAlignedRouteContributions(
        actual,original_reference,actual_ids,full_reference_ids,2,512,0.99f,window),std::invalid_argument);
    EXPECT_THROW(RouteContributionReferenceWindow::checked(8,3,4),std::invalid_argument);
    const auto split_row=RouteContributionReferenceWindow::checked(8,2,6);
    EXPECT_THROW(split_row.projectIds(full_reference_ids,actual_ids.size(),6,2),std::invalid_argument);
    const auto wrong_width=RouteContributionReferenceWindow::checked(12,8,4);
    EXPECT_THROW(wrong_width.projectIds(full_reference_ids,actual_ids.size(),4,2),std::invalid_argument);
}
} // namespace llaminar2::test::parity
