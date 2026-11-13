/**
 * @file Test__PlanningAdmissionPreparation.cpp
 * @brief Device-free exact source declaration replay and bounded expensive compiler work.
 */
#include "planning/PlanningAdmissionPreparation.h"
#include "loaders/ModelLoader.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
using namespace llaminar2;
/** @test Repeated physical searches reuse source scans while every original interval equals the canonical builder. */
TEST(PlanningAdmissionPreparation, IntervalReplayMatchesCanonicalSourceWithBoundedScans)
{
    test::PlanningGGUFFixture file(true,false);
    PlanningModelSource source(file.path());
    PlanningAdmissionPreparation prepared(source.loader(),source.metadata(),{});
    const auto &family=prepared.graphFamily(MoEOverlayMTPGraphFamilyPolicy::MainOnly,1,64,1,1,0);
    const int layers=family.routedLayerCapacity();
    for(int replay=0;replay<20;++replay)
        for(int first=0;first<layers;++first)
            for(int count=1;count<=layers-first;++count)
            {
                const auto &actual=prepared.manifest(first,count,layers);
                const auto expected=buildMoEOverlayLayerWeightManifestFromLoader(source.loader(),count,
                    source.metadata().memoryProfile().expert_count,first);
                ASSERT_EQ(actual.size(),expected.size());
                for(size_t index=0;index<actual.size();++index)
                { EXPECT_EQ(actual[index].layer_idx,expected[index].layer_idx); EXPECT_EQ(actual[index].projections,expected[index].projections); }
                EXPECT_EQ(&prepared.graphFamily(MoEOverlayMTPGraphFamilyPolicy::MainOnly,1,64,1,1,0),&family);
            }
    const auto counts=prepared.counts();
    EXPECT_EQ(counts.upload_geometry_builds,1u);
    EXPECT_EQ(counts.graph_family_builds,1u);
    EXPECT_EQ(counts.full_manifest_builds,1u);
    EXPECT_EQ(counts.interval_manifest_builds,static_cast<size_t>(layers*(layers+1)/2));
    EXPECT_THROW(prepared.manifest(-1,1,layers),std::invalid_argument);
    EXPECT_THROW(prepared.manifest(0,layers+1,layers),std::invalid_argument);
    test::PlanningGGUFFixture other(true,false);PlanningModelSource foreign(other.path());
    EXPECT_THROW(prepared.requireOriginal(foreign.loader(),foreign.metadata(),{}),std::invalid_argument);
    EXPECT_THROW(prepared.requireOriginal(source.loader(),source.metadata(),{.staging_stream_count=2}),std::invalid_argument);
    EXPECT_THROW(prepared.requireOriginal(source.loader(),source.metadata(),{.staging_budget_bytes=4096}),std::invalid_argument);
    // Shape/family changes are new typed declarations, not a reuse of the original graph.
    (void)prepared.graphFamily(MoEOverlayMTPGraphFamilyPolicy::MainOnly,1,128,1,1,0);
    EXPECT_EQ(prepared.counts().graph_family_builds,2u);
    EXPECT_THROW(prepared.requireMTP(),std::invalid_argument);
    EXPECT_EQ(prepared.counts().mtp_validations,0u);
}
