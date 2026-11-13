/**
 * @file CPUJointMTPSourceContract.inc
 * @brief Actual loader-authenticated predictor source coverage before preparation.
 *
 * The original tiny GGUF layer stands for the graph family's selected source
 * coordinate. No engines, sockets or native readiness are fabricated here.
 */
#include "execution/moe/MoEOverlayMTPSourceContract.h"

/** @test Whole experts and every original joint fragment retain complete local source roles. */
TEST(CPUJointMTPSourceContract, OriginalPredictorSourceRequiresCompleteOriginalCohort)
{
    using namespace llaminar2;
    const std::array<int,1> layers{7},foreign_layer{8};
    for(int rank=0;rank<2;++rank) {
        SCOPED_TRACE(rank);
        const auto original=staticJointOriginal(rank);
        const auto &map=*original.preparation->cpuJointOwnerMap();
        EXPECT_NO_THROW(requireMoEOverlayMTPSourceCoverage(map,original.preparation.get(),layers,512,rank));
        // Original local CPU requests are absent, despite the same retained map.
        const auto partial=original.preparation->filteredForDevice(DeviceId::rocm(0));
        EXPECT_EQ(partial.cpuJointOwnerMap(),original.preparation->cpuJointOwnerMap());
        EXPECT_THROW(requireMoEOverlayMTPSourceCoverage(map,&partial,layers,512,rank),std::invalid_argument);
        EXPECT_THROW(requireMoEOverlayMTPSourceCoverage(map,nullptr,layers,512,rank),std::invalid_argument);
        const auto foreign=staticJointOriginal(rank);
        EXPECT_THROW(requireMoEOverlayMTPSourceCoverage(map,foreign.preparation.get(),layers,512,rank),std::invalid_argument);
        EXPECT_THROW(requireMoEOverlayMTPSourceCoverage(map,original.preparation.get(),foreign_layer,512,rank),std::invalid_argument);
        EXPECT_THROW(requireMoEOverlayMTPSourceCoverage(map,original.preparation.get(),layers,511,rank),std::invalid_argument);
        const std::array<int,2> repeated{7,7};
        EXPECT_THROW(requireMoEOverlayMTPSourceCoverage(map,original.preparation.get(),repeated,512,rank),std::invalid_argument);
    }
}
