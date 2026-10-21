/**
 * @file Test__PredictorFCGeometry.cpp
 * @brief Device-free whole-storage branch-FC reshape admission and adversarial arithmetic.
 *
 * Current live rows never resize the roots. A counted empty/small replay uses
 * the same admitted physical metadata, while the existing descriptor gates
 * execution against the original authoritative scalar.
 */
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPredictorBuilder.h"
#include <gtest/gtest.h>
#include <array>
using namespace llaminar2;
TEST(PredictorFCGeometry, WholeStorageBranchReshapePreservesEveryNativeByte)
{
    const std::array<size_t,2> concat{64,20480},projected{64,10240};
    const auto geometry=Qwen4ExpPredictorFCGeometry::admit(concat,projected,4,2560);
    EXPECT_EQ(geometry.input,(std::array<size_t,2>{256,5120}));
    EXPECT_EQ(geometry.output,(std::array<size_t,2>{256,2560}));
    EXPECT_EQ(geometry.input[0]*geometry.input[1],concat[0]*concat[1]);
    EXPECT_EQ(geometry.output[0]*geometry.output[1],projected[0]*projected[1]);
}
TEST(PredictorFCGeometry, OriginalCountChangesNeitherRootShapeNorExpandedCountOwner)
{
    int32_t live=4;
    const std::vector<int> host{4};
    const auto rows=Qwen4ExpPredictorRows::admit(1,4,DeviceId::cuda(0),nullptr,&live);
    const std::array<size_t,2> concat{64,20480},projected{64,10240};
    const auto geometry=Qwen4ExpPredictorFCGeometry::admit(concat,projected,rows.tokens().physicalRows(),2560);
    for(const int current : {4,1,0,4}) {
        live=current;
        EXPECT_EQ(rows.branches().countOwner(),rows.tokens().countOwner());
        EXPECT_EQ(rows.branches().activeRowsFor(current),current*4);
        const auto repeat=Qwen4ExpPredictorFCGeometry::admit(concat,projected,rows.tokens().physicalRows(),2560);
        EXPECT_EQ(repeat.input,geometry.input);EXPECT_EQ(repeat.output,geometry.output);
    }
}
TEST(PredictorFCGeometry, RejectsOldHWidthWrongBranchOrderAndMismatchedCapacity)
{
    const std::array<size_t,2> concat{8,20480},projected{8,10240};
    const std::array<size_t,2> oldConcat{8,5120},oldProjected{8,2560},differentRows{9,10240};
    EXPECT_THROW((void)Qwen4ExpPredictorFCGeometry::admit(oldConcat,projected,8,2560),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorFCGeometry::admit(concat,oldProjected,8,2560),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorFCGeometry::admit(concat,differentRows,8,2560),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorFCGeometry::admit(concat,projected,9,2560),std::invalid_argument);
}
TEST(PredictorFCGeometry, RejectsRankZeroAndNativeDimensionOverflowBeforeAnyPreparation)
{
    const std::array<size_t,2> concat{8,20480},projected{8,10240};
    const std::array<size_t,3> rank3{8,4,5120};
    const std::array<size_t,2> huge{static_cast<size_t>(std::numeric_limits<int>::max())/4+1,20480};
    const std::array<size_t,2> hugeOutput{huge[0],10240};
    EXPECT_THROW((void)Qwen4ExpPredictorFCGeometry::admit(rank3,projected,8,2560),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorFCGeometry::admit(concat,projected,0,2560),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorFCGeometry::admit(concat,projected,8,0),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorFCGeometry::admit(concat,projected,8,std::numeric_limits<int>::max()),std::overflow_error);
    EXPECT_THROW((void)Qwen4ExpPredictorFCGeometry::admit(huge,hugeOutput,8,2560),std::overflow_error);
}
