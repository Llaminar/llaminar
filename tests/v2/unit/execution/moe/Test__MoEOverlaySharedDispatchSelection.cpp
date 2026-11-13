/**
 * @file Test__MoEOverlaySharedDispatchSelection.cpp
 * @brief Replay production row selection/copy without GPU or CPU-serving readiness.
 *
 * Fixed original target geometry and actual live bank/router prefixes drive the
 * same stateless production helpers used by the captured mapped-row writer.
 * Native metadata validation/event/capture tests remain a separate obligation.
 */
#include "execution/moe/MoEOverlaySharedDispatchSelection.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <limits>
#include <algorithm>
using namespace llaminar2;
namespace {
/** @brief Original cold operands, with no allocation/native-ready authority. */
struct SharedRows {
 static constexpr int capacity=65,width=7;
 std::array<float,capacity*width> input{},output{};
 std::array<float,capacity*2> ids{},weights{};
 std::array<std::int32_t, 4> routes{99, MoEOverlayRouteTarget::cpuCohort(4).word(), 31,
                                    MoEOverlayRouteTarget::cpuCohort(4).word()};
 std::vector<MoEOverlaySharedDispatchTarget> targets;
 std::vector<MoEOverlayActivationEpochControl> controls;
 int active=capacity;
 MoEOverlayActivationDispatchPackLaunch launch{};
 /** @brief Build source-cohort2..8 plus ordinary consumer31, independent of dense IDs. */
 explicit SharedRows(int degree) {
  const std::array<std::int32_t,2> experts{1,3};
  for(int i=0;i<degree;++i){const int id=7+13*i;
      targets.push_back({id, MoEOverlayJointRouteTarget::fromOriginalCohort(4, id, 9, 4)});
  }
  targets.push_back({31,{}});
  for(int row=0;row<capacity;++row){ids[2*row]=row%4;ids[2*row+1]=row%4;weights[2*row]=0.75f;weights[2*row+1]=0.25f;}
  for(int i=0;i<capacity*width;++i)input[i]=static_cast<float>(i+1);
  launch.hidden_rows_fp32=input.data();launch.route_expert_ids_fp32=ids.data();launch.route_weights=weights.data();
  launch.placement.expert_count=4;launch.packet.hidden_rows_fp32=output.data();launch.packet.d_model=width;
  launch.packet.top_k=2;launch.active_row_count_device=&active;launch.physical_rows=capacity;launch.model_layer_index=9;
  bind();
 }
 /** @brief Bind immutable recorded original target control rows, without readiness. */
 void bind()
 {
     controls.resize(targets.size());
     for (std::size_t i = 0; i < targets.size(); ++i)
     {
         auto &channel = controls[i].channel;
         channel.target_participant_id = targets[i].participant;
         channel.cohort_target = targets[i].membership.cohortTarget();
         channel.return_role = static_cast<std::uint32_t>(
             targets[i].membership.enabled() ? MoEOverlayActivationReturnRole::JointContributor
                                             : MoEOverlayActivationReturnRole::Ordinary);
     }
     launch.shared_row_controls = controls.data();
     launch.shared_row_target_count = controls.size();
 }
 /** @brief Replay exact production row decision and useful element writer. */
 std::uint64_t replay() {
  std::uint64_t bytes=0;
  for(int row=0;row<capacity;++row){
   if(selectMoEOverlaySharedRow(launch,routes.data(),41,row)!=MoEOverlaySharedRowDecision::Needed)continue;
   for(int column=0;column<width;++column)copyMoEOverlaySharedRowElement(launch,row,column);
   bytes+=width*sizeof(float);
  }
  return bytes;
 }
};
}
/** @test Each useful physical row is written once across overlapping/noncontiguous targets. */
TEST(MoEOverlaySharedDispatchSelection, UnionPreservesUntouchedRowsAndExactLiveBytesAcrossReplay) {
 for(int degree:{2,4,8}){
  SharedRows f(degree);
  for(int reverse:{0,1}){
   if(reverse){std::reverse(f.targets.begin(),f.targets.end());f.bind();}
  for(int live:{65,1,0,17,65}){
   f.active=live;f.output.fill(-137.0f);
   std::uint64_t expected=0;
   for(int row=0;row<live;++row)if(row%4!=0)expected+=SharedRows::width*sizeof(float);
   EXPECT_EQ(f.replay(),expected);
   for(int row=0;row<SharedRows::capacity;++row)for(int col=0;col<SharedRows::width;++col)
    EXPECT_EQ(f.output[row*SharedRows::width+col],row<live && row%4!=0?f.input[row*SharedRows::width+col]:-137.0f);
  }
  }
 }
}
/** @test Empty outsider, singleton, stale epoch and conflicting whole ownership remain explicit. */
TEST(MoEOverlaySharedDispatchSelection, SingletonOutsiderAndMalformedMetadataNeverAuthorizeExtraRows) {
 SharedRows f(2);f.targets={{999,{}}};f.bind();
 f.output.fill(-137.0f);EXPECT_EQ(f.replay(),0u);
 f.launch.shared_row_controls=nullptr;f.launch.shared_row_target_count=0;f.launch.target_participant_id=31;
 EXPECT_EQ(selectMoEOverlaySharedRow(f.launch,f.routes.data(),41,2),MoEOverlaySharedRowDecision::Needed);
 EXPECT_EQ(selectMoEOverlaySharedRow(f.launch,f.routes.data(),41,1),MoEOverlaySharedRowDecision::Unneeded);
 EXPECT_EQ(f.replay(),16u*SharedRows::width*sizeof(float));
 f.targets = {{7, MoEOverlayJointRouteTarget::fromOriginalCohort(4, 7, 9, 4)}};
 f.bind();
 f.controls[0].channel.cohort_target = -1;
 EXPECT_EQ(selectMoEOverlaySharedRow(f.launch,f.routes.data(),41,1),MoEOverlaySharedRowDecision::Invalid);
 f.bind();
 f.controls[0].channel.return_role = 3u;
 EXPECT_EQ(selectMoEOverlaySharedRow(f.launch, f.routes.data(), 41, 1),
           MoEOverlaySharedRowDecision::Invalid);
 f.bind();
 f.ids[2] = std::numeric_limits<float>::quiet_NaN();
 EXPECT_EQ(selectMoEOverlaySharedRow(f.launch,f.routes.data(),41,1),MoEOverlaySharedRowDecision::Invalid);
 f.active=66;EXPECT_EQ(selectMoEOverlaySharedRow(f.launch,f.routes.data(),41,0),MoEOverlaySharedRowDecision::Invalid);
}
/** @test Changing live membership and zero-weight holes preserve useful unique bytes. */
TEST(MoEOverlaySharedDispatchSelection, LiveTargetAndZeroWeightReplayNeverCopiesUnusedRows)
{
    SharedRows f(2);
    for (int epoch : {41, 42, 43})
    {
        f.routes = {99, 0, 31, 0};
        f.routes[epoch == 42 ? 0 : 1] = MoEOverlayRouteTarget::cpuCohort(4).word();
        f.ids.fill(1.f);
        f.weights.fill(0.f);
        f.active = 9;
        f.output.fill(-137.f);
        EXPECT_EQ(f.replay(), 0u);
        f.weights[3] = 1.f; // one original slot after a zero route
        const auto expected = epoch == 42 ? 0u : SharedRows::width * sizeof(float);
        EXPECT_EQ(f.replay(), expected);
        for (int row = 0; row < SharedRows::capacity; ++row)
            for (int col = 0; col < SharedRows::width; ++col)
                EXPECT_EQ(f.output[row * SharedRows::width + col],
                          expected && row == 1 ? f.input[row * SharedRows::width + col] : -137.f);
    }
}

/** @test Original-group validation survives retirement of the frozen expert union. */
TEST(MoEOverlaySharedDispatchSelection, OriginalGroupRejectsDuplicatesAndForeignGeometry)
{
    SharedRows source(2);
    EXPECT_TRUE(validateMoEOverlaySharedDispatchGroup(4, 9, source.targets));
    auto duplicate = source.targets;
    duplicate.push_back(duplicate.front());
    EXPECT_FALSE(validateMoEOverlaySharedDispatchGroup(4, 9, duplicate));
    auto foreign = source.targets;
    foreign[0].membership =
        MoEOverlayJointRouteTarget::fromOriginalCohort(4, foreign[0].participant, 10, 4);
    EXPECT_FALSE(validateMoEOverlaySharedDispatchGroup(4, 9, foreign));
    foreign = source.targets;
    foreign[0].membership = MoEOverlayJointRouteTarget::fromOriginalCohort(4, 999, 9, 4);
    EXPECT_FALSE(validateMoEOverlaySharedDispatchGroup(4, 9, foreign));
    EXPECT_FALSE(validateMoEOverlaySharedDispatchGroup(4, 9, {}));
}

/** @test Exact signed zero filtering never drops nonzero subnormal coefficients under FTZ. */
TEST(MoEOverlaySharedDispatchSelection, SignedZerosAndSubnormalWeightsHaveExactLiveSemantics)
{
    EXPECT_TRUE(moeOverlayJointRouteWeightIsZero(0.f));
    EXPECT_TRUE(moeOverlayJointRouteWeightIsZero(-0.f));
    EXPECT_FALSE(moeOverlayJointRouteWeightIsZero(std::bit_cast<float>(std::uint32_t{1})));
    EXPECT_FALSE(
        moeOverlayJointRouteWeightIsZero(std::bit_cast<float>(std::uint32_t{0x80000001u})));
    SharedRows source(2);
    source.active = 1;
    source.ids[0] = source.ids[1] = 1.f;
    source.weights[0] = -0.f;
    source.weights[1] = 0.f;
    EXPECT_EQ(source.replay(), 0u);
    source.weights[1] = std::bit_cast<float>(std::uint32_t{1});
    EXPECT_EQ(source.replay(), SharedRows::width * sizeof(float));
}
