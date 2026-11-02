/**
 * @file Test__PooledImmutableParticipantExecutionPolicy.cpp
 * @brief Original remote participant admission preserves Off capacity and bounds active request policies.
 */
#include "execution/qwen4exp/PooledImmutableParticipantExecutionPolicy.h"
#include <gtest/gtest.h>
#include "execution/mtp/MTPDeviceGenerationPolicy.h"
namespace llaminar2::test {
TEST(PooledImmutableParticipantExecutionPolicy, OffRetainsCapacityAndBoundsRequestSwitches){
    using Policy=qwen4exp::PooledImmutableParticipantExecutionPolicy;
    for(const int capacity:{0,1,3,sampling_math::DeviceGenerationPolicy::kMaximumSupportedDraftDepth}){
        MTPRuntimeConfig off;off.enabled=false;off.graph_capacity_draft_tokens=capacity;
        const auto before=off.graph_capacity_draft_tokens;
        EXPECT_NO_THROW(Policy::requireConfiguration(off));
        EXPECT_EQ(off.graph_capacity_draft_tokens,before);
        EXPECT_EQ(retainsMTPGraphCapacity(off),capacity>0);
        EXPECT_NO_THROW(Policy::requireAdmission(sampling_math::DeviceGenerationPolicy::ordinary(),off));
        EXPECT_NO_THROW(Policy::requireAdmission(sampling_math::DeviceGenerationPolicy::forwardOnly(),off));
        for(const int depth:{1,3}){
            const auto request=sampling_math::DeviceGenerationPolicy::fixed(depth);
            if(depth<=capacity)EXPECT_NO_THROW(Policy::requireAdmission(request,off));
            else EXPECT_THROW(Policy::requireAdmission(request,off),std::invalid_argument);
        }
        auto dynamic=sampling_math::DeviceGenerationPolicy::fixed(1);
        dynamic.mode=sampling_math::DeviceGenerationPolicyMode::Dynamic;dynamic.maximum_depth=3;
        ASSERT_TRUE(dynamic.valid());
        if(capacity>=3)EXPECT_NO_THROW(Policy::requireAdmission(dynamic,off));
        else EXPECT_THROW(Policy::requireAdmission(dynamic,off),std::invalid_argument);

    }
    EXPECT_THROW(Policy::requireAdmission(sampling_math::DeviceGenerationPolicy::fixed(0),MTPRuntimeConfig{}),std::invalid_argument);
}
/** @test Real host-to-controller conversion admits fixed and dynamic requests within the same retained capacity. */
TEST(PooledImmutableParticipantExecutionPolicy, ActiveCanonicalPoliciesRequireOriginalRetainedCapacity){
 using Policy=qwen4exp::PooledImmutableParticipantExecutionPolicy;
 for(const int depth:{1,3}){
  MTPRuntimeConfig active;active.enabled=true;active.draft_tokens=depth;active.graph_capacity_draft_tokens=15;
  const auto before=active.graph_capacity_draft_tokens;
  const auto request=resolveMTPDeviceGenerationDepthPolicy(active);ASSERT_TRUE(request.valid());
  EXPECT_NO_THROW(Policy::requireConfiguration(active));
  EXPECT_NO_THROW(Policy::requireAdmission(request,active));
  EXPECT_EQ(active.graph_capacity_draft_tokens,before);EXPECT_TRUE(active.enabled);
  auto tooSmall=active;tooSmall.graph_capacity_draft_tokens=depth-1;
  if(depth>1)EXPECT_THROW(Policy::requireConfiguration(tooSmall),std::invalid_argument);
 }
 MTPRuntimeConfig dynamic;dynamic.enabled=true;dynamic.graph_capacity_draft_tokens=15;
 dynamic.depth_policy.mode=MTPDepthPolicyMode::Dynamic;dynamic.depth_policy.min_depth=1;
 dynamic.depth_policy.max_depth=15;dynamic.depth_policy.initial_depth=1;
 const auto request=resolveMTPDeviceGenerationDepthPolicy(dynamic);ASSERT_TRUE(request.valid());
 EXPECT_NO_THROW(Policy::requireConfiguration(dynamic));
 EXPECT_NO_THROW(Policy::requireAdmission(request,dynamic));
 auto insufficient=dynamic;insufficient.depth_policy.max_depth=3;insufficient.graph_capacity_draft_tokens=3;
 EXPECT_THROW(Policy::requireAdmission(request,insufficient),std::invalid_argument);
 auto batched=dynamic;batched.max_request_batch=2;EXPECT_THROW(Policy::requireConfiguration(batched),std::invalid_argument);
 auto negative=dynamic;negative.graph_capacity_draft_tokens=-1;EXPECT_THROW(Policy::requireConfiguration(negative),std::invalid_argument);
 auto malformed=request;malformed.minimum_depth=0;EXPECT_THROW(Policy::requireAdmission(malformed,dynamic),std::invalid_argument);
}

}
