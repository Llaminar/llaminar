/**
 * @file Test__Qwen4ExpReplicatedPLEAdmission.cpp
 * @brief Complete immutable admission identity at the mirrored generation boundary.
 *
 * This is an actual production validator used after every child materializes.
 * It proves request-law/seed ownership without reading a token/count/history
 * value or issuing a cohort/query capability.
 */
#include "models/qwen4exp/Qwen4ExpReplicatedPLECohort.h"
#include "execution/mtp/DeviceGenerationContract.h"
#include <gtest/gtest.h>
using namespace llaminar2;
/** @test Every immutable record remains exact; equal policy labels alone are insufficient. */
TEST(Qwen4ExpReplicatedPLEAdmission, SeedsBudgetAndCompleteSamplingLawMustMatch)
{
    DeviceGenerationAdmissionRequest original;
    original.request_count=1;original.max_new_tokens=8;
    original.depth_policy=sampling_math::DeviceGenerationPolicy::ordinary();
    original.sampling_seeds=GenerationRequestSeeds({123});
    original.ordinary_sampling=SamplingParams{.temperature=0.7F,.top_k=16,.top_p=0.9F,.seed=123};
    ASSERT_TRUE(original.valid());
    EXPECT_NO_THROW(Qwen4ExpReplicatedPLECohort::requireSameGenerationAdmission(original,original));
    for(const int change:{0,1,2,3,4,5}) {
        auto participant=original;
        switch(change) {
        case 0:participant.sampling_seeds=GenerationRequestSeeds({124});break;
        case 1:participant.max_new_tokens++;break;
        case 2:participant.ordinary_sampling->top_p=0.8F;break;
        case 3:participant.ordinary_sampling->presence_penalty=0.1F;break;
        case 4:participant.ordinary_sampling->dry_sequence_breakers.push_back("x");break;
        case 5:participant.initial_leading_row_disposition=sampling_math::DeviceGenerationLeadingRowDisposition::AlreadyEmitted;break;
        }
        EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireSameGenerationAdmission(original,participant),std::invalid_argument);
    }
    auto forward=original;forward.depth_policy=sampling_math::DeviceGenerationPolicy::forwardOnly();
    forward.ordinary_sampling.reset();forward.sampling_seeds.reset();
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireSameGenerationAdmission(forward,forward),std::invalid_argument);
}
/** @test Dynamic selector policy metadata cannot be silently reinterpreted by one participant. */
TEST(Qwen4ExpReplicatedPLEAdmission, FrozenDepthPolicyIdentityIncludesAllControllerParameters)
{
    DeviceGenerationAdmissionRequest original;original.request_count=1;original.max_new_tokens=16;
    original.depth_policy=sampling_math::DeviceGenerationPolicy::fixed(2);
    original.sampling_seeds=GenerationRequestSeeds({789});
    ASSERT_TRUE(original.valid());
    auto participant=original;participant.depth_policy.cooldown_steps++;
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireSameGenerationAdmission(original,participant),std::invalid_argument);
    participant=original;participant.depth_policy.learned.model_class=MTPDepthPolicyModelClass::MoE;
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireSameGenerationAdmission(original,participant),std::invalid_argument);
    EXPECT_NO_THROW(Qwen4ExpReplicatedPLECohort::requireSameGenerationAdmission(original,original));
}
