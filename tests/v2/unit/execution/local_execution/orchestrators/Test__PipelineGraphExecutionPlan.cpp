/**
 * @file Test__PipelineGraphExecutionPlan.cpp
 * @brief Device-free proof of complete pipeline domain identity and setup state.
 *
 * A TP domain is not its leader. These tests retain all participants, vary
 * vendor order and width, and reject malformed membership before any graph
 * capture or allocation. They certify topology/lifecycle contracts, not GPU
 * execution support or an E2E result.
 */
#include <gtest/gtest.h>
#include "execution/local_execution/orchestrators/PipelineGraphExecutionPlan.h"

using namespace llaminar2;

TEST(PipelineGraphExecutionPlanTest, HeterogeneousPlanRequiresExactMaterializationAndReplayTraversal)
{
    PipelineGraphExecutionPlan plan({
        {.stage_index = 0, .participants = {DeviceId::cuda(0)},
         .execution = PipelineGraphSegmentExecution::NativeDeviceExecutable},
        {.stage_index = 1, .participants = {DeviceId::cpu()},
         .execution = PipelineGraphSegmentExecution::HostDeclarative}});
    EXPECT_TRUE(plan.hasHeterogeneousBoundary());
    EXPECT_EQ(plan.segmentCount(), 2u);
    EXPECT_EQ(plan.participantCount(), 2u);
    EXPECT_EQ(plan.nativeSegmentCount(), 1u);
    EXPECT_EQ(plan.hostSegmentCount(), 1u);
    std::string error;
    EXPECT_FALSE(plan.certifiesReplay(2, &error));
    EXPECT_NE(error.find("preceded"), std::string::npos);
    EXPECT_FALSE(plan.markMaterialized(0, &error));
    EXPECT_NE(error.find("requires 1"), std::string::npos);
    EXPECT_TRUE(plan.markMaterialized(1, &error)) << error;
    EXPECT_TRUE(plan.markMaterialized(1, &error)) << error;
    EXPECT_TRUE(plan.certifiesReplay(2, &error)) << error;
    EXPECT_FALSE(plan.certifiesReplay(1, &error));
    EXPECT_NE(error.find("requires 2"), std::string::npos);
}

TEST(PipelineGraphExecutionPlanTest, SameBackendGpuOrdinalsDoNotClaimHeterogeneousSegmentation)
{
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        PipelineGraphExecutionPlan plan({
            {.stage_index = 0, .participants = {DeviceId{type, 5}, DeviceId{type, 3}},
             .execution = PipelineGraphSegmentExecution::NativeDeviceExecutable},
            {.stage_index = 1, .participants = {DeviceId{type, 7}},
             .execution = PipelineGraphSegmentExecution::NativeDeviceExecutable}});
        EXPECT_FALSE(plan.hasHeterogeneousBoundary());
        EXPECT_EQ(plan.nativeSegmentCount(), 2u);
        EXPECT_EQ(plan.participantCount(), 3u);
        EXPECT_EQ(plan.segments()[0].primaryDevice(), (DeviceId{type, 5}));
        EXPECT_TRUE(plan.markMaterialized(2));
        EXPECT_TRUE(plan.certifiesReplay(2));
    }
}

TEST(PipelineGraphExecutionPlanTest, PreservesUnequalTPWidthsAndBothVendorOrders)
{
    for (const auto first : {DeviceType::CUDA, DeviceType::ROCm})
        for (const int first_width : {1, 2, 4, 8})
            for (const int second_width : {1, 2, 4, 8})
            {
                const auto second = first == DeviceType::CUDA ? DeviceType::ROCm : DeviceType::CUDA;
                std::vector<PipelineGraphExecutionSegment> domains;
                for (size_t index = 0; index < 2; ++index)
                {
                    PipelineGraphExecutionSegment domain{.stage_index = index,
                        .execution = PipelineGraphSegmentExecution::NativeDeviceExecutable};
                    const int width = index ? second_width : first_width;
                    // Preserve communicator order; sorting physical ordinals
                    // would silently change which rank owns each tensor shard.
                    for (int member = width; member > 0; --member)
                        domain.participants.emplace_back(index ? second : first, member);
                    domains.push_back(std::move(domain));
                }
                PipelineGraphExecutionPlan plan(std::move(domains));
                EXPECT_TRUE(plan.hasHeterogeneousBoundary());
                EXPECT_EQ(plan.participantCount(), static_cast<size_t>(first_width + second_width));
                EXPECT_EQ(plan.segments()[0].primaryDevice(), (DeviceId{first, first_width}));
                EXPECT_EQ(plan.segments()[1].primaryDevice(), (DeviceId{second, second_width}));
            }
}

TEST(PipelineGraphExecutionPlanTest, RejectsMalformedDomainMembershipBeforeMaterialization)
{
    const std::vector<std::vector<DeviceId>> invalid{
        {}, {DeviceId::invalid()}, {DeviceId::cuda(0), DeviceId::invalid()},
        {DeviceId::cuda(1), DeviceId::cuda(1)},
        {DeviceId::rocm(2), DeviceId::rocm(2)},
        {DeviceId::cuda(0), DeviceId::rocm(0)},
        {DeviceId::rocm(0), DeviceId::cpu()}};
    for (const auto &members : invalid)
    {
        EXPECT_THROW((PipelineGraphExecutionPlan({
            {.stage_index = 0, .participants = members,
             .execution = PipelineGraphSegmentExecution::NativeDeviceExecutable}})), std::invalid_argument);
    }
    EXPECT_THROW((PipelineGraphExecutionPlan({
        {.stage_index = 1, .participants = {DeviceId::cpu()},
         .execution = PipelineGraphSegmentExecution::HostDeclarative}})), std::invalid_argument);
    EXPECT_THROW((PipelineGraphExecutionPlan({
        {.stage_index = 0, .participants = {DeviceId::cuda(0)},
         .execution = PipelineGraphSegmentExecution::HostDeclarative}})), std::invalid_argument);
}

TEST(PipelineGraphExecutionPlanTest, CPUExecutionIdentityDoesNotInventPhysicalNUMAOwnership)
{
    // DeviceId is a rank-local execution kind. Existing CPU collective context
    // membership, not this graph projection, distinguishes physical NUMA nodes.
    PipelineGraphExecutionPlan plan({
        {.stage_index = 0, .participants = {DeviceId::cpu(), DeviceId::cpu()},
         .execution = PipelineGraphSegmentExecution::HostDeclarative},
        {.stage_index = 1, .participants = {DeviceId::rocm(4)},
         .execution = PipelineGraphSegmentExecution::NativeDeviceExecutable}});
    EXPECT_EQ(plan.participantCount(), 3u);
    EXPECT_TRUE(plan.hasHeterogeneousBoundary());
}
