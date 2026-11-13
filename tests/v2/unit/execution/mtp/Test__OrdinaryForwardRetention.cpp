/**
 * @file Test__OrdinaryForwardRetention.cpp
 * @brief Device-free ownership and immutable retained-forward identity regressions.
 *
 * Hosted export is selected by the declared complete execution policy, never
 * by failure of native composition. The retained view carries no capture or
 * ownership extension; every frozen cache coordinate must still match.
 */
#include "execution/mtp/OrdinaryForwardRetention.h"
#include "execution/local_execution/engine/ForwardExecutionEngine.h"
#include <gtest/gtest.h>
namespace llaminar2::test
{
TEST(OrdinaryForwardRetention, ParticipantLocalTransactionsRetainNativeComposition)
{
    for (const auto policy : {DeviceGenerationExecutionPolicy::NativeConditionalGraph,
                             DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions})
        EXPECT_EQ(resolveOrdinaryForwardRetention(policy, OrdinaryForwardDomain::ParticipantLocal),
                  OrdinaryForwardRetentionKind::NativeComposable);
}
TEST(OrdinaryForwardRetention, HostedOverlayRequiresItsExplicitExecutionPolicy)
{
    EXPECT_EQ(resolveOrdinaryForwardRetention(DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions,
        OrdinaryForwardDomain::HostedExpertOverlay), OrdinaryForwardRetentionKind::HostedExpertOverlay);
    EXPECT_THROW((void)resolveOrdinaryForwardRetention(DeviceGenerationExecutionPolicy::NativeConditionalGraph,
        OrdinaryForwardDomain::HostedExpertOverlay), std::invalid_argument);
    for (const auto domain : {OrdinaryForwardDomain::ParticipantLocal, OrdinaryForwardDomain::HostedExpertOverlay})
    {
        EXPECT_THROW((void)resolveOrdinaryForwardRetention(DeviceGenerationExecutionPolicy::Unsupported, domain), std::invalid_argument);
        EXPECT_THROW((void)resolveOrdinaryForwardRetention(static_cast<DeviceGenerationExecutionPolicy>(255), domain), std::invalid_argument);
    }
    EXPECT_THROW((void)resolveOrdinaryForwardRetention(DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions,
        static_cast<OrdinaryForwardDomain>(255)), std::invalid_argument);
}
TEST(OrdinaryForwardRetention, HostedCacheIdentityRejectsEveryReplacedOwnerCoordinate)
{
    int owners[3]{};
    ForwardExecutionEngine::RetainedDecodeGraphView baseline;
    baseline.signature.device = baseline.device = DeviceId::cuda(0);
    baseline.signature.decode = true;
    baseline.graph_owner = reinterpret_cast<const ComputeGraph *>(owners);
    baseline.stream = owners + 1;
    baseline.topology_generation = 3;
    baseline.workspace_generation = 4;
    baseline.capture_variant_signature = 5;
    baseline.capture_generation = 6;
    baseline.replay_unit_count = baseline.source_capture_unit_count = 2;
    baseline.segmented = true;
    const auto reject = [&](auto change) { auto changed = baseline; change(changed); EXPECT_NE(changed, baseline); };
    reject([&](auto &v) { v.graph_owner = reinterpret_cast<const ComputeGraph *>(owners + 1); });
    reject([&](auto &v) { v.stream = owners + 2; });
    reject([](auto &v) { ++v.topology_generation; });
    reject([](auto &v) { ++v.workspace_generation; });
    reject([](auto &v) { ++v.capture_variant_signature; });
    reject([](auto &v) { ++v.capture_generation; });
    reject([](auto &v) { ++v.replay_unit_count; });
    reject([](auto &v) { ++v.source_capture_unit_count; });
    reject([](auto &v) { v.segmented = false; });
    reject([](auto &v) { v.composed_parent = true; });
    reject([](auto &v) { v.signature.seq_len = 2; });
    reject([](auto &v) { v.device = DeviceId::rocm(0); });
    EXPECT_EQ(baseline, baseline);
}
}
