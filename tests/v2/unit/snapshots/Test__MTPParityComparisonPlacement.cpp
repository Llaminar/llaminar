/**
 * @file Test__MTPParityComparisonPlacement.cpp
 * @brief Device-free regression for placement-owned serial/verifier comparison dispatch.
 *
 * These tests execute the same dispatch helper as full parity. The fixture
 * records API selection and exception propagation, never fabricates native
 * parent capture, physical placement or accelerator qualification.
 */
#include "../../utils/MTPParityComparisonPlacement.h"
#include "config/OrchestrationConfig.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace llaminar2::test::parity
{
    namespace
    {
        /** Original dispatch surface only; native scope behavior has its own replay tests. */
        struct ComparisonRunner
        {
            OrchestrationConfig configuration;
            int host_calls = 0;
            int device_calls = 0;
            bool reject_device_scope = false;

            /** @return Immutable admitted configuration used by the real helper. */
            const OrchestrationConfig &config() const { return configuration; }

            /** @brief Record exactly one original host scope around the unchanged callback. */
            void withMoEOverlayInferencePlacementScope(const std::function<void()> &body)
            {
                ++host_calls;
                body();
            }

            /** @brief Preserve a native admission rejection without executing the callback. */
            void withMoEOverlayDevicePlacementScope(const std::function<void()> &body)
            {
                ++device_calls;
                if (reject_device_scope)
                    throw std::runtime_error("original native scope unavailable");
                body();
            }
        };

        /** @brief Build only canonical authority metadata, not a physical admission. */
        std::shared_ptr<MoERoutedExpertPlacementPlan> placement(
            std::vector<GlobalDeviceAddress> participants, RoutedExpertComputePolicy policy)
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->domains.push_back(RoutedExpertDomain{
                .name = "experts", .participants = std::move(participants),
                .routed_compute_policy = policy});
            plan->routed_tiers.push_back(RoutedExpertTier{.name = "capacity", .domain = "experts"});
            plan->authority_execution = resolveMoEOverlayAuthorityExecutionKind(*plan);
            return plan;
        }
    }

    /** @test CUDA, ROCm and mixed-vendor metadata select the original device scope. */
    TEST(MTPParityComparisonPlacement, DeviceAuthorityUsesOriginalScope)
    {
        for (const auto &participants : std::vector<std::vector<GlobalDeviceAddress>>{
                 {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                 {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                 {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)}})
        {
            ComparisonRunner runner;
            runner.configuration.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
            runner.configuration.moe_routed_expert_plan = placement(participants, RoutedExpertComputePolicy::TensorSharded);
            int calls = 0;
            withMTPParityComparisonPlacement(runner, [&] { ++calls; });
            EXPECT_EQ(calls, 1);
            EXPECT_EQ(runner.device_calls, 1);
            EXPECT_EQ(runner.host_calls, 0);
        }
    }

    /** @test Tensor-sharded CPU cohorts retain their original host-owned scope. */
    TEST(MTPParityComparisonPlacement, CPUJointRetainsOriginalScope)
    {
        for (const int sockets : {2, 4, 8})
        {
            std::vector<GlobalDeviceAddress> participants;
            for (int socket = 0; socket < sockets; ++socket)
                participants.push_back(GlobalDeviceAddress::cpu(socket));
            ComparisonRunner runner;
            runner.configuration.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
            runner.configuration.moe_routed_expert_plan = placement(participants, RoutedExpertComputePolicy::TensorSharded);
            int calls = 0;
            withMTPParityComparisonPlacement(runner, [&] { ++calls; });
            EXPECT_EQ(calls, 1);
            EXPECT_EQ(runner.host_calls, 1);
            EXPECT_EQ(runner.device_calls, 0);
        }
    }

    /** @test Static, non-overlay and apportioned CPU behavior is preserved. */
    TEST(MTPParityComparisonPlacement, ExistingUnscopedPoliciesRemainUnchanged)
    {
        ComparisonRunner runner;
        runner.configuration.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        int calls = 0;
        withMTPParityComparisonPlacement(runner, [&] { ++calls; });
        runner.configuration.moe_routed_expert_plan = placement(
            {GlobalDeviceAddress::cuda(0)}, RoutedExpertComputePolicy::Apportioned);
        withMTPParityComparisonPlacement(runner, [&] { ++calls; });
        runner.configuration.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        runner.configuration.moe_routed_expert_plan = placement(
            {GlobalDeviceAddress::cpu(0)}, RoutedExpertComputePolicy::Apportioned);
        withMTPParityComparisonPlacement(runner, [&] { ++calls; });
        EXPECT_EQ(calls, 3);
        EXPECT_EQ(runner.host_calls, 0);
        EXPECT_EQ(runner.device_calls, 0);
    }

    /** @test An unavailable scope rejects; no unprotected second attempt runs. */
    TEST(MTPParityComparisonPlacement, NativeRejectionNeverExecutesUnprotectedBody)
    {
        ComparisonRunner runner;
        runner.configuration.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        runner.configuration.moe_routed_expert_plan = placement(
            {GlobalDeviceAddress::rocm(0)}, RoutedExpertComputePolicy::Apportioned);
        runner.reject_device_scope = true;
        int calls = 0;
        EXPECT_THROW(withMTPParityComparisonPlacement(runner, [&] { ++calls; }), std::runtime_error);
        EXPECT_EQ(calls, 0);
        EXPECT_EQ(runner.device_calls, 1);
        EXPECT_EQ(runner.host_calls, 0);
    }

    /** @test The callback's original exception escapes unchanged after one invocation. */
    TEST(MTPParityComparisonPlacement, OriginalCallbackFailureIsNotRetried)
    {
        ComparisonRunner runner;
        runner.configuration.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        runner.configuration.moe_routed_expert_plan = placement(
            {GlobalDeviceAddress::cuda(0)}, RoutedExpertComputePolicy::Apportioned);
        int calls = 0;
        const auto failure = std::make_exception_ptr(std::runtime_error("original comparison failed"));
        try
        {
            withMTPParityComparisonPlacement(runner, [&] { ++calls; std::rethrow_exception(failure); });
            FAIL() << "Original failure was swallowed";
        }
        catch (...)
        {
            EXPECT_EQ(std::current_exception(), failure);
        }
        EXPECT_EQ(calls, 1);
        EXPECT_EQ(runner.device_calls, 1);
    }

    /** @test Missing or contradictory frozen identity fails before any scope/body is entered. */
    TEST(MTPParityComparisonPlacement, RejectsUnresolvedOrContradictoryAuthority)
    {
        for (const auto authority : {MoEOverlayAuthorityExecutionKind::Unresolved,
                                     MoEOverlayAuthorityExecutionKind::HostResident})
        {
            ComparisonRunner runner;
            runner.configuration.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
            auto plan = placement({GlobalDeviceAddress::cuda(0)}, RoutedExpertComputePolicy::Apportioned);
            plan->authority_execution = authority;
            runner.configuration.moe_routed_expert_plan = plan;
            int calls = 0;
            EXPECT_THROW(withMTPParityComparisonPlacement(runner, [&] { ++calls; }), std::logic_error);
            EXPECT_EQ(calls, 0);
            EXPECT_EQ(runner.host_calls + runner.device_calls, 0);
        }
    }
} // namespace llaminar2::test::parity
