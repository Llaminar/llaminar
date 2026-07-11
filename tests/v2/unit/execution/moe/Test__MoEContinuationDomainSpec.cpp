#include <gtest/gtest.h>

#include "execution/moe/MoERoutedExpertPlacementPlan.h"

#include <algorithm>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {

        ExecutionDomainDefinition denseDomain(
            const std::string &name,
            ExecutionDomainScope scope,
            std::vector<GlobalDeviceAddress> participants)
        {
            ExecutionDomainDefinition domain;
            domain.name = name;
            domain.scope = scope;
            domain.participants = std::move(participants);
            domain.backend = CollectiveBackendType::HOST;
            return domain;
        }

        RoutedExpertDomain routedSingleDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain routedTensorParallelDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::LOCAL;
            domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
            domain.backend = CollectiveBackendType::RCCL;
            domain.routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
            return domain;
        }

        RoutedExpertTier routedTier(const std::string &name, const std::string &domain, bool fallback = false)
        {
            RoutedExpertTier tier;
            tier.name = name;
            tier.domain = domain;
            tier.priority = 0;
            tier.fallback = fallback;
            return tier;
        }

        bool hasErrorContaining(const MoERoutedExpertPlacementValidationResult &result, const std::string &needle)
        {
            return std::any_of(result.errors.begin(), result.errors.end(), [&](const std::string &error)
                               { return error.find(needle) != std::string::npos; });
        }

        MoERoutedExpertPlacementPlan basePlanWithContinuation(ExecutionDomainDefinition continuation_domain)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = continuation_domain.name;
            plan.shared_expert_domain = continuation_domain.name;
            plan.continuation_domain_spec.domain = continuation_domain.name;
            plan.continuation_domain_spec.dense_tp_enabled = continuation_domain.participants.size() > 1;
            plan.continuation_domain_spec.logical_root_participant = 0;
            plan.continuation_domain_spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
            plan.dense_domains = {std::move(continuation_domain)};
            plan.domains = {routedSingleDomain("cpu_cold")};
            plan.routed_tiers = {routedTier("cold", "cpu_cold", true)};
            return plan;
        }

    } // namespace

    TEST(Test__MoEContinuationDomainSpec, AcceptsSingleLocalNodeLocalAndGlobalContinuationScopes)
    {
        const std::vector<ExecutionDomainDefinition> domains = {
            denseDomain("single_cont", ExecutionDomainScope::SINGLE, {GlobalDeviceAddress::cuda(0)}),
            denseDomain("local_cont", ExecutionDomainScope::LOCAL, {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}),
            denseDomain("node_cont", ExecutionDomainScope::NODE_LOCAL, {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)}),
            denseDomain("global_cont", ExecutionDomainScope::GLOBAL, {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)}),
        };

        for (const auto &domain : domains)
        {
            MoEContinuationDomainSpec spec;
            spec.domain = domain.name;
            spec.logical_root_participant = domain.participants.size() > 1 ? 1 : 0;
            spec.dense_tp_enabled = domain.participants.size() > 1;
            spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;

            const auto result = validateMoEContinuationDomainSpec(spec, domain);
            EXPECT_TRUE(result.ok()) << domain.name << ": "
                                     << (result.errors.empty() ? "" : result.errors.front());
        }
    }

    TEST(Test__MoEContinuationDomainSpec, AllowsGlobalContinuationDomainWithoutRoutedDomainConversion)
    {
        auto plan = basePlanWithContinuation(
            denseDomain("global_cont", ExecutionDomainScope::GLOBAL,
                        {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)}));

        const auto result = validateMoERoutedExpertPlacementPlan(plan);

        EXPECT_TRUE(result.ok()) << (result.errors.empty() ? "" : result.errors.front());
    }

    TEST(Test__MoEContinuationDomainSpec, AcceptsExplicitRoutedExpertTensorSharding)
    {
        auto plan = basePlanWithContinuation(
            denseDomain("local_cont", ExecutionDomainScope::LOCAL,
                        {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}));
        plan.domains = {routedTensorParallelDomain("rocm_warm")};
        plan.routed_tiers = {routedTier("warm", "rocm_warm", true)};

        const auto result = validateMoERoutedExpertPlacementPlan(plan);

        EXPECT_TRUE(result.ok()) << (result.errors.empty() ? "" : result.errors.front());
    }

    TEST(Test__MoEContinuationDomainSpec, RejectsTensorShardingWithoutCollectiveParticipants)
    {
        auto plan = basePlanWithContinuation(
            denseDomain("local_cont", ExecutionDomainScope::LOCAL,
                        {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}));
        auto invalid_domain = routedTensorParallelDomain("rocm_warm");
        invalid_domain.scope = ExecutionDomainScope::SINGLE;
        invalid_domain.participants = {GlobalDeviceAddress::rocm(0)};
        plan.domains = {std::move(invalid_domain)};
        plan.routed_tiers = {routedTier("warm", "rocm_warm", true)};

        const auto result = validateMoERoutedExpertPlacementPlan(plan);

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "routed_compute=tensor-sharded"));
    }

} // namespace llaminar2::test
