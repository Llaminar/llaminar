#include <gtest/gtest.h>

#include "execution/moe/MoEExpertParallelPlan.h"

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

        ExpertComputeDomain routedSingleDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain routedTensorParallelDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::LocalTP;
            domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
            domain.backend = CollectiveBackendType::RCCL;
            domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
            return domain;
        }

        ExpertRoutedTier routedTier(const std::string &name, const std::string &domain, bool fallback = false)
        {
            ExpertRoutedTier tier;
            tier.name = name;
            tier.domain = domain;
            tier.priority = 0;
            tier.fallback = fallback;
            return tier;
        }

        bool hasErrorContaining(const MoEExpertParallelValidationResult &result, const std::string &needle)
        {
            return std::any_of(result.errors.begin(), result.errors.end(), [&](const std::string &error)
                               { return error.find(needle) != std::string::npos; });
        }

        MoEExpertParallelPlan basePlanWithContinuation(ExecutionDomainDefinition continuation_domain)
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
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

        const auto result = validateMoEExpertParallelPlan(plan);

        EXPECT_TRUE(result.ok()) << (result.errors.empty() ? "" : result.errors.front());
    }

    TEST(Test__MoEContinuationDomainSpec, RejectsRoutedTensorParallelExpertsByDefault)
    {
        auto plan = basePlanWithContinuation(
            denseDomain("local_cont", ExecutionDomainScope::LOCAL,
                        {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}));
        plan.domains = {routedTensorParallelDomain("rocm_warm")};
        plan.routed_tiers = {routedTier("warm", "rocm_warm", true)};

        const auto result = validateMoEExpertParallelPlan(plan);

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "TensorParallelExperts"));
        EXPECT_TRUE(hasErrorContaining(result, "disabled by default"));
    }

    TEST(Test__MoEContinuationDomainSpec, CanExplicitlyAllowLegacyRoutedTensorParallelExperts)
    {
        auto plan = basePlanWithContinuation(
            denseDomain("local_cont", ExecutionDomainScope::LOCAL,
                        {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}));
        plan.domains = {routedTensorParallelDomain("rocm_warm")};
        plan.routed_tiers = {routedTier("warm", "rocm_warm", true)};

        MoEExpertParallelValidationOptions options;
        options.allow_routed_tensor_parallel_experts = true;
        const auto result = validateMoEExpertParallelPlan(plan, options);

        EXPECT_TRUE(result.ok()) << (result.errors.empty() ? "" : result.errors.front());
    }

} // namespace llaminar2::test