#include <gtest/gtest.h>

#include "config/OrchestrationConfig.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {

        DomainDefinition continuationDomain(const std::string &name, TPScope scope)
        {
            DomainDefinition domain;
            domain.name = name;
            domain.devices = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
            domain.scope = scope;
            domain.backend = scope == TPScope::RANK_LOCAL ? CollectiveBackendType::HOST : CollectiveBackendType::UPI;
            if (scope == TPScope::GLOBAL || scope == TPScope::NODE_LOCAL)
                domain.explicit_ranks = {0, 1};
            else
                domain.owner_rank = 0;
            return domain;
        }

        RoutedExpertDomain routedTensorShardedDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::RANK_LOCAL;
            domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
            domain.backend = CollectiveBackendType::RCCL;
            domain.routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
            domain.owner_rank = 0;
            return domain;
        }

        RoutedExpertTier fallbackTier(const std::string &domain)
        {
            RoutedExpertTier tier;
            tier.name = "fallback";
            tier.domain = domain;
            tier.priority = 0;
            tier.fallback = true;
            return tier;
        }

        ExecutionDomainScope expectedExecutionScope(TPScope scope)
        {
            switch (scope)
            {
            case TPScope::RANK_LOCAL:
                return ExecutionDomainScope::RANK_LOCAL;
            case TPScope::NODE_LOCAL:
                return ExecutionDomainScope::NODE_LOCAL;
            case TPScope::GLOBAL:
                return ExecutionDomainScope::GLOBAL;
            case TPScope::AUTO:
            case TPScope::HYBRID:
                return ExecutionDomainScope::AUTO;
            }
            return ExecutionDomainScope::AUTO;
        }

        OrchestrationConfig configWithIndependentRoutedTensorSharding(TPScope continuation_scope)
        {
            OrchestrationConfig config;
            config.domain_definitions.push_back(continuationDomain("dense_cont", continuation_scope));

            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "dense_cont";
            plan->base_model_domain = "dense_cont";
            plan->shared_expert_domain = "dense_cont";
            plan->continuation_domain_spec.domain = "dense_cont";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->continuation_domain_spec.dense_tp_enabled = true;
            plan->continuation_domain_spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
            plan->domains = {routedTensorShardedDomain("routed_tp")};
            plan->routed_tiers = {fallbackTier("routed_tp")};
            config.moe_routed_expert_plan = std::move(plan);
            return config;
        }

        void expectContinuationConfigPreservesIndependentRoutedTensorSharding(
            TPScope continuation_scope)
        {
            auto config = configWithIndependentRoutedTensorSharding(continuation_scope);
            auto normalize_errors = normalizeMoERoutedExpertPlacementDomains(config);
            ASSERT_TRUE(normalize_errors.empty()) << (normalize_errors.empty() ? "" : normalize_errors.front());
            ASSERT_NE(config.moe_routed_expert_plan, nullptr);

            const auto &plan = *config.moe_routed_expert_plan;
            ASSERT_FALSE(plan.dense_domains.empty());
            EXPECT_EQ(plan.dense_domains.front().scope, expectedExecutionScope(continuation_scope));

            const auto result = validateMoERoutedExpertPlacementPlan(plan);
            EXPECT_TRUE(result.ok())
                << (result.errors.empty() ? "" : result.errors.front());
        }

    } // namespace

    TEST(Test__MoEContinuationConfig, LocalTPContinuationKeepsRoutedTensorShardingIndependent)
    {
        expectContinuationConfigPreservesIndependentRoutedTensorSharding(TPScope::RANK_LOCAL);
    }

    TEST(Test__MoEContinuationConfig, GlobalTPContinuationKeepsRoutedTensorShardingIndependent)
    {
        expectContinuationConfigPreservesIndependentRoutedTensorSharding(TPScope::GLOBAL);
    }

} // namespace llaminar2::test
