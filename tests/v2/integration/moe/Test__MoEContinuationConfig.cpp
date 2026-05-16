#include <gtest/gtest.h>

#include "config/OrchestrationConfig.h"
#include "execution/moe/MoEExpertParallelPlan.h"

#include <algorithm>
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
            domain.backend = scope == TPScope::LOCAL ? CollectiveBackendType::HOST : CollectiveBackendType::UPI;
            if (scope == TPScope::GLOBAL || scope == TPScope::NODE_LOCAL)
                domain.explicit_ranks = {0, 1};
            else
                domain.owner_rank = 0;
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
            domain.owner_rank = 0;
            return domain;
        }

        ExpertRoutedTier fallbackTier(const std::string &domain)
        {
            ExpertRoutedTier tier;
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
            case TPScope::LOCAL:
                return ExecutionDomainScope::LOCAL;
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

        OrchestrationConfig configWithContinuationAndRoutedTensorParallelExperts(TPScope continuation_scope)
        {
            OrchestrationConfig config;
            config.domain_definitions.push_back(continuationDomain("dense_cont", continuation_scope));

            auto plan = std::make_shared<MoEExpertParallelPlan>();
            plan->enabled = true;
            plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan->continuation_domain = "dense_cont";
            plan->base_model_domain = "dense_cont";
            plan->shared_expert_domain = "dense_cont";
            plan->continuation_domain_spec.domain = "dense_cont";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->continuation_domain_spec.dense_tp_enabled = true;
            plan->continuation_domain_spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
            plan->domains = {routedTensorParallelDomain("routed_tp")};
            plan->routed_tiers = {fallbackTier("routed_tp")};
            config.moe_expert_parallel_plan = std::move(plan);
            return config;
        }

        bool hasErrorContaining(const MoEExpertParallelValidationResult &result, const std::string &needle)
        {
            return std::any_of(result.errors.begin(), result.errors.end(), [&](const std::string &error)
                               { return error.find(needle) != std::string::npos; });
        }

        void expectContinuationConfigRejectsOnlyRoutedTensorParallelExperts(TPScope continuation_scope)
        {
            auto config = configWithContinuationAndRoutedTensorParallelExperts(continuation_scope);
            auto normalize_errors = normalizeMoEExpertOverlayDomains(config);
            ASSERT_TRUE(normalize_errors.empty()) << (normalize_errors.empty() ? "" : normalize_errors.front());
            ASSERT_NE(config.moe_expert_parallel_plan, nullptr);

            const auto &plan = *config.moe_expert_parallel_plan;
            ASSERT_FALSE(plan.dense_domains.empty());
            EXPECT_EQ(plan.dense_domains.front().scope, expectedExecutionScope(continuation_scope));

            const auto result = validateMoEExpertParallelPlan(plan);
            EXPECT_FALSE(result.ok());
            EXPECT_TRUE(hasErrorContaining(result, "TensorParallelExperts"));
            EXPECT_TRUE(hasErrorContaining(result, "disabled by default"));
            EXPECT_FALSE(hasErrorContaining(result, "scope=global"));
        }

    } // namespace

    TEST(Test__MoEContinuationConfig, LocalTPContinuationStillRejectsRoutedTensorParallelExpertsByDefault)
    {
        expectContinuationConfigRejectsOnlyRoutedTensorParallelExperts(TPScope::LOCAL);
    }

    TEST(Test__MoEContinuationConfig, GlobalTPContinuationStillRejectsRoutedTensorParallelExpertsByDefault)
    {
        expectContinuationConfigRejectsOnlyRoutedTensorParallelExperts(TPScope::GLOBAL);
    }

} // namespace llaminar2::test