/**
 * @file Test__MoEGraphNativeProductionHardening.cpp
 * @brief Phase 21 graph-native MoE overlay production hardening tests.
 */

#include "config/OrchestrationConfigParser.h"
#include "execution/config/RoutedExpertPolicy.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    TEST(Test__MoEGraphNativeProductionHardening,
         RouteAccumulationSelectsOnlySingleDeviceROCmGroupedVerifier)
    {
        using Workload = MoERouteAccumulationWorkload;
        using Policy = MoERouteAccumulationPolicy;

        EXPECT_EQ(
            selectMoERouteAccumulationPolicy(
                MoERouteAccumulationSelection{
                    .backend = DeviceType::ROCm,
                    .participant_count = 1,
                    .workload = Workload::GroupedVerifier}),
            Policy::IndependentRouteSlotsThenOrderedFold);

        for (const DeviceType backend :
             {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        {
            SCOPED_TRACE(static_cast<int>(backend));
            EXPECT_EQ(
                selectMoERouteAccumulationPolicy(
                    MoERouteAccumulationSelection{
                        .backend = backend,
                        .participant_count = 1,
                        .workload = Workload::Ordinary}),
                Policy::DirectOrderedFold);
            EXPECT_EQ(
                selectMoERouteAccumulationPolicy(
                    MoERouteAccumulationSelection{
                        .backend = backend,
                        .participant_count = 2,
                        .workload = Workload::GroupedVerifier}),
                Policy::DirectOrderedFold);
        }

        EXPECT_EQ(
            selectMoERouteAccumulationPolicy(
                MoERouteAccumulationSelection{
                    .backend = DeviceType::CUDA,
                    .participant_count = 1,
                    .workload = Workload::GroupedVerifier}),
            Policy::DirectOrderedFold);
    }

    namespace
    {

        RoutedExpertDomain singleDeviceDomain(
            const std::string &name,
            GlobalDeviceAddress participant,
            CollectiveBackendType backend,
            int owner_rank)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = backend;
            domain.participants = {std::move(participant)};
            domain.owner_rank = owner_rank;
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain localTPDomain(
            const std::string &name,
            CollectiveBackendType backend,
            int owner_rank = 1)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::LOCAL;
            domain.backend = backend;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0), GlobalDeviceAddress::rocm(1, 0)};
            domain.owner_rank = owner_rank;
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertTier routedTier(
            const std::string &name,
            const std::string &domain,
            int priority,
            int capacity,
            bool fallback = false)
        {
            RoutedExpertTier tier;
            tier.name = name;
            tier.domain = domain;
            tier.priority = priority;
            tier.max_experts_per_layer = capacity;
            tier.fallback = fallback;
            return tier;
        }

        MoERoutedExpertPlacementPlan productionPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.base_model_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.continuation_domain_spec.domain = "cuda_hot";
            plan.continuation_domain_spec.logical_root_participant = 0;
            plan.continuation_domain_spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
            plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan.domains = {
                singleDeviceDomain("cuda_hot", GlobalDeviceAddress::cuda(0, 0), CollectiveBackendType::NCCL, 0),
                localTPDomain("rocm_warm", CollectiveBackendType::RCCL, 1),
                singleDeviceDomain("cpu_cold", GlobalDeviceAddress::cpu(0), CollectiveBackendType::HOST, 2),
            };
            plan.dense_domains = {plan.domains.front().toExecutionDomainDefinition()};
            plan.routed_tiers = {
                routedTier("hot", "cuda_hot", 0, 2),
                routedTier("warm", "rocm_warm", 1, 1),
                routedTier("cold", "cpu_cold", 2, 0, true),
            };
            return plan;
        }

        bool hasErrorContaining(const MoERoutedExpertPlacementValidationResult &result, const std::string &needle)
        {
            return std::any_of(result.errors.begin(), result.errors.end(), [&](const std::string &error)
                               { return error.find(needle) != std::string::npos; });
        }

        std::string parseArgsError(std::initializer_list<const char *> args)
        {
            std::vector<std::string> strings;
            std::vector<char *> argv;
            for (const char *arg : args)
                strings.emplace_back(arg);
            for (auto &arg : strings)
                argv.push_back(arg.data());

            OrchestrationConfigParser parser;
            try
            {
                (void)parser.parseArgs(static_cast<int>(argv.size()), argv.data());
            }
            catch (const std::invalid_argument &error)
            {
                return error.what();
            }
            return {};
        }

    } // namespace

    TEST(Test__MoEGraphNativeProductionHardening, ExplanationUsesExplicitRoutedExpertPolicyAxes)
    {
        const auto plan = productionPlan();
        const std::string explanation = renderMoERoutedExpertPlacementPlanExplanation(plan);

        EXPECT_NE(explanation.find("topology: graph-native tiered-overlay"), std::string::npos);
        EXPECT_NE(explanation.find("routed_compute=apportioned"), std::string::npos);
        EXPECT_NE(explanation.find("routed_decode_assignment=static-owner"), std::string::npos);
        EXPECT_NE(explanation.find("routed_prefill_assignment=static-owner"), std::string::npos);
        EXPECT_NE(explanation.find("continuation_domain: cuda_hot"), std::string::npos);
        EXPECT_NE(explanation.find("routed_domains:"), std::string::npos);
        EXPECT_NE(explanation.find("routed_tiers:"), std::string::npos);
        EXPECT_NE(explanation.find("total_non_fallback_capacity: 3 experts/layer"), std::string::npos);
        EXPECT_NE(explanation.find("fallback: count=1 domains=[cpu_cold] cpu_fallback=true"), std::string::npos);
        EXPECT_NE(explanation.find("model expert count is not known at config parse time"), std::string::npos);
    }

    TEST(Test__MoEGraphNativeProductionHardening, OrchestrationToStringIncludesRoutedExpertPlacementExplanation)
    {
        OrchestrationConfig config;
        config.explain_placement = true;
        config.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(productionPlan());

        const std::string text = config.toString();

        EXPECT_NE(text.find("moe_routed_expert_placement:"), std::string::npos);
        EXPECT_NE(text.find("topology: graph-native tiered-overlay"), std::string::npos);
        EXPECT_NE(text.find("coverage is validated during model-aware resolution"), std::string::npos);
    }

    TEST(Test__MoEGraphNativeProductionHardening, ExplanationReportsKnownCoverageWithoutFallback)
    {
        auto plan = productionPlan();
        plan.domains.pop_back();
        plan.routed_tiers.pop_back();

        const std::string explanation = renderMoERoutedExpertPlacementPlanExplanation(plan, 4);

        EXPECT_NE(explanation.find("coverage: incomplete without fallback (3/4)"), std::string::npos);
    }

    TEST(Test__MoEGraphNativeProductionHardening, ValidatorRejectsNoFallbackInsufficientCapacityWhenModelCountKnown)
    {
        auto plan = productionPlan();
        plan.domains.pop_back();
        plan.routed_tiers.pop_back();

        const auto result = validateMoERoutedExpertPlacementPlan(
            plan,
            MoERoutedExpertPlacementValidationOptions{.routed_expert_count = 4});

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "no fallback tier"));
        EXPECT_TRUE(hasErrorContaining(result, "covers only 3 of 4 routed experts"));
    }

    TEST(Test__MoEGraphNativeProductionHardening, PlacementValidatorAcceptsRoutedTensorSharding)
    {
        auto plan = productionPlan();
        plan.domains[1].routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;

        const auto result = validateMoERoutedExpertPlacementPlan(plan);

        EXPECT_TRUE(result.ok())
            << (result.errors.empty() ? "" : result.errors.front());
    }

    TEST(Test__MoEGraphNativeProductionHardening, ValidatorReportsMissingContinuationBaseAndSharedReferences)
    {
        auto plan = productionPlan();
        plan.continuation_domain = "missing_continuation";
        plan.base_model_domain = "missing_base";
        plan.shared_expert_domain = "missing_shared";
        plan.continuation_domain_spec.domain = "missing_continuation";

        const auto result = validateMoERoutedExpertPlacementPlan(plan);

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "continuation domain references unknown execution domain"));
        EXPECT_TRUE(hasErrorContaining(result, "base/non-expert model domain references unknown execution domain"));
        EXPECT_TRUE(hasErrorContaining(result, "shared expert domain references unknown execution domain"));
    }

    TEST(Test__MoEGraphNativeProductionHardening, ValidatorRejectsMoreThanOneFallbackTier)
    {
        auto plan = productionPlan();
        plan.routed_tiers[0].fallback = true;
        plan.routed_tiers[1].fallback = true;

        const auto result = validateMoERoutedExpertPlacementPlan(plan);

        EXPECT_FALSE(result.ok());
        EXPECT_TRUE(hasErrorContaining(result, "at most one fallback tier"));
    }

    TEST(Test__MoEGraphNativeProductionHardening, ParserRequiresRoutedDomainScopeAndComputePolicy)
    {
        const std::string missing_scope = parseArgsError({"llaminar2",
                                                          "--moe-routed-expert-placement", "tiered-overlay",
                                                          "--moe-routed-expert-continuation-domain", "cuda_hot",
                                                          "--moe-routed-expert-shared-domain", "cuda_hot",
                                                          "--moe-routed-expert-domain", "cuda_hot=0:cuda:0;backend=nccl;routed_compute=apportioned",
                                                          "--moe-routed-expert-tier", "hot@cuda_hot;priority=0;max-experts-per-layer=4"});

        EXPECT_NE(missing_scope.find("missing scope"), std::string::npos);

        const std::string missing_compute = parseArgsError({"llaminar2",
                                                            "--moe-routed-expert-placement", "tiered-overlay",
                                                            "--moe-routed-expert-continuation-domain", "cuda_hot",
                                                            "--moe-routed-expert-shared-domain", "cuda_hot",
                                                            "--moe-routed-expert-domain", "cuda_hot=0:cuda:0;scope=single;backend=nccl",
                                                            "--moe-routed-expert-tier", "hot@cuda_hot;priority=0;max-experts-per-layer=4"});

        EXPECT_NE(missing_compute.find("missing routed_compute"), std::string::npos);
    }

    TEST(Test__MoEGraphNativeProductionHardening, ExecutionPlanResolverReportsAmbiguousLocalTPRankHints)
    {
        auto plan = productionPlan();
        plan.domains[1] = localTPDomain("rocm_warm", CollectiveBackendType::RCCL, -1);

        try
        {
            (void)resolveMoEExpertOverlayExecutionPlan(
                std::make_shared<MoERoutedExpertPlacementPlan>(plan),
                MoEExpertOverlayExecutionPlanResolverOptions{.current_world_rank = 0, .world_size = 3});
            FAIL() << "Expected ambiguous LocalTP rank ownership to fail";
        }
        catch (const std::invalid_argument &error)
        {
            const std::string message = error.what();
            EXPECT_NE(message.find("rocm_warm"), std::string::npos);
            EXPECT_NE(message.find("ambiguous rank ownership"), std::string::npos);
            EXPECT_NE(message.find("set owner=<rank> or ranks=<rank-list>"), std::string::npos);
        }
    }

} // namespace llaminar2::test
