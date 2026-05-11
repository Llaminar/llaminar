#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {

        ExpertComputeDomain cudaSingleDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::NCCL;
            domain.participants = {GlobalDeviceAddress::cuda(0)};
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain rocmLocalTPDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::LocalTP;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
            return domain;
        }

        ExpertComputeDomain cpuNodeLocalTPDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::NodeLocalTP;
            domain.backend = CollectiveBackendType::UPI;
            domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
            domain.world_ranks = {0, 1};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
            return domain;
        }

        ExpertComputeDomain cpuSingleFallbackDomain(const std::string &name, int owner_rank)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::UPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.world_ranks = {owner_rank};
            domain.owner_rank = owner_rank;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain remoteCudaDomain(const std::string &name)
        {
            ExpertComputeDomain domain = cudaSingleDomain(name);
            domain.participants = {GlobalDeviceAddress::cuda(0, 0, "remote-node")};
            return domain;
        }

        ExpertRoutedTier tier(const std::string &name, const std::string &domain, int priority, bool fallback = false)
        {
            ExpertRoutedTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.fallback = fallback;
            return result;
        }

        std::shared_ptr<MoEExpertParallelPlan> layoutAPlan()
        {
            auto plan = std::make_shared<MoEExpertParallelPlan>();
            plan->enabled = true;
            plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan->continuation_domain = "rocm_hot";
            plan->shared_expert_domain = "rocm_hot";
            plan->residency_policy = ExpertResidencyPolicy::StaticById;
            plan->domains = {
                rocmLocalTPDomain("rocm_hot"),
                cpuNodeLocalTPDomain("cpu_cold"),
            };
            plan->routed_tiers = {
                tier("hot", "rocm_hot", 0),
                tier("cold", "cpu_cold", 1, true),
            };
            return plan;
        }

        std::shared_ptr<MoEExpertParallelPlan> layoutBPlan()
        {
            auto plan = std::make_shared<MoEExpertParallelPlan>();
            plan->enabled = true;
            plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan->continuation_domain = "cuda_fast";
            plan->shared_expert_domain = "cuda_fast";
            plan->residency_policy = ExpertResidencyPolicy::StaticById;
            plan->domains = {
                cudaSingleDomain("cuda_fast"),
                rocmLocalTPDomain("rocm_warm"),
                cpuNodeLocalTPDomain("cpu_cold"),
            };
            plan->routed_tiers = {
                tier("hottest", "cuda_fast", 0),
                tier("warm", "rocm_warm", 1),
                tier("cold", "cpu_cold", 2, true),
            };
            plan->placements = {
                ExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 2, 2}},
            };
            return plan;
        }

        std::shared_ptr<MoEExpertParallelPlan> layoutBThreeRankPlan()
        {
            auto plan = layoutBPlan();
            plan->domains[0].world_ranks = {0};
            plan->domains[0].owner_rank = 0;
            plan->domains[1].owner_rank = 1;
            plan->domains[2] = cpuSingleFallbackDomain("cpu_cold", 2);
            return plan;
        }

        std::string thrownMessageFor(
            std::shared_ptr<MoEExpertParallelPlan> plan,
            MoEExpertOverlayRuntimeResolverOptions options = {.current_world_rank = 0})
        {
            try
            {
                (void)resolveMoEExpertOverlayRuntimePlan(std::move(plan), options);
            }
            catch (const std::exception &e)
            {
                return e.what();
            }
            return {};
        }

        bool containsDomain(const OverlayRankPlan &rank_plan, const std::string &domain_name)
        {
            return rank_plan.ownsDomain(domain_name);
        }

        bool containsDevice(const OverlayRankPlan &rank_plan, DeviceId device)
        {
            return rank_plan.hasLocalDevice(device);
        }

    } // namespace

    TEST(Test__MoEExpertOverlayRuntimePlan, LayoutAResolvesRocmContinuationAndCpuFallback)
    {
        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(layoutAPlan());

        ASSERT_NE(runtime_plan, nullptr);
        EXPECT_EQ(runtime_plan->continuationDevice(), DeviceId::rocm(0));
        EXPECT_EQ(runtime_plan->sharedExpertDomain().primary_device, DeviceId::rocm(0));
        EXPECT_EQ(runtime_plan->sharedExpertDeviceForMVP(0), DeviceId::rocm(0));
        EXPECT_EQ(runtime_plan->tierDeviceForMVP(0), DeviceId::rocm(0));
        EXPECT_EQ(runtime_plan->tierDeviceForMVP(1), DeviceId::cpu());

        const auto *rocm_domain = runtime_plan->domainForName("rocm_hot");
        const auto *cpu_domain = runtime_plan->domainForName("cpu_cold");
        ASSERT_NE(rocm_domain, nullptr);
        ASSERT_NE(cpu_domain, nullptr);
        EXPECT_EQ(rocm_domain->backend, CollectiveBackendType::RCCL);
        EXPECT_EQ(cpu_domain->backend, CollectiveBackendType::UPI);
        EXPECT_EQ(rocm_domain->compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
        EXPECT_EQ(cpu_domain->compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
        ASSERT_EQ(cpu_domain->participants.size(), 2u);
        EXPECT_EQ(cpu_domain->participants[0].world_rank, 0);
        EXPECT_TRUE(cpu_domain->participants[0].owned_by_current_rank);
        EXPECT_EQ(cpu_domain->participants[1].world_rank, 1);
        EXPECT_FALSE(cpu_domain->participants[1].owned_by_current_rank);
        EXPECT_FALSE(rocm_domain->multi_participant_execution_pending);
        EXPECT_FALSE(cpu_domain->multi_participant_execution_pending);
        EXPECT_TRUE(rocm_domain->domain_scoped_collective_context_ready);
        EXPECT_TRUE(cpu_domain->domain_scoped_collective_context_ready);
        EXPECT_TRUE(rocm_domain->local_reachable_for_mvp);

        const std::string diagnostics = runtime_plan->diagnostics();
        EXPECT_NE(diagnostics.find("continuation_device=ROCm:0"), std::string::npos);
        EXPECT_EQ(diagnostics.find("multi_participant_execution_pending=true"), std::string::npos);
        EXPECT_NE(diagnostics.find("collective_context=ready"), std::string::npos);
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, LayoutBResolvesCudaContinuationWithRocmAndCpuTiers)
    {
        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(layoutBPlan());

        ASSERT_NE(runtime_plan, nullptr);
        EXPECT_EQ(runtime_plan->continuationDevice(), DeviceId::cuda(0));
        EXPECT_EQ(runtime_plan->sharedExpertDeviceForMVP(0), DeviceId::cuda(0));
        ASSERT_EQ(runtime_plan->routedTiers().size(), 3u);
        EXPECT_EQ(runtime_plan->tierDeviceForMVP(0), DeviceId::cuda(0));
        EXPECT_EQ(runtime_plan->tierDeviceForMVP(1), DeviceId::rocm(0));
        EXPECT_EQ(runtime_plan->tierDeviceForMVP(2), DeviceId::cpu());

        const auto &warm_domain = runtime_plan->domainForTier(1);
        const auto &cold_domain = runtime_plan->domainForTier(2);
        EXPECT_EQ(warm_domain.name, "rocm_warm");
        EXPECT_EQ(cold_domain.name, "cpu_cold");
        EXPECT_FALSE(warm_domain.multi_participant_execution_pending);
        EXPECT_FALSE(cold_domain.multi_participant_execution_pending);
        EXPECT_TRUE(warm_domain.domain_scoped_collective_context_ready);
        EXPECT_TRUE(cold_domain.domain_scoped_collective_context_ready);
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, InvalidRemoteContinuationFailsBeforeGraphExecution)
    {
        auto plan = layoutAPlan();
        plan->domains.push_back(remoteCudaDomain("remote_continuation"));
        plan->continuation_domain = "remote_continuation";
        plan->shared_expert_domain = "remote_continuation";

        const std::string message = thrownMessageFor(std::move(plan));

        ASSERT_FALSE(message.empty());
        EXPECT_NE(message.find("continuation domain"), std::string::npos) << message;
        EXPECT_NE(message.find("not locally reachable"), std::string::npos) << message;
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, InvalidSharedDomainOwnershipFailsBeforeGraphExecution)
    {
        auto plan = layoutBPlan();
        auto shared_domain = cudaSingleDomain("shared_remote_rank");
        shared_domain.owner_rank = 1;
        plan->domains.push_back(shared_domain);
        plan->shared_expert_domain = "shared_remote_rank";

        const std::string message = thrownMessageFor(std::move(plan));

        ASSERT_FALSE(message.empty());
        EXPECT_NE(message.find("shared expert domain"), std::string::npos);
        EXPECT_NE(message.find("not locally reachable"), std::string::npos);
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, TieredOverlayDescriptorsRemainSameLayerRoles)
    {
        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(layoutBPlan());

        ASSERT_NE(runtime_plan, nullptr);
        EXPECT_EQ(runtime_plan->sourcePlan().execution_kind, MoEExpertExecutionKind::TieredExpertOverlay);
        ASSERT_EQ(runtime_plan->sourcePlan().placements.size(), 1u);
        EXPECT_EQ(runtime_plan->sourcePlan().placements[0].layer, 0);
        EXPECT_EQ(runtime_plan->routedTiers()[0].domain_name, "cuda_fast");
        EXPECT_EQ(runtime_plan->routedTiers()[1].domain_name, "rocm_warm");
        EXPECT_EQ(runtime_plan->routedTiers()[2].domain_name, "cpu_cold");
        EXPECT_EQ(runtime_plan->continuationDevice(), DeviceId::cuda(0));
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, LayoutARank0PlansContinuationRootAndRocmLocalTPOwner)
    {
        const auto execution_plan = resolveMoEExpertOverlayExecutionPlan(layoutAPlan(), 0);
        const auto &rank = execution_plan.currentRankPlan();

        EXPECT_EQ(rank.world_rank, 0);
        EXPECT_EQ(rank.role, OverlayRankRole::ContinuationRoot);
        EXPECT_TRUE(rank.hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_TRUE(rank.hasRole(OverlayRankRole::LocalAcceleratorParticipant));
        EXPECT_FALSE(rank.hasRole(OverlayRankRole::RelayOnly));
        EXPECT_TRUE(rank.builds_root_graph);
        EXPECT_TRUE(containsDomain(rank, "rocm_hot"));
        EXPECT_TRUE(containsDevice(rank, DeviceId::rocm(0)));
        EXPECT_TRUE(containsDevice(rank, DeviceId::rocm(1)));
        EXPECT_TRUE(rank.loads_tokenizer);
        EXPECT_TRUE(rank.loads_full_model_metadata);
        EXPECT_TRUE(rank.loads_root_weights);
        EXPECT_TRUE(rank.loads_expert_weights);
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, LayoutARank1PlansCpuFallbackOnlyWithoutRootGraph)
    {
        const auto execution_plan = resolveMoEExpertOverlayExecutionPlan(layoutAPlan(), 1);
        const auto &rank = execution_plan.currentRankPlan();

        EXPECT_EQ(rank.world_rank, 1);
        EXPECT_EQ(rank.role, OverlayRankRole::CpuFallbackParticipant);
        EXPECT_TRUE(rank.hasRole(OverlayRankRole::CpuFallbackParticipant));
        EXPECT_FALSE(rank.hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_FALSE(rank.hasRole(OverlayRankRole::LocalAcceleratorParticipant));
        EXPECT_FALSE(rank.hasRole(OverlayRankRole::RelayOnly));
        EXPECT_FALSE(rank.builds_root_graph);
        EXPECT_EQ(rank.owned_domains, (std::vector<std::string>{"cpu_cold"}));
        EXPECT_TRUE(containsDevice(rank, DeviceId::cpu()));
        EXPECT_FALSE(rank.loads_tokenizer);
        EXPECT_TRUE(rank.loads_full_model_metadata);
        EXPECT_FALSE(rank.loads_root_weights);
        EXPECT_TRUE(rank.loads_expert_weights);
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, LayoutBThreeRankPlanDistinguishesCudaRocmAndCpuRoles)
    {
        const auto root_execution = resolveMoEExpertOverlayExecutionPlan(layoutBThreeRankPlan(), 0);
        const auto rocm_execution = resolveMoEExpertOverlayExecutionPlan(layoutBThreeRankPlan(), 1);
        const auto cpu_execution = resolveMoEExpertOverlayExecutionPlan(layoutBThreeRankPlan(), 2);

        const auto &root = root_execution.currentRankPlan();
        EXPECT_EQ(root.role, OverlayRankRole::ContinuationRoot);
        EXPECT_TRUE(root.builds_root_graph);
        EXPECT_TRUE(containsDomain(root, "cuda_fast"));
        EXPECT_TRUE(containsDevice(root, DeviceId::cuda(0)));

        const auto &rocm = rocm_execution.currentRankPlan();
        EXPECT_EQ(rocm.role, OverlayRankRole::LocalAcceleratorParticipant);
        EXPECT_FALSE(rocm.builds_root_graph);
        EXPECT_TRUE(rocm.hasRole(OverlayRankRole::LocalAcceleratorParticipant));
        EXPECT_FALSE(rocm.hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_TRUE(containsDomain(rocm, "rocm_warm"));
        EXPECT_TRUE(containsDevice(rocm, DeviceId::rocm(0)));
        EXPECT_TRUE(containsDevice(rocm, DeviceId::rocm(1)));

        const auto &cpu = cpu_execution.currentRankPlan();
        EXPECT_EQ(cpu.role, OverlayRankRole::CpuFallbackParticipant);
        EXPECT_FALSE(cpu.builds_root_graph);
        EXPECT_TRUE(cpu.hasRole(OverlayRankRole::CpuFallbackParticipant));
        EXPECT_FALSE(cpu.hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_TRUE(containsDomain(cpu, "cpu_cold"));
        EXPECT_TRUE(containsDevice(cpu, DeviceId::cpu()));
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, LayoutARank1RegressionIsOrchestrationGapNotGraphMathGap)
    {
        // Regression guard for the production limitation documented in
        // docs/v2/MOE_EXPERT_OVERLAY_ORCHESTRATION_REFACTOR_PLAN.md: rank 1 is an
        // auxiliary CPU fallback participant. The failure is orchestration trying
        // to build a continuation-root runner on rank 1, not overlay graph math.
        const std::string strict_message = thrownMessageFor(
            layoutAPlan(),
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = 1,
                .validate_mvp_root_reachability = true,
            });
        EXPECT_NE(strict_message.find("continuation domain"), std::string::npos) << strict_message;
        EXPECT_NE(strict_message.find("not locally reachable"), std::string::npos) << strict_message;

        const auto execution_plan = resolveMoEExpertOverlayExecutionPlan(layoutAPlan(), 1);
        const auto &rank = execution_plan.currentRankPlan();
        EXPECT_EQ(rank.role, OverlayRankRole::CpuFallbackParticipant);
        EXPECT_FALSE(rank.builds_root_graph);
        EXPECT_FALSE(rank.hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_TRUE(rank.hasRole(OverlayRankRole::CpuFallbackParticipant));
        EXPECT_NE(execution_plan.diagnostics().find("builds_root_graph=false"), std::string::npos);
    }

} // namespace llaminar2::test
