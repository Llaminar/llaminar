#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"

#include <gtest/gtest.h>

#include <algorithm>
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

        ExpertComputeDomain remoteCudaWorkerDomain(const std::string &name, int owner_rank)
        {
            ExpertComputeDomain domain = remoteCudaDomain(name);
            domain.owner_rank = owner_rank;
            domain.world_ranks = {owner_rank};
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

        std::shared_ptr<MoEExpertParallelPlan> remoteExpertWorkerPlan()
        {
            auto plan = std::make_shared<MoEExpertParallelPlan>();
            plan->enabled = true;
            plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan->continuation_domain = "cuda_fast";
            plan->shared_expert_domain = "cuda_fast";
            plan->residency_policy = ExpertResidencyPolicy::StaticById;
            plan->domains = {
                cudaSingleDomain("cuda_fast"),
                remoteCudaWorkerDomain("remote_experts", 1),
                cpuSingleFallbackDomain("cpu_cold", 2),
            };
            plan->domains[0].owner_rank = 0;
            plan->domains[0].world_ranks = {0};
            plan->routed_tiers = {
                tier("fast", "cuda_fast", 0),
                tier("remote", "remote_experts", 1),
                tier("cold", "cpu_cold", 2, true),
            };
            return plan;
        }

        std::shared_ptr<MoEExpertParallelPlan> continuationOnlyDensePlan()
        {
            auto plan = std::make_shared<MoEExpertParallelPlan>();
            plan->enabled = true;
            plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan->continuation_domain = "dense_cont";
            plan->shared_expert_domain = "dense_cont";
            plan->residency_policy = ExpertResidencyPolicy::StaticById;
            plan->domains = {
                cudaSingleDomain("dense_cont"),
                cpuSingleFallbackDomain("cpu_routed", 0),
            };
            plan->routed_tiers = {
                tier("cold", "cpu_routed", 0, true),
            };
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

        std::string executionPlanThrownMessageFor(
            std::shared_ptr<MoEExpertParallelPlan> plan,
            MoEExpertOverlayExecutionPlanResolverOptions options)
        {
            try
            {
                (void)resolveMoEExpertOverlayExecutionPlan(std::move(plan), options);
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
        EXPECT_TRUE(rocm_domain->routed_rebalance_controller_eligible);
        EXPECT_EQ(rocm_domain->rebalance_domain_id, "overlay_routed_rocm_hot");
        EXPECT_EQ(rocm_domain->routed_tier_count, 1);
        EXPECT_TRUE(cpu_domain->routed_rebalance_controller_eligible);
        EXPECT_EQ(cpu_domain->rebalance_domain_id, "overlay_routed_cpu_cold");
        EXPECT_EQ(cpu_domain->routed_tier_count, 1);
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
        EXPECT_NE(diagnostics.find("routed_rebalance=overlay_routed_cpu_cold"), std::string::npos);
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, LocalTPReplicatedExpertsDomainIsGraphNativeReady)
    {
        auto plan = layoutAPlan();
        plan->domains[0].compute_kind = ExpertDomainComputeKind::ReplicatedExperts;

        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);

        ASSERT_NE(runtime_plan, nullptr);
        const auto *rocm_domain = runtime_plan->domainForName("rocm_hot");
        ASSERT_NE(rocm_domain, nullptr);
        EXPECT_EQ(rocm_domain->kind, ExpertDomainKind::LocalTP);
        EXPECT_EQ(rocm_domain->compute_kind, ExpertDomainComputeKind::ReplicatedExperts);
        EXPECT_FALSE(rocm_domain->multi_participant_execution_pending);
        EXPECT_TRUE(rocm_domain->domain_scoped_collective_context_ready);
        EXPECT_TRUE(rocm_domain->pending_reason.empty());

        const std::string diagnostics = runtime_plan->diagnostics();
        EXPECT_NE(diagnostics.find("rocm_hot"), std::string::npos) << diagnostics;
        EXPECT_EQ(diagnostics.find("multi_participant_execution_pending=true"),
                  std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("collective_context=ready"), std::string::npos)
            << diagnostics;
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, ContinuationOnlyDomainIsNotRoutedRebalanceEligible)
    {
        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(continuationOnlyDensePlan());

        const auto *dense_domain = runtime_plan->domainForName("dense_cont");
        const auto *routed_domain = runtime_plan->domainForName("cpu_routed");
        ASSERT_NE(dense_domain, nullptr);
        ASSERT_NE(routed_domain, nullptr);

        EXPECT_FALSE(dense_domain->routed_rebalance_controller_eligible);
        EXPECT_TRUE(dense_domain->rebalance_domain_id.empty());
        EXPECT_EQ(dense_domain->routed_tier_count, 0);

        EXPECT_TRUE(routed_domain->routed_rebalance_controller_eligible);
        EXPECT_EQ(routed_domain->rebalance_domain_id, "overlay_routed_cpu_routed");
        EXPECT_EQ(routed_domain->routed_tier_count, 1);
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
        EXPECT_FALSE(rank.loads_worker_tokenizer_state);
        EXPECT_TRUE(rank.loads_full_model_metadata);
        EXPECT_TRUE(rank.loads_root_weights);
        EXPECT_TRUE(rank.loads_shared_expert_weights);
        EXPECT_TRUE(rank.loads_accelerator_routed_experts);
        EXPECT_TRUE(rank.loads_cpu_fallback_experts);
        EXPECT_FALSE(rank.loads_worker_fallback_experts);
        EXPECT_TRUE(rank.loads_expert_weights);
        EXPECT_NE(std::find(rank.root_weight_domains.begin(), rank.root_weight_domains.end(), "rocm_hot"),
              rank.root_weight_domains.end());
        EXPECT_NE(std::find(rank.shared_expert_weight_domains.begin(), rank.shared_expert_weight_domains.end(), "rocm_hot"),
              rank.shared_expert_weight_domains.end());
        EXPECT_NE(std::find(rank.accelerator_routed_expert_domains.begin(), rank.accelerator_routed_expert_domains.end(), "rocm_hot"),
              rank.accelerator_routed_expert_domains.end());
        EXPECT_NE(std::find(rank.cpu_fallback_expert_domains.begin(), rank.cpu_fallback_expert_domains.end(), "cpu_cold"),
              rank.cpu_fallback_expert_domains.end());
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, LayoutAFullPlanRendersRootAndCpuFallbackRanks)
    {
        const auto execution_plan = resolveMoEExpertOverlayExecutionPlan(
            layoutAPlan(),
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 0,
                .world_size = 2,
            });

        ASSERT_EQ(execution_plan.rank_plans.size(), 2u);
        EXPECT_EQ(execution_plan.continuation_root_rank, 0);
        const auto *rank0 = execution_plan.rankPlanFor(0);
        const auto *rank1 = execution_plan.rankPlanFor(1);
        ASSERT_NE(rank0, nullptr);
        ASSERT_NE(rank1, nullptr);

        EXPECT_TRUE(rank0->hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_TRUE(rank0->hasRole(OverlayRankRole::LocalAcceleratorParticipant));
        EXPECT_TRUE(rank0->hasRole(OverlayRankRole::CpuFallbackParticipant));
        EXPECT_TRUE(rank0->builds_root_graph);
        EXPECT_TRUE(containsDomain(*rank0, "rocm_hot"));
        EXPECT_TRUE(containsDomain(*rank0, "cpu_cold"));

        EXPECT_EQ(rank1->role, OverlayRankRole::CpuFallbackParticipant);
        EXPECT_TRUE(rank1->hasRole(OverlayRankRole::CpuFallbackParticipant));
        EXPECT_FALSE(rank1->builds_root_graph);
        EXPECT_TRUE(containsDomain(*rank1, "cpu_cold"));

        const std::string diagnostics = execution_plan.diagnostics();
        EXPECT_NE(diagnostics.find("rank[0]"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("rank[1]"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("ContinuationRoot"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("CpuFallbackParticipant"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("rocm_hot"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("cpu_cold"), std::string::npos) << diagnostics;
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
        EXPECT_TRUE(rank.loads_worker_tokenizer_state);
        EXPECT_TRUE(rank.loads_full_model_metadata);
        EXPECT_FALSE(rank.loads_root_weights);
        EXPECT_FALSE(rank.loads_shared_expert_weights);
        EXPECT_FALSE(rank.loads_accelerator_routed_experts);
        EXPECT_TRUE(rank.loads_cpu_fallback_experts);
        EXPECT_TRUE(rank.loads_worker_fallback_experts);
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
        EXPECT_TRUE(root.loads_root_weights);
        EXPECT_TRUE(root.loads_shared_expert_weights);
        EXPECT_TRUE(root.loads_accelerator_routed_experts);
        EXPECT_FALSE(root.loads_cpu_fallback_experts);
        EXPECT_FALSE(root.loads_worker_fallback_experts);

        const auto &rocm = rocm_execution.currentRankPlan();
        EXPECT_EQ(rocm.role, OverlayRankRole::LocalAcceleratorParticipant);
        EXPECT_FALSE(rocm.builds_root_graph);
        EXPECT_TRUE(rocm.hasRole(OverlayRankRole::LocalAcceleratorParticipant));
        EXPECT_FALSE(rocm.hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_TRUE(containsDomain(rocm, "rocm_warm"));
        EXPECT_TRUE(containsDevice(rocm, DeviceId::rocm(0)));
        EXPECT_TRUE(containsDevice(rocm, DeviceId::rocm(1)));
        EXPECT_FALSE(rocm.loads_root_weights);
        EXPECT_FALSE(rocm.loads_shared_expert_weights);
        EXPECT_TRUE(rocm.loads_accelerator_routed_experts);
        EXPECT_FALSE(rocm.loads_cpu_fallback_experts);
        EXPECT_FALSE(rocm.loads_worker_fallback_experts);

        const auto &cpu = cpu_execution.currentRankPlan();
        EXPECT_EQ(cpu.role, OverlayRankRole::CpuFallbackParticipant);
        EXPECT_FALSE(cpu.builds_root_graph);
        EXPECT_TRUE(cpu.hasRole(OverlayRankRole::CpuFallbackParticipant));
        EXPECT_FALSE(cpu.hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_TRUE(containsDomain(cpu, "cpu_cold"));
        EXPECT_TRUE(containsDevice(cpu, DeviceId::cpu()));
        EXPECT_FALSE(cpu.loads_root_weights);
        EXPECT_FALSE(cpu.loads_shared_expert_weights);
        EXPECT_FALSE(cpu.loads_accelerator_routed_experts);
        EXPECT_TRUE(cpu.loads_cpu_fallback_experts);
        EXPECT_TRUE(cpu.loads_worker_fallback_experts);
        EXPECT_TRUE(cpu.loads_worker_tokenizer_state);
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, LayoutBFullPlanIncludesRelayOnlyRank)
    {
        const auto execution_plan = resolveMoEExpertOverlayExecutionPlan(
            layoutBThreeRankPlan(),
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 0,
                .world_size = 4,
            });

        ASSERT_EQ(execution_plan.rank_plans.size(), 4u);
        ASSERT_NE(execution_plan.rankPlanFor(0), nullptr);
        ASSERT_NE(execution_plan.rankPlanFor(1), nullptr);
        ASSERT_NE(execution_plan.rankPlanFor(2), nullptr);
        ASSERT_NE(execution_plan.rankPlanFor(3), nullptr);

        EXPECT_EQ(execution_plan.rankPlanFor(0)->role, OverlayRankRole::ContinuationRoot);
        EXPECT_EQ(execution_plan.rankPlanFor(1)->role, OverlayRankRole::LocalAcceleratorParticipant);
        EXPECT_EQ(execution_plan.rankPlanFor(2)->role, OverlayRankRole::CpuFallbackParticipant);
        EXPECT_EQ(execution_plan.rankPlanFor(3)->role, OverlayRankRole::RelayOnly);
        EXPECT_FALSE(execution_plan.rankPlanFor(3)->builds_root_graph);
        EXPECT_FALSE(execution_plan.rankPlanFor(3)->loads_full_model_metadata);

        const std::string diagnostics = execution_plan.diagnostics();
        EXPECT_NE(diagnostics.find("rank[3]: primary_role=RelayOnly"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("rocm_warm"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("cpu_cold"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("root_weight_domains"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("shared_expert_domains"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("accelerator_routed_domains"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("worker_fallback_domains"), std::string::npos) << diagnostics;
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, RemoteExpertWorkerRankIsDistinctFromLocalAccelerator)
    {
        const auto execution_plan = resolveMoEExpertOverlayExecutionPlan(
            remoteExpertWorkerPlan(),
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 1,
                .world_size = 3,
            });
        const auto &rank = execution_plan.currentRankPlan();

        EXPECT_EQ(rank.world_rank, 1);
        EXPECT_EQ(rank.role, OverlayRankRole::RemoteExpertParticipant);
        EXPECT_TRUE(rank.hasRole(OverlayRankRole::RemoteExpertParticipant));
        EXPECT_FALSE(rank.hasRole(OverlayRankRole::LocalAcceleratorParticipant));
        EXPECT_FALSE(rank.hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_FALSE(rank.builds_root_graph);
        EXPECT_TRUE(containsDomain(rank, "remote_experts"));
        EXPECT_TRUE(containsDevice(rank, DeviceId::cuda(0)));
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, InvalidExecutionTopologyReportsRankAndDomain)
    {
        const std::string too_small_world = executionPlanThrownMessageFor(
            layoutAPlan(),
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 0,
                .world_size = 1,
            });
        ASSERT_FALSE(too_small_world.empty());
        EXPECT_NE(too_small_world.find("cpu_cold"), std::string::npos) << too_small_world;
        EXPECT_NE(too_small_world.find("rank 1"), std::string::npos) << too_small_world;
        EXPECT_NE(too_small_world.find("world size 1"), std::string::npos) << too_small_world;

        const std::string bad_current_rank = executionPlanThrownMessageFor(
            layoutAPlan(),
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 3,
                .world_size = 2,
            });
        ASSERT_FALSE(bad_current_rank.empty());
        EXPECT_NE(bad_current_rank.find("current rank 3"), std::string::npos) << bad_current_rank;
        EXPECT_NE(bad_current_rank.find("world size 2"), std::string::npos) << bad_current_rank;
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, OnlyContinuationOwnerBuildsRootGraph)
    {
        const auto execution_plan = resolveMoEExpertOverlayExecutionPlan(
            layoutBThreeRankPlan(),
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 0,
                .world_size = 4,
            });

        for (const auto &rank : execution_plan.rank_plans)
        {
            EXPECT_EQ(rank.builds_root_graph,
                      rank.world_rank == execution_plan.continuation_root_rank)
                << execution_plan.diagnostics();
            if (rank.world_rank != execution_plan.continuation_root_rank)
            {
                EXPECT_FALSE(rank.hasRole(OverlayRankRole::ContinuationRoot))
                    << execution_plan.diagnostics();
            }
        }
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
