/**
 * @file Test__MoEExpertOverlayRuntimePlan.cpp
 * @brief Unit coverage for expert-overlay rank, device, and topology resolution.
 *
 * These tests keep the configuration-to-runtime ownership boundary device-free.
 * In particular, a concrete inventory hostname is an address label, not proof
 * that a rank-local participant is remote; the resolved MPI rank is the
 * authority used by production graph construction.
 */

#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEExpertOverlayAuthorityPlan.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "config/ConfigValidator.h"
#include "config/OrchestrationConfig.h"

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

        RoutedExpertDomain cudaSingleDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = CollectiveBackendType::NCCL;
            domain.participants = {GlobalDeviceAddress::cuda(0)};
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain rocmLocalTPDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::RANK_LOCAL;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
            domain.owner_rank = 0;
            domain.routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
            return domain;
        }

        RoutedExpertDomain cpuNodeTPDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::NODE_LOCAL;
            domain.backend = CollectiveBackendType::UPI;
            domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
            domain.world_ranks = {0, 1};
            domain.owner_rank = 0;
            domain.routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
            return domain;
        }

        RoutedExpertDomain cpuSingleFallbackDomain(const std::string &name, int owner_rank)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = CollectiveBackendType::UPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.world_ranks = {owner_rank};
            domain.owner_rank = owner_rank;
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain remoteCudaDomain(const std::string &name)
        {
            RoutedExpertDomain domain = cudaSingleDomain(name);
            domain.participants = {GlobalDeviceAddress::cuda(0, 0, "remote-node")};
            return domain;
        }

        RoutedExpertDomain remoteCudaWorkerDomain(const std::string &name, int owner_rank)
        {
            RoutedExpertDomain domain = remoteCudaDomain(name);
            domain.owner_rank = owner_rank;
            domain.world_ranks = {owner_rank};
            return domain;
        }

        RoutedExpertTier tier(const std::string &name, const std::string &domain, int priority, bool fallback = false)
        {
            RoutedExpertTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.fallback = fallback;
            return result;
        }

        std::shared_ptr<MoERoutedExpertPlacementPlan> layoutAPlan()
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "rocm_hot";
            plan->shared_expert_domain = "rocm_hot";
            plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan->domains = {
                rocmLocalTPDomain("rocm_hot"),
                cpuNodeTPDomain("cpu_cold"),
            };
            plan->routed_tiers = {
                tier("hot", "rocm_hot", 0),
                tier("cold", "cpu_cold", 1, true),
            };
            return plan;
        }

        std::shared_ptr<MoERoutedExpertPlacementPlan> layoutBPlan()
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "cuda_fast";
            plan->shared_expert_domain = "cuda_fast";
            plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan->domains = {
                cudaSingleDomain("cuda_fast"),
                rocmLocalTPDomain("rocm_warm"),
                cpuNodeTPDomain("cpu_cold"),
            };
            plan->routed_tiers = {
                tier("hottest", "cuda_fast", 0),
                tier("warm", "rocm_warm", 1),
                tier("cold", "cpu_cold", 2, true),
            };
            plan->placements = {
                RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 2, 2}},
            };
            return plan;
        }

        std::shared_ptr<MoERoutedExpertPlacementPlan> layoutBThreeRankPlan()
        {
            auto plan = layoutBPlan();
            plan->domains[0].world_ranks = {0};
            plan->domains[0].owner_rank = 0;
            plan->domains[1].owner_rank = 1;
            plan->domains[2] = cpuSingleFallbackDomain("cpu_cold", 2);
            return plan;
        }

        std::shared_ptr<MoERoutedExpertPlacementPlan> remoteExpertWorkerPlan()
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "cuda_fast";
            plan->shared_expert_domain = "cuda_fast";
            plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
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

        std::shared_ptr<MoERoutedExpertPlacementPlan> continuationOnlyDensePlan()
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "dense_cont";
            plan->shared_expert_domain = "dense_cont";
            plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan->domains = {
                cudaSingleDomain("dense_cont"),
                cpuSingleFallbackDomain("cpu_routed", 0),
            };
            plan->routed_tiers = {
                tier("cold", "cpu_routed", 0, true),
            };
            return plan;
        }

        /** @brief Build a one-tier CPU plan whose dense graph is NodeTP. */
        std::shared_ptr<MoERoutedExpertPlacementPlan>
        cpuNodeLocalContinuationPlan()
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::SingleDomain;
            plan->continuation_domain = "cpu_overlay";
            plan->base_model_domain = "cpu_overlay";
            plan->shared_expert_domain = "cpu_overlay";
            plan->continuation_domain_spec.domain = "cpu_overlay";
            plan->continuation_domain_spec.logical_root_participant = 0;
            plan->continuation_domain_spec.setDensePolicy(
                DenseParallelPolicy::TensorParallel);
            plan->residency_policy =
                RoutedExpertResidencyPolicy::StaticById;
            auto domain = cpuNodeTPDomain("cpu_overlay");
            domain.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            plan->domains = {domain};
            plan->routed_tiers = {
                tier("priority_0", "cpu_overlay", 0, true),
            };
            return plan;
        }

        std::string thrownMessageFor(
            std::shared_ptr<MoERoutedExpertPlacementPlan> plan,
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
            std::shared_ptr<MoERoutedExpertPlacementPlan> plan,
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

        RoutedExpertDomain rankAgnosticGpuDomain(
            const std::string &name,
            GlobalDeviceAddress participant,
            CollectiveBackendType backend)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = backend;
            domain.participants = {std::move(participant)};
            domain.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        std::shared_ptr<MoERoutedExpertPlacementPlan>
        rankAgnosticThreeTierPlan()
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "cuda_hot";
            plan->shared_expert_domain = "cuda_hot";
            plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan->domains = {
                rankAgnosticGpuDomain(
                    "cuda_hot",
                    GlobalDeviceAddress::cuda(0),
                    CollectiveBackendType::NCCL),
                rankAgnosticGpuDomain(
                    "rocm_warm",
                    GlobalDeviceAddress::rocm(0),
                    CollectiveBackendType::RCCL),
                cpuNodeTPDomain("cpu_cold"),
            };
            plan->domains.back().routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            plan->domains.back().world_ranks.clear();
            plan->domains.back().owner_rank = -1;
            plan->routed_tiers = {
                tier("hot", "cuda_hot", 0),
                tier("warm", "rocm_warm", 1),
                tier("cold", "cpu_cold", 2, true),
            };
            plan->placements = {
                RoutedExpertLayerPlacement{
                    .layer = 0,
                    .routed_expert_tier = {0, 1, 2, 2}},
            };
            return plan;
        }

        DeviceInfo syntheticGpu(
            DeviceType type,
            int ordinal,
            int numa_node)
        {
            DeviceInfo result;
            result.type = type;
            result.local_device_id = ordinal;
            result.numa_node = numa_node;
            result.memory_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
            result.free_memory_bytes = result.memory_bytes;
            result.compute_units = type == DeviceType::CUDA ? 82 : 60;
            return result;
        }

        ClusterInventory syntheticTwoSocketInventory(
            int cuda_rank,
            int rocm_rank)
        {
            ClusterInventory inventory;
            inventory.world_size = 2;
            inventory.node_count = 1;
            for (int rank = 0; rank < 2; ++rank)
            {
                RankInventory rank_inventory;
                rank_inventory.rank = rank;
                rank_inventory.node_id = 0;
                rank_inventory.local_rank = rank;
                rank_inventory.hostname = "test-node";
                rank_inventory.cpu.type = DeviceType::CPU;
                rank_inventory.cpu.local_device_id = 0;
                if (rank == cuda_rank)
                {
                    rank_inventory.gpus.push_back(
                        syntheticGpu(DeviceType::CUDA, 0, cuda_rank));
                }
                if (rank == rocm_rank)
                {
                    rank_inventory.gpus.push_back(
                        syntheticGpu(DeviceType::ROCm, 0, rocm_rank));
                }
                inventory.total_gpus +=
                    static_cast<int>(rank_inventory.gpus.size());
                inventory.ranks.push_back(std::move(rank_inventory));
            }
            inventory.buildNodeAggregations();
            return inventory;
        }

        std::shared_ptr<MoERoutedExpertPlacementPlan>
        rankAgnosticLocalCudaTPPlan()
        {
            RoutedExpertDomain cuda_hot;
            cuda_hot.name = "cuda_hot";
            cuda_hot.scope = ExecutionDomainScope::RANK_LOCAL;
            cuda_hot.backend = CollectiveBackendType::NCCL;
            cuda_hot.participants = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::cuda(1),
            };
            cuda_hot.routed_compute_policy =
                RoutedExpertComputePolicy::TensorSharded;

            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = cuda_hot.name;
            plan->shared_expert_domain = cuda_hot.name;
            plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
            plan->domains = {cuda_hot};
            plan->dense_domains = {cuda_hot.toExecutionDomainDefinition()};
            plan->routed_tiers = {
                tier("hot", cuda_hot.name, 0, true),
            };
            return plan;
        }

        ClusterInventory syntheticSingleRankTwoCudaInventory()
        {
            ClusterInventory inventory;
            inventory.world_size = 1;

            RankInventory rank;
            rank.rank = 0;
            rank.node_id = 0;
            rank.local_rank = 0;
            rank.hostname = "test-node";
            rank.cpu.type = DeviceType::CPU;
            rank.cpu.local_device_id = 0;
            rank.gpus = {
                syntheticGpu(DeviceType::CUDA, 0, 0),
                syntheticGpu(DeviceType::CUDA, 1, 0),
            };
            inventory.ranks.push_back(std::move(rank));
            inventory.buildNodeAggregations();
            return inventory;
        }

        /**
         * @brief Build a node inventory with several devices per MPI rank.
         *
         * CUDA remains rank-local while the ROCm pool spans ranks and owns two
         * devices on each. This is the adversarial shape that distinguishes a
         * participant-rank map from a deduplicated communicator membership.
         */
        ClusterInventory syntheticTwoRankHeterogeneousGpuInventory()
        {
            ClusterInventory inventory;
            inventory.world_size = 2;
            inventory.node_count = 1;
            for (int world_rank = 0; world_rank < 2; ++world_rank)
            {
                RankInventory rank;
                rank.rank = world_rank;
                rank.node_id = 0;
                rank.local_rank = world_rank;
                rank.hostname = "test-node";
                rank.cpu.type = DeviceType::CPU;
                rank.cpu.local_device_id = 0;
                if (world_rank == 0)
                {
                    rank.gpus = {
                        syntheticGpu(DeviceType::CUDA, 0, 0),
                        syntheticGpu(DeviceType::CUDA, 1, 0),
                        syntheticGpu(DeviceType::ROCm, 0, 0),
                        syntheticGpu(DeviceType::ROCm, 1, 0),
                    };
                }
                else
                {
                    rank.gpus = {
                        syntheticGpu(DeviceType::ROCm, 2, 1),
                        syntheticGpu(DeviceType::ROCm, 3, 1),
                    };
                }
                inventory.total_gpus += static_cast<int>(rank.gpus.size());
                inventory.ranks.push_back(std::move(rank));
            }
            inventory.buildNodeAggregations();
            return inventory;
        }

        /** @brief Build portable CUDA/ROCm tier intent with discovered scope. */
        std::shared_ptr<MoERoutedExpertPlacementPlan>
        rankAgnosticAutoScopeGpuTierPlan()
        {
            RoutedExpertDomain cuda;
            cuda.name = "cuda_priority_0";
            cuda.scope = ExecutionDomainScope::AUTO;
            cuda.backend = CollectiveBackendType::NCCL;
            cuda.participants = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::cuda(1),
            };
            cuda.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;

            RoutedExpertDomain rocm;
            rocm.name = "rocm_priority_1";
            rocm.scope = ExecutionDomainScope::AUTO;
            rocm.backend = CollectiveBackendType::RCCL;
            rocm.participants = {
                GlobalDeviceAddress::rocm(0),
                GlobalDeviceAddress::rocm(1),
                GlobalDeviceAddress::rocm(2),
                GlobalDeviceAddress::rocm(3),
            };
            rocm.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;

            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology =
                RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = cuda.name;
            plan->base_model_domain = cuda.name;
            plan->shared_expert_domain = cuda.name;
            plan->continuation_domain_spec.domain = cuda.name;
            plan->continuation_domain_spec.setDensePolicy(
                DenseParallelPolicy::TensorParallel);
            plan->residency_policy =
                RoutedExpertResidencyPolicy::StaticById;
            plan->domains = {cuda, rocm};
            plan->dense_domains = {
                cuda.toExecutionDomainDefinition(),
            };
            plan->routed_tiers = {
                tier("priority_0", cuda.name, 0),
                tier("priority_1", rocm.name, 1, true),
            };
            return plan;
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
        EXPECT_EQ(rocm_domain->routed_compute_policy, RoutedExpertComputePolicy::TensorSharded);
        EXPECT_EQ(cpu_domain->routed_compute_policy, RoutedExpertComputePolicy::TensorSharded);
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

    TEST(Test__MoEExpertOverlayRuntimePlan, LocalTPApportionedRoutedComputeIsGraphNativeReady)
    {
        auto plan = layoutAPlan();
        plan->domains[0].routed_compute_policy = RoutedExpertComputePolicy::Apportioned;

        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);

        ASSERT_NE(runtime_plan, nullptr);
        const auto *rocm_domain = runtime_plan->domainForName("rocm_hot");
        ASSERT_NE(rocm_domain, nullptr);
        EXPECT_EQ(rocm_domain->scope, ExecutionDomainScope::RANK_LOCAL);
        EXPECT_EQ(rocm_domain->routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
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

    TEST(Test__MoEExpertOverlayRuntimePlan, LocalTPReplicatedRoutedComputeIsGraphNativeReady)
    {
        auto plan = layoutAPlan();
        plan->domains[0].routed_compute_policy =
            RoutedExpertComputePolicy::Replicated;

        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(plan);

        ASSERT_NE(runtime_plan, nullptr);
        const auto *rocm_domain = runtime_plan->domainForName("rocm_hot");
        ASSERT_NE(rocm_domain, nullptr);
        EXPECT_EQ(rocm_domain->scope, ExecutionDomainScope::RANK_LOCAL);
        EXPECT_EQ(
            rocm_domain->routed_compute_policy,
            RoutedExpertComputePolicy::Replicated);
        EXPECT_FALSE(rocm_domain->multi_participant_execution_pending);
        EXPECT_TRUE(rocm_domain->domain_scoped_collective_context_ready);
        EXPECT_TRUE(rocm_domain->pending_reason.empty());

        const std::string diagnostics = runtime_plan->diagnostics();
        EXPECT_EQ(
            diagnostics.find("multi_participant_execution_pending=true"),
            std::string::npos)
            << diagnostics;
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
        EXPECT_EQ(runtime_plan->sourcePlan().topology, RoutedExpertPlacementTopology::TieredOverlay);
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
        EXPECT_TRUE(rank.ownsContinuationGraph());
        EXPECT_EQ(rank.execution_kind,
                  OverlayRankExecutionKind::ContinuationAuthority);
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
        EXPECT_TRUE(rank0->ownsContinuationGraph());
        EXPECT_TRUE(containsDomain(*rank0, "rocm_hot"));
        EXPECT_TRUE(containsDomain(*rank0, "cpu_cold"));

        EXPECT_EQ(rank1->role, OverlayRankRole::CpuFallbackParticipant);
        EXPECT_TRUE(rank1->hasRole(OverlayRankRole::CpuFallbackParticipant));
        EXPECT_FALSE(rank1->ownsContinuationGraph());
        EXPECT_TRUE(containsDomain(*rank1, "cpu_cold"));

        const std::string diagnostics = execution_plan.diagnostics();
        EXPECT_NE(diagnostics.find("rank[0]"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("rank[1]"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("ContinuationRoot"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("CpuFallbackParticipant"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("rocm_hot"), std::string::npos) << diagnostics;
        EXPECT_NE(diagnostics.find("cpu_cold"), std::string::npos) << diagnostics;
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, NodeLocalContinuationBuildsOneDenseShardPerRank)
    {
        const auto execution_plan = resolveMoEExpertOverlayExecutionPlan(
            cpuNodeLocalContinuationPlan(),
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 0,
                .world_size = 2,
            });

        ASSERT_EQ(execution_plan.continuation_root_rank, 0);
        const auto *root = execution_plan.rankPlanFor(0);
        const auto *participant = execution_plan.rankPlanFor(1);
        ASSERT_NE(root, nullptr);
        ASSERT_NE(participant, nullptr);

        EXPECT_TRUE(root->hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_TRUE(root->ownsContinuationGraph());
        EXPECT_EQ(root->execution_kind,
                  OverlayRankExecutionKind::ContinuationAuthority);
        EXPECT_TRUE(root->loads_tokenizer);
        EXPECT_TRUE(root->loads_root_weights);

        EXPECT_TRUE(participant->hasRole(
            OverlayRankRole::ContinuationParticipant));
        EXPECT_TRUE(participant->hasRole(
            OverlayRankRole::CpuFallbackParticipant));
        EXPECT_TRUE(participant->ownsContinuationGraph());
        EXPECT_EQ(participant->execution_kind,
                  OverlayRankExecutionKind::ContinuationPeer);
        EXPECT_FALSE(participant->loads_tokenizer);
        EXPECT_TRUE(participant->loads_root_weights);
        EXPECT_TRUE(participant->loads_shared_expert_weights);
        EXPECT_TRUE(participant->loads_cpu_fallback_experts);
        EXPECT_FALSE(participant->loads_worker_fallback_experts);
        EXPECT_TRUE(containsDomain(*participant, "cpu_overlay"));
        EXPECT_TRUE(containsDevice(*participant, DeviceId::cpu()));
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
        EXPECT_FALSE(rank.ownsContinuationGraph());
        EXPECT_EQ(rank.execution_kind,
                  OverlayRankExecutionKind::ExpertOnlyFollower);
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
        EXPECT_TRUE(root.ownsContinuationGraph());
        EXPECT_TRUE(containsDomain(root, "cuda_fast"));
        EXPECT_TRUE(containsDevice(root, DeviceId::cuda(0)));
        EXPECT_TRUE(root.loads_root_weights);
        EXPECT_TRUE(root.loads_shared_expert_weights);
        EXPECT_TRUE(root.loads_accelerator_routed_experts);
        EXPECT_FALSE(root.loads_cpu_fallback_experts);
        EXPECT_FALSE(root.loads_worker_fallback_experts);

        const auto &rocm = rocm_execution.currentRankPlan();
        EXPECT_EQ(rocm.role, OverlayRankRole::LocalAcceleratorParticipant);
        EXPECT_FALSE(rocm.ownsContinuationGraph());
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
        EXPECT_FALSE(cpu.ownsContinuationGraph());
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
        EXPECT_FALSE(execution_plan.rankPlanFor(3)->ownsContinuationGraph());
        EXPECT_EQ(execution_plan.rankPlanFor(3)->execution_kind,
                  OverlayRankExecutionKind::RelayOnly);
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
        EXPECT_FALSE(rank.ownsContinuationGraph());
        EXPECT_EQ(rank.execution_kind,
                  OverlayRankExecutionKind::ExpertOnlyFollower);
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
            EXPECT_EQ(rank.ownsContinuationGraph(),
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
        // docs/v2/projects/2026-06/MOE_EXPERT_OVERLAY_ORCHESTRATION_REFACTOR_PLAN.md: rank 1 is an
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
        EXPECT_FALSE(rank.ownsContinuationGraph());
        EXPECT_FALSE(rank.hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_TRUE(rank.hasRole(OverlayRankRole::CpuFallbackParticipant));
        EXPECT_NE(execution_plan.diagnostics().find("owns_continuation_graph=false"), std::string::npos);
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, HardwareBindingFollowsSplitGpuSocketInventory)
    {
        const auto requested = rankAgnosticThreeTierPlan();
        const auto inventory = syntheticTwoSocketInventory(
            /*cuda_rank=*/0,
            /*rocm_rank=*/1);

        const auto resolved = bindMoEExpertOverlayPlanToClusterInventory(
            *requested, inventory);

        ASSERT_NE(resolved, nullptr);
        ASSERT_EQ(resolved->domains.size(), 3u);
        EXPECT_EQ(resolved->domains[0].world_ranks, (std::vector<int>{0}));
        EXPECT_EQ(resolved->domains[1].world_ranks, (std::vector<int>{1}));
        EXPECT_EQ(resolved->domains[2].world_ranks, (std::vector<int>{0, 1}));
        EXPECT_EQ(resolved->domains[0].participants[0].numa_node, 0);
        EXPECT_EQ(resolved->domains[1].participants[0].numa_node, 1);
        EXPECT_EQ(resolved->domains[0].participants[0].hostname, "test-node");
        EXPECT_EQ(resolved->domains[1].participants[0].hostname, "test-node");

        const auto execution = resolveMoEExpertOverlayExecutionPlan(
            resolved,
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 1,
                .world_size = 2,
            });
        EXPECT_TRUE(execution.currentRankPlan().hasRole(
            OverlayRankRole::RemoteExpertParticipant));
        EXPECT_TRUE(execution.currentRankPlan().hasRole(
            OverlayRankRole::CpuFallbackParticipant));
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, HardwareBindingKeepsLocalTPRankLocal)
    {
        const auto requested = rankAgnosticLocalCudaTPPlan();
        requested->domains.front().routed_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        const auto inventory = syntheticSingleRankTwoCudaInventory();

        const auto resolved = bindMoEExpertOverlayPlanToClusterInventory(
            *requested, inventory);

        ASSERT_NE(resolved, nullptr);
        ASSERT_EQ(resolved->domains.size(), 1u);
        const auto &routed = resolved->domains.front();
        EXPECT_EQ(routed.owner_rank, 0);
        EXPECT_TRUE(routed.world_ranks.empty());
        ASSERT_EQ(routed.participants.size(), 2u);
        EXPECT_EQ(routed.participants[0].hostname, "test-node");
        EXPECT_EQ(routed.participants[1].hostname, "test-node");

        // The dense and routed views must encode identical rank-local intent.
        ASSERT_EQ(resolved->dense_domains.size(), 1u);
        const auto &dense = resolved->dense_domains.front();
        ASSERT_TRUE(dense.owner_rank.has_value());
        EXPECT_EQ(*dense.owner_rank, 0);
        EXPECT_TRUE(dense.ranks.empty());
        EXPECT_TRUE(dense.validate().empty());

        const auto validation = validateMoERoutedExpertPlacementPlan(*resolved);
        EXPECT_TRUE(validation.ok());
        EXPECT_TRUE(validation.errors.empty());

        /*
         * Inventory binding replaces localhost with a concrete host and
         * intentionally represents LocalTP with one owner rank rather than a
         * repeated participant-rank vector.  Every device in that pool must
         * still resolve as local; otherwise only participant zero can prepare
         * weights and the production overlay graph is impossible to build.
         */
        const auto runtime = resolveMoEExpertOverlayRuntimePlan(
            resolved,
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = 0,
            });
        const auto *runtime_domain = runtime->domainForName("cuda_hot");
        ASSERT_NE(runtime_domain, nullptr);
        ASSERT_EQ(runtime_domain->participants.size(), 2u);
        for (size_t participant_index = 0;
             participant_index < runtime_domain->participants.size();
             ++participant_index)
        {
            const auto &participant =
                runtime_domain->participants[participant_index];
            EXPECT_TRUE(participant.world_rank_known);
            EXPECT_EQ(participant.world_rank, 0);
            EXPECT_TRUE(participant.owned_by_current_rank);
            EXPECT_TRUE(participant.locally_addressable);
            EXPECT_EQ(
                participant.local_device,
                DeviceId::cuda(static_cast<int>(participant_index)));
        }
        EXPECT_TRUE(runtime_domain->domain_scoped_collective_context_ready);
        EXPECT_FALSE(runtime_domain->multi_participant_execution_pending);

        OrchestrationConfig installed = OrchestrationConfig::defaults();
        installed.moe_routed_expert_plan = requested;
        installed.domain_definitions.push_back(
            DomainDefinition::fromExecutionDomainDefinition(
                requested->domains.front().toExecutionDomainDefinition()));
        const auto install_errors = installResolvedMoEExpertOverlayPlan(
            installed,
            resolved,
            MoEExpertOverlayPlanInstallOrigin::UserDeclaredDomains);
        EXPECT_TRUE(install_errors.empty());
        ASSERT_NE(installed.moe_routed_expert_plan, nullptr);
        EXPECT_EQ(
            installed.moe_routed_expert_plan->domains.front().owner_rank, 0);
        EXPECT_TRUE(
            installed.moe_routed_expert_plan->domains.front().world_ranks.empty());
        ASSERT_EQ(installed.domain_definitions.size(), 1u);
        EXPECT_TRUE(installed.domain_definitions.front().explicit_ranks.empty());

        const auto execution = resolveMoEExpertOverlayExecutionPlan(
            resolved,
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 0,
                .world_size = 1,
            });
        EXPECT_TRUE(execution.currentRankPlan().ownsContinuationGraph());
        EXPECT_TRUE(containsDevice(
            execution.currentRankPlan(), DeviceId::cuda(0)));
        EXPECT_TRUE(containsDevice(
            execution.currentRankPlan(), DeviceId::cuda(1)));
    }

    TEST(Test__MoEExpertOverlayRuntimePlan,
         AutoScopeTracksDeviceMovesAndPreservesParticipantRankMultiplicity)
    {
        const auto requested = rankAgnosticAutoScopeGpuTierPlan();
        auto resolved = bindMoEExpertOverlayPlanToClusterInventory(
            *requested,
            syntheticTwoRankHeterogeneousGpuInventory());

        ASSERT_NE(resolved, nullptr);
        ASSERT_EQ(resolved->domains.size(), 2u);
        const auto &cuda = resolved->domains[0];
        const auto &rocm = resolved->domains[1];

        EXPECT_EQ(cuda.scope, ExecutionDomainScope::RANK_LOCAL);
        EXPECT_EQ(cuda.owner_rank, 0);
        EXPECT_TRUE(cuda.world_ranks.empty());

        EXPECT_EQ(rocm.scope, ExecutionDomainScope::NODE_LOCAL);
        EXPECT_EQ(rocm.owner_rank, 0);
        EXPECT_EQ(
            rocm.world_ranks,
            (std::vector<int>{0, 0, 1, 1}));
        EXPECT_TRUE(rocm.toExecutionDomainDefinition().validate().empty());

        const auto validation =
            validateMoERoutedExpertPlacementPlan(*resolved);
        EXPECT_TRUE(validation.ok());
        EXPECT_TRUE(validation.errors.empty());

        const auto execution = resolveMoEExpertOverlayExecutionPlan(
            resolved,
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 1,
                .world_size = 2,
            });
        EXPECT_TRUE(execution.currentRankPlan().hasRole(
            OverlayRankRole::RemoteExpertParticipant));
        EXPECT_TRUE(containsDevice(
            execution.currentRankPlan(), DeviceId::rocm(2)));
        EXPECT_TRUE(containsDevice(
            execution.currentRankPlan(), DeviceId::rocm(3)));

        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.moe_routed_expert_plan = requested;
        for (const auto &domain : requested->domains)
        {
            config.domain_definitions.push_back(
                DomainDefinition::fromExecutionDomainDefinition(
                    domain.toExecutionDomainDefinition()));
        }
        const auto install_errors = installResolvedMoEExpertOverlayPlan(
            config,
            std::move(resolved),
            MoEExpertOverlayPlanInstallOrigin::UserDeclaredDomains);
        EXPECT_TRUE(install_errors.empty());
        ASSERT_NE(config.moe_routed_expert_plan, nullptr);
        EXPECT_EQ(
            config.moe_routed_expert_plan->domains[0].scope,
            ExecutionDomainScope::RANK_LOCAL);
        EXPECT_EQ(
            config.moe_routed_expert_plan->domains[1].scope,
            ExecutionDomainScope::NODE_LOCAL);
        EXPECT_EQ(
            config.moe_routed_expert_plan->domains[1].world_ranks,
            (std::vector<int>{0, 0, 1, 1}));
    }

    TEST(
        Test__MoEExpertOverlayRuntimePlan,
        ImplicitInstallAtomicallyReplacesSimpleTPSelectors)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.tp_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
        };
        config.tp_weights = {0.6f, 0.4f};
        config.tp_degree = 2;
        config.tp_scope = TPScope::RANK_LOCAL;

        const auto normalized = normalizeMoEExpertOverlayAuthorityPlan({
            .model_has_routed_experts = true,
            .world_rank = 0,
            .local_tp_participants = config.tp_devices,
            .local_tp_weights = config.tp_weights,
            .local_tp_backend = CollectiveBackendType::NCCL,
            .residency_maintenance = MoERebalanceRuntimeMode::Off,
        });
        ASSERT_TRUE(normalized.synthesized());
        config.moe_routed_expert_plan = normalized.plan;
        auto resolved = bindMoEExpertOverlayPlanToClusterInventory(
            *normalized.plan,
            syntheticSingleRankTwoCudaInventory());

        const auto install_errors = installResolvedMoEExpertOverlayPlan(
            config,
            std::move(resolved),
            MoEExpertOverlayPlanInstallOrigin::SynthesizedSimpleTP);

        EXPECT_TRUE(install_errors.empty());
        EXPECT_TRUE(config.tp_devices.empty());
        EXPECT_TRUE(config.tp_weights.empty());
        EXPECT_EQ(config.tp_degree, 1);
        EXPECT_EQ(config.tp_scope, TPScope::AUTO);
        ASSERT_EQ(config.domain_definitions.size(), 1u);
        EXPECT_EQ(
            config.domain_definitions.front().name,
            "implicit_moe_local_tp");
        EXPECT_EQ(config.domain_definitions.front().devices.size(), 2u);
        EXPECT_EQ(
            config.domain_definitions.front().weights,
            (std::vector<float>{0.6f, 0.4f}));

        const auto validation =
            ConfigValidator::createStandard().validate(config);
        const auto stale_conflict = std::find_if(
            validation.begin(),
            validation.end(),
            [](const ConfigValidationError &entry)
            {
                return entry.rule_id ==
                       "tp-devices-named-domains-mutex";
            });
        EXPECT_EQ(stale_conflict, validation.end())
            << "The canonical named domain must be the only TP authority";
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, InventoryBoundHostnameRemainsLocalToOwningRank)
    {
        const auto requested = rankAgnosticThreeTierPlan();
        const auto inventory = syntheticTwoSocketInventory(
            /*cuda_rank=*/0,
            /*rocm_rank=*/1);
        const auto resolved = bindMoEExpertOverlayPlanToClusterInventory(
            *requested, inventory);

        ASSERT_NE(resolved, nullptr);

        // Binding deliberately replaces the portable localhost selector with
        // the inventory host name.  Runtime resolution must use the resolved
        // rank, rather than reinterpreting that host label as a remote address.
        const auto rank_zero_runtime = resolveMoEExpertOverlayRuntimePlan(
            resolved,
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = 0,
            });
        const auto *cuda_hot = rank_zero_runtime->domainForName("cuda_hot");
        ASSERT_NE(cuda_hot, nullptr);
        ASSERT_EQ(cuda_hot->participants.size(), 1u);
        const auto &cuda_participant = cuda_hot->participants.front();
        EXPECT_EQ(cuda_participant.address.hostname, "test-node");
        EXPECT_TRUE(cuda_participant.world_rank_known);
        EXPECT_EQ(cuda_participant.world_rank, 0);
        EXPECT_TRUE(cuda_participant.owned_by_current_rank);
        EXPECT_TRUE(cuda_participant.locally_addressable);
        EXPECT_EQ(cuda_participant.local_device, DeviceId::cuda(0));
        EXPECT_TRUE(cuda_hot->primary_is_local);
        EXPECT_TRUE(cuda_hot->local_reachable_for_mvp);

        const auto rank_one_runtime = resolveMoEExpertOverlayRuntimePlan(
            resolved,
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = 1,
                .validate_mvp_root_reachability = false,
            });
        const auto *cuda_hot_remote = rank_one_runtime->domainForName("cuda_hot");
        ASSERT_NE(cuda_hot_remote, nullptr);
        ASSERT_EQ(cuda_hot_remote->participants.size(), 1u);
        EXPECT_FALSE(cuda_hot_remote->participants.front().owned_by_current_rank);
        EXPECT_FALSE(cuda_hot_remote->participants.front().locally_addressable);
        EXPECT_EQ(cuda_hot_remote->primary_device, DeviceId::invalid());
        EXPECT_FALSE(cuda_hot_remote->local_reachable_for_mvp);
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, HardwareBindingAllowsBothGpuFamiliesOnOneSocket)
    {
        const auto requested = rankAgnosticThreeTierPlan();
        const auto inventory = syntheticTwoSocketInventory(
            /*cuda_rank=*/0,
            /*rocm_rank=*/0);

        const auto resolved = bindMoEExpertOverlayPlanToClusterInventory(
            *requested, inventory);

        ASSERT_NE(resolved, nullptr);
        EXPECT_EQ(resolved->domains[0].world_ranks, (std::vector<int>{0}));
        EXPECT_EQ(resolved->domains[1].world_ranks, (std::vector<int>{0}));
        EXPECT_EQ(resolved->domains[2].world_ranks, (std::vector<int>{0, 1}));

        const auto execution = resolveMoEExpertOverlayExecutionPlan(
            resolved,
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = 0,
                .world_size = 2,
            });
        const auto &rank = execution.currentRankPlan();
        EXPECT_TRUE(rank.hasRole(OverlayRankRole::ContinuationRoot));
        EXPECT_TRUE(rank.hasRole(OverlayRankRole::LocalAcceleratorParticipant));
        EXPECT_TRUE(rank.hasRole(OverlayRankRole::CpuFallbackParticipant));
        EXPECT_TRUE(containsDevice(rank, DeviceId::cuda(0)));
        EXPECT_TRUE(containsDevice(rank, DeviceId::rocm(0)));
        EXPECT_TRUE(containsDevice(rank, DeviceId::cpu()));
    }

    TEST(Test__MoEExpertOverlayRuntimePlan, ResolvedInstallUpdatesCanonicalDomainInventoryAtomically)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.moe_routed_expert_plan = rankAgnosticThreeTierPlan();
        for (const auto &domain : config.moe_routed_expert_plan->domains)
        {
            config.domain_definitions.push_back(
                DomainDefinition::fromExecutionDomainDefinition(
                    domain.toExecutionDomainDefinition()));
        }
        const auto inventory = syntheticTwoSocketInventory(
            /*cuda_rank=*/0,
            /*rocm_rank=*/1);
        auto resolved = bindMoEExpertOverlayPlanToClusterInventory(
            *config.moe_routed_expert_plan, inventory);

        const auto errors = installResolvedMoEExpertOverlayPlan(
            config,
            std::move(resolved),
            MoEExpertOverlayPlanInstallOrigin::UserDeclaredDomains);

        EXPECT_TRUE(errors.empty());
        ASSERT_NE(config.moe_routed_expert_plan, nullptr);
        ASSERT_EQ(config.domain_definitions.size(), 3u);
        for (const auto &domain : config.domain_definitions)
        {
            EXPECT_TRUE(domain.owner_rank.has_value()) << domain.name;
            EXPECT_EQ(domain.explicit_ranks.size(), domain.devices.size())
                << domain.name;
        }
        EXPECT_EQ(
            config.moe_routed_expert_plan->domains[1].world_ranks,
            (std::vector<int>{1}));
    }

} // namespace llaminar2::test
